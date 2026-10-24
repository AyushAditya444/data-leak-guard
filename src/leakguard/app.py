"""The guard's web app: an OpenAI-compatible /v1/chat/completions endpoint.

Phase 1 forwards requests unchanged. Later phases insert detection + masking
before `upstream.send(...)` and restoration on the way back.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from leakguard.config import Settings
from leakguard.errors import openai_error
from leakguard.logs import log_request
from leakguard.upstream import UpstreamClient

CHAT_PATH = "/chat/completions"


@dataclass
class RequestContext:
    """Per-request bookkeeping for the log line."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    upstream_started: float | None = None
    stream: bool = False

    def finish(self, status: int) -> None:
        now = time.perf_counter()
        upstream_ms = None if self.upstream_started is None else (now - self.upstream_started) * 1000
        log_request(
            request_id=self.request_id,
            status=status,
            stream=self.stream,
            upstream_ms=upstream_ms,
            total_ms=(now - self.started) * 1000,
        )


def _forwarded_headers(request: Request) -> dict[str, str]:
    # Pass the caller's own provider key through; the guard holds no keys itself.
    # Everything else (host, content-length, hop-by-hop headers) is set by httpx.
    auth = request.headers.get("authorization")
    return {"authorization": auth} if auth else {}


async def _read_chat_request(request: Request, settings: Settings) -> dict | Response:
    """Return the parsed request body, or an OpenAI-shaped error response."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_request_bytes:
        return _too_large(settings)
    raw = await request.body()
    if len(raw) > settings.max_request_bytes:  # size check happens before parsing
        return _too_large(settings)
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return openai_error(400, "Request body is not valid JSON.",
                            "invalid_request_error", "invalid_json")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return openai_error(400, "Request body must be an object with a 'messages' list.",
                            "invalid_request_error", "invalid_request")
    return body


def _too_large(settings: Settings) -> Response:
    return openai_error(413, f"Request body exceeds {settings.max_request_bytes} bytes.",
                        "invalid_request_error", "request_too_large")


async def _send_upstream(
    upstream: UpstreamClient, body: dict, headers: dict, stream: bool, ctx: RequestContext
) -> httpx.Response | Response:
    """Send to the LLM provider, translating network failures into OpenAI-shaped errors."""
    ctx.upstream_started = time.perf_counter()
    try:
        return await upstream.send(CHAT_PATH, body, headers, stream=stream)
    except httpx.TimeoutException:
        return openai_error(504, "The upstream LLM did not respond in time.",
                            "upstream_error", "upstream_timeout")
    except httpx.TransportError:
        return openai_error(502, "Could not reach the upstream LLM.",
                            "upstream_error", "upstream_unreachable")


def _relayed_headers(resp: httpx.Response) -> dict[str, str]:
    """Provider headers SDKs rely on (retry timing, rate limits). An allowlist, so
    cookies and hop-by-hop headers from the provider never reach the caller."""
    return {
        name: value
        for name, value in resp.headers.items()
        if name in ("retry-after", "retry-after-ms", "openai-processing-ms")
        or name.startswith("x-ratelimit-")
    }


def _relay(resp: httpx.Response) -> Response:
    return Response(content=resp.content, status_code=resp.status_code,
                    headers=_relayed_headers(resp), media_type=resp.headers.get("content-type"))


async def relay_stream(resp: httpx.Response) -> AsyncIterator[bytes]:
    """Pass streamed bytes through unchanged, and always close the upstream response —
    on normal end, on error, and when the client disconnects mid-stream.

    Uses aiter_bytes (decoded), not aiter_raw: the guard doesn't forward the
    Content-Encoding header, so compressed raw bytes would reach the client unreadable.
    Phase 6 replaces this pass-through with an SSE-aware placeholder restorer.
    """
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()


async def _stream_and_log(resp: httpx.Response, ctx: RequestContext) -> AsyncIterator[bytes]:
    # aclosing() guarantees relay_stream's `finally` runs even if we are closed early.
    try:
        async with aclosing(relay_stream(resp)) as chunks:
            async for chunk in chunks:
                yield chunk
    finally:
        ctx.finish(resp.status_code)  # logged when the stream ends, so timings are complete


async def _proxy_chat(
    request: Request, settings: Settings, upstream: UpstreamClient, ctx: RequestContext
) -> Response:
    body = await _read_chat_request(request, settings)
    if isinstance(body, Response):
        ctx.finish(body.status_code)
        return body

    ctx.stream = body.get("stream") is True
    sent = await _send_upstream(upstream, body, _forwarded_headers(request), ctx.stream, ctx)
    if isinstance(sent, Response):
        ctx.finish(sent.status_code)
        return sent

    if ctx.stream and sent.is_success:
        return StreamingResponse(_stream_and_log(sent, ctx), status_code=sent.status_code,
                                 headers=_relayed_headers(sent), media_type="text/event-stream")

    if ctx.stream:
        # An error reply to a stream request (e.g. 429) is plain JSON, not SSE:
        # read it fully and relay it like any other error.
        await sent.aread()
        await sent.aclose()
    response = _relay(sent)
    ctx.finish(response.status_code)
    return response


def create_app(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Build the guard. Tests pass `transport` to route upstream calls in-process."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.upstream = UpstreamClient(settings, transport)
        yield
        await app.state.upstream.aclose()

    app = FastAPI(title="Data Leak Guard", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        ctx = RequestContext()
        response = await _proxy_chat(request, settings, request.app.state.upstream, ctx)
        response.headers["x-request-id"] = ctx.request_id
        return response

    return app


def build_app() -> FastAPI:
    """Entry point for `uvicorn --factory leakguard.app:build_app`."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings.from_env())
