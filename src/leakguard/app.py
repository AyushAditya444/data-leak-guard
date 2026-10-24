"""The guard's web app: an OpenAI-compatible /v1/chat/completions endpoint.

Phase 1 forwards requests unchanged. Later phases insert detection + masking
before `upstream.send(...)` and restoration on the way back.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Request, Response

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


def _relay(resp: httpx.Response) -> Response:
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


async def _proxy_chat(
    request: Request, settings: Settings, upstream: UpstreamClient, ctx: RequestContext
) -> Response:
    body = await _read_chat_request(request, settings)
    if isinstance(body, Response):
        ctx.finish(body.status_code)
        return body

    sent = await _send_upstream(upstream, body, _forwarded_headers(request), False, ctx)
    response = sent if isinstance(sent, Response) else _relay(sent)
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
