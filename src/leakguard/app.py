"""The guard's web app: an OpenAI-compatible /v1/chat/completions endpoint.

Request flow:
    read + validate  ->  detect + apply policy  ->  forward masked request upstream
    ->  restore placeholders in the reply  ->  log metadata only
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
from starlette.concurrency import run_in_threadpool

from leakguard.config import Settings
from leakguard.detect.engine import Detector, EntityModel, build_detector
from leakguard.detect.gazetteer import load_gazetteer
from leakguard.errors import openai_error
from leakguard.logs import log_detection_failed, log_ner_disabled, log_request
from leakguard.masking import PlaceholderMap, mask_messages
from leakguard.ner.model import NameModel, NameModelError
from leakguard.policy import FailureMode, Policy, load_policy
from leakguard.restore import Restorer, restore_response
from leakguard.upstream import UpstreamClient

CHAT_PATH = "/chat/completions"


@dataclass
class RequestContext:
    """Per-request bookkeeping for the log line."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.perf_counter)
    upstream_started: float | None = None
    stream: bool = False
    detection_ms: float | None = None
    findings: dict[str, int] | None = None  # e.g. {"PHONE:mask": 2} - counts, never values
    blocked: bool = False

    def finish(self, status: int) -> None:
        now = time.perf_counter()
        upstream_ms = None if self.upstream_started is None else (now - self.upstream_started) * 1000
        log_request(
            request_id=self.request_id,
            status=status,
            stream=self.stream,
            upstream_ms=upstream_ms,
            total_ms=(now - self.started) * 1000,
            findings=self.findings,
            blocked=self.blocked,
            detection_ms=self.detection_ms,
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
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        return openai_error(400, "Request body must be an object with a 'messages' list "
                            "of message objects.", "invalid_request_error", "invalid_request")
    return body


def _too_large(settings: Settings) -> Response:
    return openai_error(413, f"Request body exceeds {settings.max_request_bytes} bytes.",
                        "invalid_request_error", "request_too_large")


def _apply_policy(
    body: dict, policy: Policy, ctx: RequestContext, detect: Detector
) -> tuple[dict, PlaceholderMap | None] | Response:
    """Mask the request per policy. Returns (body to send, mapping) or an error response."""
    started = time.perf_counter()
    try:
        result = mask_messages(body["messages"], policy, detect)
    # Deliberately broad: whatever a detector bug raises, the request must not slip
    # through unchecked by accident (fail-closed), so every exception is handled here.
    except Exception:  # noqa: BLE001
        ctx.detection_ms = (time.perf_counter() - started) * 1000
        log_detection_failed(request_id=ctx.request_id, failure_mode=policy.failure_mode)
        if policy.failure_mode is FailureMode.OPEN:
            return body, None  # explicitly configured: availability over safety
        return openai_error(500, "The guard could not inspect this request, so it was not sent.",
                            "guard_error", "detection_failed")
    ctx.detection_ms = (time.perf_counter() - started) * 1000
    ctx.findings = {f"{entity}:{action}": n for (entity, action), n in sorted(result.counts.items())}

    if result.blocked:
        ctx.blocked = True
        kinds = ", ".join(entity.value for entity in result.blocked)
        return openai_error(
            400, f"Request blocked: it contains sensitive data ({kinds}) that this guard's "
            "policy does not allow to be sent to an LLM.",
            "invalid_request_error", "sensitive_data_blocked")
    return {**body, "messages": result.messages}, result.mapping


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


def _restored_content(resp: httpx.Response, restorer: Restorer | None) -> bytes:
    """The reply body with placeholders swapped back for real values (JSON replies only)."""
    is_json = "application/json" in resp.headers.get("content-type", "")
    if restorer is None or not resp.is_success or not is_json:
        return resp.content
    try:
        data = resp.json()
    except ValueError:
        return resp.content
    if not isinstance(data, dict):
        return resp.content
    return json.dumps(restore_response(data, restorer), ensure_ascii=False).encode("utf-8")


def _relay(resp: httpx.Response, restorer: Restorer | None = None) -> Response:
    return Response(content=_restored_content(resp, restorer), status_code=resp.status_code,
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
    request: Request, settings: Settings, policy: Policy, detect: Detector,
    upstream: UpstreamClient, ctx: RequestContext,
) -> Response:
    body = await _read_chat_request(request, settings)
    if isinstance(body, Response):
        ctx.finish(body.status_code)
        return body

    # Detection is CPU-bound (the name model takes milliseconds). Run in a worker thread,
    # it can't stall the event loop, so other requests keep being served meanwhile.
    applied = await run_in_threadpool(_apply_policy, body, policy, ctx, detect)
    if isinstance(applied, Response):
        ctx.finish(applied.status_code)
        return applied
    outgoing, mapping = applied
    restorer = Restorer(mapping) if mapping else None

    ctx.stream = outgoing.get("stream") is True
    sent = await _send_upstream(upstream, outgoing, _forwarded_headers(request), ctx.stream, ctx)
    if isinstance(sent, Response):
        ctx.finish(sent.status_code)
        return sent

    if ctx.stream and sent.is_success:
        # The request was masked, so nothing leaks; restoring placeholders inside the
        # stream arrives in Phase 6 (until then the stream shows [TYPE_N] placeholders).
        return StreamingResponse(_stream_and_log(sent, ctx), status_code=sent.status_code,
                                 headers=_relayed_headers(sent), media_type="text/event-stream")

    if ctx.stream:
        # An error reply to a stream request (e.g. 429) is plain JSON, not SSE:
        # read it fully and relay it like any other error.
        await sent.aread()
        await sent.aclose()
    response = _relay(sent, restorer)
    ctx.finish(response.status_code)
    return response


def _load_name_model(settings: Settings) -> Detector | None:
    """The name model's detect(): names only. Its location tags are not used by the
    address detector: on dev they raised recall and coverage but lowered precision, so the
    pre-stated rule rejected them (docs/evaluation/address-tuning.md)."""
    if not settings.ner_enabled:
        log_ner_disabled()
        return None
    try:
        return NameModel.load(settings.ner_model_dir, settings.ner_threads).detect
    except NameModelError as exc:
        raise NameModelError(
            f"NER model not found or invalid at {settings.ner_model_dir}: {exc}. "
            'Run "python -m leakguard.ner.export" or set LEAKGUARD_NER=off.') from None


def create_app(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
    policy: Policy | None = None,
    name_model: Detector | None = None,
    ner: EntityModel | None = None,
) -> FastAPI:
    """Build the guard. Tests pass `transport` to route upstream calls in-process, and
    `name_model` (names only) or `ner` (names + location tags) to stand in for the real
    NER model. The policy and the model are loaded here, so an invalid policy file or a
    missing model stops the guard at startup."""
    policy = policy or load_policy(settings.policy_path)
    load_gazetteer()  # the PIN directory (~1 MB): load now, not on the first request
    if ner is not None:
        detect = build_detector(ner=ner)
    else:
        detect = build_detector(name_model or _load_name_model(settings))

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
        response = await _proxy_chat(request, settings, policy, detect,
                                     request.app.state.upstream, ctx)
        response.headers["x-request-id"] = ctx.request_id
        return response

    return app


def build_app() -> FastAPI:
    """Entry point for `uvicorn --factory leakguard.app:build_app`."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings.from_env())
