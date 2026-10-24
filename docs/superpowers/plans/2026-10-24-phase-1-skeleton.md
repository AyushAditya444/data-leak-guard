# Phase 1 — Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working pass-through proxy (no detection yet): an OpenAI-compatible `/v1/chat/completions` endpoint that forwards to a configurable upstream, streaming and non-streaming, plus a bundled mock LLM, Docker Compose, and CI.

**Architecture:** A FastAPI app (`create_app`) holds one shared `httpx.AsyncClient` (via `UpstreamClient`) for its lifetime and relays requests to `{upstream_base_url}/chat/completions`. Streaming responses are relayed byte-for-byte. A second small FastAPI app (`create_mock_app`) imitates an OpenAI-compatible LLM by echoing the last user message, splitting streamed replies into tiny chunks so later phases can test placeholders split across chunks. Tests wire the guard to the mock in-process with `httpx.ASGITransport` — no network, no cost.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, httpx, PyYAML (later phases), pytest, anyio (pytest plugin, ships with httpx), openai SDK (tests/examples only), ruff, Docker, GitHub Actions.

**Spec:** [docs/superpowers/specs/2026-10-24-data-leak-guard-design.md](../specs/2026-10-24-data-leak-guard-design.md) — this plan implements §9 phase 1, and the parts of §6.1, §6.5, §6.7, §6.8 that don't involve detection.

**Who writes it:** Claude implements everything (spec §8, changed 2026-10-24). Each task ends by adding a section to `docs/walkthrough/phase-1-skeleton.md` so the author can study and explain the code later.

**Project root:** `data-leak-guard/` (all paths below are relative to it).

## Global Constraints

- Python **3.11**; package name **`leakguard`**, in a `src/` layout.
- **Zero cost**: no test or step may require a paid API or network access; tests use the in-process mock.
- **Never log** message content, request/response bodies, or the `Authorization` header (spec §6.7, anti-goal "the guard becoming the leak").
- Errors returned by the guard use OpenAI's shape: `{"error": {"message": str, "type": str, "param": null, "code": str | null}}`.
- Only `POST /v1/chat/completions` and `GET /healthz` exist (spec §2 non-goals).
- The client's `Authorization` header is passed through to upstream; the guard holds no provider keys (spec §6.5).
- Unknown request fields are forwarded unchanged (spec §6.5).
- Must run on the author's Windows 11 laptop and on Linux (Docker, CI).

## Review Focus

1. **Client disconnects mid-stream** → the upstream stream must be closed, not leaked. Test: `test_relay_stream_closes_upstream_on_early_exit` (Task 3).
2. **Upstream base URLs with different shapes** (trailing slash, Gemini's `/v1beta/openai/`) → must join to the right URL. Test: `test_upstream_url_join` (Task 2).
3. **Malformed bodies** (invalid JSON, a JSON array, missing `messages`) → 400 in OpenAI shape, never a 500. Tests: `test_invalid_json_returns_400`, `test_non_object_or_missing_messages_returns_400` (Task 2).
4. **Upstream error while `stream: true`** (e.g. 429 rate limit from a free tier) → the status and JSON error must be relayed, not a 200 event-stream with garbage. Test: `test_stream_upstream_error_relayed` (Task 3).
5. **Upstream unreachable or slow** (Ollama not running, free tier timing out) → 502 / 504 in OpenAI shape. Tests: `test_upstream_unreachable_returns_502`, `test_upstream_timeout_returns_504` (Task 2).

---

## File Structure

```
data-leak-guard/
├── pyproject.toml              # package metadata, deps, pytest + ruff config
├── .gitignore
├── README.md                   # quickstart (Task 4)
├── Dockerfile                  # one image, used for both guard and mock (Task 4)
├── docker-compose.yml          # guard + mock-llm services (Task 4)
├── .github/workflows/ci.yml    # ruff + pytest on push/PR (Task 4)
├── examples/chat.py            # official openai SDK → guard, the "drop-in" proof (Task 4)
├── src/leakguard/
│   ├── __init__.py
│   ├── config.py               # Settings dataclass + from_env (Task 2)
│   ├── errors.py               # openai_error() helper (Task 2)
│   ├── logs.py                 # log_request(): JSON log line, metadata only (Task 2)
│   ├── upstream.py             # upstream_url(), UpstreamClient (Task 2)
│   ├── app.py                  # create_app(), build_app(), relay_stream() (Tasks 2–3)
│   └── mock_llm.py             # create_mock_app(): fake OpenAI-compatible LLM (Task 1)
└── tests/
    ├── conftest.py             # fixtures: mock_app, guard client, anyio_backend
    ├── test_mock_llm.py        # Task 1
    ├── test_config_upstream.py # Task 2
    ├── test_proxy.py           # Task 2
    ├── test_streaming.py       # Task 3
    └── test_sdk.py             # Task 3
```

---

### Task 1: Repo scaffold + mock LLM

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/leakguard/__init__.py`, `src/leakguard/mock_llm.py`
- Test: `tests/conftest.py`, `tests/test_mock_llm.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `leakguard.mock_llm.create_mock_app(chunk_size: int = 3) -> fastapi.FastAPI`
  - Mock contract: `POST /v1/chat/completions` replies with content `"You said: " + <text of the last message whose role is "user">`. If that message's `content` is a list of parts, the `text` of every `{"type": "text"}` part is concatenated.
    - Non-stream body: `{"id": "mock-1", "object": "chat.completion", "created": 0, "model": <request model>, "choices": [{"index": 0, "message": {"role": "assistant", "content": <reply>}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}`
    - Stream (`"stream": true`): `text/event-stream`; events are `data: <json>\n\n`; each JSON is `{"id": "mock-1", "object": "chat.completion.chunk", "created": 0, "model": <model>, "choices": [{"index": 0, "delta": {...}, "finish_reason": null}]}`. First event's delta is `{"role": "assistant"}`; then one event per `chunk_size` characters of the reply with delta `{"content": <piece>}`; then delta `{}` with `finish_reason: "stop"`; then `data: [DONE]\n\n`.
  - `tests/conftest.py` fixture `mock_app` → `create_mock_app(chunk_size=3)`, fixture `anyio_backend` → `"asyncio"`.

- [ ] **Step 1: Initialise the repo and packaging**

`git init` in `data-leak-guard/`, branch `main`. `pyproject.toml`: project `leakguard`, version `0.1.0`, `requires-python = ">=3.11"`, dependencies `fastapi`, `uvicorn[standard]`, `httpx`, `pyyaml`; optional `dev` extra: `pytest`, `openai`, `ruff`. Setuptools `src` layout. `[tool.pytest.ini_options] testpaths = ["tests"]`. `[tool.ruff] line-length = 100, target-version = "py311"`. `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `.env`.

Then:
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux: source .venv/bin/activate
pip install -e ".[dev]"
```
Expected: install succeeds; `python -c "import leakguard"` exits 0.

- [ ] **Step 2: Write the failing tests** in `tests/test_mock_llm.py`

```python
import json
from fastapi.testclient import TestClient

def _sse_payloads(text: str) -> list[str]:
    return [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]

def test_mock_non_stream_echoes_last_user_message(mock_app):
    client = TestClient(mock_app)
    r = client.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "system", "content": "x"},
                     {"role": "user", "content": "first"},
                     {"role": "assistant", "content": "ok"},
                     {"role": "user", "content": "hi [PHONE_1]"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "m"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "You said: hi [PHONE_1]"}
    assert body["choices"][0]["finish_reason"] == "stop"

def test_mock_content_parts_are_joined(mock_app):
    client = TestClient(mock_app)
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
    assert r.json()["choices"][0]["message"]["content"] == "You said: ab"

def test_mock_stream_splits_into_small_chunks(mock_app):
    client = TestClient(mock_app)
    r = client.post("/v1/chat/completions", json={
        "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    payloads = _sse_payloads(r.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    pieces = [c["choices"][0]["delta"].get("content", "") for c in chunks]
    assert "".join(pieces) == "You said: hi"
    assert all(len(p) <= 3 for p in pieces)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_mock_llm.py -v`
Expected: FAIL — `ImportError` / fixture `mock_app` not found.

- [ ] **Step 4: Implement `create_mock_app(chunk_size: int = 3) -> FastAPI`** in `src/leakguard/mock_llm.py`, and the `mock_app` + `anyio_backend` fixtures in `tests/conftest.py`

Stream with `fastapi.responses.StreamingResponse(media_type="text/event-stream")` over a generator that yields the events in the contract above.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mock_llm.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit** (include the existing `docs/` so the design is in history)

```bash
git add .
git commit -m "feat: scaffold leakguard package and mock OpenAI-compatible LLM"
```

- [ ] **Step 7: Walkthrough notes** — add a section to `docs/walkthrough/phase-1-skeleton.md` explaining: what an OpenAI chat-completion request/response looks like; what SSE is and why streamed replies arrive as `data:` lines; why the mock splits into 3-character chunks (so `[PHONE_1]` will be split across chunks in Phase 6). Include a Q&A: *"Why do we need a mock LLM at all?"*

---

### Task 2: Non-streaming pass-through proxy

**Files:**
- Create: `src/leakguard/config.py`, `src/leakguard/errors.py`, `src/leakguard/logs.py`, `src/leakguard/upstream.py`, `src/leakguard/app.py`
- Modify: `tests/conftest.py` (add `guard` fixture)
- Test: `tests/test_config_upstream.py`, `tests/test_proxy.py`

**Interfaces:**
- Consumes: `create_mock_app` (Task 1).
- Produces:
  - `leakguard.config.Settings` — `@dataclass(frozen=True)` with fields `upstream_base_url: str = "http://localhost:9000/v1"`, `upstream_timeout_s: float = 60.0`, `max_request_bytes: int = 1_048_576`.
  - `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings` — reads `LEAKGUARD_UPSTREAM_BASE_URL`, `LEAKGUARD_UPSTREAM_TIMEOUT_S`, `LEAKGUARD_MAX_REQUEST_BYTES`; missing → defaults; `None` → `os.environ`.
  - `leakguard.errors.openai_error(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse`.
  - `leakguard.logs.log_request(*, request_id: str, status: int, stream: bool, upstream_ms: float | None, total_ms: float) -> None` — one JSON line at INFO on logger `"leakguard"`. Accepts **only** these fields.
  - `leakguard.upstream.upstream_url(base_url: str, path: str) -> str`.
  - `leakguard.upstream.UpstreamClient(settings: Settings, transport: httpx.AsyncBaseTransport | None = None)` with `async def send(self, path: str, body: dict, headers: Mapping[str, str], stream: bool) -> httpx.Response` (when `stream=True` the response body is **not** read; caller must `aclose()` it) and `async def aclose(self) -> None`.
  - `leakguard.app.create_app(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI` — `UpstreamClient` created/closed in the app's lifespan. Every response carries an `x-request-id` header (uuid4 hex).
  - `leakguard.app.build_app() -> FastAPI` — `create_app(Settings.from_env())`, and configures `logging.basicConfig(level=INFO)` if the root logger has no handlers. Used by `uvicorn --factory`.
  - `tests/conftest.py` fixture `make_guard(transport=None, **settings_overrides) -> TestClient` (default transport: `httpx.ASGITransport(app=mock_app)`, default base URL `"http://mock/v1"`; each client is entered as a context manager so the lifespan runs, and closed at teardown), and fixture `guard` = `make_guard()`.
- Error mapping (all OpenAI-shaped):

  | Condition | Status | `type` | `code` |
  |---|---|---|---|
  | body bytes > `max_request_bytes` | 413 | `invalid_request_error` | `request_too_large` |
  | body not valid JSON | 400 | `invalid_request_error` | `invalid_json` |
  | body not an object, or no `messages` list | 400 | `invalid_request_error` | `invalid_request` |
  | `httpx.ConnectError` | 502 | `upstream_error` | `upstream_unreachable` |
  | `httpx.TimeoutException` | 504 | `upstream_error` | `upstream_timeout` |
  | upstream non-2xx | upstream's status | upstream body relayed as-is | — |

- [ ] **Step 1: Write the failing tests** in `tests/test_config_upstream.py`

```python
import pytest
from leakguard.config import Settings
from leakguard.upstream import upstream_url

@pytest.mark.parametrize("base,expected", [
    ("http://a/v1", "http://a/v1/chat/completions"),
    ("http://a/v1/", "http://a/v1/chat/completions"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/",
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
])
def test_upstream_url_join(base, expected):
    assert upstream_url(base, "/chat/completions") == expected

def test_settings_from_env_reads_values():
    s = Settings.from_env({"LEAKGUARD_UPSTREAM_BASE_URL": "http://x/v1",
                           "LEAKGUARD_UPSTREAM_TIMEOUT_S": "5",
                           "LEAKGUARD_MAX_REQUEST_BYTES": "100"})
    assert (s.upstream_base_url, s.upstream_timeout_s, s.max_request_bytes) == ("http://x/v1", 5.0, 100)

def test_settings_from_env_defaults():
    assert Settings.from_env({}) == Settings()
```

- [ ] **Step 2: Write the failing tests** in `tests/test_proxy.py`

```python
import json
import httpx

CHAT = "/v1/chat/completions"

def _req(content="hi", **extra):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}

def test_healthz(guard):
    r = guard.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_non_stream_passthrough_keeps_unicode(guard):
    r = guard.post(CHAT, json=_req("नमस्ते 🙏 hi"))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "You said: नमस्ते 🙏 hi"
    assert r.headers["x-request-id"]

def test_forwards_body_and_auth_unchanged(make_guard):
    seen = {}
    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})
    guard = make_guard(transport=httpx.MockTransport(handler), upstream_base_url="http://up/v1")
    sent = _req(temperature=0.2, some_future_field={"x": 1})
    guard.post(CHAT, json=sent, headers={"Authorization": "Bearer sk-test"})
    assert seen["url"] == "http://up/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == sent

def test_upstream_error_status_relayed(make_guard):
    err = {"error": {"message": "bad key", "type": "invalid_request_error", "param": None, "code": "invalid_api_key"}}
    guard = make_guard(transport=httpx.MockTransport(lambda req: httpx.Response(401, json=err)))
    r = guard.post(CHAT, json=_req())
    assert r.status_code == 401 and r.json() == err

def test_upstream_unreachable_returns_502(make_guard):
    def handler(req):
        raise httpx.ConnectError("refused")
    r = make_guard(transport=httpx.MockTransport(handler)).post(CHAT, json=_req())
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error"
    assert r.json()["error"]["code"] == "upstream_unreachable"

def test_upstream_timeout_returns_504(make_guard):
    def handler(req):
        raise httpx.ReadTimeout("slow")
    r = make_guard(transport=httpx.MockTransport(handler)).post(CHAT, json=_req())
    assert r.status_code == 504 and r.json()["error"]["code"] == "upstream_timeout"

def test_request_too_large_returns_413(make_guard):
    r = make_guard(max_request_bytes=100).post(CHAT, json=_req("x" * 200))
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"

def test_invalid_json_returns_400(guard):
    r = guard.post(CHAT, content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"

def test_non_object_or_missing_messages_returns_400(guard):
    for body in ([1, 2], {"model": "m"}, {"model": "m", "messages": "nope"}):
        r = guard.post(CHAT, json=body)
        assert r.status_code == 400, body
        assert r.json()["error"]["code"] == "invalid_request"

def test_logs_contain_no_content_or_auth(guard, caplog):
    caplog.set_level("INFO", logger="leakguard")
    guard.post(CHAT, json=_req("secret-value-123"), headers={"Authorization": "Bearer sk-live-xyz"})
    assert "secret-value-123" not in caplog.text
    assert "sk-live-xyz" not in caplog.text
    assert "request_id" in caplog.text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config_upstream.py tests/test_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError: leakguard.config` / fixture `make_guard` not found.

- [ ] **Step 4: Implement** `config.py`, `errors.py`, `logs.py`, `upstream.py`, `app.py` (non-streaming path + `/healthz`), and the `make_guard` / `guard` fixtures, per the Interfaces block

Read the body as bytes first (size check before parsing). `upstream_url` = `base_url.rstrip("/") + path`. Forward only the `Authorization` header (when present); let httpx set `Content-Type`. Relay the upstream response's status, body bytes and `content-type` header; don't copy hop-by-hop headers.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest -v`
Expected: all Task 1 + Task 2 tests pass.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add src tests
git commit -m "feat: non-streaming OpenAI-compatible pass-through proxy"
```
Expected: ruff reports no errors.

- [ ] **Step 7: Walkthrough notes** — add a section to `docs/walkthrough/phase-1-skeleton.md` explaining: the app-factory pattern and why tests inject an `httpx` transport; why one shared `AsyncClient` (connection pooling); why the size check happens before JSON parsing; the error table and why errors copy OpenAI's shape (SDKs parse them); why the log function *only accepts* metadata fields (makes leaking content impossible by construction). Include a Q&A: *"What happens if Ollama isn't running?"*

---

### Task 3: Streaming pass-through + official SDK drop-in test

**Files:**
- Modify: `src/leakguard/app.py`
- Test: `tests/test_streaming.py`, `tests/test_sdk.py`

**Interfaces:**
- Consumes: `UpstreamClient.send(..., stream=True)`, `create_app`, `make_guard`/`guard` fixtures (Task 2).
- Produces:
  - `leakguard.app.relay_stream(resp: httpx.Response) -> AsyncIterator[bytes]` — yields `resp.aiter_raw()` chunks unchanged; closes `resp` in `finally` (so it closes on normal end, error, and client disconnect). Phase 6 will replace the body of this relay with an SSE-aware restorer.
  - Behaviour: when the request body has `"stream": true` and upstream returns 2xx → `StreamingResponse(relay_stream(resp), media_type="text/event-stream")`. Upstream non-2xx → read the body, close, relay status + JSON body (same as non-streaming). The request log line is emitted when the stream finishes.

- [ ] **Step 1: Write the failing tests** in `tests/test_streaming.py`

```python
import json
import httpx
import pytest
from leakguard.app import relay_stream

CHAT = "/v1/chat/completions"

def _stream_req(content="hello"):
    return {"model": "m", "stream": True, "messages": [{"role": "user", "content": content}]}

def test_stream_passthrough_reassembles(guard):
    with guard.stream("POST", CHAT, json=_stream_req()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        payloads = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    text = "".join(json.loads(p)["choices"][0]["delta"].get("content", "") for p in payloads[:-1])
    assert text == "You said: hello"

def test_stream_upstream_error_relayed(make_guard):
    err = {"error": {"message": "slow down", "type": "rate_limit_error", "param": None, "code": None}}
    guard = make_guard(transport=httpx.MockTransport(lambda req: httpx.Response(429, json=err)))
    r = guard.post(CHAT, json=_stream_req())
    assert r.status_code == 429
    assert r.json() == err

@pytest.mark.anyio
async def test_relay_stream_closes_upstream_on_early_exit():
    async def body():
        yield b"data: 1\n\n"
        yield b"data: 2\n\n"
    resp = httpx.Response(200, content=body())
    agen = relay_stream(resp)
    assert await agen.__anext__() == b"data: 1\n\n"
    await agen.aclose()
    assert resp.is_closed
```

- [ ] **Step 2: Write the failing test** in `tests/test_sdk.py` — proves "change one line" works with the official SDK

```python
from openai import OpenAI

def _client(guard):
    return OpenAI(base_url="http://testserver/v1", api_key="sk-test", http_client=guard)

def test_openai_sdk_non_stream(guard):
    out = _client(guard).chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "You said: hi"

def test_openai_sdk_stream(guard):
    stream = _client(guard).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "You said: hi"
```
(`TestClient` is an `httpx.Client` subclass, so the SDK accepts it as `http_client`.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_streaming.py tests/test_sdk.py -v`
Expected: FAIL — `ImportError: relay_stream`; streaming tests get a non-stream response.

- [ ] **Step 4: Implement `relay_stream` and the streaming branch** in `src/leakguard/app.py`, per the Interfaces block

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add src tests
git commit -m "feat: streaming pass-through and OpenAI SDK drop-in tests"
```

- [ ] **Step 7: Walkthrough notes** — add a section to `docs/walkthrough/phase-1-skeleton.md` explaining: why streaming needs a generator instead of returning a body; what happens to the upstream connection if the user closes the chat mid-reply (and why `finally` matters); why an upstream error during a stream request must be relayed as JSON, not as an event stream; what `test_sdk.py` proves for the README's "change one line" claim. Include a Q&A: *"Where will Phase 6 plug the placeholder restorer in?"*

---

### Task 4: Docker, Compose, CI, README quickstart, example client

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml`, `README.md`, `examples/chat.py`

**Interfaces:**
- Consumes: `leakguard.app:build_app`, `leakguard.mock_llm:create_mock_app` (both used as `uvicorn --factory` targets).
- Produces:
  - Image: `python:3.11-slim`, installs the package (not dev extras), non-root user, default `CMD` runs the guard: `uvicorn --factory leakguard.app:build_app --host 0.0.0.0 --port 8080`.
  - Compose services: `mock-llm` (same image, command `uvicorn --factory leakguard.mock_llm:create_mock_app --host 0.0.0.0 --port 9000`) and `guard` (port `8080:8080`, env `LEAKGUARD_UPSTREAM_BASE_URL=http://mock-llm:9000/v1`, `depends_on: mock-llm`).
  - CI: on push and pull_request; ubuntu-latest; Python 3.11; `pip install -e ".[dev]"`; `ruff check .`; `pytest -v`.
  - `examples/chat.py`: uses `openai.OpenAI(base_url=os.environ.get("LEAKGUARD_URL", "http://localhost:8080/v1"), api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-mock"))`, sends one message, prints the reply, then streams the same message and prints chunks as they arrive.
  - README: one-paragraph description (from spec §1), "Status: Phase 1 — pass-through proxy, no detection yet", Quickstart (`docker compose up --build`, then `python examples/chat.py`), how to point at Ollama / Gemini / Groq via `LEAKGUARD_UPSTREAM_BASE_URL`, how to run tests.

- [ ] **Step 1: Write Dockerfile, .dockerignore, docker-compose.yml** per the Interfaces block

- [ ] **Step 2: Verify the stack runs**

Run:
```bash
docker compose up -d --build
curl -s http://localhost:8080/healthz
python examples/chat.py
docker compose down
```
Expected: health check prints `{"status":"ok"}`; the example prints `You said: ...` once in full and once streamed in pieces.

- [ ] **Step 3: Write `.github/workflows/ci.yml`** per the Interfaces block, and verify locally that the same commands pass

Run: `ruff check . && pytest -v`
Expected: no lint errors; all tests pass. (CI itself runs once the author pushes to GitHub — creating the public repo is the author's step, done when they choose.)

- [ ] **Step 4: Write `README.md` and `examples/chat.py`** per the Interfaces block

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml .github README.md examples
git commit -m "chore: docker compose stack, CI workflow, README quickstart, example client"
```

- [ ] **Step 6: Walkthrough notes** — add a section to `docs/walkthrough/phase-1-skeleton.md` explaining: what each Dockerfile line does and why a non-root user; how Compose's service names become hostnames (`mock-llm`); what CI checks and why it matters (green badge = tests actually run); how to switch upstream to Ollama with one env var. Include step-by-step instructions for running the whole stack and the example from a fresh terminal.

---

## Roadmap (later phases — each gets its own plan when we reach it)

| Phase | Plan written |
|---|---|
| 2. Rules + secrets (incl. bank account, IFSC), policy, masking/blocking, non-stream restore | after Phase 1 |
| 3. Evaluation v0 (generator, dev/test split, harness) | after Phase 2 |
| 4. NER (model comparison, ONNX + quantisation) | after Phase 3 |
| 5. Addresses | after Phase 4 |
| 6. Streaming restore + tolerant matching | after Phase 5 |
| 7. Presidio baseline, report, README, threat model (demo video recorded by the author) | after Phase 6 |

Claude implements every phase and writes `docs/walkthrough/phase-N-<name>.md` for each.

Plans are written one phase at a time because later phases depend on what earlier ones teach us (e.g. which NER model wins, how often free LLMs mangle placeholders).
