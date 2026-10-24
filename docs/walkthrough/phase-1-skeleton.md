# Phase 1 Walkthrough — The Skeleton

> **What Phase 1 builds:** a proxy that forwards AI requests unchanged (no detection yet), a fake AI for free testing, Docker, and CI. Every later phase plugs into this skeleton.

---

## 1. Project setup and the fake AI (mock LLM)

### Files
| File | Purpose |
|---|---|
| `pyproject.toml` | The project's ID card: name, Python version, dependencies, test and lint settings |
| `.gitattributes` | Forces Unix line endings (LF) so Docker/CI files don't break when edited on Windows |
| `src/leakguard/__init__.py` | Marks `leakguard` as a Python package |
| `src/leakguard/mock_llm.py` | The fake AI |
| `tests/conftest.py` | Shared test "fixtures" (ready-made objects tests can ask for by name) |
| `tests/test_mock_llm.py` | Tests for the fake AI |

### Why a `src/` layout?
Code lives in `src/leakguard/`, not directly in `leakguard/`. This forces tests to import the **installed** package (via `pip install -e .`), so you catch packaging mistakes early instead of tests accidentally importing files from the current folder.

### What an OpenAI chat request looks like
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a support bot."},
    {"role": "user", "content": "My phone is 9876543210"}
  ],
  "stream": false
}
```
- `messages` is the **whole conversation so far** — this is why the guard can be stateless (see design §6.2).
- `content` can be a plain string **or** a list of parts like `[{"type": "text", "text": "..."}]` (used for images + text). `_text_of()` handles both.

### What the reply looks like
```json
{"id": "...", "object": "chat.completion", "model": "...",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
 "usage": {...}}
```

### Streaming and SSE
With `"stream": true`, the reply arrives in pieces using **Server-Sent Events (SSE)** — plain text lines over one long HTTP response:
```
data: {"choices":[{"delta":{"role":"assistant"}}], ...}

data: {"choices":[{"delta":{"content":"You"}}], ...}

data: {"choices":[{"delta":{"content":" sa"}}], ...}

data: [DONE]
```
Each event is `data: <json>` followed by a blank line. The client glues the `content` pieces together. `[DONE]` means finished.

### How `mock_llm.py` works
- `create_mock_app(chunk_size=3)` builds a small FastAPI app with one endpoint, `POST /v1/chat/completions`.
- It finds the **last user message** (`_last_user_text`) and replies `"You said: <that text>"`.
- **Why echo?** When the guard later masks `9876543210` → `[PHONE_1]`, the mock echoes `[PHONE_1]` back — giving the guard's restore step a realistic job, for free.
- **Why 3-character chunks?** So `[PHONE_1]` gets split like `[PH` / `ONE` / `_1]` across chunks. That's the hardest case for restoring placeholders in streams (Phase 6). Real LLMs split text unpredictably too; the mock makes it happen *every* time so tests catch it.
- `_stream_events` yields: a first chunk with the role, one chunk per 3 characters, a final chunk with `finish_reason: "stop"`, then `[DONE]` — the same order real OpenAI uses.
- `ensure_ascii=False` keeps Hindi/emoji readable in the JSON, instead of turning each character into an escape code (a backslash, the letter u, and four hex digits).

### Fixtures (`conftest.py`)
pytest looks at a test's parameter names and supplies matching fixtures. `def test_x(mock_app)` automatically receives `create_mock_app(chunk_size=3)`. `anyio_backend` tells async tests to run on asyncio.

### Q&A
**Q: Why do you need a mock LLM at all?**
A: Three reasons. (1) **Cost** — thousands of test requests to a real LLM cost money and hit rate limits; the mock is free. (2) **Determinism** — a real LLM answers differently each time, so tests would be flaky; the mock always answers the same way. (3) **Targeted edge cases** — the mock deliberately splits replies into tiny chunks to force the hardest streaming case. Real APIs are still used, but only in a few end-to-end checks.

---

## 2. The non-streaming proxy

### Files
| File | Purpose |
|---|---|
| `src/leakguard/config.py` | `Settings`: upstream URL, timeout, max request size — read from environment variables |
| `src/leakguard/errors.py` | `openai_error()`: builds errors in OpenAI's JSON shape |
| `src/leakguard/logs.py` | `log_request()`: one JSON log line per request, metadata only |
| `src/leakguard/upstream.py` | `upstream_url()` and `UpstreamClient`: talks to the LLM provider |
| `src/leakguard/app.py` | `create_app()`: the FastAPI app with `/healthz` and `/v1/chat/completions` |
| `tests/conftest.py` | `make_guard` / `guard` fixtures: a guard wired to the mock LLM, in-process |
| `tests/test_config_upstream.py`, `tests/test_proxy.py` | Tests |

### The request's journey (`app.py`)
```
POST /v1/chat/completions
  → RequestContext()            new request id + start timer
  → _read_chat_request()        size check → parse JSON → check it has a 'messages' list
  → _forwarded_headers()        keep only the caller's Authorization header
  → _send_upstream()            POST to {upstream_base_url}/chat/completions
  → _relay()                    copy status, body, content-type back to the caller
  → ctx.finish()                write the log line
  → add x-request-id header
```
In Phase 2, detection + masking slot in **between** reading the request and sending it upstream, and restoring slots in **before** relaying the reply.

### Key design decisions

**1. App factory (`create_app(settings, transport)`).**
Instead of one global `app`, a function builds it. Tests can build many guards with different settings, and inject a **transport** — httpx's pluggable "how bytes actually travel" layer:
- `httpx.ASGITransport(app=mock_app)` → requests go straight into the mock LLM's Python code, no network.
- `httpx.MockTransport(handler)` → a function decides the response, so tests can fake a 401, a timeout, or a dead server.

**2. One shared `httpx.AsyncClient` (in `lifespan`).**
`lifespan` runs once at startup/shutdown. Creating the client there means **connection pooling**: the TCP + TLS handshake to the provider (~100 ms+) happens once and is reused, instead of once per request.

**3. Size check *before* JSON parsing.**
Parsing a 500 MB body could exhaust memory. The guard first trusts-but-verifies `Content-Length`, then checks the actual byte count, and only then parses. Cheap checks first — a basic denial-of-service defence.

**4. OpenAI-shaped errors.**
OpenAI SDKs expect `{"error": {"message", "type", "param", "code"}}`. Matching it means a developer using the guard gets a clean `openai.APIStatusError` with a readable message, not a confusing crash.

| Situation | Status | `code` |
|---|---|---|
| Body too large | 413 | `request_too_large` |
| Not valid JSON | 400 | `invalid_json` |
| Not an object / no `messages` list | 400 | `invalid_request` |
| Upstream unreachable (connection refused, DNS, broken connection) | 502 | `upstream_unreachable` |
| Upstream too slow | 504 | `upstream_timeout` |
| Upstream returns an error (e.g. 401 bad key, 429 rate limit) | same as upstream | relayed unchanged |

502 vs 504: both mean "the problem is behind me". 502 *Bad Gateway* = the upstream failed; 504 *Gateway Timeout* = the upstream didn't answer in time.

**5. Logs that *can't* leak.**
`log_request()` only accepts named metadata arguments (`request_id`, `status`, `stream`, timings). There's no parameter for message text, so nobody can accidentally log an Aadhaar number. A test (`test_logs_contain_no_content_or_auth`) sends a secret and an API key and checks neither appears in the logs. **Security by construction beats security by discipline.**

**6. Only `Authorization` is forwarded.**
The guard passes the caller's own provider key through and holds none itself. Other headers (`Host`, `Content-Length`, hop-by-hop headers like `Connection`) belong to *this* hop and must not be copied to the next one — httpx sets the correct ones.

**7. Unknown fields are forwarded unchanged.**
The guard parses the body to check `messages`, but forwards the whole object, so new OpenAI features (`some_future_field` in the test) keep working without updating the guard.

### Q&A
**Q: What happens if Ollama isn't running?**
A: The upstream connection is refused, httpx raises `ConnectError` (a `TransportError`), and the guard returns **502** with `{"error": {"type": "upstream_error", "code": "upstream_unreachable", ...}}`. The OpenAI SDK raises a clean error the developer can read, and the log line still records the request id, status 502, and timing — but never the message text.
