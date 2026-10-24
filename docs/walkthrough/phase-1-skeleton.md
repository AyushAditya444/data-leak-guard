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

**7. Rate-limit headers are relayed (an allowlist).**
When a provider says `429 Too Many Requests`, it also sends `Retry-After` and `x-ratelimit-*` headers. The OpenAI SDK reads `Retry-After` to decide how long to wait before retrying. The guard copies exactly these headers back (`_relayed_headers()`), and nothing else — so a provider's cookies or connection headers never leak through to the caller. This was caught in the final review: without it, the SDK's retry timing silently changes behind the guard, which breaks the "drop-in" promise.

**8. Unknown fields are forwarded unchanged.**
The guard parses the body to check `messages`, but forwards the whole object, so new OpenAI features (`some_future_field` in the test) keep working without updating the guard.

### Q&A
**Q: What happens if Ollama isn't running?**
A: The upstream connection is refused, httpx raises `ConnectError` (a `TransportError`), and the guard returns **502** with `{"error": {"type": "upstream_error", "code": "upstream_unreachable", ...}}`. The OpenAI SDK raises a clean error the developer can read, and the log line still records the request id, status 502, and timing — but never the message text.


---

## 3. Streaming pass-through and the "change one line" proof

### Files
| File | Purpose |
|---|---|
| `src/leakguard/app.py` | adds `relay_stream()`, `_stream_and_log()` and the streaming branch in `_proxy_chat()` |
| `tests/test_streaming.py` | streaming behaviour tests |
| `tests/test_sdk.py` | the **official OpenAI SDK** talking to the guard |

### How the streaming branch works
```
"stream": true in the request?
  ├─ send upstream with stream=True        (response body NOT read yet)
  ├─ upstream status 2xx?
  │    yes → return StreamingResponse(_stream_and_log(...))
  │           bytes flow upstream → guard → client as they arrive
  │    no  → read the (small) JSON error fully, relay it as a normal error
  └─ the log line is written when the stream *ends*
```

### Key design decisions

**1. Why a generator?** A normal response is built once and sent. A stream must send bytes *as they arrive* from the LLM, so the user sees words appear. A Python async generator (`async def ... yield`) produces values one at a time; `StreamingResponse` sends each one immediately.

**2. Always close the upstream connection (`finally`).**
If a user closes the chat window mid-reply, the client disconnects. FastAPI stops our generator — and the `finally` block in `relay_stream()` runs `resp.aclose()`. Without it, the guard would keep an open connection to the provider (and keep paying for tokens!) for an answer nobody reads. `aclosing(...)` in `_stream_and_log()` makes sure the inner generator is closed too, not left for the garbage collector. The test `test_relay_stream_closes_upstream_on_early_exit` reads one chunk, stops, and checks the upstream response is closed.

**3. Errors during a stream request are JSON, not SSE.**
If the free tier says "429 Too Many Requests", the body is a small JSON error — not an event stream. Relaying it as `text/event-stream` with status 200 would hide the error. So the guard checks `sent.is_success` first; errors are read fully and relayed with their real status (`test_stream_upstream_error_relayed`).

**4. `aiter_bytes()` not `aiter_raw()`.**
Providers may compress responses (gzip). `aiter_raw()` gives the compressed bytes; `aiter_bytes()` gives them decompressed. The guard doesn't forward the `Content-Encoding` header, so it must send decompressed bytes or the client receives garbage. (Phase 6 needs the decoded text anyway, to restore placeholders.)

**5. How the test proves real streaming.**
Fun fact found while building: the SDK streaming test passed even *before* streaming was implemented, because the old code collected the whole reply and sent it at once — functionally fine, but the user would wait for the entire answer. The fix was a test that checks real streaming behaviour: a streamed response has **no `Content-Length` header** (the total size isn't known in advance; it's sent in chunks). That's a good lesson: *a passing test only proves what it actually checks.*

**6. `test_sdk.py` — the drop-in proof.**
It uses the real `openai` Python library with only `base_url` changed. Both normal and streaming calls work. That backs the README claim "change one line" with a test, not a promise.

### Where Phase 6 plugs in
`relay_stream()` currently passes bytes through untouched. In Phase 6 it becomes an SSE-aware restorer: parse each `data:` event, find `delta.content`, hold back anything that might be the start of a split placeholder like `[PHO`, restore when complete, and re-emit the event.

### Q&A
**Q: What happens to the upstream connection if the user closes the chat mid-reply?**
A: The client disconnect makes the server stop iterating our async generator; its `finally` block calls `resp.aclose()`, which closes the connection to the provider immediately — so we don't keep generating (and paying for) tokens no one will read. I have a unit test that stops a stream after one chunk and asserts the upstream response is closed.


---

## 4. Docker, CI, README, and the example client

### Files
| File | Purpose |
|---|---|
| `Dockerfile` | recipe for the container image (used for both the guard and the mock LLM) |
| `.dockerignore` | files *not* copied into the image (git history, tests, docs, caches) |
| `docker-compose.yml` | starts `mock-llm` + `guard` together with one command |
| `.github/workflows/ci.yml` | GitHub runs lint + all tests on every push |
| `examples/chat.py` | the official OpenAI SDK talking to the guard — normal + streamed |
| `README.md` | what it is, quickstart, configuration |

### The Dockerfile, line by line
| Line | Why |
|---|---|
| `FROM python:3.11-slim` | small official Python image (no compilers/docs) → smaller, faster, fewer vulnerabilities |
| `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED` | no `.pyc` clutter; logs appear immediately instead of being buffered |
| `PIP_NO_CACHE_DIR=1` | don't keep pip's download cache inside the image → smaller |
| `COPY pyproject.toml ...` then `RUN pip install .` | installs the package and its dependencies |
| `useradd ... appuser` + `USER appuser` | **never run as root**: if someone exploits the app, they don't get root in the container |
| `CMD ["uvicorn", "--factory", "leakguard.app:build_app", ...]` | `--factory` tells uvicorn to *call* `build_app()` to get the app, which reads settings from env vars at startup |

### docker-compose: two services, one image
- `mock-llm` runs the same image with a different command (`create_mock_app` on port 9000).
- `guard` exposes port 8080 to your machine and points at `http://mock-llm:9000/v1`.
- **Service names are hostnames:** inside Compose's private network, `mock-llm` resolves to the mock's container. That's why the URL isn't `localhost` — inside the guard's container, `localhost` means *the guard's own container*.
- To reach Ollama running on your laptop from inside a container, use `host.docker.internal` (Docker Desktop's name for "the host machine").
- `${LEAKGUARD_UPSTREAM_BASE_URL:-http://mock-llm:9000/v1}` = "use the env var if set, otherwise the mock".

### CI (`ci.yml`)
On every push or pull request, GitHub starts a fresh Ubuntu machine, installs Python 3.11 and the project, runs `ruff check .` (lint) and `pytest -v` (all tests). A green check mark on the repo shows visitors the tests **really run** — not just exist. It runs once the code is pushed to GitHub.

### Verified end to end
```
$ docker compose up -d --build
$ curl -s http://localhost:8080/healthz
{"status":"ok"}
$ python examples/chat.py
Full reply:    You said: Hi, my phone number is 9876543210.
Streamed reply: You| sa|id:| Hi|, m|y p|hon|e n|umb|er |is |987|654|321|0.|
```
The `|` marks show each streamed chunk — the mock's 3-character chunks travelling through the guard. And the guard's log line contains **only metadata**:
```
{"event": "request", "request_id": "c38f...", "status": 200, "stream": true, "upstream_ms": 14.9, "total_ms": 55.7}
```
(Notice the phone number is still visible in the reply — Phase 1 has no detection yet. Phase 2 will turn it into `[PHONE_1]` before it leaves.)

### Running it yourself from a fresh terminal
1. Start **Docker Desktop** and wait until it says "Engine running".
2. `cd data-leak-guard` (your clone of the repo)
3. `docker compose up --build` (leave it running; logs appear here)
4. In a second terminal, same folder: activate your virtual environment (e.g. `.venv\Scripts\activate`), then `python examples/chat.py`
5. Run the tests: `pytest` (with the venv activated).
6. Stop: `Ctrl+C` in the first terminal, then `docker compose down`.

### Q&A
**Q: Why does the guard's config say `http://mock-llm:9000` and not `http://localhost:9000`?**
A: Each container has its own network namespace — `localhost` inside the guard's container is the guard itself. Docker Compose puts both services on a private network with built-in DNS, where each **service name** resolves to that container. So the guard reaches the mock at `mock-llm`. Same idea in Kubernetes, where services are reached by name.
