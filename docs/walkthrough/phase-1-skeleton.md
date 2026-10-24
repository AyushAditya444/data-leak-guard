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
- `ensure_ascii=False` keeps Hindi/emoji readable in the JSON instead of `न...` escapes.

### Fixtures (`conftest.py`)
pytest looks at a test's parameter names and supplies matching fixtures. `def test_x(mock_app)` automatically receives `create_mock_app(chunk_size=3)`. `anyio_backend` tells async tests to run on asyncio.

### Q&A
**Q: Why do you need a mock LLM at all?**
A: Three reasons. (1) **Cost** — thousands of test requests to a real LLM cost money and hit rate limits; the mock is free. (2) **Determinism** — a real LLM answers differently each time, so tests would be flaky; the mock always answers the same way. (3) **Targeted edge cases** — the mock deliberately splits replies into tiny chunks to force the hardest streaming case. Real APIs are still used, but only in a few end-to-end checks.
