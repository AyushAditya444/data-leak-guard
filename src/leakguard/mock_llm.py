"""A fake OpenAI-compatible LLM for free, offline testing.

It replies "You said: <last user message>". Echoing is useful: if the guard masks
"9876543210" as "[PHONE_1]" on the way in, the mock echoes "[PHONE_1]" back, so the
guard's restore step has something real to restore.

Streamed replies are cut into tiny chunks (3 characters by default) on purpose, so a
placeholder like "[PHONE_1]" arrives split across chunks — the hard case the
streaming restorer has to handle.
"""

import json
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

COMPLETION_ID = "mock-1"


def _text_of(content: Any) -> str:
    """Message content is either a string or a list of parts like {"type": "text", ...}."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _text_of(message.get("content"))
    return ""


def _chunk_event(model: str, delta: dict, finish_reason: str | None = None) -> str:
    chunk = {
        "id": COMPLETION_ID,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _stream_events(model: str, reply: str, chunk_size: int) -> Iterator[str]:
    yield _chunk_event(model, {"role": "assistant"})
    for start in range(0, len(reply), chunk_size):
        yield _chunk_event(model, {"content": reply[start:start + chunk_size]})
    yield _chunk_event(model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


def create_mock_app(chunk_size: int = 3) -> FastAPI:
    app = FastAPI(title="Mock LLM")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model", "mock")
        reply = "You said: " + _last_user_text(body.get("messages", []))

        if body.get("stream"):
            return StreamingResponse(
                _stream_events(model, reply, chunk_size), media_type="text/event-stream"
            )

        return {
            "id": COMPLETION_ID,
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app
