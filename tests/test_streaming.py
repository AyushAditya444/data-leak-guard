import json

import httpx
import pytest

from leakguard import app as app_module

CHAT = "/v1/chat/completions"


def _stream_req(content="hello"):
    return {"model": "m", "stream": True, "messages": [{"role": "user", "content": content}]}


def test_stream_passthrough_reassembles(guard):
    with guard.stream("POST", CHAT, json=_stream_req()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # Sent incrementally (chunked), not buffered: a buffered reply would carry Content-Length.
        assert "content-length" not in r.headers
        payloads = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    text = "".join(
        json.loads(p)["choices"][0]["delta"].get("content", "") for p in payloads[:-1])
    assert text == "You said: hello"


def test_stream_upstream_error_relayed(make_guard):
    err = {"error": {"message": "slow down", "type": "rate_limit_error",
                     "param": None, "code": None}}
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
    agen = app_module.relay_stream(resp)
    assert await agen.__anext__() == b"data: 1\n\n"
    await agen.aclose()
    assert resp.is_closed
