import json

import httpx
import pytest

from leakguard import app as app_module
from leakguard.detect.spans import Span
from leakguard.entities import EntityType as T
from leakguard.sse import StreamRestorer

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


def test_stream_success_relays_rate_limit_headers(make_guard):
    sse = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    def handler(req):
        return httpx.Response(200, content=sse, headers={
            "content-type": "text/event-stream", "x-ratelimit-remaining-tokens": "42"})

    guard = make_guard(transport=httpx.MockTransport(handler))
    with guard.stream("POST", CHAT, json=_stream_req()) as r:
        assert r.status_code == 200
        assert r.headers["x-ratelimit-remaining-tokens"] == "42"


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


# --- Phase 6: placeholders restored inside streams ---

PHONE_MSG = "my phone is 9876543210"


def _payloads(client, body):
    with client.stream("POST", CHAT, json=body) as r:
        assert r.status_code == 200
        return [line[6:] for line in r.iter_lines() if line.startswith("data: ")]


def _content(payloads):
    text = ""
    for p in payloads:
        if p == "[DONE]":
            continue
        for choice in json.loads(p).get("choices", []):
            text += (choice.get("delta") or {}).get("content") or ""
    return text


def _sse(*contents):
    events = "".join(
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": c}}]}) + "\n\n"
        for c in contents)
    return events + "data: [DONE]\n\n"


def _request_records(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records
            if r.getMessage().startswith('{"event": "request"')]


def test_stream_restores_placeholders_with_mock(guard):  # the mock sends 3-char chunks
    assert _content(_payloads(guard, _stream_req(PHONE_MSG))) == "You said: " + PHONE_MSG


def test_stream_without_findings_is_byte_identical(make_guard):
    body = b'data: {"choices":[{"delta":{"content":"[PH"}}]}\r\n\r\n: x\n\ndata: [DONE]\n\n'
    guard = make_guard(transport=httpx.MockTransport(lambda req: httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"})))
    with guard.stream("POST", CHAT, json=_stream_req("hello")) as r:
        assert r.read() == body


def test_non_sse_reply_to_stream_request_restored(make_guard):
    def handler(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "Got " + sent.split()[-1]}}]})

    r = make_guard(transport=httpx.MockTransport(handler)).post(CHAT, json=_stream_req(PHONE_MSG))
    assert r.json()["choices"][0]["message"]["content"] == "Got 9876543210"


def test_restore_failure_is_fail_safe(guard, monkeypatch, caplog):
    def crash(self, event):
        raise RuntimeError("bug")

    monkeypatch.setattr(StreamRestorer, "process", crash)
    caplog.set_level("INFO", logger="leakguard")
    text = _content(_payloads(guard, _stream_req(PHONE_MSG)))
    assert text == "You said: my phone is [PHONE_1]"
    assert '"event": "restore_failed"' in caplog.text and "9876543210" not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
def test_logs_restored_and_unrestored_counts(make_guard, caplog, stream):
    reply = "[PHONE_1] and phone 1 too"  # lowercase bare form: looks like ours, not accepted

    def handler(request):
        if json.loads(request.content).get("stream"):
            chunks = [reply[i:i + 3] for i in range(0, len(reply), 3)]
            return httpx.Response(200, content=_sse(*chunks),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": reply}}]})

    caplog.set_level("INFO", logger="leakguard")
    guard = make_guard(transport=httpx.MockTransport(handler))
    if stream:
        assert _content(_payloads(guard, _stream_req(PHONE_MSG))) == "9876543210 and phone 1 too"
    else:
        guard.post(CHAT, json={**_stream_req(PHONE_MSG), "stream": False})
    (record,) = _request_records(caplog)
    assert (record["restored"], record["unrestored"]) == (1, 1)
    assert "9876543210" not in caplog.text


def test_hindi_value_restored_across_byte_split(make_guard):  # Review Focus 5
    address = "Flat 302,\nSai Krupa Apts, Kothrud, Pune 411038"
    message = f"मेरा नाम रवि है, address:\n{address}"

    def fake_names(text):
        start = text.find("रवि")
        return [] if start == -1 else [Span(start, start + 3, T.NAME, 0.9, "ner")]

    def handler(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        assert "[NAME_1]" in sent and "[ADDRESS_1]" in sent
        events = (
            "data: " + json.dumps({"choices": [{"index": 0, "delta": {
                "content": "[NAME_1] ji, नमस्ते"}}]}, ensure_ascii=False) + "\n\n"
            + "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "t", "type": "function", "function": {
                    "name": "save", "arguments": '{"address": "[ADDRESS_1]"}'}}]}}]}) + "\n\n"
            + "data: [DONE]\n\n").encode("utf-8")

        async def two_bytes_at_a_time():
            for i in range(0, len(events), 2):
                yield events[i:i + 2]

        return httpx.Response(200, content=two_bytes_at_a_time(),
                              headers={"content-type": "text/event-stream"})

    guard = make_guard(transport=httpx.MockTransport(handler), name_model=fake_names)
    payloads = _payloads(guard, _stream_req(message))
    assert _content(payloads) == "रवि ji, नमस्ते"
    args = "".join(call["function"].get("arguments", "")
                   for p in payloads if p != "[DONE]"
                   for choice in json.loads(p)["choices"]
                   for call in (choice.get("delta") or {}).get("tool_calls") or [])
    assert json.loads(args) == {"address": address}
