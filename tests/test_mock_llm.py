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
    assert body["choices"][0]["message"] == {
        "role": "assistant", "content": "You said: hi [PHONE_1]"}
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
