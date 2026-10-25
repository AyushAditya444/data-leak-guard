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
    err = {"error": {"message": "bad key", "type": "invalid_request_error",
                     "param": None, "code": "invalid_api_key"}}
    guard = make_guard(transport=httpx.MockTransport(lambda req: httpx.Response(401, json=err)))
    r = guard.post(CHAT, json=_req())
    assert r.status_code == 401 and r.json() == err


def test_rate_limit_headers_relayed(make_guard):
    # The OpenAI SDK reads retry-after to schedule retries; a drop-in proxy must keep it.
    def handler(req):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={
            "retry-after": "7", "x-ratelimit-remaining-requests": "0", "set-cookie": "a=b"})

    r = make_guard(transport=httpx.MockTransport(handler)).post(CHAT, json=_req())
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.headers["x-ratelimit-remaining-requests"] == "0"
    assert "set-cookie" not in r.headers


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
    # a streamed reply with a restored value: the value must not reach the logs either
    body = {**_req("call 9876543210"), "stream": True}
    with guard.stream("POST", CHAT, json=body) as r:
        assert "9876543210" in r.read().decode()
    assert "9876543210" not in caplog.text


def test_chunked_body_over_the_limit_is_413_without_reading_it_all(mock_app):
    # No Content-Length (chunked): the size must be enforced while reading, not after.
    # Driven through raw ASGI, because TestClient reads the whole body before calling the app.
    import asyncio

    from leakguard.app import create_app
    from leakguard.config import Settings

    app = create_app(Settings(upstream_base_url="http://mock/v1", ner_enabled=False,
                              max_request_bytes=1000), httpx.ASGITransport(app=mock_app))
    pulled, sent = [], []

    async def receive():
        pulled.append(1)  # 100 chunks x 100 bytes = 10x the limit
        return {"type": "http.request", "body": b"x" * 100, "more_body": len(pulled) < 100}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": CHAT, "raw_path": CHAT.encode(),
             "root_path": "", "query_string": b"", "server": ("t", 80), "client": ("c", 1),
             "headers": [(b"content-type", b"application/json"),
                         (b"transfer-encoding", b"chunked")]}
    async def run():
        async with app.router.lifespan_context(app):  # creates the upstream client
            await app(scope, receive, send)

    asyncio.run(run())
    assert sent[0]["status"] == 413
    assert len(pulled) <= 11  # stopped just past the limit, not after all 100 chunks
