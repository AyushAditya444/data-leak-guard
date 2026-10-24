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
