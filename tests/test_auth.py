"""The optional guard key: LEAKGUARD_API_KEY / X-Leakguard-Key."""

import json
import logging

import httpx
import pytest

from leakguard.config import Settings

CHAT = "/v1/chat/completions"
KEY = "k" * 16


def _req(content="hi"):
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


def _is_guard_key_error(r) -> bool:
    return r.status_code == 401 and r.json()["error"]["code"] == "invalid_guard_key"


def test_guard_key_off_by_default(guard):
    assert guard.post(CHAT, json=_req()).status_code == 200


def test_guard_key_missing_or_wrong_is_401(make_guard):
    guard = make_guard(api_key=KEY)
    for headers in ({}, {"X-Leakguard-Key": "k" * 15 + "x"}, {"X-Leakguard-Key": ""}):
        assert _is_guard_key_error(guard.post(CHAT, json=_req(), headers=headers))


def test_guard_key_right_passes(make_guard):
    guard = make_guard(api_key=KEY)
    r = guard.post(CHAT, json=_req(), headers={"X-Leakguard-Key": KEY})
    assert r.status_code == 200


def test_guard_key_checked_before_body(make_guard):
    # Without the key, nothing about the body is looked at: no 413, no 400.
    guard = make_guard(api_key=KEY, max_request_bytes=10)
    assert _is_guard_key_error(guard.post(CHAT, json=_req("x" * 100)))
    assert _is_guard_key_error(guard.post(CHAT, content=b"not json"))


def test_guard_key_non_ascii_header_is_401(make_guard):
    guard = make_guard(api_key=KEY)
    for value in ("ключ".encode(), (KEY + " ").encode(), b"\xff\xfe"):
        assert _is_guard_key_error(guard.post(CHAT, json=_req(),
                                              headers={"X-Leakguard-Key": value}))


def test_guard_key_covers_every_v1_path(make_guard):
    guard = make_guard(api_key=KEY)
    assert _is_guard_key_error(guard.get("/v1/models"))
    assert guard.get("/healthz").status_code == 200


def test_guard_key_not_forwarded(make_guard):
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    guard = make_guard(transport=httpx.MockTransport(handler), api_key=KEY)
    guard.post(CHAT, json=_req(), headers={"X-Leakguard-Key": KEY,
                                           "Authorization": "Bearer sk-test"})
    assert "x-leakguard-key" not in seen["headers"]
    assert seen["headers"]["authorization"] == "Bearer sk-test"


def test_guard_key_never_logged(make_guard, caplog):
    guard = make_guard(api_key=KEY)
    wrong = "w" * 20
    with caplog.at_level(logging.INFO, logger="leakguard"):
        guard.post(CHAT, json=_req(), headers={"X-Leakguard-Key": KEY})
        guard.post(CHAT, json=_req(), headers={"X-Leakguard-Key": wrong})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert KEY not in text and wrong not in text
    events = [json.loads(r.getMessage())["event"] for r in caplog.records]
    assert "auth_failed" in events


def test_short_key_refuses_start():
    with pytest.raises(ValueError, match="at least 16"):
        Settings(api_key="short")


def test_api_key_from_env():
    assert Settings.from_env({"LEAKGUARD_API_KEY": KEY}).api_key == KEY
    assert Settings.from_env({"LEAKGUARD_API_KEY": ""}).api_key is None
    assert Settings.from_env({}).api_key is None


def test_guard_key_enforced_under_a_root_path(make_guard, mock_app):
    # Behind a reverse proxy serving the guard under /guard, the request path the app sees
    # includes the prefix; the key must still be required.
    from fastapi.testclient import TestClient

    from leakguard.app import create_app

    app = create_app(Settings(upstream_base_url="http://mock/v1", ner_enabled=False,
                              api_key=KEY), httpx.ASGITransport(app=mock_app))
    # Like uvicorn --root-path: the ASGI path carries the prefix, and routing strips it.
    with TestClient(app, root_path="/guard") as client:
        assert _is_guard_key_error(client.post("/guard" + CHAT, json=_req()))
        r = client.post("/guard" + CHAT, json=_req(), headers={"X-Leakguard-Key": KEY})
        assert r.status_code == 200


def test_auth_failed_path_is_capped(caplog):
    # The path is the caller's input: an unauthenticated client must not fill the logs.
    from leakguard.logs import log_auth_failed

    with caplog.at_level(logging.INFO, logger="leakguard"):
        log_auth_failed(request_id="r1", path="/v1/" + "a" * 5000)
    [record] = caplog.records
    assert len(json.loads(record.getMessage())["path"]) <= 200
