import json

import httpx

import leakguard.app as app_module
from leakguard.policy import DEFAULT_POLICY, FailureMode, Policy

CHAT = "/v1/chat/completions"


def _req(content, **extra):
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}


def test_end_to_end_mask_and_restore(make_guard):
    seen = {}

    def handler(request):
        seen["sent"] = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "Refund to " + seen["sent"].split()[-1]}}]})

    guard = make_guard(transport=httpx.MockTransport(handler))
    r = guard.post(CHAT, json=_req("my upi is rahul.s@okicici"))
    assert seen["sent"] == "my upi is [UPI_1]"
    assert r.json()["choices"][0]["message"]["content"] == "Refund to rahul.s@okicici"


def test_with_mock_llm_round_trip(guard):
    r = guard.post(CHAT, json=_req("Aadhaar 2345 6789 0124"))
    assert r.json()["choices"][0]["message"]["content"] == "You said: Aadhaar 2345 6789 0124"


def test_secret_blocked_with_400_naming_type_not_value(make_guard):
    calls = []
    guard = make_guard(transport=httpx.MockTransport(lambda req: calls.append(1)))
    r = guard.post(CHAT, json=_req("debug with sk-abcdefghijklmnopqrstuvwxyz12"))
    assert r.status_code == 400 and calls == []
    err = r.json()["error"]
    assert err["code"] == "sensitive_data_blocked" and "SECRET" in err["message"]
    assert "sk-abc" not in err["message"]


def test_non_object_message_items_rejected(guard):
    r = guard.post(CHAT, json={"model": "m", "messages": ["hi"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def _crash(*args, **kwargs):
    raise RuntimeError("detector crashed")


def test_detection_failure_fails_closed(make_guard, monkeypatch):
    monkeypatch.setattr(app_module, "mask_messages", _crash)
    r = make_guard().post(CHAT, json=_req("hi"))
    assert r.status_code == 500 and r.json()["error"]["code"] == "detection_failed"


def test_detection_failure_open_mode_forwards(make_guard, monkeypatch):
    monkeypatch.setattr(app_module, "mask_messages", _crash)
    pol = Policy(actions=DEFAULT_POLICY.actions, failure_mode=FailureMode.OPEN)
    r = make_guard(policy=pol).post(CHAT, json=_req("hi"))
    assert r.status_code == 200


def test_streaming_request_is_masked(make_guard):
    seen = {}

    def handler(request):
        seen["sent"] = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    with make_guard(transport=httpx.MockTransport(handler)).stream(
            "POST", CHAT, json=_req("pan ABCPS1234K", stream=True)) as r:
        r.read()
    assert seen["sent"] == "pan [PAN_1]"


def test_logs_counts_but_never_values(guard, caplog):
    caplog.set_level("INFO", logger="leakguard")
    guard.post(CHAT, json=_req("phone 9876543210 pan ABCPS1234K"))
    assert "9876543210" not in caplog.text and "ABCPS1234K" not in caplog.text
    assert '"PHONE:mask": 1' in caplog.text and '"PAN:mask": 1' in caplog.text


def test_non_streaming_restore_failure_is_fail_safe(make_guard, monkeypatch, caplog):
    # final review I4: a restorer bug must not turn a good reply into a bare 500
    def crash(body, restorer):
        raise RuntimeError("bug")

    def handler(request):
        sent = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "Got " + sent.split()[-1]}}]})

    monkeypatch.setattr(app_module, "restore_response", crash)
    caplog.set_level("INFO", logger="leakguard")
    r = make_guard(transport=httpx.MockTransport(handler)).post(CHAT, json=_req("call 9876543210"))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Got [PHONE_1]"
    assert '"event": "restore_failed"' in caplog.text and '"event": "request"' in caplog.text
    assert "9876543210" not in caplog.text
