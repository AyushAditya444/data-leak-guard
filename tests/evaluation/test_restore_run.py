import asyncio
import json

import httpx
import pytest

from leakguard.entities import EntityType as T
from leakguard.evaluation.restore_run import (
    Conversation,
    api_key_from_env,
    build_conversations,
    parse_reset,
    run_conversations,
)

PHONE_MSG = "Please note my phone is 9876543210, confirm it back."


def _conversation(text=PHONE_MSG, tool=False):
    return Conversation(id="t-000", sample_id="t-1",
                        messages=[{"role": "user", "content": text}],
                        tools=None, tool_choice=None)


def _fake_groq(reply, calls=None, first_status=None):
    """A fake provider: replies with `reply` (masked side), streamed in 3-char chunks when
    asked. `first_status` makes the very first call fail with that status."""
    calls = [] if calls is None else calls

    def handler(request):
        calls.append(json.loads(request.content))
        if first_status and len(calls) == 1:
            return httpx.Response(first_status, headers={"retry-after": "0"},
                                  json={"error": {"message": "slow down"}})
        if calls[-1].get("stream"):
            events = "".join(
                "data: " + json.dumps({"id": "x", "choices": [
                    {"index": 0, "delta": {"content": reply[i:i + 3]}}]}) + "\n\n"
                for i in range(0, len(reply), 3)) + "data: [DONE]\n\n"
            return httpx.Response(200, content=events.encode(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "x", "choices": [{"index": 0, "message": {
            "role": "assistant", "content": reply}, "finish_reason": "stop"}]})

    return httpx.MockTransport(handler)


def _run(transport, conversations=None, **kw):
    return asyncio.run(run_conversations(conversations or [_conversation()], ["m1"], transport,
                                         api_key="gsk-test-key", max_tokens=100, **kw))


def test_conversations_are_deterministic_and_safe():
    a, b = build_conversations("dev", 100, seed=1), build_conversations("dev", 100, seed=1)
    assert a == b and len(a) == 100
    assert 15 <= sum(c.tools is not None for c in a) <= 25
    assert all(c.tool_choice for c in a if c.tools)
    assert len({c.sample_id for c in a}) == 100
    assert all("SECRET" not in json.dumps(c.messages) for c in a)


def test_conversations_skip_secrets_negatives_and_hard_samples():
    from leakguard.evaluation.loading import load_split
    by_id = {s.id: s for s in load_split("dev")}
    for c in build_conversations("dev", 100, seed=1):
        sample = by_id[c.sample_id]
        assert sample.spans and sample.category not in ("hard", "negative")
        assert T.SECRET not in {span.type for span in sample.spans}


def test_dev_and_test_use_their_own_samples():
    dev = {c.sample_id for c in build_conversations("dev", 100, seed=1)}
    test = {c.sample_id for c in build_conversations("test", 300, seed=2)}
    assert all(s.startswith("dev-") for s in dev) and all(s.startswith("test-") for s in test)


def test_outcomes_count_forms_and_guard_counters():
    result = _run(_fake_groq("[PHONE_1] / phone 1 / {PHONE_1}"))
    assert [(o["stream"], o["status"]) for o in result["outcomes"]] == [(False, 200), (True, 200)]
    for outcome in result["outcomes"]:
        assert outcome["mentions"] == {"exact": 1, "bracketless": 1, "brackets": 1}
        assert (outcome["restored"], outcome["unrestored"]) == (2, 1)
        assert outcome["mentioned"] is True


def test_reply_without_placeholders_is_not_mentioned():
    (non_stream, stream) = _run(_fake_groq("Sorry, I can't help with that."))["outcomes"]
    assert non_stream["mentioned"] is False and non_stream["mentions"] == {}
    assert (stream["restored"], stream["unrestored"]) == (0, 0)


def test_429_is_retried():
    calls = []
    result = _run(_fake_groq("[PHONE_1]", calls, first_status=429))
    assert [o["status"] for o in result["outcomes"]] == [200, 200]
    assert len(calls) == 3


def test_requests_carry_key_model_and_cap():
    calls = []
    _run(_fake_groq("ok", calls))
    assert calls[0]["model"] == "m1" and calls[0]["max_tokens"] == 100
    assert calls[0]["messages"][0]["content"] == "Please note my phone is [PHONE_1], confirm it back."


def test_results_contain_no_text_values_or_key():
    dumped = json.dumps(_run(_fake_groq("Your number is [PHONE_1], noted.")))
    for secret in ("9876543210", "gsk-test-key", "noted", "Please note"):
        assert secret not in dumped


def test_saved_streams_hold_placeholders_and_raw_sse():
    result = _run(_fake_groq("Your number is [PHONE_1]."), save_streams=1)
    (fixture,) = result["fixtures"]
    assert fixture["placeholders"] == {"[PHONE_1]": "9876543210"}
    assert fixture["sse"].endswith("data: [DONE]\n\n") and fixture["model"] == "m1"


def test_missing_key_exits_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        api_key_from_env()


@pytest.mark.parametrize("text, seconds", [("7.66s", 7.66), ("2m59.5s", 179.5), ("120ms", 0.12),
                                           ("1h0m1s", 3601.0), ("", 0.0), ("junk", 0.0)])
def test_parse_reset(text, seconds):
    assert parse_reset(text) == pytest.approx(seconds)
