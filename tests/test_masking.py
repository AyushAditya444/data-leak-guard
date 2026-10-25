import pytest

from leakguard.entities import EntityType as T
from leakguard.masking import mask_messages
from leakguard.policy import DEFAULT_POLICY, Action, Policy


def _user(content):
    return [{"role": "user", "content": content}]


def test_masks_and_numbers_in_order_of_appearance():
    msgs = [{"role": "system", "content": "Bot"},
            {"role": "user", "content": "I'm at 9876543210 and pan ABCPS1234K"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "also 9123456789, and again 9876543210"}]
    r = mask_messages(msgs, DEFAULT_POLICY)
    assert r.messages[1]["content"] == "I'm at [PHONE_1] and pan [PAN_1]"
    assert r.messages[3]["content"] == "also [PHONE_2], and again [PHONE_1]"
    assert msgs[1]["content"] == "I'm at 9876543210 and pan ABCPS1234K"  # input untouched
    assert r.counts[(T.PHONE, Action.MASK)] == 3 and r.blocked == []


def test_same_value_different_format_same_placeholder():
    r = mask_messages(_user("98765 43210 or +91 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "[PHONE_1] or [PHONE_1]"


def test_reserved_numbers_cover_every_accepted_form():
    r = mask_messages(_user("what is [phone 1] and {PHONE-2}? my number is 9876543210"),
                      DEFAULT_POLICY)
    assert r.messages[0]["content"].endswith("my number is [PHONE_3]")
    assert r.mapping.value_for(T.PHONE, 1) is None
    assert r.mapping.value_for(T.PHONE, 3) == "9876543210"
    assert r.mapping.entities() == {T.PHONE}


def test_secret_blocks_and_log_only_leaves_text():
    pol = Policy(actions={**DEFAULT_POLICY.actions, T.EMAIL: Action.LOG_ONLY})
    r = mask_messages(_user("mail a@b.com key sk-abcdefghijklmnopqrstuvwxyz12"), pol)
    assert r.blocked == [T.SECRET]
    assert "a@b.com" in r.messages[0]["content"]
    assert r.counts[(T.EMAIL, Action.LOG_ONLY)] == 1


def test_masks_content_parts_and_tool_call_arguments():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "pan ABCPS1234K"},
                                         {"type": "image_url", "image_url": {"url": "x"}}]},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "f", "arguments": '{"phone": "9876543210"}'}}]}]
    r = mask_messages(msgs, DEFAULT_POLICY)
    assert r.messages[0]["content"][0]["text"] == "pan [PAN_1]"
    assert r.messages[0]["content"][1] == msgs[0]["content"][1]
    assert r.messages[1]["tool_calls"][0]["function"]["arguments"] == '{"phone": "[PHONE_1]"}'


def test_unicode_lookalike_in_request_still_masks():  # final review C1
    r = mask_messages(_user("[EMA\u0130L 1] call 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "[EMA\u0130L 1] call [PHONE_1]"


@pytest.mark.parametrize("typed", ["phone 1", "PHONE 1", "phone_1", "[phone 1", "Phone-1"])
def test_near_forms_a_user_types_are_reserved(typed):  # final review I3
    r = mask_messages(_user(f"{typed}; my number is 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"].endswith("my number is [PHONE_2]")
    assert r.mapping.value_for(T.PHONE, 1) is None


def test_reserving_many_typed_placeholders_is_fast():  # final review I1
    import time
    started = time.perf_counter()
    mask_messages(_user("[PHONE_1] PHONE_2 " * 10_000), DEFAULT_POLICY)
    assert time.perf_counter() - started < 2.0
