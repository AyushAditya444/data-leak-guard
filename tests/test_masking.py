from leakguard.entities import EntityType as T
from leakguard.masking import PlaceholderMap, mask_messages, restore_response
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


def test_restore_round_trip():
    r = mask_messages(_user("upi rahul.s@okicici"), DEFAULT_POLICY)
    assert r.mapping.restore("refund to [UPI_1] done") == "refund to rahul.s@okicici done"


def test_user_typed_placeholder_is_never_restored():
    r = mask_messages(_user("what is [PHONE_1]? my number is 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "what is [PHONE_1]? my number is [PHONE_2]"
    assert r.mapping.restore("[PHONE_1] / [PHONE_2] / [PHONE_9]") == \
        "[PHONE_1] / 9876543210 / [PHONE_9]"


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


def test_restore_response_body():
    m = PlaceholderMap()
    ph = m.placeholder_for(T.PHONE, "9876543210")
    body = {"choices": [{"message": {
        "role": "assistant", "content": f"call {ph}",
        "tool_calls": [{"function": {"name": "f", "arguments": f'{{"p": "{ph}"}}'}}]}}]}
    out = restore_response(body, m)
    assert out["choices"][0]["message"]["content"] == "call 9876543210"
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == \
        '{"p": "9876543210"}'
