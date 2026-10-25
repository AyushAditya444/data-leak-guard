import json
import random

import pytest

from leakguard.entities import EntityType as T
from leakguard.masking import PlaceholderMap, mask_messages
from leakguard.placeholders import max_placeholder_len
from leakguard.policy import DEFAULT_POLICY
from leakguard.restore import Restorer, restore_response


def _user(content):
    return [{"role": "user", "content": content}]


def _restorer(*pairs, bracketless=False):  # pairs of (EntityType, value), numbered in order
    m = PlaceholderMap()
    for entity, value in pairs:
        m.placeholder_for(entity, value)
    return Restorer(m, bracketless=bracketless)


def test_restore_round_trip():
    r = mask_messages(_user("upi rahul.s@okicici"), DEFAULT_POLICY)
    assert Restorer(r.mapping).restore_text("refund to [UPI_1] done") == \
        "refund to rahul.s@okicici done"


def test_user_typed_placeholder_is_never_restored():
    r = mask_messages(_user("what is [PHONE_1]? my number is 9876543210"), DEFAULT_POLICY)
    assert r.messages[0]["content"] == "what is [PHONE_1]? my number is [PHONE_2]"
    assert Restorer(r.mapping).restore_text("[PHONE_1] / [PHONE_2] / [PHONE_9]") == \
        "[PHONE_1] / 9876543210 / [PHONE_9]"


def test_restore_text_counts_every_form():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("call [phone 1] or {PHONE_1}") == "call 9876543210 or 9876543210"
    assert (r.stats.restored, r.stats.unrestored) == (2, 0)


def test_reserved_never_restored_in_any_form():
    res = mask_messages(_user("what is [PHONE_1]? my number is 9876543210"), DEFAULT_POLICY)
    r = Restorer(res.mapping, bracketless=False)
    assert r.restore_text("[phone 1] {PHONE-2}") == "[phone 1] 9876543210"
    assert (r.stats.restored, r.stats.unrestored) == (1, 0)


def test_unknown_placeholder_left_and_not_counted():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("[EMAIL_1] [PHONE_9]") == "[EMAIL_1] [PHONE_9]"
    assert (r.stats.restored, r.stats.unrestored) == (0, 0)


def test_unrestored_counts_bare_and_unclosed():
    r = _restorer((T.PHONE, "9876543210"))
    assert r.restore_text("call PHONE_1 or [PHONE_1 now") == "call PHONE_1 or [PHONE_1 now"
    assert (r.stats.restored, r.stats.unrestored) == (0, 2)


def test_bracketless_restores_when_enabled():
    r = _restorer((T.PHONE, "9876543210"), bracketless=True)
    assert r.restore_text("call PHONE_1 now") == "call 9876543210 now"


def test_json_escaped_into_tool_arguments():
    address = 'Flat 2,\n"Sai" Apts, Kothrud, Pune 411038'
    r = _restorer((T.ADDRESS, address))
    out = r.restore_text('{"a": "[ADDRESS_1]"}', json_escape=True)
    assert json.loads(out)["a"] == address


def test_restore_response_all_fields():
    address = 'Flat 2,\n"Sai" Apts, Pune 411038'
    r = _restorer((T.PHONE, "9876543210"), (T.ADDRESS, address))
    body = {"id": "x", "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "call [PHONE_1]", "refusal": "not [Phone_1]",
        "reasoning": "user gave [PHONE 1]",
        "tool_calls": [{"id": "t", "type": "function", "function": {
            "name": "f", "arguments": '{"p": "[PHONE_1]", "a": "[ADDRESS_1]"}'}}]}}]}
    msg = restore_response(body, r)["choices"][0]["message"]
    assert msg["content"] == "call 9876543210"
    assert msg["refusal"] == "not 9876543210"
    assert msg["reasoning"] == "user gave 9876543210"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == \
        {"p": "9876543210", "a": address}
    assert r.stats.restored == 5


def test_ordinary_brackets_pass_through():  # Review Focus 1
    r = _restorer((T.NAME, "Ravi"))
    text = 'See [docs](http://x), arr[0], (1), <br>, {x}, [NOTE_1] and {"k": [1]}'
    f = r.field()
    assert f.feed(text) + f.flush() == text
    for cut in range(len(text)):  # whatever the chunking, little is ever held back
        g = r.field()
        released = g.feed(text[:cut])
        assert cut - len(released) <= max_placeholder_len({T.NAME})


def test_adjacent_and_repeated_placeholders():  # Review Focus 2
    r = _restorer((T.NAME, "Ravi"), (T.PHONE, "9876543210"))
    assert r.restore_text("[NAME_1][PHONE_1]23 " + "[NAME_1]" * 50) == \
        "Ravi987654321023 " + "Ravi" * 50


def _all_two_cut_chunkings(text):
    return [[text[:i], text[i:j], text[j:]]
            for i in range(len(text) + 1) for j in range(i, len(text) + 1)]


def _random_chunkings(text, seed, n):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pieces, i = [], 0
        while i < len(text):
            step = rng.randint(1, 4)
            pieces.append(text[i:i + step])
            i += step
        out.append(pieces)
    return out


TEXTS = ["call [PHONE_1] now", "[Name 1], hi {NAME_1}!", "x[PH", "[[NAME_1]]", "a [NAME_1",
         "NAME_1 [phone-1]", "tail [PHONE_1]", "", "XNAME_1 NAME_1 (name_1) PHONE_1x",
         "[PHONE_[PHONE_1]"]


@pytest.mark.parametrize("bracketless", [False, True])
@pytest.mark.parametrize("text", TEXTS)
def test_streaming_equals_whole_text_for_every_cut(text, bracketless):
    pairs = ((T.NAME, "Ravi"), (T.PHONE, "9876543210"))
    whole = _restorer(*pairs, bracketless=bracketless)
    expected = whole.restore_text(text)
    for chunks in _all_two_cut_chunkings(text) + _random_chunkings(text, seed=0, n=50):
        r = _restorer(*pairs, bracketless=bracketless)
        f = r.field()
        out = "".join(f.feed(c) for c in chunks) + f.flush()
        f.close()
        assert out == expected, chunks
        assert r.stats == whole.stats, chunks


def test_nothing_held_without_opener():
    assert _restorer((T.NAME, "Ravi")).field().feed("hello world") == "hello world"


def test_hold_is_bounded():
    f = _restorer((T.PHONE, "9876543210")).field()
    assert f.feed("ok [PHONE_123") == "ok "
    assert f.feed("4567") == "[PHONE_1234567"  # 7 digits: can never be a placeholder


def test_flush_twice_and_close_counts_once():
    r = _restorer((T.PHONE, "9876543210"))
    f = r.field()
    assert f.feed("x PHONE_") == "x PHONE_"
    assert f.feed("1 [PH") == "1 "
    assert f.flush() == "[PH" and f.flush() == ""
    f.close()
    assert (r.stats.restored, r.stats.unrestored) == (0, 1)
