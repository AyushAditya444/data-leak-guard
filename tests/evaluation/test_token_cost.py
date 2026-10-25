from leakguard.detect import detect_rules
from leakguard.evaluation.samples import Sample
from leakguard.evaluation.token_cost import token_cost
from leakguard.policy import DEFAULT_POLICY


def _sample(text, sid="s1"):
    return Sample(sid, "support", "tpl", text, ())


def _cost(*texts):
    # characters stand in for tokens, so the expected numbers are easy to work out
    return token_cost([_sample(t, str(i)) for i, t in enumerate(texts)], detect_rules,
                      DEFAULT_POLICY, count=len)


def test_token_cost_counts_before_after():
    phone = "9876543210"
    cost = _cost(f"call {phone} now")
    assert cost["messages"] == 1 and cost["blocked"] == 0
    assert cost["tokens_after"] - cost["tokens_before"] == len("[PHONE_1]") - len(phone)
    assert cost["by_type"]["PHONE"] == {"values": 1, "value_tokens": 10.0,
                                        "placeholder_tokens": 9.0}


def test_token_cost_excludes_blocked():
    cost = _cost("password=Xk9#mQ2$vL7pW4zR", "call 9876543210 now")
    assert cost["blocked"] == 1 and cost["messages"] == 1
    assert cost["tokens_before"] == len("call 9876543210 now")


def test_token_cost_unchanged_message():
    cost = _cost("hello, where is my order?")
    assert cost["unchanged"] == 1 and cost["mean_change"] == 0.0
    assert cost["tokens_before"] == cost["tokens_after"]
