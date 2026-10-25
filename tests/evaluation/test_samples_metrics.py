import json

import pytest

from leakguard.detect.spans import Span
from leakguard.entities import EntityType as T
from leakguard.evaluation.metrics import evaluation_from_json, evaluation_to_json, score
from leakguard.evaluation.samples import Sample, parse_marked, read_jsonl, write_jsonl


def _sample(marked, category="support", sid="s1"):
    text, spans = parse_marked(marked)
    return Sample(sid, category, "tpl", text, spans)


def test_parse_marked():
    text, spans = parse_marked("call [[PHONE:98765 43210]] or mail [[EMAIL:a@b.in]]")
    assert text == "call 98765 43210 or mail a@b.in"
    assert [(s.type, text[s.start:s.end]) for s in spans] == [
        (T.PHONE, "98765 43210"), (T.EMAIL, "a@b.in")]
    with pytest.raises(ValueError):
        parse_marked("[[PASSPORT:X1]]")


def test_jsonl_round_trip(tmp_path):
    s = _sample("₹500 to [[UPI:ravi@ybl]]\nthanks")
    write_jsonl(tmp_path / "x.jsonl", [s])
    assert read_jsonl(tmp_path / "x.jsonl") == [s]


def test_overlap_same_type_is_tp_and_coverage_counts_chars():
    s = _sample("a/c [[BANK_ACCOUNT:50100123456789]]")
    gold = s.spans[0]
    partial = Span(gold.start, gold.end - 4, T.BANK_ACCOUNT)
    ev = score([s], lambda text: [partial])
    st = ev.by_type[T.BANK_ACCOUNT]
    assert (st.tp, st.fp, st.fn, st.exact) == (1, 0, 0, 0)
    assert st.coverage == pytest.approx(10 / 14)


def test_wrong_type_is_fp_and_fn_but_still_covers():
    s = _sample("a/c [[BANK_ACCOUNT:9876543210]]")
    g = s.spans[0]
    ev = score([s], lambda text: [Span(g.start, g.end, T.PHONE)])
    assert ev.by_type[T.BANK_ACCOUNT].fn == 1 and ev.by_type[T.PHONE].fp == 1
    assert ev.by_type[T.BANK_ACCOUNT].coverage == 1.0


def test_one_prediction_matches_one_gold():
    s = _sample("[[PHONE:9876543210]] [[PHONE:9123456789]]")
    whole = Span(0, len(s.text), T.PHONE)
    st = score([s], lambda text: [whole]).by_type[T.PHONE]
    assert (st.tp, st.fn) == (1, 1)


def test_negatives_fp_rate_and_errors():
    neg = _sample("order 234567890124", category="negative", sid="n1")
    ev = score([neg], lambda text: [Span(6, 18, T.AADHAAR)])
    assert (ev.negatives_total, ev.negatives_with_fp) == (1, 1)
    assert ev.errors[0].kind == "FP" and ev.errors[0].value == "234567890124"


def test_score_propagates_detector_errors():
    def broken(text):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        score([_sample("hi")], broken)


def test_cover_extends_coverage_not_typed_scores():
    s = _sample("mail [[EMAIL:a@b.in]] now")
    g = s.spans[0]
    ev = score([s], lambda text: [], cover=lambda text: [(g.start, g.end)])
    st = ev.by_type[T.EMAIL]
    assert st.coverage == 1.0 and st.recall == 0.0 and st.fp == 0


def test_cover_ranges_count_each_char_once():
    s = _sample("mail [[EMAIL:a@b.in]] now")
    g = s.spans[0]
    pred = Span(g.start, g.end, T.EMAIL)
    ev = score([s], lambda text: [pred],
               cover=lambda text: [(g.start, g.end), (g.start - 2, g.end + 1)])
    st = ev.by_type[T.EMAIL]
    assert st.covered_chars == st.gold_chars == g.end - g.start


def test_evaluation_json_round_trip():
    s = _sample("call [[PHONE:98765 43210]] or [[EMAIL:a@b.in]]")
    ev = score([s, _sample("order 12345", "negative", "n1")],
               lambda text: [Span(5, 16, T.PHONE)] if "call" in text else [])
    data = json.loads(json.dumps(evaluation_to_json(ev)))
    back = evaluation_from_json(data)
    assert back.by_type == ev.by_type
    assert (back.negatives_total, back.negatives_with_fp) == (1, 0)
