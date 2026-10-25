"""The Presidio baseline's pure core, tested with a fake analyser (no Presidio in CI)."""

from leakguard.entities import EntityType as T
from leakguard.evaluation.presidio_baseline import (
    BOTH_SUPPORT,
    GRID,
    MAPPING,
    UNSUPPORTED,
    Found,
    PresidioDetector,
    apply_thresholds,
    resolve_conflicts,
    tune_thresholds,
)
from leakguard.evaluation.samples import Sample, parse_marked


def _sample(marked, sid="s1", category="support"):
    text, spans = parse_marked(marked)
    return Sample(sid, category, "tpl", text, spans)


def test_mapping_matches_spec():
    assert MAPPING == {"IN_AADHAAR": T.AADHAAR, "IN_PAN": T.PAN, "PHONE_NUMBER": T.PHONE,
                       "EMAIL_ADDRESS": T.EMAIL, "CREDIT_CARD": T.CARD, "PERSON": T.NAME,
                       "LOCATION": T.ADDRESS}
    assert set(UNSUPPORTED) == {T.UPI, T.IFSC, T.BANK_ACCOUNT, T.SECRET}
    assert set(UNSUPPORTED) | set(MAPPING.values()) == set(T)
    assert set(BOTH_SUPPORT) == {T.AADHAAR, T.PAN, T.PHONE, T.EMAIL, T.CARD, T.NAME}
    assert GRID[0] == 0.0 and GRID[-1] == 0.95 and len(GRID) == 20


def test_thresholds_drop_below_and_keep_untuned():
    found = [Found("PERSON", 0, 5, 0.4), Found("PERSON", 6, 9, 0.6), Found("URL", 10, 20, 0.1)]
    kept = apply_thresholds(found, {"PERSON": 0.5})
    assert kept == [Found("PERSON", 6, 9, 0.6), Found("URL", 10, 20, 0.1)]


def test_resolve_same_type_overlap_merges():
    kept, n = resolve_conflicts([Found("PERSON", 0, 5, 0.6), Found("PERSON", 3, 9, 0.8)],
                                "x" * 20)
    assert kept == [Found("PERSON", 0, 9, 0.8)] and n == 1


def test_resolve_contained_any_type_removed():
    kept, n = resolve_conflicts([Found("PHONE_NUMBER", 10, 20, 0.9),
                                 Found("CREDIT_CARD", 5, 25, 0.5)], "x" * 30)
    assert kept == [Found("CREDIT_CARD", 5, 25, 0.5)] and n == 1


def test_resolve_identical_spans_keep_higher_score():
    kept, _ = resolve_conflicts([Found("IN_AADHAAR", 0, 14, 0.5),
                                 Found("PHONE_NUMBER", 0, 14, 0.4)], "x" * 14)
    assert kept == [Found("IN_AADHAAR", 0, 14, 0.5)]
    kept, _ = resolve_conflicts([Found("IN_AADHAAR", 0, 14, 0.4),
                                 Found("PHONE_NUMBER", 0, 14, 0.4)], "x" * 14)
    assert kept == [Found("PHONE_NUMBER", 0, 14, 0.4)]  # a tie keeps the later one


def test_resolve_space_separated_same_type_merges():
    text = "I am Rahul Sharma"
    kept, n = resolve_conflicts([Found("PERSON", 5, 10, 0.85), Found("PERSON", 11, 17, 0.7)],
                                text)
    # Presidio's anonymiser keeps the second result's score when it joins the two
    assert kept == [Found("PERSON", 5, 17, 0.7)] and n == 1
    kept, _ = resolve_conflicts([Found("PERSON", 5, 10, 0.85), Found("PERSON", 11, 17, 0.7)],
                                "I am Rahul,Sharma")
    assert len(kept) == 2  # something other than spaces in between: not merged


def test_resolve_drops_empty_and_out_of_range():
    kept, _ = resolve_conflicts([Found("PERSON", 3, 3, 0.9), Found("PERSON", 5, 50, 0.9),
                                 Found("PERSON", -1, 2, 0.9), Found("URL", 0, 4, 0.5)], "x" * 10)
    assert kept == [Found("URL", 0, 4, 0.5)]


def test_detector_maps_and_counts_unmapped():
    text = "mail a@b.in on 2026-10-25"
    found = [Found("EMAIL_ADDRESS", 5, 11, 1.0), Found("URL", 7, 11, 0.5),
             Found("DATE_TIME", 15, 25, 0.85)]
    det = PresidioDetector(lambda t: found)
    spans = det.detect(text)
    assert [(s.type, s.start, s.end, s.detector) for s in spans] == [(T.EMAIL, 5, 11, "presidio")]
    assert (15, 25) in det.cover(text)
    assert det.unmapped == {"DATE_TIME": 1}  # the URL inside the email was resolved away
    assert det.conflicts == 1


def test_detector_analyses_each_text_once():
    calls = []

    def analyze(text):
        calls.append(text)
        return [Found("PERSON", 0, 5, 0.85)]

    det = PresidioDetector(analyze)
    det.detect("Rahul here")
    det.cover("Rahul here")
    det.detect("Rahul here")
    assert calls == ["Rahul here"]
    assert det.unmapped == {} and det.conflicts == 0


def test_tuning_picks_best_f1_tie_to_higher():
    named = _sample("hi [[NAME:Rahul]] here", "a")
    negative = _sample("hello there", "b", "negative")

    def analyze(text):
        if text == named.text:
            return [Found("PERSON", 3, 8, 0.9)]
        return [Found("PERSON", 0, 5, 0.25)]  # a false alarm with a low score

    thresholds, rows = tune_thresholds([named, negative], analyze)
    # 0.0-0.25 keep the false alarm (F1 0.67); 0.3-0.9 tie at F1 1.0 -> the highest, 0.9
    assert thresholds["PERSON"] == 0.9
    person = {r.threshold: r for r in rows if r.presidio_type == "PERSON"}
    assert person[0.25].fp == 1 and person[0.3].fp == 0 and person[0.95].tp == 0
    assert set(thresholds) == set(MAPPING)


def test_evaluation_pins_math_libraries_to_one_thread():
    # Latency is reported "on one CPU thread": numpy/BLAS (used by spaCy) must not fan out.
    import os

    from leakguard import evaluation

    for name in evaluation.THREAD_VARS:
        assert name in os.environ
    assert "OMP_NUM_THREADS" in evaluation.THREAD_VARS
