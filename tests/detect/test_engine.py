from leakguard.detect.engine import build_detector, detect_rules
from leakguard.detect.spans import Span
from leakguard.entities import EntityType as T


def _fake_names(text):
    """Flags every occurrence of 'rahul' (any case), as a careless NER model might."""
    spans, start = [], text.lower().find("rahul")
    while start != -1:
        spans.append(Span(start, start + 5, T.NAME, 0.9, "ner"))
        start = text.lower().find("rahul", start + 1)
    return spans


def test_build_detector_adds_names_and_rules_win_overlaps():
    text = "Rahul wrote from rahul.sharma@gmail.com"
    spans = build_detector(_fake_names)(text)
    assert [(s.type, s.text(text)) for s in spans] == [
        (T.NAME, "Rahul"), (T.EMAIL, "rahul.sharma@gmail.com")]


def test_build_detector_without_names_is_rules_only():
    text = "I am Rahul, mail rahul@gmail.com"
    assert build_detector(None)(text) == detect_rules(text)
