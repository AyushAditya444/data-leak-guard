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


def test_rules_include_name_cues():
    # lowercase names have no capital letter for the model: the cue detector is a rule
    text = "mera naam rahul hai, number 9876543210"
    assert [(s.type, s.text(text)) for s in detect_rules(text)] == [
        (T.NAME, "rahul"), (T.PHONE, "9876543210")]


def _kinds(text, spans):
    return [(s.type, s.text(text)) for s in spans]


def test_address_stops_at_phone():
    text = "Flat 9, Kothrud, Pune 411038 9876543210"
    assert _kinds(text, detect_rules(text)) == [
        (T.ADDRESS, "Flat 9, Kothrud, Pune 411038"), (T.PHONE, "9876543210")]
    text = "B-204, Baner,Pune-411045,ph:9820031476"
    assert _kinds(text, detect_rules(text)) == [
        (T.ADDRESS, "B-204, Baner,Pune-411045"), (T.PHONE, "9820031476")]


def test_name_inside_address_absorbed():
    def fake_ner(text):
        start = text.find("Sai Krupa")
        return [Span(start, start + 9, T.NAME, 0.9, "ner")] if start >= 0 else []
    text = "Flat 302, Sai Krupa Apts, Kothrud, Pune"
    assert _kinds(text, build_detector(fake_ner)(text)) == [(T.ADDRESS, text)]


def test_address_next_to_email_kept_separate():
    text = "B-204, Baner, Pune, email a@b.com"
    assert _kinds(text, detect_rules(text)) == [
        (T.ADDRESS, "B-204, Baner, Pune"), (T.EMAIL, "a@b.com")]
