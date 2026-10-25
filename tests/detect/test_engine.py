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


def test_ner_locations_reach_address():
    from leakguard.ner.model import Entities
    text = "Flat 2, Hiranandani Meadows"  # an invented locality: not in the PIN directory
    start = text.index("Hiranandani")

    def fake_ner(t):
        return Entities(names=[], locations=[(start, len(t))] if t == text else [])

    assert _kinds(text, build_detector(name_model=None)(text)) == []
    assert _kinds(text, build_detector(ner=fake_ner)(text)) == [(T.ADDRESS, text)]


def test_name_partly_inside_address_is_absorbed():
    # A surname can also be a village name ("Sharma"), so an address may start inside a
    # name. Merging would then drop the NAME and leak "Kavya": the address takes it in.
    text = "customer Kavya Sharma, Flat 302, Kothrud, Pune"
    start = text.index("Kavya")

    def fake_ner(t):
        return [Span(start, start + 12, T.NAME, 0.9, "ner")] if t == text else []

    spans = build_detector(fake_ner)(text)
    covered = {i for s in spans for i in range(s.start, s.end)}
    assert set(range(start, len(text))) <= covered  # nothing from "Kavya" on is visible


def _hidden(text, spans, value):
    """Is every character of `value` (inside `text`) covered by some span?"""
    start = text.index(value)
    covered = {i for s in spans for i in range(s.start, s.end)}
    return set(range(start, start + len(value))) <= covered


def test_address_glued_to_a_phone_is_not_dropped():
    # Widening to whole words must not reach into the phone: merging would then keep
    # the PHONE and drop the whole address.
    for text, address in [
        ("Flat 2, Kothrud, Pune 411038ph9876543210", "Flat 2, Kothrud, Pune 411038"),
        ("Flat 2, Kothrud, Pune 411038Mob9876543210", "Flat 2, Kothrud, Pune 411038"),
        ("Flat 2, Kothrud, near Hanuman Mandir9876543210", "Flat 2, Kothrud, near Hanuman Mandir"),
    ]:
        spans = detect_rules(text)
        assert T.PHONE in {s.type for s in spans}, text
        assert _hidden(text, spans, address), text


def test_phone_inside_an_address_does_not_split_it():
    # Both halves of the address stay masked, the phone in between stays a PHONE.
    for text, parts in [
        ("Flat 302, Green Valley Society, ph 9876543210, Kothrud, Pune 411038",
         ["Flat 302, Green Valley Society", "Kothrud, Pune 411038"]),
        ("Deliver to H.No 45, MG Road, call 9876543210, Kothrud, Pune",
         ["H.No 45, MG Road", "Kothrud, Pune"]),
        ("H.No 45, MG Road (9876543210) Baner, Pune", ["H.No 45, MG Road", "Baner, Pune"]),
    ]:
        spans = detect_rules(text)
        assert T.PHONE in {s.type for s in spans}, text
        assert all(_hidden(text, spans, part) for part in parts), text
        assert all(s.type in (T.ADDRESS, T.PHONE) for s in spans), text
