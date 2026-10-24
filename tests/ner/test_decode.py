import pytest

from leakguard.entities import EntityType as T
from leakguard.ner.decode import decode_names, person_label_ids

# dslim/bert-base-NER's label set
ID2LABEL = {0: "O", 1: "B-MISC", 2: "I-MISC", 3: "B-PER", 4: "I-PER", 5: "B-ORG", 6: "I-ORG",
            7: "B-LOC", 8: "I-LOC"}
O, B, I = 0, 3, 4  # label ids: outside, begin-person, inside-person


def _offsets(text, pieces):
    """Character offsets of each token piece, found left to right (like a tokenizer's)."""
    out, cursor = [], 0
    for piece in pieces:
        start = text.index(piece, cursor)
        out.append((start, start + len(piece)))
        cursor = start + len(piece)
    return out


def _decode(text, pieces, labels, scores=None, id2label=ID2LABEL, threshold=0.5):
    begin, inside = person_label_ids(id2label)
    scores = scores or [0.99] * len(pieces)
    return decode_names(text, _offsets(text, pieces), labels, scores, begin, inside, threshold)


def _texts(text, spans):
    return [s.text(text) for s in spans]


def test_bio_tokens_become_one_span():
    text = "I am Rahul Sharma"
    spans = _decode(text, ["I", "am", "Rahul", "Sharma"], [O, O, B, I])
    assert _texts(text, spans) == ["Rahul Sharma"]
    assert spans[0].type is T.NAME and spans[0].detector == "ner"


def test_span_expands_to_whole_word():
    # The model tagged only the first word piece; masking "Venk" alone would leak "atesh".
    text = "I am Venkatesh"
    assert _texts(text, _decode(text, ["I", "am", "Venk", "atesh"], [O, O, B, O])) == [
        "Venkatesh"]


def test_two_adjacent_names_split_on_begin():
    text = "Rahul Priya"
    assert _texts(text, _decode(text, ["Rahul", "Priya"], [B, B])) == ["Rahul", "Priya"]


def test_names_on_different_lines_are_separate():
    text = "Rahul\nPriya"
    assert _texts(text, _decode(text, ["Rahul", "Priya"], [B, I])) == ["Rahul", "Priya"]


def test_low_score_group_dropped():
    text = "I am Rahul"
    assert _decode(text, ["I", "am", "Rahul"], [O, O, B], scores=[0.9, 0.9, 0.4]) == []


def test_group_score_is_mean_of_tokens():
    text = "Rahul Sharma"
    [span] = _decode(text, ["Rahul", "Sharma"], [B, I], scores=[0.9, 0.7])
    assert span.score == pytest.approx(0.8)


def test_non_bio_person_label():
    id2label = {0: "O", 1: "location", 2: "person"}
    assert person_label_ids(id2label) == (frozenset({2}), frozenset({2}))
    text = "met Priya Nair today"
    spans = _decode(text, ["met", "Priya", "Nair", "today"], [0, 2, 2, 0], id2label=id2label)
    assert _texts(text, spans) == ["Priya Nair"]


def test_no_person_label_raises():
    with pytest.raises(ValueError, match="person"):
        person_label_ids({0: "O", 1: "B-LOC"})


def test_initials_kept_whole():
    text = "Call R. K. Narayan now"
    pieces = ["Call", "R", ".", "K", ".", "Narayan", "now"]
    assert _texts(text, _decode(text, pieces, [O, B, O, I, O, I, O])) == ["R. K. Narayan"]


def test_apostrophe_name_kept_whole():
    text = "Ask D'Souza"
    assert _texts(text, _decode(text, ["Ask", "D", "'", "Souza"], [O, B, O, I])) == ["D'Souza"]


def test_name_inside_email_stays_inside_the_word_part():
    # Merging later drops it in favour of the EMAIL; decode must not swallow the domain.
    text = "mail rahul.sharma@gmail.com"
    pieces = ["mail", "rahul", ".", "sharma", "@", "gmail", ".", "com"]
    assert _texts(text, _decode(text, pieces, [O, B, O, O, O, O, O, O])) == ["rahul"]
