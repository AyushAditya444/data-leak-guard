"""Turn an NER model's per-token labels into NAME spans over the original text.

A model labels *tokens* (word pieces like "Venk" + "atesh"), each with character offsets
into the text. Most models use BIO tags: B-PER begins a person's name, I-PER continues it,
O is anything else. This module groups person tokens into names, widens each name to whole
words, and drops names the model wasn't confident about.
"""

import re
from collections.abc import Collection, Sequence

from leakguard.detect.spans import Span
from leakguard.entities import EntityType

# What may sit between two tokens of one name: spaces, the dot of an initial ("R. K."),
# an apostrophe ("D'Souza") or a hyphen. Never a newline: a name doesn't span two lines.
_NAME_GAP = re.compile(r"[ .'-]{1,3}")
_STRIP = " \t\r\n.,;:!?()[]{}<>\"'-"


def person_label_ids(id2label: dict[int, str]) -> tuple[frozenset[int], frozenset[int]]:
    """(begin ids, inside ids) of the person labels. A label without BIO prefix ("person")
    counts as both: each such token may start a name or continue one."""
    begin, inside = set(), set()
    for label_id, label in id2label.items():
        name = label.upper()
        if name in ("B-PER", "B-PERSON"):
            begin.add(int(label_id))
        elif name in ("I-PER", "I-PERSON"):
            inside.add(int(label_id))
        elif name in ("PER", "PERSON"):
            begin.add(int(label_id))
            inside.add(int(label_id))
    if not begin:
        raise ValueError(f"No person label in the model's labels: {sorted(id2label.values())}")
    return frozenset(begin), frozenset(inside)


def _continues(text: str, prev_end: int, start: int, is_inside: bool) -> bool:
    """Does a person token starting at `start` continue the name that ended at `prev_end`?"""
    gap = text[prev_end:start]
    if gap == "":
        return True  # another piece of the same word
    return is_inside and _NAME_GAP.fullmatch(gap) is not None


def _is_word_char(text: str, i: int) -> bool:
    if text[i].isalnum():
        return True
    # an apostrophe between letters belongs to the word: D'Souza
    return text[i] == "'" and 0 < i < len(text) - 1 and text[i - 1].isalpha() \
        and text[i + 1].isalpha()


def _whole_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) so it never cuts a word: masking "Venk" would leak "atesh"."""
    while start > 0 and _is_word_char(text, start - 1):
        start -= 1
    while end < len(text) and _is_word_char(text, end):
        end += 1
    return start, end


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _STRIP:
        start += 1
    while end > start and text[end - 1] in _STRIP:
        end -= 1
    return start, end


def decode_names(
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    scores: Sequence[float],
    begin_ids: Collection[int],
    inside_ids: Collection[int],
    threshold: float,
) -> list[Span]:
    """NAME spans from per-token predictions. `scores[i]` is token i's probability of
    being part of a person's name; a name is kept if its tokens' mean score >= threshold."""
    groups: list[list[int]] = []
    prev_end: int | None = None
    for i, label in enumerate(label_ids):
        start, end = offsets[i]
        if start == end or (label not in begin_ids and label not in inside_ids):
            continue
        if prev_end is not None and _continues(text, prev_end, start, label in inside_ids):
            groups[-1].append(i)
        else:
            groups.append([i])
        prev_end = end  # a word between two person tokens fails _NAME_GAP, splitting them

    spans: list[Span] = []
    for group in groups:
        score = sum(scores[i] for i in group) / len(group)
        if score < threshold:
            continue
        start, end = _whole_words(text, offsets[group[0]][0], offsets[group[-1]][1])
        start, end = _strip(text, start, end)
        if start < end:
            spans.append(Span(start, end, EntityType.NAME, score, detector="ner"))
    return spans
