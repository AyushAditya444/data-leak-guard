"""Names the model can't see: lowercase ("mera naam rahul hai") and ALL CAPS ("NAME: RAHUL
SHARMA"). NER models lean heavily on capital letters, so on dev the chosen model found only
14% of lowercase and 44% of all-caps names (docs/evaluation/ner-comparison.md, model alone).

This detector needs no name list. It only fires in a slot right after (or before) a phrase
that announces a name, and the slot must be 1-3 plain words with none of them a common
word: "main theek hoon" ("I'm fine") is not a name. Everything else is left to the model.
"""

import re

from leakguard.detect.spans import Span
from leakguard.entities import EntityType

_MAX_WORDS = 3
_SCORE = 0.6  # below a confident model hit; equal-priority overlaps keep the longer span

# Words that end or rule out a name slot (Hinglish + English). Lowercase.
_STOP_WORDS = """
hai h he hain hoon hun hu hoo aur and ka ki ke ko se ne pe par mein me main mai bhi ji
ha haan nahi nhi na kya kyu kyon kaise kab kahan toh to yeh ye woh wo bas abhi yahan wahan
theek thik pareshan ghar bahut busy ready free online offline here from is was the a an not
wrong correct spelled misspelled missing changed incorrect incorrectly updated also still
on in at of for with by my your our his her their fine ok okay sorry back sir madam mam
bhai bhaiya didi sahab please pls same different new old
"""
_STOP = frozenset(_STOP_WORDS.split())
_ENDS_HOON = frozenset({"hoon", "hun", "hu", "hoo"})

_WORD = re.compile(r"[^\W\d_]+")
_SPACES = re.compile(r"[ \t]+")

# Phrases followed by the name. (pattern, the word that must follow the slot, or None)
_BEFORE_NAME = (
    # mera naam rahul hai / my name is rahul / my name rahul
    (re.compile(r"\b(?:mera|meraa|mere|my)\s+(?:naam|name)\s+(?:is\s+)?", re.IGNORECASE), None),
    # main anjali hoon ("I am anjali"): only with the closing "hoon"
    (re.compile(r"\b(?:main|mai)\s+", re.IGNORECASE), _ENDS_HOON),
    # record fields: "Name: ...", "Cust name - ...", "| APPLICANT NAME - ...". The field must
    # start a line or follow a separator, so "Bank name: SBI" or "file name: x.pdf" don't count.
    (re.compile(r"(?:^|(?<=[|,;(/]))[ \t]*"
                r"(?:(?:full|cust|customer|applicant|account[ \t]+holder|holder|beneficiary"
                r"|nominee|contact|father'?s?|mother'?s?|guardian)[ \t]+)?"
                r"(?:name|naam)[ \t]*[:\-–][ \t]*", re.IGNORECASE | re.MULTILINE), None),
)
# Phrases preceded by the name: "suraj kumar bol raha hu"
_AFTER_NAME = re.compile(
    r"[ \t]+(?:bol|bool)[ \t]+(?:raha|rahi|rha|rhi)[ \t]+(?:hoon|hun|hu|hoo|h)\b", re.IGNORECASE)


def _words_forward(text: str, pos: int) -> list[re.Match]:
    """Up to _MAX_WORDS non-stop words starting at `pos`, separated only by spaces."""
    words: list[re.Match] = []
    while len(words) < _MAX_WORDS:
        m = _WORD.match(text, pos)
        if m is None or m.group().lower() in _STOP:
            break
        words.append(m)
        gap = _SPACES.match(text, m.end())
        if gap is None:
            break
        pos = gap.end()
    return words


def _next_word(text: str, pos: int) -> str:
    gap = _SPACES.match(text, pos)
    m = _WORD.match(text, gap.end() if gap else pos)
    return m.group().lower() if m else ""


def _words_backward(text: str, end: int) -> tuple[int, int] | None:
    """The trailing run (up to _MAX_WORDS) of non-stop words that ends exactly at `end`."""
    line_start = text.rfind("\n", 0, end) + 1
    words = list(_WORD.finditer(text, line_start, end))
    run: list[re.Match] = []
    for m in reversed(words):
        # the nearest word touches the cue (which starts with the spaces); earlier ones
        # must be separated from the run by spaces only
        touches = m.end() == end if not run else \
            _SPACES.fullmatch(text, m.end(), run[0].start()) is not None
        if m.group().lower() in _STOP or len(run) == _MAX_WORDS or not touches:
            break
        run.insert(0, m)
    return (run[0].start(), run[-1].end()) if run else None


def detect_name_cues(text: str) -> list[Span]:
    spans: list[Span] = []
    for cue, must_end_with in _BEFORE_NAME:
        for m in cue.finditer(text):
            words = _words_forward(text, m.end())
            if not words:
                continue
            if must_end_with and _next_word(text, words[-1].end()) not in must_end_with:
                continue
            spans.append(Span(words[0].start(), words[-1].end(), EntityType.NAME, _SCORE,
                              "name_cues"))
    for m in _AFTER_NAME.finditer(text):
        run = _words_backward(text, m.start())
        if run:
            spans.append(Span(run[0], run[1], EntityType.NAME, _SCORE, "name_cues"))
    return spans
