"""Find the pieces an Indian address is made of.

Each finder marks one kind of piece:

    unit       Flat 302, H.No. 45, Plot 7, 3rd floor      (bare: 12/3, B-204, #45)
    building   Sai Krupa Apts, Green Valley Society
    street     MG Road, Nehru Marg, gali no 4, Sector 15, 2nd Cross
    landmark   near Hanuman Mandir, opp. SBI ATM, Shiv Mandir ke paas
    pin        411038 (a real PIN, not after "OTP", "order"...)
    place      Kothrud, Pune, Maharashtra (from the PIN directory), Palm Grove Layout
               (a name with a locality ending), or an NER location

No single piece is an address. address.py groups pieces that sit close together and
decides which groups are. A "bare" piece (a unit like "12/3", a PIN without "pincode"
before it) is too common in other text to count unless a non-bare piece is next to it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from leakguard.detect.gazetteer import Gazetteer
from leakguard.detect.spans import Intervals

SignalKind = Literal["pin", "unit", "building", "street", "landmark", "place"]
DETAIL_KINDS: frozenset[str] = frozenset({"pin", "unit", "building", "street", "landmark"})
PLACE_KINDS: frozenset[str] = frozenset({"pin", "place"})


@dataclass(frozen=True)
class Signal:
    start: int
    end: int
    kind: SignalKind
    bare: bool = False


# --- vocabularies ----------------------------------------------------------------------

UNIT_WORDS = (r"flat(?:\s*no\.?)?", r"h\.?\s?no\.?", r"house\s*(?:no\.?|number)",
              r"plot(?:\s*no\.?)?", r"door\s*no\.?", r"room(?:\s*no\.?)?", r"shop\s*no\.?",
              r"(?:quarter|qtr\.?)\s*no\.?", r"makaan\s*no\.?", r"makan\s*no\.?", r"unit",
              r"tower", r"wing", r"no\.")
BUILDING_WORDS = ("apartments", "apartment", "apts", "apt", "society", "soc", "residency",
                  "towers", "heights", "enclave", "niwas", "nivas", "sadan", "bhavan", "bhawan",
                  "complex", "chs", "kunj", "villa", "villas", "hostel", "plaza", "mansion",
                  "campus", "residences", "gardens")
STREET_WORDS = ("road", "rd", "marg", "street", "st", "lane", "path", "salai", "chowk", "bazaar",
                "galli", "veedhi")
# Endings of Indian locality names. Many new layouts and colonies are not post offices, so
# the directory doesn't know them, but "<Name> Layout" or "<Name> Colony" is a place.
LOCALITY_ENDINGS = ("nagar", "colony", "layout", "vihar", "puram", "bagh", "ganj", "pura",
                    "wadi", "peth", "palya", "halli", "extension", "extn", "township", "estate")
NUMBERED_STREETS = ("gali", "galli", "lane", "sector", "phase", "block", "ward")
LANDMARK_CUES = (r"near", r"nr\.?", r"opp\.?", r"opposite", r"behind", r"beside", r"besides",
                 r"next\s+to", r"in\s+front\s+of", r"close\s+to")
HINGLISH_POSTPOSITIONS = (r"ke\s+(?:paas|pass|saamne|samne|peeche|piche|peechhe)",)
LANDMARK_NOUNS = frozenset({
    "mandir", "masjid", "church", "gurudwara", "temple", "school", "college", "hospital",
    "mall", "atm", "bank", "pump", "station", "stand", "chowk", "park", "market", "talkies",
    "cinema", "theatre", "theater", "office", "tank", "ground", "circle", "garden", "hotel",
    "dharamshala", "dairy", "bridge", "flyover", "showroom", "petrol", "depot", "tower",
})
PIN_CUES = re.compile(r"(?:pin\s*code|pincode|pin|postal\s*code|zip(?:\s*code)?)"
                      r"(?:\s+(?:is|hai|no\.?|number))?\W{0,3}$", re.IGNORECASE)
PIN_BLOCKERS = frozenset({"otp", "code", "order", "ticket", "invoice", "ref", "txn", "id",
                          "rs", "₹", "inr", "amount", "pnr", "awb", "tracking", "booking"})

# Words that never belong to the name of a building, street or landmark.
_NOT_A_NAME = frozenset({"the", "a", "an", "my", "our", "your", "his", "her", "their", "this",
                         "that", "in", "at", "on", "of", "to", "from", "and", "or", "is", "are",
                         "was", "near", "opp", "behind", "beside", "via", "for", "with", "mera",
                         "meri", "hamara", "ghar", "address", "deliver", "ship", "send", "pe",
                         "par", "mein", "hai", "se", "ko", "parcel", "i", "we", "you", "it"})

# --- patterns --------------------------------------------------------------------------

_NAME_WORD = r"[A-Za-z][A-Za-z.'&]*"
_WORDS = re.compile(rf"{_NAME_WORD}")

_UNIT = re.compile(
    rf"(?<![\w.])(?:{'|'.join(UNIT_WORDS)})\s*[:#-]?\s*[A-Za-z]?\d+[A-Za-z]?"
    r"(?:\s?[/-]\s?[A-Za-z0-9]+)*(?![\w/])", re.IGNORECASE)
_FLOOR = re.compile(r"\b(?:\d+(?:st|nd|rd|th)|ground|first|second|third|top)\s+floor\b",
                    re.IGNORECASE)
# Bare units: "12/3", "3-4-56" (but not dates: no third part with 4 digits), "B-204", "A/7",
# "#45".
_BARE_UNIT = re.compile(
    r"(?<![\w/#.-])(?:\d{1,4}\s?/\s?\d{1,4}[A-Za-z]?|[A-Za-z]{1,2}\s?[-/]\s?\d{1,4}[A-Za-z]?"
    r"|#\s?\d{1,5}[A-Za-z]?)(?![\w/.-]|-\w)")
# Both start with a "not inside a word" lookbehind: otherwise a failed match is retried
# from every letter of a long word, which is quadratic (100,000 letters took a minute).
_BUILDING = re.compile(
    rf"(?<![\w.'&])(?:{_NAME_WORD} ){{0,3}}(?:{'|'.join(BUILDING_WORDS)})\b(?: \d+)?",
    re.IGNORECASE)
_NAMED_STREET = re.compile(
    rf"(?<![\w.'&])(?:\d{{1,4}} )?(?:{_NAME_WORD} ){{0,3}}(?:{'|'.join(STREET_WORDS)})\b\.?",
    re.IGNORECASE)
_LOCALITY = re.compile(
    rf"(?<![\w.'&])(?:{_NAME_WORD} ){{1,3}}(?:{'|'.join(LOCALITY_ENDINGS)})\b\.?",
    re.IGNORECASE)
_ORDINAL_STREET = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:cross|main)(?:\s+(?:road|rd))?\b",
                             re.IGNORECASE)
_NUMBERED_STREET = re.compile(
    rf"\b(?:{'|'.join(NUMBERED_STREETS)})\s*(?:no\.?\s*)?[:#-]?\s*(?:\d{{1,4}}[A-Za-z]?|[A-Z])\b",
    re.IGNORECASE)
_LANDMARK = re.compile(
    rf"(?<![\w.])(?:{'|'.join(LANDMARK_CUES)})(?![\w])\s*((?:{_NAME_WORD}\s?){{1,4}})",
    re.IGNORECASE)
_POSTPOSITION = re.compile(rf"(?:{'|'.join(HINGLISH_POSTPOSITIONS)})\b", re.IGNORECASE)
_SIX_DIGITS = re.compile(r"(?<![\d])[1-9]\d{5}(?![\d])")


# --- finders ---------------------------------------------------------------------------

def _trim_name(text: str, start: int, end: int, keep_last: bool) -> int:
    """Start of the name in text[start:end] after dropping leading words that can't be
    part of a name ("our society" -> "society"; "the" in "the metro station" is kept by
    the landmark finder instead). The last word (the suffix) is always kept."""
    words = list(_WORDS.finditer(text, start, end))
    first = 0
    for i, w in enumerate(words[:-1] if keep_last else words):
        if w.group().lower().rstrip(".") in _NOT_A_NAME:
            first = i + 1
    if first == 0:
        return start  # nothing dropped; keeps a leading house number ("45 Anna Salai")
    return words[first].start() if first < len(words) else end


def _named(pattern: re.Pattern, text: str, kind: SignalKind, need_name: bool) -> list[Signal]:
    found = []
    for m in pattern.finditer(text):
        start = _trim_name(text, m.start(), m.end(), keep_last=True)
        name_words = _WORDS.findall(text, start, m.end())
        if need_name and len(name_words) < 2 and not re.search(r"\d", text[start:m.end()]):
            continue  # "society" alone is not a building, "Tower 4" or "Rose Apts" is
        found.append(Signal(start, m.end(), kind))
    return found


def _units(text: str) -> list[Signal]:
    found = [Signal(m.start(), m.end(), "unit") for m in _UNIT.finditer(text)]
    found += [Signal(m.start(), m.end(), "unit") for m in _FLOOR.finditer(text)]
    taken = Intervals((s.start, s.end) for s in found)
    for m in _BARE_UNIT.finditer(text):
        if not taken.overlaps(m.start(), m.end()):
            found.append(Signal(m.start(), m.end(), "unit", bare=True))
    return found


def _streets(text: str) -> list[Signal]:
    found = _named(_NAMED_STREET, text, "street", need_name=True)
    found += [Signal(m.start(), m.end(), "street") for m in _ORDINAL_STREET.finditer(text)]
    found += [Signal(m.start(), m.end(), "street") for m in _NUMBERED_STREET.finditer(text)]
    return found


def _landmark_end(text: str, words: list[re.Match]) -> int | None:
    """End of the landmark's name among the words after a cue: up to the last landmark
    noun ("hanuman mandir", "the bus stand"), else the run of Capitalised words."""
    nouns = [w for w in words if w.group().lower().rstrip(".") in LANDMARK_NOUNS]
    if nouns:
        return nouns[-1].end()
    end = None
    for w in words:
        if not w.group()[0].isupper() or w.group().lower() in _NOT_A_NAME:
            break
        end = w.end()
    return end


def _landmarks(text: str, gazetteer: Gazetteer) -> list[Signal]:
    found = []
    for m in _LANDMARK.finditer(text):
        words = [w for w in _WORDS.finditer(text, m.start(1), m.end(1))]
        if words and words[0].group().lower() == "the":
            rest = words[1:]
        else:
            rest = words
        end = _landmark_end(text, rest)
        if end is None:
            continue
        name_start = rest[0].start()
        has_noun = any(w.group().lower().rstrip(".") in LANDMARK_NOUNS for w in rest)
        places = gazetteer.find_places(text[name_start:end])
        if not has_noun and places and places[0][:2] == (0, end - name_start):
            continue  # "near Kothrud" is a place, not a landmark
        found.append(Signal(m.start(), end, "landmark"))
    for m in _POSTPOSITION.finditer(text):
        before = list(_WORDS.finditer(text, max(0, m.start() - 60), m.start()))[-4:]
        start = None
        for w in reversed(before):
            if w.group().lower() in _NOT_A_NAME or text[w.end():m.start()].count("\n"):
                break
            start = w.start()
        if start is not None:
            found.append(Signal(start, m.end(), "landmark"))
    return found


def _pins(text: str, gazetteer: Gazetteer) -> list[Signal]:
    found = []
    for m in _SIX_DIGITS.finditer(text):
        if m.group() not in gazetteer.pins:
            continue
        before = text[max(0, m.start() - 30):m.start()]
        cued = PIN_CUES.search(before) is not None
        recent = re.findall(r"[a-z₹]+", before.lower())[-3:]
        if not cued and any(word in PIN_BLOCKERS for word in recent):
            continue  # "OTP 411038", "order no 560034"
        found.append(Signal(m.start(), m.end(), "pin", bare=not cued))
    return found


def find_signals(text: str, gazetteer: Gazetteer,
                 locations: Sequence[tuple[int, int]] = ()) -> list[Signal]:
    """Every address piece in `text`, sorted by position. Pieces may overlap, except
    that a place inside a landmark is dropped: some post offices are named "Bus Stand"
    or "Hanuman Mandir", and "I'm near the bus stand" must not have a place of its own."""
    landmarks = _landmarks(text, gazetteer)
    places = [Signal(s, e, "place") for s, e, _ in gazetteer.find_places(text)]
    places += _named(_LOCALITY, text, "place", need_name=True)
    places += [Signal(s, e, "place") for s, e in locations]
    in_landmark = Intervals((lm.start, lm.end) for lm in landmarks)
    places = [p for p in places if not in_landmark.contains(p.start, p.end)]
    signals = [
        *_pins(text, gazetteer),
        *_units(text),
        *_named(_BUILDING, text, "building", need_name=True),
        *_streets(text),
        *landmarks,
        *places,
    ]
    return sorted(signals, key=lambda s: (s.start, -s.end))
