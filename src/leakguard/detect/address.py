"""Indian postal addresses: group nearby address pieces into one ADDRESS span.

An address is masked when it has at least one specific DETAIL (flat or house number,
building, street, landmark, PIN) and at least one PLACE (locality, city, district, state,
PIN), close together. So:

    "Flat 302, Sai Krupa Apts, Kothrud, Pune 411038"   masked, as ONE span
    "near Hanuman Mandir, Kothrud"                      masked (landmark + locality)
    "I live in Kothrud, Pune"                           not masked (places only)
    "stuck in traffic near Kothrud"                     not masked (no landmark name)
    "near the metro station"                            not masked (no place)
    "OTP 411038"                                        not masked (not used as a PIN)

One span from the first piece to the last, so masking "Kothrud, Pune 411038" while
leaving "Flat 302, Sai Krupa Apts" visible can't happen.
"""

import re
from bisect import bisect_left
from collections.abc import Sequence

from leakguard.detect.address_signals import DETAIL_KINDS, PLACE_KINDS, Signal, find_signals
from leakguard.detect.gazetteer import load_gazetteer
from leakguard.detect.spans import Span
from leakguard.entities import EntityType

# How many characters of "other text" may sit between two pieces of one address, after
# separators are ignored. Tuned on the dev split (docs/evaluation/address-tuning.md).
DEFAULT_GAP = 24

# Separators between the parts of an address; they don't count towards the gap.
_SEPARATORS = re.compile(r"[\s,;:/.\-&()]+|\band\b", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\n[ \t\r]*\n")
_EDGE = " \t\r\n,;:.-/&("


def _chains(text: str, group: list[Signal], end: int, signal: Signal, gap: int) -> bool:
    """May `signal` join `group`, whose pieces end at `end`?"""
    if signal.start <= end:
        return True  # overlapping pieces
    between = text[end:signal.start]
    if _BLANK_LINE.search(between):
        return False  # a blank line ends a block of text
    other_text = _SEPARATORS.sub("", between)
    if (other_text and signal.kind == "unit" and not signal.bare
            and any(s.kind in PLACE_KINDS for s in group)):
        return False  # "Flat 2, Baner, Pune to H.No. 45, ...": a second address starts
    return len(other_text) <= gap


def _drop_leading_places(text: str, group: list[Signal]) -> list[Signal]:
    """An address starts with a detail or with places listed one after another ("Kothrud,
    Pune 411038"). A place joined to the rest by ordinary words is not part of it: in
    "Reddy here, my new address is behind Axis Bank, Kothrud" the surname "Reddy" is also a
    village name."""
    while (len(group) > 1 and group[0].kind == "place" and group[0].end < group[1].start
           and _SEPARATORS.sub("", text[group[0].end:group[1].start])):
        group = group[1:]
    return group


def _is_address(group: list[Signal]) -> bool:
    kinds = {s.kind for s in group}
    return (bool(kinds & DETAIL_KINDS) and bool(kinds & PLACE_KINDS)
            and any(not s.bare for s in group))


class _Stops:
    """Spans of other detectors, sorted, with O(log n) lookups (a pasted log can hold
    thousands of phone numbers)."""

    def __init__(self, stops: Sequence[Span]):
        self._spans = sorted(stops, key=lambda s: s.start)
        self._starts = [s.start for s in self._spans]

    def _first_from(self, position: int) -> Span | None:
        i = bisect_left(self._starts, position)
        return self._spans[i] if i < len(self._spans) else None

    def overlaps(self, start: int, end: int) -> bool:
        # stops never overlap each other (they come out of merging), so only the stop
        # just before `start` and the first one from `start` on can touch [start, end)
        i = bisect_left(self._starts, start)
        before = self._spans[i - 1] if i > 0 else None
        after = self._first_from(start)
        return bool((before and before.end > start) or (after and after.start < end))

    def between(self, end: int, start: int) -> bool:
        stop = self._first_from(end)
        return stop is not None and stop.end <= start


def _group(signals: list[Signal], text: str, stops: _Stops, gap: int) -> list[list[Signal]]:
    groups: list[list[Signal]] = []
    end = -1
    for signal in signals:
        if (groups and not stops.between(end, signal.start)
                and _chains(text, groups[-1], end, signal, gap)):
            groups[-1].append(signal)
            end = max(end, signal.end)
        else:
            groups.append([signal])
            end = signal.end
    return groups


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _EDGE:
        start += 1
    while end > start and text[end - 1] in _EDGE:
        end -= 1
    # never cut a word: masking "Kothru" would leak the "d"
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def detect_address(text: str, stops: Sequence[Span] = (),
                   locations: Sequence[tuple[int, int]] = (),
                   gap: int = DEFAULT_GAP) -> list[Span]:
    """ADDRESS spans in `text`.

    `stops` are spans other detectors already found (phones, emails...): an address never
    overlaps or reaches across one, so it can't swallow a phone number — and merging
    can't then drop the whole address in favour of the higher-priority phone.
    `locations` are extra place ranges, e.g. location tags from the NER model.
    """
    blocked = _Stops(stops)
    signals = [s for s in find_signals(text, load_gazetteer(), locations)
               if not blocked.overlaps(s.start, s.end)]
    spans = []
    for group in _group(signals, text, blocked, gap):
        group = _drop_leading_places(text, group)
        if _is_address(group):
            start, end = _trim(text, group[0].start, max(s.end for s in group))
            if start < end:
                spans.append(Span(start, end, EntityType.ADDRESS, 1.0, "address"))
    return spans
