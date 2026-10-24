"""A Span marks one piece of sensitive data inside a text: where it is and what it is."""

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from leakguard.entities import PRIORITY, EntityType


@dataclass(frozen=True)
class Span:
    start: int  # index of the first character
    end: int  # index just after the last character (like Python slicing)
    type: EntityType
    score: float = 1.0
    detector: str = ""

    def text(self, source: str) -> str:
        return source[self.start:self.end]

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Resolve overlapping detections: keep the highest-priority span, then the longest,
    then the earliest. Returns non-overlapping spans sorted by position.

    Kept spans never overlap, so they stay sorted by start and a new candidate only needs
    checking against its two neighbours, found by binary search: O(n log n) overall.
    (Comparing against every kept span was quadratic: 20,000 phone numbers took 16 s.)
    """
    ranked = sorted(spans, key=lambda s: (-PRIORITY[s.type], -(s.end - s.start), s.start))
    starts: list[int] = []
    kept: list[Span] = []
    for span in ranked:
        i = bisect_right(starts, span.start)
        overlaps_left = i > 0 and kept[i - 1].end > span.start
        overlaps_right = i < len(kept) and kept[i].start < span.end
        if not (overlaps_left or overlaps_right):
            starts.insert(i, span.start)
            kept.insert(i, span)
    return kept
