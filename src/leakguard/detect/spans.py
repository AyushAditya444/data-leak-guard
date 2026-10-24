"""A Span marks one piece of sensitive data inside a text: where it is and what it is."""

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
    then the earliest. Returns non-overlapping spans sorted by position."""
    ranked = sorted(spans, key=lambda s: (-PRIORITY[s.type], -(s.end - s.start), s.start))
    kept: list[Span] = []
    for span in ranked:
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)
