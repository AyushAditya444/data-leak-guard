"""Microsoft Presidio as a baseline, scored exactly like our engine.

This module is the part that needs no Presidio: the type mapping, score thresholds,
Presidio's default way of resolving overlapping results, an adapter that turns Presidio
results into our Spans, and the dev-set threshold tuning. presidio_run.py plugs the real
Presidio in. Keeping the two apart lets CI test every decision here with a fake analyser.

Fairness rules (Phase 7 design §4):
- Presidio types we don't track (DATE_TIME, URL, ...) are not counted as false alarms, but
  they still count for coverage: an anonymiser built on Presidio would hide them too.
- Overlaps are resolved the way Presidio's own anonymiser does by default, so we score what
  a developer using Presidio would actually get masked.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from leakguard.detect.spans import Span
from leakguard.entities import EntityType
from leakguard.evaluation.metrics import score
from leakguard.evaluation.samples import Sample

MAPPING: dict[str, EntityType] = {
    "IN_AADHAAR": EntityType.AADHAAR,
    "IN_PAN": EntityType.PAN,
    "PHONE_NUMBER": EntityType.PHONE,
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "CREDIT_CARD": EntityType.CARD,
    "PERSON": EntityType.NAME,
    "LOCATION": EntityType.ADDRESS,  # place names, not full addresses: reported with a caveat
}
UNSUPPORTED = (EntityType.UPI, EntityType.IFSC, EntityType.BANK_ACCOUNT, EntityType.SECRET)
# The like-for-like average: types both engines detect (ADDRESS is shown on its own).
BOTH_SUPPORT = (EntityType.AADHAAR, EntityType.PAN, EntityType.PHONE, EntityType.EMAIL,
                EntityType.CARD, EntityType.NAME)
GRID = tuple(i / 20 for i in range(20))  # 0.0, 0.05, ..., 0.95


@dataclass(frozen=True)
class Found:
    """One Presidio result, without depending on Presidio's own class."""

    entity: str
    start: int
    end: int
    score: float


Analyze = Callable[[str], list[Found]]


def apply_thresholds(found: Iterable[Found], thresholds: Mapping[str, float]) -> list[Found]:
    """Keep results scoring at least their type's threshold; untuned types are all kept."""
    return [f for f in found if f.score >= thresholds.get(f.entity, 0.0)]


def _merge(a: Found, b: Found) -> Found:
    return Found(a.entity, min(a.start, b.start), max(a.end, b.end), max(a.score, b.score))


def _merge_same_type_overlaps(found: list[Found]) -> list[Found]:
    merged: list[Found] = []
    for entity in dict.fromkeys(f.entity for f in found):
        group = sorted((f for f in found if f.entity == entity), key=lambda f: (f.start, f.end))
        current = group[0]
        for f in group[1:]:
            if f.start < current.end:
                current = _merge(current, f)
            else:
                merged.append(current)
                current = f
        merged.append(current)
    return sorted(merged, key=lambda f: (f.start, f.end))


def _remove_contained(found: list[Found]) -> list[Found]:
    """Drop results inside another one (any type). Identical spans keep the higher score;
    a tie keeps the later one. Quadratic, but Presidio returns a handful per message."""
    def beaten(i: int, r: Found) -> bool:
        for j, o in enumerate(found):
            if j == i or not (o.start <= r.start and r.end <= o.end):
                continue
            if (o.start, o.end) != (r.start, r.end) or o.score > r.score:
                return True
            if o.score == r.score and j > i:
                return True
        return False
    return [r for i, r in enumerate(found) if not beaten(i, r)]


def _merge_space_separated(found: list[Found], text: str) -> list[Found]:
    merged: list[Found] = []
    for f in sorted(found, key=lambda f: (f.start, f.end)):
        prev = merged[-1] if merged else None
        gap = text[prev.end:f.start] if prev else ""
        if prev and prev.entity == f.entity and gap and set(gap) == {" "}:
            merged[-1] = _merge(prev, f)
        else:
            merged.append(f)
    return merged


def resolve_conflicts(found: Iterable[Found], text: str) -> tuple[list[Found], int]:
    """Presidio anonymiser's default strategy (MERGE_SIMILAR_OR_CONTAINED). Returns the
    results that would be masked and how many were removed or merged away."""
    found = list(found)
    valid = [f for f in found if 0 <= f.start < f.end <= len(text)]
    kept = _merge_same_type_overlaps(valid) if valid else []
    kept = _remove_contained(kept)
    kept = [f for f in _merge_space_separated(kept, text) if f.end > f.start]
    return kept, len(found) - len(kept)


class PresidioDetector:
    """Presidio as a detector for metrics.score(): `detect` gives mapped Spans (typed
    metrics), `cover` gives every kept result (coverage). Each text is analysed once."""

    def __init__(self, analyze: Analyze, thresholds: Mapping[str, float] | None = None):
        self._analyze = analyze
        self._thresholds = dict(thresholds or {})
        self._kept: dict[str, list[Found]] = {}
        self.unmapped: Counter[str] = Counter()
        self.conflicts = 0

    def _results(self, text: str) -> list[Found]:
        if text not in self._kept:
            kept, removed = resolve_conflicts(
                apply_thresholds(self._analyze(text), self._thresholds), text)
            self.conflicts += removed
            self.unmapped.update(f.entity for f in kept if f.entity not in MAPPING)
            self._kept[text] = kept
        return self._kept[text]

    def detect(self, text: str) -> list[Span]:
        return [Span(f.start, f.end, MAPPING[f.entity], f.score, "presidio")
                for f in self._results(text) if f.entity in MAPPING]

    def cover(self, text: str) -> list[tuple[int, int]]:
        return [(f.start, f.end) for f in self._results(text)]


@dataclass(frozen=True)
class TuningRow:
    presidio_type: str
    threshold: float
    tp: int
    fp: int
    fn: int
    f1: float


def tune_thresholds(dev: Sequence[Sample], analyze: Analyze
                    ) -> tuple[dict[str, float], list[TuningRow]]:
    """For each mapped Presidio type: the threshold in GRID with the best dev F1, every
    other type left at 0.0. Ties go to the higher threshold (fewer false alarms)."""
    cache: dict[str, list[Found]] = {}

    def cached(text: str) -> list[Found]:
        if text not in cache:
            cache[text] = analyze(text)
        return cache[text]

    rows: list[TuningRow] = []
    chosen: dict[str, float] = {}
    for presidio_type, entity in MAPPING.items():
        best: TuningRow | None = None
        for threshold in GRID:
            det = PresidioDetector(cached, {presidio_type: threshold})
            st = score(dev, det.detect).by_type.get(entity)
            tp, fp, fn = (st.tp, st.fp, st.fn) if st else (0, 0, 0)
            row = TuningRow(presidio_type, threshold, tp, fp, fn, st.f1 if st else 0.0)
            rows.append(row)
            if best is None or row.f1 >= best.f1:  # ascending grid: >= keeps the higher one
                best = row
        chosen[presidio_type] = best.threshold
    return chosen, rows
