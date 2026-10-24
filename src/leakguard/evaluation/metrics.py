"""Scoring a detector against gold labels.

Matching rule: a prediction is a true positive when it has the SAME TYPE as a gold span
and OVERLAPS it. Each gold span and each prediction can be matched at most once (largest
overlaps are paired first). Unmatched predictions are false positives, unmatched gold
spans are false negatives.

Coverage answers the question that matters for leaks — "was the value hidden?" — as the
share of gold characters covered by ANY prediction, whatever its type or boundaries.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from leakguard.detect.spans import Span
from leakguard.entities import EntityType
from leakguard.evaluation.samples import Sample

NEGATIVE = "negative"


def _ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


@dataclass
class TypeScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact: int = 0  # true positives whose boundaries match the gold span exactly
    gold_chars: int = 0
    covered_chars: int = 0

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def coverage(self) -> float:
        return _ratio(self.covered_chars, self.gold_chars)

    @property
    def support(self) -> int:
        return self.tp + self.fn


@dataclass(frozen=True)
class ErrorCase:
    sample_id: str
    kind: str  # "FP" (flagged but not sensitive) or "FN" (sensitive but missed)
    type: EntityType
    value: str
    text: str


@dataclass
class Evaluation:
    by_type: dict[EntityType, TypeScores] = field(default_factory=dict)
    by_category: dict[str, TypeScores] = field(default_factory=dict)
    negatives_total: int = 0
    negatives_with_fp: int = 0
    errors: list[ErrorCase] = field(default_factory=list)


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _match(gold: list[Span], preds: list[Span]) -> dict[int, int]:
    """gold index -> prediction index, pairing the largest same-type overlaps first."""
    pairs = sorted(
        ((_overlap(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(preds)
         if g.type is p.type and _overlap(g, p) > 0),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used_preds: set[int] = set()
    for _, gi, pi in pairs:
        if gi not in matched and pi not in used_preds:
            matched[gi] = pi
            used_preds.add(pi)
    return matched


def _covered(gold: Span, preds: list[Span]) -> int:
    return sum(1 for i in range(gold.start, gold.end)
               if any(p.start <= i < p.end for p in preds))


def score(samples: Iterable[Sample], detect: Callable[[str], list[Span]]) -> Evaluation:
    by_type: dict[EntityType, TypeScores] = defaultdict(TypeScores)
    by_category: dict[str, TypeScores] = defaultdict(TypeScores)
    ev = Evaluation()

    for sample in samples:
        preds = list(detect(sample.text))  # detector errors propagate on purpose
        gold = list(sample.spans)
        matched = _match(gold, preds)
        category = by_category[sample.category]

        for gi, g in enumerate(gold):
            scores = by_type[g.type]
            covered = _covered(g, preds)
            scores.gold_chars += g.end - g.start
            scores.covered_chars += covered
            category.gold_chars += g.end - g.start
            category.covered_chars += covered
            if gi in matched:
                p = preds[matched[gi]]
                scores.tp += 1
                category.tp += 1
                scores.exact += (p.start, p.end) == (g.start, g.end)
            else:
                scores.fn += 1
                category.fn += 1
                ev.errors.append(ErrorCase(sample.id, "FN", g.type, g.text(sample.text),
                                           sample.text))

        used = set(matched.values())
        unmatched = [p for pi, p in enumerate(preds) if pi not in used]
        for p in unmatched:
            by_type[p.type].fp += 1
            category.fp += 1
            ev.errors.append(ErrorCase(sample.id, "FP", p.type, p.text(sample.text), sample.text))

        if sample.category == NEGATIVE:
            ev.negatives_total += 1
            ev.negatives_with_fp += bool(unmatched)

    ev.by_type = dict(by_type)
    ev.by_category = dict(by_category)
    return ev
