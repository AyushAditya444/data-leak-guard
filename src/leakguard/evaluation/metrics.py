"""Scoring a detector against gold labels.

Matching rule: a prediction is a true positive when it has the SAME TYPE as a gold span
and OVERLAPS it. Each gold span and each prediction can be matched at most once (largest
overlaps are paired first). Unmatched predictions are false positives, unmatched gold
spans are false negatives.

Coverage answers the question that matters for leaks — "was the value hidden?" — as the
share of gold characters covered by ANY prediction, whatever its type or boundaries. An
optional `cover` adds ranges that count for coverage only: e.g. a baseline's results of a
type we don't score, which its anonymiser would still hide.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

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
    # (category, type) -> scores, so a report can leave out types with no detector yet
    by_category_type: dict[tuple[str, EntityType], TypeScores] = field(default_factory=dict)
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


def _covered(gold: Span, ranges: list[tuple[int, int]]) -> int:
    """Gold characters inside at least one range; each character counted once."""
    return sum(1 for i in range(gold.start, gold.end)
               if any(start <= i < end for start, end in ranges))


def score(samples: Iterable[Sample], detect: Callable[[str], list[Span]],
          cover: Callable[[str], Iterable[tuple[int, int]]] | None = None) -> Evaluation:
    by_type: dict[EntityType, TypeScores] = defaultdict(TypeScores)
    by_category: dict[str, TypeScores] = defaultdict(TypeScores)
    by_category_type: dict[tuple[str, EntityType], TypeScores] = defaultdict(TypeScores)
    ev = Evaluation()

    for sample in samples:
        preds = list(detect(sample.text))  # detector errors propagate on purpose
        gold = list(sample.spans)
        matched = _match(gold, preds)
        hidden = [(p.start, p.end) for p in preds] + list(cover(sample.text) if cover else [])

        for gi, g in enumerate(gold):
            # the same counts go into three views: per type, per category, per both
            views = (by_type[g.type], by_category[sample.category],
                     by_category_type[(sample.category, g.type)])
            covered = _covered(g, hidden)
            hit = gi in matched
            for scores in views:
                scores.gold_chars += g.end - g.start
                scores.covered_chars += covered
                scores.tp += hit
                scores.fn += not hit
            if hit:
                p = preds[matched[gi]]
                by_type[g.type].exact += (p.start, p.end) == (g.start, g.end)
            else:
                ev.errors.append(ErrorCase(sample.id, "FN", g.type, g.text(sample.text),
                                           sample.text))

        used = set(matched.values())
        unmatched = [p for pi, p in enumerate(preds) if pi not in used]
        for p in unmatched:
            for scores in (by_type[p.type], by_category[sample.category],
                           by_category_type[(sample.category, p.type)]):
                scores.fp += 1
            ev.errors.append(ErrorCase(sample.id, "FP", p.type, p.text(sample.text), sample.text))

        if sample.category == NEGATIVE:
            ev.negatives_total += 1
            ev.negatives_with_fp += bool(unmatched)

    ev.by_type = dict(by_type)
    ev.by_category = dict(by_category)
    ev.by_category_type = dict(by_category_type)
    return ev


def evaluation_to_json(ev: Evaluation) -> dict:
    """Per-type counts only (no error examples, so no message text): safe to commit."""
    return {"by_type": {str(entity): asdict(st) for entity, st in ev.by_type.items()},
            "negatives_total": ev.negatives_total,
            "negatives_with_fp": ev.negatives_with_fp}


def evaluation_from_json(data: dict) -> Evaluation:
    return Evaluation(
        by_type={EntityType(entity): TypeScores(**st) for entity, st in data["by_type"].items()},
        negatives_total=data["negatives_total"], negatives_with_fp=data["negatives_with_fp"])
