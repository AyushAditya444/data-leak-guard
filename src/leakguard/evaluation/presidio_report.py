"""The Presidio pages: dev tuning, and the side-by-side comparison on test.

Both render from the committed JSON written by presidio_run.py, so regenerating them needs
neither Presidio nor the name model.
"""

from collections.abc import Sequence

from leakguard.entities import EntityType
from leakguard.evaluation.metrics import TypeScores, evaluation_from_json
from leakguard.evaluation.presidio_baseline import BOTH_SUPPORT, MAPPING, UNSUPPORTED

ENGINES = (("ours", "Ours"),
           ("presidio_out_of_box", "Presidio, out of the box"),
           ("presidio_best_shot", "Presidio, best shot"))
CAVEAT = "†"
NOT_SUPPORTED = "not supported"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _sum(scores: Sequence[TypeScores]) -> TypeScores:
    total = TypeScores()
    for st in scores:
        for name in ("tp", "fp", "fn", "exact", "gold_chars", "covered_chars"):
            setattr(total, name, getattr(total, name) + getattr(st, name))
    return total


def _supported(engine: str, entity: EntityType) -> bool:
    return engine == "ours" or entity not in UNSUPPORTED


def _label(entity: EntityType) -> str:
    return f"{entity} {CAVEAT}" if entity is EntityType.ADDRESS else str(entity)


def _scores(results: dict, split: str) -> dict[str, dict[EntityType, TypeScores]]:
    return {engine: evaluation_from_json(results["engines"][engine][split]).by_type
            for engine, _ in ENGINES}


def _headline(results: dict) -> list[str]:
    scores = _scores(results, "test")
    names = [label for _, label in ENGINES]
    lines = ["| Type | " + " | ".join(f"{n}: F1" for n in names) + " | "
             + " | ".join(f"{n}: coverage" for n in names) + " |",
             "|---|" + "---:|" * (2 * len(ENGINES))]
    for entity in EntityType:
        rows = {engine: scores[engine].get(entity) for engine, _ in ENGINES}
        if not rows["ours"] or rows["ours"].support == 0:
            continue
        f1 = [_pct(st.f1) if _supported(engine, entity) else NOT_SUPPORTED
              for engine, st in rows.items()]
        cov = [_pct(st.coverage) if st else "–" for st in rows.values()]
        lines.append(f"| {_label(entity)} | " + " | ".join(f1 + cov) + " |")
    both = {engine: _sum([s[e] for e in BOTH_SUPPORT if e in s]) for engine, s in scores.items()}
    lines.append("| **Types both support** | "
                 + " | ".join(f"**{_pct(st.f1)}**" for st in both.values()) + " | "
                 + " | ".join(f"**{_pct(st.coverage)}**" for st in both.values()) + " |")
    return lines


def _detail(results: dict, engine: str, split: str) -> list[str]:
    ev = evaluation_from_json(results["engines"][engine][split])
    lines = ["| Type | Support | Precision | Recall | F1 | Coverage |",
             "|---|---:|---:|---:|---:|---:|"]
    for entity in EntityType:
        st = ev.by_type.get(entity)
        if st is None or (st.support == 0 and st.fp == 0):
            continue
        if _supported(engine, entity):
            cells = [_pct(st.precision), _pct(st.recall), _pct(st.f1)]
        else:
            cells = [NOT_SUPPORTED] * 3
        lines.append(f"| {_label(entity)} | {st.support} | " + " | ".join(cells)
                     + f" | {_pct(st.coverage)} |")
    lines += ["", (f"Negative messages with at least one false alarm: {ev.negatives_with_fp} "
              f"of {ev.negatives_total}.")]
    return lines


def _indian_sentence(meta: dict) -> str:
    defaults = meta["indian_defaults"]
    version = meta["versions"].get("presidio-analyzer", "?")
    on = sorted(name for name, enabled in defaults.items() if enabled)
    if not on:
        return (f"In presidio-analyzer {version}, all of Presidio's Indian recognizers "
                f"({', '.join(sorted(defaults))}) are **off by default**, so the out-of-the-box "
                "setup cannot find Aadhaar or PAN numbers at all. The best-shot setup turns "
                "them on.")
    return (f"In presidio-analyzer {version}, these Indian recognizers are on by default: "
            f"{', '.join(on)}; the rest are off and are turned on only in the best-shot setup.")


def render_presidio_baseline(results: dict, notes: Sequence[str]) -> str:
    meta = results["meta"]
    versions = ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
    lines = [
        "# Presidio baseline",
        "",
        (f"Generated {meta['date']} at commit `{meta['guard_commit']}` with "
        f"`python -m leakguard.evaluation presidio-report`, from "
        "`data/eval/presidio-test.json`."),
        "",
        ("Microsoft Presidio is the best-known open-source PII detector. This page runs it on "
        f"the same **test** split ({meta['sizes']['test']} messages) and **hand-written** set "
        f"({meta['sizes']['handwritten']} messages) as [our report](report.md), scored the "
        "same way (same type and overlapping; coverage = share of sensitive characters hidden)."),
        "",
        "- **Out of the box:** `AnalyzerEngine()` exactly as installed, `language=\"en\"`.",
        ("- **Best shot:** the same, plus all of Presidio's Indian recognizers turned on and a "
        "minimum score per type tuned on dev ([presidio-tuning.md](presidio-tuning.md)). Only "
        "Presidio's own settings: no custom recognizers, patterns or models."),
        "",
        _indian_sentence(meta),
        "",
        f"Versions: {versions}.",
        "",
        "## Test split: F1 and coverage",
        "",
        *_headline(results),
        "",
        (f"*not supported*: Presidio has no detector for this type; its coverage column still "
        "shows how much of these values Presidio hid anyway through other detectors. "
        f"{CAVEAT} Presidio's `LOCATION` finds place names, not full postal addresses, so its "
        "ADDRESS precision and recall are not like for like; coverage is the fairer measure. "
        "*Types both support* adds up AADHAAR, PAN, PHONE, EMAIL, CARD and NAME the same way "
        "for every engine."),
        "",
    ]
    for split, title in (("test", "Test split"), ("handwritten", "Hand-written set")):
        for engine, label in ENGINES:
            lines += [f"## {title}: {label}", "", *_detail(results, engine, split), ""]
    lines += ["## Latency (detection only, test split)", "",
              "Best of 3 runs per message, one CPU thread.", "",
              "| Engine | p50 | p95 |", "|---|---:|---:|"]
    for engine, label in ENGINES:
        lat = results["engines"][engine]["latency"]
        lines.append(f"| {label} | {lat['p50_ms']:.3f} ms | {lat['p95_ms']:.3f} ms |")
    lines += ["", "## Presidio results we don't score (test split)", "",
              ("Presidio types outside the mapping are not counted as false alarms, but they do "
              "count for coverage. How often each fired, after conflicts were resolved:"), "",
              "| Presidio type | " + " | ".join(label for _, label in ENGINES[1:]) + " |",
              "|---|" + "---:|" * (len(ENGINES) - 1)]
    unmapped = [results["engines"][engine]["unmapped"] for engine, _ in ENGINES[1:]]
    for name in sorted(set().union(*unmapped)):
        lines.append(f"| `{name}` | " + " | ".join(str(u.get(name, 0)) for u in unmapped) + " |")
    conflicts = [str(results["engines"][engine]["conflicts"]) for engine, _ in ENGINES[1:]]
    lines += ["", ("Overlapping results resolved the way Presidio's anonymiser does by default "
              f"(removed or merged): {' and '.join(conflicts)} (out of the box, best shot)."),
              "", "Mapping: " + ", ".join(f"`{p}`→{e}" for p, e in MAPPING.items()) + ".",
              "", "Best-shot thresholds: " + ", ".join(
                  f"`{t}` {v:.2f}" for t, v in sorted(results["thresholds"].items())) + ".",
              "", "## Notes", ""]
    lines += [f"- {note}" for note in notes] or ["- none"]
    return "\n".join(lines)


def render_presidio_tuning(dev: dict) -> str:
    meta = dev["meta"]
    rows = dev["tuning"]
    types = list(dict.fromkeys(r["presidio_type"] for r in rows))
    f1 = {(r["presidio_type"], r["threshold"]): r["f1"] for r in rows}
    thresholds = sorted({r["threshold"] for r in rows})
    lines = [
        "# Presidio threshold tuning (dev)",
        "",
        (f"Generated {meta['date']} at commit `{meta['guard_commit']}` from "
        f"`data/eval/presidio-dev.json` ({meta['sizes']['dev']} dev messages)."),
        "",
        ("Presidio's best-shot setup gets a minimum score per type, chosen here on **dev** only. "
        "Rule, fixed before the run: for each type, the threshold from 0.00 to 0.95 (steps of "
        "0.05) with the best F1, every other type left at 0.00; a tie goes to the higher "
        "threshold. Types whose scores don't vary show a flat line."),
        "",
        "| Presidio type | Chosen | F1 at 0.00 | F1 at chosen |",
        "|---|---:|---:|---:|",
    ]
    for t in types:
        chosen = dev["thresholds"][t]
        lines.append(f"| {t} | {chosen:.2f} | {_pct(f1[(t, 0.0)])} | {_pct(f1[(t, chosen)])} |")
    lines += ["", "## F1 by threshold", "",
              "| Threshold | " + " | ".join(f"`{t}`" for t in types) + " |",
              "|---:|" + "---:|" * len(types)]
    for threshold in thresholds:
        lines.append(f"| {threshold:.2f} | "
                     + " | ".join(_pct(f1[(t, threshold)]) for t in types) + " |")
    return "\n".join(lines)
