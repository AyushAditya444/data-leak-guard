"""Latency measurement and the Markdown report."""

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable

from leakguard.detect.spans import Span
from leakguard.entities import EntityType
from leakguard.evaluation.metrics import Evaluation, TypeScores
from leakguard.evaluation.samples import Sample

# Types with labels in the data but no detector yet, mapped to the phase that builds them.
# Every type is built as of Phase 5; the mechanism stays for future types.
UNBUILT: dict[EntityType, str] = {}

_BUCKETS = (("short (<200 chars)", 0, 200), ("medium (200-1000)", 200, 1000),
            ("long (>1000)", 1000, math.inf))


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def measure_latency(
    samples: Iterable[Sample], detect: Callable[[str], list[Span]], repeats: int = 3,
) -> dict[str, dict[str, float]]:
    """Detection time per message: best of `repeats` runs (ignores one-off OS hiccups)."""
    timings: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        best = math.inf
        for _ in range(repeats):
            started = time.perf_counter()
            detect(sample.text)
            best = min(best, (time.perf_counter() - started) * 1000)
        timings["all"].append(best)
        for name, low, high in _BUCKETS:
            if low <= len(sample.text) < high:
                timings[name].append(best)
    return {
        name: {"n": len(values), "p50_ms": _percentile(values, 0.50),
               "p95_ms": _percentile(values, 0.95), "max_ms": max(values)}
        for name, values in timings.items() if values
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _sum(scores: Iterable[TypeScores]) -> TypeScores:
    total = TypeScores()
    for st in scores:
        for attr in ("tp", "fp", "fn", "exact", "gold_chars", "covered_chars"):
            setattr(total, attr, getattr(total, attr) + getattr(st, attr))
    return total


def _overall(ev: Evaluation) -> TypeScores:
    return _sum(st for entity, st in ev.by_type.items() if entity not in UNBUILT)


def _type_table(ev: Evaluation) -> list[str]:
    lines = ["| Type | Support | Precision | Recall | F1 | Coverage | Exact |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for entity in EntityType:
        st = ev.by_type.get(entity)
        if st is None or (st.support == 0 and st.fp == 0):
            continue
        if entity in UNBUILT:
            lines.append(f"| {entity} *(detector not built yet: {UNBUILT[entity]})* "
                         f"| {st.support} | – | – | – | – | – |")
            continue
        exact = _pct(st.exact / st.tp) if st.tp else "–"
        lines.append(f"| {entity} | {st.support} | {_pct(st.precision)} | {_pct(st.recall)} "
                     f"| {_pct(st.f1)} | {_pct(st.coverage)} | {exact} |")
    o = _overall(ev)
    lines.append(f"| **All built types** | **{o.support}** | **{_pct(o.precision)}** "
                 f"| **{_pct(o.recall)}** | **{_pct(o.f1)}** | **{_pct(o.coverage)}** "
                 f"| **{_pct(o.exact / o.tp) if o.tp else '–'}** |")
    return lines


def _category_table(ev: Evaluation) -> list[str]:
    lines = ["| Category | Recall | Coverage | False positives |", "|---|---:|---:|---:|"]
    for category in sorted(ev.by_category):
        # types without a detector yet would drag every category down for no information
        st = _sum(s for (cat, entity), s in ev.by_category_type.items()
                  if cat == category and entity not in UNBUILT)
        recall = _pct(st.recall) if st.support else "–"
        coverage = _pct(st.coverage) if st.gold_chars else "–"
        lines.append(f"| {category} | {recall} | {coverage} | {st.fp} |")
    return lines


def _short(text: str, limit: int = 110) -> str:
    flat = text.replace("\n", " ⏎ ")
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def _examples(ev: Evaluation, per_group: int = 5) -> list[str]:
    groups: dict[tuple[str, EntityType], list] = defaultdict(list)
    for err in ev.errors:
        if err.type not in UNBUILT:
            groups[(err.kind, err.type)].append(err)
    lines: list[str] = []
    for (kind, entity), errs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        label = "missed" if kind == "FN" else "wrongly flagged"
        lines.append(f"- **{kind} {entity}** ({len(errs)} {label})")
        lines += [f"  - `{_short(e.value, 50)}` in “{_short(e.text)}”" for e in errs[:per_group]]
    return lines or ["- none"]


def _latency_table(latency: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| Message length | n | p50 | p95 | max |", "|---|---:|---:|---:|---:|"]
    for name, row in latency.items():
        lines.append(f"| {name} | {int(row['n'])} | {row['p50_ms']:.3f} ms "
                     f"| {row['p95_ms']:.3f} ms | {row['max_ms']:.3f} ms |")
    return lines


# The report pages, in the order the summary lists them, with what each one answers.
SUMMARY_PAGES = (
    ("presidio-baseline.md", ("Microsoft Presidio on the same data, set up two ways, side by "
     "side with this engine.")),
    ("presidio-tuning.md", "how Presidio's best-shot settings were chosen, on dev only."),
    ("performance.md", ("the guard's added latency, throughput, memory, the smallest machine "
     "it fits, and LLM token cost.")),
    ("restore-report.md", ("how reliably real LLM replies get their values put back, streamed "
     "or not.")),
    ("restore-tuning.md", "the dev run that decided which rewritten placeholders to accept."),
    ("ner-comparison.md", "the five name models compared on dev, and why this one was chosen."),
    ("address-tuning.md", "how the address detector's settings were chosen, on dev."),
)


def render_summary() -> str:
    """Where every published number lives: a map of the report pages, no numbers itself."""
    lines = ["## Summary", "",
             ("Every number in the README comes from one of these pages, each generated from "
              "committed results:"), "",
             ("- **This page:** detection quality on the held-out test set and the hand-written "
              "set, per type and per category, with latency and error examples.")]
    lines += [f"- [{page}]({page}): {what}" for page, what in SUMMARY_PAGES]
    return "\n".join(lines)


def render_report(
    results: dict[str, Evaluation],
    latencies: dict[str, dict[str, dict[str, float]]],
    dataset_sizes: dict[str, int],
    hard_cases: dict[str, TypeScores] | None = None,
    notes: list[str] | None = None,
    name_forms: dict[str, TypeScores] | None = None,
    model_info: dict | None = None,
    address_styles: dict[str, TypeScores] | None = None,
) -> str:
    """`latencies` maps a layer label ("rules only", "rules + names") to measure_latency()
    output; `model_info` describes the name model (candidate, repo, revision, size_mb,
    threads)."""
    out = [
        "# Evaluation report",
        "",
        "_Generated by `python -m leakguard.evaluation report`. Do not edit by hand._",
        "",
        render_summary(),
        "",
        "## What was measured",
        "",
        "- Datasets: " + ", ".join(f"**{name}** ({n} messages)" for name, n in
                                   dataset_sizes.items()) + ". All data is synthetic or "
        "hand-written; no real person's data is used.",
        ("- **Matching rule:** a detection is correct when it has the **same type** as a gold "
        "span and **overlaps** it; each gold span and each detection is matched at most once."),
        ("- **Coverage:** share of gold characters hidden by *any* detection, whatever its type. "
        "This is the leak metric: 100% means nothing of that value reached the LLM."),
        "- **Exact:** share of correct detections whose boundaries match the gold span exactly.",
        ("- Thresholds were tuned on the **dev** split only; the numbers below are from data "
        "the tuning never saw."),
    ]
    if model_info:
        out.append(f"- **Name model:** `{model_info['candidate']}` = "
                   f"`{model_info['repo']}@{model_info['revision'][:7]}`, ONNX int8, "
                   f"{model_info['size_mb']:.0f} MB, {model_info['threads']} CPU thread(s). "
                   "Chosen on dev: see [ner-comparison.md](ner-comparison.md).")
    out.append("")
    for name, ev in results.items():
        neg = (f"{ev.negatives_with_fp} of {ev.negatives_total} "
               f"({_pct(ev.negatives_with_fp / ev.negatives_total)})"
               if ev.negatives_total else "n/a")
        out += [f"## Results: {name}", "", *_type_table(ev), "",
                f"Negative messages (nothing sensitive) with at least one false alarm: {neg}.",
                "", f"### By category ({name})", "", *_category_table(ev), ""]

    if name_forms:
        out += ["## Names by form (test)", "",
                "How NAME recall depends on how the name is written.", "",
                "| Form | Recall | Coverage |", "|---|---:|---:|"]
        out += [f"| {form} | {_pct(st.recall)} | {_pct(st.coverage)} |"
                for form, st in name_forms.items()]
        out.append("")

    if address_styles:
        out += ["## Addresses by style (test)", "",
                ("How ADDRESS detection depends on how the address is written. "
                 "`landmark_only` (\"near the metro station\", no place) is not an address by "
                 "the detector's definition, so it is an expected miss."), "",
                "| Style | Support | Recall | Coverage | Exact |", "|---|---:|---:|---:|---:|"]
        out += [f"| {style} | {st.support} | {_pct(st.recall)} | {_pct(st.coverage)} | "
                f"{_pct(st.exact / st.tp) if st.tp else '–'} |"
                for style, st in address_styles.items()]
        out.append("")

    out += ["## Latency (detection only)", "",
            "Best of 3 runs per message, on the test split, single CPU thread.", ""]
    for layer, latency in latencies.items():
        out += [f"### {layer}", "", *_latency_table(latency), ""]

    if hard_cases:
        out += ["## Known-hard formats (test split)", "",
                "Formats the rules are known not to handle; measured so the gaps are explicit.",
                "", "| Template | Recall | Coverage |", "|---|---:|---:|"]
        out += [f"| `{_short(t, 70)}` | {_pct(st.recall)} | {_pct(st.coverage)} |"
                for t, st in hard_cases.items()]
        out.append("")

    if notes:
        out += ["## Methodology notes", "", *[f"- {note}" for note in notes], ""]

    if "handwritten" in results:
        out += ["## Error examples (hand-written set)", "", *_examples(results["handwritten"]),
                ""]
    return "\n".join(out)


def render_errors(ev: Evaluation, limit: int = 40) -> str:
    groups: dict[tuple[str, EntityType], list] = defaultdict(list)
    for err in ev.errors:
        groups[(err.kind, err.type)].append(err)
    lines: list[str] = []
    for (kind, entity), errs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"== {kind} {entity} ({len(errs)}) ==")
        lines += [f"  [{e.sample_id}] {e.value!r} | {_short(e.text, 140)}" for e in errs[:limit]]
    return "\n".join(lines)
