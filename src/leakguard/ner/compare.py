"""Compare the exported NER candidates on the DEV split and pick the one to ship.

    python -m leakguard.ner.export --all     # first: writes models/candidates/<key>/
    python -m leakguard.ner.compare          # writes docs/evaluation/ner-comparison.md

Dev only, by design: there is no option to run this on test. Every candidate is scored
as deployed (all rule detectors, including the cue-based name fallback, plus that model,
merged), and also as the model alone, which shows what the model itself can do.

Two settings are tuned per candidate on dev, like any hyperparameter: the score threshold
and the int8 quantisation mode (one scale per tensor or per channel), since neither is
best for every model.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from leakguard.detect.engine import RULE_DETECTORS
from leakguard.detect.name_cues import detect_name_cues
from leakguard.detect.spans import Span, merge_spans
from leakguard.entities import EntityType
from leakguard.evaluation.generate import SPLIT_SEEDS, build_split, name_form
from leakguard.evaluation.metrics import Evaluation, score
from leakguard.evaluation.report import measure_latency
from leakguard.evaluation.samples import Sample
from leakguard.ner.candidates import CANDIDATES
from leakguard.ner.export import CANDIDATES_DIR, QUANT_FILES
from leakguard.ner.model import NameModel

OUTPUT = Path("docs/evaluation/ner-comparison.md")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
WITHIN_F1 = 0.01  # "within 1.0 F1 point": then the faster model wins
LOWERCASE_TRIGGER = 0.80  # use the cue-based fallback if the model's lowercase recall is below

Cache = dict[str, list[Span]]  # message text -> spans


@dataclass
class CandidateResult:
    key: str
    repo: str
    quantisation: str  # the int8 mode chosen on dev: "per-tensor" or "per-channel"
    threshold: float  # the best one from the sweep; the rest of the row uses it
    precision: float  # NAME, as deployed
    recall: float
    f1: float
    model_f1: float  # NAME F1 of the model alone (no cue fallback), at its own best threshold
    negative_fp: int  # NAMEs flagged in messages that contain no personal data at all
    p50_ms: float
    p95_ms: float
    size_mb: float
    forms: dict[str, float]  # NAME recall per name form, as deployed
    model_forms: dict[str, float]  # ... and for the model alone
    sweep: dict[float, float]  # threshold -> F1, as deployed
    quant_f1: dict[str, float]  # int8 mode -> best deployed F1
    fp32: dict | None  # deployed F1, p50 and size of the unquantised model


def select_winner(rows: list[CandidateResult]) -> str:
    """Highest F1; but if others are within 1 F1 point of it, the fastest (p95) of those."""
    best = max(row.f1 for row in rows)
    close = [row for row in rows if row.f1 >= best - WITHIN_F1]
    return min(close, key=lambda row: row.p95_ms).key


def _names(model: NameModel, samples: list[Sample]) -> Cache:
    """Each message through the model once, at threshold 0, so thresholds can be swept on
    the cached spans: grouping never depends on the score, so this equals re-running."""
    model.threshold = 0.0
    return {s.text: model.detect(s.text) for s in samples}


def _best(samples: list[Sample], rules: Cache, names: Cache,
          ) -> tuple[float, Evaluation, dict[float, float]]:
    """(best threshold, its evaluation, threshold -> NAME F1) for rules + names."""
    def detector_at(threshold: float):
        def detect(text: str) -> list[Span]:
            return merge_spans([*rules[text], *(n for n in names[text] if n.score >= threshold)])
        return detect

    evals = {t: score(samples, detector_at(t)) for t in THRESHOLDS}
    sweep = {t: ev.by_type[EntityType.NAME].f1 for t, ev in evals.items()}
    threshold = max(sweep, key=lambda t: (sweep[t], -t))  # ties: the lower threshold
    return threshold, evals[threshold], sweep


def _forms(samples: list[Sample], rules: Cache, names: Cache, threshold: float,
           ) -> dict[str, float]:
    def detect(text: str) -> list[Span]:
        return merge_spans([*rules[text], *(n for n in names[text] if n.score >= threshold)])

    out = {}
    for form in sorted({f for s in samples if (f := name_form(s.template))}):
        subset = [s for s in samples if name_form(s.template) == form]
        out[form] = score(subset, detect).by_type[EntityType.NAME].recall
    return out


def evaluate_candidate(key: str, samples: list[Sample], rules: Cache,
                       rules_without_cues: Cache) -> CandidateResult:
    model_dir = CANDIDATES_DIR / key
    runs = {}
    for mode, filename in QUANT_FILES.items():
        model = NameModel.load(model_dir, model_file=filename)
        names = _names(model, samples)
        runs[mode] = (model, names, *_best(samples, rules, names))
    mode = max(runs, key=lambda m: runs[m][3].by_type[EntityType.NAME].f1)
    model, names, threshold, ev, sweep = runs[mode]
    st = ev.by_type[EntityType.NAME]
    model_threshold, model_ev, _ = _best(samples, rules_without_cues, names)
    negative = ev.by_category_type.get(("negative", EntityType.NAME))
    model.threshold = threshold
    latency = measure_latency(samples, model.detect)["all"]

    fp32 = None
    if (model_dir / "model.fp32.onnx").is_file():
        fp32_model = NameModel.load(model_dir, model_file="model.fp32.onnx")
        fp32_names = _names(fp32_model, samples)
        fp32_threshold, fp32_ev, _ = _best(samples, rules, fp32_names)
        fp32_model.threshold = fp32_threshold
        fp32 = {"f1": fp32_ev.by_type[EntityType.NAME].f1,
                "p50_ms": measure_latency(samples, fp32_model.detect)["all"]["p50_ms"],
                "size_mb": (model_dir / "model.fp32.onnx").stat().st_size / 1e6}

    return CandidateResult(
        key=key, repo=CANDIDATES[key].repo, quantisation=mode, threshold=threshold,
        precision=st.precision, recall=st.recall, f1=st.f1,
        model_f1=model_ev.by_type[EntityType.NAME].f1,
        negative_fp=negative.fp if negative else 0,
        p50_ms=latency["p50_ms"], p95_ms=latency["p95_ms"],
        size_mb=(model_dir / QUANT_FILES[mode]).stat().st_size / 1e6,
        forms=_forms(samples, rules, names, threshold),
        model_forms=_forms(samples, rules_without_cues, names, model_threshold),
        sweep=sweep, quant_f1={m: run[3].by_type[EntityType.NAME].f1 for m, run in runs.items()},
        fp32=fp32)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _forms_table(rows: list[CandidateResult], attr: str) -> list[str]:
    forms = sorted({f for row in rows for f in getattr(row, attr)})
    lines = ["| Candidate | " + " | ".join(forms) + " |", "|---|" + "---:|" * len(forms)]
    for r in rows:
        values = getattr(r, attr)
        lines.append(f"| `{r.key}` | "
                     + " | ".join(_pct(values[f]) if f in values else "–" for f in forms) + " |")
    return lines


def render_comparison(rows: list[CandidateResult], n_samples: int) -> str:
    rows = sorted(rows, key=lambda r: -r.f1)
    winner = select_winner(rows)
    chosen = next(row for row in rows if row.key == winner)
    out = [
        "# NER model comparison (dev split)",
        "",
        "_Generated by `python -m leakguard.ner.compare`. Do not edit by hand._",
        "",
        (f"- Data: the **dev** split ({n_samples} messages). The test split is never used "
         "here; its numbers are only reported in [report.md](report.md)."),
        ("- **As deployed:** all rule detectors (including the cue-based fallback for "
         "lowercase and all-caps names) plus the model, merged. **Model alone:** the same "
         "without the cue fallback, to show what the model itself can do."),
        ("- Tuned per candidate on dev: the score threshold (best NAME F1 over "
         f"{', '.join(f'{t:.1f}' for t in THRESHOLDS)}) and the int8 quantisation mode "
         "(one scale per tensor or per channel)."),
        "- Latency: the model alone, best of 3 runs per message, 1 CPU thread, int8.",
        ("- Not a candidate: `ai4bharat/IndicNER`. It is gated on Hugging Face (download "
         "needs a logged-in account that accepted its terms), so a stranger could not build "
         "the guard with one command."),
        "",
        "## Results",
        "",
        ("| Candidate | Model | int8 mode | Threshold | Precision | Recall | F1 "
         "| F1 model alone | FP in negatives | p50 | p95 | Size |"),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        out.append(f"| `{r.key}` | `{r.repo}` | {r.quantisation} | {r.threshold:.1f} "
                   f"| {_pct(r.precision)} | {_pct(r.recall)} | {_pct(r.f1)} "
                   f"| {_pct(r.model_f1)} | {r.negative_fp} | {r.p50_ms:.1f} ms "
                   f"| {r.p95_ms:.1f} ms | {r.size_mb:.0f} MB |")
    out += ["", "## NAME recall by name form: model alone", "", *_forms_table(rows, "model_forms"),
            "", "## NAME recall by name form: as deployed", "", *_forms_table(rows, "forms"),
            "", "## Quantisation: what int8 costs and gains", "",
            "F1 as deployed, each at its best threshold.", "",
            ("| Candidate | F1 fp32 | F1 int8 per-tensor | F1 int8 per-channel "
             "| p50 fp32 → int8 | Size fp32 → int8 |"),
            "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        fp32_f1 = _pct(r.fp32["f1"]) if r.fp32 else "–"
        speed = f"{r.fp32['p50_ms']:.1f} → {r.p50_ms:.1f} ms" if r.fp32 else "–"
        size = f"{r.fp32['size_mb']:.0f} → {r.size_mb:.0f} MB" if r.fp32 else "–"
        quant = [_pct(r.quant_f1[m]) if m in r.quant_f1 else "–"
                 for m in ("per-tensor", "per-channel")]
        out.append(f"| `{r.key}` | {fp32_f1} | {quant[0]} | {quant[1]} | {speed} | {size} |")
    out += ["", "## Threshold sweep (NAME F1 as deployed, chosen int8 mode)", "",
            "| Candidate | " + " | ".join(f"{t:.1f}" for t in THRESHOLDS) + " |",
            "|---|" + "---:|" * len(THRESHOLDS)]
    for r in rows:
        cells = [_pct(r.sweep[t]) if t in r.sweep else "–" for t in THRESHOLDS]
        out.append(f"| `{r.key}` | " + " | ".join(cells) + " |")
    out += ["", "## Decision", "",
            ("Rule (fixed before running): highest NAME F1 on dev, as deployed; if other "
             "candidates are within 1.0 F1 point of it, the one with the lowest p95 latency "
             "among them."), "",
            (f"**Selected: `{winner}`** (`{chosen.repo}`), int8 {chosen.quantisation}, "
             f"threshold {chosen.threshold:.1f}."), ""]
    lower = chosen.model_forms.get("lower")
    if lower is not None:
        used = "is used" if lower < LOWERCASE_TRIGGER else "is not needed"
        out.append(f"Lowercase names: the selected model alone finds {_pct(lower)} of them "
                   f"(trigger: below {_pct(LOWERCASE_TRIGGER)}), so the cue-based fallback "
                   f"{used}; as deployed, lowercase recall is "
                   f"{_pct(chosen.forms.get('lower', 0.0))}.")
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    argparse.ArgumentParser(prog="python -m leakguard.ner.compare",
                            description=__doc__.splitlines()[0]).parse_args()
    keys = [k for k in CANDIDATES
            if all((CANDIDATES_DIR / k / f).is_file() for f in QUANT_FILES.values())]
    if not keys:
        sys.exit(f"No exported candidates in {CANDIDATES_DIR}. "
                 "Run: python -m leakguard.ner.export --all")
    dev = build_split("dev", SPLIT_SEEDS["dev"])
    rules = {s.text: [sp for d in RULE_DETECTORS for sp in d(s.text)] for s in dev}
    without_cues = {s.text: [sp for d in RULE_DETECTORS if d is not detect_name_cues
                             for sp in d(s.text)] for s in dev}
    rows = []
    for key in keys:
        print(f"evaluating {key} on {len(dev)} dev messages...", flush=True)
        rows.append(evaluate_candidate(key, dev, rules, without_cues))
        r = rows[-1]
        print(f"  F1 {_pct(r.f1)} (model alone {_pct(r.model_f1)}), int8 {r.quantisation}, "
              f"threshold {r.threshold:.1f}, p95 {r.p95_ms:.1f} ms", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_comparison(rows, len(dev)), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}; selected {select_winner(rows)}")


if __name__ == "__main__":
    main()
