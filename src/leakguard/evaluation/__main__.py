"""Command line: python -m leakguard.evaluation {generate,report,errors}. Run from repo root."""

import argparse
import json
import sys
from pathlib import Path

from leakguard.config import Settings
from leakguard.detect import build_detector, detect_rules
from leakguard.detect.address import detect_address
from leakguard.detect.engine import Detector
from leakguard.entities import EntityType
from leakguard.evaluation.generate import SPLIT_SEEDS, address_style, build_split, name_form
from leakguard.evaluation.metrics import score
from leakguard.evaluation.report import measure_latency, render_errors, render_report
from leakguard.evaluation.samples import Sample, read_jsonl, write_jsonl
from leakguard.ner.model import NameModel, NameModelError

DATA = Path("data/eval")
REPORT = Path("docs/evaluation/report.md")

# Anything that affects how far the test numbers can be trusted is disclosed here.
NOTES = [
    ("After the final round of dev-set tuning, the test-set errors were inspected once. "
    "They revealed one defect class the dev set did not contain: key names with a prefix "
    "(`DB_PASSWORD=...`, `db_password: ...`). A dev template with that pattern was added, "
    "the detector was fixed against dev, and the report regenerated. Test SECRET recall "
    "before this fix was 72.9%; the test set was not used for any other decision."),
    ("UPI recall on the test split is pulled down by one hard template with two unknown UPI "
    "handles per message; see *Known-hard formats*."),
    ("Phase 4 extended the dataset (name forms, separate dev/test name pools, capitalised "
    "words that are not names). The seeded generator therefore produced new samples, so "
    "non-NAME rows moved slightly from the Phase 3 report. The name model, its threshold "
    "and int8 mode, and the cue-based fallback for lowercase/all-caps names were all "
    "chosen on dev only; no test-set inspection informed any Phase 4 decision."),
    ("The cue phrases of the name fallback (e.g. `mera naam ... hai`, `... bol raha hoon`) "
    "were written by the same author as the message templates, so the fallback's test "
    "numbers may be optimistic. The hand-written set and the lowercase test template "
    "without any cue (`bhaiya ... ko parcel de dena`) are the more independent checks."),
    ("Phase 5 extended the dataset (address styles, places drawn from India Post's PIN "
    "directory with dev and test using different districts, invented localities that are "
    "not post offices, and address look-alikes such as real PINs used as OTPs or order "
    "numbers). `agent_record` grew from 150 to 180 and `hard` from 100 to 170 messages per "
    "split so every address style has at least 20 samples. Non-ADDRESS rows therefore moved "
    "from the Phase 4 report: UPI, PAN and PHONE recall fell because the larger `hard` "
    "category holds more of their known-hard formats (unknown UPI handles, spaced PANs, "
    "numbers as words). On dev, adding the address detector changed no other type's "
    "recall."),
    ("Every address setting (gap 24, the name model's location tags off, the place stop "
    "list) and every address detector fix (locality endings such as Layout/Colony as "
    "places, dropping a leading place joined only by ordinary words, absorbing a name that "
    "overlaps an address) was chosen from the dev split; see "
    "[address-tuning.md](address-tuning.md). No test-set inspection informed any Phase 5 "
    "decision."),
    ("The address keyword lists, the generator's address vocabulary and the hand-written "
    "addresses were all written by the same author, so address numbers may be optimistic. "
    "In particular, the generator's invented localities (in both splits) often end in "
    "Layout, Colony or Extn, which the detector recognises as locality endings. Places and "
    "PINs in generated addresses come from the same PIN directory the detector uses "
    "(snapshot downloaded 2026-10-25), so directory places are found by construction; "
    "about a quarter of generated addresses use an invented locality instead."),
    ("A final code review of Phase 5 (a fresh reviewer probing the detector with its own "
    "inputs) found address leaks around phone numbers and false addresses in everyday "
    "prose (\"The ratio is 3/4, are you sure?\"). The fixes, a stop list of everyday words "
    "built from a hand-written common-word list, and six everyday-prose negatives were "
    "added; the dev tuning was re-run and this report regenerated. The test split was "
    "not inspected for these fixes. The negatives still cover only a few kinds of everyday "
    "prose, so precision on arbitrary chat may be lower than measured here."),
    ("A name right next to an address can be absorbed into the ADDRESS span (surnames like "
    "Sharma are also village names, and an address must never cut a name in half). The "
    "name is still hidden, but the typed metrics count it as a missed NAME. On dev, the "
    "address detector lowers NAME recall by 1.4 points while NAME coverage rises slightly."),
    ("`landmark_only` addresses (\"near the metro station\", no place) are labelled as "
    "addresses but are not addresses by the detector's definition, so they are counted as "
    "misses on purpose; see *Addresses by style*."),
]


def load_split(split: str) -> list[Sample]:
    """dev/test are rebuilt from the seeded generator, not read from disk: they are not
    committed, because their (fake) API keys look real to GitHub's secret scanning."""
    if split in SPLIT_SEEDS:
        return build_split(split, SPLIT_SEEDS[split])
    return read_jsonl(DATA / f"{split}.jsonl")


def load_detector(require_names: bool) -> tuple[Detector, NameModel | None, Settings]:
    """The full detector, configured like the guard (LEAKGUARD_NER, ..._MODEL_DIR)."""
    settings = Settings.from_env()
    if not settings.ner_enabled:
        if require_names:
            sys.exit("The report needs the name model, but LEAKGUARD_NER=off is set.")
        print("note: LEAKGUARD_NER=off, so names are not detected", file=sys.stderr)
        return detect_rules, None, settings
    try:
        model = NameModel.load(settings.ner_model_dir, settings.ner_threads)
    except NameModelError as exc:
        sys.exit(f"The name model could not be loaded ({exc}). "
                 "Run: python -m leakguard.ner.export")
    return build_detector(model.detect), model, settings  # as deployed: names only


def _memo(detect: Detector) -> Detector:
    """Score many slices of the same messages while running the model once per message."""
    cache: dict[str, list] = {}

    def cached(text: str) -> list:
        if text not in cache:
            cache[text] = detect(text)
        return cache[text]

    return cached


def _model_info(model: NameModel, settings: Settings) -> dict:
    model_dir = Path(settings.ner_model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"candidate": model.name, "repo": manifest["repo"],
            "revision": manifest["revision"], "threads": settings.ner_threads,
            "size_mb": (model_dir / "model.onnx").stat().st_size / 1e6}


def cmd_generate(_: argparse.Namespace) -> None:
    for split, seed in SPLIT_SEEDS.items():
        samples = build_split(split, seed)
        write_jsonl(DATA / f"{split}.jsonl", samples)
        print(f"wrote {len(samples)} samples to {DATA / f'{split}.jsonl'}")


def cmd_report(_: argparse.Namespace) -> None:
    detect, model, settings = load_detector(require_names=True)
    cached = _memo(detect)
    test, handwritten = load_split("test"), load_split("handwritten")
    results = {"test": score(test, cached), "handwritten": score(handwritten, cached)}
    hard_templates = sorted({s.template for s in test if s.category == "hard"})
    hard_cases = {}
    for template in hard_templates:
        ev = score([s for s in test if s.template == template], cached)
        total = next(iter(ev.by_category.values()))
        hard_cases[template] = total
    forms = sorted({f for s in test if (f := name_form(s.template))})
    name_forms = {form: score([s for s in test if name_form(s.template) == form],
                              cached).by_type[EntityType.NAME] for form in forms}
    styles = sorted({st for s in test if (st := address_style(s.template))})
    address_styles = {style: score([s for s in test if address_style(s.template) == style],
                                   cached).by_type[EntityType.ADDRESS] for style in styles}
    latencies = {"rules only": measure_latency(test, detect_rules),
                 "rules + name model": measure_latency(test, detect),
                 "address detector alone": measure_latency(test, detect_address)}
    md = render_report(results, latencies,
                       {"test": len(test), "handwritten": len(handwritten)}, hard_cases,
                       NOTES, name_forms, _model_info(model, settings), address_styles)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {REPORT}")


def cmd_errors(args: argparse.Namespace) -> None:
    detect, _, _ = load_detector(require_names=False)
    print(render_errors(score(load_split(args.split), detect), limit=args.limit))


def main() -> None:
    # Hindi text and symbols like "⏎" must not crash a legacy (cp1252) Windows console:
    # characters the console can't show are replaced instead of raising.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="python -m leakguard.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="write data/eval/dev.jsonl and test.jsonl (for browsing)")
    sub.add_parser("report", help="write docs/evaluation/report.md from test + handwritten")
    errors = sub.add_parser("errors", help="print false positives/negatives for one split")
    errors.add_argument("--split", default="dev", choices=["dev", "test", "handwritten"])
    errors.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    {"generate": cmd_generate, "report": cmd_report, "errors": cmd_errors}[args.command](args)


if __name__ == "__main__":
    main()
