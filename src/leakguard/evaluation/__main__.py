"""Command line: python -m leakguard.evaluation {generate,report,errors}. Run from repo root."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from leakguard.config import Settings
from leakguard.detect import detect_rules
from leakguard.detect.address import detect_address
from leakguard.detect.engine import Detector
from leakguard.entities import EntityType
from leakguard.evaluation.generate import SPLIT_SEEDS, address_style, build_split, name_form
from leakguard.evaluation.loading import DATA, load_detector, load_split
from leakguard.evaluation.metrics import score
from leakguard.evaluation.report import measure_latency, render_errors, render_report
from leakguard.evaluation.samples import write_jsonl
from leakguard.ner.model import NameModel

REPORT = Path("docs/evaluation/report.md")
FIXTURES = Path("tests/fixtures/groq_streams")

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


def cmd_restore_run(args: argparse.Namespace) -> None:
    import httpx

    from leakguard.evaluation import restore_run as rr

    key = rr.api_key_from_env()
    conversations = rr.build_conversations(args.split, rr.SIZES[args.split], rr.SEEDS[args.split])
    conversations = conversations[:args.limit] if args.limit else conversations
    _, model, _ = load_detector(require_names=False)  # the guard as shipped: names on
    print(f"{len(conversations)} conversations x {len(rr.MODELS)} models x 2 modes",
          file=sys.stderr)
    result = asyncio.run(rr.run_conversations(
        conversations, rr.MODELS, httpx.AsyncHTTPTransport(), key, rr.MAX_TOKENS,
        args.save_streams, model.detect if model else None))
    result["meta"]["split"] = args.split
    for i, fixture in enumerate(result.pop("fixtures", []), 1):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        path = FIXTURES / f"{args.split}-{i:02d}.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    out = Path(args.out) if args.out else DATA / f"restore-{args.split}.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    statuses: dict[int, int] = {}
    for outcome in result["outcomes"]:
        statuses[outcome["status"]] = statuses.get(outcome["status"], 0) + 1
    print(f"wrote {out}: {len(result['outcomes'])} replies, statuses {statuses}")


RESTORE_NOTES = [
    ("Groq's models and their behaviour change over time, and replies are sampled (provider "
     "default temperature), so a re-run will not give identical numbers."),
    ("Placeholders whose number is written as a word (\"Name One\") are neither restored nor "
     "counted as unrestored: they cannot be told apart from ordinary words."),
    ("The conversations, instructions and the evaluation messages were written by the same "
     "author as the guard. Samples from the `hard` category (formats the detector is known to "
     "miss) and anything containing a secret (blocked, so nothing comes back) are excluded."),
    ("Which placeholder forms are restored was decided on the dev run only "
     "([restore-tuning.md](restore-tuning.md)); this test run was made once, after that, and "
     "changed no code or setting."),
    ("A strict bare `PHONE_1` is restored even inside an unclosed `[PHONE_1`, which shows the "
     "value with a stray bracket; such mentions are still listed as `unclosed` in the forms "
     "table, which is read from the raw replies."),
    ("Errors are HTTP 400 `tool_use_failed` from Groq: the provider rejected a malformed tool "
     "call from the model. The guard blocked nothing (checked by re-sending one on dev)."),
]

DEV_NOTES = [
    ("Decision: bracketless placeholders accepted (`ACCEPT_BRACKETLESS = True` in "
     "`src/leakguard/placeholders.py`), strictly: uppercase type, underscore, word boundaries "
     "on both sides. Users typing that form get those numbers reserved, as for every accepted "
     "form."),
    ("The forms table counts every bracketless-looking mention (any case, any separator). "
     "On dev, 28 of the 29 unrestored placeholders were bracketless (the other was the unclosed "
     "one); how many of them the strict form restores is visible in the test run's "
     "`unrestored` column."),
    ("No other mutation reached 2%: one unclosed `[PHONE_1` (0.1%), so nothing else was added. "
     "Case, separator and bracket variants were accepted from the start and are rare."),
    ("The 3 errors are HTTP 400 `tool_use_failed` from Groq (gpt-oss-120b, non-streaming, "
     "forced tool call): the provider rejected the model's malformed tool call. The guard "
     "blocked nothing: one was re-sent directly (no blocked types, same error)."),
]


def cmd_restore_report(args: argparse.Namespace) -> None:
    from leakguard.evaluation import restore_report as rep

    results = json.loads((DATA / f"restore-{args.split}.json").read_text(encoding="utf-8"))
    if args.split == "dev":
        out = Path("docs/evaluation/restore-tuning.md")
        md = rep.render_restore_tuning(results, DEV_NOTES)
    else:
        out = Path("docs/evaluation/restore-report.md")
        md = rep.render_restore_report(results, rep.measure_stream_overhead(), RESTORE_NOTES)
    out.write_text(md + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


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
    run = sub.add_parser("restore-run", help="measure restoring on Groq (needs GROQ_API_KEY)")
    run.add_argument("--split", required=True, choices=["dev", "test"])
    run.add_argument("--limit", type=int, default=0, help="only the first N conversations")
    run.add_argument("--save-streams", type=int, default=0,
                     help="save N raw streamed replies as test fixtures")
    run.add_argument("--out", help="results path (default data/eval/restore-<split>.json)")
    rep = sub.add_parser("restore-report", help="render the restore tuning page (dev) or "
                         "report (test) from data/eval/restore-<split>.json")
    rep.add_argument("--split", required=True, choices=["dev", "test"])
    args = parser.parse_args()
    {"generate": cmd_generate, "report": cmd_report, "errors": cmd_errors,
     "restore-run": cmd_restore_run, "restore-report": cmd_restore_report}[args.command](args)


if __name__ == "__main__":
    main()
