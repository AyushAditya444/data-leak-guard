"""Loading evaluation data and the detector, shared by the evaluation commands."""

import sys
from pathlib import Path

from leakguard.config import Settings
from leakguard.detect import build_detector, detect_rules
from leakguard.detect.engine import Detector
from leakguard.evaluation.generate import SPLIT_SEEDS, build_split
from leakguard.evaluation.samples import Sample, read_jsonl
from leakguard.ner.model import NameModel, NameModelError

DATA = Path("data/eval")


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
