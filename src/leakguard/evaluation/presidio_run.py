"""Runs the real Presidio for the baseline (needs `pip install -e ".[baseline]"`).

Presidio and spaCy are imported inside functions so that the rest of the evaluation, and CI,
never need them.
"""

import datetime
import os
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

from leakguard.evaluation import THREAD_VARS
from leakguard.evaluation.loading import load_detector, load_split
from leakguard.evaluation.metrics import evaluation_to_json, score
from leakguard.evaluation.presidio_baseline import (
    Analyze,
    Found,
    PresidioDetector,
    tune_thresholds,
)
from leakguard.evaluation.report import measure_latency
from leakguard.evaluation.restore_run import _git_commit

SETUPS = ("out_of_box", "best_shot")
INDIAN = ("IN_AADHAAR", "IN_PAN", "IN_PASSPORT", "IN_VEHICLE_REGISTRATION", "IN_VOTER",
          "IN_GSTIN")


def _engine():
    from presidio_analyzer import AnalyzerEngine
    return AnalyzerEngine()  # default registry, default spaCy model, default threshold (0)


def make_analyzer(setup: str) -> Analyze:
    if setup not in SETUPS:
        raise ValueError(f"unknown setup {setup!r}")
    engine = _engine()
    if setup == "best_shot":
        from presidio_analyzer import predefined_recognizers
        for name in dir(predefined_recognizers):
            if name.startswith("In") and name.endswith("Recognizer"):
                engine.registry.add_recognizer(getattr(predefined_recognizers, name)())

    def analyze(text: str) -> list[Found]:
        return [Found(r.entity_type, r.start, r.end, r.score)
                for r in engine.analyze(text, language="en")]
    return analyze


def indian_defaults() -> dict[str, bool]:
    """Which of Presidio's Indian types a default AnalyzerEngine() detects."""
    supported = set(_engine().get_supported_entities("en"))
    return {name: name in supported for name in INDIAN}


def versions() -> dict[str, str]:
    found = {}
    for package in ("presidio-analyzer", "spacy", "en_core_web_lg"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not installed"
    return found


def _meta(sizes: Mapping[str, int]) -> dict:
    return {"date": datetime.date.today().isoformat(), "guard_commit": _git_commit(),
            "versions": versions(), "indian_defaults": indian_defaults(),
            "threads": {name: os.environ.get(name) for name in THREAD_VARS},
            "sizes": dict(sizes)}


def run_dev() -> dict:
    dev = load_split("dev")
    thresholds, rows = tune_thresholds(dev, make_analyzer("best_shot"))
    return {"meta": _meta({"dev": len(dev)}), "thresholds": thresholds,
            "tuning": [asdict(row) for row in rows]}


def _score_presidio(analyze: Analyze, thresholds: Mapping[str, float], splits: dict) -> dict:
    result: dict = {}
    for split, samples in splits.items():
        det = PresidioDetector(analyze, thresholds)  # fresh counters per split
        result[split] = evaluation_to_json(score(samples, det.detect, cover=det.cover))
        if split == "test":
            result["unmapped"] = dict(det.unmapped)
            result["conflicts"] = det.conflicts
    # Latency: the analyser itself, un-memoised, so every message is really analysed
    # (thresholds and conflict resolution are negligible next to spaCy).
    result["latency"] = measure_latency(splits["test"], analyze)["all"]
    return result


def run_test(thresholds: Mapping[str, float]) -> dict:
    splits = {"test": load_split("test"), "handwritten": load_split("handwritten")}
    detect, _, _ = load_detector(require_names=True)
    ours = {split: evaluation_to_json(score(samples, detect))
            for split, samples in splits.items()}
    ours["latency"] = measure_latency(splits["test"], detect)["all"]
    engines = {"ours": ours,
               "presidio_out_of_box": _score_presidio(make_analyzer("out_of_box"), {}, splits),
               "presidio_best_shot": _score_presidio(make_analyzer("best_shot"), thresholds,
                                                     splits)}
    return {"meta": _meta({split: len(s) for split, s in splits.items()}),
            "thresholds": dict(thresholds), "engines": engines}
