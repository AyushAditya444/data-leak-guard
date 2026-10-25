"""The Presidio pages render from committed JSON alone (no Presidio needed)."""

from dataclasses import asdict

from leakguard.evaluation.metrics import TypeScores
from leakguard.evaluation.presidio_report import (
    render_presidio_baseline,
    render_presidio_tuning,
)

LATENCY = {"n": 10, "p50_ms": 1.0, "p95_ms": 2.0, "max_ms": 3.0}


def _ev(**types: TypeScores) -> dict:
    return {"by_type": {t: asdict(st) for t, st in types.items()},
            "negatives_total": 10, "negatives_with_fp": 1}


def _results(indian=None) -> dict:
    ours = _ev(AADHAAR=TypeScores(tp=9, fp=1, fn=1, gold_chars=100, covered_chars=95),
               NAME=TypeScores(tp=8, fp=0, fn=2, gold_chars=50, covered_chars=40),
               BANK_ACCOUNT=TypeScores(tp=5, fp=0, fn=0, gold_chars=60, covered_chars=60))
    presidio = _ev(AADHAAR=TypeScores(tp=4, fp=0, fn=6, gold_chars=100, covered_chars=40),
                   NAME=TypeScores(tp=6, fp=4, fn=4, gold_chars=50, covered_chars=30),
                   BANK_ACCOUNT=TypeScores(tp=0, fp=0, fn=5, gold_chars=60, covered_chars=36))
    engine = {"test": presidio, "handwritten": presidio, "latency": LATENCY,
              "unmapped": {"DATE_TIME": 3}, "conflicts": 2}
    return {"meta": {"date": "2026-10-25", "guard_commit": "abc1234",
                     "versions": {"presidio-analyzer": "2.2.364", "spacy": "3.8.16",
                                  "en_core_web_lg": "3.8.0"},
                     "indian_defaults": indian or {"IN_AADHAAR": False, "IN_PAN": False},
                     "threads": {"OMP_NUM_THREADS": "1"},
                     "sizes": {"test": 20, "handwritten": 5}},
            "thresholds": {"PERSON": 0.5},
            "engines": {"ours": {"test": ours, "handwritten": ours, "latency": LATENCY},
                        "presidio_out_of_box": engine, "presidio_best_shot": engine}}


def _row(md: str, start: str) -> list[str]:
    line = next(line for line in md.splitlines() if line.startswith(start))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_baseline_page_marks_unsupported_but_shows_coverage():
    md = render_presidio_baseline(_results(), [])
    row = _row(md, "| BANK_ACCOUNT")
    assert row[2] == row[3] == "not supported"  # both Presidio F1 cells
    assert row[1] == "100.0%"  # ours
    assert row[5] == row[6] == "60.0%"  # Presidio coverage is still shown


def test_baseline_page_both_support_row_uses_six_types():
    md = render_presidio_baseline(_results(), [])
    row = _row(md, "| **Types both support**")
    # ours over AADHAAR + NAME: tp 17, fp 1, fn 3 -> P 94.4%, R 85.0%, F1 89.5%
    assert row[1] == "**89.5%**"
    # Presidio: tp 10, fp 4, fn 10 -> P 71.4%, R 50.0%, F1 58.8%; BANK_ACCOUNT not included
    assert row[2] == "**58.8%**"


def test_baseline_page_reports_indian_defaults():
    md = render_presidio_baseline(_results(), [])
    assert "off by default" in md
    md = render_presidio_baseline(_results({"IN_AADHAAR": True, "IN_PAN": False}), [])
    assert "IN_AADHAAR" in md and "on by default" in md


def test_tuning_page_lists_chosen_thresholds():
    dev = {"meta": {"date": "2026-10-25", "guard_commit": "abc1234", "versions": {},
                    "sizes": {"dev": 20}},
           "thresholds": {"PERSON": 0.35},
           "tuning": [{"presidio_type": "PERSON", "threshold": t / 20, "tp": 5, "fp": 1,
                       "fn": 1, "f1": 0.5 + t / 100} for t in range(20)]}
    md = render_presidio_tuning(dev)
    assert _row(md, "| PERSON")[1] == "0.35"


def test_baseline_page_scopes_false_alarms():
    results = _results()
    for engine in ("presidio_out_of_box", "presidio_best_shot"):
        results["engines"][engine]["test"] = {
            **results["engines"][engine]["test"], "negatives_with_fp": 3,
            "negative_fp_types": [["ADDRESS"], ["NAME"], ["ADDRESS", "PHONE"]]}
    results["engines"]["ours"]["test"]["negative_fp_types"] = [["PHONE"]]
    md = render_presidio_baseline(results, [])
    row = _row(md, "| Presidio, best shot |")
    assert row[1] == "3 of 10" and row[2] == "2 of 10"  # ADDRESS-only alarms leave scope
    assert _row(md, "| Ours |")[1:3] == ["1 of 10", "1 of 10"]


def test_presidio_run_reexecs_with_a_fixed_hash_seed(monkeypatch):
    # Presidio returns equal-score results in an order that depends on Python's hash seed;
    # the run re-starts itself with PYTHONHASHSEED=0 so it can be reproduced exactly.
    import argparse
    import subprocess

    import pytest

    from leakguard.evaluation import __main__ as cli

    calls = []
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda cmd, env, check: calls.append((cmd, env))
                        or subprocess.CompletedProcess(cmd, 0))
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_presidio_run(argparse.Namespace(split="test"))
    assert exit_info.value.code == 0
    cmd, env = calls[0]
    assert env["PYTHONHASHSEED"] == "0"
    assert cmd[-3:] == ["presidio-run", "--split", "test"]


def test_both_support_summary_matches_committed_page():
    # The demo's results scene shows these; they must equal the committed Presidio page.
    import json
    from pathlib import Path

    from leakguard.evaluation.presidio_report import both_support_summary

    results = json.loads(Path("data/eval/presidio-test.json").read_text(encoding="utf-8"))
    summary = both_support_summary(results)
    assert list(summary) == ["ours", "presidio_out_of_box", "presidio_best_shot"]
    assert [round(100 * s["f1"], 1) for s in summary.values()] == [95.3, 62.6, 74.5]
    assert [round(100 * s["coverage"], 1) for s in summary.values()] == [93.9, 79.6, 94.3]
    assert [s["false_alarms"] for s in summary.values()] == [4, 67, 68]
    assert {s["negatives"] for s in summary.values()} == {350}
    assert summary["presidio_best_shot"]["label"] == "Presidio, best shot"


def test_zero_over_zero_precision_shows_dash():
    # Nothing predicted: precision is undefined (0/0), not 0%. Recall and F1 are 0% for real.
    results = _results()
    results["engines"]["presidio_out_of_box"]["test"]["by_type"]["AADHAAR"].update(
        tp=0, fp=0, fn=10)
    md = render_presidio_baseline(results, [])
    assert "| AADHAAR | 10 | – | 0.0% | 0.0% |" in md
