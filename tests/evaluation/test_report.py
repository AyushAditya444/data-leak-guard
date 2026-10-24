from leakguard.detect import detect_rules
from leakguard.evaluation.metrics import score
from leakguard.evaluation.report import measure_latency, render_errors, render_report
from leakguard.evaluation.samples import Sample, parse_marked


def _samples():
    out = []
    for i, marked in enumerate(["call [[PHONE:9876543210]]", "order 123456789012",
                                "[[NAME:Ravi Kumar]] here",
                                "a/c [[BANK_ACCOUNT:50100123456789]]"]):
        text, spans = parse_marked(marked)
        out.append(Sample(f"s{i}", "negative" if i == 1 else "support", "t", text, spans))
    return out


def test_report_contains_tables_rule_and_unbuilt_detectors():
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev, "handwritten": ev},
                       measure_latency(_samples(), detect_rules, 1),
                       {"test": 4, "handwritten": 4})
    assert "| Type | Support | Precision | Recall | F1 | Coverage | Exact |" in md
    assert "overlaps" in md and "same type" in md
    assert "detector not built yet" in md
    assert "p95" in md


def test_category_table_excludes_unbuilt_types():
    # "[[NAME:...]] here" is the only gold in its sample; NAME has no detector yet, so the
    # support category's recall must come from PHONE + BANK_ACCOUNT only (both found).
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev}, measure_latency(_samples(), detect_rules, 1), {"test": 4})
    assert "| support | 100.0% | 100.0% | 0 |" in md


def test_report_includes_methodology_notes():
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev}, measure_latency(_samples(), detect_rules, 1), {"test": 4},
                       notes=["Test errors were inspected once."])
    assert "## Methodology notes" in md and "- Test errors were inspected once." in md


def test_latency_buckets():
    lat = measure_latency(_samples(), detect_rules, repeats=1)
    assert lat["all"]["n"] == 4 and lat["all"]["p95_ms"] >= lat["all"]["p50_ms"] >= 0


def test_errors_cli_works_on_a_legacy_windows_console():
    # Windows consoles often use cp1252, which can't encode "⏎" or Hindi text.
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    run = subprocess.run([sys.executable, "-m", "leakguard.evaluation", "errors",
                          "--split", "handwritten"], check=False, capture_output=True,
                         env=env)
    assert run.returncode == 0, run.stderr.decode(errors="replace")[-500:]


def test_render_errors_lists_misses():
    ev = score(_samples(), detect_rules)
    assert "FN NAME" in render_errors(ev)
