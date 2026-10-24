from leakguard.detect import detect_rules
from leakguard.evaluation.metrics import TypeScores, score
from leakguard.evaluation.report import measure_latency, render_errors, render_report
from leakguard.evaluation.samples import Sample, parse_marked


def _samples():
    out = []
    for i, marked in enumerate(["call [[PHONE:9876543210]]", "order 123456789012",
                                "[[ADDRESS:Flat 2, MG Road, Pune]] here",
                                "a/c [[BANK_ACCOUNT:50100123456789]]"]):
        text, spans = parse_marked(marked)
        out.append(Sample(f"s{i}", "negative" if i == 1 else "support", "t", text, spans))
    return out


def _lat():
    return {"rules only": measure_latency(_samples(), detect_rules, 1)}


def test_report_contains_tables_rule_and_unbuilt_detectors():
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev, "handwritten": ev}, _lat(),
                       {"test": 4, "handwritten": 4})
    assert "| Type | Support | Precision | Recall | F1 | Coverage | Exact |" in md
    assert "overlaps" in md and "same type" in md
    assert "detector not built yet" in md
    assert "p95" in md


def test_category_table_excludes_unbuilt_types():
    # "[[ADDRESS:...]] here" is the only gold in its sample; ADDRESS has no detector yet, so
    # the support category's recall must come from PHONE + BANK_ACCOUNT only (both found).
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev}, _lat(), {"test": 4})
    assert "| support | 100.0% | 100.0% | 0 |" in md


def test_report_includes_methodology_notes():
    ev = score(_samples(), detect_rules)
    md = render_report({"test": ev}, _lat(), {"test": 4},
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
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "LEAKGUARD_NER": "off"}
    run = subprocess.run([sys.executable, "-m", "leakguard.evaluation", "errors",
                          "--split", "handwritten"], check=False, capture_output=True,
                         env=env)
    assert run.returncode == 0, run.stderr.decode(errors="replace")[-500:]


def test_render_errors_lists_misses():
    ev = score(_samples(), detect_rules)
    assert "FN ADDRESS" in render_errors(ev)


def test_report_has_layer_latency_tables():
    lat = measure_latency(_samples(), detect_rules, 1)
    md = render_report({"test": score(_samples(), detect_rules)},
                       {"rules only": lat, "rules + names": lat}, {"test": 4})
    assert "### rules only" in md and "### rules + names" in md


def test_name_no_longer_marked_unbuilt():
    text, spans = parse_marked("[[NAME:Ravi Kumar]] here")
    ev = score([Sample("n", "support", "t", text, spans)], detect_rules)
    md = render_report({"test": ev}, _lat(), {"test": 1})
    assert "| NAME | 1 |" in md and "NAME *(detector not built" not in md


def test_report_has_name_forms_table_and_model_line():
    forms = {"lower": TypeScores(tp=3, fn=1, gold_chars=40, covered_chars=30)}
    info = {"candidate": "bert-base", "repo": "dslim/bert-base-NER",
            "revision": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc", "size_mb": 109.0,
            "threads": 1}
    md = render_report({"test": score(_samples(), detect_rules)}, _lat(), {"test": 4},
                       name_forms=forms, model_info=info)
    assert "## Names by form (test)" in md and "| lower | 75.0% | 75.0% |" in md
    assert "`dslim/bert-base-NER@d1a3e8f`" in md and "109 MB" in md


def test_report_cli_refuses_without_the_name_model():
    import os
    import subprocess
    import sys
    env = {**os.environ, "LEAKGUARD_NER": "off"}
    run = subprocess.run([sys.executable, "-m", "leakguard.evaluation", "report"],
                         check=False, capture_output=True, env=env)
    assert run.returncode != 0 and b"name model" in run.stderr
