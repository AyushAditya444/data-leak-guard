from leakguard.detect import detect_all
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
    ev = score(_samples(), detect_all)
    md = render_report({"test": ev, "handwritten": ev},
                       measure_latency(_samples(), detect_all, 1),
                       {"test": 4, "handwritten": 4})
    assert "| Type | Support | Precision | Recall | F1 | Coverage | Exact |" in md
    assert "overlaps" in md and "same type" in md
    assert "detector not built yet" in md
    assert "p95" in md


def test_category_table_excludes_unbuilt_types():
    # "[[NAME:...]] here" is the only gold in its sample; NAME has no detector yet, so the
    # support category's recall must come from PHONE + BANK_ACCOUNT only (both found).
    ev = score(_samples(), detect_all)
    md = render_report({"test": ev}, measure_latency(_samples(), detect_all, 1), {"test": 4})
    assert "| support | 100.0% | 100.0% | 0 |" in md


def test_latency_buckets():
    lat = measure_latency(_samples(), detect_all, repeats=1)
    assert lat["all"]["n"] == 4 and lat["all"]["p95_ms"] >= lat["all"]["p50_ms"] >= 0


def test_render_errors_lists_misses():
    ev = score(_samples(), detect_all)
    assert "FN NAME" in render_errors(ev)
