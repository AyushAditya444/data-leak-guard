from leakguard.ner.compare import CandidateResult, render_comparison, select_winner


def _row(key, f1, p95, **kw):
    defaults = {"precision": f1, "recall": f1, "threshold": 0.5, "negative_fp": 0,
                "p50_ms": p95 / 2, "size_mb": 100.0, "forms": {"lower": 0.5},
                "sweep": {0.5: f1}, "fp32": None}
    return CandidateResult(key=key, repo=f"org/{key}", f1=f1, p95_ms=p95,
                           **{**defaults, **kw})


def test_selection_picks_f1_leader():
    rows = [_row("a", 0.80, 5.0), _row("b", 0.90, 20.0)]
    assert select_winner(rows) == "b"


def test_selection_prefers_faster_within_one_point():
    rows = [_row("slow", 0.905, 30.0), _row("fast", 0.897, 8.0), _row("worse", 0.85, 1.0)]
    assert select_winner(rows) == "fast"


def test_render_states_rule_winner_and_fallback_decision():
    rows = [_row("a", 0.80, 5.0, forms={"lower": 0.40}), _row("b", 0.90, 20.0,
                                                             forms={"lower": 0.85})]
    md = render_comparison(rows, n_samples=10)
    assert "**Selected: `b`**" in md and "within 1.0 F1 point" in md
    assert "dev" in md and "IndicNER" in md
    assert "fallback is not needed" in md  # b's lowercase recall 85% >= 80%
