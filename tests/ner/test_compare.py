from leakguard.ner.compare import CandidateResult, render_comparison, select_winner


def _row(key, f1, p95, **kw):
    defaults = {"precision": f1, "recall": f1, "threshold": 0.5, "quantisation": "per-channel",
                "negative_fp": 0, "p50_ms": p95 / 2, "size_mb": 100.0,
                "forms": {"lower": 0.9}, "model_f1": f1 - 0.1, "model_forms": {"lower": 0.5},
                "sweep": {0.5: f1}, "quant_f1": {"per-channel": f1, "per-tensor": f1 - 0.02},
                "fp32": None}
    return CandidateResult(key=key, repo=f"org/{key}", f1=f1, p95_ms=p95,
                           **{**defaults, **kw})


def test_selection_picks_f1_leader():
    rows = [_row("a", 0.80, 5.0), _row("b", 0.90, 20.0)]
    assert select_winner(rows) == "b"


def test_selection_prefers_faster_within_one_point():
    rows = [_row("slow", 0.905, 30.0), _row("fast", 0.897, 8.0), _row("worse", 0.85, 1.0)]
    assert select_winner(rows) == "fast"


def test_render_states_rule_winner_and_fallback_decision():
    rows = [_row("a", 0.80, 5.0), _row("b", 0.90, 20.0, model_forms={"lower": 0.40})]
    md = render_comparison(rows, n_samples=10)
    assert "**Selected: `b`**" in md and "within 1.0 F1 point" in md
    assert "dev" in md and "IndicNER" in md
    # the trigger is judged on the model alone (40% < 80%), so the fallback is used
    assert "model alone" in md and "40.0%" in md and "fallback is used" in md


def test_render_shows_both_quantisation_modes():
    md = render_comparison([_row("a", 0.80, 5.0)], n_samples=10)
    assert "per-tensor" in md and "per-channel" in md and "78.0%" in md
