from leakguard.evaluation.restore_report import (
    bracketless_share,
    measure_stream_overhead,
    render_restore_report,
    render_restore_tuning,
    summarize,
)


def _outcome(model="m1", stream=False, status=200, mentions=None, restored=0, unrestored=0):
    mentions = mentions or {}
    return {"conversation": "c", "model": model, "stream": stream, "tool": False,
            "status": status, "mentions": mentions, "restored": restored,
            "unrestored": unrestored, "mentioned": sum(mentions.values()) > 0}


def _results(*outcomes):
    return {"meta": {"date": "2026-10-25", "split": "dev", "models": ["m1"],
                     "conversations": 4, "max_tokens": 600, "temperature": "provider default",
                     "guard_commit": "abc1234"},
            "outcomes": list(outcomes)}


def test_summarize_rates():
    results = _results(
        _outcome(mentions={"exact": 2}, restored=2),
        _outcome(mentions={"exact": 1, "bracketless": 1}, restored=1, unrestored=1),
        _outcome(),
        _outcome(status=429),
        _outcome(stream=True, mentions={"case": 1}, restored=1))
    s = summarize(results)[("m1", False)]
    assert (s.replies, s.mentioned, s.fully_restored, s.no_mention, s.errors) == (4, 2, 1, 1, 1)
    assert (s.restored, s.unrestored) == (3, 1)
    assert s.forms == {"exact": 3, "bracketless": 1}
    assert summarize(results)[("m1", True)].fully_restored == 1


def test_bracketless_share():
    results = _results(_outcome(mentions={"exact": 98}), _outcome(stream=True,
                                                                  mentions={"bracketless": 2}))
    assert bracketless_share(results) == 0.02


def test_bracketless_share_without_mentions_is_zero():
    assert bracketless_share(_results(_outcome())) == 0.0


def test_tuning_verdict_at_threshold():
    accept = render_restore_tuning(_results(_outcome(mentions={"exact": 98, "bracketless": 2})))
    assert "Verdict: accept" in accept and "2.0%" in accept
    keep = render_restore_tuning(_results(_outcome(mentions={"exact": 9801,
                                                             "bracketless": 199})))
    assert "Verdict: keep off" in keep


def test_report_has_table_and_notes():
    overhead = {"pass-through": {"ttfb_p50": 0.1, "ttfb_p95": 0.2, "total_p50": 1.0,
                                 "total_p95": 2.0},
                "restoring": {"ttfb_p50": 0.3, "ttfb_p95": 0.4, "total_p50": 3.0,
                              "total_p95": 4.0}}
    md = render_restore_report(_results(_outcome(mentions={"exact": 1}, restored=1)), overhead,
                               ["first note", "second note"])
    assert "| Model | Mode | Replies |" in md
    assert "first note" in md and "second note" in md
    assert "abc1234" in md and "100.0%" in md


def test_overhead_keys():
    overhead = measure_stream_overhead(n=5)
    assert set(overhead) == {"pass-through", "restoring"}
    for timings in overhead.values():
        assert set(timings) == {"ttfb_p50", "ttfb_p95", "total_p50", "total_p95"}
        assert all(v > 0 for v in timings.values())


def test_tuning_page_lists_notes():
    md = render_restore_tuning(_results(_outcome(mentions={"exact": 1})), ["the decision"])
    assert "## Notes" in md and "- the decision" in md
