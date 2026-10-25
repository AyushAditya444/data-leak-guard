from leakguard.evaluation.address_tuning import TuningRow, select_setting


def _row(gap, loc, p, r, f1, cov):
    return TuningRow(gap=gap, loc=loc, precision=p, recall=r, f1=f1, coverage=cov,
                     negatives_flagged=0, p95_ms=1.0)


def test_select_setting_picks_the_f1_leader():
    rows = [_row(6, False, .90, .80, .847, .90), _row(12, False, .92, .85, .884, .92),
            _row(24, False, .88, .86, .870, .93)]
    assert select_setting(rows) == (12, False)


def test_coverage_breaks_near_ties():
    # 24 is within 1 F1 point of 12 and hides more of each address
    rows = [_row(12, False, .92, .85, .884, .92), _row(24, False, .90, .86, .880, .96)]
    assert select_setting(rows) == (24, False)


def test_locations_kept_only_if_they_help_without_hurting_precision():
    off = _row(12, False, .92, .85, .884, .92)
    helps = _row(12, True, .93, .90, .915, .95)
    assert select_setting([off, helps]) == (12, True)
    costs_precision = _row(12, True, .85, .95, .897, .97)
    assert select_setting([off, costs_precision]) == (12, False)
    no_gain = _row(12, True, .92, .84, .878, .91)
    assert select_setting([off, no_gain]) == (12, False)


def test_render_lists_the_dev_stopwords():
    from leakguard.evaluation.address_tuning import dev_stopwords, render
    words = dev_stopwords()
    assert "late" in words and "station" not in words  # "station" was seeded before dev
    md = render([_row(12, False, .9, .9, .9, .9)], (12, False), 10, None)
    assert all(f"`{w}`" in md for w in words)
