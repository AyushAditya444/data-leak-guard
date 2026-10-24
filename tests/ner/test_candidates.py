import re

from leakguard.ner.candidates import CANDIDATES, DEFAULT_CANDIDATE


def test_revisions_are_pinned_shas():
    # A branch name like "main" could change under us; a commit sha can't.
    assert all(re.fullmatch(r"[0-9a-f]{40}", c.revision) for c in CANDIDATES.values())


def test_default_candidate_exists():
    assert DEFAULT_CANDIDATE in CANDIDATES


def test_thresholds_in_range():
    assert all(0 < c.threshold < 1 for c in CANDIDATES.values())


def test_keys_match():
    assert all(key == c.key for key, c in CANDIDATES.items())
