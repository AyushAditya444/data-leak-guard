from importlib.resources import files

import pytest

from leakguard.detect.gazetteer import load_gazetteer, normalise_office


@pytest.fixture(scope="module")
def g():
    return load_gazetteer()


def _found(g, text: str) -> list[tuple[str, str]]:
    return [(text[s:e], kind) for s, e, kind in g.find_places(text)]


@pytest.mark.parametrize("raw, clean", [
    ("Kothrud S.O", "Kothrud"),
    ("Baner B.O", "Baner"),
    ("Pune H.O", "Pune"),
    ("Andheri (West) S.O", "Andheri"),
    ("Laxmi Nagar SO East Delhi", "Laxmi Nagar"),
    ("Baner DSO (Pune)", "Baner"),
    ("MARKET YARD S.O (Beed)", "Market Yard"),
    ("Papanpet BO", "Papanpet"),
])
def test_normalise_office(raw, clean):
    assert normalise_office(raw) == clean


def test_known_pins_and_places(g):
    assert "411038" in g.pins and "110092" in g.pins
    assert "999999" not in g.pins
    found = dict(_found(g, "Flat 2, Kothrud, Pune"))
    assert found["Kothrud"] == "locality"
    assert found["Pune"] in ("city", "district")


def test_longest_match_first(g):
    assert _found(g, "B-12, Navi Mumbai") == [("Navi Mumbai", "city")]


def test_direction_word_belongs_to_the_place(g):
    # "Flat 2, Andheri West" must not leave "West" outside the mask.
    assert ("Andheri West", "locality") in _found(g, "Flat 2, Andheri West")


def test_whole_words_only(g):
    assert _found(g, "Punekar pune123") == []


def test_common_words_are_not_places(g):
    assert _found(g, "meet me at the station near the market camp") == []


def test_bundle_is_small():
    bundle = files("leakguard").joinpath("data/pin_directory.tsv.gz")
    assert len(bundle.read_bytes()) < 5_000_000
