import pytest

from leakguard.detect.address_signals import find_signals
from leakguard.detect.gazetteer import load_gazetteer


@pytest.fixture(scope="module")
def g():
    return load_gazetteer()


def _signals(g, text, locations=()):
    return [(text[s.start:s.end], s.kind, s.bare) for s in find_signals(text, g, locations)]


def _of_kind(g, text, kind):
    return [(value, bare) for value, k, bare in _signals(g, text) if k == kind]


def test_pin_with_cue(g):
    assert _of_kind(g, "pincode 411038", "pin") == [("411038", False)]
    assert _of_kind(g, "PIN: 411038", "pin") == [("411038", False)]
    assert _of_kind(g, "Pin-411038", "pin") == [("411038", False)]
    assert _of_kind(g, "411038", "pin") == [("411038", True)]


def test_invalid_or_blocked_pins(g):
    assert _of_kind(g, "999999", "pin") == []
    assert _of_kind(g, "OTP 411038", "pin") == []
    assert _of_kind(g, "order no 560034", "pin") == []
    assert _of_kind(g, "use code 560034 at checkout", "pin") == []
    assert _of_kind(g, "call 9876543210", "pin") == []


@pytest.mark.parametrize("text", ["Flat 302", "Flat No. 1102", "H.No. 45", "H.No 3-4-56",
                                  "house no 12", "Plot 7", "Door No 4-5", "Shop No. 3",
                                  "3rd floor", "Room 12", "QTR NO 45", "Flat 3B",
                                  "makaan no 227", "No. 14"])
def test_unit_forms(g, text):
    assert _of_kind(g, text, "unit") == [(text, False)]


@pytest.mark.parametrize("text", ["12/3", "B-204", "#45", "A/7"])
def test_bare_unit_forms(g, text):
    assert _of_kind(g, text, "unit") == [(text, True)]


def test_dates_are_not_units(g):
    assert _of_kind(g, "on 12/03/2024", "unit") == []


@pytest.mark.parametrize("text", ["Sai Krupa Apts", "Green Valley Society", "Shanti Niwas",
                                  "Krishna Kunj", "Shubham CHS", "krishna heights",
                                  "Cyber Towers"])
def test_building(g, text):
    assert _of_kind(g, text, "building") == [(text, False)]


def test_society_in_small_talk_is_not_a_building(g):
    assert _of_kind(g, "the society meeting is on Sunday", "building") == []
    assert _of_kind(g, "our society has 12 wings", "building") == []


@pytest.mark.parametrize("text", ["MG Road", "Nehru Marg", "gali no 4", "2nd Cross", "5th Main",
                                  "Sector 15", "Phase 2", "Block C", "Lake View Road",
                                  "45 Anna Salai", "lane no 7"])
def test_street(g, text):
    assert _of_kind(g, text, "street") == [(text, False)]


@pytest.mark.parametrize("text", ["near Hanuman Mandir", "opp. SBI ATM", "behind City Mall",
                                  "Hanuman Mandir ke paas", "nr. city mall",
                                  "next to the bus stand", "Near Rly Station"])
def test_landmark(g, text):
    assert _of_kind(g, text, "landmark") == [(text, False)]


def test_cue_before_a_place_is_not_a_landmark(g):
    assert _of_kind(g, "near Kothrud", "landmark") == []
    assert ("Kothrud", "place", False) in _signals(g, "near Kothrud")


def test_places_and_locations(g):
    found = _signals(g, "Kothrud, Pune")
    assert ("Kothrud", "place", False) in found and ("Pune", "place", False) in found
    assert ("Zorba", "place", False) in _signals(g, "Zorba", locations=[(0, 5)])


def test_place_inside_a_landmark_is_not_a_place(g):
    # Some post offices are named "Bus Stand" or "Hanuman Mandir".
    kinds = [k for _, k, _ in _signals(g, "I'm near the bus stand")]
    assert kinds == ["landmark"]


@pytest.mark.parametrize("text", ["Silver Springs Colony", "Palm Grove Layout", "Shastri Vihar",
                                  "Rose Garden Extn", "Kalyani Nagar"])
def test_locality_by_its_ending(g, text):
    # New layouts and colonies are often not post office names; their ending says "place".
    assert (text, "place", False) in _signals(g, text)


def test_locality_ending_alone_is_not_a_place(g):
    assert [k for _, k, _ in _signals(g, "the colony")] == []
