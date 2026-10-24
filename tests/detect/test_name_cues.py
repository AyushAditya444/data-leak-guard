import pytest

from leakguard.detect.name_cues import detect_name_cues
from leakguard.entities import EntityType as T


def _names(text):
    spans = detect_name_cues(text)
    assert all(s.type is T.NAME and s.detector == "name_cues" for s in spans)
    return [s.text(text) for s in spans]


@pytest.mark.parametrize("text,expected", [
    ("mera naam rahul hai, order cancel kar do", ["rahul"]),
    ("MY NAME IS RAHUL SHARMA", ["RAHUL SHARMA"]),
    ("my name is priya nair and my order is late", ["priya nair"]),
    ("hello ji suraj kumar bol raha hu, parcel kab aayega", ["suraj kumar"]),
    ("main anjali hoon, mera number 9811122233 hai", ["anjali"]),
    ("NAME: RAHUL SHARMA\nMOBILE: 9876543210", ["RAHUL SHARMA"]),
    ("APPLICANT NAME - PRIYA NAIR / DOB - 01/02/1990", ["PRIYA NAIR"]),
    ("Cust name: sunita devi", ["sunita devi"]),
    ("Order 123 | Name - PRIYA NAIR | Mob 9876543210", ["PRIYA NAIR"]),
    ("Beneficiary name: ramesh kumar", ["ramesh kumar"]),
])
def test_cue_finds_name(text, expected):
    assert _names(text) == expected


@pytest.mark.parametrize("text", [
    "main theek hoon, bas refund chahiye",        # "I'm fine"
    "main ghar pe hoon, delivery kab hogi",        # "I'm at home"
    "my name is wrong on the invoice",
    "my name is spelled incorrectly",
    "naam kya hai aapka?",
    "rahul dravid batting kar raha hai",           # a name, but no cue: left to the model
    "Name:",
    "Name: 9876543210",
    "username: rahul123",
    # "name" fields that are not about a person
    "Bank name: SBI, branch Andheri",
    "Branch name - Andheri West",
    "file name: report.pdf",
    "Product name: Redmi Note",
])
def test_no_cue_no_name(text):
    assert _names(text) == []


def test_slot_is_at_most_three_words():
    text = "my name is ram lal gupta verma"
    assert _names(text) == ["ram lal gupta"]
