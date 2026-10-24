import random
import re

from leakguard.detect.checksums import luhn_valid, verhoeff_valid
from leakguard.detect.contact import UPI_HANDLES
from leakguard.evaluation.values import (
    NAME_POOLS,
    VALUE_MAKERS,
    name_makers,
    verhoeff_check_digit,
)


def _many(key, n=200):
    rng = random.Random(7)
    return [VALUE_MAKERS[key](rng) for _ in range(n)]


def _digits(s):
    return re.sub(r"\D", "", s)


def test_verhoeff_check_digit():
    assert verhoeff_check_digit("236") == "3"
    assert all(verhoeff_valid(d + verhoeff_check_digit(d)) for d in ["23456789012", "31234567890"])


def test_generated_values_are_valid_by_construction():
    # Checked with checksums/structure only — never with the detectors themselves.
    assert all(verhoeff_valid(_digits(v)) and _digits(v)[0] in "23456789"
               and len(_digits(v)) == 12 for v in _many("AADHAAR"))
    assert all(luhn_valid(_digits(v)) for v in _many("CARD"))
    assert all(re.fullmatch(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]", v.upper())
               for v in _many("PAN"))
    assert all(re.fullmatch(r"[A-Z]{4}0\d{6}", v) for v in _many("IFSC"))
    phones = [_digits(v)[-10:] for v in _many("PHONE")]
    assert all(len(p) == 10 and p[0] in "6789" for p in phones)
    assert all(v.split("@")[1] in UPI_HANDLES for v in _many("UPI"))
    assert all(v.split("@")[1] not in UPI_HANDLES for v in _many("UPI.unknown_handle"))
    assert all(9 <= len(v) <= 18 and v.isdigit() for v in _many("BANK_ACCOUNT"))
    assert all(v.isalpha() for v in _many("SECRET.letters"))


def test_every_declared_maker_exists_and_is_deterministic():
    required = {"AADHAAR", "AADHAAR.mixed", "PAN", "PAN.spaced", "PHONE", "PHONE.words",
                "PHONE.dots", "UPI", "UPI.unknown_handle", "EMAIL", "CARD", "IFSC",
                "BANK_ACCOUNT", "NAME", "ADDRESS", "SECRET.openai", "SECRET.aws",
                "SECRET.github", "SECRET.slack", "SECRET.google", "SECRET.stripe",
                "SECRET.jwt", "SECRET.pem", "SECRET.password", "SECRET.letters", "SECRET.token",
                "ORDER_ID", "OTP", "TICKET", "TIMESTAMP", "INVOICE", "DATE", "AMOUNT", "SHA",
                "UUID", "TOLLFREE", "CAMEL", "TRACKING", "COUPON", "BANK"}
    assert required <= set(VALUE_MAKERS)
    assert [VALUE_MAKERS["ADDRESS"](random.Random(1)) for _ in range(2)] == \
        [VALUE_MAKERS["ADDRESS"](random.Random(1)) for _ in range(2)]


def test_values_never_contain_marker_syntax():
    for key in VALUE_MAKERS:
        assert all("[[" not in v and "]]" not in v for v in _many(key, 50)), key


def test_name_pools_are_disjoint_between_splits():
    (dev_f, dev_l), (test_f, test_l) = NAME_POOLS["dev"], NAME_POOLS["test"]
    assert not set(dev_f) & set(test_f) and not set(dev_l) & set(test_l)
    assert min(len(dev_f), len(test_f)) >= 50 and min(len(dev_l), len(test_l)) >= 30


def test_name_forms():
    m = name_makers("dev")
    for seed in range(20):
        lower = m["NAME.lower"](random.Random(seed))
        assert lower == lower.lower()
        caps = m["NAME.caps"](random.Random(seed))
        assert caps == caps.upper()
        assert re.fullmatch(r"([A-Z]\. )+[A-Z][a-z]+", m["NAME.initial"](random.Random(seed)))
        assert len(m["NAME.three"](random.Random(seed)).split()) == 3
        assert " " not in m["NAME.surname"](random.Random(seed))
        assert " " not in m["NAME.first"](random.Random(seed))
