from leakguard.detect.checksums import luhn_valid, verhoeff_valid
from leakguard.detect.spans import Span, merge_spans
from leakguard.entities import EntityType as T


def test_verhoeff_known_values():
    assert verhoeff_valid("2363")
    assert verhoeff_valid("234567890124") and verhoeff_valid("987654321012")
    assert not verhoeff_valid("234567890125")
    assert not verhoeff_valid("") and not verhoeff_valid("12a4")


def test_luhn_known_values():
    assert luhn_valid("4111111111111111") and luhn_valid("378282246310005")
    assert not luhn_valid("4111111111111112") and not luhn_valid("")


def test_merge_prefers_priority_then_length():
    a = Span(0, 10, T.PHONE)
    b = Span(0, 10, T.BANK_ACCOUNT)
    c = Span(20, 30, T.EMAIL)
    assert merge_spans([a, b, c]) == [b, c]
    upi = Span(5, 20, T.UPI)
    email = Span(5, 24, T.EMAIL)
    assert merge_spans([email, upi]) == [upi]
    short = Span(0, 4, T.PHONE)
    long = Span(0, 12, T.PHONE)
    assert merge_spans([short, long]) == [long]


def test_merge_keeps_non_overlapping_in_order():
    spans = [Span(30, 35, T.PAN), Span(0, 5, T.PHONE), Span(5, 9, T.EMAIL)]  # touching ≠ overlap
    assert [s.start for s in merge_spans(spans)] == [0, 5, 30]
