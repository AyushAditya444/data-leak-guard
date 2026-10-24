from leakguard.detect.contact import (
    detect_bank_account,
    detect_email,
    detect_phone,
    detect_upi,
)


def _vals(spans, text):
    return [s.text(text) for s in spans]


def test_phone_formats():
    text = "call 9876543210, +91 98765 43210, +91-9876543210, 09876543210 or 98765-43210"
    assert _vals(detect_phone(text), text) == [
        "9876543210", "+91 98765 43210", "+91-9876543210", "09876543210", "98765-43210"]


def test_phone_rejects_lookalikes():
    for text in ["OTP 482913", "id 5876543210", "98765432101234", "ref 1234567890"]:
        assert detect_phone(text) == [], text


def test_upi_vs_email():
    text = "pay rahul.s@okicici or 9876543210@ybl, mail rahul@gmail.com, x@unknownbank"
    assert _vals(detect_upi(text), text) == ["rahul.s@okicici", "9876543210@ybl"]
    assert _vals(detect_email(text), text) == ["rahul@gmail.com"]
    assert detect_upi("me@okicici.com") == []  # a domain, not a UPI handle


def test_upi_at_end_of_sentence():
    # Found in dev error analysis: the full stop made the handle look like a domain.
    text = "wants refund to priya.bose@upi. Thanks"
    assert _vals(detect_upi(text), text) == ["priya.bose@upi"]
    assert detect_upi("mail me@okicici.com") == []  # still a domain, not a UPI ID


def test_bank_account_with_context():
    text = "My a/c no 50100123456789 and account number 1234 5678 9012"
    assert _vals(detect_bank_account(text), text) == ["50100123456789", "1234 5678 9012"]
    text2 = "Send to 30012345678 IFSC SBIN0005943"
    assert _vals(detect_bank_account(text2), text2) == ["30012345678"]


def test_bank_account_requires_context():
    for text in ["Ticket 50100123456789 is open", "order no 123456789012 shipped"]:
        assert detect_bank_account(text) == [], text
