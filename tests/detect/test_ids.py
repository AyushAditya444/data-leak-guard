from leakguard.detect.ids import detect_aadhaar, detect_card, detect_ifsc, detect_pan
from leakguard.entities import EntityType as T


def _vals(spans, text):
    return [s.text(text) for s in spans]


def test_aadhaar_formats():
    text = "My Aadhaar is 2345 6789 0124, old one 9876-5432-1012, also 234567890124."
    assert _vals(detect_aadhaar(text), text) == [
        "2345 6789 0124", "9876-5432-1012", "234567890124"]
    assert all(s.type is T.AADHAAR for s in detect_aadhaar(text))


def test_aadhaar_rejects_bad_checksum_and_lookalikes():
    for text in ["order 123456789012", "2345 6789 0125", "0345 6789 0124",
                 "1234567890124", "id 12345678901234"]:
        assert detect_aadhaar(text) == [], text


def test_pan():
    text = "PAN: ABCPS1234K and abcps1234k"
    assert _vals(detect_pan(text), text) == ["ABCPS1234K", "abcps1234k"]
    assert detect_pan("ABCXS1234K") == []  # X is not a holder type
    assert detect_pan("XABCPS1234K") == []  # touching letters


def test_card_luhn():
    text = "card 4111 1111 1111 1111 or 4111-1111-1111-1111 or 378282246310005"
    assert len(detect_card(text)) == 3
    assert detect_card("4111 1111 1111 1112") == []


def test_ifsc():
    text = "IFSC HDFC0001234, sbin0005943"
    assert _vals(detect_ifsc(text), text) == ["HDFC0001234", "sbin0005943"]
    assert detect_ifsc("HDFC1001234") == [] and detect_ifsc("XHDFC0001234") == []
