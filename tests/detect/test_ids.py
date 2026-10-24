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


def test_aadhaar_not_found_inside_a_grouped_card_number():
    # Found in dev error analysis: the first 12 digits of a grouped card number can pass
    # Verhoeff; as AADHAAR outranks CARD, the card's last 4 digits then leaked.
    for text in ["card 4023-5925-6341-6859 on file", "card 4639 0765 6828 0813 on file",
                 "x 4111 2345 6789 0124 y"]:
        assert detect_aadhaar(text) == [], text
    # a real Aadhaar followed by an unrelated number is still found
    text = "aadhaar 2345 6789 0124 9876543210"
    assert _vals(detect_aadhaar(text), text) == ["2345 6789 0124"]


def test_reference_number_context_overrides_aadhaar_shape():
    # Found in dev error analysis: every remaining Aadhaar false positive was an order
    # number that happened to pass Verhoeff (1 in 10 do).
    for text in ["but order 234567890124 still shows unpaid", "invoice no. 2345 6789 0124",
                 "txn ref 987654321012 failed"]:
        assert detect_aadhaar(text) == [], text
    # an Aadhaar keyword wins over a reference word
    text = "order blocked, my aadhaar 234567890124 is verified"
    assert _vals(detect_aadhaar(text), text) == ["234567890124"]
    text = "my id is 2345 6789 0124"  # "id" alone is too vague to rule Aadhaar out
    assert _vals(detect_aadhaar(text), text) == ["2345 6789 0124"]


def test_pan():
    text = "PAN: ABCPS1234K and abcps1234k"
    assert _vals(detect_pan(text), text) == ["ABCPS1234K", "abcps1234k"]
    assert detect_pan("ABCXS1234K") == []  # X is not a holder type
    assert detect_pan("XABCPS1234K") == []  # touching letters


def test_card_luhn():
    text = "card 4111 1111 1111 1111 or 4111-1111-1111-1111 or 378282246310005"
    assert len(detect_card(text)) == 3
    assert detect_card("4111 1111 1111 1112") == []


def test_unusual_card_lengths_need_card_context():
    # Found in dev error analysis: 1 in 10 random 14-digit ticket numbers pass Luhn.
    # Indian cards are almost always 15 (Amex) or 16 digits; other lengths need context.
    assert detect_card("Escalation 34702481911668: customer called") == []  # 14 digits, Luhn ok
    assert len(detect_card("my diners card 3470 2481 9116 68")) == 1
    assert len(detect_card("pay with 4111111111111111")) == 1  # 16 digits, no context needed


def test_ifsc():
    text = "IFSC HDFC0001234, sbin0005943"
    assert _vals(detect_ifsc(text), text) == ["HDFC0001234", "sbin0005943"]
    assert detect_ifsc("HDFC1001234") == [] and detect_ifsc("XHDFC0001234") == []
