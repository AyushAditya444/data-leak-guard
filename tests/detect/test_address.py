import random
import time

import pytest

from leakguard.detect.address import detect_address
from leakguard.detect.gazetteer import load_gazetteer
from leakguard.entities import EntityType as T


def _found(text):
    spans = detect_address(text)
    assert all(s.type is T.ADDRESS for s in spans)
    return [s.text(text) for s in spans]


def test_full_address_is_one_span():
    text = "Deliver to Flat 302, Sai Krupa Apts, Kothrud, Pune 411038. Call me"
    assert _found(text) == ["Flat 302, Sai Krupa Apts, Kothrud, Pune 411038"]


def test_landmark_address():
    assert _found("I'm near Hanuman Mandir, Kothrud") == ["near Hanuman Mandir, Kothrud"]


@pytest.mark.parametrize("text", ["I live in Pune", "I live in Kothrud, Pune",
                                  "stuck in traffic near Kothrud", "near the metro station",
                                  "OTP 411038", "score was 12/3", "order no 560034"])
def test_not_addresses(text):
    assert _found(text) == []


def test_cued_pin_alone():
    assert _found("my pincode is 411038") == ["411038"]


def test_crlf_multiline_is_one_span():
    text = "Ship to:\r\nFlat 302,\r\nSai Krupa Apts,\r\nKothrud, Pune 411038\r\nThanks"
    assert _found(text) == ["Flat 302,\r\nSai Krupa Apts,\r\nKothrud, Pune 411038"]


def test_small_talk_not_masked():
    assert _found("I'll meet you at the station near the market") == []
    assert _found("I'm near the bus stand") == []


def test_two_addresses_two_spans():
    text = "from Flat 2, Baner, Pune to H.No. 45, Laxmi Nagar, Delhi"
    assert _found(text) == ["Flat 2, Baner, Pune", "H.No. 45, Laxmi Nagar, Delhi"]


def test_many_six_digit_numbers_is_fast():
    pins = sorted(load_gazetteer().pins)
    rng = random.Random(1)
    text = " ".join(rng.choice(pins) for _ in range(20_000))
    started = time.perf_counter()
    assert _found(text) == []  # bare PINs never chain into an address on their own
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("text", ["", "   ", "🙏🙏", "फ्लैट 302, कोथरूड, पुणे ४११०३८",
                                  "４１１０３８"])
def test_odd_inputs_return_lists(text):
    assert isinstance(detect_address(text), list)


def test_stop_spans_break_addresses():
    from leakguard.detect.spans import Span
    text = "Flat 2, Baner 9876543210 Pune"
    phone = Span(14, 24, T.PHONE)
    spans = detect_address(text, stops=[phone])
    assert all(not s.overlaps(phone) for s in spans)
    assert [s.text(text) for s in spans] == ["Flat 2, Baner"]


def test_leading_place_joined_by_words_is_not_part_of_the_address():
    # "Reddy" is also a village name; words between it and the address keep it out
    text = "Reddy here, my new address is behind Axis Bank, Kothrud, Pune"
    assert _found(text) == ["behind Axis Bank, Kothrud, Pune"]
    # places joined by separators only stay in: "Kothrud, Pune 411038"
    assert _found("Kothrud, Pune 411038") == ["Kothrud, Pune 411038"]
