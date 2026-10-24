import time

from leakguard.detect import detect_all
from leakguard.detect.secrets import detect_secrets, shannon_entropy
from leakguard.entities import EntityType as T


def test_entropy():
    assert shannon_entropy("aaaa") == 0.0
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9


def test_known_key_prefixes():
    for key in ["sk-proj-abcDEF1234567890abcdefXYZ", "AKIAIOSFODNN7EXAMPLE",
                "ghp_" + "a1B2" * 9, "AIza" + "Sy" + "A1b2C3d4E5" * 3 + "xyz"]:
        assert [s.type for s in detect_secrets(f"use {key} please")] == [T.SECRET], key


def test_key_value_and_bearer():
    text = 'password=hunter22 and api_key: "zx81" Authorization: Bearer abc.def.ghi123'
    assert len(detect_secrets(text)) == 3


def test_token_key_value():
    # Found in dev error analysis: a word-heavy random token after "token:" was missed.
    text = "token: WDnBzqgmlWvfblKeohSsxhf3ubmmUwGa"
    assert [s.text(text) for s in detect_secrets(text)] == ["WDnBzqgmlWvfblKeohSsxhf3ubmmUwGa"]
    assert detect_secrets("token: what is it?") == []
    assert detect_secrets("the token expired") == []


def test_natural_language_password_mentions_not_flagged():
    for text in ["my password is wrong", "reset password: how?", "I forgot my password",
                 "password: please help", "secret santa list", "the api key expired yesterday"]:
        assert detect_secrets(text) == [], text


def test_entropy_catches_random_tokens_not_hex_ids():
    assert detect_secrets("token Zx8Kq2LmP9wRt4Vb7Nc1Hy6Jd3Fg") != []
    assert detect_secrets("commit 9fceb02d0ae598e95dc970b74767f19372d61af8") == []
    assert detect_secrets("id 550e8400-e29b-41d4-a716-446655440000") == []


def test_code_identifiers_are_not_secrets():
    # Developers paste code; blocking every long camelCase name would make the guard unusable.
    for ident in ["getUserAccountBalance2024Handler", "MyApplicationConfigV2Loader",
                  "parseInvoiceLineItemsFromCsvV3"]:
        assert detect_secrets(f"see {ident}()") == [], ident


def test_detect_all_merges_everything():
    text = "Aadhaar 2345 6789 0124, a/c 9876543210, key sk-abcdefghijklmnopqrstuvwxyz12"
    assert [s.type for s in detect_all(text)] == [T.AADHAAR, T.BANK_ACCOUNT, T.SECRET]


def test_detect_all_is_fast_with_many_findings():
    # A request near the size limit can hold tens of thousands of numbers; merging the
    # detections must not be quadratic in their count.
    text = "call 9876543210 " * 20_000
    started = time.perf_counter()
    spans = detect_all(text)
    assert len(spans) == 20_000
    assert time.perf_counter() - started < 2.0


def test_detect_all_is_fast_on_adversarial_input():
    for text in ["9 " * 50_000, "a" * 100_000 + "@", "-" * 100_000, "A1b2" * 25_000]:
        started = time.perf_counter()
        detect_all(text)
        assert time.perf_counter() - started < 2.0
