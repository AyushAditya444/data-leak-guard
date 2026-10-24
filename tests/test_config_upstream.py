import pytest

from leakguard.config import Settings
from leakguard.upstream import upstream_url


@pytest.mark.parametrize("base,expected", [
    ("http://a/v1", "http://a/v1/chat/completions"),
    ("http://a/v1/", "http://a/v1/chat/completions"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/",
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
])
def test_upstream_url_join(base, expected):
    assert upstream_url(base, "/chat/completions") == expected


def test_settings_from_env_reads_values():
    s = Settings.from_env({"LEAKGUARD_UPSTREAM_BASE_URL": "http://x/v1",
                           "LEAKGUARD_UPSTREAM_TIMEOUT_S": "5",
                           "LEAKGUARD_MAX_REQUEST_BYTES": "100"})
    assert (s.upstream_base_url, s.upstream_timeout_s, s.max_request_bytes) == (
        "http://x/v1", 5.0, 100)


def test_settings_from_env_defaults():
    assert Settings.from_env({}) == Settings()


def test_ner_settings_from_env():
    s = Settings.from_env({"LEAKGUARD_NER": "off", "LEAKGUARD_NER_MODEL_DIR": "/m",
                           "LEAKGUARD_NER_THREADS": "4"})
    assert (s.ner_enabled, s.ner_model_dir, s.ner_threads) == (False, "/m", 4)
    assert Settings.from_env({"LEAKGUARD_NER": "ON"}).ner_enabled is True


def test_ner_env_rejects_unclear_values():
    with pytest.raises(ValueError, match="LEAKGUARD_NER"):
        Settings.from_env({"LEAKGUARD_NER": "maybe"})
