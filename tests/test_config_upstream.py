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
