"""Secret detection: API keys, tokens, passwords, private keys.

Three complementary methods:
1. Known prefixes — many providers stamp their keys (sk-, AKIA, ghp_, ...).
2. Key/value context — "password=...", "api_key: ...", "Bearer ...".
3. Entropy — long tokens that look random are probably secrets, even in unknown formats.

Secrets are *blocked* by default, so a false positive rejects the user's whole message.
The rules therefore lean towards precision: ordinary sentences about passwords must
never be flagged.
"""

import math
import re
from collections import Counter

from leakguard.detect.spans import Span, merge_spans
from leakguard.entities import EntityType

_START = r"(?<![A-Za-z0-9_-])"  # a match may only begin at the start of a token

_PREFIXED = [re.compile(_START + p) for p in (
    r"sk-[A-Za-z0-9_-]{20,}",  # OpenAI (sk-proj-..., sk-svcacct-...), Anthropic (sk-ant-...)
    r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])",  # AWS access key id
    r"gh[pousr]_[A-Za-z0-9]{36,}",  # GitHub tokens
    r"github_pat_[A-Za-z0-9_]{22,}",  # GitHub fine-grained token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack
    r"AIza[0-9A-Za-z_-]{35}",  # Google API key
    r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}",  # Stripe
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
)] + [re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")]  # PEM private key header

_KEY_VALUE = re.compile(
    # an optional WORD_ / WORD- prefix covers DB_PASSWORD, db_password, APP_SECRET, MY_API_KEY
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9]+[_-])*"
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?token"
    r"|auth[_-]?token|client[_-]?secret)\s*[:=]\s*[\"']?([^\s\"',;]+)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+([A-Za-z0-9._~+/-]{8,}=*)", re.IGNORECASE)

_TOKEN = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_=-]{24,}")
_MIN_ENTROPY = 4.0  # bits/char; hex (16 symbols) can't exceed 4.0, so git SHAs/UUIDs pass
_WORD = re.compile(r"[A-Z]?[a-z]{3,}|[A-Z]{4,}")
_MAX_WORD_FRACTION = 0.75  # above this, a token reads like a code identifier (Phase 3 tunes it)

_TRAILING_PUNCTUATION = ".,;:!?"


def shannon_entropy(s: str) -> float:
    """Average information per character: H = -sum(p * log2(p)) over character frequencies."""
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _looks_like_credential(value: str) -> bool:
    # "reset password: how?" -> "how": too short. "password: please" -> letters only.
    return len(value) >= 4 and any(not ch.isalpha() for ch in value)


def word_fraction(token: str) -> float:
    """Share of the token made of word-like chunks ("Invoice", "line", "HTTP").
    Code identifiers are mostly words (getUserAccountBalance2024Handler: 0.88);
    random keys almost never are (median 0.25, 99.9th percentile 0.79 over 9,000 samples)."""
    return sum(len(m.group()) for m in _WORD.finditer(token)) / len(token)


def _looks_random(token: str) -> bool:
    has_classes = (any(c.islower() for c in token) and any(c.isupper() for c in token)
                   and any(c.isdigit() for c in token))
    return (has_classes and shannon_entropy(token) > _MIN_ENTROPY
            and word_fraction(token) < _MAX_WORD_FRACTION)


def _secret(start: int, end: int, detector: str) -> Span:
    return Span(start, end, EntityType.SECRET, detector=detector)


def detect_secrets(text: str) -> list[Span]:
    found: list[Span] = []
    for pattern in _PREFIXED:
        found += [_secret(m.start(), m.end(), "secret_prefix") for m in pattern.finditer(text)]

    for m in _KEY_VALUE.finditer(text):
        value = m.group(1).rstrip(_TRAILING_PUNCTUATION)
        if _looks_like_credential(value):
            found.append(_secret(m.start(1), m.start(1) + len(value), "secret_key_value"))

    found += [_secret(m.start(1), m.end(1), "secret_bearer") for m in _BEARER.finditer(text)]

    found += [_secret(m.start(), m.end(), "secret_entropy")
              for m in _TOKEN.finditer(text) if _looks_random(m.group())]

    return merge_spans(found)  # one key can match several rules; report it once
