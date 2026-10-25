"""The optional guard key: a shared secret callers must send before the guard does anything.

Off by default (the guard is meant to sit on localhost or a private network). Turn it on
with LEAKGUARD_API_KEY when the guard is reachable by others, so it can't be used as an
open relay to whatever upstream it points at (e.g. a keyless local Ollama).
"""

import hmac

GUARD_KEY_HEADER = "x-leakguard-key"


def guard_key_ok(given: str | None, expected: str) -> bool:
    """Constant-time comparison: how long it takes doesn't depend on how many leading
    characters were right, so the key can't be guessed one character at a time."""
    if given is None:
        return False
    # Bytes, not str: compare_digest rejects non-ASCII str, and a header value may hold any
    # byte (Starlette decodes headers as latin-1).
    return hmac.compare_digest(given.encode("utf-8", "surrogateescape"),
                               expected.encode("utf-8"))
