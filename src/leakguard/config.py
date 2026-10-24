"""Runtime settings, read from environment variables so Docker can configure the guard."""

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible base URL, written the way the OpenAI SDK's `base_url` is:
    # the mock, Ollama (http://localhost:11434/v1), Gemini, Groq, or OpenAI itself.
    upstream_base_url: str = "http://localhost:9000/v1"
    upstream_timeout_s: float = 60.0
    max_request_bytes: int = 1_048_576  # 1 MiB

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            upstream_base_url=env.get("LEAKGUARD_UPSTREAM_BASE_URL", defaults.upstream_base_url),
            upstream_timeout_s=float(
                env.get("LEAKGUARD_UPSTREAM_TIMEOUT_S", defaults.upstream_timeout_s)),
            max_request_bytes=int(
                env.get("LEAKGUARD_MAX_REQUEST_BYTES", defaults.max_request_bytes)),
        )
