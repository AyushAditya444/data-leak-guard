"""Runtime settings, read from environment variables so Docker can configure the guard."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

_ON, _OFF = ("on", "true", "1", "yes"), ("off", "false", "0", "no")


def _switch(env: Mapping[str, str], name: str, default: bool) -> bool:
    """An on/off setting. Anything unclear is an error: a typo must not silently turn a
    protection off."""
    value = env.get(name)
    if value is None:
        return default
    if value.strip().lower() in _ON:
        return True
    if value.strip().lower() in _OFF:
        return False
    raise ValueError(f"{name} must be on or off, not {value!r}")


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible base URL, written the way the OpenAI SDK's `base_url` is:
    # the mock, Ollama (http://localhost:11434/v1), Gemini, Groq, or OpenAI itself.
    upstream_base_url: str = "http://localhost:9000/v1"
    upstream_timeout_s: float = 60.0
    max_request_bytes: int = 1_048_576  # 1 MiB
    policy_path: str | None = None  # None = built-in default policy
    # Person-name detection. On by default: if the model is missing the guard refuses to
    # start rather than silently letting names through. LEAKGUARD_NER=off disables it.
    ner_enabled: bool = True
    ner_model_dir: str = "models/ner"
    ner_threads: int = 1  # CPU threads per inference; requests already run in parallel

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
            policy_path=env.get("LEAKGUARD_POLICY_PATH") or None,
            ner_enabled=_switch(env, "LEAKGUARD_NER", defaults.ner_enabled),
            ner_model_dir=env.get("LEAKGUARD_NER_MODEL_DIR", defaults.ner_model_dir),
            ner_threads=int(env.get("LEAKGUARD_NER_THREADS", defaults.ner_threads)),
        )
