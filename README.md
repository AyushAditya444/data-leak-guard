# Data Leak Guard

A drop-in, OpenAI-compatible proxy that stops Indian personal data and secrets from leaking into LLMs. It sits between your app and the LLM provider, finds sensitive data (Aadhaar, PAN, phone numbers, UPI IDs, bank accounts, IFSC codes, addresses, names, API keys), hides it before the request leaves, and puts the real values back into the reply, so your app works normally. You adopt it by changing **one line**: your SDK's `base_url`.

> **Status: Phase 2 of 7.** Rule-based detection, masking, blocking and restoring work for non-streaming replies. Streamed replies are masked (nothing leaks) but still show placeholders until Phase 6. Person names (Phase 4) and addresses (Phase 5) are not detected yet. See the [design](docs/superpowers/specs/2026-10-24-data-leak-guard-design.md).

## What it does

```
Your app sends:   "My Aadhaar is 2345 6789 0124, refund to rahul.s@okicici"
The LLM sees:     "My Aadhaar is [AADHAAR_1], refund to [UPI_1]"
The LLM replies:  "Refund to [UPI_1] is on its way."
Your app gets:    "Refund to rahul.s@okicici is on its way."
```

| Detected | How | Default |
|---|---|---|
| Aadhaar | 12 digits + Verhoeff checksum | mask |
| PAN | `ABCPS1234K` structure + holder-type letter | mask |
| Indian mobile | 10 digits starting 6–9, `+91`/`0` prefixes | mask |
| UPI ID | `name@handle` with a known UPI handle | mask |
| Email | standard format | mask |
| Card | 13–19 digits + Luhn checksum | mask |
| IFSC | `HDFC0001234` structure | mask |
| Bank account | 9–18 digits **with** account context ("a/c", "account", IFSC nearby) | mask |
| Secrets | known key prefixes, `password=…`, `Bearer …`, high-entropy tokens | **block** (HTTP 400) |

Logs record **counts only** (`"PHONE:mask": 2`), never values. Placeholder mappings live in memory for one request and are never stored.

## Quickstart (free, no API key needed)

Requires Docker.

```bash
docker compose up --build
```

This starts the guard on `http://localhost:8080` and a free **mock LLM** behind it that replies `"You said: <your message>"`.

In another terminal:

```bash
pip install openai
python examples/chat.py
```

## Use it from your code

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="<your provider key>")  # only change
client.chat.completions.create(model="...", messages=[...])
```

The guard passes your `Authorization` header through to the provider. It holds no keys itself and never logs message content or keys.

## Point it at a real LLM

Set `LEAKGUARD_UPSTREAM_BASE_URL` to any OpenAI-compatible base URL:

| Provider | `LEAKGUARD_UPSTREAM_BASE_URL` | Cost |
|---|---|---|
| Mock (default) | `http://mock-llm:9000/v1` | free |
| Ollama on your machine | `http://host.docker.internal:11434/v1` | free |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | free tier |
| Groq | `https://api.groq.com/openai/v1` | free tier |
| OpenAI | `https://api.openai.com/v1` | paid |

```bash
LEAKGUARD_UPSTREAM_BASE_URL=http://host.docker.internal:11434/v1 docker compose up --build
```

| Setting | Default | Meaning |
|---|---|---|
| `LEAKGUARD_UPSTREAM_BASE_URL` | `http://localhost:9000/v1` | where requests are forwarded |
| `LEAKGUARD_UPSTREAM_TIMEOUT_S` | `60` | upstream timeout in seconds |
| `LEAKGUARD_MAX_REQUEST_BYTES` | `1048576` | larger requests are rejected with 413 |
| `LEAKGUARD_POLICY_PATH` | built-in default | YAML policy file, see [`config/policy.yaml`](config/policy.yaml) |

### Policy

Choose per type: `mask` (hide + restore), `block` (reject the request) or `log_only` (send, but count it). Types you leave out keep their defaults:

```yaml
actions:
  NAME: log_only
failure_mode: closed   # if detection itself fails: closed = reject, open = forward unmasked
```

## Development

```bash
python -m venv .venv              # any location works
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
pytest
ruff check .
```

## Docs

- [Design](docs/superpowers/specs/2026-10-24-data-leak-guard-design.md)
- [Phase 1 walkthrough](docs/walkthrough/phase-1-skeleton.md): the proxy skeleton
- [Phase 2 walkthrough](docs/walkthrough/phase-2-detection.md): detection, policy, masking

## License

MIT
