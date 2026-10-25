# Data Leak Guard

A drop-in, OpenAI-compatible proxy that stops Indian personal data and secrets from leaking into LLMs. It sits between your app and the LLM provider, finds sensitive data (Aadhaar, PAN, phone numbers, UPI IDs, bank accounts, IFSC codes, addresses, names, API keys), hides it before the request leaves, and puts the real values back into the reply, so your app works normally. You adopt it by changing **one line**: your SDK's `base_url`.

> **Status: Phase 5 of 7.** Detection (rules, Indian postal addresses, and a person-name model on CPU), masking, blocking and restoring work for non-streaming replies, and are measured on a held-out evaluation set (below). Streamed replies are masked (nothing leaks) but still show placeholders until Phase 6. See the [design](docs/superpowers/specs/2026-10-24-data-leak-guard-design.md).

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
| Person name | NER model ([`xlm-roberta-base-ner-hrl`](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl), ONNX int8 on CPU, chosen from 5 candidates on dev: [comparison](docs/evaluation/ner-comparison.md)) + cue phrases for lowercase/all-caps names (`mera naam … hai`, `Name: …`) | mask |
| Postal address | pieces (flat/house no., building, street, landmark, PIN code) + places from India Post's PIN directory and the name model's location tags, grouped into **one** span | mask |

**What counts as an address:** at least one specific detail (flat or house number, building, street, landmark like "near Hanuman Mandir", PIN code) **and** at least one place (locality, city, PIN), close together. So `Flat 302, Sai Krupa Apts, Kothrud, Pune 411038` and `near Hanuman Mandir, Kothrud` are masked as one `[ADDRESS_1]` each, while `I live in Pune`, `I live in Kothrud, Pune`, `stuck near Kothrud`, `near the metro station` and `OTP 411038` are not. A 6-digit number counts as a PIN only if it is a real PIN code and is not after words like "OTP" or "order".

Logs record **counts only** (`"PHONE:mask": 2`), never values. Placeholder mappings live in memory for one request and are never stored.

## Results

Measured on a **held-out test set** of 1,600 synthetic messages (English, Hinglish, pasted customer records, developer snippets, look-alike negatives and known-hard formats) whose phrasings were never used for tuning. Full report, with methodology, per-category results, latency, known-hard formats and error examples: [docs/evaluation/report.md](docs/evaluation/report.md).

| Type | Support | Precision | Recall | F1 | Coverage | Exact |
|---|---:|---:|---:|---:|---:|---:|
| AADHAAR | 95 | 97.9% | 100.0% | 99.0% | 100.0% | 100.0% |
| PAN | 197 | 100.0% | 87.3% | 93.2% | 85.1% | 100.0% |
| PHONE | 397 | 97.4% | 93.7% | 95.5% | 94.2% | 100.0% |
| UPI | 92 | 100.0% | 47.8% | 64.7% | 55.4% | 100.0% |
| EMAIL | 168 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| CARD | 72 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 72 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| BANK_ACCOUNT | 94 | 100.0% | 97.9% | 98.9% | 100.0% | 100.0% |
| SECRET | 248 | 100.0% | 90.3% | 94.9% | 96.7% | 66.1% |
| NAME | 364 | 97.6% | 89.3% | 93.3% | 89.8% | 95.4% |
| ADDRESS | 271 | 99.6% | 89.7% | 94.4% | 94.3% | 81.9% |
| **All built types** | **2070** | **98.9%** | **90.8%** | **94.7%** | **93.8%** | **92.8%** |

- Negative messages (nothing sensitive) with at least one false alarm: 6 of 350 (1.7%).
- UPI and PAN recall are pulled down by the known-hard formats (unknown UPI handles, spaced PANs); ADDRESS recall by `landmark_only` addresses ("near the metro station", no place), which are not addresses by the detector's definition. Both are broken down in the report.
- Detection latency per message, one CPU thread: rules (including addresses) **0.167 ms** (p50); address detector alone **0.107 ms**; rules + name model p50 **17.003 ms**, p95 **40.905 ms**.
- Hand-written set (93 messy real-style messages): **97.9%** precision, **93.0%** recall over built types.
- *Matching rule:* same type and overlapping. *Coverage* = share of the sensitive characters actually hidden from the LLM.
- Reproduce: `python -m leakguard.evaluation report`.

## Quickstart (free, no API key needed)

Requires Docker.

```bash
docker compose up --build
```

This starts the guard on `http://localhost:8080` and a free **mock LLM** behind it that replies `"You said: <your message>"`.

The first build downloads CPU-only PyTorch and the name model's weights (~1.3 GB) to export the model inside a throwaway build stage, so it takes a while; later builds reuse Docker's cache. The image that runs contains only the exported ONNX model, not PyTorch.

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
| `LEAKGUARD_NER` | `on` | person-name detection (an ONNX model on CPU). `off` disables it; while `on`, the guard refuses to start if the model is missing |
| `LEAKGUARD_NER_MODEL_DIR` | `models/ner` | exported name model (the Docker image includes it) |
| `LEAKGUARD_NER_THREADS` | `1` | CPU threads per name-model inference |

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
pytest                            # model-backed tests skip until the model is exported
ruff check .
```

The name model is exported from a pinned Hugging Face revision (needs PyTorch once; the
guard itself only needs ONNX Runtime):

```bash
pip install "torch>=2.14,<3" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[export]"
python -m leakguard.ner.export           # -> models/ner/ (what the guard loads)
python -m leakguard.evaluation report    # regenerate the evaluation report
```

To redo the model choice: `python -m leakguard.ner.export --all` then
`python -m leakguard.ner.compare` (dev split only; writes
[docs/evaluation/ner-comparison.md](docs/evaluation/ner-comparison.md)).

To redo the address tuning: `python -m leakguard.evaluation.address_tuning` (dev split
only; writes [docs/evaluation/address-tuning.md](docs/evaluation/address-tuning.md)). The
bundled PIN directory is rebuilt from India Post's CSV with
`python -m leakguard.detect.gazetteer build --csv data/raw/<file>.csv` (source and licence:
[src/leakguard/data/README.md](src/leakguard/data/README.md)).

## Docs

- [Design](docs/superpowers/specs/2026-10-24-data-leak-guard-design.md)
- [Phase 1 walkthrough](docs/walkthrough/phase-1-skeleton.md): the proxy skeleton
- [Phase 2 walkthrough](docs/walkthrough/phase-2-detection.md): detection, policy, masking
- [Phase 3 walkthrough](docs/walkthrough/phase-3-evaluation.md): evaluation
- [Phase 4 walkthrough](docs/walkthrough/phase-4-names.md): person names (NER)
- [Phase 5 walkthrough](docs/walkthrough/phase-5-addresses.md): Indian postal addresses
- [Evaluation report](docs/evaluation/report.md)

## License

MIT. Contains data from the All India Pincode Directory, Department of Posts, Government
of India, published on data.gov.in under the Government Open Data License – India.
