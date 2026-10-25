# Data Leak Guard

A drop-in, OpenAI-compatible proxy that stops Indian personal data and secrets from leaking into LLMs. It sits between your app and the LLM provider, finds sensitive data (Aadhaar, PAN, phone numbers, UPI IDs, bank accounts, IFSC codes, addresses, names, API keys), hides it before the request leaves, and puts the real values back into the reply, so your app works normally. You adopt it by changing **one line**: your SDK's `base_url`.

> **Status: Phase 7a of 7.** Detection (rules, Indian postal addresses, and a person-name model on CPU), masking, blocking and restoring work for both streaming and non-streaming replies. Detection is measured on a held-out evaluation set and compared with Microsoft Presidio, restoring is measured on two real LLMs, and speed and running cost are measured (below). Phase 7b adds a demo video. See the [design](docs/superpowers/specs/2026-10-24-data-leak-guard-design.md).

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
| Postal address | pieces (flat/house no., building, street, landmark, PIN code) + places from India Post's PIN directory and locality endings (Nagar, Colony, Layout…), grouped into **one** span; tuned on dev: [address tuning](docs/evaluation/address-tuning.md) | mask |

**What counts as an address:** at least one specific detail (flat or house number, building, street, landmark like "near Hanuman Mandir", PIN code) **and** at least one place (locality, city, PIN), close together. So `Flat 302, Sai Krupa Apts, Kothrud, Pune 411038` and `near Hanuman Mandir, Kothrud` are masked as one `[ADDRESS_1]` each, while `I live in Pune`, `I live in Kothrud, Pune`, `stuck near Kothrud`, `near the metro station` and `OTP 411038` are not. A 6-digit number counts as a PIN only if it is a real PIN code and is not after words like "OTP" or "order".

**Restoring** works in streamed replies too, even when a placeholder arrives split across chunks (`[PHO` + `NE_1]`): the guard holds back only a tail that could still become a placeholder (at most ~20 characters) and sends everything else at once. It also accepts the small ways LLMs rewrite placeholders (`[Phone_1]`, `[PHONE 1]`, `{PHONE_1}`, a bare `PHONE_1`), in reply text, "reasoning" text and tool-call arguments. It never guesses: anything else is left as written. Placeholders a *user* types, in any of those forms, are never filled with a value.

Logs record **counts only** (`"PHONE:mask": 2`, `"restored": 3, "unrestored": 0`), never values. Placeholder mappings live in memory for one request and are never stored.

## Results

Measured on a **held-out test set** of 1,600 synthetic messages (English, Hinglish, pasted customer records, developer snippets, look-alike negatives and known-hard formats) whose phrasings were never used for tuning. Full report, with methodology, per-category results, latency, known-hard formats and error examples: [docs/evaluation/report.md](docs/evaluation/report.md).

| Type | Support | Precision | Recall | F1 | Coverage | Exact |
|---|---:|---:|---:|---:|---:|---:|
| AADHAAR | 95 | 97.9% | 100.0% | 99.0% | 100.0% | 100.0% |
| PAN | 197 | 100.0% | 87.3% | 93.2% | 85.1% | 100.0% |
| PHONE | 397 | 96.9% | 93.7% | 95.3% | 94.2% | 100.0% |
| UPI | 92 | 100.0% | 47.8% | 64.7% | 56.6% | 100.0% |
| EMAIL | 168 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| CARD | 72 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| IFSC | 72 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| BANK_ACCOUNT | 94 | 100.0% | 97.9% | 98.9% | 100.0% | 100.0% |
| SECRET | 248 | 100.0% | 90.3% | 94.9% | 96.7% | 66.5% |
| NAME | 364 | 97.8% | 87.4% | 92.3% | 88.2% | 95.9% |
| ADDRESS | 271 | 99.6% | 88.9% | 94.0% | 93.7% | 89.2% |
| **All built types** | **2070** | **98.8%** | **90.3%** | **94.4%** | **93.5%** | **93.9%** |

- Negative messages (nothing sensitive) with at least one false alarm: 5 of 350 (1.4%).
- UPI and PAN recall are pulled down by the known-hard formats (unknown UPI handles, spaced PANs); ADDRESS recall by `landmark_only` addresses ("near the metro station", no place), which are not addresses by the detector's definition. A name absorbed into an address is still hidden but counts as a missed NAME. All broken down in the report.
- Detection latency per message, one CPU thread: rules (including addresses) **0.175 ms** (p50); address detector alone **0.116 ms**; rules + name model p50 **17.340 ms**, p95 **41.144 ms**.
- Hand-written set (93 messy real-style messages): **98.9%** precision, **92.0%** recall over built types.
- *Matching rule:* same type and overlapping. *Coverage* = share of the sensitive characters actually hidden from the LLM.
- Reproduce: `python -m leakguard.evaluation report`.

### Restoring replies on real LLMs

300 held-out test conversations sent through the guard to two models on Groq's free tier, each once non-streaming and once streaming, with an instruction to repeat the customer's details back (about a fifth force a tool call). **Fully restored** = the reply mentions at least one of our placeholders and none of ours is left in it. Full report: [docs/evaluation/restore-report.md](docs/evaluation/restore-report.md).

| Model | Mode | Replies mentioning a placeholder | Fully restored | Mentions the model rewrote | Left unrestored |
|---|---|---:|---:|---:|---:|
| openai/gpt-oss-120b | non-streaming | 280 | 100.0% | 12.9% | 0 |
| openai/gpt-oss-120b | streaming | 273 | 100.0% | 11.8% | 0 |
| qwen/qwen3.8-27b | non-streaming | 258 | 99.6% | 6.8% | 1 |
| qwen/qwen3.8-27b | streaming | 259 | 99.6% | 3.5% | 1 |

- The most common rewrite was dropping the brackets (`PHONE_1`): 10% of gpt-oss mentions. Accepting that form was decided on a separate dev run before this test ([restore-tuning.md](docs/evaluation/restore-tuning.md)).
- Restoring costs the guard about 4 ms per streamed reply (~700 characters in 3-character chunks), against about 0.25 ms for plain pass-through.
- 9 of 1,200 requests failed at the provider (Groq rejecting a malformed tool call, or temporarily unavailable) and are not in the rates. Real models change over time, so a re-run gives similar, not identical, numbers.
- Reproduce: `python -m leakguard.evaluation restore-run --split test` (needs a free `GROQ_API_KEY`), then `restore-report --split test`, which renders from the committed results without a key.

### Compared with Presidio

Microsoft Presidio is the best-known open-source PII detector. It was run on the same test set with the same scoring, set up two ways: **out of the box**, and its **best shot** (all its Indian recognizers turned on, a minimum score per type tuned on dev). In presidio-analyzer 2.2.364, Presidio's Indian recognizers are **off by default**. Full page: [docs/evaluation/presidio-baseline.md](docs/evaluation/presidio-baseline.md).

| Type | Ours F1 | Presidio out of the box F1 | Presidio best shot F1 | Ours coverage | Presidio best shot coverage |
|---|---:|---:|---:|---:|---:|
| AADHAAR | 99.0% | 0.0% | 92.6% | 100.0% | 100.0% |
| PAN | 93.2% | 0.0% | 91.8% | 85.1% | 85.1% |
| PHONE | 95.3% | 78.0% | 78.0% | 94.2% | 100.0% |
| EMAIL | 100.0% | 93.9% | 93.9% | 100.0% | 100.0% |
| CARD | 100.0% | 94.7% | 94.7% | 100.0% | 100.0% |
| NAME | 92.3% | 54.2% | 54.2% | 88.2% | 83.0% |
| ADDRESS † | 94.0% | 48.5% | 48.5% | 93.7% | 39.9% |
| UPI / IFSC / BANK_ACCOUNT / SECRET | 64.7–100.0% | not supported | not supported | 56.6–100.0% | 4.8–87.1% |
| **Types both support** (first six rows) | **95.3%** | **62.6%** | **74.5%** | **93.9%** | **94.3%** |

- On the six types both detect, Presidio's best shot hides about as many characters as ours (94.3% against 93.9% coverage), but it gets there by over-masking. 116 of 350 messages with nothing sensitive in them get a false alarm (ours: 5), and only 40.3% of what it calls a name is one. Over-masking hides words the LLM needs to answer well.
- Presidio is faster: 7.4 ms per message at p50 against our 17.7 ms, most of which is our name model.
- Presidio has no detector for UPI IDs, IFSC codes, Indian bank accounts or secrets. It still hides some of them by accident (87.1% of bank-account characters, through its US bank-number detector), which the coverage column shows.
- † Presidio's `LOCATION` finds place names, not full addresses, so its ADDRESS F1 isn't like for like; coverage is the fairer measure.

### Speed and cost

Measured on a laptop (Intel Core i7-1165G7, 8 logical cores, Windows 11 build 26200, Docker Desktop), against the mock LLM so only the guard's own cost shows. Full page, with method: [docs/evaluation/performance.md](docs/evaluation/performance.md).

| What | Name model on | Name model off |
|---|---:|---:|
| Added to each request, p50 / p95 | 19.2 / 30.4 ms | 1.7 / 3.1 ms |
| Added before the first streamed token, p50 / p95 | 19.7 / 31.2 ms | 1.5 / 2.2 ms |

- **Throughput** (the real container, CPU-capped, name model on): 42.9 messages/s on 1 CPU with one client at a time, and 56.5 messages/s on 2 CPUs with 8 concurrent clients. With 8 clients on 1 CPU, throughput *drops* to 24.5/s, because the detection threads fight over one CPU. Size CPUs (or guard replicas) to the concurrency you expect.
- **Memory:** 694 MB idle, 734 MB peak, so it fits a **1 vCPU / 1 GB** machine (with 25% headroom). The image is 1.03 GB on disk (337 MB compressed), 283 MB of it the name model.
- **LLM tokens:** masking **saves** tokens. Over 1,376 test messages the guard would forward, prompts went from 31,139 to 26,119 tokens (−16.1%, `o200k_base`), mostly because a long address becomes a short `[ADDRESS_1]`.
- A real LLM takes hundreds of milliseconds to seconds per reply, so the ~20 ms the name model adds is small next to it.

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
| `LEAKGUARD_API_KEY` | off | if set (16+ characters), every `/v1/` request must send it in an `X-Leakguard-Key` header, otherwise 401 |

### Exposing the guard beyond localhost

`docker compose up` publishes the guard on `127.0.0.1:8080` only, so other machines on your network can't use it. To serve other machines on purpose, change the port line in [`docker-compose.yml`](docker-compose.yml) **and** set a guard key, so the guard can't be used as an open relay to your LLM:

```python
client = OpenAI(base_url="http://guard-host:8080/v1", api_key="<your provider key>",
                default_headers={"X-Leakguard-Key": "<your guard key>"})
```

### Policy

Choose per type: `mask` (hide + restore), `block` (reject the request) or `log_only` (send, but count it). Types you leave out keep their defaults:

```yaml
actions:
  NAME: log_only
failure_mode: closed   # if detection itself fails: closed = reject, open = forward unmasked
```

## Security of the guard itself

A guard that leaks is worse than no guard, so the guard is treated as an attack surface too.

| Threat | Who | What the guard does | Proof |
|---|---|---|---|
| Stolen placeholder mappings (`[PHONE_1]` → real number) | anyone with access to the guard's storage | Nothing to steal: mappings live in memory for one request and are never written anywhere. The guard has no database and no cache | [`masking.py`](src/leakguard/masking.py), [`app.py`](src/leakguard/app.py) (one `PlaceholderMap` per request) |
| Sensitive data or keys in the guard's own logs | anyone who can read logs (ops, a log vendor) | The log functions only accept named metadata fields (counts, timings), never text or headers | `test_logs_counts_but_never_values`, `test_logs_contain_no_content_or_auth`, `test_guard_key_never_logged` |
| Placeholder injection: a user types `[PHONE_1]` hoping the reply gets someone else's value | a malicious user of the app | Every placeholder-like form a user types (`[PHONE_1]`, `phone 1`, `PHONE_1`, …) is reserved and never filled with a value | `test_user_typed_placeholder_is_never_restored`, `test_reserved_never_restored_in_any_form`, `test_near_forms_a_user_types_are_reserved` |
| ReDoS: a message crafted to make a regex take forever | a malicious user | Patterns avoid catastrophic backtracking; long and adversarial inputs are tested for speed | `test_detect_rules_is_fast_on_adversarial_input`, `test_long_repetitive_messages_stay_fast`, `test_reserving_many_typed_placeholders_is_fast` |
| Open relay: strangers use the guard to reach the LLM behind it (e.g. a keyless local Ollama) | anyone on the same network | Docker publishes the guard on `127.0.0.1` only. When exposing it on purpose, `LEAKGUARD_API_KEY` requires an `X-Leakguard-Key` header on every `/v1/` request | `test_guard_published_on_localhost_only`, `test_guard_key_missing_or_wrong_is_401` |
| Detection crashes on some input, and the message goes out unchecked | a bug, or input that triggers one | Fail-closed by default: the request is rejected (500), not forwarded. `failure_mode: open` exists and is documented as unsafe | `test_detection_failure_fails_closed` |
| Huge requests to exhaust memory or CPU | a malicious user | Bodies over `LEAKGUARD_MAX_REQUEST_BYTES` (1 MiB) are rejected with 413 before parsing | `test_request_too_large_returns_413` |
| Attacks on the guard key itself | anyone who can reach the guard | Compared in constant time (no timing leak), checked before the body is read, never forwarded upstream, never logged; a key under 16 characters stops the guard at startup | `test_guard_key_checked_before_body`, `test_guard_key_not_forwarded`, `test_short_key_refuses_start` |

### What it does not protect against

- **Data inside images or attached files.** Only text fields (messages, tool calls) are inspected.
- **Scripts it was not measured on.** Names and addresses in Devanagari or other non-Latin scripts were not in the evaluation, so there is no number for them.
- **Who someone is, worked out from context.** "The CFO of a Pune fintech who joined in March" contains nothing the guard masks.
- **The known-hard formats**, such as spaced PANs, numbers written as words and unknown UPI handles. UPI recall is 47.8% and PAN recall 87.3% on the test set, mostly for these reasons ([report](docs/evaluation/report.md#known-hard-formats-test-split)).
- **Traffic that doesn't go through the guard.** It protects what your app sends through `base_url`, nothing else.
- **What the LLM provider sees around the placeholders.** The provider still gets the rest of the message, and it can see that a phone number was there.

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
- [Phase 6 walkthrough](docs/walkthrough/phase-6-streaming-restore.md): restoring streamed replies
- [Phase 7a walkthrough](docs/walkthrough/phase-7a-measure-and-harden.md): Presidio comparison, speed and cost, hardening
- [Evaluation report](docs/evaluation/report.md), [Presidio baseline](docs/evaluation/presidio-baseline.md), [performance and cost](docs/evaluation/performance.md), [restore report](docs/evaluation/restore-report.md)

## License

MIT. Contains data from the All India Pincode Directory, Department of Posts, Government
of India, published on data.gov.in under the Government Open Data License – India.
