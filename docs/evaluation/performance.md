# Performance, running cost and token cost

Generated 2026-10-25 at commit `bdbb8e4` with `python -m leakguard.evaluation perf-report`, from `data/eval/perf.json` (measured with `perf-run`).

## Measured on

- CPU: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz, 8 logical cores
- RAM: 31.7 GB
- OS: Windows 11 (build 26200), Python 3.11.0
- Docker 29.7.2 (its VM: 8 CPUs, 15.5 GB)

A laptop, not a server: absolute numbers will differ elsewhere; the comparisons between rows are what carries over.

## What the guard adds to each request

Client, guard and the mock LLM run as plain processes on the same machine, talking over loopback. Overhead = time through the guard minus time straight to the mock, per message (best of 3 each), over test messages the guard would forward. *First token* is the same difference for the first piece of reply text in a streamed reply.

| Name model | Mode | Overhead p50 | Overhead p95 | First token p50 | First token p95 |
|---|---|---:|---:|---:|---:|
| on | non-streaming | 19.23 ms | 30.37 ms | – | – |
| on | streaming | 19.94 ms | 31.82 ms | 19.70 ms | 31.24 ms |
| off | non-streaming | 1.71 ms | 3.10 ms | – | – |
| off | streaming | 1.57 ms | 2.75 ms | 1.52 ms | 2.24 ms |

## Throughput

The real guard container, capped with Docker's `--cpus`, one uvicorn worker, the name model on one thread; the load client and the mock run in their own containers on the same Docker network. Errors are counted and left out of messages/s. *CPU per message* is the guard container's CPU time divided by successful requests.

| CPUs | Concurrent clients | Messages/s | p50 | p95 | Requests | Errors | CPU per message |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 42.9 | 21.66 ms | 37.03 ms | 1288 | 0 | 21.77 ms |
| 1 | 8 | 24.5 | 301.26 ms | 505.29 ms | 742 | 0 | 40.99 ms |
| 2 | 1 | 41.8 | 22.46 ms | 36.92 ms | 1254 | 0 | 22.58 ms |
| 2 | 8 | 56.5 | 116.66 ms | 212.30 ms | 1697 | 0 | 35.51 ms |

## Running cost

- Image size: 1.03GB on disk, 337 MB compressed (what a pull downloads); exported name model: 283 MB.
- Memory read from the container's cgroup memory.peak. *Fits* = the smallest of 0.5, 1, 2, 4 GB with 25% headroom over the peak.

| CPUs | Memory idle | Memory peak | Fits |
|---:|---:|---:|---|
| 1 | 694 MB | 734 MB | 1 vCPU / 1 GB |
| 2 | 694 MB | 734 MB | 2 vCPU / 1 GB |

## LLM token cost of masking

Counted with tiktoken `o200k_base` (OpenAI's current models and gpt-oss) over 1376 test messages the guard would forward (224 blocked messages left out; 476 had nothing to mask).

- Tokens before masking: 31,139; after: 26,119 (-5,020, -16.1%).
- Per message: -3.65 tokens on average.

Each value against its placeholder, tokenised on their own:

| Type | Values | Value tokens (mean) | Placeholder tokens (mean) |
|---|---:|---:|---:|
| AADHAAR | 97 | 7.0 | 7.0 |
| ADDRESS | 242 | 17.7 | 5.0 |
| BANK_ACCOUNT | 92 | 4.7 | 6.0 |
| CARD | 72 | 9.3 | 5.0 |
| EMAIL | 168 | 7.4 | 5.0 |
| IFSC | 72 | 5.3 | 6.0 |
| NAME | 325 | 4.2 | 5.0 |
| PAN | 172 | 6.0 | 5.0 |
| PHONE | 384 | 5.4 | 5.0 |
| UPI | 44 | 6.7 | 6.0 |

## Notes

- Docker Desktop on Windows runs containers in a VM, so container numbers include its overhead; that is why overhead is measured on plain processes instead.
- The mock LLM answers instantly, so these numbers are the guard's own cost; a real LLM takes hundreds of milliseconds to seconds per reply.
- With more concurrent requests than CPUs, throughput can fall instead of levelling off: the detection threads compete for the container's CPU quota and spend time switching and waiting (compare *CPU per message* across rows). Size the CPUs, or the number of guard replicas, to the expected concurrency.
