"""Runs the performance measurements (python -m leakguard.evaluation perf-run).

Two setups, because each answers a different question honestly:
- overhead and time to first token: guard and mock as plain processes on loopback, since
  Docker Desktop's port forwarding on Windows costs more than the rules take;
- throughput, memory and CPU: the real container, capped with --cpus, so "fits a 1 vCPU
  machine" is measured rather than guessed.
Needs Docker, the exported name model, and the [perf] extra (tiktoken, psutil).
"""

import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence

import httpx

from leakguard.evaluation.loading import load_detector, load_split
from leakguard.evaluation.perf import forwardable_texts, machine_info, measure_overhead
from leakguard.evaluation.restore_run import _git_commit
from leakguard.evaluation.token_cost import ENCODING, o200k_counter, token_cost
from leakguard.policy import DEFAULT_POLICY

OVERHEAD_MESSAGES = 300  # evenly spread over the test split: enough for a stable p95
COMPOSE = ["docker", "compose", "-f", "docker-compose.perf.yml", "-p", "leakguard-perf"]
IMAGE = "leakguard-perf:latest"
CGROUP = "/sys/fs/cgroup"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(url: str, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            httpx.post(url, json={"messages": []}, timeout=2)  # any HTTP reply means it's up
            return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not come up within {timeout_s} s")


def _uvicorn(factory: str, port: int, env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", factory, "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def overhead_texts() -> list[str]:
    texts = forwardable_texts(load_split("test"))
    return texts[::max(1, len(texts) // OVERHEAD_MESSAGES)][:OVERHEAD_MESSAGES]


def native_overhead(ner: bool, texts: Sequence[str]) -> dict:
    mock_port, guard_port = _free_port(), _free_port()
    env = {**os.environ, "LEAKGUARD_UPSTREAM_BASE_URL": f"http://127.0.0.1:{mock_port}/v1",
           "LEAKGUARD_NER": "on" if ner else "off"}
    procs = [_uvicorn("leakguard.mock_llm:create_mock_app", mock_port, env),
             _uvicorn("leakguard.app:build_app", guard_port, env)]
    try:
        for port in (mock_port, guard_port):
            _wait_until_up(f"http://127.0.0.1:{port}/v1/chat/completions")
        with httpx.Client(base_url=f"http://127.0.0.1:{mock_port}", timeout=60) as direct, \
                httpx.Client(base_url=f"http://127.0.0.1:{guard_port}", timeout=60) as guard:
            return {"non_streaming": measure_overhead(direct, guard, texts, stream=False),
                    "streaming": measure_overhead(direct, guard, texts, stream=True)}
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=30)


def _compose(*args: str, env: dict | None = None, check: bool = True) -> str:
    return subprocess.run([*COMPOSE, *args], capture_output=True, text=True, check=check,
                          env=env, encoding="utf-8", errors="replace").stdout


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _guard_read(path: str) -> str:
    return _compose("exec", "-T", "guard", "cat", path).strip()


def _cpu_usec() -> int:
    stats = dict(line.split() for line in _guard_read(f"{CGROUP}/cpu.stat").splitlines())
    return int(stats["usage_usec"])


def _memory_mb() -> float:
    return int(_guard_read(f"{CGROUP}/memory.current")) / 2**20


def _peak_mb(polled: list[float]) -> tuple[float, str]:
    try:
        return int(_guard_read(f"{CGROUP}/memory.peak")) / 2**20, "cgroup memory.peak"
    except subprocess.CalledProcessError:
        return max(polled), "cgroup memory.current, polled every second"


def _wait_for_guard(timeout_s: float = 180) -> None:
    probe = ("import urllib.request; "
             "urllib.request.urlopen('http://localhost:8080/healthz', timeout=2)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if subprocess.run([*COMPOSE, "exec", "-T", "guard", "python", "-c", probe],
                          capture_output=True, check=False).returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError("the guard container did not become healthy")


def _one_cap(cpus: int, concurrency: Sequence[int], seconds: float) -> dict:
    env = {**os.environ, "PERF_CPUS": str(cpus)}
    _compose("up", "-d", "--force-recreate", "guard", "mock", env=env)
    _wait_for_guard()
    time.sleep(10)  # let start-up settle before reading idle memory
    run = {"cpus": cpus, "idle_mb": _memory_mb(), "loads": []}
    polled = [run["idle_mb"]]
    stop = threading.Event()

    def poll() -> None:  # only used if the kernel has no memory.peak
        while not stop.wait(1):
            try:
                polled.append(_memory_mb())
            except subprocess.CalledProcessError:
                pass

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        for n in concurrency:
            before = _cpu_usec()
            out = _compose("run", "--rm", "client", "load", "--url", "http://guard:8080",
                           "--concurrency", str(n), "--seconds", str(seconds), env=env)
            load = json.loads(out.strip().splitlines()[-1])
            load["cpu_usec"] = _cpu_usec() - before
            run["loads"].append(load)
    finally:
        stop.set()
        poller.join()
    run["peak_mb"], run["memory_source"] = _peak_mb(polled)
    return run


def container_runs(cpus: Sequence[int] = (1, 2), concurrency: Sequence[int] = (1, 8),
                   seconds: float = 30) -> dict:
    _compose("build", "guard")
    try:
        runs = [_one_cap(n, concurrency, seconds) for n in cpus]
        model_bytes = int(_compose("exec", "-T", "guard", "du", "-sb",
                                   "/app/models/ner").split()[0])
    finally:
        _compose("down", "--remove-orphans", check=False)
    # With Docker's containerd image store, `inspect` gives the compressed size (what a pull
    # downloads) and `image ls` the unpacked size on disk: report both.
    image_bytes = int(_docker("image", "inspect", IMAGE, "--format", "{{.Size}}"))
    image_disk = _docker("image", "ls", IMAGE, "--format", "{{.Size}}")
    return {"image_mb": image_bytes / 2**20, "image_disk": image_disk,
            "model_mb": model_bytes / 2**20,
            "memory_source": runs[0].pop("memory_source"),
            "runs": [{k: v for k, v in r.items() if k != "memory_source"} for r in runs]}


def run() -> dict:
    detect, _, _ = load_detector(require_names=True)
    texts = overhead_texts()
    print(f"overhead on {len(texts)} messages, name model on and off", file=sys.stderr)
    overhead = {"ner_on": native_overhead(True, texts), "ner_off": native_overhead(False, texts)}
    print("throughput and memory in the capped container", file=sys.stderr)
    containers = container_runs()
    tokens = token_cost(load_split("test"), detect, DEFAULT_POLICY, o200k_counter())
    return {"meta": {"date": datetime.date.today().isoformat(), "guard_commit": _git_commit(),
                     "machine": machine_info()},
            "overhead": overhead, "containers": containers, "tokens": tokens,
            "encoding": ENCODING}
