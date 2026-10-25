from pathlib import Path

import yaml


def test_guard_published_on_localhost_only():
    # "8080:8080" would make Docker listen on every network interface, so anyone on the
    # same Wi-Fi could use the guard (and whatever keyless upstream it points at).
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    ports = compose["services"]["guard"]["ports"]
    assert ports and all(str(p).startswith("127.0.0.1:") for p in ports)
