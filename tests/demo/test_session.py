"""verify_session checks every claim the narration makes against a recorded session."""

import copy
import json
from pathlib import Path

import pytest

from examples.demo.session import load_session, save_session, verify_session

FIXTURE = Path("tests/demo/fixtures/session-ok.json")


@pytest.fixture
def session() -> dict:
    return copy.deepcopy(load_session(FIXTURE))


def _problem(session: dict, scene_id: str) -> str:
    problems = verify_session(session)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"{scene_id}:"), problems
    return problems[0]


def test_verify_ok_fixture_passes(session):
    assert verify_session(session) == []


def test_verify_value_on_wire(session):
    session["scenes"]["core"]["wire"][0]["messages"][1]["content"] += " 9876543210"
    assert "left the machine" in _problem(session, "core")


def test_verify_nothing_restored(session):
    session["scenes"]["core"]["app_got"]["text"] = "Thanks, noted."
    assert "restored" in _problem(session, "core")


def test_verify_placeholder_left_in_reply(session):
    session["scenes"]["core"]["app_got"]["text"] += " [PHONE_1]"
    assert "placeholder" in _problem(session, "core")


def test_verify_log_contains_value(session):
    session["scenes"]["core"]["log"]["note"] = "Priya Sharma"
    assert "log" in _problem(session, "core")


def test_verify_log_without_findings(session):
    session["scenes"]["core"]["log"]["findings"] = {}
    assert "log" in _problem(session, "core")


def test_verify_stream_without_split_placeholder(session):
    session["scenes"]["stream"]["wire"][0]["chunks"] = [[0.2, "I'll call "], [0.3, "[PHONE_1]"],
                                                        [0.35, " soon."]]
    assert "split" in _problem(session, "stream")


def test_verify_blocked_but_sent(session):
    session["scenes"]["secret"]["wire"] = [{"messages": [], "status": 200, "chunks": []}]
    assert "sent" in _problem(session, "secret")


def test_verify_blocked_wrong_status(session):
    session["scenes"]["secret"]["app_got"]["status"] = 200
    assert "400" in _problem(session, "secret")


def test_verify_lookalike_changed(session):
    session["scenes"]["lookalike"]["wire"][0]["messages"] = [
        {"role": "user", "content": "My OTP is [ADDRESS_1]."}]
    assert "unchanged" in _problem(session, "lookalike")


def test_verify_results_claim_false(session):
    session["results"]["presidio_best_shot"]["f1"] = 0.99
    assert "F1" in _problem(session, "results")


def test_verify_results_false_alarms_claim(session):
    session["results"]["ours"]["false_alarms"] = 100
    assert "false alarms" in _problem(session, "results")


def test_save_load_round_trip_keeps_unicode(session, tmp_path):
    session["scenes"]["core"]["app_got"]["text"] = "₹500 refund, मेरा नाम Priya Sharma"
    path = tmp_path / "s.json"
    save_session(session, path)
    assert load_session(path) == session
    raw = path.read_text(encoding="utf-8")
    assert "₹500" in raw and raw.endswith("\n")
    assert json.loads(raw) == session


def test_verify_missing_request_scene(session):
    del session["scenes"]["secret"]
    assert "blocked" in _problem(session, "session")


def test_committed_session_passes_verification():
    # The session the video replays (recorded against a real model) must keep every claim.
    assert verify_session(load_session(Path("examples/demo/session.json"))) == []
