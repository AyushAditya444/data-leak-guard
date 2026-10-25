"""The parts of the demo build that run before Docker does."""

import shutil
from pathlib import Path

import pytest

from examples.demo import build

FIXTURE = Path("tests/demo/fixtures/session-ok.json")  # a mock session


def test_committed_video_needs_a_live_groq_session(tmp_path, monkeypatch, capsys):
    # The narration says "recorded live against a real model on Groq"; a mock session must
    # not become the committed video. The check runs before any Docker step.
    committed = tmp_path / "session.json"
    shutil.copy(FIXTURE, committed)
    monkeypatch.setattr(build, "DEFAULT_SESSION", committed)
    monkeypatch.setattr(build, "narrate", lambda: pytest.fail("Docker step reached"))
    assert build.main(["--session", str(committed)]) == 1
    assert "Groq" in capsys.readouterr().err


def test_own_voice_outside_the_repo_is_copied_in(tmp_path):
    voice = tmp_path / "my recordings"
    voice.mkdir()
    (voice / "core.wav").write_bytes(b"RIFF")
    (voice / "notes.wav").write_bytes(b"RIFF")
    dest = tmp_path / "copied"
    found, ignored = build.collect_own_voice(voice, ["title", "core"], dest)
    assert found == {"core": dest / "core.wav"}
    assert (dest / "core.wav").read_bytes() == b"RIFF"
    assert ignored == ["notes.wav"]


def test_own_voice_without_matching_files_is_an_error(tmp_path):
    (tmp_path / "take1.wav").write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="title.wav"):
        build.collect_own_voice(tmp_path, ["title", "core"], tmp_path / "copied")
