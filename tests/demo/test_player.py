"""The scene player draws the recorded session; it runs on the VHS image's Python."""

import ast
import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest

from examples.demo import player
from examples.demo.scenes import SCREEN_COLS, SCREEN_ROWS
from examples.demo.session import load_session

FIXTURE = Path("tests/demo/fixtures/session-ok.json")
ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


def _visible(line: str) -> str:
    return ANSI.sub("", line)


def _long_session() -> dict:
    session = copy.deepcopy(load_session(FIXTURE))
    session["scenes"]["core"]["app_got"]["text"] = "word " * 600
    session["scenes"]["core"]["wire"][0]["messages"][-1]["content"] = "[NAME_1] " * 300
    session["scenes"]["stream"]["wire"][0]["chunks"] += [[0.4, " more"]] * 400
    return session


@pytest.fixture(params=["ok", "long"])
def session(request) -> dict:
    return load_session(FIXTURE) if request.param == "ok" else _long_session()


def test_box_pads_coloured_text_to_width():
    lines = player.paint("Refund ₹500 to priya@okaxis now", ["priya@okaxis"], 30)
    out = player.box("YOUR APP SENT", lines, 40, player.CYAN, max_lines=5)
    assert len({len(_visible(line)) for line in out}) == 1
    assert len(_visible(out[0])) == 40
    assert "\033[" in "".join(out)


def test_paint_colours_a_value_split_across_lines():
    text = "I live at Flat 302, Sai Krupa Apartments, Kothrud, Pune 411038 now"
    lines = player.paint(text, ["Flat 302, Sai Krupa Apartments, Kothrud, Pune 411038"], 30)
    assert len(lines) > 1
    assert all(player.YELLOW in line for line in lines[:2])
    assert " ".join(_visible(line).strip() for line in lines) == text


def test_paint_marks_chunk_borders():
    text = "call [PHONE_1] now"
    lines = player.paint(text, [], 80, borders=[7, 10])
    assert _visible(lines[0]) == "call [P·HON·E_1] now"


def test_box_truncates_with_pointer():
    lines = [f"line {i}" for i in range(10)]
    out = player.box("T", lines, 60, player.CYAN, max_lines=3)
    assert len(out) == 3 + 2
    assert "session.json" in _visible(out[-2])


def test_every_scene_fits_screen(session):
    for scene_id, scene in session["scenes"].items():
        lines = player.render(scene_id, session, scene["show"]["beats"] - 1,
                              SCREEN_COLS, SCREEN_ROWS)
        assert len(lines) <= SCREEN_ROWS, scene_id
        assert all(len(_visible(line)) <= SCREEN_COLS for line in lines), scene_id


def test_stream_wire_shows_chunk_borders():
    session = load_session(FIXTURE)
    text = "\n".join(_visible(line) for line in player.render("stream", session, 2, 108, 22))
    assert "[PHO·NE_1]" in text


def test_secret_scene_says_nothing_sent():
    session = load_session(FIXTURE)
    first = "\n".join(_visible(line) for line in player.render("secret", session, 0, 108, 22))
    last = "\n".join(_visible(line) for line in player.render("secret", session, 1, 108, 22))
    assert "nothing was sent" not in first
    assert "nothing was sent" in last and "400" in last


def test_results_scene_reads_session_numbers():
    session = copy.deepcopy(load_session(FIXTURE))
    before = "\n".join(player.render("results", session, 1, 108, 22))
    session["results"]["ours"]["f1"] = 0.123
    after = "\n".join(_visible(line) for line in player.render("results", session, 1, 108, 22))
    assert before != after and "12.3%" in after


def test_request_scene_header_says_recorded_live():
    session = copy.deepcopy(load_session(FIXTURE))
    session["meta"] |= {"upstream": "groq", "model": "openai/gpt-oss-120b"}
    header = _visible(player.render("core", session, 0, 108, 22)[0])
    assert "recorded live" in header and "openai/gpt-oss-120b" in header
    assert "2026-10-25" in header
    assert header.endswith("session.json")  # not cut off, even with the longest title


def test_player_imports_stdlib_only():
    tree = ast.parse(Path(player.__file__).read_text(encoding="utf-8"))
    modules = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert modules <= sys.stdlib_module_names, modules - sys.stdlib_module_names


def test_cli_prints_every_scene():
    for scene_id in load_session(FIXTURE)["scenes"]:
        done = subprocess.run([sys.executable, "examples/demo/player.py", scene_id,
                               "--session", str(FIXTURE)], capture_output=True, timeout=30,
                              check=False)
        assert done.returncode == 0, (scene_id, done.stderr)
        assert done.stdout
