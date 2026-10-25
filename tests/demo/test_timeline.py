"""The narration script, and turning its audio timings into subtitles and a VHS tape."""

import re
from pathlib import Path

import pytest

from examples.demo.scenes import SCENES
from examples.demo.timeline import (
    HIDDEN_START_S,
    LEAD_IN_S,
    SCENE_PAD_S,
    min_durations_from_session,
    parse_narration,
    scene_plan,
    srt,
    tape,
    voice_overrun,
)

NARRATION = Path("examples/demo/narration.md").read_text(encoding="utf-8")


def test_parse_narration_sections_in_order():
    text = "## a\nFirst line.\nSecond line.\n\n## b\nOnly line.\n"
    assert parse_narration(text) == {"a": ["First line.", "Second line."], "b": ["Only line."]}


def test_parse_narration_ignores_header_and_blank_lines():
    text = "# Title\n\nHow to use this file.\n\n## a\n\n  Spoken.  \n\n"
    assert parse_narration(text) == {"a": ["Spoken."]}


def test_narration_covers_every_scene():
    assert list(parse_narration(NARRATION)) == [s.id for s in SCENES]


def test_narration_has_enough_cues_for_beats():
    cues = parse_narration(NARRATION)
    for s in SCENES:
        assert len(cues[s.id]) >= s.beats, s.id


def test_narration_fits_two_minutes():
    words = sum(len(line.split()) for lines in parse_narration(NARRATION).values()
                for line in lines)
    assert words <= 320


def test_results_narration_has_no_numbers():
    # Numbers on that screen come from the committed results; the voice states none.
    spoken = " ".join(parse_narration(NARRATION)["results"]).replace("F1", "")
    assert not re.search(r"\d", spoken)


# --- timing --------------------------------------------------------------------------

TIMINGS = {
    "a": {"duration": 5.0, "cues": [{"text": "One.", "start": 0.0, "end": 2.0},
                                    {"text": "Two.", "start": 2.35, "end": 5.0}]},
    "b": {"duration": 3.0, "cues": [{"text": "Three " * 20, "start": 0.0, "end": 3.0}]},
}
BEATS = {"a": 2, "b": 1}


def _secs(stamp: str) -> float:
    h, m, rest = stamp.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def test_scene_plan_durations_cover_audio_and_minimums():
    plan = scene_plan(TIMINGS, {"b": 10.0}, BEATS)
    assert [p["id"] for p in plan] == ["a", "b"]
    assert plan[0]["duration"] == pytest.approx(LEAD_IN_S + 5.0 + SCENE_PAD_S)
    assert plan[1]["duration"] == 10.0
    assert plan[0]["beats_at"] == [0.0, round(LEAD_IN_S + 2.35, 3)]


def test_scene_plan_scenes_never_overlap():
    plan = scene_plan(TIMINGS, {}, BEATS)
    assert plan[0]["start"] == 0.0
    assert plan[1]["start"] == pytest.approx(plan[0]["start"] + plan[0]["duration"])


def test_srt_format_and_numbering():
    plan = scene_plan(TIMINGS, {}, BEATS)
    blocks = srt(plan, TIMINGS).strip().split("\n\n")
    assert [b.splitlines()[0] for b in blocks] == ["1", "2", "3"]
    times = [tuple(_secs(t) for t in b.splitlines()[1].split(" --> ")) for b in blocks]
    assert times[0] == pytest.approx((LEAD_IN_S, LEAD_IN_S + 2.0))
    assert all(start < end for start, end in times)
    assert all(times[i][1] <= times[i + 1][0] for i in range(len(times) - 1))
    b_start = plan[1]["start"]
    assert b_start <= times[2][0] and times[2][1] <= b_start + plan[1]["duration"]
    assert all(len(line) <= 70 for b in blocks for line in b.splitlines()[2:])


def test_one_tape_per_scene():
    # One clip per scene: VHS drops a few % of frames, so each clip is stretched back to
    # its planned length before joining, and timing errors cannot add up across scenes.
    plan = scene_plan(TIMINGS, {}, BEATS)
    tapes = tape(plan, "examples/demo/session.json", font_size=18)
    assert list(tapes) == ["a", "b"]
    for scene in plan:
        text = tapes[scene["id"]]
        assert "Set Width 1280" in text and "Set Height 600" in text
        assert "Set FontSize 18" in text
        assert f"Output build/demo/scenes/{scene['id']}.mp4" in text
        # Out of shot: start the player, give it time to clear the screen, then film.
        # Otherwise the typed command shows for the first frames (and in the GIF).
        steps = [line for line in text.splitlines() if line.split(" ")[0] in
                 ("Hide", "Type", "Enter", "Show", "Sleep")]
        assert [s.split(" ")[0] for s in steps] == ["Hide", "Type", "Enter", "Sleep", "Show",
                                                    "Sleep"]
        assert steps[3] == f"Sleep {HIDDEN_START_S}s"
        assert steps[5] == f"Sleep {round(scene['duration'], 2)}s"
    # Beats after the first, and the hold, are shifted by the time spent out of shot.
    assert ("player.py a --session examples/demo/session.json --at 0.0,"
            f"{round(LEAD_IN_S + 2.35 + HIDDEN_START_S, 3)} "
            f"--hold {round(plan[0]['duration'] + HIDDEN_START_S, 2)}") in tapes["a"]


def test_min_duration_covers_stream_replay():
    session = {"scenes": {"stream": {
        "wire": [{"chunks": [[1.0, "a"], [4.0, "b"]]}],
        "app_got": {"chunks": [[1.1, "a"], [3.1, "b"]]}}, "title": {}}}
    assert min_durations_from_session(session) == {
        "stream": pytest.approx(LEAD_IN_S + 3.0 + 2.0 + 1.0)}


def test_own_voice_too_long_is_rejected():
    plan = scene_plan(TIMINGS, {}, BEATS)
    assert voice_overrun({"a": 4.0}, plan) == []
    [message] = voice_overrun({"a": 9.0, "b": 1.0}, plan)
    assert message.startswith("a:") and "s too long" in message


def test_every_narration_line_fits_two_subtitle_lines():
    # The band under the terminal holds two lines; a third would cover the terminal.
    timings = {sid: {"duration": 1.0, "cues": [{"text": t, "start": 0.0, "end": 1.0}
                                               for t in cues]}
               for sid, cues in parse_narration(NARRATION).items()}
    plan = scene_plan(timings, {}, {})
    for block in srt(plan, timings).strip().split("\n\n"):
        assert len(block.splitlines()[2:]) <= 2, block
