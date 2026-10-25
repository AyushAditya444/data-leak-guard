"""The narration script, and turning its audio timings into subtitles and a VHS tape."""

import re
from pathlib import Path

from examples.demo.scenes import SCENES
from examples.demo.timeline import parse_narration

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
