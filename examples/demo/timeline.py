"""From narration to timing: the script's cues, then (from the voice's timings) scene
durations, subtitles and the VHS tape.

Standard library only, and imports nothing from the repo: voice.py imports it inside the
voice container, where only this folder is mounted.
"""

import re


def parse_narration(text: str) -> dict[str, list[str]]:
    """`## <scene_id>` sections, in order, each a list of cues (one per non-empty line).
    Anything before the first section is notes for humans and is ignored."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"##\s+(\S+)\s*$", line)
        if heading:
            current = sections.setdefault(heading.group(1), [])
        elif current is not None and line.strip():
            current.append(line.strip())
    return sections
