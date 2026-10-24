"""Labelled samples and the "marked text" format used to write them.

Marked text puts each gold value inline with its type:
    "call [[PHONE:98765 43210]] or mail [[EMAIL:a@b.in]]"
parse_marked() turns that into clean text plus exact gold spans, so nobody ever has
to count character offsets by hand.
"""

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from leakguard.detect.spans import Span
from leakguard.entities import EntityType

_MARK = re.compile(r"\[\[([A-Z_]+):(.*?)\]\]", re.DOTALL)


@dataclass(frozen=True)
class Sample:
    id: str
    category: str
    template: str
    text: str
    spans: tuple[Span, ...]  # gold spans, detector="gold"


def parse_marked(marked: str) -> tuple[str, tuple[Span, ...]]:
    pieces: list[str] = []
    spans: list[Span] = []
    length = 0  # length of the clean text built so far
    cursor = 0
    for m in _MARK.finditer(marked):
        if m.group(1) not in EntityType.__members__:
            raise ValueError(f"Unknown entity type in marked text: {m.group(1)}")
        before = marked[cursor:m.start()]
        value = m.group(2)
        pieces += [before, value]
        start = length + len(before)
        spans.append(Span(start, start + len(value), EntityType(m.group(1)), detector="gold"))
        length = start + len(value)
        cursor = m.end()
    pieces.append(marked[cursor:])
    return "".join(pieces), tuple(spans)


def _to_json(sample: Sample) -> str:
    return json.dumps({
        "id": sample.id,
        "category": sample.category,
        "template": sample.template,
        "text": sample.text,
        "spans": [{"start": s.start, "end": s.end, "type": s.type.value} for s in sample.spans],
    }, ensure_ascii=False)


def _from_json(line: str) -> Sample:
    raw = json.loads(line)
    spans = tuple(Span(s["start"], s["end"], EntityType(s["type"]), detector="gold")
                  for s in raw["spans"])
    return Sample(raw["id"], raw["category"], raw["template"], raw["text"], spans)


def write_jsonl(path: str | os.PathLike, samples: Iterable[Sample]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(_to_json(sample) + "\n" for sample in samples)


def read_jsonl(path: str | os.PathLike) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        return [_from_json(line) for line in f if line.strip()]
