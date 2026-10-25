"""The recorded demo session, and the checks that keep the narration honest.

A session is plain JSON (see examples/demo/README.md for the shape). The video replays it;
`verify_session` checks that every claim the narration makes is true of it. A recording
that fails a check is recorded again, never committed: the narration is not bent to fit.
"""

import json
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")


def load_session(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_session(session: dict, path: Path) -> None:
    text = json.dumps(session, ensure_ascii=False, indent=1) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def wire_reply(scene: dict) -> str:
    """What the LLM sent back (still masked), from the last upstream request."""
    return "".join(text for _, text in scene["wire"][-1]["chunks"]) if scene["wire"] else ""


def _split_placeholder(chunks: list) -> bool:
    """True if some placeholder starts in one chunk and ends in a later one."""
    text, borders = "", []
    for _, piece in chunks:
        text += piece
        borders.append(len(text))
    return any(m.start() < b < m.end() for m in PLACEHOLDER.finditer(text) for b in borders)


def _check_masked(scene: dict) -> list[str]:
    values = scene["show"]["highlight"]
    sent = json.dumps([w["messages"] for w in scene["wire"]], ensure_ascii=False)
    reply = scene["app_got"]["text"]
    log = json.dumps(scene["log"], ensure_ascii=False)
    problems = []
    if not scene["wire"]:
        problems.append("nothing reached the LLM")
    for value in values:
        if value in sent:
            problems.append(f"{value!r} left the machine")
            continue
        # Half a value is a leak too ("Priya [NAME_1]"): no word of it may be on the wire.
        parts = [w for w in re.findall(r"\w{3,}", value)
                 if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", sent)]
        if parts:
            problems.append(f"part of {value!r} left the machine: "
                            + ", ".join(repr(w) for w in parts))
    if not any(v in reply for v in values):
        problems.append("no real value was restored in the reply")
    if PLACEHOLDER.search(reply):
        problems.append("a placeholder is left in the reply")
    if not scene["log"] or not scene["log"].get("findings"):
        problems.append("the log line has no findings")
    problems += [f"the log line contains {v!r}" for v in values if v in log]
    if scene["stream"] and scene["wire"] and not _split_placeholder(scene["wire"][-1]["chunks"]):
        problems.append("no placeholder was split across streamed chunks")
    return problems


def _check_blocked(scene: dict) -> list[str]:
    problems = []
    if scene["app_got"]["status"] != 400:
        problems.append(f"expected HTTP 400, got {scene['app_got']['status']}")
    if scene["wire"]:
        problems.append("the request was sent to the LLM")
    return problems


def _check_unchanged(scene: dict) -> list[str]:
    if not scene["wire"] or scene["app_got"]["status"] != 200:
        return ["the request did not complete"]
    if scene["wire"][0]["messages"][-1] != scene["app_sent"][-1]:
        return ["the message was not sent unchanged"]
    return []


def _check_results(results: dict) -> list[str]:
    ours = results["ours"]
    others = [v for k, v in results.items() if k != "ours"]
    problems = []
    if not all(ours["f1"] > o["f1"] for o in others):
        problems.append("ours does not have the highest F1")
    if not all(ours["false_alarms"] < o["false_alarms"] for o in others):
        problems.append("ours does not have the fewest false alarms")
    return problems


CHECKS = {"masked": _check_masked, "blocked": _check_blocked, "unchanged": _check_unchanged}


def verify_session(session: dict, require_live: bool = False) -> list[str]:
    """Every narrated claim that does not hold, as "<scene>: <what is wrong>".
    `require_live`: the session the video ships with, where the narration also says it was
    recorded live against a real model on Groq."""
    problems = []
    if require_live and session["meta"]["upstream"] != "groq":
        problems.append(f"meta: the narration says this was recorded live on Groq, but its "
                        f"upstream is {session['meta']['upstream']!r}")
    for scene_id, scene in session["scenes"].items():
        check = CHECKS.get(scene["show"]["expect"])
        problems += [f"{scene_id}: {p}" for p in (check(scene) if check else [])]
    expects = {scene["show"]["expect"] for scene in session["scenes"].values()}
    problems += [f"session: no {e!r} scene" for e in CHECKS if e not in expects]
    problems += [f"results: {p}" for p in _check_results(session["results"])]
    return problems
