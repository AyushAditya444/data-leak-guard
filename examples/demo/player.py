"""Replay one scene of the recorded demo session in the terminal. VHS runs this to film it.

    python3 examples/demo/player.py core --at 0,3.9,7.5 --hold 12

It draws only what is in examples/demo/session.json (recorded by record.py); it never
calls an LLM. Standard library only: it runs on the VHS image's own Python, not the venv.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"

PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
TRUNCATED = "… (full text in examples/demo/session.json)"
REPO = "github.com/AyushAditya444/data-leak-guard"
BORDER = "·"  # drawn between streamed chunks


def visible_len(line: str) -> int:
    return len(ANSI.sub("", line))


# --- text: wrap, colour, frame -------------------------------------------------------

def _wrap(text: str, width: int) -> list[tuple[int, int]]:
    """Greedy word wrap that returns (start, end) ranges into `text`, so colours computed
    on the whole text still apply after wrapping."""
    lines, base = [], 0
    for para in text.split("\n"):
        start = end = None
        for word in re.finditer(r"\S+", para):
            ws, we = base + word.start(), base + word.end()
            while we - ws > width:  # a word longer than a line: hard split
                if start is not None:
                    lines.append((start, end))
                lines.append((ws, ws + width))
                ws, start, end = ws + width, None, None
            if start is None:
                start, end = ws, we
            elif we - start <= width:
                end = we
            else:
                lines.append((start, end))
                start, end = ws, we
        lines.append((start, end) if start is not None else (base, base))
        base += len(para) + 1
    return lines


def paint(text: str, values, width: int, borders=()) -> list[str]:
    """Wrap `text` to `width` columns, sensitive `values` in yellow, placeholders in green.
    `borders` are positions in `text` where a dim chunk border is drawn."""
    colours = [""] * len(text)
    for value in values:
        at = text.find(value)
        while value and at >= 0:
            colours[at:at + len(value)] = [YELLOW] * len(value)
            at = text.find(value, at + len(value))
    for match in PLACEHOLDER.finditer(text):
        colours[match.start():match.end()] = [GREEN] * (match.end() - match.start())
    inside = {b for b in borders if 0 < b < len(text)}
    chars, styles = [], []
    for i, char in enumerate(text):
        if i in inside:
            chars.append(BORDER)
            styles.append(DIM)
        chars.append(char)
        styles.append(colours[i])
    shown = "".join(chars)
    out = []
    for start, end in _wrap(shown, width):
        line, current = "", ""
        for i in range(start, end):
            if styles[i] != current:
                line += RESET + styles[i] if current else styles[i]
                current = styles[i]
            line += shown[i]
        out.append(line + (RESET if current else ""))
    return out


def box(title: str, lines: list[str], width: int, color: str, max_lines: int) -> list[str]:
    """`lines` (already wrapped to width - 4) in a frame with the title in its top border."""
    inner = width - 4
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [DIM + TRUNCATED[:inner] + RESET]
    title = title[:width - 6]
    out = [f"{color}┌─ {BOLD}{title}{RESET}{color} {'─' * (width - 5 - len(title))}┐{RESET}"]
    for line in lines:
        pad = " " * (inner - visible_len(line))
        out.append(f"{color}│{RESET} {line}{pad} {color}│{RESET}")
    out.append(f"{color}└{'─' * (width - 2)}┘{RESET}")
    return out


def _centre(lines: list[str], cols: int, rows: int) -> list[str]:
    top = max(0, (rows - len(lines)) // 2 - 1)
    return [""] * top + [" " * max(0, (cols - visible_len(line)) // 2) + line for line in lines]


# --- scenes --------------------------------------------------------------------------

def _header(scene_id: str, session: dict, cols: int) -> str:
    scene, meta = session["scenes"][scene_id], session["meta"]
    title = scene["show"]["title"]
    if "app_sent" not in scene:
        return f"{BOLD}{CYAN}▌ {title}{RESET}"
    if meta["upstream"] == "mock":
        where = f"live run against the mock LLM on {meta['recorded']}"
    else:
        where = (f"recorded live against {meta['upstream'].capitalize()} {meta['model']} "
                 f"on {meta['recorded']}, replayed from session.json")
    where = where[:max(0, cols - len(title) - 5)]
    return f"{BOLD}{CYAN}▌ {title}{RESET}{DIM} · {where}{RESET}"


def _user_text(messages: list[dict]) -> str:
    return messages[-1]["content"] if messages else ""


def _chunks_until(chunks: list, elapsed: float | None) -> list:
    if elapsed is None or not chunks:
        return chunks
    first = chunks[0][0]
    return [c for c in chunks if c[0] - first <= elapsed]


def _streamed(chunks: list, values, width: int, borders: bool) -> list[str]:
    text, ends = "", []
    for _, piece in chunks:
        text += piece
        ends.append(len(text))
    return paint(text, values, width, ends[:-1] if borders else ())


def _request_boxes(scene_id: str, scene: dict, beat: int, cols: int,
                   elapsed: float | None) -> list[list[str]]:
    w = cols - 4
    values = scene["show"]["highlight"]
    sent = box("YOUR APP SENT", paint(_user_text(scene["app_sent"]), values, w), cols, CYAN, 4)
    wire = scene["wire"][-1] if scene["wire"] else None
    got = scene["app_got"]
    left_title = "WHAT LEFT YOUR MACHINE"
    if scene_id == "secret":
        nothing = [f"{RED}nothing was sent: the guard blocked the request{RESET}"]
        error = paint(f"HTTP {got['status']} · {got['text']}", [], w)
        error = [f"{RED}{line}{RESET}" for line in error]
        return [sent] + ([box(left_title, nothing, cols, RED, 2),
                          box("YOUR APP GOT BACK", error, cols, RED, 3)] if beat >= 1 else [])
    left = paint(_user_text(wire["messages"]), values, w) if wire else []
    if scene_id == "lookalike":
        left[-1:] = [left[-1] + f"{DIM}  (sent unchanged){RESET}"] if left else []
        reply = paint(got["text"], values, w)
        return [sent] + ([box(left_title, left, cols, GREEN, 3),
                          box("YOUR APP GOT BACK", reply, cols, CYAN, 3)] if beat >= 1 else [])
    if scene_id == "stream":
        boxes = [box("YOUR APP SENT", paint(_user_text(scene["app_sent"]), values, w),
                     cols, CYAN, 2), box(left_title, left, cols, GREEN, 2)]
        if beat >= 1:
            chunks = _chunks_until(wire["chunks"], elapsed if beat == 1 else None)
            boxes.append(box(f"STREAMED BACK BY THE MODEL ({BORDER} = chunk border)",
                             _streamed(chunks, values, w, borders=True), cols, GREEN, 3))
        if beat >= 2:
            chunks = _chunks_until(got["chunks"], elapsed)
            boxes.append(box("YOUR APP GOT BACK", _streamed(chunks, values, w, borders=False),
                             cols, CYAN, 3))
        return boxes
    boxes = [sent, box(left_title, left, cols, GREEN, 4),
             box("YOUR APP GOT BACK", paint(got["text"], values, w), cols, CYAN, 4)]
    return boxes[:beat + 1]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _results_box(session: dict, beat: int, cols: int) -> list[str]:
    rows = [f"{'':<28}{'F1':>8}{'Coverage':>12}" + (f"{'False alarms (clean msgs)':>30}"
                                                     if beat >= 1 else "")]
    for engine, r in session["results"].items():
        line = f"{r['label']:<28}{_pct(r['f1']):>8}{_pct(r['coverage']):>12}"
        if beat >= 1:
            line += f"{str(r['false_alarms']) + ' of ' + str(r['negatives']):>30}"
        rows.append(f"{BOLD}{GREEN}{line}{RESET}" if engine == "ours" else line)
    scope = "Test split, types both support (Aadhaar, PAN, phone, email, card, name)."
    rows += ["", f"{DIM}{scope}{RESET}", f"{DIM}Read from data/eval/presidio-test.json.{RESET}"]
    return box("OUR ENGINE vs MICROSOFT PRESIDIO (same test set, same scoring)",
               rows, cols, CYAN, 10)


def _log_box(session: dict, cols: int) -> list[str]:
    record = session["scenes"]["core"]["log"]
    lines = ["{"]
    for i, (key, value) in enumerate(record.items()):
        line = f'  "{key}": {json.dumps(value)}' + ("," if i < len(record) - 1 else "")
        lines.append(f"{GREEN}{line}{RESET}" if key == "findings" else line)
    lines += ["}", "", f"{DIM}Counts only: no values, no message text.{RESET}"]
    return box("THE GUARD'S LOG LINE FOR THE SUPPORT MESSAGE", lines, cols, CYAN, 18)


def _card(scene_id: str, cols: int) -> list[str]:
    if scene_id == "title":
        return [f"{BOLD}{CYAN}Data Leak Guard{RESET}", "",
                "An OpenAI-compatible proxy that keeps Indian personal data",
                "and secrets from reaching LLM APIs.", "", f"{DIM}{REPO}{RESET}"]
    return [f"{GREEN}$ docker compose up{RESET}", "", REPO, "",
            f"{YELLOW}Known limit: UPI IDs with handles it doesn't know are often missed.{RESET}",
            f"{DIM}Results, method and threat model: see the README.{RESET}"]


def _one_line_box(cols: int) -> list[str]:
    lines = ["from openai import OpenAI", "",
             f"{RED}- client = OpenAI(api_key=key){RESET}",
             f'{GREEN}+ client = OpenAI(base_url="http://localhost:8080/v1", api_key=key){RESET}',
             "", f"{DIM}Everything else in your code stays the same.{RESET}"]
    return box("YOUR APP (official OpenAI SDK)", lines, cols, CYAN, 8)


def _stack(header: str, boxes: list[list[str]], rows: int) -> list[str]:
    body = sum(len(b) for b in boxes)
    gap = 1 if 2 + body + len(boxes) - 1 <= rows else 0
    out = [header, ""]
    for i, b in enumerate(boxes):
        out += ([""] * gap if i else []) + b
    return out[:rows]


def render(scene_id: str, session: dict, beat: int, cols: int, rows: int,
           elapsed: float | None = None) -> list[str]:
    """The whole screen for `scene_id` after `beat` reveals (0-based). `elapsed` limits a
    streaming box to the chunks that had arrived by then (None = all)."""
    scene = session["scenes"][scene_id]
    header = _header(scene_id, session, cols)
    if scene_id in ("title", "outro"):
        return _centre(_card(scene_id, cols), cols, rows)[:rows]
    if "app_sent" in scene:
        boxes = _request_boxes(scene_id, scene, beat, cols, elapsed)
    elif scene_id == "one_line":
        boxes = [_one_line_box(cols)]
    elif scene_id == "log":
        boxes = [_log_box(session, cols)]
    else:
        boxes = [_results_box(session, beat, cols)]
    return _stack(header, boxes, rows)


# --- playing ---------------------------------------------------------------------------

ANIMATED = {"stream": {1: "wire", 2: "got"}}  # beat -> whose chunks it plays


def _draw(lines: list[str]) -> None:
    sys.stdout.write("\033[H" + "".join(line + "\033[K\n" for line in lines) + "\033[J")
    sys.stdout.flush()


def play(scene_id: str, session: dict, at: list[float], hold: float = 0.0,
         cols: int = 108, rows: int = 22) -> None:
    """Draw beat k at `at[k]` seconds; stream beats play their chunks at the recorded
    offsets. Stays on screen until `hold` seconds, so the shell prompt is never filmed."""
    started = time.monotonic()

    def wait_until(t: float) -> None:
        time.sleep(max(0.0, started + t - time.monotonic()))

    sys.stdout.write("\033[?25l\033[2J")  # hide the cursor, clear the screen
    scene = session["scenes"][scene_id]
    beats = scene["show"]["beats"]
    for beat in range(beats):
        begin = at[min(beat, len(at) - 1)] if at else 0.0
        wait_until(begin)
        whose = ANIMATED.get(scene_id, {}).get(beat)
        if whose:
            chunks = scene["wire"][-1]["chunks"] if whose == "wire" else scene["app_got"]["chunks"]
            first = chunks[0][0] if chunks else 0.0
            for t, _ in chunks:
                wait_until(begin + t - first)
                _draw(render(scene_id, session, beat, cols, rows, elapsed=t - first))
        _draw(render(scene_id, session, beat, cols, rows))
    wait_until(hold)
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one scene of the demo session.")
    parser.add_argument("scene")
    parser.add_argument("--session", default="examples/demo/session.json")
    parser.add_argument("--at", help="seconds at which each beat appears, e.g. 0,3.9,7.5")
    parser.add_argument("--hold", type=float, default=0.0, help="seconds to stay on screen")
    parser.add_argument("--cols", type=int, default=108)
    parser.add_argument("--rows", type=int, default=22)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    session = json.loads(Path(args.session).read_text(encoding="utf-8"))
    beats = session["scenes"][args.scene]["show"]["beats"]
    if args.at is None:  # no timing: print the finished scene once
        print("\n".join(render(args.scene, session, beats - 1, args.cols, args.rows)))
        return 0
    play(args.scene, session, [float(x) for x in args.at.split(",")], args.hold,
         args.cols, args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
