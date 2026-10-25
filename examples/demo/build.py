"""Build the demo video and GIF from the recorded session. No LLM is called.

    python -m examples.demo.build                     # Kokoro voice, committed session
    python -m examples.demo.build --voice-dir my-voice  # your own <scene>.wav files
    python -m examples.demo.build --session build/demo/session-mock.json  # a dry run

Steps: narrate (Kokoro, Docker) -> scene timings, subtitles and one VHS tape per scene ->
film each scene (VHS, Docker) -> stretch each clip to its planned length, join them with
the voice and subtitles, and cut the GIF (ffmpeg, from the VHS image). Outputs: build/demo/demo.mp4 (not committed), docs/demo/demo.gif, and the
generated examples/demo/demo.srt and tapes/. Needs Docker; nothing to install.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from examples.demo.scenes import FONT_SIZE
from examples.demo.session import load_session, verify_session
from examples.demo.timeline import (
    LEAD_IN_S,
    min_durations_from_session,
    scene_plan,
    srt,
    tape,
    voice_overrun,
)

REPO = Path.cwd()
DEMO = Path("examples/demo")
OUT = Path("build/demo")
DEFAULT_SESSION = DEMO / "session.json"
VHS = "ghcr.io/charmbracelet/vhs:v0.12.1"
VOICE_IMAGE = "leakguard-demo-voice"
GIF_MAX_MB = 5
# Subtitles sit in the 120 px band added under the 600 px terminal. Sizes are in libass's
# script units (288 lines high), scaled to the 720 px frame.
SUBTITLE_STYLE = "FontName=Noto Sans,FontSize=12,MarginV=10,Outline=1,Shadow=0"
BAND_COLOUR = "0x1e1e2e"  # the terminal theme's background


def _run(args: list[str]) -> str:
    print("+", " ".join(args[:6]), "..." if len(args) > 6 else "", flush=True)
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _in_vhs_image(entrypoint: str, *args: str) -> str:
    """Run ffmpeg/ffprobe/vhs from the VHS image with the repo mounted at /vhs."""
    return _run(["docker", "run", "--rm", "-v", f"{REPO}:/vhs", "-w", "/vhs",
                 "--entrypoint", entrypoint, VHS, *args])


def narrate() -> dict:
    voice = OUT / "voice"
    voice.mkdir(parents=True, exist_ok=True)
    _run(["docker", "build", "-q", "-f", str(DEMO / "voice.Dockerfile"), "-t", VOICE_IMAGE,
          str(DEMO)])
    _run(["docker", "run", "--rm", "-v", f"{REPO / DEMO}:/demo:ro", "-v", f"{REPO / voice}:/out",
          VOICE_IMAGE, "--narration", "/demo/narration.md", "--out", "/out"])
    return json.loads((voice / "timings.json").read_text(encoding="utf-8"))


def collect_own_voice(voice_dir: Path, scene_ids: list[str],
                      dest: Path) -> tuple[dict[str, Path], list[str]]:
    """Copy the author's `<scene_id>.wav` files into `dest` (inside the repo, so the
    ffmpeg container can see them wherever they were recorded). Returns the copies by scene
    and the names of WAV files that match no scene."""
    found, ignored = {}, []
    for wav in sorted(voice_dir.glob("*.wav")):
        if wav.stem not in scene_ids:
            ignored.append(wav.name)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        found[wav.stem] = Path(shutil.copyfile(wav, dest / wav.name))
    if not found:
        expected = ", ".join(f"{sid}.wav" for sid in scene_ids)
        raise ValueError(f"no recordings in {voice_dir}: expected files named {expected}")
    return found, ignored


def own_voice_durations(own: dict[str, Path]) -> dict[str, float]:
    return {scene_id: _duration(wav) for scene_id, wav in own.items()}


def mix_voice(plan: list[dict], own: dict[str, Path]) -> Path:
    """One audio track: each scene's narration (the author's own where given) placed at
    its scene's start."""
    inputs, filters = [], []
    for i, scene in enumerate(plan):
        wav = own.get(scene["id"], OUT / "voice" / f"{scene['id']}.wav")
        inputs += ["-i", wav.as_posix()]
        delay = round((scene["start"] + LEAD_IN_S) * 1000)
        filters.append(f"[{i}:a]aresample=48000,adelay={delay}:all=1[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(plan)))
    graph = ";".join(filters) + f";{mix}amix=inputs={len(plan)}:normalize=0[out]"
    track = OUT / "voice.wav"
    _in_vhs_image("ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
                  "-map", "[out]", track.as_posix())
    return track


def _duration(path: Path) -> float:
    out = _in_vhs_image("ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path.as_posix())
    return float(out.strip())


def film(plan: list[dict], tapes: Path) -> list[tuple[Path, float]]:
    """One VHS clip per scene, with its filmed length (a little short of the plan: VHS
    drops frames when the machine is busy)."""
    clips = []
    for scene in plan:
        _in_vhs_image("vhs", (tapes / f"{scene['id']}.tape").as_posix())
        clip = OUT / "scenes" / f"{scene['id']}.mp4"
        clips.append((clip, _duration(clip)))
    return clips


def join(plan: list[dict], clips: list[tuple[Path, float]], track: Path, subtitles: Path,
         video: Path) -> None:
    """Stretch each clip to its planned length, join them, add the subtitle band, burn in
    the subtitles and add the voice."""
    inputs, parts = [], []
    for i, (scene, (clip, filmed)) in enumerate(zip(plan, clips, strict=True)):
        inputs += ["-i", clip.as_posix()]
        stretch = scene["duration"] / filmed
        parts.append(f"[{i}:v]setpts=(PTS-STARTPTS)*{stretch:.6f},fps=50,"
                     f"tpad=stop_mode=clone:stop_duration=1,"
                     f"trim=duration={scene['duration']},setpts=PTS-STARTPTS[v{i}]")
    joined = "".join(f"[v{i}]" for i in range(len(plan)))
    graph = (";".join(parts) + f";{joined}concat=n={len(plan)}:v=1:a=0,"
             f"pad=1280:720:0:0:color={BAND_COLOUR},"
             f"subtitles={subtitles.as_posix()}:force_style='{SUBTITLE_STYLE}'[out]")
    _in_vhs_image("ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", track.as_posix(),
                  "-filter_complex", graph, "-map", "[out]", "-map", f"{len(plan)}:a",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                  "-c:a", "aac", "-b:a", "128k", video.as_posix())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the demo video and GIF.")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--voice-dir", type=Path, help="your own <scene_id>.wav recordings")
    parser.add_argument("--skip-voice", action="store_true",
                        help="reuse build/demo/voice/ from the last run")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    committed = args.session.resolve() == DEFAULT_SESSION.resolve()
    generated = DEMO if committed else OUT  # a dry run never overwrites committed files
    gif = Path("docs/demo/demo.gif") if committed else OUT / "demo.gif"

    # Checked before any Docker step. The committed video's narration says it was recorded
    # live on Groq, so only such a session may become it.
    session = load_session(args.session)
    problems = verify_session(session, require_live=committed)
    if problems:
        print("this session does not support the narration:\n" + "\n".join(problems),
              file=sys.stderr)
        return 1
    own: dict[str, Path] = {}
    if args.voice_dir:
        try:
            own, ignored = collect_own_voice(args.voice_dir, list(session["scenes"]),
                                             OUT / "own-voice")
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        if ignored:
            print("ignored (no scene with that name): " + ", ".join(ignored), file=sys.stderr)

    timings_file = OUT / "voice" / "timings.json"
    if args.skip_voice and timings_file.is_file():
        timings = json.loads(timings_file.read_text(encoding="utf-8"))
    else:
        timings = narrate()

    beats = {sid: scene["show"]["beats"] for sid, scene in session["scenes"].items()}
    plan = scene_plan(timings, min_durations_from_session(session), beats)
    if own:
        problems = voice_overrun(own_voice_durations(own), plan)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1

    subtitles = generated / "demo.srt"
    subtitles.write_text(srt(plan, timings), encoding="utf-8", newline="\n")
    tapes = generated / "tapes"
    tapes.mkdir(exist_ok=True)
    for scene_id, text in tape(plan, args.session.as_posix(), FONT_SIZE).items():
        (tapes / f"{scene_id}.tape").write_text(text, encoding="utf-8", newline="\n")
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")

    clips = film(plan, tapes)
    track = mix_voice(plan, own)
    video = OUT / "demo.mp4"
    join(plan, clips, track, subtitles, video)

    core = next(scene for scene in plan if scene["id"] == "core")
    gif.parent.mkdir(parents=True, exist_ok=True)
    _in_vhs_image(
        "ffmpeg", "-y", "-loglevel", "error", "-ss", str(core["start"]),
        "-t", str(core["duration"]), "-i", video.as_posix(),
        "-vf", "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];"
               "[b][p]paletteuse=dither=bayer", gif.as_posix())

    total = sum(scene["duration"] for scene in plan)
    size_mb = gif.stat().st_size / 1e6
    print(f"video: {video} ({total:.0f} s)\ngif:   {gif} ({size_mb:.1f} MB)")
    if size_mb > GIF_MAX_MB:
        print(f"the GIF is over {GIF_MAX_MB} MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
