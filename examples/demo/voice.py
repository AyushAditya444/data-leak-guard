"""Read the narration aloud with Kokoro, offline. Runs inside the voice image
(voice.Dockerfile), started by build.py:

    python /demo/voice.py --narration /demo/narration.md --out /out

Writes one WAV per scene and timings.json: each scene's length and each cue's start and end
(seconds from the scene's start), which build.py turns into subtitles and scene durations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from timeline import parse_narration  # same folder: mounted next to this script

VOICE, LANG, SPEED = "af_heart", "en-us", 1.0  # af_heart chosen by the author
CUE_GAP_S = 0.35  # silence between two sentences of a scene
MODEL, VOICES = "/models/kokoro-v1.0.onnx", "/models/voices-v1.0.bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kokoro = Kokoro(MODEL, VOICES)
    timings = {}
    for scene_id, cues in parse_narration(args.narration.read_text(encoding="utf-8")).items():
        pieces, cue_times, at, rate = [], [], 0.0, 24000
        for i, cue in enumerate(cues):
            samples, rate = kokoro.create(cue, voice=VOICE, speed=SPEED, lang=LANG)
            if i:
                pieces.append(np.zeros(int(CUE_GAP_S * rate), dtype=np.float32))
                at += CUE_GAP_S
            pieces.append(samples)
            cue_times.append({"text": cue, "start": round(at, 3),
                              "end": round(at + len(samples) / rate, 3)})
            at += len(samples) / rate
        sf.write(args.out / f"{scene_id}.wav", np.concatenate(pieces), rate)
        timings[scene_id] = {"duration": round(at, 3), "cues": cue_times}
        print(f"{scene_id}: {at:.1f} s", flush=True)
    (args.out / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
    print(f"total: {sum(t['duration'] for t in timings.values()):.1f} s")


if __name__ == "__main__":
    main()
