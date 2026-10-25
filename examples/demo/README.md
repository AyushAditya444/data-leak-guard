# The demo video

A ~2-minute narrated, subtitled video of the guard at work, and the GIF at the top of the
main README, both generated from the files in this folder.

## Record, then replay

The video is **recorded once, then replayed**:

1. `record.py` runs every scene through the real guard (built by `create_app`, exactly as
   the Docker image runs it), with your app's side played by the official OpenAI SDK. A
   small **wire tap** (`tap.py`, demo-only, not part of the guard) sits on the guard's
   connection to the LLM and records exactly what left the machine and what came back. It
   never reads headers, so it never sees the API key. Everything is saved to
   [session.json](session.json).
2. `build.py` films [player.py](player.py) replaying that session, scene by scene, and
   adds the voice and subtitles. Building the video calls no LLM.

Why not film a live run? Every render would get a different reply (a box could overflow,
a provider error could break the take), the API key would have to go into the recording
container, and nobody could check the video against anything. With a replay, the video
shows a real run that you can open and read, and every claim the narration makes is
checked against it by `verify_session` (in [session.py](session.py)). A recording that
fails a check is recorded again, never committed. The narration is not changed to fit it.

The committed session was recorded against Groq's `openai/gpt-oss-120b`. It says so on
screen.

## Commands

Run from the repository root with the dev environment active (`pip install -e ".[dev]"`)
and the name model exported (`python -m leakguard.ner.export`). The build needs Docker
and nothing else: VHS, ffmpeg and the Kokoro voice run in containers.

```bash
# Record with a real model (needs a free Groq key in GROQ_API_KEY; never put it in a file)
python -m examples.demo.record --upstream groq

# Or record against the built-in mock: free, no key, works offline (what CI runs)
python -m examples.demo.record --upstream mock --out build/demo/session-mock.json

# Build the video and GIF from the committed session (about 4 minutes)
python -m examples.demo.build

# A dry run from another session never touches the committed files
python -m examples.demo.build --session build/demo/session-mock.json
```

Outputs:

| File | Committed? |
|---|---|
| `build/demo/demo.mp4`, the video | no. Uploaded to the main README by hand through GitHub's web editor |
| `docs/demo/demo.gif`, the support-message scene | yes |
| `examples/demo/demo.srt`, the subtitles (also burned into the video) | yes |
| `examples/demo/tapes/*.tape`, the VHS script for each scene | yes (generated) |

## Using your own voice

The narrator is [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (voice `af_heart`).
To replace it with your own voice, read each `## <scene>` section of
[narration.md](narration.md) into a file named after the scene (`title.wav`, `core.wav`,
…). Leave out any scene you want to keep in Kokoro's voice. Then:

```bash
python -m examples.demo.build --voice-dir path/to/your/wavs --skip-voice
```

The subtitles stay as written, so follow the script. Each scene keeps the length the
Kokoro reading gave it; if one of your files is too long for its scene, the build stops
and says which scene, and by how much.

## Files

| File | What it does |
|---|---|
| [scenes.py](scenes.py) | the scenes, their made-up personal data, and the screen size |
| [tap.py](tap.py) | the wire tap |
| [record.py](record.py) | runs the guard and the scenes, writes `session.json` |
| [session.py](session.py) | the session format, and `verify_session` |
| [player.py](player.py) | draws one scene from the session (standard library only: it runs in the VHS image) |
| [narration.md](narration.md) | the script: voice and subtitles |
| [voice.py](voice.py), [voice.Dockerfile](voice.Dockerfile) | Kokoro text-to-speech in Docker |
| [timeline.py](timeline.py) | scene timings, subtitles and tapes from the voice's timings |
| [build.py](build.py) | the one command that builds everything |
| [LICENSES.md](LICENSES.md) | the tools used to make the video, and their licences |

Timing: each scene lasts as long as its narration plus 0.8 s, so one scene's narration
never runs into the next. VHS drops a few percent of frames on a busy machine, so each
scene is filmed as its own clip and stretched back to its planned length before the clips
are joined.
