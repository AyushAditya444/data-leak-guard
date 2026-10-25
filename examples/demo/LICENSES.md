# Tools used to make the demo video

None of these are part of the guard or its Docker image, and none are stored in this
repository: they are downloaded into separate Docker images when the video is built.
Licences checked on 2026-10-25 from each project's own page.

| Tool | Used for | Licence |
|---|---|---|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | the narrator's voice (model weights, voice `af_heart`) | Apache-2.0 |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | running Kokoro on CPU with ONNX Runtime; its release `model-files-v1.0` provides the converted model files | MIT |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng), via [espeakng-loader](https://github.com/thewh1teagle/espeakng-loader) | turning text into phonemes inside the voice image | GPL-3.0 (espeak-ng), MIT (loader) |
| [VHS](https://github.com/charmbracelet/vhs) | filming the terminal (`ghcr.io/charmbracelet/vhs:v0.12.1`) | MIT |
| [FFmpeg](https://ffmpeg.org/legal.html) | joining video, voice and subtitles; making the GIF (the build bundled in the VHS image) | GPL (that build is configured with `--enable-gpl`) |

The voice is synthetic. The recorded session the video replays is
[session.json](session.json); all personal data in it is made up.
