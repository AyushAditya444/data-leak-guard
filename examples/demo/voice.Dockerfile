# syntax=docker/dockerfile:1
# The demo's narrator: Kokoro text-to-speech on CPU, offline once built. Built and run by
# `python -m examples.demo.build`; the scripts and narration are mounted at run time, so
# editing the narration does not rebuild this image.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
RUN pip install kokoro-onnx==0.6.1 onnxruntime==1.31.0 soundfile==0.14.0 numpy==2.4.6

# Kokoro-82M (Apache-2.0) converted to ONNX by the kokoro-onnx project. Checksums pinned so
# a changed download fails the build instead of changing the voice. The folder is made
# first: ADD's --chmod would otherwise apply to it too and make it unreadable.
RUN mkdir -m 755 /models
ADD --chmod=644 --checksum=sha256:7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
    /models/kokoro-v1.0.onnx
ADD --chmod=644 --checksum=sha256:bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d \
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
    /models/voices-v1.0.bin

RUN useradd --create-home --uid 1000 narrator
USER narrator
WORKDIR /demo
ENTRYPOINT ["python", "/demo/voice.py"]
