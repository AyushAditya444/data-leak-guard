# syntax=docker/dockerfile:1

# --- Stage 1: build the name model --------------------------------------------------
# PyTorch is only needed to export the NER model to ONNX, so it lives in this throwaway
# stage. The final image copies just the exported files (~70-110 MB), never PyTorch.
FROM python:3.11-slim AS model

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
# CPU-only PyTorch first (the default PyPI wheel bundles GPU libraries: gigabytes), in its
# own layer before the source is copied, so code changes don't reinstall it.
RUN pip install "torch>=2.14,<3" --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[export]"
# The download cache survives rebuilds, so changing app code doesn't re-download weights.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -m leakguard.ner.export --out /models/ner

# --- Stage 2: the guard ---------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEAKGUARD_NER_MODEL_DIR=/app/models/ner

WORKDIR /app

# Install dependencies first so code-only changes reuse this cached layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY --from=model /models/ner /app/models/ner

# Never run as root inside the container.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "--factory", "leakguard.app:build_app", "--host", "0.0.0.0", "--port", "8080"]
