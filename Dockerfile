FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code-only changes reuse this cached layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Never run as root inside the container.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "--factory", "leakguard.app:build_app", "--host", "0.0.0.0", "--port", "8080"]
