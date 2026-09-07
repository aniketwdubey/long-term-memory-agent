FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[otel]"

COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Offline by default: no credentials, no network, deterministic.
ENV ENGRAM_CHAT_PROVIDER=stub \
    ENGRAM_EMBEDDER=hashing

CMD ["python", "scripts/demo.py"]
