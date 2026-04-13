# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy dependency files only (better Docker layer caching)
COPY pyproject.toml poetry.lock* ./

# Configure Poetry:
# - no virtual env inside Docker (we're already isolated)
# - no interaction prompts
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="your-team@company.com"
LABEL description="Legal Contract Entity Extractor — Production Image"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/data/raw /app/data/processed /app/tmp \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser app/        ./app/
COPY --chown=appuser:appuser ocr/        ./ocr/
COPY --chown=appuser:appuser config/     ./config/
COPY --chown=appuser:appuser training/   ./training/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT} --workers ${WORKERS:-1}"]