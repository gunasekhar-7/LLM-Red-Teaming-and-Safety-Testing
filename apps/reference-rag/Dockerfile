# Multi-stage build for reference RAG (build context: repo root)
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY libs/guards ./libs/guards
COPY apps/reference-rag ./apps/reference-rag

RUN uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0" && \
    uv pip install --system --no-cache \
        ./libs/guards \
        ./apps/reference-rag && \
    python -m spacy download en_core_web_sm

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY apps/reference-rag/src/ ./src/
COPY config/ ./config/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "reference_rag.main"]
