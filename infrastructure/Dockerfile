# ──────────── Stage 1: Build dependencies ────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install "."

# ──────────── Stage 2: Slim runtime ────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (no build tools, no source of pyproject.toml)
COPY data_pipeline/ data_pipeline/
COPY quant_engine/ quant_engine/
COPY api_gateway/ api_gateway/

# Pre-trained models directory (mounted as volume in production)
RUN mkdir -p models logs/decisions

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"]

CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
