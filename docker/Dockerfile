# ─────────────────────────────────────────────────────────────────────────────
# Exam Stress Balancer — FastAPI Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: dependency builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="exam-stress-balancer"
LABEL description="Exam Stress Balancer — RL-powered study recommendation API"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/         ./src/
COPY models/      ./models/
COPY data/        ./data/
COPY evaluations/ ./evaluations/
COPY logs/        ./logs/

# Ensure directories exist and are writable
RUN mkdir -p models data evaluations logs \
 && chown -R appuser:appuser /app

# Switch to non-root
USER appuser

# Environment defaults (override via docker run -e or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=2 \
    LOG_LEVEL=INFO

EXPOSE 8000

# Health-check: ping /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", \
     "uvicorn src.api.main:app \
      --host $API_HOST \
      --port $API_PORT \
      --workers $API_WORKERS \
      --log-level info"]
