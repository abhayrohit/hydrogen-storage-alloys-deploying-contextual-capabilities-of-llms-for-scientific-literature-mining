# Multi-stage for smaller final image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (optional: locales & basic build tools)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Default env for cloud demos (override in production if needed)
ENV PORT=8000 \
    DEMO_MODE=1

EXPOSE 8000

# Use ${PORT} if provided by platform (Render/Heroku), else fallback to 8000
CMD ["sh", "-c", "uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000}"]
