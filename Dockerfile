# spectra/Dockerfile
# Railway deployment container (CPU inference)
# For GPU deployment, swap the base for pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# and reinstate the torch-geometric install block.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for OpenCV / Pillow + git for torch.hub (DINOv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (keeps image small on Railway free tier)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 torchvision==0.16.0

# Remaining Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Create checkpoint dir (weights are NOT in the image — see TODO below)
RUN mkdir -p weights checkpoints

# Railway injects $PORT; default to 8080 for local docker runs
ENV PORT=8080
EXPOSE 8080

# Health check (Railway also probes /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use shell form so $PORT expands at runtime
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT}
