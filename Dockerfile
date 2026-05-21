# spectra/Dockerfile
# Production container for SPECTRA inference API
# Base: PyTorch + CUDA 11.8

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyG (separate due to CUDA-specific wheels)
RUN pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-sparse  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-geometric

# Copy source
COPY . .

# Create checkpoint dir
RUN mkdir -p checkpoints

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["python", "api/server.py"]
