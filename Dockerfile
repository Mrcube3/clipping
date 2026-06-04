# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# Install system deps: FFmpeg, OpenCV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download latest yt-dlp binary for the build architecture
RUN mkdir -p /app/bin && \
    arch=$(uname -m); \
    case "$arch" in \
        x86_64|amd64) suffix="_linux" ;; \
        aarch64|arm64) suffix="_linux_aarch64" ;; \
        *) echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    wget -q -O /app/bin/yt-dlp "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp${suffix}" && \
    chmod +x /app/bin/yt-dlp

COPY smart_reframe.py .

EXPOSE 7860

CMD ["uvicorn", "smart_reframe:app", "--host", "0.0.0.0", "--port", "7860"]
