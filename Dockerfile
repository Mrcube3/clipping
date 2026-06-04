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
    curl \
    ca-certificates \
    unzip \
    && update-ca-certificates --fresh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

# Install deno (JS runtime required by yt-dlp for YouTube extraction)
RUN arch=$(uname -m); \
    case "$arch" in \
        x86_64|amd64) suffix="x86_64-unknown-linux-gnu" ;; \
        aarch64|arm64) suffix="aarch64-unknown-linux-gnu" ;; \
        *) echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/denoland/deno/releases/latest/download/deno-${suffix}.zip" -o /tmp/deno.zip && \
    unzip -o /tmp/deno.zip -d /usr/local/bin/ && \
    chmod +x /usr/local/bin/deno && \
    rm /tmp/deno.zip

COPY smart_reframe.py .

EXPOSE 7860

CMD ["uvicorn", "smart_reframe:app", "--host", "0.0.0.0", "--port", "7860"]
