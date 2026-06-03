# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# Install system deps: FFmpeg, OpenCV libs, build tools for MediaPipe wheels
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

COPY smart_reframe.py .

EXPOSE 7860

CMD ["uvicorn", "smart_reframe:app", "--host", "0.0.0.0", "--port", "7860"]
