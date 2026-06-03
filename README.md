# Smart Reframe

Smart Reframe is a FastAPI service for extracting vertical 9:16 clips from horizontal video. It supports both uploaded video files and YouTube URLs, detects faces frame-by-frame, and renders vertical clips using FFmpeg.

## Features

- Upload local videos or process YouTube URLs
- Auto-detect faces and track speakers
- Render portrait (1080x1920) or square (1080x1080) clips
- Smart mode can auto-select engaging segments
- Built-in web UI at `/`
- Output served via `/api/v1/output/{job_id}/{filename}`

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the API:

   ```bash
   uvicorn smart_reframe:app --host 0.0.0.0 --port 8000
   ```

3. Open in browser:

   ```text
   http://localhost:8000
   ```

## API Endpoints

- `POST /api/v1/upload`
  - Upload a video file and receive an `upload_id`.
- `POST /api/v1/process`
  - Process clips using either `upload_id` or `youtube_url`.
  - Required form fields: `clips` (JSON array of `{start,end}` intervals).
  - Optional form fields: `smart`, `format`, `denoise`.
- `GET /api/v1/output/{job_id}/{filename}`
  - Download rendered clip files.
- `GET /api/v1/history`
  - List previous jobs and their output clips.
- `GET /api/v1/health`
  - Health check.
- `GET /api/v1/probe`
  - Probe uploaded video or YouTube URL for metadata.

## Example `process` payload

```json
{
  "upload_id": "<id>",
  "clips": [
    {"start": 12.0, "end": 32.0},
    {"start": 45.0, "end": 65.0}
  ],
  "format": "portrait",
  "denoise": "1"
}
```

## Notes

- The service requires `ffmpeg`; it will try to use the bundled binary or `imageio-ffmpeg`.
- Uploaded files are stored temporarily and cleaned up after processing.
- YouTube downloads use `yt-dlp`.

