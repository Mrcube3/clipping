#!/bin/bash
# start.sh — Smart Reframe API
# Installs deps on first run, starts the server.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Install dependencies if missing
pip3 install --user -r requirements.txt 2>/dev/null || true

echo "Starting Smart Reframe API at http://localhost:8000"
echo "Open that URL in your browser to test."
echo ""
exec python3 -m uvicorn smart_reframe:app --host 0.0.0.0 --port 8000 --reload
