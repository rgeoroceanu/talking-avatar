#!/bin/bash
set -e

echo "=========================================="
echo "  MuseTalk Live Talking Head Server"
echo "=========================================="

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo ""
echo "Starting server on port ${PORT:-8080}..."
echo ""

# MuseTalk preprocessing.py uses relative paths like ./musetalk/utils/dwpose/...
# so we must run from the MuseTalk repo directory
cd /app/MuseTalk

exec uvicorn server:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1 \
    --log-level info
