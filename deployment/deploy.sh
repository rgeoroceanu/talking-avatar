#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

PROJECT_ID=$(gcloud config get-value project)
REGION=${1:-europe-west4}

echo "Deploying musetalk-live to Cloud Run"
echo "  Project: $PROJECT_ID"
echo "  Region:  $REGION"
echo ""

# Build and push container image
gcloud builds submit "$ROOT_DIR" \
    --config "$SCRIPT_DIR/cloudbuild.yaml" \
    --substitutions "_IMAGE=gcr.io/$PROJECT_ID/musetalk-live"

# Deploy to Cloud Run with L4 GPU
gcloud run deploy musetalk-live \
  --image gcr.io/$PROJECT_ID/musetalk-live \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi \
  --max-instances 1 --timeout 300 \
  --port 8080 --region $REGION \
  --allow-unauthenticated
