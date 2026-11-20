#!/bin/bash
set -e

echo "RetailLab API starting..."
echo "Model directory: ${MODEL_DIR:-/app/models}"
echo "Log level: ${LOG_LEVEL:-INFO}"

# Ensure directories exist
mkdir -p /app/logs/decisions /app/models /app/data

# Start the API server
exec uvicorn api_gateway.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS:-4}" \
    --log-level "${LOG_LEVEL:-info}"
