#!/usr/bin/env bash
# Run the two-tower serving API locally.
# Usage:  bash run_serving.sh            # reads host/port from serving/config.yaml
#         RECSYS_PORT=9000 bash run_serving.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f artifacts/twotower.pt ] || [ ! -f artifacts/dataset.pt ]; then
  echo "ERROR: artifacts/twotower.pt and artifacts/dataset.pt are required." >&2
  echo "Run 'python train.py' first (this script never retrains)." >&2
  exit 1
fi

# Load host/port from config (env RECSYS_HOST/RECSYS_PORT override via uvicorn args below).
HOST="${RECSYS_HOST:-}"
PORT="${RECSYS_PORT:-}"

exec python -m uvicorn serving.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1
