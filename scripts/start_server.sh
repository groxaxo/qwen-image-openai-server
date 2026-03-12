#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# Try common conda locations.
for CANDIDATE in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  if [ -f "$CANDIDATE" ]; then
    source "$CANDIDATE"
    break
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load conda first or edit this script for your env path." >&2
  exit 1
fi

conda activate qwenimg
exec python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1
