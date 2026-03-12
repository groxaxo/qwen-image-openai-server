#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

conda env create -f environment.yml || conda env update -f environment.yml --prune

echo
echo "Environment ready. Activate with: conda activate qwenimg"
