#!/bin/bash
# Start the backend. Models load in the background; watch /api/health for models_loaded.
set -e
export SATQUERY_DATA_ROOT="${SATQUERY_DATA_ROOT:-/media/cyborg-prithwish/Expansion/satquery-data}"
export SATQUERY_ARTIFACT_ROOT="${SATQUERY_ARTIFACT_ROOT:-/media/cyborg-prithwish/Expansion/satquery-artifacts}"
export PYTHONPATH="${PYTHONPATH:-}:$HOME/PycharmProjects/SIH_2026_SatQuery/src"
cd "$(dirname "$0")"
exec ~/miniconda3/envs/tf-torch/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
