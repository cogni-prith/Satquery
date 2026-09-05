#!/bin/bash
# Start the backend. Models load in the background; watch /api/health for models_loaded.
set -e
export SATQUERY_DATA_ROOT="${SATQUERY_DATA_ROOT:-/media/cyborg-prithwish/Expansion/satquery-data}"
export SATQUERY_ARTIFACT_ROOT="${SATQUERY_ARTIFACT_ROOT:-/media/cyborg-prithwish/Expansion/satquery-artifacts}"
# Which satquery package to serve. v1 is the VLM-first repo, v2 the specialist-first
# rewrite; this service binds whichever one is on PYTHONPATH.
ML_ROOT="${SATQUERY_ML_ROOT:-$HOME/PycharmProjects/SIH_2026_SatQuery}"
# Both repos vendor EarthDial's Phi-3, which breaks on transformers v5's generation
# rewrite. .compat/ holds 4.49 and goes ahead of site-packages so tf-torch stays untouched.
export PYTHONPATH="$HOME/PycharmProjects/SIH_2026_SatQuery/.compat:$ML_ROOT/src:${PYTHONPATH:-}"
# v2 loads the 4 GB backbone only when asked, so an import never costs a demo its VRAM.
export SATQUERY_LOAD_VLM="${SATQUERY_LOAD_VLM:-1}"
export SATQUERY_DEMO_DIR="${SATQUERY_DEMO_DIR:-$ML_ROOT/demo_inputs}"
export SATQUERY_MODELS="${SATQUERY_MODELS:-/media/cyborg-prithwish/Expansion/satquery-models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"
exec ~/miniconda3/envs/tf-torch/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
