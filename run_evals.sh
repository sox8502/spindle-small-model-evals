#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
uv sync --locked
for model in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; do
  uv run --locked python -m evals.ifeval --backend tinker --model "$model"
  uv run --locked python -m evals.math500 --backend tinker --model "$model"
done
