#!/usr/bin/env bash
set -euo pipefail
# SPINDLE_BASE_URL / SPINDLE_API_KEY (and TINKER_API_KEY for hosted Tinker) live outside the repo.
set -a; . "$HOME/.config/spindle-evals/env"; set +a

for model in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; do
  uv run --locked python -m evals.math500 --backend tinker --model "$model"
done
