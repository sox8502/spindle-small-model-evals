#!/usr/bin/env bash
set -euo pipefail
# TINKER_BASE_URL / TINKER_API_KEY live outside the repo.
set -a; . "$HOME/.config/spindle-evals/env"; set +a

for model in Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B; do
  uv run --locked python -m evals.ifeval --model "$model"
done
