#!/usr/bin/env bash
set -euo pipefail
# SPINDLE_BASE_URL / SPINDLE_API_KEY (and TINKER_API_KEY for hosted Tinker) live outside the repo.
set -a; . "$HOME/.config/spindle-evals/env"; set +a

# model, IFEval concurrency, MATH-500 concurrency: in-flight sequences fit one replica's SGLang running limit.
while read -r model ifeval_conc math_conc; do
  # Both benchmarks share one sampler session per model.
  uv run --locked python -m evals.ifeval --backend spindle --model "$model" --concurrency "$ifeval_conc"
  uv run --locked python -m evals.math500 --backend spindle --model "$model" --concurrency "$math_conc"
done <<'MODELS'
Qwen/Qwen3.5-4B 64 16
Qwen/Qwen3.5-9B 32 8
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 64 32
MODELS
