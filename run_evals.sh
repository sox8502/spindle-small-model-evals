#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MODAL_ENVIRONMENT=main
uv sync --locked
export SPINDLE_API_KEY="$(uv run --locked python -c 'import secrets; print("tml-spindle-" + secrets.token_urlsafe(32))')"
uv run --locked python - <<'SETUP'
import os
import modal
modal.Secret.objects.create("spindle-rerun1103-api", {"TINKER_API_KEY": os.environ["SPINDLE_API_KEY"]}, allow_existing=True, environment_name="main")
modal.Secret.from_name("spindle-rerun1103-api", environment_name="main").update({"TINKER_API_KEY": os.environ["SPINDLE_API_KEY"]})
SETUP
uv run --locked spindle deploy --env main spindle_configs/qwen35_4b.py spindle_configs/qwen35_9b.py spindle_configs/nemotron35_lightning.py
export SPINDLE_BASE_URL="$(uv run --locked python - <<'URL'
import modal
print(modal.Function.from_name("spindle-rerun1103", "server", environment_name="main").get_web_url())
URL
)"
for model in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; do
  uv run --locked python -m evals.ifeval --backend spindle --model "$model"
  uv run --locked python -m evals.math500 --backend spindle --model "$model"
done
