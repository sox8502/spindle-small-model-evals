#!/usr/bin/env bash
set -euo pipefail
if command -v caffeinate >/dev/null; then caffeinate -dimsu -w $$ & fi
export PYTHONUNBUFFERED=1
export MODAL_ENVIRONMENT=main
uv sync --locked
trap 'uv run --locked python -m evals.controlled --cleanup-only' EXIT
export SPINDLE_API_KEY="$(uv run --locked python -c 'import secrets; print("tml-spindle-" + secrets.token_urlsafe(32))')"
uv run --locked python - <<'SETUP'
import os
import modal
modal.Secret.objects.create("spindle-ctlg1103-api", {"TINKER_API_KEY": os.environ["SPINDLE_API_KEY"]}, allow_existing=True, environment_name="main")
modal.Secret.from_name("spindle-ctlg1103-api", environment_name="main").update({"TINKER_API_KEY": os.environ["SPINDLE_API_KEY"]})
SETUP
uv run --locked spindle deploy --env main spindle_configs/qwen35_4b.py spindle_configs/qwen35_9b.py spindle_configs/nemotron35_lightning.py
export SPINDLE_BASE_URL="$(uv run --locked python - <<'URL'
import modal
print(modal.Function.from_name("spindle-ctlg1103", "server", environment_name="main").get_web_url())
URL
)"
uv run --locked python -m evals.controlled
