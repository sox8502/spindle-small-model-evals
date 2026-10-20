# Spindle evals

Evaluate small open models by sampling through a self-hosted
[Spindle](https://modal.com/docs/guide/spindle) server on Modal. The server speaks the Tinker API, so eval scripts
are plain Tinker SDK clients; GPUs are only allocated on Modal while sampling.

Spindle is pinned in `pyproject.toml` to commit `9ad4123` of
[modal-projects/spindle](https://github.com/modal-projects/spindle). It requires Python 3.12. The Tinker SDK is pinned to 0.33.x, which both Spindle and hosted Tinker accept.

## Layout

| Path | Purpose |
| --- | --- |
| `spindle_configs/` | Deploy configs (thin subclasses of Spindle's bundled Qwen3.5 2B / 4B recipes) |
| `evals/common.py` | Shared sampling via `create_sampling_client(base_model=...)`, chat templating, result writing |
| `evals/ifeval.py` | IFEval strict/loose prompt- and instruction-level accuracy (541 prompts), using lm-eval's reference checkers |
| `evals/pricing.py` | Tinker list prices used to price each run's tokens |
| `evals/spindle_cost.py` | Actual Modal spend of the `spindle-evals` apps, per config, from the hourly billing report |
| `results/` | Per-example JSONL and summary JSON (git-ignored) |

## One-time server setup

```bash
uv sync --locked
export MODAL_ENVIRONMENT=main

# The configs use their own `spindle-evals` frontend and `spindle-evals-api` secret.
export TINKER_API_KEY="tml-$(openssl rand -hex 32)"
modal secret create spindle-evals-api TINKER_API_KEY="$TINKER_API_KEY"

modal workspace proxy-tokens create          # prints Modal-Key / Modal-Secret
export MODAL_PROXY_TOKEN_ID='<Modal-Key>'
export MODAL_PROXY_TOKEN_SECRET='<Modal-Secret>'
modal workspace proxy-tokens allow "$MODAL_PROXY_TOKEN_ID" "$MODAL_ENVIRONMENT"
modal secret create spindle-proxy \
  MODAL_PROXY_TOKEN_ID="$MODAL_PROXY_TOKEN_ID" \
  MODAL_PROXY_TOKEN_SECRET="$MODAL_PROXY_TOKEN_SECRET"

# Always pass config files: a bare `spindle deploy` deploys every bundled model.
uv run --locked spindle deploy spindle_configs/*.py
```

Save the `server` URL printed by the deploy. `run_evals.sh` reads credentials from
`~/.config/spindle-evals/env` (keep it `chmod 600`, outside the repo). `TINKER_*` is reserved for hosted Tinker:

```bash
SPINDLE_BASE_URL=https://<server>.modal.run
SPINDLE_API_KEY=tml-...     # the key stored in spindle-evals-api
TINKER_API_KEY=tml-...      # optional: a hosted Tinker key from the Tinker console
```

## Run an eval

`bash run_evals.sh` runs IFEval on every model in the script. For a single ad-hoc eval:

```bash
set -a; . ~/.config/spindle-evals/env; set +a

uv run --locked python -m evals.ifeval --backend spindle --model Qwen/Qwen3.5-2B --limit 20   # smoke test
uv run --locked python -m evals.ifeval --backend spindle --model Qwen/Qwen3.5-4B              # all 541 prompts
uv run --locked python -m evals.ifeval --backend tinker  --model Qwen/Qwen3.5-4B              # hosted Tinker
```

`--backend` is required and is recorded in each summary. Evals also accept `--limit`, `--concurrency`, `--max-tokens`, `--temperature` (default greedy) and `--thinking`
(enables the Qwen chat template's thinking mode; the thinking block is stripped before scoring).
IFEval downloads NLTK's `punkt_tab` on first import.

The first request on a cold pool waits for an H100 sampler replica to start and load the model; Tinker retries
the HTTP 408s until it is ready. Replicas scale to zero after 5 idle minutes.

## Cost tracking: Spindle vs Tinker

Each eval summary records `prompt_tokens`, `sample_tokens`, `wall_s`, UTC `started_at`/`finished_at`, and
`tinker_list_cost_usd` (those tokens at Tinker's list price; `null` for models Tinker doesn't serve, e.g. Qwen3.5-2B).

Spindle's real cost is GPU time, including cold starts and the 5-minute idle scale-down. Modal billing is hourly and
can lag by about a day, so reconcile it afterwards for the run's window:

```bash
uv run --locked python -m evals.spindle_cost --since 2026-10-16T22:00:00 --until 2026-10-17T00:00:00
```

To measure hosted Tinker directly, run the same eval with `--backend tinker`.

## Adding models or tasks

- **Model:** `uv run --locked spindle config init --preset <bundled_config_name>` prints a subclass to save in
  `spindle_configs/`; redeploy with all config files.
- **Task:** add a module under `evals/` that builds prompts, calls `evals.common.generate`, scores, and calls `write_results`.

## Teardown

`modal app list`, then `modal app stop` the `spindle-fft-*` sampler apps first and the `spindle-evals` frontend last.
