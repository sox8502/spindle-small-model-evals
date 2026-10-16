# Spindle evals

Evaluate small open models by sampling through a self-hosted
[Spindle](https://modal.com/docs/guide/spindle) server on Modal. The server speaks the Tinker API, so eval scripts
are plain Tinker SDK clients; GPUs are only allocated on Modal while sampling.

Spindle is pinned in `pyproject.toml` to commit `14d2379` of
[modal-projects/spindle](https://github.com/modal-projects/spindle). It requires Python 3.12.

## Layout

| Path | Purpose |
| --- | --- |
| `spindle_configs/` | Deploy configs (thin subclasses of Spindle's bundled Qwen3.5 2B / 4B recipes) |
| `evals/common.py` | Shared sampling via `create_sampling_client(base_model=...)`, chat templating, result writing |
| `evals/ifeval.py` | IFEval strict/loose prompt- and instruction-level accuracy (541 prompts), using lm-eval's reference checkers |
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

Save the `server` URL printed by the deploy. `run_evals.sh` reads both values from
`~/.config/spindle-evals/env` (keep it `chmod 600`, outside the repo):

```bash
TINKER_BASE_URL=https://<server>.modal.run
TINKER_API_KEY=tml-...
```

## Run an eval

`bash run_evals.sh` runs IFEval on every model in the script. For a single ad-hoc eval:

```bash
export TINKER_BASE_URL='https://<server>.modal.run'
export TINKER_API_KEY='<spindle api key>'

uv run --locked python -m evals.ifeval --model Qwen/Qwen3.5-2B --limit 20   # smoke test
uv run --locked python -m evals.ifeval --model Qwen/Qwen3.5-4B              # all 541 prompts
```

Evals accept `--limit`, `--concurrency`, `--max-tokens`, `--temperature` (default greedy) and `--thinking`
(enables the Qwen chat template's thinking mode; the thinking block is stripped before scoring).
IFEval downloads NLTK's `punkt_tab` on first import.

The first request on a cold pool waits for an H100 sampler replica to start and load the model; Tinker retries
the HTTP 408s until it is ready. Replicas scale to zero after 5 idle minutes.

## Adding models or tasks

- **Model:** `uv run --locked spindle config init --preset <bundled_config_name>` prints a subclass to save in
  `spindle_configs/`; redeploy with all config files.
- **Task:** add a module under `evals/` that builds prompts, calls `evals.common.generate`, scores, and calls `write_results`.

## Teardown

`modal app list`, then `modal app stop` the `spindle-fft-*` sampler apps first and the `spindle-evals` frontend last.
