"""Shared sampling and output helpers for evals against Spindle or hosted Tinker (same Tinker SDK)."""

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import tinker
from tinker import types

from evals.pricing import tinker_list_cost


def parser(*, max_tokens: int) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--backend", required=True, choices=["spindle", "tinker"])
    result.add_argument("--model", required=True, help="Base model id, e.g. Qwen/Qwen3.5-2B")
    result.add_argument("--limit", type=int, default=None, help="Evaluate the first N problems")
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--max-tokens", type=int, default=max_tokens)
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--thinking", action="store_true", help="Enable the chat template's thinking mode")
    result.add_argument("--out", type=Path, default=Path("results"))
    return result


async def generate(args: argparse.Namespace, prompts: list[str]) -> tuple[list[str], dict]:
    """Sample one chat completion per user prompt; return texts after any thinking block, plus usage."""
    if args.backend == "spindle":
        base_url, api_key = os.environ["SPINDLE_BASE_URL"], os.environ["SPINDLE_API_KEY"]
    else:
        # None selects the SDK's hosted Tinker endpoint.
        base_url, api_key = None, os.environ["TINKER_API_KEY"]
    service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
    sampler = service.create_sampling_client(base_model=args.model)
    tokenizer = sampler.get_tokenizer()
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    gate = asyncio.Semaphore(args.concurrency)
    usage = {"backend": args.backend, "base_url": base_url, "prompt_tokens": 0, "sample_tokens": 0}

    async def complete(prompt: str) -> str:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=args.thinking,
            tokenize=True,
        )["input_ids"]
        async with gate:
            response = await sampler.sample_async(
                prompt=types.ModelInput.from_ints(tokens), num_samples=1, sampling_params=params
            )
        sampled = response.sequences[0].tokens
        # Approximates cold start plus queueing; the rest of wall_s is sampling.
        usage.setdefault("first_result_s", round(time.monotonic() - start, 1))
        usage["prompt_tokens"] += len(tokens)
        usage["sample_tokens"] += len(sampled)
        text = tokenizer.decode(sampled, skip_special_tokens=True)
        return text.split("</think>")[-1].strip()

    usage["started_at"] = datetime.now(UTC).isoformat()
    start = time.monotonic()
    completions = await asyncio.gather(*(complete(prompt) for prompt in prompts))
    usage["wall_s"] = round(time.monotonic() - start, 1)
    usage["finished_at"] = datetime.now(UTC).isoformat()
    usage["tinker_list_cost_usd"] = tinker_list_cost(
        args.model, usage["prompt_tokens"], usage["sample_tokens"]
    )
    return completions, usage


def write_results(
    args: argparse.Namespace, task: str, rows: list[dict], metrics: dict, usage: dict
) -> None:
    summary = {
        "task": task,
        "model": args.model,
        "n": len(rows),
        **metrics,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        **usage,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{args.model.replace('/', '__')}"
    (args.out / f"{stem}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
