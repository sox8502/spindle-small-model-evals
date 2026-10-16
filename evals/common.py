"""Shared sampling and output helpers for evals against a Spindle (Tinker-compatible) server."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import tinker
from tinker import types


def parser(*, max_tokens: int) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model", required=True, help="Base model id, e.g. Qwen/Qwen3.5-2B")
    result.add_argument("--limit", type=int, default=None, help="Evaluate the first N problems")
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--max-tokens", type=int, default=max_tokens)
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--thinking", action="store_true", help="Enable the chat template's thinking mode")
    result.add_argument("--out", type=Path, default=Path("results"))
    return result


async def generate(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    """Sample one chat completion per user prompt, returning text after any thinking block."""
    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"], api_key=os.environ["TINKER_API_KEY"]
    )
    sampler = service.create_sampling_client(base_model=args.model)
    tokenizer = sampler.get_tokenizer()
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    gate = asyncio.Semaphore(args.concurrency)

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
        text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
        return text.split("</think>")[-1].strip()

    return await asyncio.gather(*(complete(prompt) for prompt in prompts))


def write_results(args: argparse.Namespace, task: str, rows: list[dict], metrics: dict) -> None:
    summary = {
        "task": task,
        "model": args.model,
        "n": len(rows),
        **metrics,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{args.model.replace('/', '__')}"
    (args.out / f"{stem}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
