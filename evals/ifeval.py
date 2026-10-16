"""IFEval (google/IFEval) scored with the reference instruction checkers vendored in lm-eval."""

import asyncio

from datasets import load_dataset
from lm_eval.tasks.ifeval.utils import process_results

from evals.common import generate, parser, write_results


async def main() -> None:
    args = parser(max_tokens=1280).parse_args()
    problems = load_dataset("google/IFEval", split="train")
    if args.limit is not None:
        problems = problems.select(range(min(args.limit, len(problems))))

    completions, usage = await generate(args, [p["prompt"] for p in problems])

    rows = [
        {
            "key": problem["key"],
            "instruction_id_list": problem["instruction_id_list"],
            **process_results(problem, [completion]),
            "completion": completion,
        }
        for problem, completion in zip(problems, completions)
    ]
    metrics = {
        f"prompt_level_{mode}_acc": sum(r[f"prompt_level_{mode}_acc"] for r in rows) / len(rows)
        for mode in ("strict", "loose")
    } | {
        f"inst_level_{mode}_acc": sum(sum(r[f"inst_level_{mode}_acc"]) for r in rows)
        / sum(len(r[f"inst_level_{mode}_acc"]) for r in rows)
        for mode in ("strict", "loose")
    }
    write_results(args, "ifeval", rows, metrics, usage)


if __name__ == "__main__":
    asyncio.run(main())
