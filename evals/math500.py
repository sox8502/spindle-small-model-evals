"""MATH-500 (HuggingFaceH4/MATH-500) with k samples per problem, scored by math_verify."""

import asyncio

from datasets import load_dataset
from math_verify import parse, verify

from evals.common import generate, parser, write_results

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def is_correct(gold: str, completion: str) -> bool:
    return verify(parse(f"${gold}$"), parse(completion))


async def main() -> None:
    args = parser(max_tokens=2048, temperature=0.6, num_samples=4).parse_args()
    problems = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit is not None:
        problems = problems.select(range(min(args.limit, len(problems))))

    completions, usage = await generate(args, [f"{p['problem']}\n\n{INSTRUCTION}" for p in problems])

    rows = []
    for problem, samples in zip(problems, completions):
        correct = [is_correct(problem["answer"], sample) for sample in samples]
        rows.append(
            {
                "unique_id": problem["unique_id"],
                "subject": problem["subject"],
                "level": problem["level"],
                "gold": problem["answer"],
                "correct": correct,
                "completions": samples,
            }
        )
    metrics = {
        f"mean_acc@{args.num_samples}": sum(sum(r["correct"]) for r in rows)
        / sum(len(r["correct"]) for r in rows),
        f"pass@{args.num_samples}": sum(any(r["correct"]) for r in rows) / len(rows),
    }
    write_results(args, "math500", rows, metrics, usage)


if __name__ == "__main__":
    asyncio.run(main())
