"""Tinker list prices (USD per 1M tokens), from https://tinker-docs.thinkingmachines.ai/tinker/models/ on 2026-10-16."""

# (prefill, sample); uncached prefill since IFEval prompts are unique. Tinker does not serve Qwen3.5-2B.
TINKER_PRICES = {
    "Qwen/Qwen3.5-4B": (0.33, 1.005),
    "Qwen/Qwen3.5-9B": (0.30, 0.807),
    "Qwen/Qwen3-8B": (0.195, 0.60),
    # Limited-time discounted rate (list: 0.368, 0.99).
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.495),
}


def tinker_list_cost(model: str, prompt_tokens: int, sample_tokens: int) -> float | None:
    if model not in TINKER_PRICES:
        return None
    prefill, sample = TINKER_PRICES[model]
    return round((prompt_tokens * prefill + sample_tokens * sample) / 1e6, 6)
