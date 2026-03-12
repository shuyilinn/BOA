#!/usr/bin/env python3
"""Generate BOA harmful benchmark from JBB and HarmBench."""

import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

RANDOM_SEED = 2026
OUTPUT_PATH = Path(__file__).resolve().parent / "boa_benchmark" / "jailbreak_oracle_benchmark.json"


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def main() -> None:
    jbb_samples = list(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"))
    harmbench_train = load_dataset("walledai/HarmBench", "standard", split="train")
    harmbench_samples = [s for s in harmbench_train if s["category"] == "chemical_biological"]

    samples = []
    seen_prompts = set()

    for s in jbb_samples:
        category = s["Category"]

        prompt = normalize(s["Goal"])
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        samples.append(
            {
                "id": len(samples) + 1,
                "dataset": "JailbreakBench",
                "prompt": prompt,
                "original_prompt": prompt,
                "target": s["Target"],
                "meta": {
                    "patch": None,
                    "method": "Direct",
                    "category": category,
                },
            }
        )

    for s in harmbench_samples:
        prompt = normalize(s["prompt"])
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        samples.append(
            {
                "id": len(samples) + 1,
                "dataset": "HarmBench",
                "prompt": prompt,
                "original_prompt": prompt,
                "target": "",
                "meta": {
                    "patch": None,
                    "method": "Direct",
                    "category": s["category"],
                },
            }
        )

    jbb_count = sum(1 for s in samples if s["dataset"] == "JailbreakBench")
    harmbench_count = sum(1 for s in samples if s["dataset"] == "HarmBench")

    benchmark = {
        "metadata": {
            "name": "Jailbreak Oracle Benchmark (Converted)",
            "version": "1.0",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "random_seed": RANDOM_SEED,
            "total_samples": len(samples),
            "jbb_samples": jbb_count,
            "harmbench_samples": harmbench_count,
            "duplicates_removed": (len(jbb_samples) + len(harmbench_samples)) - len(samples),
        },
        "samples": samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"Saved benchmark to: {OUTPUT_PATH}")
    print(f"Total={len(samples)} | JailbreakBench={jbb_count} | HarmBench={harmbench_count}")


if __name__ == "__main__":
    main()
