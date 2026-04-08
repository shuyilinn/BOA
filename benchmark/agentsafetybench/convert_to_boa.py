from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.agent_data_loader import adapt_agent_safetybench_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Agent-SafetyBench data to BOA format.")
    parser.add_argument(
        "--input",
        type=str,
        default="./benchmark/agentsafetybench/released_data.json",
        help="Path to Agent-SafetyBench released_data.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark/agentsafetybench/released_data_for_boa.json",
        help="Path to write converted BOA json",
    )
    parser.add_argument(
        "--env_dir",
        type=str,
        default=None,
        help="Directory containing Agent-SafetyBench environment *.json schema files",
    )
    parser.add_argument(
        "--sys_prompt_variant",
        type=str,
        default="basic",
        help="Function-calling system prompt variant (e.g. basic, without_strict_jsonformat, with_failure_modes)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Agent-SafetyBench input should be a top-level list of samples.")

    converted = adapt_agent_safetybench_samples(
        data,
        env_dir=args.env_dir,
        sys_prompt_variant=args.sys_prompt_variant,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "name": "Agent-SafetyBench for BOA",
                    "total_samples": len(converted["samples"]),
                    "sys_prompt_variant": args.sys_prompt_variant,
                },
                "samples": converted["samples"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Converted {len(converted['samples'])} samples to {output_path}")


if __name__ == "__main__":
    main()
