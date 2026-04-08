#!/usr/bin/env python3
"""Run naive sampling baseline experiments for each model.

Split into two phases per experiment:
  Phase 1 (generate): --generate_only   → produces prompts/
  Phase 2 (judge):    --judge_only      → produces results.jsonl
If the output of a phase already exists, it is skipped.
"""

import argparse
import os
from pyexpat import model
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / ".." / ".." / "results"
OUTPUT_DIR = SCRIPT_DIR / "baseline_results"

EXPERIMENTS = [
    # {
    #     "folder": "merged_20260331-235438_single_turn_vicuna-7b-v1.5_hf_p0.6_k-1_t0.9_lh0.0001_cs1_greedy_cacheon_sbufon_jbufon_topkon_refPaton_refJudgeon",
    #     "model": "lmsys/vicuna-7b-v1.5",
    #     "top_p": "0.6",
    #     "top_k": "-1",
    #     "temperature": "0.9",
    # },

    #    { 
    #         "folder": "merge_llama2_7b",
    #         "model": "meta-llama/Llama-2-7b-chat-hf",
    #         "top_p": "0.9",
    #         "top_k": "-1",
    #         "temperature": "0.6",
    #    },
    #     {

    #         "folder": "merged_qwen3_8b",
    #         "model": "Qwen/Qwen3-8B",
    #         "top_p": "0.95",
    #         "top_k": "20",
    #         "temperature": "0.6",
    #    }

       {

            "folder": "merged_single_turn_gemma-3-4b-it_hf_p0.95_k64_t1_lh0.0001_cs1_greedy_cacheon_sbufon_jbufon_topkon",
            "model": "google/gemma-3-4b-it",
            "top_p": "0.95",
            "top_k": "64",
            "temperature": "1",
       }




    #     {
    #     "folder": "/home/shuyi/BOA/results/merged_Llama-3.1-8B-Instruct",
    #     "model": "meta-llama/Llama-3.1-8B-Instruct",
    #     "top_p": "0.9",
    #     "top_k": "-1",
    #     "temperature": "0.6",
    # },
    # {
    #     "folder": "merged_all_Qwen2.5-32B-Instruct-AWQ_hf_p0.95_k20_t0.6_lh0.0001_cs1_cacheon_sbufon_jbufon_topkon_refPaton_refJudgeon",
    #     "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    #     "top_p": "0.95",
    #     "top_k": "20",
    #     "temperature": "0.6",
    # },
    # {
    #     "folder": "merged_all_Qwen2.5-72B-Instruct-AWQ_hf_p0.95_k20_t0.6_lh0.0001_cs1_cacheon_sbufon_jbufon_topkon_refPaton_refJudgeon",
    #     "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    #     "top_p": "0.95",
    #     "top_k": "20",
    #     "temperature": "0.6",
    # },
]

PROMPT_END = 128
GPU_MEM_UTIL = "0.7"
MAX_MODEL_LEN = "1024"


def missing_generated_prompts(exp_dir: Path) -> list[int]:
    """Return sorted list of prompt indices (1..PROMPT_END) with no prompts/ file."""
    prompts_dir = exp_dir / "prompts"
    if not prompts_dir.is_dir():
        return list(range(1, PROMPT_END + 1))
    existing = set()
    for p in prompts_dir.glob("prompt_*.jsonl"):
        try:
            existing.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(set(range(1, PROMPT_END + 1)) - existing)


def missing_judged_prompts(exp_dir: Path) -> list[int]:
    """Return sorted list of prompt indices (1..PROMPT_END) missing from results.jsonl."""
    results = exp_dir / "results.jsonl"
    if not results.exists():
        return list(range(1, PROMPT_END + 1))
    import json
    existing = set()
    with open(results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("total_samples", 0) > 0 or rec.get("prompt", ""):
                    existing.add(rec["prompt_index"])
            except Exception:
                pass
    return sorted(set(range(1, PROMPT_END + 1)) - existing)


def clean_empty_results(exp_dir: Path, resume_from: int) -> None:
    """Remove empty (total_samples=0) entries for prompts >= resume_from."""
    import json
    results = exp_dir / "results.jsonl"
    if not results.exists():
        return
    lines = results.read_text().splitlines()
    kept = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec["prompt_index"] >= resume_from and rec.get("total_samples", 0) == 0 and not rec.get("prompt", ""):
                removed += 1
                continue
        except Exception:
            pass
        kept.append(line)
    if removed:
        results.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"  Cleaned {removed} empty entries from results.jsonl")


def _remove_judged_entries(exp_dir: Path, indices: set[int]) -> None:
    """Remove results.jsonl entries for the given prompt indices so they can be re-judged."""
    import json
    results = exp_dir / "results.jsonl"
    if not results.exists():
        return
    lines = results.read_text().splitlines()
    kept = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec["prompt_index"] in indices:
                removed += 1
                continue
        except Exception:
            pass
        kept.append(line)
    if removed:
        results.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"  Removed {removed} existing entries from results.jsonl for re-judging")


def make_tag(folder: str) -> str:
    tag = re.sub(r"\d[\d_-]*single_turn_", "", folder)
    tag = re.sub(r"^merged_all_", "", tag)
    return tag


def run(cmd: list[str], label: str, env: dict):
    print(f"\n{'='*60}")
    print(f"[{label}] Starting ...")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"[{label}] Done.")
    except subprocess.CalledProcessError as e:
        print(f"[{label}] FAILED (exit code {e.returncode}), continuing...",
              file=sys.stderr)
    except Exception as e:
        print(f"[{label}] FAILED ({e}), continuing...", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=2, help="Physical GPU index")
    parser.add_argument("--prompt_indices", type=str, default=None,
                        help="Comma-separated 1-based prompt indices to process (e.g. '1,2,5,10')")
    parser.add_argument("--prompt_start", type=int, default=None,
                        help="1-based start of prompt range (inclusive)")
    parser.add_argument("--prompt_end", type=int, default=None,
                        help="1-based end of prompt range (inclusive, default: PROMPT_END)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run specified prompts even if results already exist")
    args = parser.parse_args()

    # Determine which prompts to process
    if args.prompt_indices:
        subset = sorted(int(x) for x in args.prompt_indices.split(","))
    elif args.prompt_start is not None:
        end = args.prompt_end if args.prompt_end is not None else PROMPT_END
        subset = list(range(args.prompt_start, end + 1))
    else:
        subset = None  # all prompts (use PROMPT_END)

    subset_set = set(subset) if subset is not None else None
    effective_end = max(subset) if subset else PROMPT_END

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.cuda)}
    dev = "0"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = []  # (tag, exp_dir, base_cmd)

    for exp in EXPERIMENTS:
        tag = make_tag(exp["folder"])
        runs_jsonl = str(RESULTS_ROOT / exp["folder"] / "runs.jsonl")
        common_sampling = [
            "--temperature", exp["temperature"],
            "--top_p", exp["top_p"],
            "--top_k", exp["top_k"],
        ]

        print(f"\n{'#'*60}")
        print(f"# {tag}")
        print(f"{'#'*60}")

        exp_dir = OUTPUT_DIR / f"naive_{tag}"
        base_cmd = [
            sys.executable, str(SCRIPT_DIR / "naive_sampling.py"),
            "--model", exp["model"],
            "--model_cuda", dev,
            *common_sampling,
            "--default_budget", "250000",
            "--max_new_tokens", "512",
            "--samples_per_call", "50",
            "--runs_jsonl", runs_jsonl,
            "--prompt_end", str(effective_end),
            "--judger_cuda", dev,
            "--judger_batch_size", "512",
            "--output_dir", str(exp_dir),
        ]
        all_tasks.append((tag, exp_dir, base_cmd))

    # ── Phase 1: all generation ──────────────────────────────────────
    print(f"\n{'#'*60}")
    print("# Phase 1: Generation")
    print(f"{'#'*60}")
    for tag, exp_dir, base_cmd in all_tasks:
        if args.force and subset is not None:
            to_gen = list(subset)
            print(f"[naive gen  | {tag}] FORCE re-generating {len(to_gen)} prompts: {to_gen}")
        else:
            to_gen = missing_generated_prompts(exp_dir)
            if subset is not None:
                to_gen = [i for i in to_gen if i in subset_set]
        if not to_gen:
            print(f"[naive gen  | {tag}] SKIP — all requested prompts already generated")
        else:
            run(base_cmd + ["--generate_only", "--prompt_indices", ",".join(str(i) for i in to_gen)],
                label=f"naive gen | {tag}", env=env)

    # ── Phase 2: all judging ─────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("# Phase 2: Judging")
    print(f"{'#'*60}")
    for tag, exp_dir, base_cmd in all_tasks:
        if args.force and subset is not None:
            to_judge = list(subset)
            # Remove old results for these prompts so they get re-judged
            _remove_judged_entries(exp_dir, subset_set)
            print(f"[naive judge | {tag}] FORCE re-judging {len(to_judge)} prompts: {to_judge}")
        else:
            to_judge = missing_judged_prompts(exp_dir)
            if subset is not None:
                to_judge = [i for i in to_judge if i in subset_set]
        if not to_judge:
            print(f"[naive judge | {tag}] SKIP — all requested prompts already judged")
        else:
            clean_empty_results(exp_dir, min(to_judge))
            run(base_cmd + ["--judge_only", "--samples_dir", str(exp_dir),
                            "--prompt_indices", ",".join(str(i) for i in to_judge)],
                label=f"naive judge | {tag}", env=env)

    print(f"\nAll experiments completed. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
