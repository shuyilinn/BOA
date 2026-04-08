#!/usr/bin/env python3
"""
Measure average refusal length (in tokens) for each model.

For each of the 128 benchmark prompts, generate one greedy response
and record its token count. No judger is used.

Usage:
  python measure_refusal_length.py --model meta-llama/Llama-3.1-8B-Instruct --cuda 0
  python measure_refusal_length.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --cuda 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import time

import torch

# Repo root (must be on sys.path before any repo imports)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Fix AWQ + transformers compatibility (PytorchGELUTanh rename)
from engines.hf_engine import _ensure_awq_transformers_compat
_ensure_awq_transformers_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse chat template from naive_sampling
from experiment.baseline.naive_sampling import apply_chat_template, load_benchmark_prompts


def main():
    parser = argparse.ArgumentParser(description="Measure average refusal token length")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--prompt_end", type=int, default=128, help="Number of prompts to process")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load prompts
    all_prompts = load_benchmark_prompts()
    prompts = all_prompts[: args.prompt_end]

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()

    # Generate one greedy response per prompt
    token_lengths = []
    for i, raw_prompt in enumerate(prompts, 1):
        prompt_str = apply_chat_template(raw_prompt, tokenizer, model_name=args.model)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][prompt_len:]
        n_tokens = len(new_tokens)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        token_lengths.append(n_tokens)

        print(f"[{i:3d}/{len(prompts)}] {n_tokens:4d} tokens | {text[:80]}...")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {len(token_lengths)}")
    print(f"{'='*60}")
    print(f"  Mean token length:   {statistics.mean(token_lengths):.1f}")
    print(f"  Median token length: {statistics.median(token_lengths):.1f}")
    print(f"  Stdev:               {statistics.stdev(token_lengths):.1f}")
    print(f"  Min:                 {min(token_lengths)}")
    print(f"  Max:                 {max(token_lengths)}")
    print(f"{'='*60}")

    # Save results
    out_path = os.path.join(
        os.path.dirname(__file__),
        f"refusal_lengths_{args.model.replace('/', '_')}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "prompt_count": len(token_lengths),
            "mean": statistics.mean(token_lengths),
            "median": statistics.median(token_lengths),
            "stdev": statistics.stdev(token_lengths),
            "min": min(token_lengths),
            "max": max(token_lengths),
            "lengths": token_lengths,
        }, f, indent=2)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
