#!/usr/bin/env python3
"""
Analyze BOA experiment results and compare across chunk sizes.

Usage:
    python analyze_results.py <result_dir> [<result_dir> ...] [--csv OUTPUT.csv]

Example:
    python analyze_results.py results/*cs1* results/*cs2* results/*cs3* results/*cs4*
    python analyze_results.py results/*cs* --csv chunk_comparison.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def load_dir(result_dir: Path) -> dict:
    """Load and aggregate stats from one result directory."""
    metadata_path = result_dir / "metadata.json"
    runs_path = result_dir / "runs.jsonl"

    if not runs_path.exists():
        print(f"[WARN] No runs.jsonl in {result_dir}, skipping.", file=sys.stderr)
        return None

    # --- metadata ---
    chunk_size = None
    run_id = result_dir.name
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        chunk_size = meta.get("config", {}).get("chunk_size")

    # --- runs.jsonl ---
    runs = []
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            runs.append(entry.get("stats", entry))

    if not runs:
        return None

    n = len(runs)
    n_unsafe = sum(1 for r in runs if r.get("jailbreak_found", False))
    n_unsafe_attack = sum(
        1 for r in runs
        if r.get("jailbreak_found") and r.get("exit_reason") == "unsafe_detected_attack_sampling"
    )
    n_unsafe_tree = sum(
        1 for r in runs
        if r.get("jailbreak_found") and r.get("exit_reason") == "unsafe_detected_tree_search"
    )
    total_tokens = sum(r.get("total_tokens_generated", 0) for r in runs)
    total_duration = sum(r.get("duration", 0.0) for r in runs)
    total_nodes = sum(r.get("evaluated_nodes_count", 0) for r in runs)

    cache_hit_rates = [
        r["cache"]["full_hit_rate_pct"]
        for r in runs
        if r.get("cache") and r["cache"].get("lookups", 0) > 0
    ]
    mean_cache_hit = sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else float("nan")

    tokens_per_sec = total_tokens / total_duration if total_duration > 0 else float("nan")

    exit_reasons = {}
    for r in runs:
        reason = r.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "run_id": run_id,
        "chunk_size": chunk_size,
        "n_runs": n,
        "n_unsafe": n_unsafe,
        "n_unsafe_attack": n_unsafe_attack,
        "n_unsafe_tree": n_unsafe_tree,
        "n_safe": n - n_unsafe,
        "unsafe_pct": 100.0 * n_unsafe / n if n else 0.0,
        "total_tokens": total_tokens,
        "mean_tokens": total_tokens / n,
        "total_duration_s": total_duration,
        "mean_duration_s": total_duration / n,
        "tokens_per_sec": tokens_per_sec,
        "mean_nodes": total_nodes / n,
        "mean_nodes_x_cs": (total_nodes / n) * (chunk_size or 1),
        "mean_cache_hit_pct": mean_cache_hit,
        "exit_reasons": exit_reasons,
    }


def fmt(val, fmt_str):
    try:
        return format(val, fmt_str)
    except (TypeError, ValueError):
        return str(val)


def print_table(rows: list[dict]):
    cols = [
        ("CS",              "chunk_size",        "<4",   "{}"),
        ("Runs",            "n_runs",             ">4",   "{}"),
        ("Unsafe",          "n_unsafe",           ">6",   "{}"),
        ("AttackSmp",       "n_unsafe_attack",    ">9",   "{}"),
        ("TreeSearch",      "n_unsafe_tree",      ">10",  "{}"),
        ("Unsafe%",         "unsafe_pct",         ">7",   ".1f"),
        ("MeanTokens",      "mean_tokens",        ">10",  ".0f"),
        ("MeanDur(s)",      "mean_duration_s",    ">10",  ".1f"),
        ("Tok/s",           "tokens_per_sec",     ">8",   ".1f"),
        ("MeanNodes",       "mean_nodes",         ">9",   ".1f"),
        ("Nodes*CS",        "mean_nodes_x_cs",    ">8",   ".1f"),
        ("CacheHit%",       "mean_cache_hit_pct", ">9",   ".1f"),
    ]

    header = "  ".join(format(h, w) for h, k, w, _ in cols)
    sep    = "  ".join("-" * int(w.lstrip("<>")) for _, _, w, _ in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(
            format(fmt(row.get(k), f), w)
            for _, k, w, f in cols
        )
        print(line)

    print()
    print("Exit reasons per run:")
    for row in rows:
        cs = row["chunk_size"]
        reasons = row["exit_reasons"]
        print(f"  CS{cs}: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")


def write_csv(rows: list[dict], path: str):
    fields = [
        "chunk_size", "n_runs", "n_unsafe", "n_unsafe_attack", "n_unsafe_tree", "n_safe", "unsafe_pct",
        "total_tokens", "mean_tokens", "total_duration_s", "mean_duration_s",
        "tokens_per_sec", "mean_nodes", "mean_cache_hit_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BOA chunk-size experiment results")
    parser.add_argument("dirs", nargs="+", help="Result directories to analyze")
    parser.add_argument("--csv", metavar="FILE", help="Also write CSV output to FILE")
    args = parser.parse_args()

    rows = []
    for d in args.dirs:
        result_dir = Path(d)
        if not result_dir.is_dir():
            print(f"[WARN] Not a directory: {d}", file=sys.stderr)
            continue
        row = load_dir(result_dir)
        if row:
            rows.append(row)

    if not rows:
        print("No valid result directories found.", file=sys.stderr)
        sys.exit(1)

    # Sort by chunk_size
    rows.sort(key=lambda r: (r["chunk_size"] is None, r["chunk_size"]))

    print(f"\n{'='*70}")
    print(f"  Chunk Size Comparison  ({len(rows)} experiments)")
    print(f"{'='*70}\n")
    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
