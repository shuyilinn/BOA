#!/usr/bin/env python3
"""
Analyze search tree structure to evaluate search strategy behavior.

Examines:
1. Tree shape: breadth at each depth, branching factor
2. Exploration balance: node allocation across root-child subtrees
3. Gradient factor distribution: is the searcher prioritizing improving paths?
4. Score progression: are explored paths actually improving?
5. Depth vs score: is depth decay working as intended?

Usage:
    python scripts/analyze_search_trees.py <result_dir>
    python scripts/analyze_search_trees.py <result_dir> --prompt 1
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_tree(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_lookup(tree: dict) -> Dict[int, dict]:
    return {n["id"]: n for n in tree["nodes"]}


def get_root_child_id(node_id: int, lookup: Dict[int, dict], root_id: int) -> Optional[int]:
    """Walk up to find which root-child subtree this node belongs to."""
    nid = node_id
    while nid is not None:
        parent_id = lookup[nid].get("parent_id")
        if parent_id == root_id:
            return nid
        nid = parent_id
    return None


def analyze_one_tree(tree: dict) -> dict:
    lookup = build_lookup(tree)
    root_id = tree["root_id"]
    nodes = tree["nodes"]

    # Basic stats
    total = len(nodes)
    status_counts = Counter(n["status"] for n in nodes)
    max_depth = max(n["depth"] for n in nodes)

    # Depth distribution (only evaluated/explored nodes, skip root)
    evaluated = [n for n in nodes if n["status"] in ("evaluated", "Jailbreak") and n["depth"] > 0]
    queued = [n for n in nodes if n["status"] == "queued"]

    depth_eval_counts = Counter(n["depth"] for n in evaluated)
    depth_queued_counts = Counter(n["depth"] for n in queued)

    # Branching factor (nodes that have children)
    branching = [len(n["child_ids"]) for n in nodes if n["child_ids"]]

    # --- Subtree balance ---
    root_children = [lookup[cid] for cid in lookup[root_id]["child_ids"]]
    subtree_sizes = {}
    for rc in root_children:
        count = 0
        for n in evaluated:
            rc_id = get_root_child_id(n["id"], lookup, root_id)
            if rc_id == rc["id"]:
                count += 1
        subtree_sizes[rc["id"]] = {
            "text_preview": rc["text"][:30].replace("\n", "\\n"),
            "evaluated_nodes": count,
        }

    # --- Gradient factor & score analysis ---
    gradient_factors = []
    effective_scores = []
    depth_decay_factors = []
    selection_scores_by_depth = defaultdict(list)
    scores_by_depth = defaultdict(list)

    for n in evaluated:
        meta = n.get("metadata", {})
        gf = meta.get("gradient_factor")
        ef = meta.get("effective_score")
        dd = meta.get("depth_decay_factor")
        if gf is not None:
            gradient_factors.append(gf)
        if ef is not None:
            effective_scores.append(ef)
        if dd is not None:
            depth_decay_factors.append(dd)
        sel_score = n.get("selection_score", n.get("score", -1))
        if sel_score >= 0:
            selection_scores_by_depth[n["depth"]].append(sel_score)
        if n.get("score", -1) >= 0:
            scores_by_depth[n["depth"]].append(n["score"])

    # Gradient factor distribution
    gf_bins = {"<0.5 (stagnant)": 0, "0.5-1.0": 0, "1.0 (neutral)": 0, "1.0-1.5": 0, ">=1.5 (improving)": 0}
    for gf in gradient_factors:
        if gf < 0.5:
            gf_bins["<0.5 (stagnant)"] += 1
        elif gf < 1.0:
            gf_bins["0.5-1.0"] += 1
        elif gf == 1.0:
            gf_bins["1.0 (neutral)"] += 1
        elif gf < 1.5:
            gf_bins["1.0-1.5"] += 1
        else:
            gf_bins[">=1.5 (improving)"] += 1

    # Score progression: for each depth, what's the max selection_score?
    max_score_by_depth = {}
    mean_score_by_depth = {}
    for d in sorted(selection_scores_by_depth.keys()):
        vals = selection_scores_by_depth[d]
        max_score_by_depth[d] = max(vals)
        mean_score_by_depth[d] = sum(vals) / len(vals)

    # Exploration order analysis: check if nodes with higher gradient_factor
    # were explored earlier (lower node index = explored earlier in many cases)
    explored_nodes_with_gf = []
    for i, n in enumerate(nodes):
        meta = n.get("metadata", {})
        gf = meta.get("gradient_factor")
        if n["status"] in ("evaluated", "Jailbreak") and gf is not None and n["depth"] > 0:
            explored_nodes_with_gf.append((i, gf, n["depth"], n.get("selection_score", -1)))

    return {
        "total_nodes": total,
        "max_depth": max_depth,
        "status_counts": dict(status_counts),
        "evaluated_count": len(evaluated),
        "queued_count": len(queued),
        "depth_eval_counts": dict(sorted(depth_eval_counts.items())),
        "depth_queued_counts": dict(sorted(depth_queued_counts.items())),
        "branching_factor": {
            "mean": sum(branching) / len(branching) if branching else 0,
            "max": max(branching) if branching else 0,
            "min": min(branching) if branching else 0,
        },
        "subtree_balance": subtree_sizes,
        "gradient_factor_distribution": gf_bins,
        "gradient_factor_stats": {
            "count": len(gradient_factors),
            "mean": sum(gradient_factors) / len(gradient_factors) if gradient_factors else 0,
            "min": min(gradient_factors) if gradient_factors else 0,
            "max": max(gradient_factors) if gradient_factors else 0,
        },
        "max_score_by_depth": max_score_by_depth,
        "mean_score_by_depth": mean_score_by_depth,
    }


def print_report(prompt_idx: int, stats: dict):
    print(f"\n{'='*70}")
    print(f"  Prompt {prompt_idx} — Search Tree Analysis")
    print(f"{'='*70}")

    print(f"\n[Tree Shape]")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Evaluated: {stats['evaluated_count']}, Queued: {stats['queued_count']}")
    print(f"  Status: {stats['status_counts']}")
    bf = stats['branching_factor']
    print(f"  Branching factor: mean={bf['mean']:.2f}, min={bf['min']}, max={bf['max']}")

    print(f"\n[Depth Distribution — evaluated nodes]")
    eval_counts = stats["depth_eval_counts"]
    max_count = max(eval_counts.values()) if eval_counts else 1
    for d in sorted(eval_counts.keys()):
        bar = "█" * int(40 * eval_counts[d] / max_count)
        print(f"  d={d:3d}: {eval_counts[d]:5d} {bar}")

    print(f"\n[Subtree Balance — nodes per root-child subtree]")
    subtrees = stats["subtree_balance"]
    if subtrees:
        total_eval = sum(s["evaluated_nodes"] for s in subtrees.values())
        for sid, info in sorted(subtrees.items(), key=lambda x: -x[1]["evaluated_nodes"]):
            pct = 100 * info["evaluated_nodes"] / total_eval if total_eval else 0
            bar = "█" * int(30 * pct / 100)
            print(f"  [{info['text_preview']:30s}] {info['evaluated_nodes']:5d} ({pct:5.1f}%) {bar}")
    else:
        print("  (no root children)")

    print(f"\n[Gradient Factor Distribution]")
    gf = stats["gradient_factor_distribution"]
    total_gf = sum(gf.values())
    for label, count in gf.items():
        pct = 100 * count / total_gf if total_gf else 0
        bar = "█" * int(30 * pct / 100)
        print(f"  {label:22s}: {count:5d} ({pct:5.1f}%) {bar}")

    gfs = stats["gradient_factor_stats"]
    print(f"  Stats: mean={gfs['mean']:.3f}, min={gfs['min']:.3f}, max={gfs['max']:.3f}")

    print(f"\n[Score Progression by Depth — max selection_score]")
    max_scores = stats["max_score_by_depth"]
    mean_scores = stats["mean_score_by_depth"]
    if max_scores:
        global_max = max(max_scores.values()) if max_scores.values() else 1
        global_max = max(global_max, 0.01)
        for d in sorted(max_scores.keys()):
            bar = "█" * int(30 * max_scores[d] / global_max) if global_max > 0 else ""
            print(f"  d={d:3d}: max={max_scores[d]:6.3f}  mean={mean_scores.get(d, 0):6.3f}  {bar}")
    else:
        print("  (no scored nodes)")


def print_summary(all_stats: List[dict]):
    print(f"\n{'='*70}")
    print(f"  Cross-Prompt Summary ({len(all_stats)} prompts)")
    print(f"{'='*70}")

    # Aggregate gradient factor distributions
    agg_gf = Counter()
    all_gf_means = []
    all_depths = []
    all_eval = []
    all_subtree_ginis = []

    for s in all_stats:
        for label, count in s["gradient_factor_distribution"].items():
            agg_gf[label] += count
        all_gf_means.append(s["gradient_factor_stats"]["mean"])
        all_depths.append(s["max_depth"])
        all_eval.append(s["evaluated_count"])

        # Compute Gini coefficient for subtree balance
        sizes = [info["evaluated_nodes"] for info in s["subtree_balance"].values()]
        if len(sizes) > 1 and sum(sizes) > 0:
            n = len(sizes)
            mean_s = sum(sizes) / n
            if mean_s > 0:
                gini = sum(abs(a - b) for a in sizes for b in sizes) / (2 * n * n * mean_s)
                all_subtree_ginis.append(gini)

    total_gf = sum(agg_gf.values())
    print(f"\n[Aggregate Gradient Factor Distribution]")
    for label in ["<0.5 (stagnant)", "0.5-1.0", "1.0 (neutral)", "1.0-1.5", ">=1.5 (improving)"]:
        count = agg_gf[label]
        pct = 100 * count / total_gf if total_gf else 0
        bar = "█" * int(30 * pct / 100)
        print(f"  {label:22s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n[Key Metrics]")
    print(f"  Mean gradient factor:     {sum(all_gf_means)/len(all_gf_means):.3f}")
    print(f"  Mean max depth:           {sum(all_depths)/len(all_depths):.1f}")
    print(f"  Mean evaluated nodes:     {sum(all_eval)/len(all_eval):.0f}")
    if all_subtree_ginis:
        mean_gini = sum(all_subtree_ginis) / len(all_subtree_ginis)
        print(f"  Mean subtree Gini index:  {mean_gini:.3f}  (0=perfectly balanced, 1=all in one subtree)")

    # Diagnosis
    print(f"\n[Diagnosis]")
    stagnant_pct = 100 * agg_gf.get("<0.5 (stagnant)", 0) / total_gf if total_gf else 0
    improving_pct = 100 * agg_gf.get(">=1.5 (improving)", 0) / total_gf if total_gf else 0
    neutral_pct = 100 * agg_gf.get("1.0 (neutral)", 0) / total_gf if total_gf else 0

    if neutral_pct > 50:
        print(f"  ⚠ {neutral_pct:.0f}% of nodes have gradient_factor=1.0 (neutral).")
        print(f"    This means most nodes are at depth < gradient_window, so gradient")
        print(f"    isn't actually influencing selection. Consider reducing gradient_window.")
    if stagnant_pct > 40:
        print(f"  ⚠ {stagnant_pct:.0f}% of nodes are stagnant (gradient_factor < 0.5).")
        print(f"    The searcher is heavily penalizing many paths. This could be too aggressive")
        print(f"    if the score landscape is inherently flat early on.")
    if improving_pct > 30:
        print(f"  ✓ {improving_pct:.0f}% of nodes show fast improvement (gradient_factor >= 1.5).")
        print(f"    The strategy is finding and following improving paths.")
    if all_subtree_ginis and mean_gini > 0.6:
        print(f"  ⚠ Subtree Gini={mean_gini:.2f} — exploration is concentrated in few subtrees.")
        print(f"    The strategy may not be diversifying enough across different paths.")
    elif all_subtree_ginis and mean_gini < 0.3:
        print(f"  ✓ Subtree Gini={mean_gini:.2f} — exploration is well-balanced across subtrees.")


def main():
    parser = argparse.ArgumentParser(description="Analyze BOA search tree behavior")
    parser.add_argument("result_dir", help="Result directory to analyze")
    parser.add_argument("--prompt", type=int, help="Analyze only this prompt index")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    trees_dir = result_dir / "trees"

    if not trees_dir.exists():
        print(f"No trees/ directory in {result_dir}", file=sys.stderr)
        sys.exit(1)

    tree_files = sorted(trees_dir.glob("prompt_*_tree.json"))
    if not tree_files:
        print("No tree JSON files found.", file=sys.stderr)
        sys.exit(1)

    if args.prompt:
        tree_files = [f for f in tree_files if f"prompt_{args.prompt:04d}" in f.name]
        if not tree_files:
            print(f"No tree file found for prompt {args.prompt}", file=sys.stderr)
            sys.exit(1)

    all_stats = []
    for tf in tree_files:
        prompt_idx = int(tf.stem.split("_")[1])
        tree = load_tree(tf)
        stats = analyze_one_tree(tree)
        print_report(prompt_idx, stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print_summary(all_stats)


if __name__ == "__main__":
    main()
