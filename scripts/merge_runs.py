"""
Merge runs from multiple experiment directories.
For each prompt (identified by absolute benchmark index), prefer the unsafe result.
Produces runs.jsonl, runs.txt, and trees/ identical in format to a single experiment.

Usage: python scripts/merge_runs.py <output_dir> <result_dir1> <result_dir2> ...
"""

import json
import os
import shutil
import sys


def load_runs(result_dir):
    meta_path = os.path.join(result_dir, "metadata.json")
    runs_path = os.path.join(result_dir, "runs.jsonl")

    with open(meta_path) as f:
        meta = json.load(f)

    harmful_prompt_start = meta["config"]["harmful_prompt_start"]

    runs = []
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompt_index = entry["stats"]["prompt_index"]
            abs_index = harmful_prompt_start + prompt_index - 1
            # Remember which source dir this came from
            entry["_source_dir"] = result_dir
            runs.append((abs_index, entry))

    return runs


def merge_runs(result_dirs):
    # abs_index -> best entry (prefer unsafe)
    best: dict = {}

    for d in result_dirs:
        print(f"Loading: {d}")
        runs = load_runs(d)
        for abs_index, entry in runs:
            if abs_index not in best:
                best[abs_index] = entry
            else:
                current_jailbreak = best[abs_index]["stats"].get("jailbreak_found", False)
                new_jailbreak = entry["stats"].get("jailbreak_found", False)
                if new_jailbreak and not current_jailbreak:
                    best[abs_index] = entry

    return sorted(best.items())


def render_runs_txt_block(abs_index, total_prompts, stats, tree_txt_path, tree_json_path):
    """Reproduce the exact runs.txt block format from reporter.py."""
    original_prompt_str = stats.get("original_prompt") or "N/A"
    effective_prompt_str = stats.get("effective_prompt") or stats.get("prompt") or "N/A"
    prompt_metadata = stats.get("prompt_metadata", {}) or {}
    fulfillable = prompt_metadata.get("fulfillable")
    prompt_str = stats.get("prompt") or original_prompt_str
    final_out = stats.get("final_output") or "N/A"
    jailbreak = stats.get("jailbreak_found", False)
    layer3_score = stats.get("layer3_score")
    layer4_score = stats.get("layer4_score")
    duration = stats.get("duration")
    elapsed = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)
    tok = stats.get("total_tokens_generated")
    tok_str = tok if tok is not None else "N/A"
    tok_sources = stats.get("token_sources", {}) or {}
    evaluated_nodes_count = stats.get("evaluated_nodes_count")
    max_evaluated_depth = stats.get("max_evaluated_depth")
    prob = stats.get("probability")
    log_prob = stats.get("log_probability")
    prob_str = f"{float(prob):.6g}" if isinstance(prob, (int, float)) else "N/A"
    log_prob_str = f"{float(log_prob):.6f}" if isinstance(log_prob, (int, float)) else "N/A"
    safe_str = "NO" if jailbreak else "YES"
    generated_display = final_out if jailbreak else "(No jailbreak found)"
    exit_reason = stats.get("exit_reason", "N/A")
    tree_stats = stats.get("tree_stats", {}) or {}

    prompt_title = f"Prompt ({abs_index}/{total_prompts}):"

    separator = "=" * 96
    tail_separator = "-" * 96
    block = []
    block.append("")
    block.append(separator)
    block.append(prompt_title)
    block.append(prompt_str)
    block.append(f"Original prompt: {original_prompt_str}")
    block.append(f"Effective prompt: {effective_prompt_str}")
    if fulfillable is not None:
        block.append(f"Fulfillable: {fulfillable}")
    block.append("")
    block.append("Generated text (current best):")
    block.append(generated_display)
    block.append(f"Layer3 score: {layer3_score}")
    block.append(f"Layer4 score: {layer4_score}")
    block.append(f"Safe: {safe_str}")
    block.append(f"Exit reason: {exit_reason}")
    block.append(f"Log probability (full seq): {log_prob_str}")
    tree_path_lp = stats.get("tree_path_log_probability")
    block.append(
        f"Log probability (tree path): {tree_path_lp:.6f}"
        if tree_path_lp is not None
        else "Log probability (tree path): N/A"
    )
    block.append(f"Probability: {prob_str}")
    block.append(f"Elapsed time: {elapsed}")
    block.append(f"Tokens generated: {tok_str}")
    block.append(
        f"Search depth stats: evaluated_nodes={evaluated_nodes_count}, max_evaluated_depth={max_evaluated_depth}"
    )
    block.append(
        f"Tree stats: total={tree_stats.get('total')}, max_depth={tree_stats.get('max_depth')}, "
        f"evaluated={tree_stats.get('evaluated')}, queued={tree_stats.get('queued')}"
    )
    block.append(
        f"Tokens by source: attack_sampling={tok_sources.get('attack_sampling')}, "
        f"sampling={tok_sources.get('sampling')}, "
        f"full_response_extend={tok_sources.get('full_response_extend')}"
    )
    cache_stats = stats.get("cache", {}) or {}
    block.append(
        f"Cache: lookups={cache_stats.get('lookups')}, full_hits={cache_stats.get('full_hits')}, "
        f"partial_hits={cache_stats.get('partial_hits')}, "
        f"full_hit_rate={cache_stats.get('full_hit_rate_pct')}%, "
        f"any_hit_rate={cache_stats.get('any_hit_rate_pct')}%"
    )
    refusal_filter = stats.get("refusal_filter", {}) or {}
    if refusal_filter:
        block.append(
            f"Refusal filter: refusal={refusal_filter.get('refusal_total')}, "
            f"no_refusal_checked={refusal_filter.get('no_refusal_checked')}, "
            f"reverted={refusal_filter.get('reverted_total')}, "
            f"total={refusal_filter.get('total')}, "
            f"refusal_ratio={refusal_filter.get('confirmed_ratio_pct')}%, "
            f"no_refusal_ratio={refusal_filter.get('no_refusal_checked_ratio_pct')}%, "
            f"reverted_ratio={refusal_filter.get('reverted_ratio_pct')}%"
        )
    buffer_stats = stats.get("buffer", {}) or {}
    sample_buf = buffer_stats.get("sample_buffer", {}) or {}
    judge_buf = buffer_stats.get("judging_buffer", {}) or {}
    block.append(
        f"Sample buffer: capacity={sample_buf.get('capacity')}, "
        f"enqueued_items={sample_buf.get('enqueued_items')}, "
        f"items={sample_buf.get('items')}, batches={sample_buf.get('batches')}, "
        f"mean_batch_size={sample_buf.get('mean_batch_size')}, "
        f"max_queue_size={sample_buf.get('max_queue_size')}"
    )
    block.append(
        f"Judging buffer: capacity={judge_buf.get('capacity')}, "
        f"enqueued_items={judge_buf.get('enqueued_items')}, "
        f"items={judge_buf.get('items')}, batches={judge_buf.get('batches')}, "
        f"mean_batch_size={judge_buf.get('mean_batch_size')}, "
        f"max_queue_size={judge_buf.get('max_queue_size')}"
    )
    block.append(f"Tree txt file: {tree_txt_path}")
    block.append(f"Tree json file: {tree_json_path}")
    if "profiling" in stats:
        block.append("")
        block.append("[Profiling Report]")
        block.append(str(stats["profiling"]))
    sampled = stats.get("sampled_responses", [])
    if sampled:
        block.append("")
        block.append("Sampled Complete Responses:")
        for si, s in enumerate(sampled):
            block.append(f"  #{si+1} | Cum Log Prob: {s['cum_log_prob']:.6f} | Prob: {s['probability']:.6g}")
            block.append(f"  {s['text'][:500]}")
            block.append("")
    block.append(tail_separator)
    block.append("")
    return "\n".join(block) + "\n"


def write_merged(output_dir, sorted_entries):
    trees_dir = os.path.join(output_dir, "trees")
    os.makedirs(trees_dir, exist_ok=True)

    runs_jsonl_path = os.path.join(output_dir, "runs.jsonl")
    runs_txt_path = os.path.join(output_dir, "runs.txt")

    total = len(sorted_entries)
    unsafe_count = 0

    with open(runs_jsonl_path, "w") as jf, open(runs_txt_path, "w") as tf:
        for abs_index, entry in sorted_entries:
            stats = entry["stats"]
            source_dir = entry.pop("_source_dir")

            # New tree filenames based on absolute index
            new_tree_name = f"prompt_{abs_index:04d}_tree"
            new_tree_txt = f"trees/{new_tree_name}.txt"
            new_tree_json = f"trees/{new_tree_name}.json"

            # Copy tree files from source
            old_tree_txt = entry.get("tree_file_txt")
            old_tree_json = entry.get("tree_file_json")
            for old_rel, new_rel in [(old_tree_txt, new_tree_txt), (old_tree_json, new_tree_json)]:
                if old_rel:
                    src = os.path.join(source_dir, old_rel)
                    dst = os.path.join(output_dir, new_rel)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

            # Update entry
            stats["prompt_index"] = abs_index
            stats["total_prompts"] = total
            entry["tree_file_txt"] = new_tree_txt
            entry["tree_file_json"] = new_tree_json

            # Write runs.jsonl
            jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Write runs.txt block
            tf.write(render_runs_txt_block(abs_index, total, stats, new_tree_txt, new_tree_json))

            if stats.get("jailbreak_found", False):
                unsafe_count += 1

    asr = unsafe_count / total * 100 if total > 0 else 0.0
    print(f"\nMerged {total} prompts -> {output_dir}")
    print(f"ASR: {unsafe_count}/{total} = {asr:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/merge_runs.py <output_dir> <result_dir1> <result_dir2> ...")
        sys.exit(1)

    output_dir = sys.argv[1]
    result_dirs = sys.argv[2:]

    sorted_entries = merge_runs(result_dirs)
    write_merged(output_dir, sorted_entries)
