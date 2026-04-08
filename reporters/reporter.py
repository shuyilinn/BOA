import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from config import Config
from utils.run_naming import build_result_dir, build_run_id

class Reporter:
    def __init__(self, config: Config):
        self.config = config

        # Global metadata from config (model, sampling, acceleration)
        self.global_metadata: Dict[str, Any] = {
            "target_model": self.config.target_model,
            "target_engine_name": self.config.target_engine_name,
            "sampling_params": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "likelihood": self.config.likelihood,
                "sampler_number": self.config.sampler_number,
                "attack_sampler_number": self.config.attack_sampler_number,
                # Clear semantic names for downstream analysis.
                "short_response_max_new_tokens": self.config.sample_new_tokens,
                "attack_sample_max_new_tokens": self.config.attack_sample_new_tokens,
                "full_response_max_new_tokens": self.config.sample_full_new_tokens,
            },
            "acceleration": {
                "enable_sampling_cache": self.config.enable_sampling_cache,
                "enable_sampling_buffer": self.config.enable_sampling_buffer,
                "enable_judging_buffer": self.config.enable_judging_buffer,
                "enable_topk_optimization": self.config.enable_topk_optimization,
                "enable_refuse_pattern_matching": self.config.enable_refuse_pattern_matching,
                "enable_refuse_judger": self.config.enable_refuse_judger,
            },
        }

        run_id = getattr(self.config, "run_id", None)
        if run_id is None:
            run_id = build_run_id(self.config)
        self.run_id = run_id
        result_dir = build_result_dir(self.config, run_id=run_id)
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.runs_jsonl_path = os.path.join(self.result_dir, "runs.jsonl")
        self.trees_dir = os.path.join(self.result_dir, "trees")
        os.makedirs(self.trees_dir, exist_ok=True)
        self._prompt_counter = 0

        # Write global metadata (json + txt)
        self._write_global_metadata()

    def _config_as_dict(self) -> Dict[str, Any]:
        if is_dataclass(self.config):
            return asdict(self.config)
        # Fallback: serialize plain instance attributes only.
        allowed = (str, int, float, bool, type(None), dict, list)
        raw = vars(self.config) if hasattr(self.config, "__dict__") else {}
        return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, allowed)}

    def _write_global_metadata(self) -> None:
        payload = {
            "metadata": self.global_metadata,
            "config": self._config_as_dict(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        json_path = os.path.join(self.result_dir, "metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        txt_path = os.path.join(self.result_dir, "metadata.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 30 + " GLOBAL METADATA " + "=" * 30 + "\n")
            f.write(f"target_model: {self.global_metadata.get('target_model')}\n")
            f.write(f"target_engine_name: {self.global_metadata.get('target_engine_name')}\n")
            f.write("\n[sampling_params]\n")
            for k, v in (self.global_metadata.get("sampling_params") or {}).items():
                f.write(f"- {k}: {v}\n")
            f.write("\n[acceleration]\n")
            for k, v in (self.global_metadata.get("acceleration") or {}).items():
                f.write(f"- {k}: {v}\n")

    def _render_tree(self, node: Any, indent: str = "") -> str:
        """Build TXT tree with iterative DFS."""
        if not node:
            return ""

        lines = []
        stack = [(node, indent, 0, frozenset())]

        while stack:
            cur, cur_indent, depth, ancestors = stack.pop()
            node_id = id(cur)
            if node_id in ancestors:
                lines.append(f"{cur_indent}- [CYCLE] node_id={node_id} (branch skipped)\n")
                continue
            token_ids = getattr(cur, "token_ids", None) or []
            token_id = token_ids[-1] if token_ids else "N/A"
            status = str(getattr(getattr(cur, "status", None), "value", getattr(cur, "status", "N/A"))).upper()
            metadata = getattr(cur, "metadata", {}) or {}
            hit_tag = " [HIT]" if bool(metadata.get("is_jailbreak_hit", False)) else ""
            path_tag = " 🔓" if bool(metadata.get("is_jailbreak_path", False)) else ""
            text = getattr(cur, "text", "")
            log_prob = getattr(cur, "log_prob", None)
            cum_log_prob = getattr(cur, "cum_log_prob", None)
            score = getattr(cur, "score", None)
            node_depth = getattr(cur, "depth", None)
            role = cur.role.value

            log_prob_str = f"{log_prob:.2f}" if isinstance(log_prob, (int, float)) else str(log_prob)
            lines.append(
                f"{cur_indent}- [{status}]{hit_tag}{path_tag} {repr(text)} "
                f"(ID: {token_id}), DEPTH: {node_depth}, ROLE: {role}, "
                f"LOGP: {log_prob_str}, CUM_LOGP: {cum_log_prob}, SCORE: {score}\n"
            )

            children = getattr(cur, "children", []) or []
            next_ancestors = ancestors | {node_id}
            for child in reversed(children):
                stack.append((child, cur_indent + "  ", depth + 1, next_ancestors))

        return "".join(lines)

    def _serialize_tree(self, node: Any) -> Dict:
        """Serialize tree into a flat graph JSON to avoid recursive encoding depth issues."""
        if not node:
            return {}

        def node_payload(n: Any, node_id: int, parent_id: int | None) -> Dict[str, Any]:
            return {
                "id": node_id,
                "parent_id": parent_id,
                "text": getattr(n, "text", ""),
                "token_ids": getattr(n, "token_ids", None),
                "depth": getattr(n, "depth", None),
                "status": getattr(getattr(n, "status", None), "value", getattr(n, "status", None)),
                "role": n.role.value,
                "score": getattr(n, "score", None),
                "selection_score": getattr(n, "selection_score", None),
                "scores": getattr(n, "scores", None) or [],
                "log_prob": getattr(n, "log_prob", None),
                "cum_log_prob": getattr(n, "cum_log_prob", None),
                "child_ids": [],
                "metadata": getattr(n, "metadata", None) or {},
            }

        root_id = id(node)
        nodes: list[Dict[str, Any]] = []
        stack = [(node, None, 0, frozenset({root_id}))]

        while stack:
            src_node, parent_id, depth, ancestors = stack.pop()
            src_id = id(src_node)
            current = node_payload(src_node, src_id, parent_id)
            children = getattr(src_node, "children", []) or []
            cycle_child_ids = []
            for child in children:
                child_id = id(child)
                current["child_ids"].append(child_id)
                if child_id in ancestors:
                    cycle_child_ids.append(child_id)
                    continue
                stack.append((child, src_id, depth + 1, ancestors | {child_id}))
            if cycle_child_ids:
                current["cycle_child_ids"] = cycle_child_ids
            nodes.append(current)

        return {
            "format": "flat_tree_v1",
            "root_id": root_id,
            "node_count": len(nodes),
            "nodes": nodes,
        }

    def generate_reports(self, stats: Dict[str, Any], root_node: Any):
        """
        Main entry: called by Executor per prompt.
        Structured entries are appended to result_dir/runs.jsonl.
        """
        tree_paths = self._write_tree_files_per_prompt(stats=stats, root_node=root_node)
        tree_stats = stats.get("tree_stats") or {}
        if tree_stats:
            print(
                "Tree stats (persist): total=%s max_depth=%s evaluated=%s queued=%s"
                % (
                    tree_stats.get("total"),
                    tree_stats.get("max_depth"),
                    tree_stats.get("evaluated"),
                    tree_stats.get("queued"),
                )
            )
        run_entry = {
            "stats": stats,
            "tree_file_txt": tree_paths["txt"],
            "tree_file_json": tree_paths["json"],
        }

        # 1. Append to runs.jsonl in O(1) amortized I/O per prompt.
        with open(self.runs_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_entry, ensure_ascii=False) + "\n")

        # 2. Append to runs.txt (human-readable, all prompts in one file; no tree body)
        runs_txt_path = os.path.join(self.result_dir, "runs.txt")
        original_prompt_str = stats.get("original_prompt") or "N/A"
        effective_prompt_str = stats.get("effective_prompt") or stats.get("prompt") or "N/A"
        prompt_metadata = stats.get("prompt_metadata", {}) or {}
        fulfillable = prompt_metadata.get("fulfillable")
        prompt_str = stats.get("prompt") or original_prompt_str
        final_out = stats.get("final_output") or "N/A"
        jailbreak = stats["jailbreak_found"]
        # Compatibility baseline score: only from persisted stats; do not mix in root_node.score.
        # Older runs may only have a single `score` without split layer scores.
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
        if isinstance(prob, (int, float)):
            prob_str = f"{float(prob):.6g}"
        else:
            prob_str = "N/A"
        if isinstance(log_prob, (int, float)):
            log_prob_str = f"{float(log_prob):.6f}"
        else:
            log_prob_str = "N/A"
        safe_str = "NO" if jailbreak else "YES"
        generated_display = final_out if jailbreak else "(No jailbreak found)"
        exit_reason = stats.get("exit_reason", "N/A")

        prompt_index = stats.get("prompt_index")
        total_prompts = stats.get("total_prompts")
        if isinstance(prompt_index, int) and prompt_index > 0:
            if isinstance(total_prompts, int) and total_prompts > 0:
                prompt_title = f"Prompt ({prompt_index}/{total_prompts}):"
            else:
                prompt_title = f"Prompt ({prompt_index}):"
        else:
            prompt_title = "Prompt:"

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
        block.append(f"Log probability (tree path): {tree_path_lp:.6f}" if tree_path_lp is not None else "Log probability (tree path): N/A")
        block.append(f"Probability: {prob_str}")
        block.append(f"Elapsed time: {elapsed}")
        block.append(f"Tokens generated: {tok_str}")
        block.append(
            "Search depth stats: "
            f"evaluated_nodes={evaluated_nodes_count}, "
            f"max_evaluated_depth={max_evaluated_depth}"
        )
        block.append(
            "Tree stats: "
            f"total={tree_stats.get('total')}, "
            f"max_depth={tree_stats.get('max_depth')}, "
            f"evaluated={tree_stats.get('evaluated')}, "
            f"queued={tree_stats.get('queued')}"
        )
        block.append(
            "Tokens by source: "
            f"attack_sampling={tok_sources.get('attack_sampling')}, "
            f"sampling={tok_sources.get('sampling')}, "
            f"full_response_extend={tok_sources.get('full_response_extend')}"
        )
        cache_stats = stats.get("cache", {}) or {}
        block.append(
            "Cache: "
            f"lookups={cache_stats.get('lookups')}, "
            f"full_hits={cache_stats.get('full_hits')}, "
            f"partial_hits={cache_stats.get('partial_hits')}, "
            f"full_hit_rate={cache_stats.get('full_hit_rate_pct')}%, "
            f"any_hit_rate={cache_stats.get('any_hit_rate_pct')}%"
        )
        refusal_filter = stats.get("refusal_filter", {}) or {}
        if refusal_filter:
            block.append(
                "Refusal filter: "
                f"refusal={refusal_filter.get('refusal_total')}, "
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
            "Sample buffer: "
            f"capacity={sample_buf.get('capacity')}, "
            f"enqueued_items={sample_buf.get('enqueued_items')}, "
            f"items={sample_buf.get('items')}, "
            f"batches={sample_buf.get('batches')}, "
            f"mean_batch_size={sample_buf.get('mean_batch_size')}, "
            f"max_queue_size={sample_buf.get('max_queue_size')}"
        )
        block.append(
            "Judging buffer: "
            f"capacity={judge_buf.get('capacity')}, "
            f"enqueued_items={judge_buf.get('enqueued_items')}, "
            f"items={judge_buf.get('items')}, "
            f"batches={judge_buf.get('batches')}, "
            f"mean_batch_size={judge_buf.get('mean_batch_size')}, "
            f"max_queue_size={judge_buf.get('max_queue_size')}"
        )
        block.append(f"Tree txt file: {tree_paths['txt']}")
        block.append(f"Tree json file: {tree_paths['json']}")
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
        with open(runs_txt_path, "a", encoding="utf-8") as f:
            f.write("\n".join(block) + "\n")

        try:
            rel_dir = os.path.relpath(self.result_dir)
            print(f"Reports appended under {rel_dir}: runs.jsonl, runs.txt, trees/")
        except Exception:
            print(f"Reports appended: {self.runs_jsonl_path}, {runs_txt_path}, {self.trees_dir}")

    def _write_tree_files_per_prompt(self, stats: Dict[str, Any], root_node: Any) -> Dict[str, str]:
        prompt_index = stats.get("prompt_index")
        if isinstance(prompt_index, int) and prompt_index > 0:
            stem = f"prompt_{prompt_index:04d}"
        else:
            self._prompt_counter += 1
            stem = f"prompt_auto_{self._prompt_counter:04d}"

        tree_txt_path = os.path.join(self.trees_dir, f"{stem}_tree.txt")
        with open(tree_txt_path, "w", encoding="utf-8") as f:
            f.write(self._render_tree(root_node))
        tree_json_path = os.path.join(self.trees_dir, f"{stem}_tree.json")
        with open(tree_json_path, "w", encoding="utf-8") as f:
            json.dump(self._serialize_tree(root_node), f, ensure_ascii=False, indent=2)
        return {
            "txt": os.path.relpath(tree_txt_path, self.result_dir),
            "json": os.path.relpath(tree_json_path, self.result_dir),
        }
