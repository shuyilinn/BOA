from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from utils.logger import setup_logger

logger = setup_logger("RoutingPolicy")


class RoutingPolicy:
    """
    Apply judge-stage outputs back into search state.
    """

    def __init__(self, config: Any):
        self.config = config

    def get_workload_config(self) -> Dict[str, Any]:
        workload_name = str(getattr(self.config, "workload_name", "") or "")
        workload_configs = getattr(self.config, "workload_configs", {}) or {}
        return workload_configs.get(workload_name, {})

    def is_terminal_unsafe(self, result: Dict[str, Any]) -> bool:
        workload_config = self.get_workload_config()
        if not bool(workload_config.get("terminal_unsafe_enabled", False)):
            return False
        if bool(result.get("is_safe", True)):
            return False
        terminal_layers = workload_config.get("terminal_unsafe_layers", []) or []
        return int(result.get("layer", -1)) in {int(layer) for layer in terminal_layers}

    def make_terminal_success_payload(
        self,
        *,
        node: Any,
        result: Dict[str, Any],
        make_success_payload_fn: Callable[..., dict],
        exit_reason: str,
    ) -> dict:
        return make_success_payload_fn(
            node=node,
            unsafe_result=result["response"],
            response_score=float(result.get("score", 0.0)),
            layer3_score=result.get("layer3_score"),
            layer4_score=result.get("layer4_score"),
            exit_reason=exit_reason,
        )

    def apply_judge_batch_result(
        self,
        judge_result: Any,
        *,
        cache: Any,
        node_brief_fn: Callable[[Any], str],
        check_runtime_limits_fn: Callable[[str], bool],
        add_generated_tokens_fn: Callable[[int, str], None],
        make_success_payload_fn: Callable[..., dict],
        on_judging_flush: Callable[[int], None],
        finalize_rollout_node_fn: Callable[[Any, str], bool],
    ) -> Optional[dict]:
        tasks = judge_result.tasks
        results = judge_result.results
        if not tasks:
            return None

        on_judging_flush(len(tasks))
        touched_nodes = {}

        # Persist judged sample score in cache.
        for task, result in zip(tasks, results):
            node = task.node
            node.scores.append(float(result["score"]))
            touched_nodes[id(node)] = node
            if self.config.enable_sampling_cache:
                cache.add(task.seq_ids, float(result["score"]))
            if self.is_terminal_unsafe(result):
                logger.warning(
                    "!!! TERMINAL UNSAFE HIT: node=%s layer=%s score=%.2f !!!",
                    node_brief_fn(node),
                    int(result.get("layer", -1)),
                    float(result.get("score", 0.0)),
                )
                return self.make_terminal_success_payload(
                    node=node,
                    result=result,
                    make_success_payload_fn=make_success_payload_fn,
                    exit_reason="unsafe_detected_tree_search",
                )

        for node in touched_nodes.values():
            finalized = finalize_rollout_node_fn(node, "judged_node")
            if finalized and check_runtime_limits_fn("node_scored"):
                return None

        if judge_result.generated_tokens_full_expand > 0:
            add_generated_tokens_fn(judge_result.generated_tokens_full_expand, "full_response_extend")

        if len(judge_result.expand_tasks) == 0:
            return None

        for expand_task, expand_result in zip(judge_result.expand_tasks, judge_result.expand_results):
            if self.is_terminal_unsafe(expand_result):
                logger.warning(
                    "!!! TERMINAL UNSAFE HIT: node=%s layer=%s score=%.2f !!!",
                    node_brief_fn(expand_task.node),
                    int(expand_result.get("layer", -1)),
                    float(expand_result.get("score", 0.0)),
                )
                return self.make_terminal_success_payload(
                    node=expand_task.node,
                    result=expand_result,
                    make_success_payload_fn=make_success_payload_fn,
                    exit_reason="unsafe_detected_tree_search",
                )
        return None
