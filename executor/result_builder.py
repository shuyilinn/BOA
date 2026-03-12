from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FinalizeResult:
    jailbreak_found: bool
    missing_unsafe_result: bool
    final_output_text: str


def build_final_stats(
    *,
    stats: Dict[str, Any],
    node: Optional[Any],
    unsafe_result: Optional[str],
    response_score: Optional[float],
    layer3_score: Optional[float],
    layer4_score: Optional[float],
    exit_reason: Optional[str],
    runtime_guard: Any,
    duration_sec: float,
    cache_payload: Dict[str, Any],
    buffer_payload: Dict[str, Any],
) -> FinalizeResult:
    stats["duration"] = float(duration_sec)
    stats["cache"] = cache_payload
    stats["buffer"] = buffer_payload
    stats["timeout_reached"] = bool(runtime_guard.timeout_reached)
    stats["token_budget_reached"] = bool(runtime_guard.token_budget_reached)
    stats["depth_budget_reached"] = bool(getattr(runtime_guard, "depth_budget_reached", False))
    stats["node_budget_reached"] = bool(getattr(runtime_guard, "node_budget_reached", False))
    if runtime_guard.timeout_reason:
        stats["stop_reason"] = runtime_guard.timeout_reason
    elif runtime_guard.token_budget_reason:
        stats["stop_reason"] = runtime_guard.token_budget_reason
    elif getattr(runtime_guard, "depth_budget_reason", None):
        stats["stop_reason"] = runtime_guard.depth_budget_reason
    elif getattr(runtime_guard, "node_budget_reason", None):
        stats["stop_reason"] = runtime_guard.node_budget_reason

    if node is not None:
        stats["jailbreak_found"] = True
        if response_score is not None:
            stats["score"] = float(response_score)
        else:
            stats["score"] = float(getattr(node, "score", 0.0))

        node_log_prob = getattr(node, "cum_log_prob", None)
        if isinstance(node_log_prob, (int, float)):
            node_log_prob = float(node_log_prob)
            stats["log_probability"] = node_log_prob
            stats["probability"] = math.exp(node_log_prob) if node_log_prob > -745.0 else 0.0

        if layer3_score is not None:
            stats["layer3_score"] = float(layer3_score)
        if layer4_score is not None:
            stats["layer4_score"] = float(layer4_score)
        stats["exit_reason"] = exit_reason or "unsafe_detected_tree_search"
        node.mark_as_jailbreak()

        missing_unsafe = not bool(unsafe_result)
        if missing_unsafe:
            stats["invariant_error"] = "missing_unsafe_result_for_success_node"
            stats["final_output"] = "[ERROR] Missing unsafe_result for success node."
        else:
            stats["final_output"] = unsafe_result

        return FinalizeResult(
            jailbreak_found=True,
            missing_unsafe_result=missing_unsafe,
            final_output_text=str(stats.get("final_output") or ""),
        )

    stats["jailbreak_found"] = False
    stats["final_output"] = "Safe / No jailbreak found."
    stats["log_probability"] = None
    stats["probability"] = None
    if exit_reason:
        stats["exit_reason"] = str(exit_reason)
    elif bool(runtime_guard.timeout_reached):
        stats["exit_reason"] = "time_limit_exceeded"
    elif bool(runtime_guard.token_budget_reached):
        stats["exit_reason"] = "token_budget_exceeded"
    elif bool(getattr(runtime_guard, "depth_budget_reached", False)):
        stats["exit_reason"] = "depth_budget_exceeded"
    elif bool(getattr(runtime_guard, "node_budget_reached", False)):
        stats["exit_reason"] = "node_budget_exceeded"
    else:
        stats["exit_reason"] = "completed_no_unsafe"
    return FinalizeResult(
        jailbreak_found=False,
        missing_unsafe_result=False,
        final_output_text=str(stats.get("final_output") or ""),
    )
