from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FinalizeResult:
    jailbreak_found: bool
    missing_unsafe_result: bool
    final_output_text: str


# ---------------------------------------------------------------------------
# Report-time helpers (run once per prompt after execution, zero hot-path cost)
# ---------------------------------------------------------------------------

def _find_best_node(root: Any) -> Optional[Any]:
    """Walk the tree and return the node with the highest score (> 0)."""
    if root is None:
        return None
    best = root
    stack = [root]
    while stack:
        n = stack.pop()
        if getattr(n, "score", -1) > getattr(best, "score", -1):
            best = n
        stack.extend(getattr(n, "children", []) or [])
    return best if getattr(best, "score", -1) > 0 else None


def _extract_tool_summary(root: Any) -> Dict[str, Dict[str, int]]:
    """Walk the tree and summarize tool usage from TOOL nodes' existing metadata."""
    summary: Dict[str, Dict[str, int]] = {}
    if root is None:
        return summary
    stack = [root]
    while stack:
        n = stack.pop()
        if getattr(n, "role", None) and n.role.value == "tool":
            msgs = (getattr(n, "metadata", None) or {}).get("structured_messages", [])
            for msg in msgs:
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        name = tc.get("function", {}).get("name", "unknown")
                        entry = summary.setdefault(name, {"calls": 0})
                        entry["calls"] += 1
                elif msg.get("role") == "tool":
                    name = msg.get("name", "unknown")
                    content = msg.get("content", "")
                    try:
                        parsed = json.loads(content) if isinstance(content, str) else content
                        success = parsed.get("success", True) if isinstance(parsed, dict) else True
                    except (json.JSONDecodeError, AttributeError):
                        success = True
                    entry = summary.setdefault(name, {"calls": 0})
                    entry.setdefault("success", 0)
                    entry.setdefault("fail", 0)
                    entry["success" if success else "fail"] += 1
        stack.extend(getattr(n, "children", []) or [])
    return summary


def _extract_conversation_trace(target_node: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract committed_messages from a node's conversation_state."""
    if target_node is None:
        return None
    cs = getattr(target_node, "conversation_state", None)
    if cs is None:
        return None
    msgs = getattr(cs, "committed_messages", None)
    return msgs if msgs else None


def _enrich_report_time_stats(stats: Dict[str, Any], target_node: Any, root_node: Any) -> None:
    """Collect report-time data from existing node fields. Zero hot-path cost."""
    # A2: Best safe response (only for safe case)
    if not stats.get("jailbreak_found", False):
        best = _find_best_node(root_node)
        if best is not None:
            stats["best_safe_response"] = best.conversation_state.latest_assistant_text()[-2000:]
            stats["best_safe_score"] = float(best.score)
            stats["best_safe_depth"] = best.depth

    # A3: Conversation trace
    trace = _extract_conversation_trace(target_node)
    if trace is not None:
        stats["conversation_trace"] = trace

    # A4: Tool usage summary
    tool_summary = _extract_tool_summary(root_node)
    if tool_summary:
        stats["tool_summary"] = tool_summary


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

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
    root_node: Optional[Any] = None,
    response_tau: Optional[float] = None,
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

        node_log_prob = float(node.cum_log_prob)
        full_log_prob = node_log_prob + response_tau if response_tau is not None else node_log_prob
        stats["log_probability"] = full_log_prob
        stats["probability"] = math.exp(full_log_prob) if full_log_prob > -745.0 else 0.0
        stats["tree_path_log_probability"] = node_log_prob
        stats["response_tau"] = response_tau

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

        # Report-time enrichment (unsafe case: target is the winning node)
        _enrich_report_time_stats(stats, target_node=node, root_node=root_node)

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

    # Report-time enrichment (safe case: target is best-scoring node)
    best = _find_best_node(root_node)
    _enrich_report_time_stats(stats, target_node=best, root_node=root_node)

    return FinalizeResult(
        jailbreak_found=False,
        missing_unsafe_result=False,
        final_output_text=str(stats.get("final_output") or ""),
    )
