from __future__ import annotations

from collections import deque
from typing import Any, List


def node_brief(node: Any) -> str:
    path_len = len(node.get_path_token_ids())
    text_full = (getattr(node, "text", "") or "").replace("\n", "\\n")
    status_val = getattr(getattr(node, "status", None), "value", "N/A")
    return (
        f"id={id(node)} depth={getattr(node, 'depth', 'N/A')} status={status_val} "
        f"score={float(getattr(node, 'score', 0.0)):.2f} "
        f"logP={float(getattr(node, 'cum_log_prob', 0.0)):.2f} "
        f"path_tokens={path_len} text='{text_full}'"
    )


def build_tree_snapshot(root: Any, *, max_depth: int, max_nodes: int) -> str:
    lines: List[str] = []
    shown = 0
    if root is None or max_nodes <= 0:
        return ""

    # Iterative DFS avoids Python recursion limits and explicitly guards cycles.
    stack = deque([(root, "", 0, frozenset())])
    while stack and shown < max_nodes:
        node, prefix, depth, ancestors = stack.pop()
        node_id = id(node)

        if node_id in ancestors:
            lines.append(f"{prefix}- [CYCLE] node_id={node_id} (snapshot traversal stopped on this branch)")
            shown += 1
            continue

        status = str(getattr(getattr(node, "status", None), "value", "N/A")).upper()
        text = (getattr(node, "text", "") or "").replace("\n", " ")
        token_ids = getattr(node, "token_ids", None) or []
        token_id = token_ids[-1] if token_ids else "N/A"
        lines.append(
            f"{prefix}- [{status}] '{text}' (id: {token_id}), "
            f"LOGP: {float(getattr(node, 'log_prob', 0.0)):.2f}, "
            f"CUM_LOGP: {float(getattr(node, 'cum_log_prob', 0.0)):.2f}, "
            f"score: {float(getattr(node, 'score', 0.0)):.1f}"
        )
        shown += 1

        if depth >= max_depth:
            continue

        children = getattr(node, "children", []) or []
        if not children:
            continue

        next_ancestors = ancestors | {node_id}
        for child in reversed(children):
            stack.append((child, prefix + "  ", depth + 1, next_ancestors))
    return "\n".join(lines)
