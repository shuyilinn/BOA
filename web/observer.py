"""TreeObserver: hooks into Executor to emit tree-state events for the web UI."""

from __future__ import annotations

import math
import time
from queue import Queue
from typing import Any, Dict, List, Optional

from reporters.reporter import Reporter


class TreeObserver:
    """Observes executor events and pushes serialized updates to a thread-safe queue.

    Events pushed to the queue are dicts with a ``type`` key:
    - ``tree_update``: full flat_tree_v1 snapshot
    - ``step_info``: lightweight step metrics
    - ``result``: final run stats (jailbreak_found, paths, probabilities)
    - ``error``: exception info
    - ``done``: sentinel indicating the run finished
    """

    def __init__(self, event_queue: Queue, throttle_interval: float = 1.0):
        self.queue = event_queue
        self._throttle = throttle_interval
        self._last_tree_push = 0.0
        self._reporter_helper = Reporter.__new__(Reporter)  # borrow _serialize_tree

    # ------------------------------------------------------------------
    # Observer callbacks (called from executor thread)
    # ------------------------------------------------------------------

    def on_step(self, root: Any, step_idx: int, nodes_created: int, max_depth: int, start_time: float) -> None:
        """Called at the end of each search loop iteration."""
        now = time.monotonic()
        elapsed = time.time() - start_time

        # Always send step info
        self.queue.put({
            "type": "step_info",
            "data": {
                "step": step_idx,
                "nodes_created": nodes_created,
                "max_depth": max_depth,
                "elapsed_sec": round(elapsed, 2),
            },
        })

        # Throttle tree snapshots
        if now - self._last_tree_push >= self._throttle:
            self._push_tree(root)
            self._last_tree_push = now

    def on_complete(self, root: Any, stats: Dict[str, Any]) -> None:
        """Called when the run finishes (success or safe)."""
        # Final tree snapshot (unthrottled)
        self._push_tree(root)

        # Build result payload
        result = self._build_result(root, stats)
        self.queue.put({"type": "result", "data": result})
        self.queue.put({"type": "done"})

    def on_error(self, error: Exception) -> None:
        self.queue.put({"type": "run_error", "data": {"message": str(error)}})
        self.queue.put({"type": "done"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_tree(self, root: Any) -> None:
        tree_data = self._reporter_helper._serialize_tree(root)
        self.queue.put({"type": "tree_update", "data": tree_data})

    def _build_result(self, root: Any, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract result payload for the frontend."""
        jailbreak_found = stats.get("jailbreak_found", False)
        result: Dict[str, Any] = {
            "jailbreak_found": jailbreak_found,
            "exit_reason": stats.get("exit_reason", "unknown"),
            "duration": stats.get("duration"),
            "total_tokens_generated": stats.get("total_tokens_generated"),
            "evaluated_nodes_count": stats.get("evaluated_nodes_count"),
            "max_evaluated_depth": stats.get("max_evaluated_depth"),
        }

        if jailbreak_found:
            result["unsafe_path"] = {
                "text": stats.get("final_output", ""),
                "cum_log_prob": stats.get("log_probability"),
                "log_probability": stats.get("log_probability"),
                "probability": stats.get("probability"),
                "score": stats.get("score"),
                "layer3_score": stats.get("layer3_score"),
                "layer4_score": stats.get("layer4_score"),
            }

        return result
