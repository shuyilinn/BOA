import heapq
import math
import random
from typing import Optional, List, Tuple, Dict
from boa_types.tree_node import TreeNode, NodeStatus
from utils.logger import setup_logger

logger = setup_logger("PhaseAwareSearcher")

# Key for shallow nodes (depth < min_depth) in _subtree_queues
_SHALLOW_KEY = -1

# Type alias for queue entries
Entry = Tuple[float, float, float, float, int, TreeNode]


class PhaseAwareSearcher:
    """Phase-aware search: force shallow exploration, then per-subtree fair selection.

    Rules:
    1. depth < min_depth: fixed high priority (shallow_score), expanded first.
    2. depth = min_depth: each such node is a subtree root.
    3. select_next_node: from each subtree queue, pick top subtree_top_k
       candidates, pool them together, then softmax select one.
       This ensures every subtree gets fair exploration.
    """

    def __init__(
        self,
        max_beam_width: int = 20000,
        search_temperature: float = 0.5,
        min_depth: int = 3,
        shallow_score: float = 8000,
        subtree_top_k: int = 5,
    ):
        self.priority_queue: List[Entry] = []
        self.max_beam_width = max_beam_width
        self.search_temperature = search_temperature

        self.min_depth = min_depth
        self.shallow_score = shallow_score
        self.subtree_top_k = subtree_top_k

        # Per-subtree min-heaps. Key: id(subtree_root) or _SHALLOW_KEY.
        self._subtree_queues: Dict[int, List[Entry]] = {}
        # Reverse lookup: id(node) -> subtree key
        self._node_to_subtree: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_subtree_key(self, node: TreeNode) -> int:
        """Return the subtree key for a node."""
        if node.depth < self.min_depth:
            return _SHALLOW_KEY
        # Walk up to depth = min_depth ancestor
        current = node
        while current is not None and current.depth > self.min_depth:
            current = current.parent
        if current is not None and current.depth == self.min_depth:
            return id(current)
        # Fallback: node is exactly at min_depth
        return id(node)

    def _remove_from_subtree_queue(self, entry: Entry, st_key: int):
        """Remove an entry from its subtree queue."""
        sq = self._subtree_queues.get(st_key)
        if sq is None:
            return
        try:
            sq.remove(entry)
            heapq.heapify(sq)
        except ValueError:
            pass
        if not sq:
            del self._subtree_queues[st_key]

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_next_node(self) -> Optional[TreeNode]:
        if not self.priority_queue:
            return None

        # Collect top-K candidates from each subtree
        candidates: List[Entry] = []
        for sq in self._subtree_queues.values():
            if sq:
                k = min(self.subtree_top_k, len(sq))
                candidates.extend(heapq.nsmallest(k, sq))

        if not candidates:
            return None

        # Softmax selection over pooled candidates (normalize to [0,1] first)
        scores = [-c[0] for c in candidates]
        max_s = max(scores) if scores else 1.0
        if max_s > 0:
            norm_scores = [s / max_s for s in scores]
        else:
            norm_scores = [1.0 for _ in scores]
        exps = [math.exp((s - 1.0) / self.search_temperature) for s in norm_scores]
        total = sum(exps)
        probs = [e / total for e in exps]
        chosen = random.choices(candidates, weights=probs, k=1)[0]

        # Remove from both structures
        node_id = chosen[4]
        st_key = self._node_to_subtree.pop(node_id, None)
        if st_key is not None:
            self._remove_from_subtree_queue(chosen, st_key)
        self.priority_queue.remove(chosen)
        heapq.heapify(self.priority_queue)

        _, _, _, _, _, node = chosen
        node.status = NodeStatus.EXPLORING
        return node

    def add_node(self, node: TreeNode):
        if node.status != NodeStatus.CREATED:
            return

        # --- Compute effective score ---
        if node.depth < self.min_depth:
            effective_score = self.shallow_score
        else:
            effective_score = node.selection_score

        node.metadata["effective_score"] = effective_score

        # --- Build entry and enqueue ---
        entry: Entry = (-effective_score, -node.log_prob, -node.cum_log_prob, random.random(), id(node), node)

        # Global queue
        heapq.heappush(self.priority_queue, entry)

        # Per-subtree queue
        st_key = self._get_subtree_key(node)
        self._node_to_subtree[id(node)] = st_key
        if st_key not in self._subtree_queues:
            self._subtree_queues[st_key] = []
        heapq.heappush(self._subtree_queues[st_key], entry)

        node.status = NodeStatus.QUEUED

        # --- Beam width pruning ---
        if len(self.priority_queue) > self.max_beam_width:
            logger.warning("Queue exceeded the limit; keep Top-K (best K).")
            self.priority_queue.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            pruned_entries = self.priority_queue[self.max_beam_width:]
            self.priority_queue = self.priority_queue[:self.max_beam_width]
            for pruned in pruned_entries:
                _, _, _, _, nid, pruned_node = pruned
                pruned_node.status = NodeStatus.CUT
                pruned_node.metadata["cut_reason"] = "beam_width"
                # Sync: remove from subtree queue
                sk = self._node_to_subtree.pop(nid, None)
                if sk is not None:
                    self._remove_from_subtree_queue(pruned, sk)
            heapq.heapify(self.priority_queue)
