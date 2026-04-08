import heapq
import random
from typing import Optional, List, Tuple
from boa_types.tree_node import TreeNode, NodeStatus
from utils.logger import setup_logger

logger = setup_logger("NaiveSearcher")

class NaiveSearcher:
    """Greedy search strategy: always select the highest-scoring node."""

    def __init__(self, max_beam_width: int = 20000):
        self.priority_queue: List[Tuple[float, float, float, float, int, TreeNode]] = []
        self.max_beam_width = max_beam_width

    def select_next_node(self) -> Optional[TreeNode]:
        if not self.priority_queue:
            return None

        _, _, _, _, _, node = heapq.heappop(self.priority_queue)
        node.status = NodeStatus.EXPLORING
        return node

    def add_node(self, node: TreeNode):
        if node.status != NodeStatus.CREATED:
            return

        effective_score = node.selection_score
        node.metadata["effective_score"] = effective_score

        entry = (-effective_score, -node.log_prob, -node.cum_log_prob, random.random(), id(node), node)
        heapq.heappush(self.priority_queue, entry)
        node.status = NodeStatus.QUEUED

        if len(self.priority_queue) > self.max_beam_width:
            logger.warning(f"Queue exceeded the limit; keep Top-K (best K).")
            self.priority_queue.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            pruned_entries = self.priority_queue[self.max_beam_width:]
            self.priority_queue = self.priority_queue[:self.max_beam_width]
            for _, _, _, _, _, pruned_node in pruned_entries:
                pruned_node.status = NodeStatus.CUT
                pruned_node.is_cut = True
            heapq.heapify(self.priority_queue)
