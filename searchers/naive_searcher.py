import heapq
import random
from typing import Optional, List, Tuple
from boa_types.tree_node import TreeNode, NodeStatus
from utils.logger import setup_logger

logger = setup_logger("NaiveSearcher")

class NaiveSearcher:
    def __init__(self, max_beam_width: int = 20000):
        # Core: maintain a priority queue.
        # Stored as: (-score, -log_prob, -cum_log_prob, random_tiebreak, id(node), node)
        # heapq is a min-heap; using negatives keeps higher score at the top.
        # Within equal score, random key improves exploration; id(node) is a deterministic fallback.
        self.priority_queue: List[Tuple[float, float, float, float, int, TreeNode]] = []
        self.max_beam_width = max_beam_width

    def select_next_node(self) -> Optional[TreeNode]:
        """
        [Pop] Pop the highest-scoring node from the queue.
        """
        if not self.priority_queue:
            return None
        
        # Pop heap top (smallest -score -> largest score).
        # Entry is (-score, -log_prob, -cum_log_prob, random_tiebreak, id(node), node).
        _, _, _, _, _, node = heapq.heappop(self.priority_queue)
        
        # Update status.
        node.status = NodeStatus.EXPLORING
        return node

    def add_node(self, node: TreeNode):
        """
        [Push] Push a new node and enforce beam width.
        """
        # Avoid re-queuing the same node.
        if node.status != NodeStatus.CREATED:
            return

        # 1. Build the priority tuple.
        # Tie-break order:
        #   1) higher score first
        #   2) higher single-step log-probability (current chunk likelihood)
        #   3) random tie-break to increase exploration diversity
        #   4) smaller id as deterministic technical fallback
        # NOTE: `node` itself should not participate in comparisons.
        entry = (-node.score, -node.log_prob, -node.cum_log_prob, random.random(), id(node), node)

        
        # 2. Push into the heap.
        heapq.heappush(self.priority_queue, entry)
        node.status = NodeStatus.QUEUED
        
        # 3. Enforce beam width.
        if len(self.priority_queue) > self.max_beam_width:
            logger.warning(f"Queue exceeded the limit; keep Top-K (best K).")
            # Queue exceeded the limit; keep Top-K (best K).
            # Since we store (-score), smaller is better.
            # After sorting, the front contains the best entries; the tail contains the worst.
            
            # Sorting cost: O(N log N). For N=10000 this is very fast in Python (milliseconds).
            # This "lazy truncation" is simpler than checking min/max on every push.
            self.priority_queue.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            
            # Trim the extra tail entries.
            pruned_entries = self.priority_queue[self.max_beam_width:]
            self.priority_queue = self.priority_queue[:self.max_beam_width]
            
            # Mark pruned nodes.
            for _, _, _, _, _, pruned_node in pruned_entries:
                pruned_node.status = NodeStatus.CUT
                pruned_node.is_cut = True
            
            # Re-heapify (O(N)) to restore heap property for the next heappop.
            heapq.heapify(self.priority_queue)
