import math
from typing import Optional
from boa_types.tree_node import TreeNode, NodeStatus
from utils.logger import setup_logger

logger = setup_logger("MctsSearcher")


class MctsSearcher:
    """Classic MCTS with UCT tree-policy selection.

    Selection traverses the tree from root at every step, using UCT at
    each internal node to pick the most promising child, until a
    frontier (QUEUED) leaf is reached.

    Four MCTS phases mapped to the executor loop:
      Selection    — select_next_node(): tree-policy walk root → frontier
      Expansion    — executor generates children for the selected node
      Simulation   — executor scores each child (selection_score)
      Backprop     — add_node(): propagate reward root-ward

    UCT(node) = Q/N  +  C * sqrt( ln(N_parent) / N )
    """

    def __init__(self, max_beam_width: int = 20000,
                 exploration_constant: float = 1.414):
        self.max_beam_width = max_beam_width
        self.exploration_constant = exploration_constant
        self.root: Optional[TreeNode] = None
        self.total_visits: int = 0

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _ensure_mcts_metadata(self, node: TreeNode):
        if "mcts_visits" not in node.metadata:
            node.metadata["mcts_visits"] = 0
            node.metadata["mcts_value"] = 0.0

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: TreeNode, reward: float):
        """Walk from *node* to root, incrementing visits and accumulating reward."""
        current = node
        while current is not None:
            self._ensure_mcts_metadata(current)
            current.metadata["mcts_visits"] += 1
            current.metadata["mcts_value"] += reward
            current = current.parent
        self.total_visits += 1

    # ------------------------------------------------------------------
    # UCT score
    # ------------------------------------------------------------------

    def _uct_score(self, node: TreeNode) -> float:
        self._ensure_mcts_metadata(node)
        n = node.metadata["mcts_visits"]
        if n == 0:
            return float('inf')

        exploitation = node.metadata["mcts_value"] / n

        parent = node.parent
        if parent is not None:
            self._ensure_mcts_metadata(parent)
            parent_n = max(parent.metadata["mcts_visits"], 1)
        else:
            parent_n = max(self.total_visits, 1)

        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_n) / n
        )
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Selection — classic tree-policy walk
    # ------------------------------------------------------------------

    def _tree_policy(self) -> Optional[TreeNode]:
        """Traverse from root using UCT, with backtracking on dead ends."""
        if self.root is None:
            return None

        # DFS stack: each entry is (node, remaining_children_sorted_by_uct_desc)
        # Start from root.
        stack = [(self.root, None)]

        while stack:
            node, remaining = stack[-1]

            if node.status == NodeStatus.QUEUED:
                return node

            # First visit to this node: build sorted candidate list.
            if remaining is None:
                candidates = [
                    c for c in node.children
                    if c.status not in (NodeStatus.CUT, NodeStatus.JAILBREAKED)
                    and (c.status == NodeStatus.QUEUED or c.children)
                ]
                # Sort ascending so we can pop best (last) cheaply.
                candidates.sort(key=self._uct_score)
                stack[-1] = (node, candidates)
                remaining = candidates

            if not remaining:
                # Dead end — backtrack to parent, try next sibling.
                stack.pop()
                continue

            # Pick best remaining child, descend.
            best = remaining.pop()
            stack.append((best, None))

        return None

    def select_next_node(self) -> Optional[TreeNode]:
        node = self._tree_policy()
        if node is None:
            return None

        node.mark_exploring()
        return node

    # ------------------------------------------------------------------
    # Expansion bookkeeping + backprop
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_selection_score(scores: list[float]) -> float:
        if not scores:
            return 250.0
        return sum(scores) / len(scores)

    def add_node(self, node: TreeNode):
        if node.status != NodeStatus.CREATED:
            return

        if self.root is None:
            self.root = node

        node.selection_score = self._compute_selection_score(node.scores)
        # Backpropagate reward to ancestors.
        reward = node.selection_score
        self._backpropagate(node, reward)

        # Record effective score for diagnostics.
        node.metadata["effective_score"] = self._uct_score(node)

        node.mark_queued()
