from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict

class NodeStatus(Enum):
    """Node lifecycle status."""
    CREATED = "created"        # Newly created
    EXPANDING = "expanding"    # Expanding (generating children) it means the tree sampler is expanding, and it is giving birth to the children nodes. after finished, it will be marked back as EVALUATED.
    EVALUATING = "evaluating"  # Evaluating the children nodes, it means it is in the buffer or is being evaluated by the judger.
    EVALUATED = "evaluated"    # Expanded and scored
    QUEUED = "queued"          # Waiting in the queue
    EXPLORING = "exploring"    # Exploring the node, happens after it is taken from the queue by the searcher. 
    COMPLETED = "completed"    # Terminal
    JAILBREAKED = "Jailbreak" # Jailbreak succeeded
    CUT = "cut"                # Pruned


class NodeSource(Enum):
    """Node source."""
    ROOT = "root"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    ENV = "env"


@dataclass
class TreeNode:
    """
    One node in BOA search tree.

    Field semantics (important):
    - `token_ids` / `text` are INCREMENTAL content produced at this node only.
      They are not the full prompt/path.
    - Full-path context must be read via `get_path_token_ids()` / `get_path_text()`.
    - `score` is the aggregated node score used by Searcher priority.
    - `scores` stores per-sample raw scores before aggregation into `score`.

    This convention is required by Executor/Buffer/Cache; see CONVENTIONS.md.
    """
    # ------------------------------------------------------------------
    # 1) Incremental Content (current node only; NOT full path)
    # ------------------------------------------------------------------
    # Token ids generated at this node step only.
    token_ids: List[int] = field(default_factory=list)
    # Text generated at this node step only (debug/readability).
    text: str = "" 

    
    # ------------------------------------------------------------------
    # 2) Tree Topology
    # ------------------------------------------------------------------
    # Parent node in tree; None only for root.
    parent: Optional['TreeNode'] = None
    # Direct children expanded from this node.
    children: List['TreeNode'] = field(default_factory=list, repr=False)
    # Root-based depth, auto-updated in __post_init__.
    depth: int = 0
    # Prompt token length (root path length), inherited by descendants.
    prompt_len: int = 0
    
    # ------------------------------------------------------------------
    # 3) Probability / Score Metrics
    # ------------------------------------------------------------------
    # Log-probability contribution of current incremental token chunk.
    log_prob: float = 0.0
    # Cumulative log-probability from root to this node path.
    cum_log_prob: float = 0.0
    # Aggregated node score (mean of `scores`) used for search priority.
    score: float = -1
    
    # ------------------------------------------------------------------
    # 4) Lifecycle / Control Flags
    # ------------------------------------------------------------------
    # Runtime lifecycle state (CREATED -> ... -> QUEUED/EXPLORING/...).
    status: NodeStatus = NodeStatus.CREATED
    
    # ------------------------------------------------------------------
    # 5) Metadata / Source / Raw Judger Samples
    # ------------------------------------------------------------------
    # Extra fields for runtime logic (e.g. expander flags like `is_cut`).
    metadata: Dict[str, Any] = field(default_factory=dict)

    # How this node is produced (assistant/user/tool/environment).
    source: NodeSource = NodeSource.ASSISTANT

    # Which environment policy this path should use.
    environment_type: str = "single_turn"

    # Raw per-sample judger scores before aggregation to `score`.
    scores: List[float] = field(default_factory=list)

    # Prompt-level context shared across all nodes in a path (e.g. environments, tools).
    # Set once on the root node; all descendants inherit it automatically via __post_init__.
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-init derived field computation."""
        # 1. Compute depth automatically
        if self.parent:
            self.depth = self.parent.depth + 1
            self.prompt_len = self.parent.prompt_len
            self.environment_type = self.parent.environment_type
            self.prompt_metadata = self.parent.prompt_metadata
            # 2. Accumulate probability automatically (if cum_log_prob is not provided)
            if self.cum_log_prob == 0.0:
                self.cum_log_prob = self.parent.cum_log_prob + self.log_prob
        else:
            self.depth = 0
            self.prompt_len = len(self.token_ids or [])
            if self.cum_log_prob == 0.0:
                self.cum_log_prob = self.log_prob

    # ==========================================
    #      Core methods
    # ==========================================

    def add_child(self, token_ids: List[int], text: str, log_prob: float, **kwargs) -> 'TreeNode':
        """Factory helper: quickly create and attach a child node."""
        child = TreeNode(
            token_ids=token_ids,
            text=text,
            log_prob=log_prob,
            parent=self,
            metadata=kwargs
        )
        self.children.append(child)
        return child

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_path_token_ids(self) -> List[int]:
        """
        [Backtrace] Get the full Token ID sequence from root to this node.
        Used to feed the LLM for KV-cache inference.
        """
        path = []
        node = self
        while node:
            # Note: list concatenation can be expensive; consider deque or preallocation in production.
            # Here we build in reverse order for clarity.
            if node.token_ids:
                path.extend(reversed(node.token_ids))
            node = node.parent
        return list(reversed(path))

    def get_path_text(self) -> str:
        """
        [Backtrace] Get the full text from root to this node.
        Used for Judger scoring or logging.
        """
        texts = []
        node = self
        while node:
            if node.text:
                texts.append(node.text)
            node = node.parent
        return "".join(reversed(texts))

    def get_path_output_messages(self) -> List[Dict[str, Any]]:
        """
        [Backtrace] Reconstruct structured output messages from TOOL nodes along the path.

        Mirrors the messages list built by eval.py lines 267-268: for each tool
        interaction in the path, collect the stored (assistant tool_calls + tool result)
        message pair.  The caller is responsible for appending the final assistant
        text response (if any).
        """
        path: List['TreeNode'] = []
        node: Optional['TreeNode'] = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()

        messages: List[Dict[str, Any]] = []
        for n in path:
            if n.source == NodeSource.TOOL:
                msgs = n.metadata.get("structured_messages")
                if isinstance(msgs, list):
                    messages.extend(msgs)
        return messages

    def mark_as_jailbreak(self):
        """Mark current node as jailbreak hit and annotate ancestor jailbreak path."""
        node = self
        first = True
        while node is not None:
            if first:
                node.metadata["is_jailbreak_hit"] = True
                node.status = NodeStatus.JAILBREAKED
                first = False
            else:
                node.metadata["is_jailbreak_path"] = True
            node = node.parent
        
    def __repr__(self):
        """Compact repr for debugging."""
        return (f"<TreeNode d={self.depth} score={self.score:.2f} "
                f"prob={self.log_prob:.2f} txt={repr(self.text[:20])} "
                f"status={self.status.name}>")
