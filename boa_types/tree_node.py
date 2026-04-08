from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict


@dataclass
class PendingTurn:
    """
    Stable assistant suffix state for the current branch.

    `pending` stores the assistant's current unfinished output after the committed
    conversation history. It is intentionally *not* a committed message and must
    not be passed directly into `apply_chat_template`.

    Important: `text` / `token_ids` must preserve the raw generated content,
    including special characters or partially-formed structured syntax. Do not
    clean, strip, or normalize pending content before it is committed.
    """
    text: str = ""
    token_ids: List[int] = field(default_factory=list)

    def clone(self) -> 'PendingTurn':
        return PendingTurn(
            text=self.text,
            token_ids=list(self.token_ids),
        )

@dataclass
class ConversationState:
    """
    Structured conversation state for a single tree branch.

    Semantics:
    - `committed_messages` holds the committed conversation history
      (OpenAI-style message list).
      Only completed turns should be stored here.
    - `pending` holds the assistant's current unfinished suffix after the
      committed history. The active conversation state is:
      committed_messages + pending
      but pending is not itself a committed message.

    Copy-on-write: every child node clones the parent's ConversationState via
    `clone()` so siblings never share the same conversation references.
    """
    committed_messages: List[Dict[str, Any]] = field(default_factory=list)
    pending: Optional['PendingTurn'] = None

    def clone(self) -> 'ConversationState':
        """Deep-copy committed/pending conversation content."""
        return ConversationState(
            committed_messages=deepcopy(self.committed_messages),
            pending=self.pending.clone() if self.pending is not None else None,
        )


class NodeStatus(Enum):
    """Node lifecycle status."""
    CREATED = "created"        # Newly created
    EXPANDING = "expanding"    # Expanding (generating children) it means the tree sampler is expanding, and it is giving birth to the children nodes. after finished, it will be marked back as EVALUATED.

    EVALUATED = "evaluated"    # Expanded and scored
    QUEUED = "queued"          # Waiting in the queue
    EXPLORING = "exploring"    # Exploring the node, happens after it is taken from the queue by the searcher. 
    COMPLETED = "completed"    # Terminal
    JAILBREAKED = "Jailbreak" # Jailbreak succeeded
    CUT = "cut"                # Pruned


class NodeRole(Enum):
    """Role of the incremental content carried by a node."""
    ROOT = "root"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"


@dataclass
class TreeNode:
    """
    One node in BOA search tree.

    Field semantics (important):
    - `token_ids` / `text` are INCREMENTAL content produced at this node only.
      They are not the full prompt/path.
    - Full model input must be built via `build_node_model_input()` (utils/conversation_formatter.py).
      `get_path_token_ids()` / `get_path_text()` give raw incremental concatenation (debug/legacy only).
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
    
    # ------------------------------------------------------------------
    # 3) Probability / Score Metrics
    # ------------------------------------------------------------------
    # Log-probability contribution of current incremental token chunk.
    log_prob: float = 0.0
    # Cumulative log-probability from root to this node path.
    cum_log_prob: float = 0.0
    # Aggregated node score (mean of `scores`) used for logging/stats.
    score: float = 250
    # Tail-aware selection score for search priority (separate from mean).
    selection_score: float = 250

    # ------------------------------------------------------------------
    # 4) Lifecycle / Control Flags
    # ------------------------------------------------------------------
    # Runtime lifecycle state (CREATED -> ... -> QUEUED/EXPLORING/...).
    status: NodeStatus = NodeStatus.CREATED
    
    # ------------------------------------------------------------------
    # 5) Node Attribution / Context / Raw Judger Samples
    # ------------------------------------------------------------------
    # Extra fields for runtime logic (e.g. expander flags like `is_cut`).
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Which role produced this node's incremental content.
    role: NodeRole = NodeRole.ASSISTANT

    # Which task / interaction policy this path should use.
    task_type: str = "single_turn"

    # Raw per-sample judger scores before aggregation to `score`.
    scores: List[float] = field(default_factory=list)

    # Prompt-level context shared across all nodes in a path (e.g. environments, tools).
    # Set once on the root node; all descendants inherit it automatically via __post_init__.
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 6) Structured Conversation State
    # ------------------------------------------------------------------
    # Holds the path-specific conversation state for unified single/multi-turn
    # interaction. Children inherit a clone of the parent's state via
    # __post_init__, so every node always owns a concrete ConversationState.
    conversation_state: 'ConversationState' = field(default_factory=ConversationState)

    def __post_init__(self):
        """Post-init derived field computation."""
        # 1. Compute depth automatically
        if self.parent:
            self.depth = self.parent.depth + 1
            self.task_type = self.parent.task_type
            self.prompt_metadata = self.parent.prompt_metadata
            # 2. Accumulate probability automatically (if cum_log_prob is not provided)
            if self.cum_log_prob == 0.0:
                self.cum_log_prob = self.parent.cum_log_prob + self.log_prob
            # 3. Clone parent conversation_state so siblings never share references.
            # Preserve explicitly provided non-empty state when present.
            if (
                not self.conversation_state.committed_messages
                and self.conversation_state.pending is None
            ):
                self.conversation_state = self.parent.conversation_state.clone()
            # 4. Cache root-child ancestor for O(1) subtree identification.
            if self.parent.parent is None:
                # Parent is root → this node is a depth-1 (root child)
                self._root_child: Optional['TreeNode'] = self
            else:
                self._root_child = getattr(self.parent, '_root_child', None)
        else:
            self.depth = 0
            if self.cum_log_prob == 0.0:
                self.cum_log_prob = self.log_prob
            self._root_child: Optional['TreeNode'] = None

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
            if n.role == NodeRole.TOOL:
                msgs = n.metadata.get("structured_messages")
                if isinstance(msgs, list):
                    messages.extend(msgs)
        return messages

    def mark_as_jailbreak(self):
        """Mark current node as jailbreak hit and annotate ancestor jailbreak path."""
        node = self
        first = True
        while node is not None:
            node.metadata["is_jailbreak_path"] = True
            if first:
                node.metadata["is_jailbreak_hit"] = True
                node.status = NodeStatus.JAILBREAKED
                first = False
            node = node.parent

    def __repr__(self):
        """Compact repr for debugging."""
        return (f"<TreeNode d={self.depth} score={self.score:.2f} "
                f"prob={self.log_prob:.2f} txt={repr(self.text[:20])} "
                f"status={self.status.name}>")


def collect_assistant_turn_nodes(node: TreeNode) -> List[TreeNode]:
    """Collect consecutive assistant nodes ending at ``node`` for the current turn."""
    path: List[TreeNode] = []
    current: Optional[TreeNode] = node
    while current is not None and current.role == NodeRole.ASSISTANT:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def collect_last_assistant_text(node: TreeNode, tokenizer: Any = None) -> str:
    """Collect the full text of the last assistant turn ending at *node*.

    When *tokenizer* is provided, all token IDs are decoded together so that
    SentencePiece-style tokenizers correctly reconstruct inter-word spaces.
    Without a tokenizer the per-node ``.text`` fields are concatenated, which
    may lose whitespace for single-token chunks.
    """
    nodes = collect_assistant_turn_nodes(node)
    if tokenizer is not None:
        all_ids: List[int] = []
        for n in nodes:
            all_ids.extend(n.token_ids)
        if all_ids:
            return tokenizer.decode(all_ids)
        return ""
    return "".join(n.text for n in nodes)
