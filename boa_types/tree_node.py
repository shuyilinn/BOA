from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from boa_types.conversation_delta import ConversationDelta


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
    Single source of truth for conversation content on a tree branch.

    See DESIGN_conversation_state.md for the canonical specification.

    Fields:
    - ``committed_messages``: completed turns (OpenAI-style message dicts).
    - ``pending``: the assistant's unfinished suffix *after* committed history.
      Active conversation = committed_messages + pending.
      Pending is NOT a committed message and must NOT be passed to
      ``apply_chat_template``.
    - ``tools``: OpenAI-style tool/function schema list for this conversation.
      Set once from the prompt payload; inherited by all descendants via clone.

    Copy-on-write: every child node clones the parent's ConversationState via
    ``clone()`` so siblings never share the same mutable references.
    """
    committed_messages: List[Dict[str, Any]] = field(default_factory=list)
    pending: Optional['PendingTurn'] = None
    tools: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    #  Clone
    # ------------------------------------------------------------------

    def clone(self) -> 'ConversationState':
        """Deep-copy all mutable conversation content."""
        return ConversationState(
            committed_messages=deepcopy(self.committed_messages),
            pending=self.pending.clone() if self.pending is not None else None,
            tools=deepcopy(self.tools) if self.tools is not None else None,
        )

    # ------------------------------------------------------------------
    #  Mutation: pending assistant content
    # ------------------------------------------------------------------

    def append_pending_assistant(
        self,
        chunk_text: str,
        chunk_ids: List[int],
    ) -> None:
        """Extend the assistant pending suffix with one generated chunk.

        Raw text and token IDs are preserved exactly (no stripping, no
        normalization) because the content may contain partial structured
        syntax (e.g. half of a ``<tool_call>`` tag).

        This is an in-place mutation.  Callers that need copy-on-write
        semantics should ``clone()`` first.
        """
        if self.pending is None:
            self.pending = PendingTurn(
                text=chunk_text,
                token_ids=list(chunk_ids),
            )
        else:
            self.pending.text += chunk_text
            self.pending.token_ids.extend(chunk_ids)

    # ------------------------------------------------------------------
    #  Commit: assistant turn (single-turn EOS or pre-tool-call)
    # ------------------------------------------------------------------

    def commit_assistant_turn(self, text: str) -> None:
        """Promote the current assistant generation into a committed message.

        After this call ``pending`` is ``None``.  The committed message is a
        plain ``{"role": "assistant", "content": text}`` dict — suitable for
        single-turn EOS or any case where the assistant turn contains no
        tool calls.

        For assistant turns that include tool calls, use
        ``commit_tool_interaction`` instead (it commits both the assistant
        tool-call message and the tool result in one atomic step).
        """
        self.committed_messages.append({"role": "assistant", "content": text})
        self.pending = None

    # ------------------------------------------------------------------
    #  Commit: tool interaction (assistant tool_call + tool result)
    # ------------------------------------------------------------------

    def commit_tool_interaction(
        self,
        structured_messages: List[Dict[str, Any]],
    ) -> None:
        """Commit an assistant-tool_call + tool-result message pair.

        ``structured_messages`` is expected to be a list of two dicts::

            [
                {"role": "assistant", "tool_calls": [...]},
                {"role": "tool", "content": ..., "tool_call_id": ..., "name": ...},
            ]

        This mirrors the format produced by environments (e.g.
        ``AgentSafetyBenchEnvironment``).

        After this call ``pending`` is ``None`` — the assistant content that
        led to the tool call is superseded by the structured representation
        in ``structured_messages[0]``.
        """
        self.committed_messages.extend(structured_messages)
        self.pending = None

    # ------------------------------------------------------------------
    #  Read: accessors (no mutation)
    # ------------------------------------------------------------------

    def committed_only(self) -> List[Dict[str, Any]]:
        """Return the committed message list (reference, not copy)."""
        return self.committed_messages

    def latest_assistant_text(self) -> str:
        """Return the current assistant text (pending or last committed).

        Priority:
        1. If ``pending`` exists, return its text (in-progress generation).
        2. Otherwise, return the content of the last committed assistant
           message, or ``""`` if none exists.
        """
        if self.pending is not None:
            return self.pending.text
        # Walk committed in reverse to find the last assistant message.
        for msg in reversed(self.committed_messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                # tool_calls-only assistant message has no text content
                if content is None:
                    return ""
        return ""

    def pending_token_ids(self) -> List[int]:
        """Return pending token IDs, or empty list if nothing is pending."""
        if self.pending is not None:
            return self.pending.token_ids
        return []

    def has_pending(self) -> bool:
        """Whether there is uncommitted assistant content."""
        return self.pending is not None


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
    # Whether this node was pruned by tau threshold or other cut logic.
    is_cut: bool = False

    # ------------------------------------------------------------------
    # 5) Node Attribution / Context / Raw Judger Samples
    # ------------------------------------------------------------------
    # Extra fields for runtime logic (e.g. expander metadata like cut_reason).
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

    # ------------------------------------------------------------------
    # 7) Conversation Delta
    # ------------------------------------------------------------------
    # What this node contributed to the conversation relative to its parent.
    # Set explicitly by the creator (L2Expander, L1Expander, Executor).
    # None means "not yet tagged" (migration compat); will become mandatory.
    delta: Optional['ConversationDelta'] = None

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
        # Promote metadata["is_cut"] to the dataclass field so callers
        # can read child.is_cut instead of digging into metadata.
        if not self.is_cut and self.metadata.get("is_cut"):
            self.is_cut = True

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

    # ==========================================
    #  Canonical conversation accessors
    # ==========================================
    # These delegate to conversation_state and are the stable public API
    # for reading conversation content.  See DESIGN_conversation_state.md.

    @property
    def conversation(self) -> 'ConversationState':
        """The canonical conversation state for this node's branch."""
        return self.conversation_state

    @property
    def committed_conversation(self) -> List[Dict[str, Any]]:
        """Committed message list (completed turns only)."""
        return self.conversation_state.committed_only()

    @property
    def pending_turn(self) -> Optional['PendingTurn']:
        """The in-progress assistant suffix, or None."""
        return self.conversation_state.pending

    @property
    def latest_assistant_text(self) -> str:
        """Current assistant text — pending if present, else last committed."""
        return self.conversation_state.latest_assistant_text()

    # ==========================================
    #  Legacy backtrace methods (debug / metrics only)
    # ==========================================
    # These concatenate incremental node.text / node.token_ids up the tree.
    # They do NOT represent the canonical conversation and must NOT be used
    # for model input, judger input, or cache keys.
    # See DESIGN_conversation_state.md §2.

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

    # ------------------------------------------------------------------
    #  Explicit status transitions
    # ------------------------------------------------------------------

    def mark_created(self) -> None:
        self.status = NodeStatus.CREATED

    def mark_expanding(self) -> None:
        self.status = NodeStatus.EXPANDING

    def mark_evaluated(self) -> None:
        self.status = NodeStatus.EVALUATED

    def mark_queued(self) -> None:
        self.status = NodeStatus.QUEUED

    def mark_exploring(self) -> None:
        self.status = NodeStatus.EXPLORING

    def mark_completed(self) -> None:
        if self.status != NodeStatus.JAILBREAKED:
            self.status = NodeStatus.COMPLETED

    def mark_cut(self) -> None:
        self.status = NodeStatus.CUT
        self.is_cut = True

    # ------------------------------------------------------------------
    #  Score mutations
    # ------------------------------------------------------------------

    def add_score(self, score: float) -> None:
        self.scores.append(float(score))

    def add_scores(self, new_scores: list) -> None:
        self.scores.extend(new_scores)

    def reset_scores(self) -> None:
        self.scores = []

    def set_aggregate_score(self, fallback: float = 250.0) -> None:
        """Compute mean of per-sample scores and store as aggregate score."""
        self.score = float(sum(self.scores) / len(self.scores)) if self.scores else fallback

    # ------------------------------------------------------------------

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


