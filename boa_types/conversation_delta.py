"""
conversation_delta.py
=====================
Tagged delta types describing what a single TreeNode contributes to the
conversation relative to its parent.

Every non-root TreeNode carries exactly one ConversationDelta that records
the semantic operation it performed on the conversation.  This replaces the
previous implicit encoding where the "delta" had to be reverse-engineered
from ``node.role``, ``node.text``, and scattered ``metadata`` keys.

Variant summary
---------------
- ``AssistantChunk``:  extends the pending assistant suffix (tree expansion)
- ``AssistantCommit``: commits a complete assistant turn (single-turn EOS)
- ``ToolInteractionCommit``: commits assistant tool_call + tool result pair
- ``Noop``:            control / structural node with no conversation effect

Usage::

    delta = node.delta
    if isinstance(delta, AssistantChunk):
        ...  # access delta.text, delta.token_ids
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class AssistantChunk:
    """Node extended the pending assistant suffix with a generated chunk.

    This is the most common delta type during tree expansion (L2Expander).
    The chunk has not been committed — it lives in ``conversation_state.pending``.
    """
    text: str
    token_ids: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class AssistantCommit:
    """Node committed a complete assistant turn (no tool calls).

    Produced at single-turn EOS: pending content is promoted to a committed
    ``{"role": "assistant", "content": text}`` message.
    """
    text: str


@dataclass(frozen=True)
class ToolInteractionCommit:
    """Node committed an assistant-tool_call + tool-result message pair.

    ``structured_messages`` is the list of two OpenAI-style dicts
    (assistant + tool) that were appended to ``committed_messages``.
    """
    structured_messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_result_text: str = ""


@dataclass(frozen=True)
class Noop:
    """Node carries no conversation-level change.

    Used for the root node and any future control / structural nodes.
    """
    reason: str = ""


# Union type for annotation convenience.
ConversationDelta = AssistantChunk | AssistantCommit | ToolInteractionCommit | Noop
