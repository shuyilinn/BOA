"""
conversation_formatter.py
=========================
Centralised rendering layer between semantic ConversationState and the raw
token-id sequences fed to the LLM.

Public API
----------
parse_initial_prompt      – turn a raw prompt string into a ConversationState
render_conversation_state – render committed conversation state on demand
extend_pending_assistant  – append one assistant chunk to pending suffix state
build_node_model_input    – build the exact model input for a TreeNode
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from boa_types.tree_node import ConversationState, PendingTurn, TreeNode


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_initial_prompt(
    prompt: str,
    model_name: str = "",
) -> tuple[ConversationState, List[Dict[str, Any]]]:
    """
    Parse a raw prompt string into a ConversationState plus any tool schema list.

    Accepts two formats:
    - JSON payload: {"messages": [...], "tools": [...]}
    - Plain text: wrapped into a single user message.
    """
    messages: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []

    try:
        payload = json.loads(prompt)
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            messages = payload["messages"]
            maybe_tools = payload.get("tools")
            if isinstance(maybe_tools, list):
                tools = maybe_tools
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    if not messages:
        if "gemma-3" in model_name.lower():
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": prompt}]

    return ConversationState(committed_messages=messages), tools


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_conversation_state(
    state: ConversationState,
    tokenizer: Any,
    model_name: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[int]]:
    """
    Render committed conversation history to text + token_ids using the tokenizer's
    apply_chat_template (or a plain-text fallback).
    """
    rendered_text: str
    rendered_token_ids: List[int]

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            if "Qwen3-8B" in model_name:
                rendered_text = tokenizer.apply_chat_template(
                    state.committed_messages,
                    tools=tools or None,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                rendered_text = tokenizer.apply_chat_template(
                    state.committed_messages,
                    tools=tools or None,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered_token_ids = tokenizer.encode(rendered_text)
        else:
            rendered_text, rendered_token_ids = _plain_text_fallback(state, tokenizer)
    except Exception:
        rendered_text, rendered_token_ids = _plain_text_fallback(state, tokenizer)

    return rendered_text, rendered_token_ids


def _plain_text_fallback(
    state: ConversationState,
    tokenizer: Any,
) -> tuple[str, List[int]]:
    """Fallback rendering when apply_chat_template is unavailable or fails."""
    content = ""
    if state.committed_messages:
        last = state.committed_messages[-1].get("content", "")
        if isinstance(last, list):
            content = " ".join(
                item.get("text", "") for item in last if isinstance(item, dict)
            )
        else:
            content = str(last)
    if not content.endswith(". "):
        content = content + ". "
    return content, tokenizer.encode(content)


def extend_pending_assistant(
    state: ConversationState,
    chunk_text: str,
    chunk_ids: List[int],
) -> ConversationState:
    """
    Extend the assistant pending suffix with one newly generated chunk.

    Pending content must preserve raw text/token content exactly, including
    special characters and partially-formed structured syntax.
    """
    new_state = state.clone()
    if new_state.pending is None:
        new_state.pending = PendingTurn(
            text=chunk_text,
            token_ids=list(chunk_ids),
        )
    else:
        new_state.pending.text = new_state.pending.text + chunk_text
        new_state.pending.token_ids.extend(list(chunk_ids))
    return new_state


def build_node_model_input(
    node: TreeNode,
    tokenizer: Any,
    model_name: str = "",
) -> tuple[str, List[int], int]:
    """
    Build the active model input from committed history plus the assistant
    pending suffix, if present.

    Pending content is appended *after* rendering committed messages because
    it is an unfinished assistant suffix and must not be passed directly into
    `apply_chat_template`.
    """
    tool_schemas = _resolve_tool_schemas(node)
    rendered_text, rendered_ids = render_conversation_state(
        node.conversation_state,
        tokenizer,
        model_name=model_name,
        tools=tool_schemas,
    )
    pending = node.conversation_state.pending
    if pending is None:
        return rendered_text, rendered_ids, 0
    # Decode all pending token IDs together so SentencePiece tokenizers
    # correctly reconstruct inter-word spaces (per-chunk text loses them).
    pending_text = tokenizer.decode(pending.token_ids) if pending.token_ids else ""
    return (
        rendered_text + pending_text,
        rendered_ids + list(pending.token_ids),
        len(pending.token_ids),
    )


def _resolve_tool_schemas(node: TreeNode) -> Optional[List[Dict[str, Any]]]:
    metadata = getattr(node, "prompt_metadata", {}) or {}
    tools = metadata.get("tools_openai")
    return tools if isinstance(tools, list) and tools else None
