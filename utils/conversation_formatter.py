"""
conversation_formatter.py
=========================
Centralised rendering layer between semantic ConversationState and the raw
token-id sequences / text consumed by downstream components.

Public API
----------
**Free functions** (backward-compatible, delegate to ``ConversationRenderer``):

- ``parse_initial_prompt``      – raw prompt string → ConversationState + tools
- ``render_conversation_state`` – committed messages → (text, token_ids)
- ``build_node_model_input``    – full model input for a TreeNode

**Class** (preferred for new code):

- ``ConversationRenderer``      – stateful renderer bound to tokenizer + model
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from boa_types.tree_node import ConversationState, TreeNode


# ===================================================================
# Parsing (stateless)
# ===================================================================

def parse_initial_prompt(
    prompt: str,
    model_name: str = "",
) -> Tuple[ConversationState, List[Dict[str, Any]]]:
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

    return ConversationState(
        committed_messages=messages,
        tools=tools if tools else None,
    ), tools


# ===================================================================
# ConversationRenderer (stateful — preferred for new code)
# ===================================================================

@dataclass
class ConversationRenderer:
    """Renders ConversationState into text / token-ids for different consumers.

    Bind once to a tokenizer + model_name, then call the appropriate render
    method.  This keeps rendering logic in one place and avoids threading
    tokenizer/model_name through every call site.

    All render methods are **pure reads** — they never mutate the
    ConversationState they receive.
    """
    tokenizer: Any
    model_name: str = ""

    # ------------------------------------------------------------------
    #  Model input (LLM forward pass)
    # ------------------------------------------------------------------

    def render_model_input(
        self,
        node: TreeNode,
    ) -> Tuple[str, List[int], int]:
        """Build the exact token sequence for one LLM forward pass.

        Returns ``(text, token_ids, pending_token_count)`` where:
        - ``text`` / ``token_ids``: committed messages rendered via
          ``apply_chat_template`` + pending assistant suffix appended raw.
        - ``pending_token_count``: number of tokens from the pending suffix
          (used by callers that need the ``base_generated_len`` offset).
        """
        tool_schemas = _resolve_tool_schemas(node)
        rendered_text, rendered_ids = _render_committed(
            node.conversation_state,
            self.tokenizer,
            self.model_name,
            tools=tool_schemas,
        )
        pending = node.conversation_state.pending
        if pending is None:
            return rendered_text, rendered_ids, 0
        # Decode all pending token IDs together so SentencePiece tokenizers
        # correctly reconstruct inter-word spaces.
        pending_text = (
            self.tokenizer.decode(pending.token_ids)
            if pending.token_ids else ""
        )
        return (
            rendered_text + pending_text,
            rendered_ids + list(pending.token_ids),
            len(pending.token_ids),
        )

    def render_model_input_ids(self, node: TreeNode) -> List[int]:
        """Convenience: return only token IDs (common hot path)."""
        _, ids, _ = self.render_model_input(node)
        return ids

    # ------------------------------------------------------------------
    #  Committed only (for judger, cache key, etc.)
    # ------------------------------------------------------------------

    def render_committed(
        self,
        state: ConversationState,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[int]]:
        """Render committed messages only (no pending suffix)."""
        return _render_committed(
            state, self.tokenizer, self.model_name, tools=tools,
        )

    # ------------------------------------------------------------------
    #  Debug transcript
    # ------------------------------------------------------------------

    def render_debug_transcript(
        self,
        node: TreeNode,
        *,
        max_chars: int = 2000,
    ) -> str:
        """Human-readable multi-turn transcript for logging / debug views.

        Format::

            [system] You are a helpful assistant.
            [user] Do the thing.
            [assistant] Sure, I'll call the tool.
            [tool] {"success": true}
            [assistant|pending] Here is the result so f...
        """
        lines: List[str] = []
        for msg in node.conversation_state.committed_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content")
            if content is None:
                # assistant message with tool_calls only
                tool_calls = msg.get("tool_calls", [])
                names = [
                    tc.get("function", {}).get("name", "?")
                    for tc in tool_calls
                ]
                content = f"<tool_call: {', '.join(names)}>"
            elif isinstance(content, list):
                # Multimodal content list
                content = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict)
                )
            lines.append(f"[{role}] {content}")

        pending = node.conversation_state.pending
        if pending is not None and pending.text:
            lines.append(f"[assistant|pending] {pending.text}")

        transcript = "\n".join(lines)
        if len(transcript) > max_chars:
            return "..." + transcript[-(max_chars - 3):]
        return transcript


# ===================================================================
# Free functions (backward-compatible wrappers)
# ===================================================================

def render_conversation_state(
    state: ConversationState,
    tokenizer: Any,
    model_name: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[int]]:
    """Render committed conversation history to text + token_ids.

    Backward-compatible wrapper around the internal ``_render_committed``.
    New code should use ``ConversationRenderer.render_committed`` instead.
    """
    return _render_committed(state, tokenizer, model_name, tools=tools)


def build_node_model_input(
    node: TreeNode,
    tokenizer: Any,
    model_name: str = "",
) -> Tuple[str, List[int], int]:
    """Build the active model input for a TreeNode.

    Backward-compatible wrapper.  New code should use
    ``ConversationRenderer.render_model_input`` instead.
    """
    renderer = ConversationRenderer(tokenizer=tokenizer, model_name=model_name)
    return renderer.render_model_input(node)


# ===================================================================
# Internal helpers
# ===================================================================

def _render_committed(
    state: ConversationState,
    tokenizer: Any,
    model_name: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[int]]:
    """Core rendering: committed messages → (text, token_ids)."""
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
            rendered_text, rendered_token_ids = _plain_text_fallback(
                state, tokenizer,
            )
    except Exception:
        rendered_text, rendered_token_ids = _plain_text_fallback(
            state, tokenizer,
        )
    return rendered_text, rendered_token_ids


def _plain_text_fallback(
    state: ConversationState,
    tokenizer: Any,
) -> Tuple[str, List[int]]:
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


def _resolve_tool_schemas(node: TreeNode) -> Optional[List[Dict[str, Any]]]:
    # Primary: canonical location on ConversationState.
    cs_tools = node.conversation_state.tools
    if isinstance(cs_tools, list) and cs_tools:
        return cs_tools
    # Fallback: legacy prompt_metadata (kept during migration).
    metadata = getattr(node, "prompt_metadata", {}) or {}
    tools = metadata.get("tools_openai")
    return tools if isinstance(tools, list) and tools else None
