# Design: ConversationState as Single Source of Truth

Status: **accepted** (2026-04-08)

This document defines the canonical representation for conversation content
in the BOA search tree. All current and future code must comply.

---

## Principles

### 1. ConversationState is the only semantic truth

`TreeNode.conversation_state` (structured OpenAI-style message list + pending
assistant suffix) is the **single authoritative representation** of a node's
conversation history.

Any function that needs the conversation content of a node must derive it from
`conversation_state`, not from tree-path concatenation.

### 2. Legacy path helpers are not semantic truth

The following are **debug / metric utilities only** and must never be used as
source of truth for model input, judging input, or conversation reconstruction:

- `TreeNode.text` / `TreeNode.token_ids` (incremental chunk content)
- `TreeNode.get_path_text()`
- `TreeNode.get_path_token_ids()`
- `collect_last_assistant_text()`
- `collect_assistant_turn_nodes()`
- `get_path_output_messages()`

They may be retained for logging, debug views, and backward-compatible metrics,
but no runtime decision (model input, judger input, cache key, environment
dispatch) may depend on them.

### 3. All text and token IDs are rendered from canonical state

When a component needs text or token IDs for a node:

```
conversation_state  ──render──>  (text, token_ids)
```

The only sanctioned render path is:

```python
build_node_model_input(node, tokenizer, model_name)
```

Which internally calls `render_conversation_state()` on `committed_messages`
via `apply_chat_template`, then appends the `pending` suffix.

No component may construct model input by concatenating `node.text` fields up
the tree, or by string-arithmetic on `path_text + seq_text`.

### 4. Single-turn is a degenerate case of multi-turn

Single-turn interaction is multi-turn with exactly one user message and one
assistant turn, followed by a terminal signal. There is no separate "single-turn
code path" or "legacy fallback" for conversation construction.

Specifically:
- Single-turn uses the same `ConversationState` → `build_node_model_input` →
  render pipeline as multi-turn.
- Single-turn assistant output goes through the same pending → commit lifecycle
  (commit happens at EOS, producing a committed assistant message).
- No code path may bypass `ConversationState` "because it's single-turn."

### 5. Commit is an explicit action

Promoting content from "in-progress generation" to "committed conversation
history" is an **explicit, auditable action**, not an implicit side effect.

Commit semantics:
- **Pending** (`conversation_state.pending`): the assistant's current unfinished
  output. Appended to via `append_pending_assistant()` during tree expansion.
  Not a committed message. Not passed to `apply_chat_template`.
- **Committed** (`conversation_state.committed_messages`): completed turns with
  role tags. Only modified by explicit commit operations.

Commit points (exhaustive list):
- `L1Expander.expand_after_eos()` — **single-turn EOS**: calls
  `commit_assistant_turn()` to promote pending into a committed assistant message.
- `L1Expander.expand_after_eos()` — **TOOL node creation**: calls
  `commit_tool_interaction()` to commit both the assistant tool-call message and
  tool result into `committed_messages`, clearing `pending`.

No code may silently convert pending to committed, or assume that pending
content has been committed because "the turn is done."

---

## What this means for existing code

| Consumer | Current state | Status |
|----------|--------------|--------|
| `build_node_model_input` | Reads `conversation_state` | ✅ No change needed |
| `Buffer.add_requests` | Calls `build_node_model_input` | ✅ No change needed |
| `JudgeWorker._build_full_response` | Uses `conversation_state.latest_assistant_text()` | ✅ Done |
| `JudgeWorker._full_response` metadata | Uses `_build_full_response()` | ✅ Done |
| `result_builder.best_safe_response` | Uses `conversation_state.latest_assistant_text()` | ✅ Done |
| `debug_view.node_brief` | `get_path_token_ids()` for length | ✅ Acceptable (debug only) |
| `attack_sampling` response construction | Uses `build_node_model_input()` / `stats["prompt"]` | ✅ Done |
| `CONVENTIONS.md` | References this doc | ✅ Done |

---

## Non-goals (for now)

- This document does not prescribe the exact refactoring plan or migration order.
- This document does not change `TreeNode.text`/`token_ids` field existence
  (they remain for debug/metrics).
- This document does not change tree topology, scoring, or search semantics.
