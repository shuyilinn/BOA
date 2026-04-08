# BOA Conventions (Field Ownership and Lifecycle)

This document defines implementation conventions for `TreeNode` semantics and module boundaries.
When adding or refactoring modules, align with these conventions first and keep this file updated if behavior changes.

## TreeNode Field Semantics

`TreeNode` contains two data categories that are easy to confuse:

- **Incremental fields**
  - `token_ids` / `text`: only the newly added token/text segment from parent to current node.
  - `log_prob`: only the log-probability of this incremental generation step (when applicable).
- **Canonical conversation state**
  - `conversation_state` (`ConversationState`): the single source of truth for a node's conversation history.
    Contains `committed_messages` (completed turns) + `pending` (unfinished assistant suffix) + `tools` (schemas).
  - Use `build_node_model_input(node, tokenizer, model_name)` to render model input from canonical state.
  - Use `node.conversation_state.latest_assistant_text()` to read the current assistant output.
- **Legacy debug helpers** (do NOT use for runtime decisions)
  - `get_path_token_ids()` / `get_path_text()`: incremental concatenation, retained for debug/metrics only.

Convention: **All model input, judger input, cache keys, and conversation content must derive from `conversation_state`. Never use `token_ids`/`text` concatenation as semantic truth. See `DESIGN_conversation_state.md` for full specification.**

## Probability Semantics (Sampling vs Tau / log_p)

To avoid inconsistent probability semantics across modules, use the following rules:

- Candidate filtering/sampling may use `temperature + top_p/top_k`, including re-normalized sampling inside a candidate subset.
- But cumulative `tau` / `log_p` (for example `cum_log_prob`) must use the selected token probability from the **original full-softmax distribution**:
  - Select token from the filtered candidate set.
  - Then look up that token probability `p_raw` in full softmax and accumulate `log(p_raw)`.
- Do not accumulate `tau` / `log_p` from subset re-normalized probabilities; this breaks comparability with thresholds and historical runs.

## Field Ownership

Core principle: **Creator owns structure, Executor owns runtime state, Judger owns scores, Searcher owns selection/pruning.**

In the table below, "write" means field mutation is allowed; "read-only" means readable but should not be mutated.

| Field / Capability | L3Expander | L2Expander / L1Expander | Executor | Judger | Searcher | Cache |
|---|---:|---:|---:|---:|---:|---:|
| Create `TreeNode` (`add_child`) | ✗ | **✓** | ✓ (if needed) | ✗ | ✗ | ✗ |
| `children` (topology) | ✗ | **✓** (via `add_child` only) | ✓ (via `add_child` only) | ✗ | ✗ | ✗ |
| `token_ids` / `text` (incremental content) | ✗ | **✓** (at creation) | ✓ (creation only) | ✗ | ✗ | ✗ |
| `log_prob` / `cum_log_prob` (generation metrics) | ✗ | **✓** (at creation) | ✓ (creation-time or supplement) | ✗ | ✗ | ✗ |
| `metadata` | ✓ (result metadata allowed) | ✓ (structure/generation keys) | ✓ (runtime keys) | ✓ (judging keys) | ✓ (search keys) | ✓ (cache internals) |
| `source` | ✗ | **✓** (set once by creator) | ✓ (fill only if missing) | ✗ | ✗ | ✗ |
| `status` (lifecycle) | ✗ | ✗ (forbidden) | **✓** (single owner) | ✗ | ✓ (`QUEUED` / `CUT` only; see below) | ✗ |
| `score` / `scores` (judging score) | ✗ | ✗ | ✓ (aggregate/append/persist) | **✓** (compute/write) | ✗ (read-only) | ✗ (store value only; do not mutate node) |
| `conversation_state` | ✗ | **✓** (pending append, commit) | ✓ (initial setup) | ✗ (read-only) | ✗ (read-only) | ✗ |
| `delta` (ConversationDelta) | ✗ | **✓** (set at creation/commit) | ✓ (root Noop only) | ✗ | ✗ | ✗ |

### Additional Metadata Constraints (Strongly Recommended)

To avoid key collisions across modules:

- **Metadata keys must use prefixes**, for example:
  - `sampler.l3/topk`
  - `executor/buffer_task_id`
  - `judger/refusal_state`
  - `searcher/ucb`
- Do **not** insert shadow core fields into `metadata` (for example `metadata["score"]`).
  Use canonical fields such as `TreeNode.score` / `TreeNode.scores`.

## Lifecycle Conventions (Single Entry for `status`)

Convention: `TreeNode.status` has **Executor as the single primary writer**, keeping state transitions as a single source of truth.

Recommended transitions (subset is allowed depending on implementation):

- `CREATED`: node exists but has not entered runtime pipeline.
- `EXPANDING`: Executor is requesting/generating children for this node.
- `EVALUATING`: children/candidates are waiting in buffer or being judged.
- `EVALUATED`: enough score information is available for Searcher decisions.
- `QUEUED`: marked as candidate by Searcher (optional).
- `CUT`: pruned by Searcher.
- `COMPLETED` / `JAILBREAKED`: terminal states.

Exception: if Searcher writes `QUEUED` / `CUT`, it must follow:

- Searcher writes **search-policy-only states** (`QUEUED`, `CUT`, etc.).
- Searcher does **not** write pipeline runtime states (`EXPANDING`, `EVALUATING`, `EVALUATED`).

## ConversationState Lifecycle

`ConversationState` follows a strict pending → commit lifecycle.
See `DESIGN_conversation_state.md` for the full specification.

- **Pending**: `L2Expander` appends chunks to `conversation_state.pending` via `append_pending_assistant()` during tree expansion.
- **Commit (single-turn EOS)**: `L1Expander.expand_after_eos()` calls `commit_assistant_turn()` to promote pending into a committed assistant message.
- **Commit (tool interaction)**: `L1Expander.expand_after_eos()` calls `commit_tool_interaction()` on TOOL nodes, committing both the assistant tool-call message and tool result.
- **Copy-on-write**: child nodes clone the parent's `ConversationState` in `__post_init__` so siblings never share mutable references.
- **Delta tagging**: every node sets `delta` (`ConversationDelta`) to describe what it contributed relative to its parent.

## Expander Contract (Do Not Compete with Executor for Runtime State)

- **L3Expander**: pure algorithmic black box; returns candidate tuples (`ids/text/log_p`) only; no `TreeNode` mutation.
- **L2Expander**: materializes L3 candidates into `TreeNode` children (structure + content + generation metrics); do not write runtime-state fields.
- **L1Expander**: creates user-input nodes (structure + content + `source=USER`); do not write `status` or scores.

## Executor Contract (Single Entry for Runtime State)

Executor converts "structural nodes" into "runtime nodes":

- set/advance `status`
- enqueue tasks into buffers and trigger Sampler/Judger
- write back Judger outputs into `score/scores`
- handle cache hits and write-back (cache itself does not mutate nodes)

## Quick Checklist (Before You Code)

- Did I modify `status` or `score/scores` inside an Expander? -> **Not allowed**
- Did I use `node.token_ids` or `get_path_text()` as full prompt context? -> Use `build_node_model_input()` or `conversation_state`
- Did I append with `node.children.append(...)` directly? -> Use `node.add_child(...)`
- Did I write potentially colliding metadata keys? -> Add a module prefix
