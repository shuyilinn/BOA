# BOA Conventions (Field Ownership and Lifecycle)

This document defines implementation conventions for `TreeNode` semantics and module boundaries.
When adding or refactoring modules, align with these conventions first and keep this file updated if behavior changes.

## TreeNode Field Semantics

`TreeNode` contains two data categories that are easy to confuse:

- **Incremental fields**
  - `token_ids` / `text`: only the newly added token/text segment from parent to current node.
  - `log_prob`: only the log-probability of this incremental generation step (when applicable).
- **Full path fields**
  - Use `TreeNode.get_path_token_ids()` / `TreeNode.get_path_text()` to obtain the full sequence from root to current node.

Convention: **Anything sent to model input, judger input, or cache keys must use `get_path_*()`; never treat `token_ids`/`text` as full context.**

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
- Did I use `node.token_ids` as full prompt context? -> Use `get_path_token_ids()`
- Did I append with `node.children.append(...)` directly? -> Use `node.add_child(...)`
- Did I write potentially colliding metadata keys? -> Add a module prefix
