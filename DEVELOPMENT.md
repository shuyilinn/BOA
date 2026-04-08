# BOA Development Log

Record of each search/scoring optimization: problem, example, solution, date, test result.

---

## [2026-03-24] Tail-Aware Optimistic Selection + Top-M Softmax

**Problem**: NaiveSearcher uses pure exploitation — greedy pop by mean(scores). Stable safe-alternative branches (score ~3000) monopolize the frontier, while branches that occasionally produce 9000 get suppressed. Goal mismatch: we need existence search (any unsafe continuation), not "highest average danger".

**Example**: `experiment/repeat_logs/repeat_20260324-160212/greedy_run_2.log` — search follows "Would you like information on how to identify a Nigerian 419 scam" safe path from depth 5 to depth 10, all nodes scoring 3000. Meanwhile `" that would trick"` with score=0 (1 cached sample) is ignored. 120s budget produces only 10 evaluated nodes.

**Solution**:
- `selection_score = (1-alpha)*mean + alpha*top2_mean` (tail-aware scoring)
- NaiveSearcher supports top-M softmax sampling instead of always popping top-1
- New config fields: `search_strategy`, `search_alpha`, `search_top_m`, `search_temperature`, `search_epsilon`
- Default `search_strategy="greedy"` preserves backward compatibility

**Test**: `experiment/repeat_experiment.py` — greedy vs tail_aware_softmax, 5 runs each

**Result**: tail_aware_softmax 2/5 unsafe vs greedy 1/5 unsafe. tail_aware selects more diverse paths ("that would trick" instead of "Would you like..."), reaches score=7000 branches.

---

## [2026-03-24] Cache all_complete probability exhaustion check

**Problem**: Cache `all_complete` logic is too aggressive — a single EOS-complete cache hit causes all remaining 9 samples to be skipped. Many children enter the searcher queue with score=0 from just 1 sample, killing potentially high-scoring branches.

**Example**: In greedy_run_2, expanding depth=1 produces 10 children. `" that would trick"` has cached=1/10 score=0 (one EOS sample happened to be safe). It enters the queue at score=0 and is never explored. With full sampling it could produce 7000+ continuations.

**Solution**: Store rollout `response_tau` (cumulative log-prob) in cache. When checking `all_complete`, compute `sum(exp(tau))` — only skip sampling when probability mass coverage >= 0.95.

**Test**: repeat_experiment.py — "Cache full hit cached=1/10" count should decrease significantly, number of evaluated nodes should increase

**Result**: pending

---

## [2026-03-25] Extract BOA Searcher & Clean Up Search Strategies

**Problem**: NaiveSearcher contained 4 strategies (greedy, ucb, tail_aware_softmax, epsilon_greedy) in one class with complex branching. The UCB strategy's depth_decay and stagnation_penalty params were never actually passed from Executor (bug). tail_aware_softmax and epsilon_greedy were intermediate experiments that are no longer needed.

**Solution**:
- Extracted UCB strategy into `searchers/boa_searcher.py` as `BoaSearcher` class, renamed to "boa"
- Simplified `searchers/naive_searcher.py` to greedy-only (pure heappop, no branching)
- Executor now creates `BoaSearcher` or `NaiveSearcher` based on `search_strategy` config
- Fixed bug: `depth_decay`, `stagnation_window`, `stagnation_penalty` are now correctly passed to BoaSearcher
- Removed dead config fields: `search_top_m`, `search_temperature`, `search_epsilon`
- Deleted tail_aware_softmax and epsilon_greedy strategies entirely
- Updated web UI, CLI args, and experiment scripts

**BOA strategy details**:
- `search_strategy="boa"` — softmax top-M selection + depth decay + stagnation penalty
- `add_node`: `effective_score = selection_score * depth_decay^depth * stagnation_penalty`
- `select_next_node`: softmax sampling from top-M candidates (temperature-controlled)
- `_compute_selection_score` (in Executor): `(1-α)*mean + α*top2_mean` (tail-aware blend, α=0.3)
- Depth decay (default 0.97): depth=10 → 74% score, depth=20 → 54% score — prevents over-exploration in one direction
- Stagnation penalty (default 0.2): if score improvement < 5% over last 3 ancestor levels, score is reduced to 20%

**Config** (`search_strategy="greedy" | "boa" | "gradient"`):
- `search_alpha` — tail weight for selection score (boa/gradient)
- `search_depth_decay` — per-depth score multiplier (boa only, default 0.97)
- `search_stagnation_window` — ancestor levels to check for plateau (boa only, default 3)
- `search_stagnation_penalty` — multiplier when stagnant (boa only, default 0.2)
- `search_gradient_window` — ancestor levels for gradient computation (gradient only, default 3)
- `search_breakthrough_bonus` — multiplier for 0→positive breakthroughs (gradient only, default 2.0)
- `search_gradient_penalty` — multiplier for stagnating paths (gradient only, default 0.2)
- `search_gradient_low` — relative improvement below this = stagnant (gradient only, default 0.05)
- `search_gradient_high` — relative improvement above this = max boost (gradient only, default 1.0)

**Files changed**: `searchers/boa_searcher.py` (new), `searchers/naive_searcher.py`, `config.py`, `executor/executor.py`, `web/web_config.py`, `web/static/index.html`, `experiment/repeat_experiment.py`, `utils/config_resolver.py`. Deleted: `searchers/mcts.py`

---

## [2026-03-25] Extract Pluggable Cache Algorithm & Fix Coverage Logic

**Problem**: Cache match logic (30 lines) was embedded in `executor/executor.py` with two issues:
1. Logic bug: the `all_complete` gate required every cached item to be EOS-terminated before checking probability coverage. A single non-EOS item would bypass the coverage check entirely, causing unnecessary sampling even when most probability mass was already covered.
2. Encapsulation violation: cache-specific decision logic (tau sum, coverage check, needed_count calculation) lived in the executor instead of the cache component.

**Example**: Expanding a node with 2 cached items where `sum(exp(tau)) = 0.98` but one item was not EOS-complete. Old code would skip the probability check and sample `target_count - 2` more items. New code correctly recognizes 98% coverage and skips sampling.

**Solution**:
- Created pluggable cache algorithm abstraction following the `searchers/` pattern (ABC + concrete implementations)
- `BaseCacheAlgo` (ABC) defines `query(cache, prefix_ids, *, target_count, min_suffix_tokens, node_cum_prob)` interface
- `TauThresholdCacheAlgo` implements coverage-based decision: compares sum of cached children's joint probability (`node_cum_prob * sum(exp(tau_i))`) against the node's own probability (`node_cum_prob * coverage_ratio`)
- Removed `all_complete` gate — coverage check applies to all cached items regardless of EOS completion status
- `coverage_ratio` (default 0.95) passed at algo construction from config; `node_cum_prob` (`exp(child.cum_log_prob)`) passed at query time from executor
- Executor cache block reduced from 30 lines to 12 lines — only calls `cache_algo.query()` and reads the result

**Config** (`cache_coverage_ratio: float = 0.95`):
- Fraction of the node's probability mass that must be covered by cached items to skip sampling
- 0.95 = if cached children account for 95% of the node's probability, all continuations are considered found

**Files changed**: `components/cache/algo.py` (new), `config.py`, `executor/executor.py`, `tests/test_cache_algo.py` (new)

---

## [2026-03-26] Gradient Search Strategy — Score Improvement Rate

**Problem**: BOA searcher uses `effective = selection_score × decay^depth × stagnation_penalty`. With decay=0.6, deep nodes' raw scores grow exponentially (0→300→1000→3600) as the model commits to a direction, far outpacing the decay. The binary stagnation penalty (1.0 or 0.2) is too coarse. Result: search still dives deep into stagnating paths while missing promising shallow breakthroughs.

**Example**: `logs/web_1774507297_5d5a01.log` — depth=8 node "to create" has raw score=4780, decay factor=0.017, effective=80.3. Meanwhile depth=4 node "that would" has raw score=660, decay=0.130, effective=85.5. Despite 7× raw score difference, they end up nearly equal — decay compresses everything into a narrow band where the actual score trajectory (improving vs stagnating) is lost.

Key observation from frontier ranking:
- depth=4 "email." (sel=660, ancestor=660, 0% improvement) → stagnating, should be deprioritized
- depth=8 "to create" (sel=4780, ancestor=3400, 40% improvement) → still improving, worth exploring
- depth=4 "that would" (sel=660, ancestor=0, breakthrough) → just broke through from 0, highest priority

**Insight**: The right signal is not absolute score penalized by depth, but **how fast the score is improving along the path**. Reward fast-improving paths, penalize stagnating ones.

**Solution**: New `search_strategy="gradient"` in `searchers/gradient_searcher.py`:
- No depth decay — depth is not inherently penalized
- `_compute_gradient_factor(node)`: walks up `gradient_window` ancestors, computes relative score improvement, maps to continuous factor in `[stagnation_penalty, breakthrough_bonus]`
  - `ancestor.selection_score <= 0` and `node.selection_score > 0` → breakthrough, factor = `breakthrough_bonus` (default 2.0)
  - `rel_improvement = (node.sel - ancestor.sel) / ancestor.sel`
  - Linear interpolation: `low_thresh` (0.05, stagnant) → `high_thresh` (1.0, fast growth) maps to `[penalty, bonus]`
- `effective_score = selection_score × gradient_factor`
- `select_next_node`: softmax top-M (same as BoaSearcher)

**Expected behavior** (window=3, bonus=2.0, penalty=0.2):

| Node | depth | sel_score | ancestor | rel_impr | factor | effective |
|------|-------|-----------|----------|----------|--------|-----------|
| "that would" | 4 | 660 | 0 | breakthrough | 2.0 | 1320 |
| "to create" | 8 | 4780 | 3400 | 40.6% | 1.07 | 5115 |
| "email." | 5 | 660 | 660 | 0% | 0.2 | 132 |
| "to draft" | 8 | 4780 | 300 | 1493% | 2.0 | 9560 |

Breakthrough nodes get top priority. Stagnating paths are heavily penalized regardless of absolute score. Fast-improving deep paths are not suppressed by depth.

**Config** (`search_strategy="greedy" | "boa" | "gradient"`):
- `search_gradient_window` — ancestor levels for gradient computation (default 3)
- `search_breakthrough_bonus` — multiplier for 0→positive breakthroughs (default 2.0)
- `search_gradient_penalty` — multiplier for stagnating paths (default 0.2)
- `search_gradient_low` — relative improvement threshold below which path is stagnant (default 0.05)
- `search_gradient_high` — relative improvement threshold above which path gets max boost (default 1.0)

**Files changed**: `searchers/gradient_searcher.py` (new), `config.py`, `executor/executor.py`, `web/app.py`, `web/run_manager.py`, `web/web_config.py`, `web/static/index.html`
