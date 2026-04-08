# sampler/l3_expander.py
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from engines.base_engine import TargetModelEngineBase


@dataclass
class BeamState:
    ids: List[int]
    cum_log_p: float
    stopped: bool = False
    stop_reason: Optional[str] = None
    is_cut: bool = False
    cut_reason: Optional[str] = None
    cut_threshold: Optional[float] = None
    cut_th_idx: Optional[int] = None
    dynamic: Optional[Dict[str, float]] = None
    dynamic_hit: bool = False
    decode_parent_row: Optional[int] = None
    decode_token_id: Optional[int] = None


class L3Expander:
    """
    Inner-chunk Pathfinder (Black Box).
    Responsibility: given an input path, produce a set of best candidate chunks (token sequences).
    It is unaware of the tree structure and only implements the algorithmic logic.

    L3 does not create/modify TreeNode; it only returns candidate chunks (ids/text/log_p). See CONVENTIONS.md.
    """

    def __init__(self, engine: TargetModelEngineBase, config, threshold: Optional[List[float]] = None):
        self.engine = engine
        self.config = config
        self.tokenizer = self.engine.get_tokenizer()
        self.stop_token_ids = self._resolve_stop_token_ids()
        self.chunk_size = int(config.chunk_size)
        self.chunk_width = int(config.chunk_width)
        self.temperature = float(config.temperature)
        self.top_p = float(config.top_p)
        self.top_k = int(config.top_k)
        self.enable_topk_optimization = bool(config.enable_topk_optimization)
        self.topk_prefilter_size = int(config.topk_prefilter_size)
        # Reserved for future batched L3 expansion controls.
        self.expander_batch_size = int(config.expander_batch_size)
        self.eps = 1e-12

        # Unified mode switches: off | stop | limit
        self.dynamic_mode = str(config.dynamic_stop_mode).lower()
        self.chunk_width_mode = str(config.chunk_width_mode).lower()
        self.chunk_len_mode = str(config.chunk_len_mode).lower()

        # Dynamic-stop thresholds (logits/probability based).
        self.dynamic_max_prob_threshold = config.dynamic_max_prob_threshold
        self.dynamic_entropy_threshold = config.dynamic_entropy_threshold
        self.dynamic_margin_threshold = config.dynamic_margin_threshold

        # tau baseline threshold list; when None/empty, pruning disabled
        self.threshold = threshold or None
        self._threshold_tensor: Optional[torch.Tensor] = None

    def _resolve_stop_token_ids(self) -> set[int]:
        stop_ids = {int(self.tokenizer.eos_token_id)}
        model = getattr(self.engine, "model", None)
        if model is None:
            return stop_ids

        generation_eos = model.generation_config.eos_token_id
        if isinstance(generation_eos, int):
            stop_ids.add(generation_eos)
        else:
            stop_ids.update(int(token_id) for token_id in generation_eos)
        return stop_ids

    def find_candidate_chunks(
        self,
        path_ids: List[int],
        *,
        base_cum_log_prob: float = 0.0,
        base_generated_len: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Incremental beam decode:
        - One prefill on shared prefix.
        - Then per-step single-token batch decode with branched KV cache.
        """
        active_logits, active_kv = self._prefill(path_ids)
        frontier: List[BeamState] = [self._make_state_item(ids=[], cum_log_p=0.0)]
        cut_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        cut_seq = 0

        for _ in range(self.chunk_size):
            stopped_states, active_states = self._split_frontier(frontier)
            if not active_states:
                break

            candidates, expanded_overflow, cut_seq = self._expand_candidates(
                active_states=active_states,
                active_logits=active_logits,
                base_cum_log_prob=base_cum_log_prob,
                base_generated_len=base_generated_len,
                carry_count=len(stopped_states),
                cut_heap=cut_heap,
                cut_seq=cut_seq,
            )

            expanded_pool = stopped_states + candidates
            if not expanded_pool:
                break

            frontier, _ = self._apply_chunk_width(expanded_pool)
            if expanded_overflow and self._is_stop(self.chunk_width_mode):
                break

            active_logits, active_kv, has_active = self._decode_next(frontier, active_kv)
            if not has_active:
                break

        return self._finalize(frontier, cut_heap, base_cum_log_prob)

    def _prefill(self, path_ids: List[int]) -> Tuple[torch.Tensor, Any]:
        input_ids = torch.tensor(path_ids, dtype=torch.long, device=self.engine.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return self.engine.forward_step(input_ids, kv_cache=None, attention_mask=attention_mask)

    @staticmethod
    def _split_frontier(frontier: List[BeamState]) -> Tuple[List[BeamState], List[BeamState]]:
        stopped = [s for s in frontier if s.stopped]
        active = [s for s in frontier if not s.stopped]
        return stopped, active

    def _expand_candidates(
        self,
        *,
        active_states: List[BeamState],
        active_logits: torch.Tensor,
        base_cum_log_prob: float,
        base_generated_len: int,
        carry_count: int,
        cut_heap: List[Tuple[float, int, Dict[str, Any]]],
        cut_seq: int,
    ) -> Tuple[List[BeamState], bool, int]:
        top_ids, top_log_p, valid_mask = self._get_top_candidates_batch(active_logits, self.chunk_width)
        parent_cum = torch.as_tensor(
            [s.cum_log_p for s in active_states],
            dtype=top_log_p.dtype,
            device=self.engine.device,
        ).unsqueeze(-1)
        cand_cum = parent_cum + top_log_p
        keep_mask = valid_mask.clone()

        keep_mask, cut_seq = self._apply_tau_prune(
            active_states=active_states,
            top_ids=top_ids,
            top_log_p=top_log_p,
            parent_cum=parent_cum,
            cand_cum=cand_cum,
            keep_mask=keep_mask,
            base_cum_log_prob=base_cum_log_prob,
            base_generated_len=base_generated_len,
            cut_heap=cut_heap,
            cut_seq=cut_seq,
        )

        kept_idx = keep_mask.nonzero(as_tuple=False)
        raw_kept_count = int(kept_idx.size(0))
        expanded_overflow = (carry_count + raw_kept_count) > self.chunk_width

        if kept_idx.numel() == 0:
            return [], expanded_overflow, cut_seq

        kept_scores = cand_cum[kept_idx[:, 0], kept_idx[:, 1]]
        keep_n = min(self.chunk_width, kept_idx.size(0))
        if kept_idx.size(0) > keep_n:
            top_sel = torch.topk(kept_scores, keep_n, dim=0).indices
            kept_idx = kept_idx.index_select(0, top_sel)

        states = self._build_states_from_kept(
            active_states=active_states,
            active_logits=active_logits,
            top_ids=top_ids,
            top_log_p=top_log_p,
            cand_cum=cand_cum,
            kept_idx=kept_idx,
        )
        return states, expanded_overflow, cut_seq

    def _apply_tau_prune(
        self,
        *,
        active_states: List[BeamState],
        top_ids: torch.Tensor,
        top_log_p: torch.Tensor,
        parent_cum: torch.Tensor,
        cand_cum: torch.Tensor,
        keep_mask: torch.Tensor,
        base_cum_log_prob: float,
        base_generated_len: int,
        cut_heap: List[Tuple[float, int, Dict[str, Any]]],
        cut_seq: int,
    ) -> Tuple[torch.Tensor, int]:
        if self.threshold is None:
            return keep_mask, cut_seq

        parent_next_len = torch.as_tensor(
            [int(base_generated_len) + len(s.ids) + 1 for s in active_states],
            dtype=torch.long,
            device=self.engine.device,
        ).unsqueeze(-1)
        th_idx = parent_next_len - 1
        in_range = (th_idx >= 0) & (th_idx < len(self.threshold))
        if not in_range.any():
            return keep_mask, cut_seq

        threshold_tensor = self._get_threshold_tensor(dtype=top_log_p.dtype, device=self.engine.device)
        th_vals = threshold_tensor[th_idx.clamp(min=0, max=len(self.threshold) - 1)]
        global_cum = float(base_cum_log_prob) + cand_cum
        tau_pruned = keep_mask & in_range & (global_cum < th_vals)

        if tau_pruned.any():
            pruned_idx = tau_pruned.nonzero(as_tuple=False)
            for p in pruned_idx:
                parent_i = int(p[0].item())
                cand_j = int(p[1].item())
                tid = int(top_ids[parent_i, cand_j].item())
                log_p = float(top_log_p[parent_i, cand_j].item())
                cum_log_p = float(parent_cum[parent_i, 0].item())
                global_log_p = float(global_cum[parent_i, cand_j].item())
                cut_item = self._make_cut_item(
                    current_ids=active_states[parent_i].ids,
                    cum_log_p=cum_log_p,
                    tid=tid,
                    log_p=log_p,
                    global_cum=global_log_p,
                    th_idx=int(th_idx[parent_i, 0].item()),
                )
                if len(cut_heap) < self.chunk_width:
                    heapq.heappush(cut_heap, (global_log_p, cut_seq, cut_item))
                else:
                    heapq.heappushpop(cut_heap, (global_log_p, cut_seq, cut_item))
                cut_seq += 1

        return keep_mask & ~tau_pruned, cut_seq

    def _build_states_from_kept(
        self,
        *,
        active_states: List[BeamState],
        active_logits: torch.Tensor,
        top_ids: torch.Tensor,
        top_log_p: torch.Tensor,
        cand_cum: torch.Tensor,
        kept_idx: torch.Tensor,
    ) -> List[BeamState]:
        states: List[BeamState] = []
        dynamic_cache: Dict[int, Tuple[bool, Dict[str, float]]] = {}

        for p in kept_idx:
            parent_i = int(p[0].item())
            cand_j = int(p[1].item())
            tid = int(top_ids[parent_i, cand_j].item())
            cum_log_p = float(cand_cum[parent_i, cand_j].item())
            next_ids = active_states[parent_i].ids + [tid]

            dyn = dynamic_cache.get(parent_i)
            if dyn is None:
                dyn = self._dynamic_trigger(active_logits[parent_i])
                dynamic_cache[parent_i] = dyn

            stopped, stop_reason, meta = self._check_stop_conditions(next_ids, tid, dyn)
            state = self._make_state_item(
                ids=next_ids,
                cum_log_p=cum_log_p,
                stopped=stopped,
                stop_reason=stop_reason,
                meta=meta,
            )
            if not stopped:
                state.decode_parent_row = int(parent_i)
                state.decode_token_id = int(tid)
            states.append(state)

        return states

    def _decode_next(self, frontier: List[BeamState], kv_cache: Any) -> Tuple[torch.Tensor, Any, bool]:
        next_active = [s for s in frontier if not s.stopped]
        if not next_active:
            return torch.empty(0, device=self.engine.device), kv_cache, False

        parent_rows = torch.as_tensor(
            [int(s.decode_parent_row) for s in next_active],
            dtype=torch.long,
            device=self.engine.device,
        )
        next_tokens = torch.as_tensor(
            [int(s.decode_token_id) for s in next_active],
            dtype=torch.long,
            device=self.engine.device,
        ).unsqueeze(-1)

        parent_kv = self._index_kv_cache(kv_cache, parent_rows)
        logits, next_kv = self.engine.forward_step(next_tokens, kv_cache=parent_kv)
        return logits, next_kv, True

    def _finalize(
        self,
        frontier: List[BeamState],
        cut_heap: List[Tuple[float, int, Dict[str, Any]]],
        base_cum_log_prob: float,
    ) -> List[Dict[str, Any]]:
        valid = [self._normalize_item(s, base_cum_log_prob) for s in frontier]
        cut_chunks = [
            self._normalize_item(item, base_cum_log_prob)
            for _, _, item in sorted(cut_heap, key=lambda x: x[0], reverse=True)
        ]
        results = valid + cut_chunks
        results.sort(key=lambda x: float(x["global_log_p"]), reverse=True)
        return results

    def _get_threshold_tensor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.threshold is None:
            raise ValueError("threshold is None")
        if (
            self._threshold_tensor is None
            or self._threshold_tensor.dtype != dtype
            or self._threshold_tensor.device != device
        ):
            self._threshold_tensor = torch.as_tensor(self.threshold, dtype=dtype, device=device)
        return self._threshold_tensor

    def _make_cut_item(
        self,
        *,
        current_ids: List[int],
        cum_log_p: float,
        tid: int,
        log_p: float,
        global_cum: float,
        th_idx: int,
    ) -> Dict[str, Any]:
        return {
            "ids": current_ids + [tid],
            "cum_log_p": float(cum_log_p) + float(log_p),
            "log_p": float(cum_log_p) + float(log_p),
            "global_log_p": float(global_cum),
            "is_cut": True,
            "stopped": True,
            "stop_reason": "tau_prune",
            "cut_reason": "tau_prune",
            "cut_threshold": float(self.threshold[th_idx]),
            "cut_th_idx": int(th_idx),
            "dynamic": None,
            "dynamic_hit": False,
        }

    def _make_state_item(
        self,
        *,
        ids: List[int],
        cum_log_p: float,
        stopped: bool = False,
        stop_reason: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BeamState:
        state = BeamState(ids=ids, cum_log_p=float(cum_log_p), stopped=bool(stopped), stop_reason=stop_reason)
        if meta:
            state.dynamic = meta.get("dynamic")
            state.dynamic_hit = bool(meta.get("dynamic_hit", False))
        return state

    def _normalize_item(self, state: Any, base_cum_log_prob: float) -> Dict[str, Any]:
        if isinstance(state, BeamState):
            return {
                "ids": state.ids,
                "cum_log_p": float(state.cum_log_p),
                "log_p": float(state.cum_log_p),
                "global_log_p": float(base_cum_log_prob) + float(state.cum_log_p),
                "stopped": bool(state.stopped),
                "stop_reason": state.stop_reason,
                "is_cut": bool(state.is_cut),
                "cut_reason": state.cut_reason,
                "cut_threshold": state.cut_threshold,
                "cut_th_idx": state.cut_th_idx,
                "dynamic": state.dynamic,
                "dynamic_hit": bool(state.dynamic_hit),
            }

        # Cut-item path (dict)
        cum_log_p = float(state["cum_log_p"])
        global_log_p = float(state.get("global_log_p", float(base_cum_log_prob) + cum_log_p))
        log_p = float(state.get("log_p", cum_log_p))
        return {
            "ids": state["ids"],
            "cum_log_p": cum_log_p,
            "log_p": log_p,
            "global_log_p": global_log_p,
            "stopped": bool(state.get("stopped", False)),
            "stop_reason": state.get("stop_reason"),
            "is_cut": bool(state.get("is_cut", False)),
            "cut_reason": state.get("cut_reason"),
            "cut_threshold": state.get("cut_threshold"),
            "cut_th_idx": state.get("cut_th_idx"),
            "dynamic": state.get("dynamic"),
            "dynamic_hit": bool(state.get("dynamic_hit", False)),
        }

    def _index_kv_cache(self, kv_cache: Any, index: torch.Tensor) -> Any:
        # HF Cache objects are mutable; clone before selecting rows for branching.
        if hasattr(kv_cache, "to_legacy_cache") and hasattr(kv_cache, "from_legacy_cache"):
            branched = kv_cache.__class__.from_legacy_cache(kv_cache.to_legacy_cache())
            branched.batch_select_indices(index)
            return branched
        if isinstance(kv_cache, tuple):
            return tuple(tuple(t.index_select(0, index) for t in layer_past) for layer_past in kv_cache)
        if isinstance(kv_cache, list):
            if kv_cache and isinstance(kv_cache[0], list):
                idx = index.tolist()
                return [kv_cache[i].copy() for i in idx]
            if kv_cache and isinstance(kv_cache[0], tuple):
                return [tuple(t.index_select(0, index) for t in layer_past) for layer_past in kv_cache]
            idx = index.tolist()
            return [kv_cache[i] for i in idx]
        raise TypeError(f"Unsupported kv_cache type for branching: {type(kv_cache)}")

    def _get_top_candidates_batch(self, logits_batch: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Candidate selection under sampling strategy:
        temperature -> top-p/top-k filtering -> select top-k-by-prob.
        """
        logits_temp = logits_batch / max(self.temperature, 1e-6)
        z = torch.logsumexp(logits_temp, dim=-1, keepdim=True)

        vocab_size = logits_temp.size(-1)
        if self.enable_topk_optimization:
            prefilter_k = max(int(k), self.topk_prefilter_size, int(self.top_k))
            prefilter_k = min(prefilter_k, vocab_size)
        else:
            prefilter_k = vocab_size

        logits_sorted, idx_sorted = torch.topk(logits_temp, prefilter_k, dim=-1)
        probs_sorted = torch.exp(logits_sorted - z)

        if self.top_p < 1.0:
            cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
            # Nucleus filtering:
            # keep tokens up to and including the first token where cumulative > top_p,
            # then include all ties on the boundary probability.
            to_remove = cumulative_probs > self.top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            keep_mask = ~to_remove

            # Include all equal-probability tokens at the cutoff boundary.
            keep_count = keep_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            boundary_idx = keep_count - 1
            boundary_prob = probs_sorted.gather(-1, boundary_idx)
            tie_mask = probs_sorted == boundary_prob
            keep_mask = keep_mask | tie_mask

            probs_sorted = probs_sorted * keep_mask.to(probs_sorted.dtype)

        if self.top_k > 0:
            final_k = min(self.top_k, probs_sorted.size(-1))
            probs_subset, top_idx_in_sorted = torch.topk(probs_sorted, final_k, dim=-1)
            idx_subset = idx_sorted.gather(-1, top_idx_in_sorted)
        else:
            probs_subset = probs_sorted
            idx_subset = idx_sorted

        take_n = min(max(1, int(k)), probs_subset.size(-1))
        masked_probs = probs_subset.masked_fill(probs_subset <= 0, float("-inf"))
        top_probs, top_pos = torch.topk(masked_probs, take_n, dim=-1)
        top_ids = idx_subset.gather(-1, top_pos)
        top_log_p = logits_temp.gather(-1, top_ids) - z
        valid_mask = torch.isfinite(top_probs)
        return top_ids, top_log_p, valid_mask

    def _apply_chunk_width(self, pool: List[BeamState]) -> Tuple[List[BeamState], bool]:
        # Width pruning is local-to-chunk: rank by local cumulative log-prob only.
        topn = heapq.nlargest(
            self.chunk_width,
            pool,
            key=lambda x: float(x.cum_log_p),
        )
        width_hit = len(pool) > self.chunk_width
        return topn, width_hit

    # Stop rules are centralized here:
    # 1) eos
    # 2) chunk length mode (stop/limit)
    # 3) dynamic mode (off/stop/limit)
    def _check_stop_conditions(
        self,
        next_ids: List[int],
        tid: int,
        dynamic: Tuple[bool, Dict[str, float]],
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        if self._is_eos(tid):
            return True, "eos", {}

        dyn_hit, dyn_info = dynamic
        meta = {"dynamic": dyn_info, "dynamic_hit": bool(dyn_hit)}

        # Note: outer loop already enforces a hard chunk_size cap.
        # chunk_len_mode here mainly controls boundary stop labeling.
        if len(next_ids) >= self.chunk_size:
            if self._is_stop(self.chunk_len_mode):
                return True, "chunk_len_stop", meta
            if self._is_limit(self.chunk_len_mode):
                return True, "chunk_len_limit", meta

        if dyn_hit:
            if self._is_stop(self.dynamic_mode):
                return True, "dynamic_stop", {"dynamic": dyn_info, "dynamic_hit": True}
            if self._is_limit(self.dynamic_mode):
                return True, "dynamic_limit", {"dynamic": dyn_info, "dynamic_hit": True}
        return False, None, meta

    def _dynamic_trigger(self, logits: torch.Tensor) -> Tuple[bool, Dict[str, float]]:
        assert logits.dim() == 1, f"_dynamic_trigger expects 1D logits, got shape={tuple(logits.shape)}"
        probs = torch.softmax(logits, dim=-1)
        k = min(2, probs.size(-1))
        top_probs, _ = torch.topk(probs, k=k, dim=-1)
        max_prob = float(top_probs[0].item())
        second_prob = float(top_probs[1].item()) if top_probs.numel() > 1 else 0.0
        margin = max_prob - second_prob
        entropy = float((-(probs * torch.log(probs.clamp_min(self.eps))).sum()).item())

        hit = False
        if self.dynamic_max_prob_threshold is not None:
            hit = hit or (max_prob >= float(self.dynamic_max_prob_threshold))
        if self.dynamic_margin_threshold is not None:
            hit = hit or (margin >= float(self.dynamic_margin_threshold))
        if self.dynamic_entropy_threshold is not None:
            hit = hit or (entropy <= float(self.dynamic_entropy_threshold))

        return hit, {"max_prob": max_prob, "margin": margin, "entropy": entropy}

    def _is_eos(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids

    @staticmethod
    def _is_stop(mode: str) -> bool:
        return mode == "stop"

    @staticmethod
    def _is_limit(mode: str) -> bool:
        return mode == "limit"
