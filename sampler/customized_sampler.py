import torch
from typing import Optional, Tuple, Dict
from profiler import profile, profile_block


# Now only support hf engine
# Input: logits, temperature, top_p, top_k
# Output: next_tokens
# shuyi: this file has not been viewed
"This file is for a single step sampling"
class CustomizedSampler:
    """
    BOA custom sampling policy for one generation step.
    - Token selection can use temperature + top-p/top-k + smoothing.
    - Returned log-prob is always from full distribution (full softmax over vocabulary).
    """
    def __init__(self, config):
        self.config = config
        self.eps = 1e-12

    @profile("sampler.prepare_logits")
    def prepare_logits(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling."""
        if temperature > 0:
            return logits / temperature
        return logits

    @profile("sampler.apply_smoothing")
    def apply_smoothing(self,
                        probs_subset: torch.Tensor,
                        smoothing_factor) -> torch.Tensor:
        """
        Uniform smoothing on candidate subset: (1 - alpha) * P_orig + alpha * P_uniform.
        smoothing_factor can be a float scalar or a [B, 1] tensor for per-row smoothing.
        """
        if isinstance(smoothing_factor, (int, float)) and smoothing_factor <= 0:
            return probs_subset

        # Spread smoothing only over nonzero slots
        nonzero_mask = (probs_subset > 0)
        nonzero_cnt = nonzero_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        uniform_dist = nonzero_mask.float() / nonzero_cnt

        smoothed_probs = (1 - smoothing_factor) * probs_subset + smoothing_factor * uniform_dist
        return smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    @profile("sampler.top_p_k_subset")
    def get_top_p_k_subset(self,
                           logits: torch.Tensor, 
                           top_p: float, 
                           top_k: int, 
                           enable_optimization: bool = True,
                           log_denom: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (subset probs, token indices, raw log-probs in full distribution)
        after Top-P and Top-K filtering.
        """
        if log_denom is None:
            log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        vocab_size = logits.size(-1)
        
        # Optimized: top-K first then P to save memory.
        if enable_optimization:
            prefilter_k = int(self.config.topk_prefilter_size)
            K_val = max(top_k, prefilter_k) if top_k > 0 else prefilter_k
            K_val = min(K_val, vocab_size)
        else:
            K_val = vocab_size
        logits_sorted, idx_sorted = torch.topk(logits, K_val, dim=-1)
        raw_log_probs_sorted = logits_sorted - log_denom
        probs_sorted = torch.exp(raw_log_probs_sorted)

        # Top-P filter
        if top_p < 1.0:
            cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
            # Nucleus filtering:
            # keep tokens up to and including the first token where cumulative > top_p,
            # then include all ties on the boundary probability.
            # Shift right without clone: first token always kept, rest kept if prev cumsum <= top_p
            keep_mask = torch.cat([
                cumulative_probs.new_ones(*cumulative_probs.shape[:-1], 1, dtype=torch.bool),
                cumulative_probs[..., :-1] <= top_p,
            ], dim=-1)

            # Include all equal-probability tokens at the cutoff boundary.
            keep_count = keep_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            boundary_idx = keep_count - 1
            boundary_prob = probs_sorted.gather(-1, boundary_idx)
            tie_mask = probs_sorted == boundary_prob
            keep_mask = keep_mask | tie_mask

            probs_sorted = probs_sorted * keep_mask.to(probs_sorted.dtype)

        # Top-K truncation
        if top_k > 0:
            final_k = min(top_k, probs_sorted.size(-1))
            probs_subset, top_idx_in_sorted = torch.topk(probs_sorted, final_k, dim=-1)
            idx_subset = idx_sorted.gather(-1, top_idx_in_sorted)
            raw_log_probs_subset = raw_log_probs_sorted.gather(-1, top_idx_in_sorted)
        else:
            probs_subset = probs_sorted
            idx_subset = idx_sorted
            raw_log_probs_subset = raw_log_probs_sorted

        return probs_subset, idx_subset, raw_log_probs_subset

    @profile("sampler.sample_step")
    def sample_step(
        self,
        logits: torch.Tensor,
        smoothing_factor: float,
        generator: Optional[torch.Generator] = None,
        min_log_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-step sampling.
        Returns:
        - tokens: sampled token ids
        - full_log_probs: log p(token | context) under full vocabulary distribution
        - has_valid_token: whether each row has at least one tau-legal candidate
        """
        # 1) Build statistics distribution from raw logits.
        base_logits = logits
        with profile_block("sampler.logsumexp_base"):
            base_log_denom = torch.logsumexp(base_logits, dim=-1, keepdim=True)

        # 2) Tau-gate feasibility check on raw full-distribution log-prob.
        has_valid: Optional[torch.Tensor]
        min_log_prob_col: Optional[torch.Tensor] = None
        with profile_block("sampler.tau_gate"):
            if min_log_prob is not None:
                min_log_prob_col = min_log_prob.unsqueeze(-1)
                max_logp = base_logits.amax(dim=-1, keepdim=True) - base_log_denom
                has_valid = (max_logp >= min_log_prob_col).squeeze(-1)
            else:
                has_valid = torch.ones(base_logits.size(0), dtype=torch.bool, device=base_logits.device)

        # 3) Selection distribution (temperature + top-p/top-k), constrained by tau-legal vocab.
        scaled_logits = self.prepare_logits(base_logits, self.config.temperature)

        # Only process rows with at least one tau-legal token.
        with profile_block("sampler.nonzero_sync"):
            valid_rows = has_valid.nonzero(as_tuple=True)[0]
        probs_final = None
        idx_subset = None
        if valid_rows.numel() > 0:
            with profile_block("sampler.index_select"):
                base_logits_valid = base_logits.index_select(0, valid_rows)
                scaled_logits_valid = scaled_logits.index_select(0, valid_rows)
                base_log_denom_valid = base_log_denom.index_select(0, valid_rows)
            if min_log_prob_col is not None:
                with profile_block("sampler.tau_mask"):
                    min_log_prob_valid = min_log_prob_col.index_select(0, valid_rows)
                    tau_logit_floor = base_log_denom_valid + min_log_prob_valid
                    tau_valid_vocab_valid = base_logits_valid >= tau_logit_floor
                    neg_inf = torch.finfo(scaled_logits_valid.dtype).min
                    invalid_mask = ~tau_valid_vocab_valid
                    if invalid_mask.any():
                        scaled_logits_valid.masked_fill_(invalid_mask, neg_inf)
            with profile_block("sampler.logsumexp_scaled"):
                scaled_log_denom_valid = torch.logsumexp(scaled_logits_valid, dim=-1, keepdim=True)
            probs_subset, idx_subset, _ = self.get_top_p_k_subset(
                scaled_logits_valid,
                self.config.top_p,
                self.config.top_k,
                self.config.enable_topk_optimization,
                log_denom=scaled_log_denom_valid,
            )
            # 4) Smooth over already tau-legal + top-p/top-k candidate subset.
            # If smoothing_factor is a per-row tensor [B_active, 1], index it down to [B_valid, 1].
            sf_for_valid = (
                smoothing_factor.index_select(0, valid_rows)
                if isinstance(smoothing_factor, torch.Tensor)
                else smoothing_factor
            )
            probs_final = self.apply_smoothing(probs_subset, sf_for_valid)

        # 5) Sample
        next_tokens = torch.zeros(
            base_logits.size(0),
            dtype=torch.long,
            device=base_logits.device,
        )
        selected_full_log_prob = torch.zeros(
            base_logits.size(0),
            dtype=base_logits.dtype,
            device=base_logits.device,
        )
        if valid_rows.numel() > 0 and probs_final is not None and idx_subset is not None:
            with profile_block("sampler.multinomial"):
                next_pos_in_subset = torch.multinomial(
                    probs_final,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            with profile_block("sampler.gather_logprob"):
                next_tokens_valid = idx_subset.gather(-1, next_pos_in_subset.unsqueeze(-1)).squeeze(-1)
                selected_logit_valid = base_logits_valid.gather(-1, next_tokens_valid.unsqueeze(-1)).squeeze(-1)
                selected_full_log_prob_valid = selected_logit_valid - base_log_denom_valid.squeeze(-1)
                next_tokens.index_copy_(0, valid_rows, next_tokens_valid)
                selected_full_log_prob.index_copy_(0, valid_rows, selected_full_log_prob_valid)

        return {
            "tokens": next_tokens,
            "full_log_probs": selected_full_log_prob,
            "has_valid_token": has_valid,
        }
