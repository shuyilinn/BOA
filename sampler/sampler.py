# TODO: The number of samples should vary with the logit distribution:
# the more confident it is, the fewer samples; the more uncertain it is, the more samples.
# For now, keep the original version; we will optimize it later.

# NOTE: This file is currently a lightweight scaffold. The actual sampling strategy
# should be implemented in `customized_sampler`.

# [Shuyi: NOTE: sampling will use the probability that after re-normalization, but in the tree, it will still store the probs that before re-normalization. why? I cannot remember the reason.]

import time
import torch
from typing import List, Tuple, Optional
from sampler.customized_sampler import CustomizedSampler
from utils.logger import setup_logger
from profiler import profile, maybe_torch_profile, get_torch_profile_dir

logger = setup_logger("Sampler")


class Sampler:
    def __init__(self, engine, config):
        self.config = config
        self.engine = engine
        self.tokenizer = self.engine.get_tokenizer()
        self.cs = CustomizedSampler(config)
        self.log_probability_threshold = None
        self.stop_token_ids = self._resolve_stop_token_ids()
        self._stop_lookup: Optional[torch.Tensor] = None
        logger.info(
            "Sampler stop config: stop_token_ids=%s tokenizer_eos_token_id=%s eos_token=%r",
            self.stop_token_ids,
            self.tokenizer.eos_token_id,
            getattr(self.tokenizer, "eos_token", None),
        )

    def _resolve_stop_token_ids(self) -> List[int]:
        stop_ids = {int(self.tokenizer.eos_token_id)}
        model = getattr(self.engine, "model", None)
        if model is None:
            return sorted(stop_ids)

        generation_eos = getattr(model.generation_config, "eos_token_id", None)
        if generation_eos is None:
            pass
        elif isinstance(generation_eos, int):
            stop_ids.add(int(generation_eos))
        else:
            stop_ids.update(int(token_id) for token_id in generation_eos)
        return sorted(stop_ids)

    def _get_token_id_limit(self) -> int:
        if hasattr(self, "_token_id_limit_cache"):
            return self._token_id_limit_cache
        model = getattr(self.engine, "model", None)
        embeddings = model.get_input_embeddings() if model is not None else None
        token_id_limit = getattr(embeddings, "num_embeddings", None)
        if token_id_limit is None:
            token_id_limit = getattr(getattr(model, "config", None), "vocab_size", None)
        if token_id_limit is None:
            raise ValueError("Unable to resolve model token id limit")
        self._token_id_limit_cache = int(token_id_limit)
        return self._token_id_limit_cache

    def _get_stop_lookup(self, device: torch.device) -> torch.Tensor:
        if self._stop_lookup is None or self._stop_lookup.device != device:
            token_id_limit = self._get_token_id_limit()
            lookup = torch.zeros(token_id_limit, dtype=torch.bool, device=device)
            lookup[self.stop_token_ids] = True
            self._stop_lookup = lookup
        return self._stop_lookup

    def get_batch_size(self) -> int:
        """
        Non-runtime hint used by upper layers (e.g. buffer capacity).
        Runtime sampling batch size is controlled by RuntimeOOMBatchRunner.
        """
        return max(1, int(self.config.sampler_batch_size))

    def uniform_generate(self, token_ids: List[int]) -> List[int]:
        return self.batch_generate([token_ids])[0]

    def batch_uniform_generate(
        self,
        batch_token_ids: List[List[int]],
        *,
        base_generated_lens: Optional[List[int]] = None,
        max_new_tokens: Optional[int] = None,
        return_invalid_flags: bool = False,
    ):
        threshold = self.log_probability_threshold
        tau_gate_enabled = threshold is not None and len(threshold) > 0
        if return_invalid_flags:
            tokens, invalid_flags = self.batch_uniform_generate_with_tau(
                batch_token_ids,
                base_generated_lens=base_generated_lens,
                return_tau=False,
                compute_tau=tau_gate_enabled,
                max_new_tokens=max_new_tokens,
                return_invalid_flags=True,
            )
            return tokens, invalid_flags
        tokens = self.batch_uniform_generate_with_tau(
            batch_token_ids,
            base_generated_lens=base_generated_lens,
            return_tau=False,
            compute_tau=tau_gate_enabled,
            max_new_tokens=max_new_tokens,
            return_invalid_flags=False,
        )
        return tokens

    @profile("sampler.generate")
    def batch_uniform_generate_with_tau(
        self,
        batch_token_ids: List[List[int]],
        *,
        base_generated_lens: Optional[List[int]] = None,
        return_tau: bool = True,
        compute_tau: bool = True,
        max_new_tokens: Optional[int] = None,
        return_invalid_flags: bool = False,
    ):
        """
        Uniform sampling with optional tau (cumulative log-prob) accumulation.

        - Returns generated token ids ONLY (does not include EOS).
        - Tau is always accumulated from full-distribution log-prob provided by sampler.
        - Threshold prune / tau rollback are handled in this driver loop.
        """
        
        if not batch_token_ids:
            if return_tau and return_invalid_flags:
                return [], [], []
            if return_tau:
                return [], []
            if return_invalid_flags:
                return [], []
            return []

        # Resolve generation length for this call.
        effective_max_new_tokens = int(
            max_new_tokens
            if max_new_tokens is not None
            else self.config.sample_new_tokens
        )

        input_ids, attention_mask = self._prepare_inputs(batch_token_ids)
        device = self.engine.device
        B = input_ids.size(0)
        # Prefer explicit override for this call; fallback to config.sample_new_tokens.
        T = effective_max_new_tokens
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (eos or 0)

        # Pre-allocate output on GPU: up to T tokens per sample (excl. EOS)
        out_tokens = torch.full((B, T), pad, device=device, dtype=torch.long)

        # Last written position per sample (-1 = none yet)
        last_written_step = torch.full((B,), -1, device=device, dtype=torch.long)

        # Step at which sequence terminates (EOS or tau-threshold prune), -1 = not yet
        stop_step = torch.full((B,), -1, device=device, dtype=torch.long)

        # Map active batch indices back to original batch
        active_orig = torch.arange(B, device=device, dtype=torch.long)

        tau: Optional[torch.Tensor] = None
        if compute_tau:
            tau = torch.zeros((B,), device=device, dtype=torch.float32)
        threshold = self.log_probability_threshold
        tau_gate_enabled = bool(compute_tau and threshold is not None and len(threshold) > 0)
        invalid_due_to_tau_gate = (
            torch.zeros((B,), device=device, dtype=torch.bool)
            if return_invalid_flags and tau_gate_enabled
            else None
        )
        stop_lookup = self._get_stop_lookup(device)

        smoothing_steps = self.config.uniform_smoothing_steps
        _sf_full = self.config.uniform_smoothing_factor

        # Per-row base offset for smoothing cutoff (in original batch coordinate system).
        # If not provided, defaults to all zeros (treat each sequence as starting from token 0).
        if base_generated_lens is not None:
            _base_lens = torch.tensor(base_generated_lens, device=device, dtype=torch.long)
        else:
            _base_lens = torch.zeros(B, device=device, dtype=torch.long)
        # Once all active sequences pass their smoothing window, skip per-row sf computation.
        _smoothing_all_done = not (smoothing_steps > 0 and _sf_full > 0)

        _tprof_steps = int(getattr(self.config, "torch_profiler_steps", 0))
        _tprof_dir = get_torch_profile_dir() if _tprof_steps > 0 else None

        with torch.no_grad(), maybe_torch_profile(_tprof_dir, steps=_tprof_steps) as tprof:
            # Prefill (engine.forward_step returns [B, V] logits)
            logits, past_kv = self.engine.forward_step(input_ids, kv_cache=None, attention_mask=attention_mask)
            if _tprof_steps > 0:
                torch.cuda.synchronize()
            current_logits = logits

            for step in range(T):
                tau_prev: Optional[torch.Tensor] = None
                min_log_prob: Optional[torch.Tensor] = None
                if tau is not None:
                    tau_prev = tau.index_select(0, active_orig)
                    if threshold is not None and step < len(threshold):
                        min_log_prob = float(threshold[step]) - tau_prev

                # Per-row smoothing: compute effective step for each active sequence.
                # effective_step[i] = base_generated_len[orig_i] + local_step
                if _smoothing_all_done:
                    sf = 0.0
                elif smoothing_steps == 0 or _sf_full <= 0:
                    sf = _sf_full
                else:
                    active_base_lens = _base_lens.index_select(0, active_orig)  # [B_active]
                    still_smoothing = (active_base_lens + step) < smoothing_steps  # [B_active] bool
                    if still_smoothing.any().item():
                        sf = torch.where(still_smoothing, _sf_full, 0.0).unsqueeze(1)  # [B_active, 1]
                    else:
                        sf = 0.0
                        _smoothing_all_done = True  # skip per-row computation for all future steps
                _tp1 = time.perf_counter()
                sample_res = self.cs.sample_step(
                    current_logits,
                    sf,
                    min_log_prob=min_log_prob,
                )
                _tp2 = time.perf_counter()
                if step < 5:
                    print(f"[timer] step={step} sample_step={(_tp2-_tp1)*1000:.1f}ms")
                if tprof is not None:
                    tprof.step()
                next_tokens = sample_res["tokens"]
                full_log_probs = sample_res["full_log_probs"]
                has_valid_token = sample_res["has_valid_token"]

                no_valid_token = ~has_valid_token
                is_eos = stop_lookup[next_tokens]
                if tau is not None:
                    tau_next = tau_prev + full_log_probs.to(tau.dtype)
                    if threshold is not None and step < len(threshold):
                        # Enforce per-position cumulative lower bound at the current step.
                        # If current sampled token violates threshold[t], prune immediately
                        # and rollback tau update for this token.
                        below_threshold = has_valid_token & (tau_next < float(threshold[step]))
                        commit_mask = has_valid_token & (~below_threshold)
                        tau_gate_fail_mask = below_threshold | no_valid_token
                        stop_mask = is_eos | below_threshold | no_valid_token
                    else:
                        commit_mask = has_valid_token
                        tau_gate_fail_mask = no_valid_token
                        stop_mask = is_eos | no_valid_token
                    # index_copy_ with empty indices is a no-op; skip the .any() sync
                    tau.index_copy_(
                        0,
                        active_orig[commit_mask],
                        tau_next[commit_mask],
                    )
                else:
                    tau_gate_fail_mask = no_valid_token
                    stop_mask = is_eos | no_valid_token
                still_active = ~stop_mask

                # Single GPU→CPU sync covers A/B/C/D control flow (was 4-5 separate syncs)
                n_stopped = stop_mask.sum().item()
                B_active = active_orig.numel()

                # A) Record stop_step for just-finished rows
                if n_stopped > 0:
                    finished_orig = active_orig[stop_mask]
                    stop_step[finished_orig] = step
                    if invalid_due_to_tau_gate is not None:
                        invalid_due_to_tau_gate.index_fill_(0, active_orig[tau_gate_fail_mask], True)

                # B) Write token to out_tokens for still-active rows
                if n_stopped < B_active:
                    active_orig_still = active_orig[still_active]
                    tokens_still = next_tokens[still_active]
                    out_tokens[active_orig_still, step] = tokens_still
                    last_written_step[active_orig_still] = step

                # C) All done -> break
                if n_stopped == B_active:
                    break

                # D) Some finished: shrink batch and KV cache
                if 0 < n_stopped < B_active:
                    active_rows = still_active.nonzero(as_tuple=True)[0]
                    next_tokens = next_tokens.index_select(0, active_rows)
                    active_orig = active_orig.index_select(0, active_rows)
                    past_kv = self._shrink_kv_cache(past_kv, active_rows)
                _tp3 = time.perf_counter()
                if step < 5:
                    print(f"[timer] step={step} post_process={(_tp3-_tp2)*1000:.1f}ms")
                # E) Incremental forward (batch may have shrunk)
                _t0 = time.perf_counter()
                logits, past_kv = self.engine.forward_step(next_tokens.unsqueeze(-1), kv_cache=past_kv)
                _t1 = time.perf_counter()
                if _tprof_steps > 0:
                    torch.cuda.synchronize()
                _t2 = time.perf_counter()
                if _tprof_steps > 0 and step < 5:
                    print(f"[timer] step={step} forward_launch={(_t1-_t0)*1000:.1f}ms  sync_wait={(_t2-_t1)*1000:.1f}ms  total={(_t2-_t0)*1000:.1f}ms")
                current_logits = logits

        # Assemble List[List[int]], move to CPU once
        out_tokens_cpu = out_tokens.cpu()
        stop_step_cpu = stop_step.cpu()
        last_written_cpu = last_written_step.cpu()
        lengths = torch.where(stop_step_cpu == -1, last_written_cpu + 1, stop_step_cpu).clamp(min=0)

        results: List[List[int]] = [
            out_tokens_cpu[i, :lengths[i].item()].tolist() for i in range(B)
        ]

        tau_list: List[float] = []
        if return_tau and tau is not None:
            tau_list = tau.detach().cpu().tolist()
        invalid_flags = None
        if return_invalid_flags:
            if invalid_due_to_tau_gate is None:
                invalid_flags = [False] * B
            else:
                invalid_flags = invalid_due_to_tau_gate.detach().cpu().tolist()

        if return_tau and return_invalid_flags:
            return results, tau_list, invalid_flags
        if return_tau:
            return results, tau_list
        if return_invalid_flags:
            return results, invalid_flags
        return results

    def batch_generate(self, batch_ids: List[List[int]]) -> List[List[int]]:
        return self.engine.batch_generate(batch_ids)

    def generate(self, token_ids: List[int]) -> List[int]:
        return self.engine.generate(token_ids)

    @profile("sampler.prepare_inputs")
    def _prepare_inputs(self, batch_ids: List[List[int]]):
        B = len(batch_ids)
        max_len = max((len(x) for x in batch_ids), default=0)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        input_ids_cpu = torch.full((B, max_len), pad_id, dtype=torch.long)
        mask_cpu = torch.zeros((B, max_len), dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            L = len(ids)
            if L:
                input_ids_cpu[i, -L:] = torch.tensor(ids, dtype=torch.long)
                mask_cpu[i, -L:] = 1

        device = self.engine.device
        return input_ids_cpu.to(device, non_blocking=True), mask_cpu.to(device, non_blocking=True)

    def _shrink_kv_cache(self, past_key_values, index: torch.Tensor):
        """Shrink KV cache along batch dim (index_select / list-select).
        HF: tuple of layers of tensors. vLLM: List[List[int]] (full token histories).
        """
        if past_key_values is None:
            return None
        # New HF cache API (e.g., DynamicCache/StaticCache): mutate cache object in-place.
        if hasattr(past_key_values, "batch_select_indices"):
            past_key_values.batch_select_indices(index)
            return past_key_values
        # vLLM path: list of histories
        if isinstance(past_key_values, list):
            idx = index.detach().cpu().tolist()
            return [past_key_values[i] for i in idx]
        new_pkv = []
        for layer_past in past_key_values:
            new_pkv.append(tuple(t.index_select(0, index) for t in layer_past))
        return tuple(new_pkv)
