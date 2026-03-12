import os
import math
import torch
import numpy as np
import re
from typing import List, Any, Optional
from engines.hf_engine import HuggingFaceTargetModelEngine
from engines.vllm_engine import VllmTargetModelEngine
from utils.batch_policy import RuntimeOOMBatchRunner, get_initial_batch_size
from utils.logger import setup_logger

logger = setup_logger("threshold_builder", mode="debug")


# ==========================================
# 1. Core build logic (batch optimized)
# ==========================================

def build_probability_threshold_baseline(
    config: Any,
    prompts: List[str],
    engine: Optional[Any] = None,
) -> List[float]:
    """
    Build baseline probability threshold using batched sampling.
    """

    owned_engine = False
    if engine is None:
        if config.target_engine_name == "hf":
            engine = HuggingFaceTargetModelEngine(config.target_model, cuda_ids=config.target_model_cuda_number)
        elif config.target_engine_name == "vllm":
            engine = VllmTargetModelEngine(config.target_model, cuda_ids=config.target_model_cuda_number)
        else:
            raise ValueError(f"Invalid engine: {config.target_engine_name}")
        owned_engine = True

    try:
        tokenizer = engine.get_tokenizer()
        device = engine.device

        temperature = config.temperature
        sequences_per_prompt = config.threshold_baseline_sequences_per_prompt
        gen_length = config.threshold_baseline_generation_length
        batch_size = max(1, int(config.threshold_baseline_batch_size))
        use_dynamic_bs = bool(config.use_dynamic_threshold_batch_size)
        prompt_len = int(config.threshold_estimate_prompt_len)
        util = float(
            config.threshold_gpu_memory_utilization
        )
        overhead = float(config.hf_batch_estimate_overhead)
        max_cap = int(config.max_batch_size)
        min_bs = int(config.min_batch_size)
        if use_dynamic_bs:
            batch_size = get_initial_batch_size(
                engine,
                prompt_len=prompt_len,
                max_new_tokens=int(gen_length),
                config_batch_size=batch_size,
                use_dynamic=True,
                gpu_memory_utilization=util,
                overhead_ratio=overhead,
                max_batch_cap=max_cap,
                min_batch_size=min_bs,
                logger=logger,
                policy_name="Threshold baseline",
            )
        eos_id = tokenizer.eos_token_id

        flat_tasks = []
        for i, p in enumerate(prompts):
            for _ in range(sequences_per_prompt):
                flat_tasks.append((i, p))

        all_results = [[] for _ in range(len(prompts))]

        total_tasks = len(flat_tasks)
        logger.info(
            "Step 1/3: start batch sampling (total_tasks=%s, initial_batch_size=%s)",
            total_tasks,
            batch_size,
        )
        oom_runner = RuntimeOOMBatchRunner(
            initial_batch_size=int(batch_size),
            logger=logger,
            policy_name="Threshold baseline",
            min_batch_size=min_bs,
        )
        processed_tasks = 0

        def _run_chunk(batch_slice):
            nonlocal processed_tasks
            current_batch_size = len(batch_slice)
            start = processed_tasks
            end = processed_tasks + current_batch_size
            logger.info(
                "Threshold chunk: tasks %s:%s/%s (chunk_size=%s)",
                start,
                end,
                total_tasks,
                current_batch_size,
            )
            batch_prompts = [t[1] for t in batch_slice]
            batch_indices = [t[0] for t in batch_slice]

            encodings = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = encodings["input_ids"].to(device)

            active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=device)
            cumulative_log_probs = torch.zeros(current_batch_size, device=device)
            batch_sequences_history = [[] for _ in range(current_batch_size)]

            logits, kv_cache = engine.forward_step(input_ids, kv_cache=None)

            for step in range(gen_length):
                if not active_mask.any():
                    break

                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_probs = probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
                step_log_p = torch.log(token_probs.clamp(min=1e-12))

                cumulative_log_probs += step_log_p

                for b in range(current_batch_size):
                    if active_mask[b]:
                        batch_sequences_history[b].append(cumulative_log_probs[b].item())
                        if next_tokens[b] == eos_id:
                            active_mask[b] = False

                next_input = next_tokens.unsqueeze(-1)
                logits, kv_cache = engine.forward_step(next_input, kv_cache=kv_cache)

            for b, prompt_idx in enumerate(batch_indices):
                all_results[prompt_idx].append(batch_sequences_history[b])
            processed_tasks += current_batch_size

        oom_runner.run(flat_tasks, _run_chunk)

        logger.info("Step 2/3: aggregate per-position means...")
        all_prompt_means = []
        for p_results in all_results:
            if not p_results:
                continue
            max_p_len = max(len(s) for s in p_results)
            p_means = []
            for pos in range(max_p_len):
                vals = [s[pos] for s in p_results if pos < len(s)]
                p_means.append(np.mean(vals))
            all_prompt_means.append(p_means)

        final_len = max(len(m) for m in all_prompt_means)
        final_baseline = []
        for pos in range(final_len):
            vals = [m[pos] for m in all_prompt_means if pos < len(m)]
            final_baseline.append(float(np.mean(vals)))

        logger.info(f"Step 3/3: done (baseline_len={len(final_baseline)})")
        return final_baseline
    finally:
        if owned_engine:
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


# ==========================================
# 2. Persistence (load/save)
# ==========================================

def _threshold_file_path(config: Any) -> str:
    # NOTE: target_model may contain "/" (e.g. "Qwen/Qwen3-8B"), which would
    # accidentally create sub-directories and break file writing. Sanitize it.
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(config.target_model))
    directory = "./probability_threshold/threshold"
    return (
        f"{directory}/{safe_model}_{config.target_engine_name}_{config.top_p}_{config.top_k}_{config.temperature}_"
        f"{config.threshold_baseline_sequences_per_prompt}_{config.threshold_baseline_generation_length}.txt"
    )


def _apply_likelihood_correction(config: Any, baseline: List[float]) -> List[float]:
    log_likelihood = math.log(config.likelihood)
    return [b + log_likelihood for b in baseline]


def try_load_threshold(config: Any) -> Optional[List[float]]:
    threshold_file = _threshold_file_path(config)
    if not os.path.exists(threshold_file):
        logger.info("Threshold cache not found: %s", threshold_file)
        return None

    logger.info("Threshold loaded from cache: %s", threshold_file)
    with open(threshold_file, "r") as f:
        baseline = [float(l.strip()) for l in f.readlines()]
    return _apply_likelihood_correction(config, baseline)


def build_or_load_threshold(
    config: Any,
    benign_prompts: List[str],
    engine: Optional[Any] = None,
) -> List[float]:
    """
    Load cached threshold baseline if present; otherwise build and save it.
    """
    cached_threshold = try_load_threshold(config)
    if cached_threshold is not None:
        return cached_threshold

    logger.info("Cache miss; building threshold baseline...")
    baseline = build_probability_threshold_baseline(config, benign_prompts, engine=engine)

    threshold_file = _threshold_file_path(config)
    os.makedirs(os.path.dirname(threshold_file), exist_ok=True)
    with open(threshold_file, "w") as f:
        for v in baseline:
            f.write(f"{v}\n")

    return _apply_likelihood_correction(config, baseline)
