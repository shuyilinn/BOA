"""CLI resolver for the unified Config dataclass."""

from __future__ import annotations

import argparse
import sys

from config import Config


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _provided_cli_args(argv: list[str]) -> set[str]:
    provided: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        provided.add(key.replace("-", "_"))
    return provided


def resolve_config() -> Config:
    """Parse CLI arguments and apply explicit overrides to Config."""
    parser = argparse.ArgumentParser()

    # Core runtime
    parser.add_argument("--time_limit_sec", type=float, default=None)
    parser.add_argument("--token_limit", type=int, default=None)
    parser.add_argument("--depth_limit", type=int, default=None)
    parser.add_argument("--node_limit", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--deterministic_sampling", type=_parse_bool, default=None)
    parser.add_argument("--record_trace", type=_parse_bool, default=None)
    parser.add_argument("--logger_mode", type=str, default=None, choices=["debug", "info", "warning", "error"])

    # Data / benchmark
    parser.add_argument("--benign_path", type=str, default=None)
    parser.add_argument("--benign_n", type=int, default=None)
    parser.add_argument("--harmful_prompt_start", type=int, default=None)
    parser.add_argument("--harmful_prompt_end", type=int, default=None)
    parser.add_argument("--workload_name", type=str, default=None)

    # Search / expansion
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--chunk_len_mode", type=str, default=None, choices=["off", "stop", "limit"])
    parser.add_argument("--chunk_width", type=int, default=None)
    parser.add_argument("--chunk_width_mode", type=str, default=None, choices=["off", "stop", "limit"])
    parser.add_argument("--dynamic_stop_mode", type=str, default=None, choices=["off", "stop", "limit"])
    parser.add_argument("--dynamic_max_prob_threshold", type=float, default=None)
    parser.add_argument("--dynamic_entropy_threshold", type=float, default=None)
    parser.add_argument("--dynamic_margin_threshold", type=float, default=None)

    # Target model / sampler
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--target_engine_name", type=str, default=None, choices=["hf", "vllm", "mock"])
    parser.add_argument("--target_model_cuda_number", type=int, default=None)
    parser.add_argument("--target_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--use_dynamic_batch_size", type=_parse_bool, default=None)
    parser.add_argument("--sampler_batch_size", type=int, default=None)
    parser.add_argument("--sampler_estimate_prompt_len", type=int, default=None)
    parser.add_argument("--expander_batch_size", type=int, default=None)
    parser.add_argument("--sample_new_tokens", type=int, default=None)
    parser.add_argument("--sample_full_new_tokens", type=int, default=None)
    parser.add_argument("--sampler_number", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--topk_prefilter_size", type=int, default=None)
    parser.add_argument("--likelihood", type=float, default=None)
    parser.add_argument("--uniform_smoothing_factor", type=float, default=None)
    parser.add_argument("--uniform_smoothing_steps", type=int, default=None)

    # Threshold baseline
    parser.add_argument("--threshold_baseline_generation_length", type=int, default=None)
    parser.add_argument("--threshold_baseline_sequences_per_prompt", type=int, default=None)
    parser.add_argument("--threshold_baseline_batch_size", type=int, default=None)
    parser.add_argument("--use_dynamic_threshold_batch_size", type=_parse_bool, default=None)
    parser.add_argument("--threshold_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--threshold_estimate_prompt_len", type=int, default=None)

    # Judger
    parser.add_argument("--judger_model", type=str, default=None)
    parser.add_argument("--judger_cuda_number", type=int, default=None)
    parser.add_argument("--judger_engine_name", type=str, default=None, choices=["hf", "vllm", "mock"])
    parser.add_argument("--judger_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--judger_max_model_len", type=int, default=None)
    parser.add_argument("--api_judger_model", type=str, default=None)
    parser.add_argument("--judger_batch_size", type=int, default=None)
    parser.add_argument("--judger_estimate_prompt_len", type=int, default=None)
    parser.add_argument("--judger_estimate_max_new_tokens", type=int, default=None)
    parser.add_argument("--layer3_filter_threshold", type=float, default=None)

    # Runtime optimizations
    parser.add_argument("--enable_sampling_cache", type=_parse_bool, default=None)
    parser.add_argument("--enable_sampling_buffer", type=_parse_bool, default=None)
    parser.add_argument("--enable_judging_buffer", type=_parse_bool, default=None)
    parser.add_argument("--enable_topk_optimization", type=_parse_bool, default=None)
    parser.add_argument("--enable_topp_optimization", type=_parse_bool, default=None)
    parser.add_argument("--enable_refuse_pattern_matching", type=_parse_bool, default=None)
    parser.add_argument("--enable_refuse_judger", type=_parse_bool, default=None)

    # Checkpointing
    parser.add_argument("--enable_checkpointing", type=_parse_bool, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every_expansions", type=int, default=None)
    parser.add_argument("--checkpoint_every_seconds", type=float, default=None)
    parser.add_argument("--save_trace", type=_parse_bool, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # Test / mock
    parser.add_argument("--test_mode", type=_parse_bool, default=None)
    parser.add_argument("--mock_target_text", type=str, default=None)
    parser.add_argument("--mock_target_logit_high", type=float, default=None)
    parser.add_argument("--mock_target_logit_low", type=float, default=None)
    parser.add_argument("--mock_target_noise_std", type=float, default=None)
    parser.add_argument("--mock_target_seed", type=int, default=None)
    parser.add_argument("--mock_sampler_sleep_sec", type=float, default=None)
    parser.add_argument("--mock_refused", type=_parse_bool, default=None)
    parser.add_argument("--mock_score", type=float, default=None)
    parser.add_argument("--mock_score_mode", type=str, default=None, choices=["fixed", "random"])
    parser.add_argument("--mock_score_min", type=int, default=None)
    parser.add_argument("--mock_score_max", type=int, default=None)
    parser.add_argument("--mock_high_score_prob", type=float, default=None)
    parser.add_argument("--mock_high_score_min", type=int, default=None)
    parser.add_argument("--mock_high_score_max", type=int, default=None)
    parser.add_argument("--mock_score_seed", type=int, default=None)
    parser.add_argument("--mock_judger_sleep_sec", type=float, default=None)
    parser.add_argument("--mock_api_unsafe_prob", type=float, default=None)
    parser.add_argument("--mock_api_seed", type=int, default=None)

    args = parser.parse_args()
    provided = _provided_cli_args(sys.argv[1:])
    config = Config()

    def set_if_provided(arg_name: str, attr_name: str | None = None) -> None:
        key = arg_name.replace("-", "_")
        if key not in provided:
            return
        value = getattr(args, key)
        if value is None:
            return
        setattr(config, attr_name or key, value)

    # Direct 1:1 field mapping
    direct_fields = [
        "time_limit_sec",
        "token_limit",
        "depth_limit",
        "node_limit",
        "random_seed",
        "deterministic_sampling",
        "record_trace",
        "logger_mode",
        "benign_path",
        "benign_n",
        "harmful_prompt_start",
        "harmful_prompt_end",
        "workload_name",
        "chunk_size",
        "chunk_len_mode",
        "chunk_width",
        "chunk_width_mode",
        "dynamic_stop_mode",
        "dynamic_max_prob_threshold",
        "dynamic_entropy_threshold",
        "dynamic_margin_threshold",
        "target_model",
        "target_engine_name",
        "target_model_cuda_number",
        "target_gpu_memory_utilization",
        "use_dynamic_batch_size",
        "sampler_batch_size",
        "sampler_estimate_prompt_len",
        "expander_batch_size",
        "sample_new_tokens",
        "sample_full_new_tokens",
        "sampler_number",
        "temperature",
        "top_p",
        "top_k",
        "topk_prefilter_size",
        "likelihood",
        "uniform_smoothing_factor",
        "uniform_smoothing_steps",
        "threshold_baseline_generation_length",
        "threshold_baseline_sequences_per_prompt",
        "threshold_baseline_batch_size",
        "use_dynamic_threshold_batch_size",
        "threshold_gpu_memory_utilization",
        "threshold_estimate_prompt_len",
        "judger_model",
        "judger_cuda_number",
        "judger_engine_name",
        "judger_gpu_memory_utilization",
        "judger_max_model_len",
        "api_judger_model",
        "judger_batch_size",
        "judger_estimate_prompt_len",
        "judger_estimate_max_new_tokens",
        "layer3_filter_threshold",
        "enable_sampling_cache",
        "enable_sampling_buffer",
        "enable_judging_buffer",
        "enable_topk_optimization",
        "enable_refuse_pattern_matching",
        "enable_refuse_judger",
        "enable_checkpointing",
        "checkpoint_dir",
        "checkpoint_every_expansions",
        "checkpoint_every_seconds",
        "save_trace",
        "resume_from",
        "test_mode",
        "mock_target_text",
        "mock_target_logit_high",
        "mock_target_logit_low",
        "mock_target_noise_std",
        "mock_target_seed",
        "mock_sampler_sleep_sec",
        "mock_refused",
        "mock_score",
        "mock_score_mode",
        "mock_score_min",
        "mock_score_max",
        "mock_high_score_prob",
        "mock_high_score_min",
        "mock_high_score_max",
        "mock_score_seed",
        "mock_judger_sleep_sec",
        "mock_api_unsafe_prob",
        "mock_api_seed",
    ]
    for field in direct_fields:
        set_if_provided(field)

    # Backward compatibility: old CLI flag name.
    if "enable_topk_optimization" not in provided:
        set_if_provided("enable_topp_optimization", "enable_topk_optimization")

    return config
