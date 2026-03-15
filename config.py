from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

@dataclass
class Config:
    """
    Unified configuration class.
    Parameters are grouped by runtime responsibility for easier tuning.
    """

    # -------------------------------------------------------------------------
    # Core Runtime
    # -------------------------------------------------------------------------
    # Budget
    time_limit_sec: Optional[float] = 100
    token_limit: Optional[int] = None
    depth_limit: Optional[int] = None
    node_limit: Optional[int] = None

    # Reproducibility / tracing
    random_seed: int = 2026
    deterministic_sampling: bool = True
    record_trace: bool = True

    # Profiling
    enable_profiling: bool = True
    torch_profiler_steps: int = 3   # 0 = disabled; N = profile first N decode steps per generate call

    # Logging
    logger_mode: str = "warning"  # debug|info|warning|error
    logger_tree_max_depth: int = 100000
    logger_tree_max_nodes: int = 100000
    run_id: Optional[str] = None

    # -------------------------------------------------------------------------
    # Data / Benchmark
    # -------------------------------------------------------------------------
    workload_name: str = "agent"
    workload_configs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "single_turn": {
                "benchmark_path": "./benchmark/boa_benchmark/jailbreak_oracle_benchmark.json",
                "environment_type": "single_turn",
                "clean_response": True,
                "terminal_unsafe_enabled": True,
                "terminal_unsafe_layers": [4],
                "judger_profile": "layered_single_turn",
                "judger_model": "unsloth/Qwen2.5-3B-Instruct",
            },
            "agent": {
                "benchmark_path": "./benchmark/agentsafetybench/released_data.json",
                "environment_type": "single_turn",
                "clean_response": False,
                "terminal_unsafe_enabled": True,
                "terminal_unsafe_layers": [3],
                "judger_profile": "agent_safety",
                "judger_model": "thu-coai/ShieldAgent",
            },
            "multi_turn": {
                "benchmark_path": "",
                "environment_type": "multi_turn",
                "clean_response": True,
                "terminal_unsafe_enabled": False,
                "terminal_unsafe_layers": [],
                "judger_profile": "",
            },
        }
    )
    # Benign benchmark source (HF dataset name or local path)
    benign_path: Optional[str] = "JBB-Behaviors"
    benign_n: Optional[int] = None
    # 1-based inclusive slicing on harmful prompts
    harmful_prompt_start: int = 1
    harmful_prompt_end: Optional[int] = 1

    # -------------------------------------------------------------------------
    # Search / L3 Chunk Expansion
    # -------------------------------------------------------------------------
    # Chunk-length constraint
    chunk_size: int = 1
    # Mode: off | stop | limit
    chunk_len_mode: str = "stop"

    # Chunk-width constraint
    # Maximum number of child chunks L3 can return to L2.
    chunk_width: int = 100
    # Mode: off | stop | limit
    chunk_width_mode: str = "off"

    # Dynamic stop/limit (logits-based).
    # Mode: off | stop | limit
    dynamic_stop_mode: str = "off"
    # Optional trigger thresholds (None means disabled).
    # Trigger condition is OR-composed across enabled thresholds.
    # max_prob >= threshold, margin >= threshold, entropy <= threshold.
    dynamic_max_prob_threshold: Optional[float] = None
    dynamic_entropy_threshold: Optional[float] = None
    dynamic_margin_threshold: Optional[float] = None

    # -------------------------------------------------------------------------
    # Target Model / Sampler
    # -------------------------------------------------------------------------
    # Engine
    target_model: str = "Qwen/Qwen3-8B"
    target_engine_name: str = "hf"  # hf | vllm | powerinfer | mock | gencache (future)
    target_model_cuda_number: Optional[int] = 1
    target_gpu_memory_utilization: float = 0.9

    # Sampler batching
    use_dynamic_batch_size: bool = True
    sampler_batch_size: int = 100
    sampler_estimate_prompt_len: int = 512
    expander_batch_size: int = 100
    hf_batch_estimate_overhead: float = 1
    min_batch_size: int = 1
    max_batch_size: int = 4096

    # Sampling lengths / counts (tree search)
    sample_new_tokens: int = 150
    sample_full_new_tokens: int = 500
    sampler_number: int = 10

    # Attack sampling (separate from tree-search sampling)
    attack_sample_new_tokens: int = 512
    attack_sampler_number: int = 10

    # Sampling strategy
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    likelihood: float = 0.0001
    uniform_smoothing_factor: float = 1
    uniform_smoothing_steps: int = 20  # Apply uniform smoothing only for the first N steps; 0 = all steps

    # Threshold baseline generation
    threshold_baseline_generation_length: int = 500
    threshold_baseline_sequences_per_prompt: int = 20
    threshold_baseline_batch_size: int = 100
    use_dynamic_threshold_batch_size: bool = True
    threshold_gpu_memory_utilization: float = 0.9
    threshold_estimate_prompt_len: int = 512

    # -------------------------------------------------------------------------
    # Multi-turn / Interaction Expansion
    # -------------------------------------------------------------------------
    # If true, assistant nodes ending with EOS will be marked awaiting_environment by default.
    auto_expand_environment_after_eos: bool = False
    # Optional simulated environment-side events.
    simulated_user_responses: Optional[list[str]] = None
    simulated_tool_outputs: Optional[list[str]] = None
    simulated_env_observations: Optional[list[str]] = None
    simulated_execution_errors: Optional[list[str]] = None

    # -------------------------------------------------------------------------
    # Judger
    # -------------------------------------------------------------------------
    judger_model: str = "unsloth/Qwen2.5-3B-Instruct"
    judger_cuda_number: Optional[int] = 2
    judger_engine_name: str = "vllm"  # vllm | hf | mock
    judger_gpu_memory_utilization: float = 0.3
    judger_max_model_len: int = 8192
    api_judger_model: str = "gpt-4o"
    judger_batch_size: int = 128
    judger_estimate_prompt_len: int = 1024
    judger_estimate_max_new_tokens: int = 200
    layer3_filter_threshold: float = 7000

    # -------------------------------------------------------------------------
    # Runtime Optimizations
    # -------------------------------------------------------------------------
    enable_sampling_cache: bool = True
    # Cap cache entries to avoid unbounded memory growth.
    cache_max_entries: Optional[int] = 50000
    # Minimum number of suffix tokens required for a cache hit to be considered useful.
    cache_min_suffix_tokens: int = 50
    enable_sampling_buffer: bool = True
    enable_judging_buffer: bool = True
    buffer_capacity: int = 10000
    enable_topk_optimization: bool = True
    # Top-K prefilter size used before Top-P (effective prefilter is max(prefilter, requested-k)).
    topk_prefilter_size: int = 512
    enable_refuse_pattern_matching: bool = True
    enable_refuse_judger: bool = True
    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------use_dynamic_batch_size------------------------
    enable_checkpointing: bool = True
    checkpoint_dir: str = "checkpoints/run"
    checkpoint_every_expansions: int = 10
    checkpoint_every_seconds: Optional[float] = None
    save_trace: bool = True
    resume_from: Optional[str] = None

    # -------------------------------------------------------------------------
    # Test / Mock
    # -------------------------------------------------------------------------
    test_mode: bool = False

    # Mock target model
    mock_target_text: str = "OK"
    mock_target_logit_high: float = 3
    mock_target_logit_low: float = 0.0
    mock_target_noise_std: float = 4
    mock_target_seed: int = 2026
    mock_sampler_sleep_sec: float = 2.0

    # Mock local judger
    mock_refused: bool = False
    mock_score: float = 7.0
    mock_score_mode: str = "random"  # fixed | random
    mock_score_min: int = 1
    mock_score_max: int = 7
    mock_high_score_prob: float = 0.1
    mock_high_score_min: int = 8
    mock_high_score_max: int = 10
    mock_score_seed: int = 2026
    mock_judger_sleep_sec: float = 2.0

    # Mock API judger
    mock_api_unsafe_prob: float = 0
    mock_api_seed: int = 2026

    def to_dict(self) -> dict[str, Any]:
        """Export config as a plain dict (for logging/serialization)."""
        return asdict(self)
