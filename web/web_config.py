"""Map web UI parameters to BOA Config dataclass."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from config import Config


# Models with their recommended default sampling parameters.
MODEL_PRESETS = {
    "meta-llama/Llama-3.1-8B-Instruct": {"temperature": 0.6, "top_p": 0.9, "top_k": 20},
    "meta-llama/Llama-2-7b-chat-hf": {"temperature": 0.6, "top_p": 0.9, "top_k": 20},
    "lmsys/vicuna-7b-v1.5": {"temperature": 0.9, "top_p": 0.6, "top_k": 20},
    "Qwen/Qwen3-8B": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "google/gemma-3-4b-it": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "Qwen/Qwen2.5-32B-Instruct-AWQ": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}

JUDGER_PRESETS = [
    {"id": "nuanced", "name": "Nuanced Judger", "workload": "single_turn"},
    {"id": "agent_safety", "name": "Agent Safety Judger", "workload": "agent"},
]


def build_web_config(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    likelihood: float,
    search_strategy: str = "greedy",
    pa_min_depth: int = 3,
    pa_shallow_samples: int = 20,
    pa_shallow_score: float = 8000,
    pa_subtree_top_k: int = 5,
    pa_search_temperature: float = 0.5,
    mcts_exploration_constant: float = 1.414,
    judger_type: str = "nuanced",
    custom_judger_prompt: str = "",
    time_limit_sec: float = 120,
    chunk_size: int = 2,
) -> Config:
    """Build a Config from web UI parameters.

    Creates a temporary benchmark JSON file containing the single prompt
    so the existing pipeline can load it normally.
    """
    cfg = Config()

    # Target model
    cfg.target_model = model
    cfg.target_engine_name = "hf"

    # Sampling parameters (user-controlled)
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.top_k = top_k
    cfg.likelihood = likelihood
    cfg.chunk_size = chunk_size

    if search_strategy == "phase_aware":
        cfg.search_strategy = "phase_aware"
        cfg.pa_min_depth = pa_min_depth
        cfg.pa_shallow_samples = pa_shallow_samples
        cfg.pa_shallow_score = pa_shallow_score
        cfg.pa_subtree_top_k = pa_subtree_top_k
        cfg.pa_search_temperature = pa_search_temperature
    elif search_strategy == "mcts":
        cfg.search_strategy = "mcts"
        cfg.mcts_exploration_constant = mcts_exploration_constant
    else:
        cfg.search_strategy = "greedy"

    # Time budget
    cfg.time_limit_sec = time_limit_sec

    # Single prompt mode
    cfg.harmful_prompt_start = 1
    cfg.harmful_prompt_end = 1

    # Skip threshold building for speed in web mode
    cfg.benign_n = 0

    # Workload based on judger
    if judger_type == "nuanced":
        cfg.workload_name = "single_turn"
    elif judger_type == "agent_safety":
        cfg.workload_name = "agent"
    elif judger_type == "custom":
        cfg.workload_name = "single_turn"
    else:
        cfg.workload_name = "single_turn"

    # Logging: use info level for detailed diagnostics
    cfg.logger_mode = "info"

    # Store custom judger prompt for later use by observer/run_manager
    cfg._custom_judger_prompt = custom_judger_prompt

    return cfg

