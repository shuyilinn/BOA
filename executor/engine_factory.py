from __future__ import annotations

from typing import Any, Optional

from engines.mock_engine import MockTargetModelEngine
from judgers.judger_engines.mock_api_engine import MockApiJudgerEngine
from judgers.judger_engines.mock_engine import MockJudgerEngine
from utils.logger import setup_logger

logger = setup_logger("EngineFactory")


def create_target_engine(config: Any):
    use_mock = bool(config.test_mode)
    if not use_mock:
        use_mock = config.target_engine_name == "mock"
    if use_mock:
        logger.info("Using mock target engine (test_mode=%s)", bool(config.test_mode))
        return MockTargetModelEngine(
            config.target_model,
            mock_target_text=config.mock_target_text,
            mock_target_logit_high=float(config.mock_target_logit_high),
            mock_target_logit_low=float(config.mock_target_logit_low),
            mock_target_noise_std=float(config.mock_target_noise_std),
            mock_target_seed=int(config.mock_target_seed),
            mock_sampler_sleep_sec=max(0.0, float(config.mock_sampler_sleep_sec)),
        )

    if config.target_engine_name == "hf":
        from engines.hf_engine import HuggingFaceTargetModelEngine
        return HuggingFaceTargetModelEngine(config.target_model, cuda_ids=config.target_model_cuda_number)
    if config.target_engine_name == "vllm":
        from engines.vllm_engine import VllmTargetModelEngine
        return VllmTargetModelEngine(
            config.target_model,
            cuda_ids=config.target_model_cuda_number,
            gpu_memory_utilization=config.target_gpu_memory_utilization,
            top_k=config.top_k,
            enable_topk_optimization=config.enable_topk_optimization,
        )
    raise ValueError(f"Invalid engine: {config.target_engine_name}")


def create_judger_engine(config: Any):
    judger_engine_name = config.judger_engine_name
    judger_model = config.workload_configs[config.workload_name].get("judger_model", config.judger_model)
    use_mock = bool(config.test_mode)
    if not use_mock:
        use_mock = judger_engine_name == "mock"
    if use_mock:
        logger.info("Using mock judger engine (test_mode=%s)", bool(config.test_mode))
        return MockJudgerEngine(
            name="mock-judger",
            cuda_number=0,
            mock_refused=bool(config.mock_refused),
            mock_score=float(config.mock_score),
            mock_score_mode=str(config.mock_score_mode),
            mock_score_min=int(config.mock_score_min),
            mock_score_max=int(config.mock_score_max),
            mock_high_score_prob=float(config.mock_high_score_prob),
            mock_high_score_min=int(config.mock_high_score_min),
            mock_high_score_max=int(config.mock_high_score_max),
            mock_score_seed=int(config.mock_score_seed),
            mock_judger_sleep_sec=max(0.0, float(config.mock_judger_sleep_sec)),
        )

    if judger_engine_name == "hf":
        from judgers.judger_engines.hf_engine import HFLocalJudgerEngine
        return HFLocalJudgerEngine(judger_model, config.judger_cuda_number)
    if judger_engine_name == "vllm":
        from judgers.judger_engines.vllm_engine import VLLMLocalJudgerEngine
        return VLLMLocalJudgerEngine(
            judger_model,
            config.judger_cuda_number,
            max_model_len=int(config.judger_max_model_len),
            gpu_memory_utilization=config.judger_gpu_memory_utilization,
        )
    raise ValueError(f"Invalid judger engine: {judger_engine_name}")


def create_api_judger_engine(config: Any) -> Optional[Any]:
    if bool(config.test_mode):
        return MockApiJudgerEngine(
            name="mock-api-judger",
            cuda_number=0,
            mock_api_unsafe_prob=float(config.mock_api_unsafe_prob),
            mock_api_seed=config.mock_api_seed,
        )
    try:
        from judgers.judger_engines.openai_engine import OpenAIJudgerEngine
        return OpenAIJudgerEngine(config.api_judger_model)
    except Exception as e:
        logger.warning("OpenAI api judger unavailable; disable layer4. err=%s", e)
        return None
