from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from judgers.atomic.agent_safety_judger import AgentSafetyJudger
from judgers.atomic.api_nuanced_judger import ApiNuancedJudger
from judgers.atomic.nuanced_judger import BoaJudger
from judgers.atomic.refusal_judger import RefusalJudger
from judgers.atomic.refusal_state_machine import RefusalPatternJudger
from judgers.base_judger import PipelineJudger
from judgers.tree_guide_judger import TreeGuideJudger
from utils.logger import setup_logger

logger = setup_logger("JudgerFactory")

JudgerKey = Tuple[str, str]
JudgerBuilder = Callable[[Any, Any, Any], PipelineJudger]


def _build_layered_single_turn_judger(
    local_judger_engine: Any,
    api_judger_engine: Any,
    config: Any,
) -> TreeGuideJudger:
    layer1_judger = RefusalPatternJudger(config) if bool(getattr(config, "enable_refuse_pattern_matching", True)) else None
    layer2_judger = RefusalJudger(local_judger_engine) if bool(getattr(config, "enable_refuse_judger", True)) else None
    layer3_judger = BoaJudger(local_judger_engine, filter_threshold=float(config.layer3_filter_threshold))
    layer4_judger = ApiNuancedJudger(api_judger_engine) if api_judger_engine is not None else None
    return TreeGuideJudger(
        local_judger_engine=local_judger_engine,
        api_judger_engine=api_judger_engine,
        config=config,
        layer1_judger=layer1_judger,
        layer2_judger=layer2_judger,
        layer3_judger=layer3_judger,
        layer4_judger=layer4_judger,
    )


def _build_agent_judger(
    local_judger_engine: Any,
    api_judger_engine: Any,
    config: Any,
) -> TreeGuideJudger:
    return TreeGuideJudger(
        local_judger_engine=local_judger_engine,
        api_judger_engine=api_judger_engine,
        config=config,
        layer1_judger=None,
        layer2_judger=None,
        layer3_judger=AgentSafetyJudger(local_judger_engine),
        layer4_judger=None,
    )


_JUDGER_REGISTRY: Dict[JudgerKey, JudgerBuilder] = {
    ("single_turn", "layered_single_turn"): _build_layered_single_turn_judger,
    ("agent", "agent_safety"): _build_agent_judger,
}


def register_judger(workload_name: str, judger_profile: str, builder: JudgerBuilder) -> None:
    _JUDGER_REGISTRY[(str(workload_name).strip(), str(judger_profile).strip())] = builder


def create_judger(
    config: Any,
    local_judger_engine: Any,
    api_judger_engine: Any,
) -> PipelineJudger:
    workload_name = config.workload_name
    judger_profile = config.workload_configs[workload_name]["judger_profile"]
    key = (workload_name, judger_profile)
    builder = _JUDGER_REGISTRY[key]
    logger.info(
        "Creating judger for workload=%s profile=%s",
        workload_name,
        judger_profile,
    )
    return builder(local_judger_engine, api_judger_engine, config)
