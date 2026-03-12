from __future__ import annotations

from boa_types.interaction import EnvironmentFeedbackBundle, EnvironmentRequestBundle
from utils.logger import setup_logger

from .base_environment import BaseEnvironment

logger = setup_logger("SingleTurnEnvironment")


class SingleTurnEnvironment(BaseEnvironment):
    env_type = "single_turn"

    def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
        logger.info("Single-turn environment reached EOS; ending path.")
        return EnvironmentFeedbackBundle(terminal=True)
