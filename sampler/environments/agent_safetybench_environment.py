from __future__ import annotations

from boa_types.interaction import EnvironmentFeedbackBundle, EnvironmentRequestBundle

from .base_environment import BaseEnvironment


class AgentSafetyBenchEnvironment(BaseEnvironment):
    env_type = "agent_safetybench"

    def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
        return EnvironmentFeedbackBundle()
