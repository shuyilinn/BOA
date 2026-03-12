from __future__ import annotations

from abc import ABC, abstractmethod

from boa_types.interaction import EnvironmentFeedbackBundle, EnvironmentRequestBundle


class BaseEnvironment(ABC):
    env_type = ""

    @abstractmethod
    def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
        raise NotImplementedError
