from __future__ import annotations

import random
from typing import List

from .base_engine import JudgerEngineBase


class MockApiJudgerEngine(JudgerEngineBase):
    """Mock API judger that probabilistically returns compliant/violating labels."""

    def __init__(self, name: str, cuda_number, **kwargs):
        self.name = name
        prob = float(kwargs.get("mock_api_unsafe_prob", 0.01))
        self.mock_api_unsafe_prob = min(1.0, max(0.0, prob))
        seed = kwargs.get("mock_api_seed", None)
        self._rng = random.Random(seed)

    def batch_generate_content(self, texts: List[str]) -> List[str]:
        outputs: List[str] = []
        for _ in texts:
            is_unsafe = self._rng.random() < self.mock_api_unsafe_prob
            if is_unsafe:
                outputs.append("Mock API decision.\n1_violating")
            else:
                outputs.append("Mock API decision.\n0_compliant")
        return outputs
