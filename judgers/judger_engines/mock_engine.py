from __future__ import annotations

import random
import time
from typing import List

from .base_engine import JudgerEngineBase


class MockJudgerEngine(JudgerEngineBase):
    """Deterministic judger engine for CPU-only tests."""

    def __init__(self, name: str, cuda_number, **kwargs):
        self.name = name
        self.mock_refused = bool(kwargs.get("mock_refused", False))
        self.mock_score = float(kwargs.get("mock_score", 8.0))
        self.mock_score_mode = str(kwargs.get("mock_score_mode", "fixed") or "fixed").lower()
        self.mock_score_min = int(kwargs.get("mock_score_min", 1))
        self.mock_score_max = int(kwargs.get("mock_score_max", 10))
        self.mock_high_score_prob = float(kwargs.get("mock_high_score_prob", 0.0))
        self.mock_high_score_min = int(kwargs.get("mock_high_score_min", 8))
        self.mock_high_score_max = int(kwargs.get("mock_high_score_max", 10))
        if self.mock_score_min > self.mock_score_max:
            self.mock_score_min, self.mock_score_max = self.mock_score_max, self.mock_score_min
        if self.mock_high_score_min > self.mock_high_score_max:
            self.mock_high_score_min, self.mock_high_score_max = self.mock_high_score_max, self.mock_high_score_min
        self.mock_high_score_prob = min(1.0, max(0.0, self.mock_high_score_prob))
        self._rng = random.Random(int(kwargs.get("mock_score_seed", 2026)))
        self.mock_judger_sleep_sec = max(0.0, float(kwargs.get("mock_judger_sleep_sec", 0.0) or 0.0))

    def _is_refusal_prompt(self, text: str) -> bool:
        return "1_refused" in text and "0_not_refused" in text

    def _is_score_prompt(self, text: str) -> bool:
        return "ONLY type a single integer score from 1 to 10" in text or "Score:" in text

    def batch_generate_content(self, texts: List[str]) -> List[str]:
        if self.mock_judger_sleep_sec > 0.0:
            time.sleep(self.mock_judger_sleep_sec)
        outputs: List[str] = []
        for text in texts:
            if self._is_refusal_prompt(text):
                label = "1_refused" if self.mock_refused else "0_not_refused"
                outputs.append(f"Mock refusal classification.\n{label}")
                continue
            if self._is_score_prompt(text):
                if self.mock_score_mode == "random":
                    use_high = self._rng.random() < self.mock_high_score_prob
                    if use_high:
                        score_int = self._rng.randint(self.mock_high_score_min, self.mock_high_score_max)
                    else:
                        score_int = self._rng.randint(self.mock_score_min, self.mock_score_max)
                else:
                    score_int = max(1, min(10, int(round(self.mock_score))))
                outputs.append(f"Mock nuanced scoring.\nScore: {score_int}")
                continue
            outputs.append("Mock default output.\n0_not_refused")
        return outputs
