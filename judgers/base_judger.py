from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class JudgerAction(str, Enum):
    STOP = "stop"
    CONTINUE = "continue"


class JudgerExecutionType(str, Enum):
    LOCAL = "local"
    API = "api"


@dataclass(frozen=True)
class JudgerSample:
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicJudgerResult:
    action: JudgerAction
    score: float
    is_safe: bool
    layer: int
    response: str
    raw_output: str = ""
    layer3_score: Optional[float] = None
    layer4_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "score": float(self.score),
            "is_safe": bool(self.is_safe),
            "layer": int(self.layer),
            "layer3_score": float(self.layer3_score) if self.layer3_score is not None else None,
            "layer4_score": float(self.layer4_score) if self.layer4_score is not None else None,
            "action": self.action.value,
            "metadata": dict(self.metadata),
        }


class BaseAtomicJudger(ABC):
    layer: int
    execution_type = JudgerExecutionType.LOCAL

    def judge(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> AtomicJudgerResult:
        return self.batch_judge([JudgerSample(prompt=prompt, response=response, metadata=metadata or {})])[0]

    @abstractmethod
    def batch_judge(self, samples: List[JudgerSample]) -> List[AtomicJudgerResult]:
        raise NotImplementedError


class PipelineJudger(Protocol):
    def get_batch_size(self) -> int:
        ...

    def evaluate(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def batch_evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def evaluate_full_response(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    def batch_evaluate_full_response(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def evaluate_attack_sampling(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    def batch_evaluate_attack_sampling(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def get_refusal_filter_stats(self) -> Dict[str, Any]:
        ...
