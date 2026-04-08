from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionType(Enum):
    QUEUE_SAMPLE = "queue_sample"
    QUEUE_JUDGE = "queue_judge"
    ADVANCE_TURN = "advance_turn"
    SUCCESS = "success"
    DROP = "drop"
    COMPLETE = "complete"


@dataclass
class WorkItem:
    """Unit payload routed across runtime queues."""
    path_id: str
    path_ids: List[int]
    path_text: str
    node: Any
    trace_id: str
    turn_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Stage output consumed by the single writer."""
    decision_type: DecisionType
    node: Optional[Any] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metrics_delta: Dict[str, Any] = field(default_factory=dict)


class StateStore:
    """
    Single-writer state applier for runtime events.

    M1 skeleton: collect metrics and expose one apply entrypoint.
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}

    def apply(self, decision: Decision) -> None:
        if not decision.metrics_delta:
            return
        for key, value in decision.metrics_delta.items():
            if isinstance(value, (int, float)):
                self.metrics[key] = self.metrics.get(key, 0) + value
            else:
                self.metrics[key] = value
