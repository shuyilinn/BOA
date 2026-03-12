from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from boa_types.tree_node import NodeSource

if TYPE_CHECKING:
    from boa_types.tree_node import TreeNode


META_ENDS_WITH_EOS = "ends_with_eos"
META_AWAITING_ENVIRONMENT = "awaiting_environment"


@dataclass
class EnvironmentFeedback:
    source: NodeSource
    text: str
    token_ids: List[int]
    role: Optional[str] = None
    log_prob: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentFeedbackBundle:
    sequences: List[EnvironmentFeedback] = field(default_factory=list)
    terminal: bool = False
    next_actor: Optional[str] = None
    priority_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentRequestBundle:
    assistant_node: TreeNode
    env_type: str = "single_turn"
    metadata: Dict[str, Any] = field(default_factory=dict)
