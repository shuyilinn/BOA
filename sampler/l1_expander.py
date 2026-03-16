from __future__ import annotations

from typing import Any

from boa_types.interaction import EnvironmentRequestBundle
from boa_types.tree_node import NodeSource, TreeNode
from .environments import AgentSafetyBenchEnvironment, SingleTurnEnvironment


class L1Expander:
    """
    Interaction hub.
    It routes environment requests, runs the environment, and attaches returned sequences to the tree.
    """

    def __init__(self, engine: Any, config: Any):
        self.engine = engine
        self.config = config
        self.environment_registry = self._build_environment_registry(engine, config)

    def expand_after_eos(self, request_bundle: EnvironmentRequestBundle) -> TreeNode:
        environment = self.environment_registry[request_bundle.env_type]
        feedback_bundle = environment.run(request_bundle)

        assistant_node = request_bundle.assistant_node
        if not feedback_bundle.sequences:
            if feedback_bundle.terminal:
                assistant_node.metadata["should_complete"] = True
            return assistant_node

        tokenizer = self.engine.get_tokenizer()
        current_node = assistant_node
        for sequence in feedback_bundle.sequences:
            token_ids = sequence.token_ids or tokenizer.encode(sequence.text)
            role = sequence.role or self._role_from_source(sequence.source)
            child = current_node.add_child(
                token_ids=token_ids,
                text=sequence.text,
                log_prob=float(sequence.log_prob),
                interaction_role=role,
                interaction_source=sequence.source.value,
                **sequence.metadata,
            )
            child.source = sequence.source
            if feedback_bundle.priority_hint is not None:
                child.metadata["interaction_priority_hint"] = feedback_bundle.priority_hint
            current_node = child
        if feedback_bundle.terminal:
            current_node.metadata["should_complete"] = True
        return current_node

    def _role_from_source(self, source: NodeSource) -> str:
        if source == NodeSource.TOOL:
            return "tool"
        if source == NodeSource.ENV:
            return "user"
        if source == NodeSource.ASSISTANT:
            return "assistant"
        return "user"

    def _build_environment_registry(self, engine: Any, config: Any) -> dict[str, Any]:
        return {
            AgentSafetyBenchEnvironment.env_type: AgentSafetyBenchEnvironment(),
            SingleTurnEnvironment.env_type: SingleTurnEnvironment(),
        }
