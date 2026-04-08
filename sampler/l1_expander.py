from __future__ import annotations

from typing import Any

from boa_types.interaction import EnvironmentRequestBundle
from boa_types.tree_node import NodeRole, TreeNode
from .environments import AgentSafetyBenchEnvironment, SingleTurnEnvironment


class L1Expander:
    """
    Interaction hub.
    It routes environment requests, runs the environment, and attaches returned sequences to the tree.
    """

    def __init__(self, engine: Any, config: Any):
        self.engine = engine
        self.config = config
        self._tokenizer = engine.get_tokenizer()
        self.environment_registry = self._build_environment_registry(engine, config)

    def expand_after_eos(self, request_bundle: EnvironmentRequestBundle) -> TreeNode:
        environment = self.environment_registry[request_bundle.env_type]
        feedback_bundle = environment.run(request_bundle)

        assistant_node = request_bundle.assistant_node
        if not feedback_bundle.sequences:
            if feedback_bundle.terminal:
                assistant_node.metadata["should_complete"] = True
            return assistant_node

        current_node = assistant_node
        for sequence in feedback_bundle.sequences:
            token_ids = sequence.token_ids or self._tokenizer.encode(sequence.text)
            interaction_role = sequence.role.value
            child = current_node.add_child(
                token_ids=token_ids,
                text=sequence.text,
                log_prob=float(sequence.log_prob),
                interaction_role=interaction_role,
                **sequence.metadata,
            )
            child.role = sequence.role
            if feedback_bundle.priority_hint is not None:
                child.metadata["interaction_priority_hint"] = feedback_bundle.priority_hint

            if sequence.role == NodeRole.TOOL:
                conv_msgs = sequence.metadata.get("structured_messages")
                if conv_msgs:
                    new_state = child.conversation_state.clone()
                    new_state.pending = None
                    new_state.committed_messages.extend(conv_msgs)
                    child.conversation_state = new_state

            current_node = child
        if feedback_bundle.terminal:
            current_node.metadata["should_complete"] = True
        return current_node

    def _build_environment_registry(self, engine: Any, config: Any) -> dict[str, Any]:
        tokenizer = engine.get_tokenizer()
        return {
            AgentSafetyBenchEnvironment.env_type: AgentSafetyBenchEnvironment(tokenizer=tokenizer),
            SingleTurnEnvironment.env_type: SingleTurnEnvironment(),
        }
