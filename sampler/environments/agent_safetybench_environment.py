from __future__ import annotations

import importlib
import json
import random
import re
import string
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from boa_types.interaction import EnvironmentFeedback, EnvironmentFeedbackBundle, EnvironmentRequestBundle
from boa_types.tree_node import NodeRole
from utils.logger import setup_logger

from .base_environment import BaseEnvironment

logger = setup_logger("AgentSafetyBenchEnvironment")

# Directory containing Agent-SafetyBench environment modules
_ENV_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark" / "agentsafetybench" / "environments"


class AgentSafetyBenchEnvironment(BaseEnvironment):
    env_type = "agent_safetybench"

    def __init__(self, tokenizer: Any = None) -> None:
        self._tokenizer = tokenizer
        # Cache of env instances keyed by (root_node_id, env_name) so that
        # state is shared across tool calls within the same tree path but
        # isolated between different root nodes / branches.
        self._env_cache: Dict[tuple, Any] = {}

    def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
        assistant_node = request.assistant_node

        # 1. Collect the full text of the current assistant turn from canonical state
        llm_text = assistant_node.conversation_state.latest_assistant_text()
        logger.debug("Last assistant text: %.200s", llm_text)

        # 2. Parse tool call from the LLM output
        tool_call = _parse_tool_call(llm_text)
        if tool_call is None:
            logger.info("[ENV] No tool call found; terminal. text=%.200s", llm_text)
            return EnvironmentFeedbackBundle(terminal=True)

        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments") or tool_call.get("parameters") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        logger.info("[ENV] >>> Tool call: %s | args: %s", tool_name, json.dumps(arguments, ensure_ascii=False)[:300])

        # 3. Execute the tool against the appropriate environment
        environments_config: List[Dict[str, Any]] = assistant_node.prompt_metadata.get("environments", [])
        root = assistant_node
        while root.parent is not None:
            root = root.parent
        result = _execute_tool(environments_config, tool_name, deepcopy(arguments), self._env_cache, id(root))
        logger.info("[ENV] <<< Result: success=%s | %s", result.get("success"), json.dumps(result, ensure_ascii=False)[:300])

        # 4. Produce a model-agnostic text representation of the tool result.
        # The actual model-specific rendering is handled by the conversation_formatter
        # layer (apply_chat_template on the structured messages).  Using neutral JSON
        # keeps node.text clean for debug/logging.
        result_text = json.dumps(result, ensure_ascii=False)

        # Build structured messages mirroring eval.py lines 246-268:
        # - role: assistant + tool_calls  (the model's tool call intent)
        # - role: tool + result           (the tool execution result)
        # Match Agent-SafetyBench Llama3API.parse_tool_str() random id generation
        tool_call_id = ''.join(random.sample(string.ascii_letters + string.digits, 9))
        structured_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "tool_call_id": tool_call_id,
                "name": tool_name,
            },
        ]

        return EnvironmentFeedbackBundle(
            sequences=[
                EnvironmentFeedback(
                    role=NodeRole.TOOL,
                    text=result_text,
                    metadata={"structured_messages": structured_messages},
                )
            ],
            terminal=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from the LLM output, supporting multiple formats."""
    text = text.strip()

    # Qwen / HuggingFace native function-calling format: <tool_call>...</tool_call>
    tc_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if tc_match:
        try:
            parsed = json.loads(tc_match.group(1))
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Basic system-prompt format: bare JSON {"name": ..., "arguments": ...}
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _execute_tool(
    environments_config: List[Dict[str, Any]],
    tool_name: str,
    arguments: Dict[str, Any],
    env_cache: Dict[tuple, Any],
    root_id: int,
) -> Dict[str, Any]:
    """
    Find the environment that owns this tool and call it.

    env_cache is keyed by (root_id, env_name) so the same env instance is
    reused across consecutive tool calls within a tree path — matching
    Agent-SafetyBench's stateful behaviour where one env instance lives for
    the entire task.
    """
    env_dir_str = str(_ENV_DIR)
    if env_dir_str not in sys.path:
        sys.path.insert(0, env_dir_str)

    for env_config in environments_config:
        env_name = env_config.get("name", "")
        allowed_tools = env_config.get("tools", [])
        parameters = env_config.get("parameters", {})

        if tool_name not in allowed_tools:
            continue

        cache_key = (root_id, env_name)
        env_instance = env_cache.get(cache_key)
        if env_instance is None:
            try:
                env_module = importlib.import_module(env_name)
                env_class = getattr(env_module, env_name)
                env_instance = env_class(parameters=deepcopy(parameters))
                env_cache[cache_key] = env_instance
            except Exception as exc:
                logger.warning("Error instantiating %s: %s", env_name, exc)
                return {"success": False, "message": f"Execution error: {exc}"}

        try:
            return env_instance.call_tool(tool_name, arguments)
        except Exception as exc:
            logger.warning("Error calling %s in %s: %s", tool_name, env_name, exc)
            return {"success": False, "message": f"Execution error: {exc}"}

    return {"success": False, "message": f"Tool '{tool_name}' not found in any environment."}
