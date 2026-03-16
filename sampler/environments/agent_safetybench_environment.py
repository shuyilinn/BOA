from __future__ import annotations

import importlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from boa_types.interaction import EnvironmentFeedback, EnvironmentFeedbackBundle, EnvironmentRequestBundle
from boa_types.tree_node import NodeSource, TreeNode
from utils.logger import setup_logger

from .base_environment import BaseEnvironment

logger = setup_logger("AgentSafetyBenchEnvironment")

# Directory containing Agent-SafetyBench environment modules
_ENV_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark" / "agentsafetybench" / "environments"


class AgentSafetyBenchEnvironment(BaseEnvironment):
    env_type = "agent_safetybench"

    def __init__(self) -> None:
        # Cache of env instances keyed by (root_node_id, env_name) so that
        # state is shared across tool calls within the same tree path but
        # isolated between different root nodes / branches.
        self._env_cache: Dict[tuple, Any] = {}

    def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
        assistant_node = request.assistant_node

        # 1. Collect the full text of the current assistant turn (may span multiple nodes)
        llm_text = _collect_last_assistant_text(assistant_node)
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

        # 4. Format result as text and return as feedback
        result_text = _format_tool_result(llm_text, result)

        # Build structured messages mirroring eval.py lines 246-268:
        # - role: assistant + tool_calls  (the model's tool call intent)
        # - role: tool + result           (the tool execution result)
        tool_call_id = "call_0"
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
                    source=NodeSource.TOOL,
                    text=result_text,
                    role="tool",
                    metadata={"structured_messages": structured_messages},
                )
            ],
            terminal=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_last_assistant_text(node: TreeNode) -> str:
    """Walk up the tree collecting consecutive ASSISTANT nodes to get the full last turn."""
    texts: List[str] = []
    current: Optional[TreeNode] = node
    while current is not None and current.source == NodeSource.ASSISTANT:
        texts.append(current.text)
        current = current.parent
    result = "".join(reversed(texts))
    logger.info("_collect_last_assistant_text: collected %d node(s), total length=%d, result=%s", len(texts), len(result), result)
    return result


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


def _format_tool_result(assistant_text: str, result: Dict[str, Any]) -> str:
    """
    Wrap the tool result in the format the model expects.

    - If the assistant used Qwen-style <tool_call> tags, emit the full Qwen turn:
        <|im_start|>tool\\n<tool_response>...\\n</tool_response>\\n<|im_end|>\\n<|im_start|>assistant\\n
      The trailing <|im_start|>assistant\\n acts as the generation prompt so the
      model continues generating as the assistant.
    - Otherwise fall back to plain JSON (basic system-prompt format).
    """
    result_json = json.dumps(result, ensure_ascii=False)

    if "<tool_call>" in assistant_text:
        return (
            f"\n<|im_start|>tool\n"
            f"<tool_response>\n{result_json}\n</tool_response>\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )

    # Basic system-prompt format: the model just sees the raw result
    return f"\n{result_json}\n"
