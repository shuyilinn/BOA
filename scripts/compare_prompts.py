"""
Compare BOA prompt generation with Agent-SafetyBench prompt generation.

Usage:
    python scripts/compare_prompts.py [--sample_ids 0,19,100] [--model llama3.1-8b]

Prints the rendered prompt (after apply_chat_template) from both BOA and
Agent-SafetyBench side by side for comparison.
"""

import json
import sys
import os
from argparse import ArgumentParser
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--sample_ids", type=str, default=None,
                    help="Comma-separated sample ids to compare. Default: first with-tool + first no-tool")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--data", type=str, default="/home/shuyi/Agent-SafetyBench/data/released_data.json")
args = parser.parse_args()

# ── Load tokenizer ────────────────────────────────────────────────────────
from transformers import AutoTokenizer
print(f"Loading tokenizer: {args.model} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# ── Load raw data ─────────────────────────────────────────────────────────
with open(args.data, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ── Pick samples ──────────────────────────────────────────────────────────
if args.sample_ids:
    target_ids = [int(x) for x in args.sample_ids.split(",")]
    samples = [d for d in raw_data if d["id"] in target_ids]
else:
    # auto-pick: first with-tool and first no-tool
    with_tool = next((d for d in raw_data if d.get("environments") and d["environments"][0].get("name")), None)
    no_tool = next((d for d in raw_data if not d.get("environments") or not d["environments"][0].get("name")), None)
    samples = [s for s in [with_tool, no_tool] if s is not None]

print(f"Comparing {len(samples)} sample(s): {[s['id'] for s in samples]}\n")

# ══════════════════════════════════════════════════════════════════════════
# Agent-SafetyBench side: replicate eval.py logic
# ══════════════════════════════════════════════════════════════════════════
asb_env_dir = Path("/home/shuyi/Agent-SafetyBench/environments")
sys.path.insert(0, str(asb_env_dir))
sys.path.insert(0, str(Path("/home/shuyi/Agent-SafetyBench/evaluation/model_api")))

from BaseAPI import BaseAPI

base_api = BaseAPI()

# Replicate Llama3API system prompt selection
def asb_get_system_prompt(d):
    if not d['environments'] or 'name' not in d['environments'][0] or d['environments'][0]['name'] == '':
        return "You are a helpful assistant"
    return base_api.basic_sys_prompt


def asb_load_tool_descs(d):
    """Replicate parse_envs + tool_desc wrapping from eval.py"""
    tool_descs = []
    for env_info in d.get("environments", []):
        env_name = env_info.get("name", "")
        if not env_name:
            continue
        tool_names = env_info.get("tools", [])
        # Load from JSON file (same as BaseEnv.__init__)
        json_path = asb_env_dir / f"{env_name}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            all_tools = json.load(f)
        for tool in all_tools:
            if tool["name"] in tool_names:
                tool_descs.append(tool)

    for item in tool_descs:
        item['type'] = "object"
    for i, item in enumerate(tool_descs):
        tool_descs[i] = {'type': 'function', 'function': item}
    return tool_descs


def asb_render(d):
    """Replicate Agent-SafetyBench eval.py + Llama3API.response() prompt rendering."""
    messages = [{"role": "system", "content": asb_get_system_prompt(d)}]
    if "dialog" in d:
        messages.extend(d["dialog"])
    else:
        messages.append({"role": "user", "content": d["instruction"]})

    tool_descs = asb_load_tool_descs(d)

    if tool_descs:
        rendered = tokenizer.apply_chat_template(
            messages, tools=tool_descs, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return messages, tool_descs, rendered


# ══════════════════════════════════════════════════════════════════════════
# BOA side: use agent_data_loader + conversation_formatter
# ══════════════════════════════════════════════════════════════════════════
boa_root = Path("/home/shuyi/BOA")
sys.path.insert(0, str(boa_root))

from benchmark.agent_data_loader import adapt_agent_safetybench_samples
from utils.conversation_formatter import parse_initial_prompt, render_conversation_state


def boa_render(d, all_raw_data):
    """Replicate BOA's prompt loading + rendering pipeline."""
    # adapt just this one sample
    result = adapt_agent_safetybench_samples(
        [d],
        env_dir=str(asb_env_dir),
        baseapi_path="/home/shuyi/Agent-SafetyBench/evaluation/model_api/BaseAPI.py",
        sys_prompt_variant="basic",
    )
    sample = result["samples"][0]

    # parse_initial_prompt (same as executor.py)
    conv_state, parsed_tools = parse_initial_prompt(sample["prompt"], args.model)

    # render (same as executor.py)
    rendered, _ = render_conversation_state(
        conv_state, tokenizer, model_name=args.model,
        tools=sample.get("tools_openai"),
    )
    return sample["messages"], sample.get("tools_openai", []), rendered


# ══════════════════════════════════════════════════════════════════════════
# Compare
# ══════════════════════════════════════════════════════════════════════════
for d in samples:
    sid = d["id"]
    has_tools = bool(d.get("environments") and d["environments"][0].get("name"))
    print("=" * 80)
    print(f"Sample ID: {sid}  |  has_tools: {has_tools}")
    print("=" * 80)

    asb_msgs, asb_tools, asb_rendered = asb_render(d)
    boa_msgs, boa_tools, boa_rendered = boa_render(d, raw_data)

    # Compare messages
    msgs_match = asb_msgs == boa_msgs
    tools_match = asb_tools == boa_tools
    rendered_match = asb_rendered == boa_rendered

    print(f"\n  messages match: {msgs_match}")
    print(f"  tools match:    {tools_match}")
    print(f"  rendered match: {rendered_match}")

    if not msgs_match:
        print("\n--- Messages diff ---")
        print(f"  ASB sys_prompt[:80]: {asb_msgs[0]['content'][:80]}")
        print(f"  BOA sys_prompt[:80]: {boa_msgs[0]['content'][:80]}")

    if not rendered_match:
        # Find first difference
        for i, (a, b) in enumerate(zip(asb_rendered, boa_rendered)):
            if a != b:
                ctx = 40
                print(f"\n--- First diff at char {i} ---")
                print(f"  ASB: ...{repr(asb_rendered[max(0,i-ctx):i+ctx])}...")
                print(f"  BOA: ...{repr(boa_rendered[max(0,i-ctx):i+ctx])}...")
                break
        else:
            shorter = min(len(asb_rendered), len(boa_rendered))
            print(f"\n--- Length diff: ASB={len(asb_rendered)}, BOA={len(boa_rendered)} ---")
            print(f"  Extra tail: {repr(asb_rendered[shorter:shorter+80] or boa_rendered[shorter:shorter+80])}")

    print(f"\n{'─'*40} ASB rendered (first 500 chars) {'─'*40}")
    print(asb_rendered[:500])
    print(f"\n{'─'*40} BOA rendered (first 500 chars) {'─'*40}")
    print(boa_rendered[:500])
    print()
