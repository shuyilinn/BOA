"""
Run Template (Agent) — copy this file, edit CONFIG, run it.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — only edit here
# ═══════════════════════════════════════════════════════════════════════════════

SHARED = dict(
    target_engine_name="hf",
    judger_engine_name="vllm",
    target_model_cuda_number=0,
    judger_cuda_number=1,
    target_gpu_memory_utilization=0.9,
    judger_gpu_memory_utilization=0.9,
    use_dynamic_batch_size="true",
    logger_mode="info",
)

TASKS = [
    # dict(
    #     workload_name="agent",
    #     target_model="meta-llama/Llama-3.1-8B-Instruct",
    #     benchmark_path="./benchmark/agentsafetybench/llama3.1-8b_greedy_safe_subset.json",
    #     top_p=0.9,
    #     top_k=-1,
    #     temperature=0.6,
    #     likelihood=0.0001,
    #     harmful_prompt_start=1,
    #     harmful_prompt_end=-1,
    #     chunk_size=1,
    #     chunk_width=100,
    #     time_limit_sec=600,
    # ),
        dict(
        workload_name="agent",
        target_model="Qwen/Qwen2.5-14B-Instruct",
        benchmark_path="./benchmark/agentsafetybench/qwen2.5-14b_greedy_safe_subset.json",
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        likelihood=0.0001,
        harmful_prompt_start=1,
        harmful_prompt_end=-1,
        chunk_size=1,
        chunk_width=100,
        time_limit_sec=600,
    ),
]

WAIT_PID: Optional[int] = None

# ═══════════════════════════════════════════════════════════════════════════════
# Runner — no need to edit below
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cmd(task: dict, root_dir: Path) -> list[str]:
    merged = {**SHARED, **task}
    prompt_indices = merged.pop("prompt_indices", None)
    test_mode = merged.pop("test_mode", False)

    cmd = [sys.executable, str(root_dir / "run.py")]
    for k, v in merged.items():
        cmd.extend([f"--{k}", str(v)])
    if prompt_indices:
        cmd.extend(["--prompt_indices", ",".join(str(i) for i in prompt_indices)])
    if test_mode:
        cmd.extend(["--test_mode", "true"])
    return cmd


def run_task(task_id: int, task: dict, root_dir: Path, logs_dir: Path, run_ts: str) -> int:
    command = _build_cmd(task, root_dir)
    model_short = task["target_model"].replace("/", "_")
    log_path = logs_dir / f"{run_ts}_{task_id:03d}_{model_short}.log"

    logger.info("[START] Task %d: %s", task_id, task["target_model"])
    logger.info("[CMD] %s", shlex.join(command))
    with log_path.open("a", encoding="utf-8", buffering=1) as lf:
        lf.write(f"[{datetime.utcnow().isoformat()}Z] {shlex.join(command)}\n")
        rc = subprocess.Popen(command, cwd=root_dir, stdout=lf, stderr=lf, text=True).wait()
        lf.write(f"[{datetime.utcnow().isoformat()}Z] exit_code={rc}\n")

    if rc != 0:
        logger.warning("Task %d FAILED (exit=%d) log=%s", task_id, rc, log_path)
    else:
        logger.info("Task %d done", task_id)
    return rc


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "experiment" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if WAIT_PID is not None:
        logger.info("Waiting pid %s ...", WAIT_PID)
        while True:
            try:
                os.kill(WAIT_PID, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(5)

    logger.info("Tasks: %d", len(TASKS))
    failed = 0
    for i, task in enumerate(TASKS, 1):
        if run_task(i, task, root, logs_dir, run_ts) != 0:
            failed += 1
        time.sleep(1)

    logger.info("Done. %d/%d succeeded.", len(TASKS) - failed, len(TASKS))
