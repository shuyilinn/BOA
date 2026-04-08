"""Minimum reproducibility test — mock engines, no model download needed."""
from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — only edit here
# ═══════════════════════════════════════════════════════════════════════════════

SHARED = dict(
    target_engine_name="mock",
    judger_engine_name="mock",
    target_model_cuda_number=0,
    judger_cuda_number=0,
    use_dynamic_batch_size="false",
    logger_mode="warning",
    mock_sampler_sleep_sec=0,
    mock_judger_sleep_sec=0,
    test_mode="true",
)

TASKS = [
    dict(
        workload_name="single_turn",
        target_model="mock-target",
        top_p=0.95,
        top_k=20,
        temperature=0.6,
        likelihood=0.0001,
        harmful_prompt_start=1,
        harmful_prompt_end=1,
        chunk_size=1,
        chunk_width=20,
        time_limit_sec=2,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Runner — no need to edit below
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cmd(task: dict, root_dir: Path) -> list[str]:
    merged = {**SHARED, **task}
    cmd = [sys.executable, str(root_dir / "run.py")]
    for k, v in merged.items():
        cmd.extend([f"--{k}", str(v)])
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

    logger.info("Tasks: %d", len(TASKS))
    failed = False
    for i, task in enumerate(TASKS, 1):
        if run_task(i, task, root, logs_dir, run_ts) != 0:
            failed = True
        time.sleep(1)

    if failed:
        logger.error("=" * 72)
        logger.error("%sMINIMUM REPRODUCIBILITY TEST: FAILED%s", RED, RESET)
        logger.error("=" * 72)
        raise SystemExit(1)

    logger.info("=" * 72)
    logger.info("%sMINIMUM REPRODUCIBILITY TEST: PASSED%s", GREEN, RESET)
    logger.info("=" * 72)
