from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@dataclass(frozen=True)
class SweepConfig:
    workload_name: str
    target_model: str
    top_p: float
    top_k: int
    temperature: float
    likelihood: float
    harmful_prompt_start: int
    harmful_prompt_end: int
    logger_mode: str = "warning"
    use_dynamic_batch_size: bool = True
    chunk_size: int = 1
    chunk_width: int = 100


# Lightweight real-model run.
TARGET_ENGINE_NAME = "hf"
JUDGER_ENGINE_NAME = "vllm"
TARGET_MODEL_CUDA_NUMBER = 0
JUDGER_CUDA_NUMBER = 0

LIGHT_CONFIGS: List[SweepConfig] = [
    SweepConfig(
        workload_name="single_turn",
        target_model="meta-llama/Llama-3.2-1B-Instruct",
        top_p=0.9,
        top_k=-1,
        temperature=0.6,
        likelihood=0.0001,
        harmful_prompt_start=1,
        harmful_prompt_end=1,
        chunk_size=1,
        chunk_width=20,
    )
]


def _build_cmd(cfg: SweepConfig, root_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(root_dir / "run.py"),
        "--workload_name",
        cfg.workload_name,
        "--target_model",
        cfg.target_model,
        "--target_engine_name",
        TARGET_ENGINE_NAME,
        "--judger_engine_name",
        JUDGER_ENGINE_NAME,
        "--top_p",
        str(cfg.top_p),
        "--top_k",
        str(cfg.top_k),
        "--temperature",
        str(cfg.temperature),
        "--likelihood",
        str(cfg.likelihood),
        "--logger_mode",
        cfg.logger_mode,
        "--target_model_cuda_number",
        str(TARGET_MODEL_CUDA_NUMBER),
        "--judger_cuda_number",
        str(JUDGER_CUDA_NUMBER),
        "--use_dynamic_batch_size",
        str(cfg.use_dynamic_batch_size).lower(),
        "--judger_gpu_memory_utilization",
        "0.6",
        "--benign_n",
        "0",
        "--layer3_filter_threshold",
        "100000",
        "--harmful_prompt_start",
        str(cfg.harmful_prompt_start),
        "--harmful_prompt_end",
        str(cfg.harmful_prompt_end),
        "--chunk_size",
        str(cfg.chunk_size),
        "--chunk_width",
        str(cfg.chunk_width),
        "--time_limit_sec",
        "30",
    ]


def _task_name(cfg: SweepConfig, task_id: int) -> str:
    model_short = cfg.target_model.replace("/", "_")
    batch_suffix = "_dynbon" if cfg.use_dynamic_batch_size else "_dynboff"
    return (
        f"{task_id:03d}_{cfg.workload_name}_{model_short}_light"
        f"_p{cfg.top_p}_k{cfg.top_k}_t{cfg.temperature}"
        f"_lh{cfg.likelihood}_h{cfg.harmful_prompt_start}-{cfg.harmful_prompt_end}"
        f"_cs{cfg.chunk_size}_cw{cfg.chunk_width}_log{cfg.logger_mode}{batch_suffix}"
    )


def run_task(task_id: int, cfg: SweepConfig, root_dir: Path, logs_dir: Path, run_timestamp: str) -> int:
    task_name = _task_name(cfg, task_id)
    log_path = logs_dir / f"{run_timestamp}_{task_name}.log"
    command = _build_cmd(cfg, root_dir)
    cmd_str = shlex.join(command)

    logger.info("[START] Task %s (%s)", task_id, task_name)
    logger.info("[COMMAND] %s", cmd_str)

    with log_path.open("a", encoding="utf-8", buffering=1) as lf:
        lf.write("\n" + "=" * 80 + "\n")
        lf.write(f"[{datetime.utcnow().isoformat()}Z] [START] Task {task_id} ({task_name})\n")
        lf.write(f"[COMMAND] {cmd_str}\n")
        lf.write("=" * 80 + "\n")

        proc = subprocess.Popen(
            command,
            cwd=root_dir,
            stdout=lf,
            stderr=lf,
            text=True,
        )
        return_code = proc.wait()

        lf.write(
            f"[{datetime.utcnow().isoformat()}Z] [END] Task {task_id} ({task_name}) exit_code={return_code}\n"
        )
        lf.write("=" * 80 + "\n")

    if return_code != 0:
        logger.warning("Task %s failed, exit_code=%s, log=%s", task_id, return_code, log_path)
    else:
        logger.info("Task %s completed", task_id)
    return int(return_code)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "experiment" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Total light-model tasks: %s", len(LIGHT_CONFIGS))

    failed = False
    for i, cfg in enumerate(LIGHT_CONFIGS, start=1):
        code = run_task(i, cfg, root, logs_dir, run_timestamp)
        if code != 0:
            failed = True
        time.sleep(1)

    if failed:
        logger.error("=" * 72)
        logger.error("%sLIGHTWEIGHT REAL-MODEL TEST: FAILED%s", RED, RESET)
        logger.error("%sStatus: light mode failed%s", RED, RESET)
        logger.error("=" * 72)
        raise SystemExit(1)

    logger.info("=" * 72)
    logger.info("%sLIGHTWEIGHT REAL-MODEL TEST: PASSED%s", GREEN, RESET)
    logger.info("%sStatus: light mode passed%s", GREEN, RESET)
    logger.info("=" * 72)
