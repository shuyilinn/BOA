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
    chunk_width: int = 20
    test_mode: bool = False
    use_mock_engines: bool = False


# Agent experiment defaults.
TARGET_ENGINE_NAME = "hf"
JUDGER_ENGINE_NAME = "vllm"
TARGET_MODEL_CUDA_NUMBER = 0
JUDGER_CUDA_NUMBER = 0

AGENT_CONFIGS: List[SweepConfig] = [
    SweepConfig(
        workload_name="single_turn",
        target_model="meta-llama/Llama-3.2-1B-Instruct",
        top_p=0.9,
        top_k=-1,
        temperature=0.6,
        likelihood=0.0001,
        harmful_prompt_start=1,
        harmful_prompt_end=20,
        chunk_size=1,
        chunk_width=20,
        test_mode=False,
        use_mock_engines=False,
    )
]


def _build_cmd(cfg: SweepConfig, root_dir: Path) -> List[str]:
    target_engine = "mock" if cfg.use_mock_engines else TARGET_ENGINE_NAME
    judger_engine = "mock" if cfg.use_mock_engines else JUDGER_ENGINE_NAME
    command = [
        sys.executable,
        str(root_dir / "run.py"),
        "--workload_name",
        cfg.workload_name,
        "--target_model",
        cfg.target_model,
        "--target_engine_name",
        target_engine,
        "--judger_engine_name",
        judger_engine,
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
        "60",
    ]
    if cfg.test_mode:
        command.extend(["--test_mode", "true"])
    return command


def _task_name(cfg: SweepConfig, task_id: int) -> str:
    model_short = cfg.target_model.replace("/", "_")
    mode_suffix = "_test" if cfg.test_mode else ""
    engine_suffix = "_mock" if cfg.use_mock_engines else ""
    batch_suffix = "_dynbon" if cfg.use_dynamic_batch_size else "_dynboff"
    return (
        f"{task_id:03d}_{cfg.workload_name}_{model_short}"
        f"_p{cfg.top_p}_k{cfg.top_k}_t{cfg.temperature}"
        f"_lh{cfg.likelihood}_h{cfg.harmful_prompt_start}-{cfg.harmful_prompt_end}"
        f"_cs{cfg.chunk_size}_cw{cfg.chunk_width}_log{cfg.logger_mode}{batch_suffix}"
        f"{mode_suffix}{engine_suffix}"
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
    logger.info("Total agent tasks: %s", len(AGENT_CONFIGS))

    for i, cfg in enumerate(AGENT_CONFIGS, start=1):
        run_task(i, cfg, root, logs_dir, run_timestamp)
        time.sleep(1)

    logger.info("All agent tasks done.")
