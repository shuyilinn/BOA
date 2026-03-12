from __future__ import annotations

import logging
import os
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
    test_mode: bool = False
    use_mock_engines: bool = False
    chunk_size: int = 1
    chunk_width: int = 100


# Fixed GPU assignments
TARGET_MODEL_CUDA_NUMBER = 2
JUDGER_CUDA_NUMBER = 2
TARGET_ENGINE_NAME = "hf"
JUDGER_ENGINE_NAME = "vllm"

# Profiling settings
# Set to False to disable cProfile; adjust the output directory as needed.
ENABLE_CPROFILE = False
CPROFILE_OUTPUT_DIR = "profiles"

# Wait for a PID to exit before starting; set to None to skip waiting.
WAIT_PID = 1958772
WAIT_INTERVAL_SEC = 5

MODEL_CONFIGS: List[SweepConfig] = [
    # SweepConfig("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",   0.9,  -1, 0.6,0.0001, 1, 128),
    # SweepConfig("Qwen/Qwen2.5-72B-Instruct-AWQ",        0.95, 20, 0.6, 0.0001, 1, 128),
    # SweepConfig("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 0.9, -1, 0.6, 0.0001, 1, 128, chunk_size=1, chunk_width=20),
    # SweepConfig("Qwen/Qwen2.5-32B-Instruct-AWQ",        0.95, 20, 0.6, 0.0001, 1, 128),
    SweepConfig("single_turn", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, -1, chunk_size=1, chunk_width=20),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, 128, chunk_size=5, chunk_width=20),
    # SweepConfig("Qwen/Qwen3-8B", 0.95, 20, 0.6, 0.0001, 1, 1),
    # SweepConfig("Qwen/Qwen3-8B", 0.95, 20, 0.6, 0.0001, 1, 1, test_mode=True, use_mock_engines=True),
    # SweepConfig("lmsys/vicuna-7b-v1.5", 0.6, -1, 0.9, 0.0001, 1, 128, chunk_size=1, chunk_width=20),
    # SweepConfig("meta-llama/Llama-2-7b-chat-hf", 0.9, -1, 0.6, 1, 93, 128),
    # SweepConfig("meta-llama/Llama-2-7b-chat-hf", 0.9, -1, 0.6, 0.01, 46, 128),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 1.0, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.8, -1, 1.0, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 1.0, 5, 1.0, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 1.0, 10, 1.0, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 1.0, 20, 1.0, 0.0001, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 1, 1, 1),
    # SweepConfig("meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.01, 1, 1),
    # SweepConfig("Qwen/Qwen3-8B", 0.95, 20, 0.6, 0.01, 1, 1),
    # SweepConfig("Qwen/Qwen3-8B", 0.95, 20, 0.6, 0.0001, 1, 128, chunk_size=5, chunk_width=20),
    # SweepConfig("google/gemma-3-4b-it", 0.95, 64, 1.0, 0.0001, 1, 1),
    # SweepConfig("LLM-LAT/robust-llama3-8b-instruct", 0.9, -1, 0.6, 0.0001, 1, 1),
    # SweepConfig("LLM-LAT/robust-llama3-8b-instruct", 1.0, 5, 1.0, 0.0001, 1, 1),
]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_pid_exit(pid: int, interval_sec: int) -> None:
    logger.info("Waiting pid %s to exit...", pid)
    while _pid_alive(pid):
        time.sleep(interval_sec)
    logger.info("Pid %s exited.", pid)


def _build_cmd(cfg: SweepConfig, root_dir: Path, task_name: str) -> List[str]:
    target_engine = "mock" if cfg.use_mock_engines else TARGET_ENGINE_NAME
    judger_engine = "mock" if cfg.use_mock_engines else JUDGER_ENGINE_NAME
    command = [sys.executable]
    if ENABLE_CPROFILE:
        profile_path = root_dir / "experiment" / CPROFILE_OUTPUT_DIR / f"{task_name}.prof"
        command.extend(
            [
                "-m",
                "cProfile",
                "-o",
                str(profile_path),
            ]
        )

    command.extend(
        [
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
        "--harmful_prompt_start",
        str(cfg.harmful_prompt_start),
        "--harmful_prompt_end",
        str(cfg.harmful_prompt_end),
        "--chunk_size",
        str(cfg.chunk_size),
        "--chunk_width",
        str(cfg.chunk_width),
        ]
    )
    if cfg.test_mode:
        command.extend(["--test_mode", "true"])
    return command


def _task_name(cfg: SweepConfig, task_id: int) -> str:
    model_short = cfg.target_model.replace("/", "_")
    mode_suffix = "_test" if cfg.test_mode else ""
    engine_suffix = "_mock" if cfg.use_mock_engines else ""
    batch_suffix = "_dynbon" if cfg.use_dynamic_batch_size else "_dynboff"
    return (
        f"{task_id:03d}_{cfg.workload_name}_{model_short}_p{cfg.top_p}_k{cfg.top_k}_t{cfg.temperature}"
        f"_lh{cfg.likelihood}_h{cfg.harmful_prompt_start}-{cfg.harmful_prompt_end}"
        f"_cs{cfg.chunk_size}_cw{cfg.chunk_width}_log{cfg.logger_mode}{batch_suffix}"
        f"{mode_suffix}{engine_suffix}"
    )


def run_task(task_id: int, cfg: SweepConfig, root_dir: Path, logs_dir: Path, run_timestamp: str) -> int:
    task_name = _task_name(cfg, task_id)
    log_path = logs_dir / f"{run_timestamp}_{task_name}.log"
    command = _build_cmd(cfg, root_dir, task_name)
    cmd_str = shlex.join(command)
    start_banner = f"[{datetime.utcnow().isoformat()}Z] [START] Task {task_id} ({task_name})"

    logger.info(start_banner)
    logger.info("[COMMAND] %s", cmd_str)
    with log_path.open("a", encoding="utf-8", buffering=1) as lf:
        lf.write("\n" + "=" * 80 + "\n")
        lf.write(start_banner + "\n")
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
        end_banner = f"[{datetime.utcnow().isoformat()}Z] [END] Task {task_id} ({task_name}) exit_code={return_code}"
        lf.write(end_banner + "\n")
        lf.write("=" * 80 + "\n")

    if return_code != 0:
        logger.warning("Task %s failed, exit_code=%s, log=%s", task_id, return_code, log_path)
    else:
        logger.info("Task %s completed", task_id)
    return int(return_code)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "experiment" / "logs"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_CPROFILE:
        profiles_dir = root / "experiment" / CPROFILE_OUTPUT_DIR
        profiles_dir.mkdir(parents=True, exist_ok=True)

    if WAIT_PID is not None:
        _wait_pid_exit(int(WAIT_PID), int(WAIT_INTERVAL_SEC))

    logger.info("Total tasks: %s", len(MODEL_CONFIGS))
    logger.info("-" * 80)

    for i, cfg in enumerate(MODEL_CONFIGS, start=1):
        run_task(i, cfg, root, logs_dir, run_timestamp)
        time.sleep(1)

    logger.info("-" * 80)
    logger.info("All tasks done (sequential worker).")
