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
    time_limit_sec: int = 600
    # Search strategy
    search_strategy: str = "greedy"
    search_alpha: float = 0.3
    search_depth_decay: float = 0.97
    search_stagnation_window: int = 3
    search_stagnation_penalty: float = 0.2


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
WAIT_PID = None
WAIT_INTERVAL_SEC = 5

MODEL_CONFIGS: List[SweepConfig] = [
    # ── Search strategy comparison: greedy vs boa on agent workload, prompts 1-275 ──
    # Greedy baseline
    SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 12, 12,
                logger_mode="warning", use_dynamic_batch_size=False, chunk_size=1, chunk_width=20,
                search_strategy="greedy"),
    # BOA search strategy
    SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 12, 12,
                logger_mode="warning", use_dynamic_batch_size=False, chunk_size=1, chunk_width=20,
                search_strategy="boa", search_alpha=0.3, search_depth_decay=0.97,
                search_stagnation_window=3, search_stagnation_penalty=0.2),
]

# ── Previous configs (archived) ──────────────────────────────────────────────
# MODEL_CONFIGS_ARCHIVED: List[SweepConfig] = [
#     # chunk_size sweep: 1,2,3,4 on all prompts
#     SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, -1, logger_mode="warning", use_dynamic_batch_size=False, chunk_size=1, chunk_width=20),
#     SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, -1, logger_mode="warning", use_dynamic_batch_size=False, chunk_size=2, chunk_width=20),
#     SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, -1, logger_mode="warning", use_dynamic_batch_size=False, chunk_size=4, chunk_width=20),
#     SweepConfig("agent", "meta-llama/Llama-3.1-8B-Instruct", 0.9, -1, 0.6, 0.0001, 1, -1, logger_mode="warning", use_dynamic_batch_size=False, chunk_size=16, chunk_width=20),
# ]


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
        "--time_limit_sec",
        str(cfg.time_limit_sec),
        "--search_strategy",
        cfg.search_strategy,
        "--search_alpha",
        str(cfg.search_alpha),
        "--search_depth_decay",
        str(cfg.search_depth_decay),
        "--search_stagnation_window",
        str(cfg.search_stagnation_window),
        "--search_stagnation_penalty",
        str(cfg.search_stagnation_penalty),
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
        f"_cs{cfg.chunk_size}_cw{cfg.chunk_width}_{cfg.search_strategy}_log{cfg.logger_mode}{batch_suffix}"
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
