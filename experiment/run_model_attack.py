from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    prompt_indices: Optional[List[int]] = None  # specific prompt indices to run
    benchmark_path: Optional[str] = None  # override benchmark path
    harmful_prompt_start: int = 1
    harmful_prompt_end: int = -1
    logger_mode: str = "info"
    use_dynamic_batch_size: bool = True
    test_mode: bool = False
    use_mock_engines: bool = False
    chunk_size: int = 1
    chunk_width: int = 100
    time_limit_sec: int = 600



# Fixed GPU assignments
TARGET_MODEL_CUDA_NUMBER = 1
JUDGER_CUDA_NUMBER = 3
TARGET_ENGINE_NAME = "hf"
JUDGER_ENGINE_NAME = "vllm"

# Profiling settings
ENABLE_CPROFILE = False
CPROFILE_OUTPUT_DIR = "profiles"

# Wait for a PID to exit before starting; set to None to skip waiting.
WAIT_PID = None
WAIT_INTERVAL_SEC = 5

MODEL_CONFIGS: List[SweepConfig] = [

#     SweepConfig(
#         "single_turn",
#         "Qwen/Qwen3-8B", 0.95, 20, 0.6, 0.0001,
#         prompt_indices=[111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
#  # specify which prompts to run
#         benchmark_path="./benchmark/TAP/tap_qwen3_attacked_jo_compatible.json"
#     ),
    SweepConfig(
        "single_turn",
        "meta-llama/Llama-3.1-8B-Instruct",
        0.9, -1, 0.6, 0.0001,
        prompt_indices = [125, 124, 120, 119, 118, 113, 110, 107, 105, 101, 98, 95, 93, 92, 91, 90, 81, 80, 78, 71, 70, 68, 65, 64, 62, 61, 54, 53, 47, 43, 40, 39, 36, 30, 29, 28, 27, 23, 21, 20, 19, 17],

        benchmark_path="./benchmark/GCG/jailbreak_oracle_benchmark_gcg_ccea.json",
    ),

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
        command.extend(["-m", "cProfile", "-o", str(profile_path)])

    command.extend(
        [
            str(root_dir / "run.py"),
            "--workload_name", cfg.workload_name,
            "--target_model", cfg.target_model,
            "--target_engine_name", target_engine,
            "--judger_engine_name", judger_engine,
            "--top_p", str(cfg.top_p),
            "--top_k", str(cfg.top_k),
            "--temperature", str(cfg.temperature),
            "--likelihood", str(cfg.likelihood),
            "--logger_mode", cfg.logger_mode,
            "--target_model_cuda_number", str(TARGET_MODEL_CUDA_NUMBER),
            "--judger_cuda_number", str(JUDGER_CUDA_NUMBER),
            "--use_dynamic_batch_size", str(cfg.use_dynamic_batch_size).lower(),
            "--harmful_prompt_start", str(cfg.harmful_prompt_start),
            "--harmful_prompt_end", str(cfg.harmful_prompt_end),
            "--chunk_size", str(cfg.chunk_size),
            "--chunk_width", str(cfg.chunk_width),
            "--time_limit_sec", str(cfg.time_limit_sec),
            "--judger_gpu_memory_utilization", "0.9",
            "--target_gpu_memory_utilization", "1.2",

        ]
    )
    if cfg.benchmark_path:
        command.extend(["--benchmark_path", cfg.benchmark_path])
    if cfg.prompt_indices:
        command.extend(["--prompt_indices", ",".join(str(i) for i in cfg.prompt_indices)])
    if cfg.test_mode:
        command.extend(["--test_mode", "true"])
    return command


def _task_name(cfg: SweepConfig, task_id: int) -> str:
    model_short = cfg.target_model.replace("/", "_")
    mode_suffix = "_test" if cfg.test_mode else ""
    engine_suffix = "_mock" if cfg.use_mock_engines else ""
    batch_suffix = "_dynbon" if cfg.use_dynamic_batch_size else "_dynboff"
    prompts_suffix = ""
    if cfg.prompt_indices:
        prompts_suffix = f"_pidx{'-'.join(str(i) for i in cfg.prompt_indices[:5])}"
        if len(cfg.prompt_indices) > 5:
            prompts_suffix += f"_plus{len(cfg.prompt_indices) - 5}"
    return (
        f"{task_id:03d}_{cfg.workload_name}_{model_short}_p{cfg.top_p}_k{cfg.top_k}_t{cfg.temperature}"
        f"_lh{cfg.likelihood}_h{cfg.harmful_prompt_start}-{cfg.harmful_prompt_end}"
        f"_cs{cfg.chunk_size}_cw{cfg.chunk_width}_log{cfg.logger_mode}{batch_suffix}"
        f"{prompts_suffix}{mode_suffix}{engine_suffix}"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack with specific prompts and benchmark path")
    parser.add_argument(
        "--prompt_indices", type=str, default=None,
        help="Comma-separated list of prompt indices to run, e.g. '1,3,5,10'. Overrides MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--benchmark_path", type=str, default=None,
        help="Path to benchmark file. Overrides MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--harmful_prompt_start", type=int, default=None,
        help="Start index for harmful prompts. Overrides MODEL_CONFIGS.",
    )
    parser.add_argument(
        "--harmful_prompt_end", type=int, default=None,
        help="End index for harmful prompts. Overrides MODEL_CONFIGS.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "experiment" / "logs"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_CPROFILE:
        profiles_dir = root / "experiment" / CPROFILE_OUTPUT_DIR
        profiles_dir.mkdir(parents=True, exist_ok=True)

    if WAIT_PID is not None:
        _wait_pid_exit(int(WAIT_PID), int(WAIT_INTERVAL_SEC))

    # CLI overrides for all configs
    cli_prompt_indices = None
    if args.prompt_indices:
        cli_prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(",")]

    configs = []
    for cfg in MODEL_CONFIGS:
        overrides = {}
        if cli_prompt_indices is not None:
            overrides["prompt_indices"] = cli_prompt_indices
        if args.benchmark_path is not None:
            overrides["benchmark_path"] = args.benchmark_path
        if args.harmful_prompt_start is not None:
            overrides["harmful_prompt_start"] = args.harmful_prompt_start
        if args.harmful_prompt_end is not None:
            overrides["harmful_prompt_end"] = args.harmful_prompt_end
        if overrides:
            # Rebuild frozen dataclass with overrides
            d = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
            d.update(overrides)
            cfg = SweepConfig(**d)
        configs.append(cfg)

    logger.info("Total tasks: %s", len(configs))
    if cli_prompt_indices:
        logger.info("Prompt indices (CLI override): %s", cli_prompt_indices)
    if args.benchmark_path:
        logger.info("Benchmark path (CLI override): %s", args.benchmark_path)
    if args.harmful_prompt_start is not None:
        logger.info("Harmful prompt start (CLI override): %s", args.harmful_prompt_start)
    if args.harmful_prompt_end is not None:
        logger.info("Harmful prompt end (CLI override): %s", args.harmful_prompt_end)
    logger.info("-" * 80)

    for i, cfg in enumerate(configs, start=1):
        run_task(i, cfg, root, logs_dir, run_timestamp)
        time.sleep(1)

    logger.info("-" * 80)
    logger.info("All tasks done (sequential worker).")
