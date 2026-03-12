from __future__ import annotations

from typing import Any, Callable, List, Sequence, TypeVar

import torch

from utils.memory_control import estimate_max_batch_size

T = TypeVar("T")
R = TypeVar("R")


def get_initial_batch_size(
    engine: Any,
    prompt_len: int,
    max_new_tokens: int,
    *,
    config_batch_size: int,
    use_dynamic: bool,
    gpu_memory_utilization: float,
    overhead_ratio: float,
    max_batch_cap: int,
    min_batch_size: int,
    logger: Any,
    policy_name: str,
) -> int:
    """
    Resolve initial batch size:
    - static mode: use config batch size
    - dynamic mode: use estimated batch size
    """
    config_bs = int(config_batch_size)
    min_bs = int(min_batch_size)
    if not bool(use_dynamic):
        logger.info("%s batch size (config): %s", policy_name, config_bs)
        return config_bs

    est = estimate_max_batch_size(
        engine,
        int(prompt_len),
        int(max_new_tokens),
        gpu_memory_utilization=float(gpu_memory_utilization),
        overhead_ratio=float(overhead_ratio),
        max_batch_cap=int(max_batch_cap),
        min_batch_size=min_bs,
    )
    if est is None:
        logger.warning(
            "%s dynamic batch estimate unavailable; fallback to config batch size=%s",
            policy_name,
            config_bs,
        )
        return config_bs
    result = int(est)
    logger.info(
        "%s batch size (dynamic estimate): %s [prompt_len=%s, max_new_tokens=%s, util=%.2f, overhead=%.2f]",
        policy_name,
        result,
        int(prompt_len),
        int(max_new_tokens),
        float(gpu_memory_utilization),
        float(overhead_ratio),
    )
    return result


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    if "out of memory" not in msg:
        return False
    return ("cuda" in msg) or ("cublas" in msg) or ("cudnn" in msg)


def _clear_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


class RuntimeOOMBatchRunner:
    """
    Runtime batch policy:
    - run batched workload with a fixed base batch size
    - on CUDA OOM: halve batch size only within current run() and retry
    - next run() always starts from the original base batch size
    """

    def __init__(
        self,
        *,
        initial_batch_size: int,
        logger: Any,
        policy_name: str,
        min_batch_size: int = 1,
    ) -> None:
        self.logger = logger
        self.policy_name = policy_name
        self.min_batch_size = max(1, int(min_batch_size))
        self.batch_size = max(self.min_batch_size, int(initial_batch_size))

    def run(self, items: Sequence[T], execute_batch: Callable[[List[T]], R]) -> List[R]:
        if not items:
            return []
        idx = 0
        outputs: List[R] = []
        n = len(items)
        run_batch_size = int(self.batch_size)

        while idx < n:
            chunk_size = min(run_batch_size, n - idx)
            while True:
                chunk = list(items[idx : idx + chunk_size])
                try:
                    outputs.append(execute_batch(chunk))
                    idx += chunk_size
                    break
                except Exception as exc:
                    if not _is_cuda_oom_error(exc):
                        raise
                    if chunk_size <= self.min_batch_size:
                        self.logger.error(
                            "%s runtime OOM at min chunk size=%s; re-raising.",
                            self.policy_name,
                            chunk_size,
                        )
                        raise
                    new_chunk_size = max(self.min_batch_size, chunk_size // 2)
                    if new_chunk_size == chunk_size:
                        raise
                    old_runtime_bs = int(run_batch_size)
                    if new_chunk_size < old_runtime_bs:
                        run_batch_size = int(new_chunk_size)
                    self.logger.warning(
                        "%s runtime OOM: temporary runtime_batch_size %s -> %s (this run only).",
                        self.policy_name,
                        old_runtime_bs,
                        new_chunk_size,
                    )
                    chunk_size = new_chunk_size
                    _clear_cuda_cache()
        return outputs
