from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional

from components.buffer.buffer import Buffer, BufferItem
from sampler.sampler import Sampler
from utils.batch_policy import RuntimeOOMBatchRunner, get_initial_batch_size
from utils.logger import setup_logger
from profiler import profile

logger = setup_logger("SampleWorker")


@dataclass
class SampleBatchResult:
    tasks: int
    generated_tokens: int
    pushed_to_judging: int
    requeued_to_sample: int
    produced_at: float


class SampleWorker:
    """
    Sampling stage worker.

    M2 keeps synchronous invocation from Executor while isolating stage logic.
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self._use_dynamic_batch_size = bool(self.sampler.config.use_dynamic_batch_size)
        self._oom_runner: Optional[RuntimeOOMBatchRunner] = None

    def _build_initial_batch_size(self) -> int:
        if self._use_dynamic_batch_size:
            initial_bs = get_initial_batch_size(
                self.sampler.engine,
                prompt_len=int(self.sampler.config.sampler_estimate_prompt_len),
                max_new_tokens=int(self.sampler.config.sample_new_tokens),
                config_batch_size=int(self.sampler.config.sampler_batch_size),
                use_dynamic=True,
                gpu_memory_utilization=float(self.sampler.config.target_gpu_memory_utilization),
                overhead_ratio=float(self.sampler.config.hf_batch_estimate_overhead),
                max_batch_cap=int(self.sampler.config.max_batch_size),
                min_batch_size=int(self.sampler.config.min_batch_size),
                logger=logger,
                policy_name="SampleWorker",
            )
        else:
            initial_bs = int(self.sampler.config.sampler_batch_size)
        return int(initial_bs)

    def _ensure_oom_runner(self) -> RuntimeOOMBatchRunner:
        if self._oom_runner is None:
            initial_bs = self._build_initial_batch_size()
            self._oom_runner = RuntimeOOMBatchRunner(
                initial_batch_size=initial_bs,
                logger=logger,
                policy_name="SampleWorker",
                min_batch_size=int(self.sampler.config.min_batch_size),
            )
            logger.info("Sampling batch policy initialized: initial_batch_size=%s", initial_bs)
        return self._oom_runner

    @profile("sampler.flush_once")
    def flush_once(self, sample_buffer: Buffer, judging_buffer: Buffer) -> SampleBatchResult:
        oom_runner = self._ensure_oom_runner()
        runtime_batch_size = int(oom_runner.batch_size)
        # Pop all pending tasks; runtime chunking is fully delegated to RuntimeOOMBatchRunner.
        sample_tasks: List[BufferItem] = sample_buffer.pop_all()
        if not sample_tasks:
            return SampleBatchResult(
                tasks=0,
                generated_tokens=0,
                pushed_to_judging=0,
                requeued_to_sample=0,
                produced_at=time.monotonic(),
            )

        logger.info(
            "Sampling flush start: tasks=%s (runtime_batch_size=%s, sample_buffer remaining=%s)",
            len(sample_tasks),
            runtime_batch_size,
            len(sample_buffer),
        )
        generated_tokens = 0
        pushed = 0
        requeued = 0

        def _run_chunk(chunk_tasks: List[BufferItem]) -> None:
            nonlocal generated_tokens, pushed, requeued
            sample_tasks_ids = [task.path_ids for task in chunk_tasks]
            logger.info(
                "Sampling chunk: tasks=%s (chunk_size=%s)",
                len(chunk_tasks),
                len(chunk_tasks),
            )
            new_children_groups, tau_invalid_flags = self.sampler.batch_uniform_generate(
                sample_tasks_ids,
                return_invalid_flags=True,
            )
            generated_tokens += int(sum(len(new_children) for new_children in new_children_groups))
            # tau_invalid only marks failures caused by tau/threshold gating.
            for task, new_children, tau_invalid in zip(chunk_tasks, new_children_groups, tau_invalid_flags):
                if tau_invalid:
                    sample_buffer.add_requests(
                        task.node,
                        1,
                        judger_prompt=task.judger_prompt,
                        judger_metadata=task.judger_metadata,
                        original_prompt=task.original_prompt,
                        prompt_with_chat_template=task.prompt_with_chat_template,
                    )
                    requeued += 1
                    continue
                task.seq_new_ids = new_children
                task.seq_ids = task.path_ids + new_children
                task.seq_text = self.sampler.tokenizer.decode(new_children)
                judging_buffer.add_item(task)
                pushed += 1

        oom_runner.run(sample_tasks, _run_chunk)

        logger.info(
            "Sampling done: pushed=%s to judging_buffer (size=%s), requeued=%s to sample_buffer (size=%s), runtime_batch_size=%s",
            pushed,
            len(judging_buffer),
            requeued,
            len(sample_buffer),
            oom_runner.batch_size,
        )
        return SampleBatchResult(
            tasks=len(sample_tasks),
            generated_tokens=generated_tokens,
            pushed_to_judging=pushed,
            requeued_to_sample=requeued,
            produced_at=time.monotonic(),
        )
