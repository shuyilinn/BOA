from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import time
from typing import Any, Callable, Dict, List, Tuple

from components.buffer.buffer import Buffer, BufferItem
from judgers.base_judger import PipelineJudger
from sampler.sampler import Sampler
from utils.batch_policy import RuntimeOOMBatchRunner
from utils.logger import setup_logger
from profiler import profile

logger = setup_logger("JudgeWorker")


@dataclass
class JudgeBatchResult:
    tasks: List[BufferItem]
    results: List[Dict[str, Any]]
    expand_tasks: List[BufferItem]
    expand_results: List[Dict[str, Any]]
    generated_tokens_full_expand: int
    produced_at: float


class JudgeWorker:
    """
    Judging stage worker.

    M2 keeps synchronous invocation from Executor while isolating stage logic.
    """

    def __init__(
        self,
        judger: PipelineJudger,
        target_model_engine: Any,
        sampler: Sampler,
        config: Any,
    ):
        self.judger = judger
        self.target_model_engine = target_model_engine
        self.sampler = sampler
        self.config = config
        self._oom_runner: RuntimeOOMBatchRunner | None = None
        self._expand_oom_runner: RuntimeOOMBatchRunner | None = None

    def reset_oom_runner(self) -> None:
        """Reset OOM runner batch sizes to initial value (call at prompt boundaries)."""
        if self._oom_runner is not None:
            self._oom_runner.reset()
        if self._expand_oom_runner is not None:
            self._expand_oom_runner.reset()

    @profile("judger.flush_once")
    def flush_once(
        self,
        judging_buffer: Buffer,
        batch_size: int,
        node_brief_fn: Callable[[Any], str],
    ) -> JudgeBatchResult:
        """
        Pop tasks from judging_buffer and run two-stage judging:
          Stage 1: Layer 1-3 fast evaluation on short (truncated) responses.
          Stage 2: For layer-3 high-scorers that were truncated (not EOS),
                   expand to full response and run layer-4 evaluation.
        """
        # --- OOM-aware batch size ---
        target_bs = max(1, int(batch_size))
        if self._oom_runner is None:
            self._oom_runner = RuntimeOOMBatchRunner(
                initial_batch_size=target_bs,
                logger=logger,
                policy_name="JudgeWorker",
            )
        else:
            self._oom_runner.batch_size = min(self._oom_runner.batch_size, target_bs)

        pop_count = min(target_bs, self._oom_runner.batch_size)
        tasks = judging_buffer.pop_batch(count=pop_count)
        if not tasks:
            return JudgeBatchResult(
                tasks=[],
                results=[],
                expand_tasks=[],
                expand_results=[],
                generated_tokens_full_expand=0,
                produced_at=time.monotonic(),
            )

        logger.info("Judging flush start: tasks=%s (judging_buffer remaining=%s)", len(tasks), len(judging_buffer))
        results: List[Dict[str, Any]] = []

        # --- Stage 1: Layer 1-3 evaluation on short responses ---
        def _run_chunk(chunk_tasks: List[BufferItem]) -> List[Dict[str, Any]]:
            prompts = [self._get_judger_prompt(task) for task in chunk_tasks]
            metadatas = [dict(task.judger_metadata or {}) for task in chunk_tasks]
            for i, task in enumerate(chunk_tasks):
                metadatas[i]["_full_response"] = self._build_full_response(task)
            responses = [self._build_full_response(task) for task in chunk_tasks]
            logger.info(
                "Judging chunk: tasks=%s (runtime_batch_size=%s)",
                len(chunk_tasks),
                self._oom_runner.batch_size if self._oom_runner else "N/A",
            )
            chunk_results = self.judger.batch_evaluate(prompts, responses, metadatas)
            layer_stats = Counter(r.get("layer", -1) for r in chunk_results)
            score_vals = [float(r.get("score", 0.0)) for r in chunk_results]
            logger.info(
                "Judger chunk summary: layers=%s score[min=%.2f avg=%.2f max=%.2f]",
                dict(layer_stats),
                min(score_vals) if score_vals else 0.0,
                (sum(score_vals) / len(score_vals)) if score_vals else 0.0,
                max(score_vals) if score_vals else 0.0,
            )
            return chunk_results

        assert self._oom_runner is not None
        for chunk_results in self._oom_runner.run(tasks, _run_chunk):
            results.extend(chunk_results)

        # --- Stage 2: Expand truncated layer-3 high-scorers and run layer-4 ---
        # Only expand tasks that were truncated by max_new_tokens (not naturally completed with EOS).
        expand_tasks = []
        for task, result in zip(tasks, results):
            if result["layer"] == 3 and result["score"] > self.config.layer3_filter_threshold and not task.is_complete:
                expand_tasks.append(task)

        expand_results: List[Dict[str, Any]] = []
        generated_tokens_full_expand = 0
        if expand_tasks:
            expand_results, generated_tokens_full_expand = self._expanding_and_judging(expand_tasks)

        logger.info("Judging done: runtime_batch_size=%s", self._oom_runner.batch_size)

        return JudgeBatchResult(
            tasks=tasks,
            results=results,
            expand_tasks=expand_tasks,
            expand_results=expand_results,
            generated_tokens_full_expand=generated_tokens_full_expand,
            produced_at=time.monotonic(),
        )

    def _expanding_and_judging(self, tasks: List[BufferItem]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Continue generating truncated responses to full length, then run layer-4 judging.
        Called only for truncated tasks (is_complete=False) that scored high in layer-3.
        Returns (results, generated_tokens_in_expand).
        """
        temperature = self.config.temperature
        top_p = self.config.top_p
        top_k = self.config.top_k
        # Extra tokens to generate: full budget minus what was already sampled
        extra_new_tokens = int(self.config.sample_full_new_tokens) - int(self.config.sample_new_tokens)
        initial_expand_bs = int(self._oom_runner.batch_size)
        if self._expand_oom_runner is None:
            self._expand_oom_runner = RuntimeOOMBatchRunner(
                initial_batch_size=initial_expand_bs,
                logger=logger,
                policy_name="JudgeWorker-Expand",
            )

        all_results: List[Dict[str, Any]] = []
        generated_tokens_in_expand = 0

        def _run_expand_chunk(chunk_tasks: List[BufferItem]) -> Tuple[List[Dict[str, Any]], int]:
            chunk_prompts = [self._get_judger_prompt(task) for task in chunk_tasks]
            chunk_metadatas = [dict(task.judger_metadata or {}) for task in chunk_tasks]
            for i, task in enumerate(chunk_tasks):
                chunk_metadatas[i]["_full_response"] = self._build_full_response(task)

            # Step 1: Continue generating from the truncated sequence with tau protection
            seed_ids_batch = [task.seq_ids if task.seq_ids is not None else task.path_ids for task in chunk_tasks]
            base_generated_lens = [len(task.seq_new_ids or []) for task in chunk_tasks]
            new_children_groups, tau_list, tau_invalid_flags = self.sampler.batch_uniform_generate_with_tau(
                seed_ids_batch,
                base_generated_lens=base_generated_lens,
                max_new_tokens=extra_new_tokens,
                return_tau=True,
                compute_tau=True,
                return_invalid_flags=True,
            )

            # Decode extra tokens and build full responses (short response + extended text)
            # tau-invalid tasks are marked safe and skipped from layer-4 judging.
            chunk_full_responses = []
            chunk_generated_tokens = 0
            valid_indices = []
            chunk_results: List[Dict[str, Any]] = [{} for _ in chunk_tasks]
            for i, (task, new_children, tau_invalid) in enumerate(
                zip(chunk_tasks, new_children_groups, tau_invalid_flags)
            ):
                chunk_generated_tokens += len(new_children)
                if tau_invalid:
                    chunk_results[i] = {"layer": 3, "score": 0, "is_safe": True, "action": "STOP"}
                    chunk_full_responses.append("")
                    logger.info("Expand tau-invalid: skipping layer-4 for task %d", i)
                else:
                    extra_text = self.sampler.tokenizer.decode(new_children) if new_children else ""
                    chunk_full_responses.append(self._build_full_response(task) + extra_text)
                    valid_indices.append(i)

            # Step 2: Run layer-4 judging only on tau-valid full responses
            if valid_indices:
                eval_results = self.judger.batch_evaluate_full_response(
                    [chunk_prompts[i] for i in valid_indices],
                    [chunk_full_responses[i] for i in valid_indices],
                    [chunk_metadatas[i] for i in valid_indices],
                )
                for j, idx in enumerate(valid_indices):
                    chunk_results[idx] = eval_results[j]

            return chunk_results, int(chunk_generated_tokens)

        for chunk_results, chunk_generated in self._expand_oom_runner.run(tasks, _run_expand_chunk):
            all_results.extend(chunk_results)
            generated_tokens_in_expand += int(chunk_generated)

        return all_results, int(generated_tokens_in_expand)

    @staticmethod
    def _get_judger_prompt(task: BufferItem) -> str:
        """Return the judger prompt for a task.

        Uses judger_prompt (raw user request) or original_prompt as fallback.
        Never falls back to path_text — that is the chat-template-wrapped model
        input and semantically wrong as a judger prompt.
        """
        prompt = task.judger_prompt or task.original_prompt
        assert prompt, (
            "BufferItem has neither judger_prompt nor original_prompt; "
            "cannot determine judger prompt. This indicates a bug in buffer "
            "population — every task must carry an explicit prompt."
        )
        return prompt

    def _build_full_response(self, task: BufferItem) -> str:
        """Build the assistant response text for judging.

        Uses the snapshot taken at buffer-insertion time (task.node_assistant_text)
        plus any additional tokens generated by the sampler (seq_text).
        Reading a frozen snapshot instead of live node state enforces the
        invariant that buffered tasks are not affected by later tree mutations.
        """
        assert task.node_assistant_text is not None, (
            "BufferItem must carry a frozen assistant text snapshot "
            "(set by Buffer.add_requests at enqueue time). "
            "A None snapshot indicates the item was constructed outside "
            "the canonical buffer population path."
        )
        return task.node_assistant_text + (task.seq_text or "")
