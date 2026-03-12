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

    def flush_once(
        self,
        judging_buffer: Buffer,
        batch_size: int,
        node_brief_fn: Callable[[Any], str],
    ) -> JudgeBatchResult:
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
        # preview_n = int(getattr(self.config, "logger_judger_preview_n", 3) or 3)
        # for idx, (task, result) in enumerate(zip(tasks[:preview_n], results[:preview_n]), start=1):
        #     response_preview = self._build_full_response(task).replace("\n", " ")
        #     if len(response_preview) > 120:
        #         response_preview = response_preview[:120] + "..."
        #     logger.info(
        #         "Judger item[%s/%s]: node=%s layer=%s safe=%s score=%.2f response='%s'",
        #         idx,
        #         len(tasks),
        #         id(task.node),
        #         result.get("layer"),
        #         result.get("is_safe"),
        #         float(result.get("score", 0.0)),
        #         response_preview,
        #     )

        def _run_chunk(chunk_tasks: List[BufferItem]) -> List[Dict[str, Any]]:
            prompts = [task.judger_prompt or task.original_prompt or task.path_text for task in chunk_tasks]
            metadatas = [dict(task.judger_metadata or {}) for task in chunk_tasks]
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

        expand_tasks = []
        top_hits = []
        for task, result in zip(tasks, results):
            if result["layer"] == 3 and result["score"] > self.config.layer3_filter_threshold:
                expand_tasks.append(task)
                top_hits.append((node_brief_fn(task.node), float(result.get("score", 0.0))))

        expand_results: List[Dict[str, Any]] = []
        generated_tokens_full_expand = 0
        if expand_tasks:
            candidates = "; ".join(
                f"{node_desc} score={score:.2f}" for node_desc, score in top_hits
            )
            logger.warning(
                "!!! LAYER3 HIGH-SCORE CANDIDATES: %s/%s over threshold=%.2f. candidates=[%s] !!!",
                len(expand_tasks),
                len(tasks),
                float(self.config.layer3_filter_threshold),
                candidates,
            )
            logger.warning(
                "!!! LAYER4 CHECK: sending %s task(s) to full-response judging !!!",
                len(expand_tasks),
            )
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
        Expand short responses to full responses and run layer3-4 full-response judging.
        Returns (results, generated_tokens_in_expand).
        """
        if not tasks:
            return [], 0

        temperature = self.config.temperature
        top_p = self.config.top_p
        max_new_tokens = int(self.config.sample_new_tokens)
        top_k = self.config.top_k
        sample_full_new_tokens = int(self.config.sample_full_new_tokens)
        extra_new_tokens = sample_full_new_tokens - max_new_tokens
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
            chunk_prompts = [task.judger_prompt or task.original_prompt or task.path_text for task in chunk_tasks]
            chunk_metadatas = [dict(task.judger_metadata or {}) for task in chunk_tasks]
            chunk_short_responses = [self._build_full_response(task) for task in chunk_tasks]
            if extra_new_tokens <= 0:
                chunk_full_responses = chunk_short_responses
                chunk_generated_tokens = 0
            else:
                seed_ids_batch = [task.seq_ids if task.seq_ids is not None else task.path_ids for task in chunk_tasks]
                generated_full_ids = self.target_model_engine.batch_generate(
                    seed_ids_batch,
                    max_new_tokens=extra_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                chunk_full_responses = []
                chunk_generated_tokens = 0
                for task, full_ids in zip(chunk_tasks, generated_full_ids):
                    seed_len = len(task.seq_ids) if task.seq_ids is not None else len(task.path_ids)
                    extra_ids = full_ids[seed_len:]
                    chunk_generated_tokens += len(extra_ids)
                    extra_text = self.sampler.tokenizer.decode(extra_ids) if extra_ids else ""
                    chunk_full_responses.append(self._build_full_response(task) + extra_text)

            chunk_results = self.judger.batch_evaluate_full_response(
                chunk_prompts,
                chunk_full_responses,
                chunk_metadatas,
            )
            return chunk_results, int(chunk_generated_tokens)

        for chunk_results, chunk_generated in self._expand_oom_runner.run(tasks, _run_expand_chunk):
            all_results.extend(chunk_results)
            generated_tokens_in_expand += int(chunk_generated)

        return all_results, int(generated_tokens_in_expand)

    def _build_full_response(self, task: BufferItem) -> str:
        # Build full response (path_text + current seq_text) and strip the prompt portion if present.
        full_with_prompt = (task.path_text or "") + (task.seq_text or "")
        prompt_with_chat_template = task.prompt_with_chat_template or ""
        if prompt_with_chat_template and full_with_prompt.startswith(prompt_with_chat_template):
            return full_with_prompt[len(prompt_with_chat_template):]
        return full_with_prompt
