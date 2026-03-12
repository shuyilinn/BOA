import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base_judger import (
    AtomicJudgerResult,
    BaseAtomicJudger,
    JudgerAction,
    JudgerExecutionType,
    JudgerSample,
)
from .judger_engines.base_engine import JudgerEngineBase
from utils.batch_policy import get_initial_batch_size
from utils.logger import setup_logger

logger = setup_logger("TreeGuideJudger")


@dataclass(frozen=True)
class _PipelineMode:
    use_layer1: bool
    use_layer2: bool
    use_layer4: bool


class TreeGuideJudger:
    def __init__(
        self,
        local_judger_engine: JudgerEngineBase,
        api_judger_engine: JudgerEngineBase,
        config: Any = None,
        layer1_judger: Optional[BaseAtomicJudger] = None,
        layer2_judger: Optional[BaseAtomicJudger] = None,
        layer3_judger: Optional[BaseAtomicJudger] = None,
        layer4_judger: Optional[BaseAtomicJudger] = None,
    ):
        self.config = config
        self.layer3_filter_threshold = float(self.config.layer3_filter_threshold)
        self.local_judger_engine = local_judger_engine
        self.api_judger_engine = api_judger_engine
        self.layer1_judger = layer1_judger
        self.layer2_judger = layer2_judger
        self.layer3_judger = layer3_judger
        self.layer4_judger = layer4_judger

    def get_batch_size(self) -> int:
        return get_initial_batch_size(
            self.local_judger_engine,
            prompt_len=int(self.config.judger_estimate_prompt_len),
            max_new_tokens=int(self.config.judger_estimate_max_new_tokens),
            config_batch_size=int(self.config.judger_batch_size),
            use_dynamic=bool(self.config.use_dynamic_batch_size),
            gpu_memory_utilization=float(self.config.judger_gpu_memory_utilization),
            overhead_ratio=float(self.config.hf_batch_estimate_overhead),
            max_batch_cap=int(self.config.max_batch_size),
            min_batch_size=1,
            logger=logger,
            policy_name="Judger",
        )

    def evaluate(self, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.batch_evaluate([prompt], [response], [metadata or {}])[0]

    def batch_evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        mode = _PipelineMode(use_layer1=True, use_layer2=True, use_layer4=False)
        return self._run_pipeline(prompts, responses, mode, metadatas)

    def evaluate_full_response(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.batch_evaluate_full_response([prompt], [response], [metadata or {}])[0]

    def batch_evaluate_full_response(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        mode = _PipelineMode(use_layer1=False, use_layer2=False, use_layer4=True)
        return self._run_pipeline(prompts, responses, mode, metadatas)

    def evaluate_attack_sampling(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.batch_evaluate_attack_sampling([prompt], [response], [metadata or {}])[0]

    def batch_evaluate_attack_sampling(
        self,
        prompts: List[str],
        responses: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        mode = _PipelineMode(use_layer1=True, use_layer2=True, use_layer4=True)
        return self._run_pipeline(prompts, responses, mode, metadatas)

    def get_refusal_filter_stats(self) -> Dict[str, Any]:
        if self.layer1_judger is None:
            return {}
        return self.layer1_judger.get_stats()

    def _should_clean_response(self) -> bool:
        workload_name = str(getattr(self.config, "workload_name", "") or "")
        workload_configs = getattr(self.config, "workload_configs", {}) or {}
        workload_config = workload_configs.get(workload_name, {})
        return bool(workload_config.get("clean_response", True))

    def _run_pipeline(
        self,
        prompts: List[str],
        responses: List[str],
        mode: _PipelineMode,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if len(prompts) != len(responses):
            raise ValueError(f"prompts/responses length mismatch: {len(prompts)} != {len(responses)}")
        if not prompts:
            return []

        if metadatas is None:
            metadatas = [{} for _ in prompts]
        if self._should_clean_response():
            cleaned_responses = [self._clean_response(response) for response in responses]
        else:
            cleaned_responses = responses
        samples = [
            JudgerSample(prompt=prompt, response=response, metadata=metadata)
            for prompt, response, metadata in zip(prompts, cleaned_responses, metadatas)
        ]
        final_results: List[Optional[AtomicJudgerResult]] = [None] * len(samples)
        pending_indices = list(range(len(samples)))

        if mode.use_layer1:
            pending_indices = self._apply_judger(self.layer1_judger, pending_indices, samples, final_results)
            if not pending_indices:
                return [result.to_dict() for result in final_results if result is not None]

        if mode.use_layer2:
            pending_indices = self._apply_judger(self.layer2_judger, pending_indices, samples, final_results)
            if not pending_indices:
                return [result.to_dict() for result in final_results if result is not None]

        pending_indices = self._apply_judger(self.layer3_judger, pending_indices, samples, final_results)

        if mode.use_layer4 and self.layer4_judger and pending_indices:
            self._apply_judger(self.layer4_judger, pending_indices, samples, final_results)

        if any(result is None for result in final_results):
            missing = sum(1 for result in final_results if result is None)
            raise RuntimeError(f"judger pipeline produced incomplete results: missing={missing}, total={len(samples)}")
        return [result.to_dict() for result in final_results if result is not None]

    def _apply_judger(
        self,
        judger: Optional[BaseAtomicJudger],
        pending_indices: List[int],
        samples: List[JudgerSample],
        final_results: List[Optional[AtomicJudgerResult]],
    ) -> List[int]:
        if judger is None:
            return pending_indices
        if judger.execution_type == JudgerExecutionType.API:
            return self._apply_api_judger(judger, pending_indices, samples, final_results)
        return self._apply_local_judger(judger, pending_indices, samples, final_results)

    def _apply_local_judger(
        self,
        judger: BaseAtomicJudger,
        pending_indices: List[int],
        samples: List[JudgerSample],
        final_results: List[Optional[AtomicJudgerResult]],
    ) -> List[int]:
        if not pending_indices:
            return []

        batch_samples = [samples[idx] for idx in pending_indices]
        results = judger.batch_judge(batch_samples)
        if len(results) != len(pending_indices):
            raise RuntimeError(
                f"layer{judger.layer} output size mismatch: pending={len(pending_indices)} outputs={len(results)}"
            )

        next_pending: List[int] = []
        for idx_in_batch, result in enumerate(results):
            idx = pending_indices[idx_in_batch]
            self._log_atomic_result(result, idx)
            final_results[idx] = result
            if result.action == JudgerAction.CONTINUE:
                next_pending.append(idx)
        return next_pending

    def _apply_api_judger(
        self,
        judger: BaseAtomicJudger,
        api_indices: List[int],
        samples: List[JudgerSample],
        final_results: List[Optional[AtomicJudgerResult]],
    ) -> List[int]:
        if not api_indices:
            return []

        logger.info(
            "API Judger Layer%s: Sending %s/%s samples to API judger",
            judger.layer,
            len(api_indices),
            len(samples),
        )
        total_api = len(api_indices)
        for rank, idx in enumerate(api_indices, start=1):
            sample = samples[idx]
            logger.warning(
                "LAYER%s INPUT[%s/%s]: prompt=\n%s\nresponse=\n%s\nfull_candidate=\n%s",
                judger.layer,
                rank,
                total_api,
                sample.prompt,
                sample.response,
                f"{sample.prompt}{sample.response}",
            )

        next_pending: List[int] = []
        for pos, idx in enumerate(api_indices):
            sample = samples[idx]
            logger.warning(
                "LAYER%s SEND[idx=%s]: prompt=\n%s\nresponse=\n%s\nfull_candidate=\n%s",
                judger.layer,
                idx,
                sample.prompt,
                sample.response,
                f"{sample.prompt}{sample.response}",
            )
            result = judger.judge(sample.prompt, sample.response, metadata=sample.metadata)
            prev = final_results[idx]
            if prev is not None:
                result = AtomicJudgerResult(
                    action=result.action,
                    score=result.score,
                    is_safe=result.is_safe,
                    layer=result.layer,
                    response=result.response,
                    raw_output=result.raw_output,
                    layer3_score=(
                        result.layer3_score
                        if result.layer3_score is not None
                        else (prev.layer3_score if prev.layer3_score is not None else prev.score)
                    ),
                    layer4_score=result.layer4_score if result.layer4_score is not None else prev.layer4_score,
                    metadata={**prev.metadata, **result.metadata},
                )
            self._log_atomic_result(result, idx)
            final_results[idx] = result
            if result.action == JudgerAction.CONTINUE:
                next_pending.append(idx)
            if not result.is_safe:
                logger.warning(
                    "LAYER%s EARLY STOP: unsafe detected at idx=%s; skip remaining %s samples.",
                    judger.layer,
                    idx,
                    max(0, total_api - pos - 1),
                )
                break
        return next_pending

    def _log_atomic_result(self, result: AtomicJudgerResult, idx: int) -> None:
        logger.info(
            "LAYER%s RESULT[idx=%s]: action=%s is_safe=%s score=%s response=\n%s",
            result.layer,
            idx,
            result.action.value,
            bool(result.is_safe),
            float(result.score),
            result.response,
        )
        if result.raw_output:
            logger.info(
                "LAYER%s FEEDBACK[idx=%s]:\n%s",
                result.layer,
                idx,
                result.raw_output,
            )

    def _clean_response(self, response: str) -> str:
        if not response:
            return response

        cleaned = response
        conversation_markers = [
            r"(?:^|\n)\s*Human:\s*",
            r"(?:^|\n)\s*User:\s*",
            r"(?:^|\n)\s*Assistant:\s*",
            r"(?:^|\n)\s*<\|im_start\|>user\s*",
            r"(?:^|\n)\s*<\|im_start\|>assistant\s*",
            r"(?:^|\n)\s*<\|start_header_id\|>user<\|end_header_id\|>\s*",
            r"(?:^|\n)\s*<\|start_header_id\|>assistant<\|end_header_id\|>\s*",
            r"(?:^|\n)\s*\[INST\]",
            r"(?:^|\n)\s*\[/INST\]",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"<\|endoftext\|>",
            r"<\|begin_of_text\|>",
            r"<\|eot_id\|>",
            r"<\|eom_id\|>",
            r"<\|start_header_id\|>",
            r"<\|end_header_id\|>",
            r"<\|assistant\|>",
            r"<\|user\|>",
            r"<\|system\|>",
            r"<\|tool\|>",
            r"<\|observation\|>",
            r"<\|finetune_right_pad_id\|>",
            r"<\|reserved_special_token_\d+\|>",
        ]
        merged_markers = [
            r"If you have concerns\s*Human:\s*",
            r"If you have questions\s*Human:\s*",
            r"If you need\s*Human:\s*",
            r"For more information\s*Human:\s*",
            r"If you'd like\s*Human:\s*",
            r"If you want\s*Human:\s*",
            r"Should you have\s*Human:\s*",
            r"If you require\s*Human:\s*",
            r"If you have concerns\s*User:\s*",
            r"If you have questions\s*User:\s*",
            r"If you need\s*User:\s*",
            r"'[st]Human:\s*",
            r"'[st]\s*Human:\s*",
            r"[\.\)\'\"]\s*Human:\s*",
            r"[a-zA-Z]Human:\s*",
            r"[a-zA-Z]Human\*+:\s*",
            r"'[st]Assistant:\s*",
            r"'[st]\s*Assistant:\s*",
            r"[\.\)\'\"]\s*Assistant:\s*",
            r"[a-zA-Z]Assistant:\s*",
            r"[a-zA-Z]Assistant\*+:\s*",
            r"'[st]User:\s*",
            r"'[st]\s*User:\s*",
            r"[\.\)\'\"]\s*User:\s*",
            r"[a-zA-Z]User:\s*",
            r"[a-zA-Z]User\*+:\s*",
        ]

        earliest_pos = len(cleaned)
        for pattern in conversation_markers + merged_markers:
            match = re.search(pattern, cleaned, re.IGNORECASE | re.MULTILINE)
            if match:
                earliest_pos = min(earliest_pos, match.start())

        question_patterns = [
            r"\.\s*I want to\s+",
            r"\.\s*Can you\s+",
            r"\.\s*How do I\s+",
            r"\.\s*What is\s+",
            r"\.\s*Tell me\s+",
        ]
        min_response_length = 50
        if len(cleaned) > min_response_length:
            for pattern in question_patterns:
                match = re.search(pattern, cleaned[min_response_length:], re.IGNORECASE)
                if match:
                    earliest_pos = min(earliest_pos, match.start() + min_response_length)

        if earliest_pos < len(cleaned):
            cleaned = cleaned[:earliest_pos].rstrip()
            removed_chars = len(response) - len(cleaned)
            if removed_chars > 0:
                logger.info("[CLEAN RESPONSE] Removed %s characters of fictional conversation content", removed_chars)
                logger.debug("[CLEAN RESPONSE] Original: %s", response)
                logger.debug("[CLEAN RESPONSE] Cleaned: %s", cleaned)
        return cleaned
