import gc
import os
from typing import List

import torch

from utils.logger import setup_logger

from .base_engine import JudgerEngineBase

logger = setup_logger("VLLMLocalModel")

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vllm not available")


class VLLMLocalJudgerEngine(JudgerEngineBase):
    def __init__(
        self,
        name,
        cuda_number,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization: float = 0.8,
    ):
        # Store init args for reinitialize()
        self._init_name = name
        self._init_cuda_number = cuda_number
        self._init_tensor_parallel_size = tensor_parallel_size
        self._init_max_model_len = max_model_len
        self._init_gpu_memory_utilization = gpu_memory_utilization

        self._build_engine()

    def _build_engine(self):
        """Create the vLLM engine (used by __init__ and reinitialize)."""
        cuda_devices = (
            ",".join(map(str, self._init_cuda_number))
            if isinstance(self._init_cuda_number, (list, tuple))
            else str(self._init_cuda_number)
        )
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        self.max_model_len = int(self._init_max_model_len)
        try:
            self.llm = LLM(
                model=self._init_name,
                tensor_parallel_size=self._init_tensor_parallel_size,
                max_model_len=self._init_max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=float(self._init_gpu_memory_utilization),
            )
        finally:
            if prev_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
        self.tokenizer = self.llm.get_tokenizer()

    def reinitialize(self):
        """Destroy the dead engine and create a fresh one."""
        logger.warning("Reinitializing vLLM judge engine...")
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        self._build_engine()
        logger.info("vLLM judge engine reinitialized successfully.")

    def batch_generate_content(self, texts: List[str]) -> List[str]:
        # Use deterministic generate() path to keep strict 1:1 batch cardinality.
        prompts: List[str] = []
        for t in texts:
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        if len(outputs) != len(texts):
            max_prompt_tokens = -1
            try:
                lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
                max_prompt_tokens = max(lengths) if lengths else -1
            except Exception:
                pass
            raise RuntimeError(
                "VLLM judger output size mismatch: "
                f"inputs={len(texts)} outputs={len(outputs)} "
                f"max_prompt_tokens={max_prompt_tokens} max_model_len={self.max_model_len}"
            )
        return [output.outputs[0].text for output in outputs]
