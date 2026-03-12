import os
import torch
from typing import List, Union, Tuple, Optional, Any
from engines.base_engine import TargetModelEngineBase

# [shuyi: this file has not been reviewed yet]
class VllmTargetModelEngine(TargetModelEngineBase):
    def __init__(
        self, 
        model_name: str, 
        cuda_ids: Union[int, List[int], str] = 0,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        vLLM-based inference engine. Uses prefix caching for efficient step-by-step decoding.
        """
        super().__init__(model_name, **kwargs)

        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

        # Limit GPU visibility for vLLM engine init, then restore process env
        # so other CUDA runtimes in this process are not remapped.
        cuda_devices = (
            ",".join(map(str, cuda_ids))
            if isinstance(cuda_ids, (list, tuple))
            else str(cuda_ids)
        )
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        if isinstance(cuda_ids, (list, tuple)):
            tp_size = max(1, len(cuda_ids))
        elif isinstance(cuda_ids, str) and cuda_ids not in ("", "cpu"):
            tp_size = max(1, len([x for x in cuda_ids.split(",") if x.strip()]))
        else:
            tp_size = 1

        cfg_top_k = int(kwargs.pop("top_k", 0) or 0)
        enable_topk_optimization = bool(
            kwargs.pop("enable_topk_optimization", kwargs.pop("enable_topp_optimization", True))
        )
        if enable_topk_optimization:
            logprobs_topk = max(cfg_top_k, 512) if cfg_top_k > 0 else 512
        else:
            # Precision-first mode: set to full vocab after model init.
            logprobs_topk = -1
        self.logprobs_topk = int(logprobs_topk)

        # 1. Init vLLM (enable_prefix_caching for efficient forward_step reuse)
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=True,
                trust_remote_code=True,
                **kwargs
            )
        finally:
            if prev_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
        
        # 2. Tokenizer and config (vocab size for logits tensor)
        self.tokenizer = self.llm.get_tokenizer()
        config = self.llm.llm_engine.get_model_config()
        self.vocab_size = config.get_vocab_size()
        if self.logprobs_topk <= 0:
            self.logprobs_topk = int(self.vocab_size)
        
    @property
    def device(self) -> torch.device:
        # vLLM owns GPUs; return cuda:0 as logical device
        return torch.device("cuda:0")

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def get_next_token_logits(self, batch_ids: List[List[int]]) -> torch.Tensor:
        """
        [Expander] Return dense tensor built from top-k log-probabilities
        at the next generated position (non-topk entries are -1e9).
        """
        from vllm import SamplingParams

        # vLLM does not return logits by default; request via logprobs
        # Request top-k logprobs and materialize sparse dense tensor for compatibility.
        sampling_params = SamplingParams(
            max_tokens=1, 
            logprobs=self.logprobs_topk,
            temperature=0
        )

        # use_tqdm=False to reduce log noise
        outputs = self.llm.generate(
            prompt_token_ids=batch_ids, 
            sampling_params=sampling_params, 
            use_tqdm=False
        )

        # Convert List[RequestOutput] to tensor
        logits_list = []
        for output in outputs:
            # Use first generated position's top-k logprobs.
            last_step_logprobs = output.outputs[0].logprobs[0]
            step_tensor = torch.full((self.vocab_size,), -1e9, device=self.device, dtype=torch.float16)
            if last_step_logprobs:
                ids = torch.tensor(list(last_step_logprobs.keys()), device=self.device)
                vals = torch.tensor(list(last_step_logprobs.values()), device=self.device, dtype=torch.float16)
                step_tensor[ids] = vals
            
            logits_list.append(step_tensor)

        return torch.stack(logits_list)

    def forward_step(
        self, 
        input_ids: torch.Tensor, 
        kv_cache: Optional[List[List[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        [Sampler] Single-step decode with vLLM.
        kv_cache is full history token IDs (List[List[int]]), not tensors; vLLM uses prefix caching.
        Returns (dense top-k log-prob tensor, updated_full_history_ids).
        """
        from vllm import SamplingParams
        # NOTE: vLLM does NOT consume attention_mask. For prefill, we must strip padding
        # tokens ourselves (Sampler uses left padding).
        
        # 1. Build full context (vLLM needs full history for prefix cache)
        if kv_cache is None:
            # Prefill: build history once.
            input_ids_list = input_ids.tolist()
            if attention_mask is not None:
                mask_list = attention_mask.tolist()
                full_sequences = []
                for seq, mseq in zip(input_ids_list, mask_list):
                    # Keep only non-padding tokens (Sampler uses left padding).
                    first_real = 0
                    while first_real < len(mseq) and mseq[first_real] == 0:
                        first_real += 1
                    full_sequences.append(seq[first_real:])
            else:
                full_sequences = input_ids_list
        else:
            # Decode: copy-on-write then append new token per row.
            new_token_ids = input_ids.squeeze(-1).tolist()
            full_sequences = [history.copy() for history in kv_cache]
            for history, tid in zip(full_sequences, new_token_ids):
                history.append(int(tid))

        # 2. Params: 1 step, need logits
        sampling_params = SamplingParams(
            max_tokens=1, 
            logprobs=self.logprobs_topk,
            temperature=0
        )

        # 3. vLLM with prefix caching only computes the last position
        outputs = self.llm.generate(
            prompt_token_ids=full_sequences,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # 4. Extract logits (first generated position logprobs)
        logits_list = []
        for output in outputs:
            first_gen_logprobs = output.outputs[0].logprobs[0]
            step_tensor = torch.full((self.vocab_size,), -1e9, device=self.device, dtype=torch.float16)
            if first_gen_logprobs:
                ids = torch.tensor(list(first_gen_logprobs.keys()), device=self.device)
                vals = torch.tensor(list(first_gen_logprobs.values()), device=self.device, dtype=torch.float16)
                step_tensor[ids] = vals
            
            logits_list.append(step_tensor)

        logits = torch.stack(logits_list)
        
        # 5. Return logits and updated full history as "cache"
        return logits, full_sequences


    def generate(
        self, 
        token_ids: List[int], 
        max_new_tokens: int = 50, 
        temperature: float = 1.0, 
        top_p: float = 0.9,
        **kwargs
    ) -> List[int]:
        """
        Single-sequence generate; delegates to batch_generate.
        """
        # Wrap as batch
        batch_input = [token_ids]
        
        # Call batch
        batch_output = self.batch_generate(
            batch_input, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return batch_output[0]

    def batch_generate(
        self, 
        batch_ids: List[List[int]], 
        max_new_tokens: int = 50, 
        temperature: float = 1.0, 
        top_p: float = 0.9,
        **kwargs
    ) -> List[List[int]]:
        """
        Batch generate using vLLM high-throughput API.
        """
        from vllm import SamplingParams

        # 1. Sampling params
        if temperature < 1e-5:
            params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0, 
                top_p=1.0,
            )
        else:
            params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # 2. Run (vLLM handles padding); use_tqdm=False
        outputs = self.llm.generate(
            prompt_token_ids=batch_ids,
            sampling_params=params,
            use_tqdm=False
        )

        # 3. Concat prompt + generated to match HF interface
        final_results = []
        for output in outputs:
            prompt_ids = output.prompt_token_ids
            generated_ids = output.outputs[0].token_ids
            final_results.append(prompt_ids + generated_ids)

        return final_results
