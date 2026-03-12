import torch
from typing import List, Union, Tuple, Optional, Any


def _ensure_awq_transformers_compat() -> None:
    """
    AutoAWQ expects transformers.activations.PytorchGELUTanh.
    Newer transformers versions expose GELUTanh instead.
    """
    try:
        import transformers.activations as activations
    except Exception:
        return

    if hasattr(activations, "PytorchGELUTanh"):
        return
    if hasattr(activations, "GELUTanh"):
        activations.PytorchGELUTanh = activations.GELUTanh


_ensure_awq_transformers_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer
from engines.base_engine import TargetModelEngineBase
# [shuyi: this file has not been reviewed yet]

class HuggingFaceTargetModelEngine(TargetModelEngineBase):
    def __init__(
        self, 
        model_name: str, 
        cuda_ids: Union[int, List[int], str] = 0, 
        **kwargs
    ):
        """
        HuggingFace Transformers inference engine.
        """
        super().__init__(model_name, **kwargs)
        
        # 1. Device: device_map="auto" for multi-GPU or accelerate
        self._device_map = None
        if isinstance(cuda_ids, (list, str)) and cuda_ids != "cpu":
             self._device_map = "auto"
        
        # Single-GPU manual .to(device)
        self._target_device = self._setup_device(cuda_ids)

        # 2. Tokenizer (padding)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Many models lack pad_token; set to eos for batching
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Left padding for causal LM position alignment
        self.tokenizer.padding_side = "left"

        # 3. Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self._device_map,
            trust_remote_code=True,
            **kwargs
        )
        
        # If not auto, move to target device
        if self._device_map is None and self._target_device.type != 'cpu':
            self.model.to(self._target_device)
            
        self.model.eval()

    def _setup_device(self, cuda_ids) -> torch.device:
        """Parse device argument."""
        if cuda_ids == "cpu": 
            return torch.device("cpu")
        if isinstance(cuda_ids, int): 
            return torch.device(f"cuda:{cuda_ids}")
        # For list or "auto", return cuda:0 as default main device
        return torch.device("cuda:0")

    @property
    def device(self) -> torch.device:
        """Main device of the model."""
        return self.model.device

    def get_tokenizer(self) -> Any:
        """Return tokenizer for encode/decode and eos_token_id."""
        return self.tokenizer

    def get_next_token_logits(self, batch_ids: List[List[int]]) -> torch.Tensor:
        """
        [Expander] Logits for last token of each sequence. Stateless, no KV cache.
        """
        # 1. Pad and mask via tokenizer (List[List[int]] in)
        encodings = self.tokenizer.pad(
            {"input_ids": batch_ids}, 
            padding=True, 
            return_tensors="pt"
        )
        
        # To model device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # 2. Forward (use_cache=False to save memory)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=False 
            )
        
        # 3. Logits at last position (-1 with left padding)
        # return shape: [batch_size, vocab_size]
        return outputs.logits[:, -1, :]

    def forward_step(
        self, 
        input_ids: torch.Tensor, 
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        [Sampler] Single-step forward. input_ids: Prefill [B,L] or Decode [B,1].
        kv_cache may be new HF Cache objects or legacy tuple past_key_values.
        """
        # Ensure input on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        # Prefill: pass attention_mask; Decode: HF handles causal mask
        
        model_kwargs = {
            "input_ids": input_ids,
            "past_key_values": kv_cache,
            "use_cache": True,
        }
        # Only apply attention_mask on prefill; decode stage usually doesn't need it.
        if kv_cache is None and attention_mask is not None:
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(self.device)
            model_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            outputs = self.model(**model_kwargs)
        
        # logits: [B, L, V] or [B, 1, V]; take [:, -1, :]
        next_token_logits = outputs.logits[:, -1, :]
        
        return next_token_logits, outputs.past_key_values

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
        top_k: int = 0,
        **kwargs
    ) -> List[List[int]]:
        
        # 1. Padding
        encodings = self.tokenizer.pad(
            {"input_ids": batch_ids}, 
            padding=True, 
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # 2. Gen params
        do_sample = temperature > 1e-5
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            
        # Forward extra kwargs
        gen_kwargs.update(kwargs)

        # 3. Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                **gen_kwargs
            )

        return output_ids.tolist()
