from abc import ABC, abstractmethod
from typing import List, Union, Optional, Tuple, Any
import torch
# shuyi: this file has not been viewed
class TargetModelEngineBase(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.tokenizer = None

    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return tokenizer for encode/decode and eos_token_id."""
        pass

    @abstractmethod
    def get_next_token_logits(self, batch_ids: List[List[int]]) -> torch.Tensor:
        """Stateless: full path in, logits for last token out."""
        pass

    @abstractmethod
    def forward_step(
        self, 
        input_ids: torch.Tensor, 
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        [Sampler] Single-step forward (Prefill or Decode). Returns (logits, updated kv_cache).
        """
        pass

    @abstractmethod
    def generate(
        self, 
        token_ids: List[int], 
        max_new_tokens: int = 50, 
        temperature: float = 1.0, 
        top_p: float = 0.9,
        **kwargs
    ) -> List[int]:
        """Single-sequence generate."""
        pass

    @abstractmethod
    def batch_generate(
        self, 
        batch_ids: List[List[int]], 
        max_new_tokens: int = 50, 
        temperature: float = 1.0, 
        top_p: float = 0.9,
        **kwargs
    ) -> List[List[int]]:
        """Batch generate."""
        pass