from abc import ABC, abstractmethod
from typing import List, Union


class JudgerEngineBase(ABC):
    """
    Abstract base class for local inference engines.
    Unified contract: input List[str], output List[str].
    """

    @abstractmethod
    def __init__(self, name: str, cuda_number: Union[int, List[int]], **kwargs):
        """Initialize the model/tokenizer and allocate GPU resources."""
        raise NotImplementedError

    @abstractmethod
    def batch_generate_content(self, texts: List[str]) -> List[str]:
        """Unified interface for batched text generation."""
        raise NotImplementedError

    def generate_content(self, text: str) -> str:
        """Convenience wrapper for single-text generation."""
        results = self.batch_generate_content([text])
        return results[0] if results else ""
