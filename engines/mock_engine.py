from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from engines.base_engine import TargetModelEngineBase


class _MockEmbeddings:
    def __init__(self, num_embeddings: int):
        self.num_embeddings = int(num_embeddings)


class _MockModelConfig:
    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)


class _MockGenerationConfig:
    def __init__(self, eos_token_id: int):
        self.eos_token_id = int(eos_token_id)


class _MockModel:
    def __init__(self, vocab_size: int, eos_token_id: int):
        self.config = _MockModelConfig(vocab_size)
        self.generation_config = _MockGenerationConfig(eos_token_id)
        self._embeddings = _MockEmbeddings(vocab_size)

    def get_input_embeddings(self) -> _MockEmbeddings:
        return self._embeddings


class MockTokenizer:
    """Minimal tokenizer for CPU-only tests."""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.vocab_size = 258
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        data = text.encode("utf-8", errors="ignore")
        return [int(b) + 2 for b in data]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        data: List[int] = []
        for tid in ids:
            if skip_special_tokens and tid in (self.eos_token_id, self.pad_token_id):
                continue
            if tid >= 2:
                data.append((int(tid) - 2) % 256)
        return bytes(data).decode("utf-8", errors="ignore")

    def batch_decode(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def pad(self, encoded_inputs: Dict[str, List[List[int]]], padding: bool = True, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        batch_ids = encoded_inputs["input_ids"]
        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
        padded: List[List[int]] = []
        masks: List[List[int]] = []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded.append([self.pad_token_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))
        if return_tensors != "pt":
            raise ValueError("MockTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def apply_chat_template(self, chat: List[Dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True, **kwargs):
        parts: List[str] = []
        for msg in chat:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                text_items = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_items.append(str(item.get("text", "")))
                content = "".join(text_items)
            parts.append(f"{role}: {content}")
        rendered = "\n".join(parts)
        if add_generation_prompt:
            rendered += "\nassistant: "
        if tokenize:
            return self.encode(rendered)
        return rendered


class MockTargetModelEngine(TargetModelEngineBase):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self._device = torch.device("cpu")
        self.tokenizer = MockTokenizer()
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.model = _MockModel(self.vocab_size, self.tokenizer.eos_token_id)
        self.mock_text = str(kwargs.get("mock_target_text", "OK"))
        self.mock_logit_high = float(kwargs.get("mock_target_logit_high", 8.0))
        self.mock_logit_low = float(kwargs.get("mock_target_logit_low", 0.0))
        self.mock_noise_std = float(kwargs.get("mock_target_noise_std", 0.0))
        self.mock_sampler_sleep_sec = max(0.0, float(kwargs.get("mock_sampler_sleep_sec", 0.0) or 0.0))
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(kwargs.get("mock_target_seed", 2026)))
        encoded = self.tokenizer.encode(self.mock_text)
        self._pattern_ids = encoded if encoded else [ord("O") + 2]
        # Restrict stochastic sampling to printable ASCII to keep mock outputs text-safe.
        self._allowed_token_ids = set(range(32 + 2, 127 + 2))
        self._allowed_token_ids.update(self._pattern_ids)
        self._disallowed_token_ids = [tid for tid in range(self.vocab_size) if tid not in self._allowed_token_ids]
        self._fallback_token = self._pattern_ids[0]

    @property
    def device(self):
        return self._device

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def _build_logits(self, batch_size: int, next_token_id: int) -> torch.Tensor:
        logits = torch.full((batch_size, self.vocab_size), self.mock_logit_low, dtype=torch.float32, device=self.device)
        if self._disallowed_token_ids:
            logits[:, self._disallowed_token_ids] = -1e9
        if self.mock_noise_std > 0:
            noise = torch.randn((batch_size, self.vocab_size), generator=self._rng, device=self.device) * self.mock_noise_std
            logits = logits + noise
        logits[:, int(next_token_id)] = self.mock_logit_high
        return logits

    def _next_token_for_sequence(self, seq: List[int]) -> int:
        idx = len(seq) % len(self._pattern_ids)
        return int(self._pattern_ids[idx])

    def get_next_token_logits(self, batch_ids: List[List[int]]) -> torch.Tensor:
        next_ids = [self._next_token_for_sequence(ids) if ids else self._fallback_token for ids in batch_ids]
        logits = torch.full((len(batch_ids), self.vocab_size), self.mock_logit_low, dtype=torch.float32, device=self.device)
        if self._disallowed_token_ids:
            logits[:, self._disallowed_token_ids] = -1e9
        if self.mock_noise_std > 0:
            noise = torch.randn((len(batch_ids), self.vocab_size), generator=self._rng, device=self.device) * self.mock_noise_std
            logits = logits + noise
        for row, tid in enumerate(next_ids):
            logits[row, tid] = self.mock_logit_high
        return logits

    def forward_step(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        # Uniform sampling path calls forward_step repeatedly; sleep only on prefill.
        if kv_cache is None and self.mock_sampler_sleep_sec > 0.0:
            time.sleep(self.mock_sampler_sleep_sec)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size = int(input_ids.size(0))
        logits = self._build_logits(batch_size, self._fallback_token)
        return logits, None

    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[int]:
        return self.batch_generate(
            [token_ids],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )[0]

    def batch_generate(
        self,
        batch_ids: List[List[int]],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[List[int]]:
        if self.mock_sampler_sleep_sec > 0.0:
            time.sleep(self.mock_sampler_sleep_sec)
        outputs: List[List[int]] = []
        if max_new_tokens <= 0:
            return [list(ids) for ids in batch_ids]
        for base in batch_ids:
            new_tokens = [self._pattern_ids[i % len(self._pattern_ids)] for i in range(max_new_tokens)]
            outputs.append(list(base) + new_tokens)
        return outputs
