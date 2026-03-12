"""
This file is to define the judger engines using transformers as backend.
"""
from typing import List

from .base_engine import JudgerEngineBase
import torch
from utils.logger import setup_logger

logger = setup_logger("HFJudger")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available")


# TODO: which GPU to use?
class HFLocalJudgerEngine(JudgerEngineBase):
    def __init__(self, name, cuda_number, **kwargs):
        self.device = f"cuda:{cuda_number}"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(
            self.device
        )
        self.model.eval()

    def batch_generate_content(self, texts: List[str]) -> List[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=200, temperature=0.1, do_sample=False
            )

        # Strip input prompt from output
        new_tokens = output_ids[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
