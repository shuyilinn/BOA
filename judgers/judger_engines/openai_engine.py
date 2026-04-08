import os
from openai import OpenAI
from typing import List, Optional
from .base_engine import JudgerEngineBase

class OpenAIJudgerEngine(JudgerEngineBase):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        # Prefer the provided key; otherwise read from environment variables.
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def batch_generate_content(self, texts: List[str]) -> List[str]:
        responses = []
        for text in texts:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.0,  # For judging tasks, set to 0 for stability.
                    max_tokens=500
                )
                responses.append(chat_completion.choices[0].message.content)
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                responses.append("")
        return responses