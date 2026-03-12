"""
Load benign and harmful prompts. Both loaders return the same shape for consistent usage:
  {"samples": [{"prompt": str, ...}, ...]}
- Benign: each sample has "prompt" only.
- Harmful: each sample has "prompt", "original_prompt", and optional fields (e.g. "id").
Callers use data["samples"] and extract fields as needed.
"""
import json
import os
import logging
from typing import List, Tuple, Dict, Any, Union

from benchmark.agent_data_loader import adapt_agent_safetybench_samples

logger = logging.getLogger("run")


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Library 'datasets' is required. Please install it using `pip install datasets`.") from e
    return load_dataset(*args, **kwargs)

def load_benign_prompts(dataset_name: str = "JBB-Behaviors", random_seed: int = 42) -> Dict[str, Any]:
    """
    Load benign samples. Same return shape as load_harmful_prompts for consistent usage.
    Returns: {"samples": [{"prompt": str}, ...]} (no original_prompt for benign).
    """
    logger.info(f"Loading benign prompts from HF dataset: {dataset_name}")
    prompts: List[str] = []
    if dataset_name == "JBB-Behaviors":
        ds = _load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
        prompts = list(ds["Goal"])
    else:
        logger.warning(f"Dataset name '{dataset_name}' is not strictly 'JBB-Behaviors'. Attempting generic load...")
        try:
            ds = _load_dataset(dataset_name, split="train")
            text_col = next((col for col in ["text", "prompt", "goal", "Goal"] if col in ds.column_names), None)
            if text_col:
                prompts = list(ds[text_col])
            else:
                raise ValueError(f"Could not find text column in {dataset_name}")
        except Exception as e:
            raise ValueError(f"Failed to load benign prompts from {dataset_name}: {e}")

    logger.info(f"Loaded {len(prompts)} benign prompts.")
    return {"samples": [{"prompt": p} for p in prompts]}


def load_harmful_prompts(path: str) -> Dict[str, Any]:
    """
    Load benchmark data (local JSON or Hugging Face Dataset).
    Strict: each sample must have 'original_prompt' and 'prompt'.
    Returns: {"samples": [{"prompt", "original_prompt", "id", ...}, ...]} for caller to use.
    """
    samples = []
    
    # --- 1. Try loading from local file ---
    if os.path.exists(path):
        logger.info(f"Loading harmful benchmark from local file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Support both {samples: [...]} and top-level list
            if isinstance(data, dict):
                samples = data.get("samples", [])
            elif isinstance(data, list):
                samples = data
            else:
                raise ValueError("JSON format must be a dict with 'samples' key or a list.")
                
        except Exception as e:
            logger.error(f"Failed to load local json {path}: {e}")
            raise e 

    # --- 2. Try loading from Hugging Face Hub ---
    else:
        logger.info(f"Path '{path}' not found locally. Attempting to load from Hugging Face Datasets...")
        try:
            ds = _load_dataset(path)
            # Flatten all splits (train, test, etc.)
            if hasattr(ds, 'keys'): 
                for split in ds.keys():
                    logger.info(f"Merging split '{split}' from {path}...")
                    samples.extend(ds[split].to_list())
            else:
                samples = ds.to_list()
                
        except Exception as e:
            logger.error(f"Failed to load Hugging Face dataset '{path}': {e}")
            raise e

    if not samples:
        logger.warning(f"No samples found in {path}")
        return {"samples": []}

    if "agentsafetybench" in path.lower():
        logger.info("Detected Agent-SafetyBench by json filename. Adapting to BOA sample format...")
        return adapt_agent_safetybench_samples(samples)

    # --- 3. Validate and keep full sample dict ---
    validated_samples = []
    for i, s in enumerate(samples):
        if "id" not in s:
            s = {**s, "id": i}
        missing_fields = []
        if "original_prompt" not in s:
            missing_fields.append("original_prompt")
        if "prompt" not in s:
            missing_fields.append("prompt")
        if missing_fields:
            raise ValueError(
                f"Sample ID {s.get('id', i)} in {path} is missing required fields: {missing_fields}"
            )
        validated_samples.append(s)

    logger.info(f"Successfully loaded {len(validated_samples)} harmful samples from {path}")
    return {"samples": validated_samples}
