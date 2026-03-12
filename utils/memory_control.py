from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict, Literal
import torch

Framework = Literal["hf", "vllm", "trtllm", "generic"]


@dataclass
class ModelMemInfo:
    framework: Framework
    num_layers: int
    num_heads: int
    hidden_size: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    model_dtype: Optional[torch.dtype] = None   # for inferring KV dtype default
    kv_dtype: Optional[torch.dtype] = None      # set if you know KV precision
    meta: Optional[Dict[str, Any]] = None


def extract_mem_info_from_config(
    config: Any,
    *,
    framework: Framework = "hf",
    model_dtype: Optional[torch.dtype] = None,
    kv_dtype: Optional[torch.dtype] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> ModelMemInfo:
    """
    Map config -> ModelMemInfo. Works with HF config, vLLM hf_config, or any config with same attribute names.
    """
    num_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)

    num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else None
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)

    head_dim = config.head_dim if hasattr(config, "head_dim") else None
    if head_dim is not None:
        head_dim = int(head_dim)

    # Default head_dim if not in config
    if head_dim is None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Invalid config: hidden_size({hidden_size}) not divisible by num_heads({num_heads})"
            )
        head_dim = hidden_size // num_heads

    return ModelMemInfo(
        framework=framework,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_dtype=model_dtype,
        kv_dtype=kv_dtype,
        meta=meta or {},
    )


def extract_mem_info_from_engine(
    engine: Any,
    *,
    framework: Framework = "generic",
    config: Any = None,
    model: Any = None,
    kv_dtype: Optional[torch.dtype] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> ModelMemInfo:
    """
    Prefer engine.model.config, then engine.config, then explicitly passed config.
    """
    # 1) Resolve config
    cfg = None
    if config is not None:
        cfg = config
    elif model is not None and hasattr(model, "config"):
        cfg = model.config
    elif hasattr(engine, "model") and getattr(engine, "model") is not None and hasattr(engine.model, "config"):
        cfg = engine.model.config
    elif (
        hasattr(engine, "llm")
        and getattr(engine, "llm") is not None
        and hasattr(engine.llm, "llm_engine")
        and hasattr(engine.llm.llm_engine, "get_model_config")
    ):
        model_cfg = engine.llm.llm_engine.get_model_config()
        if hasattr(model_cfg, "hf_config") and getattr(model_cfg, "hf_config") is not None:
            cfg = model_cfg.hf_config
        else:
            cfg = model_cfg
    elif hasattr(engine, "config"):
        cfg = engine.config

    if cfg is None:
        raise ValueError("No config: pass config or ensure engine/model has .config")

    # 2) Resolve dtype
    md = None
    if model is not None and hasattr(model, "dtype"):
        md = model.dtype
    elif hasattr(engine, "model") and getattr(engine, "model", None) is not None and hasattr(engine.model, "dtype"):
        md = engine.model.dtype

    return extract_mem_info_from_config(
        cfg,
        framework=framework,
        model_dtype=md,
        kv_dtype=kv_dtype,
        meta=meta,
    )


def _bytes_per_element(dtype: Optional[torch.dtype]) -> int:
    """Infer bytes per element from dtype; default 2 (fp16/bf16)."""
    if dtype is None:
        return 2
    mapping = {
        torch.float64: 8,
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return int(mapping.get(dtype, 2))


def estimate_batch_size_from_current_free_vram(
    engine: Any,
    prompt_len: int,
    max_new_tokens: int,
    *,
    framework: Framework = "hf",
    gpu_memory_utilization: float = 0.9,
    device_index: Optional[int] = None,
    overhead_ratio: float = 1.1,
    max_batch_cap: int = 4096,
    min_batch_size: int = 1,
    model: Any = None,
    config: Any = None,
    kv_dtype: Optional[torch.dtype] = None,
) -> Optional[int]:
    """
    Estimate max safe batch size from *current free VRAM* and model structure.
    Assumes model weights are already loaded and estimates incremental per-sample memory.
    HF eager inference only; returns None for other frameworks.
    Returns None if estimation fails.
    """
    if not torch.cuda.is_available():
        return None
    if framework != "hf":
        return None
    info = extract_mem_info_from_engine(
        engine,
        framework=framework,
        model=model,
        config=config,
        kv_dtype=kv_dtype,
    )

    if device_index is None:
        d = getattr(engine, "device", None)
        if getattr(d, "index", None) is not None:
            device_index = d.index
        else:
            # Avoid touching default CUDA device (often cuda:0) as a fallback,
            # which can create an unnecessary CUDA context on the wrong GPU.
            return None

    free_bytes, _ = torch.cuda.mem_get_info(device_index)

    util = max(0.05, min(float(gpu_memory_utilization), 0.99))
    bytes_available = int(free_bytes * util)

    model_obj = model if model is not None else getattr(engine, "model", None)
    act_dtype = getattr(model_obj, "dtype", None) if model_obj is not None else info.model_dtype
    act_bytes_per_el = _bytes_per_element(act_dtype)
    kv_bytes_per_el = _bytes_per_element(info.kv_dtype or info.model_dtype)

    num_layers = info.num_layers
    hidden_size = info.hidden_size
    num_kv = info.num_kv_heads if info.num_kv_heads is not None else info.num_heads
    if info.head_dim is not None:
        head_dim = int(info.head_dim)
    else:
        if hidden_size % info.num_heads != 0:
            return None
        head_dim = hidden_size // info.num_heads
    seq_len = int(prompt_len) + int(max_new_tokens)
    if seq_len <= 0:
        return None

    # Throughput-oriented heuristic for inference:
    # KV cache dominates incremental memory; activation is a small fraction and
    # most framework/runtime overhead is handled by overhead_ratio.
    kv_per_sample = 2 * num_layers * seq_len * num_kv * head_dim * kv_bytes_per_el
    act_bytes = int(0.03 * kv_per_sample)
    per_sample = int((kv_per_sample + act_bytes) * overhead_ratio)
    if per_sample <= 0:
        return None

    est = int(bytes_available // per_sample)
    est = max(min_batch_size, min(max_batch_cap, est))
    return est


def estimate_max_batch_size(
    engine: Any,
    prompt_len: int,
    max_new_tokens: int,
    *,
    gpu_memory_utilization: float = 0.9,
    device_index: Optional[int] = None,
    overhead_ratio: float = 1.1,
    max_batch_cap: int = 4096,
    min_batch_size: int = 1,
    model: Any = None,
    config: Any = None,
    kv_dtype: Optional[torch.dtype] = None,
) -> Optional[int]:
    """
    Backward-compatible alias for estimate_batch_size_from_current_free_vram().
    """
    return estimate_batch_size_from_current_free_vram(
        engine,
        prompt_len,
        max_new_tokens,
        framework="hf",
        gpu_memory_utilization=gpu_memory_utilization,
        device_index=device_index,
        overhead_ratio=overhead_ratio,
        max_batch_cap=max_batch_cap,
        min_batch_size=min_batch_size,
        model=model,
        config=config,
        kv_dtype=kv_dtype,
    )


def estimate_max_batch_size_from_config(
    config: Any,
    prompt_len: int,
    max_new_tokens: int,
    *,
    engine: Any = None,
    gpu_memory_utilization: Optional[float] = None,
    overhead_ratio: Optional[float] = None,
    max_batch_cap: Optional[int] = None,
    min_batch_size: Optional[int] = None,
) -> Optional[int]:
    """
    Read params from config and call estimate_max_batch_size. Returns None if engine not available.
    """
    if engine is None:
        return None
    util = gpu_memory_utilization if gpu_memory_utilization is not None else config.target_gpu_memory_utilization
    overhead = overhead_ratio if overhead_ratio is not None else config.hf_batch_estimate_overhead
    cap = max_batch_cap if max_batch_cap is not None else config.max_batch_size
    min_bs = min_batch_size if min_batch_size is not None else config.min_batch_size
    return estimate_max_batch_size(
        engine,
        prompt_len,
        max_new_tokens,
        gpu_memory_utilization=float(util),
        overhead_ratio=float(overhead),
        max_batch_cap=int(cap),
        min_batch_size=int(min_bs),
    )
