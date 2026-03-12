from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unknown"


def _fmt_meta(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_run_id(cfg: Any, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    workload_name = _sanitize_for_filename(str(cfg.workload_name))
    model_name = _sanitize_for_filename(str(cfg.target_model).split("/")[-1])
    engine_name = _sanitize_for_filename(str(cfg.target_engine_name))
    top_p = _sanitize_for_filename(_fmt_meta(cfg.top_p))
    top_k = _sanitize_for_filename(_fmt_meta(cfg.top_k))
    likelihood = _sanitize_for_filename(_fmt_meta(cfg.likelihood))
    temperature = _sanitize_for_filename(_fmt_meta(cfg.temperature))
    chunk_size = _sanitize_for_filename(_fmt_meta(cfg.chunk_size))

    cache = "on" if bool(cfg.enable_sampling_cache) else "off"
    sbuf = "on" if bool(cfg.enable_sampling_buffer) else "off"
    jbuf = "on" if bool(cfg.enable_judging_buffer) else "off"
    topk_opt = "on" if bool(cfg.enable_topk_optimization) else "off"
    ref_pat = "on" if bool(cfg.enable_refuse_pattern_matching) else "off"
    ref_judge = "on" if bool(cfg.enable_refuse_judger) else "off"
    flags = f"cache{cache}_sbuf{sbuf}_jbuf{jbuf}_topk{topk_opt}_refPat{ref_pat}_refJudge{ref_judge}"

    return (
        f"{ts}_{workload_name}_{model_name}_{engine_name}"
        f"_p{top_p}_k{top_k}_t{temperature}_lh{likelihood}_cs{chunk_size}_{flags}"
    )


def build_log_file_path(cfg: Any, run_id: str | None = None, logs_root: str = "logs") -> str:
    rid = run_id or build_run_id(cfg)
    return f"{logs_root}/{rid}.log"


def build_result_dir(cfg: Any, run_id: str | None = None, results_root: str = "results") -> str:
    rid = run_id or build_run_id(cfg)
    return f"{results_root}/{rid}"
