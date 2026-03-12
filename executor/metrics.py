from __future__ import annotations

from typing import Any, Dict, List


def cache_stats(*, lookups: int, partial_hits: int, full_hits: int) -> Dict[str, Any]:
    lookups_i = int(lookups or 0)
    partial_i = int(partial_hits or 0)
    full_i = int(full_hits or 0)
    any_hits = partial_i + full_i
    miss = max(0, lookups_i - any_hits)
    full_rate = (100.0 * full_i / lookups_i) if lookups_i > 0 else 0.0
    any_rate = (100.0 * any_hits / lookups_i) if lookups_i > 0 else 0.0
    return {
        "lookups": lookups_i,
        "full_hits": full_i,
        "partial_hits": partial_i,
        "miss": miss,
        "full_hit_rate_pct": full_rate,
        "any_hit_rate_pct": any_rate,
    }


def batch_summary(batch_sizes: List[int]) -> Dict[str, Any]:
    if not batch_sizes:
        return {
            "batches": 0,
            "items": 0,
            "mean_batch_size": 0.0,
            "min_batch_size": 0,
            "max_batch_size": 0,
        }
    return {
        "batches": len(batch_sizes),
        "items": int(sum(batch_sizes)),
        "mean_batch_size": float(sum(batch_sizes) / len(batch_sizes)),
        "min_batch_size": int(min(batch_sizes)),
        "max_batch_size": int(max(batch_sizes)),
    }


def buffer_stats(
    *,
    sample_buffer_capacity: int,
    sample_enqueued_items: int,
    sample_max_queue_size: int,
    sample_flush_batch_sizes: List[int],
    judging_buffer_capacity: int,
    judging_enqueued_items: int,
    judging_max_queue_size: int,
    judging_flush_batch_sizes: List[int],
) -> Dict[str, Any]:
    sample_batch_stats = batch_summary(sample_flush_batch_sizes)
    judging_batch_stats = batch_summary(judging_flush_batch_sizes)
    return {
        "sample_buffer": {
            "capacity": int(sample_buffer_capacity or 0),
            "enqueued_items": int(sample_enqueued_items or 0),
            "max_queue_size": int(sample_max_queue_size or 0),
            **sample_batch_stats,
        },
        "judging_buffer": {
            "capacity": int(judging_buffer_capacity or 0),
            "enqueued_items": int(judging_enqueued_items or 0),
            "max_queue_size": int(judging_max_queue_size or 0),
            **judging_batch_stats,
        },
    }
