from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CacheQueryResult:
    """Result of a cache query with sampling decision."""

    cached_scores: List[float]
    cached_items: List[Tuple[List[int], float, bool, Optional[float], int]]
    needed_count: int
    hit_type: str  # "full" | "partial" | "miss"


class BaseCacheAlgo(ABC):
    """Base class for cache query algorithms."""

    @abstractmethod
    def query(
        self,
        cache,
        prefix_ids: list,
        *,
        target_count: int,
        min_suffix_tokens: int = 0,
    ) -> CacheQueryResult:
        ...


class TauThresholdCacheAlgo(BaseCacheAlgo):
    """Cache algo using hit_count to decide if enough samples are cached.

    Logic:
      1. Look up cached items matching the prefix.
      2. Sum hit_counts across all unique sequences → effective sample count.
      3. If effective count >= target_count → full hit, no more sampling needed.
         Otherwise → partial hit, sample (target_count - effective_count) more.
    """

    def query(
        self,
        cache,
        prefix_ids: list,
        *,
        target_count: int,
        min_suffix_tokens: int = 0,
    ) -> CacheQueryResult:
        items = (
            cache.get(
                prefix_ids,
                top_k=target_count,
                min_suffix_tokens=min_suffix_tokens,
            )
            or []
        )

        if not items:
            return CacheQueryResult([], [], target_count, "miss")

        # item[4] = hit_count: total samples behind each unique sequence
        effective_m = sum(item[4] for item in items)

        # Build cached_scores: repeat each score by its hit_count, cap at target_count
        # Sort descending so high scores are kept when truncating
        cached_scores = []
        for item in items:
            cached_scores.extend([item[1]] * item[4])
        cached_scores.sort(reverse=True)
        cached_scores = cached_scores[:target_count]

        needed = 0 if effective_m >= target_count else target_count - effective_m
        hit_type = "full" if needed == 0 else "partial"
        return CacheQueryResult(cached_scores, items, needed, hit_type)
