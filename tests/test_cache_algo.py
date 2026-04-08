"""Tests for the TauThresholdCacheAlgo cache query algorithm."""

from __future__ import annotations

import math

from components.cache.cache import SequenceCache
from components.cache.algo import TauThresholdCacheAlgo


def _make_cache_with_entries(entries):
    """Helper: build a SequenceCache and add entries.

    Each entry is (seq_ids, score, complete, tau).
    """
    cache = SequenceCache()
    for seq_ids, score, complete, tau in entries:
        cache.add(seq_ids, score, complete=complete, tau=tau)
    return cache


class TestTauThresholdCacheAlgo:
    def test_miss_empty_cache(self):
        cache = SequenceCache()
        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2, 3], target_count=4)

        assert result.hit_type == "miss"
        assert result.needed_count == 4
        assert result.cached_scores == []
        assert result.cached_items == []

    def test_miss_no_matching_prefix(self):
        cache = _make_cache_with_entries([
            ([10, 20, 30], 0.8, True, math.log(0.5)),
        ])
        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [99, 99], target_count=2)

        assert result.hit_type == "miss"
        assert result.needed_count == 2

    def test_full_hit_count_sufficient(self):
        """3 unique entries, each hit_count=1, target=3 → full hit."""
        cache = _make_cache_with_entries([
            ([1, 2, 10], 0.9, True, math.log(0.1)),
            ([1, 2, 20], 0.8, True, math.log(0.1)),
            ([1, 2, 30], 0.7, False, math.log(0.1)),
        ])
        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=3, min_suffix_tokens=1)

        assert result.hit_type == "full"
        assert result.needed_count == 0
        assert len(result.cached_scores) == 3

    def test_partial_hit_insufficient_count(self):
        """2 unique entries (effective_m=2), target=5 → partial, need 3 more."""
        cache = _make_cache_with_entries([
            ([1, 2, 10], 0.9, True, math.log(0.2)),
            ([1, 2, 20], 0.8, False, math.log(0.2)),
        ])
        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=5, min_suffix_tokens=1)

        assert result.hit_type == "partial"
        assert result.needed_count == 3  # 5 - 2
        assert len(result.cached_scores) == 2

    def test_scores_match_items(self):
        """cached_scores should match the score field in cached_items."""
        cache = _make_cache_with_entries([
            ([1, 2, 10], 0.9, True, math.log(0.3)),
            ([1, 2, 20], 0.7, False, math.log(0.3)),
        ])
        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=5, min_suffix_tokens=1)

        for score, item in zip(result.cached_scores, result.cached_items):
            assert score == item[1]

    def test_duplicate_adds_full_hit(self):
        """Same seq_ids added 9 times → effective_m=9 → full hit even with 1 unique entry."""
        cache = SequenceCache()
        seq = [1, 2, 10]
        for i in range(9):
            cache.add(seq, score=300.0 + i * 10, complete=True, tau=math.log(0.3))

        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=9, min_suffix_tokens=1)

        assert result.hit_type == "full"
        assert result.needed_count == 0
        assert len(result.cached_scores) == 9
        # Score should be the running average: (300+310+...+380)/9 = 340
        assert abs(result.cached_scores[0] - 340.0) < 0.01

    def test_duplicate_adds_partial_hit(self):
        """Same seq_ids added 3 times, target=9 → effective_m=3, need 6 more."""
        cache = SequenceCache()
        for _ in range(3):
            cache.add([1, 2, 10], score=400.0, complete=True, tau=math.log(0.1))

        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=9, min_suffix_tokens=1)

        assert result.hit_type == "partial"
        assert result.needed_count == 6  # 9 - 3
        assert len(result.cached_scores) == 3

    def test_mixed_unique_and_duplicate(self):
        """2 unique seqs, one added 5 times, one added 3 times → effective_m=8."""
        cache = SequenceCache()
        for _ in range(5):
            cache.add([1, 2, 10], score=400.0, complete=True, tau=math.log(0.1))
        for _ in range(3):
            cache.add([1, 2, 20], score=350.0, complete=False, tau=math.log(0.1))

        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=8, min_suffix_tokens=1)

        assert result.hit_type == "full"
        assert result.needed_count == 0
        assert len(result.cached_scores) == 8

    def test_cached_scores_capped_at_target_count(self):
        """cached_scores should not exceed target_count even if hit_count is higher."""
        cache = SequenceCache()
        for _ in range(15):
            cache.add([1, 2, 10], score=500.0, complete=True, tau=None)

        algo = TauThresholdCacheAlgo()
        result = algo.query(cache, [1, 2], target_count=9, min_suffix_tokens=1)

        assert result.hit_type == "full"
        assert result.needed_count == 0
        assert len(result.cached_scores) == 9
