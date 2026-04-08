"""Tests for EOS bypass of min_suffix_tokens in SequenceCache.get()."""

import pytest
from components.cache.cache import SequenceCache


def test_eos_complete_bypasses_min_suffix_all():
    """An EOS-complete terminal shorter than min_suffix_tokens must be returned."""
    cache = SequenceCache()
    # Total sequence length = 12, prefix = 10, suffix = 2 < min_suffix_tokens=50
    seq = list(range(12))
    cache.add(seq, score=0.9, complete=True)

    results = cache.get(seq[:10], min_suffix_tokens=50)
    assert len(results) == 1
    seq_out, score_out, is_complete_out, tau_out, hit_count_out = results[0]
    assert seq_out == seq
    assert is_complete_out is True


def test_incomplete_short_sequence_not_returned():
    """A non-EOS terminal shorter than min_suffix_tokens must NOT be returned."""
    cache = SequenceCache()
    seq = list(range(12))
    cache.add(seq, score=0.9, complete=False)

    results = cache.get(seq[:10], min_suffix_tokens=50)
    assert len(results) == 0


def test_eos_complete_bypasses_min_suffix_top_k():
    """Same bypass applies when using top_k path."""
    cache = SequenceCache()
    seq = list(range(12))
    cache.add(seq, score=0.9, complete=True)

    results = cache.get(seq[:10], top_k=5, min_suffix_tokens=50)
    assert len(results) == 1
    seq_out, score_out, is_complete_out, tau_out, hit_count_out = results[0]
    assert seq_out == seq
    assert is_complete_out is True


def test_eos_zero_suffix_prefix_equals_full_sequence():
    """An EOS sequence whose length exactly equals the prefix length (suffix=0) must be returned."""
    cache = SequenceCache()
    seq = list(range(10))
    cache.add(seq, score=0.7, complete=True)

    results = cache.get(seq, min_suffix_tokens=50)
    assert len(results) == 1
    assert results[0][2] is True  # is_complete


def test_non_eos_long_sequence_is_complete_false():
    """A non-EOS sequence meeting min_suffix returns is_complete=False."""
    cache = SequenceCache()
    prefix = list(range(5))
    seq = prefix + list(range(100, 160))  # suffix_len=60 >= 50
    cache.add(seq, score=0.5, complete=False)

    results = cache.get(prefix, min_suffix_tokens=50)
    assert len(results) == 1
    assert results[0][2] is False  # is_complete


def test_top_k_score_filter_still_applies_to_eos():
    """EOS nodes bypass length filter but still compete by score for top_k selection."""
    cache = SequenceCache()
    prefix = list(range(5))

    # Two non-EOS long sequences (meet suffix requirement)
    for i in range(10):
        seq = prefix + [100 + i] * 50
        cache.add(seq, score=float(i + 1), complete=False)

    # One EOS short sequence with low score
    eos_seq = prefix + [999]
    cache.add(eos_seq, score=0.1, complete=True)

    results = cache.get(prefix, top_k=3, min_suffix_tokens=50)
    returned_seqs = [r[0] for r in results]

    # EOS seq has score 0.1, well below top-3 scores (10, 9, 8)
    assert eos_seq not in returned_seqs
    assert len(results) == 3
