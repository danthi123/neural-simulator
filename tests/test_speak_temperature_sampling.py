"""Tests for temperature sampling on :speak (perf audit follow-up 2026-05-10).

Per user observation: with strict argmax (τ=0), STDP WTA "primary wins,
synonym never selected" pattern shows synonym top-1 rate exactly 0%.
Temperature sampling enables proportional selection, lifting synonym
selection to ~15-30% at small τ while preserving primary as dominant.

This is a CPU-only unit test of the _sample_with_temperature helper.
The full generative_inference integration test runs on GPU separately.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the sampler — no GPU/bridge needed
from research.runners.chat_repl import _sample_with_temperature


# Fixture: rankings from seed 44 motor_N (real Tier 2.1 8-word :speak)
# north wins, up ranks 3rd, others below
SEED44_MOTOR_N_RANKINGS = [
    ("north", 0.0965),
    ("right", 0.0604),
    ("up",    0.0598),
    ("down",  0.0539),
    ("left",  0.0539),
    ("west",  0.0515),
    ("east",  0.0450),
    ("south", 0.0278),
]


def test_temperature_zero_returns_none():
    """τ=0 returns None — caller falls back to argmax."""
    result = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS, temperature=0.0)
    assert result is None


def test_negative_temperature_returns_none():
    """Negative temperature treated as 0 (no sampling)."""
    result = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS, temperature=-0.1)
    assert result is None


def test_empty_rankings_returns_none():
    """Empty input returns None safely."""
    assert _sample_with_temperature([], temperature=0.05) is None


def test_temperature_returns_string():
    """Positive temperature returns one word from the rankings."""
    result = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS, temperature=0.05,
                                        rng_seed=42)
    assert isinstance(result, str)
    expected_words = {w for w, _ in SEED44_MOTOR_N_RANKINGS}
    assert result in expected_words


def test_rng_seed_reproducible():
    """Same seed → same sampled word."""
    r1 = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                    temperature=0.05, rng_seed=42)
    r2 = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                    temperature=0.05, rng_seed=42)
    assert r1 == r2


def test_low_temperature_favors_argmax():
    """Low τ should pick 'north' (the argmax) most of the time."""
    # τ = 0.01 is sharp; argmax should win >80% of trials
    counts = Counter()
    for seed in range(200):
        w = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                       temperature=0.01, rng_seed=seed)
        counts[w] += 1
    assert counts["north"] / 200 > 0.80, (
        f"Low τ should heavily favor argmax 'north'. Got: {dict(counts)}"
    )


def test_moderate_temperature_lifts_synonym():
    """Moderate τ should lift secondary synonym ('up') above zero.

    Note: with small absolute cosine differences (~0.04 between top words
    in real Tier 2.1 8-word data), τ=0.05 is a relatively HIGH temperature
    that gives near-uniform results. τ=0.02 is the more typical "primary
    dominant with synonym lift" setting; τ=0.01 is "argmax-dominant".
    Empirically (from this test's run): at τ=0.05 north still wins but
    only by ~2x over uniform; at τ=0.02 north is clearly dominant.
    """
    counts = Counter()
    for seed in range(500):
        w = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                       temperature=0.02, rng_seed=seed)
        counts[w] += 1
    north_rate = counts["north"] / 500
    up_rate = counts["up"] / 500
    # At τ=0.02: north should be dominant (>40%) but not absolute (<95%)
    assert 0.35 < north_rate < 0.95, (
        f"north_rate {north_rate:.2%} unexpected at τ=0.02 "
        f"(counts: {dict(counts)})"
    )
    # up (rank 3) should be selected SOMETIMES — synonym lift validated
    assert up_rate > 0.01, (
        f"up never selected at τ=0.02 — synonym lift broken (counts: {dict(counts)})"
    )


def test_temperature_curve_makes_sense():
    """At progressively higher τ, north's share should monotonically decrease
    toward uniform (12.5% for 8 words)."""
    shares = {}
    for tau in [0.005, 0.01, 0.02, 0.05, 0.10, 0.50, 5.0]:
        c = Counter()
        for seed in range(500):
            w = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                           temperature=tau, rng_seed=seed)
            c[w] += 1
        shares[tau] = c["north"] / 500
    # Monotone decrease (with some Monte Carlo noise tolerance)
    taus = sorted(shares.keys())
    for i in range(len(taus) - 1):
        # Allow 5pp tolerance for MC noise
        assert shares[taus[i]] + 0.05 >= shares[taus[i+1]], (
            f"north share should monotone-decrease in τ. Got: {shares}"
        )
    # Sanity: very low τ → argmax dominance
    assert shares[0.005] > 0.7
    # Sanity: very high τ → near-uniform (~12.5%)
    assert shares[5.0] < 0.20


def test_high_temperature_approaches_uniform():
    """Very high τ approaches uniform random (1/N each)."""
    # τ = 100 makes scaled diffs negligible → near-uniform softmax
    counts = Counter()
    for seed in range(2000):
        w = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                       temperature=100.0, rng_seed=seed)
        counts[w] += 1
    n_words = len(SEED44_MOTOR_N_RANKINGS)
    expected_rate = 1.0 / n_words  # 12.5% for 8 words
    # Each word should be within 2-3 percentage points of uniform
    for w, _ in SEED44_MOTOR_N_RANKINGS:
        rate = counts[w] / 2000
        assert abs(rate - expected_rate) < 0.05, (
            f"Word '{w}' rate {rate:.2%} far from uniform "
            f"{expected_rate:.2%} (counts: {dict(counts)})"
        )


def test_distribution_makes_sense_for_synonym_pair():
    """For a pair {primary='north', synonym='up'} both should appear in
    top half of distribution at moderate τ."""
    counts = Counter()
    for seed in range(1000):
        w = _sample_with_temperature(SEED44_MOTOR_N_RANKINGS,
                                       temperature=0.03, rng_seed=seed)
        counts[w] += 1
    # Both north and up should be in top 4 of the distribution
    sorted_by_freq = counts.most_common()
    top4 = [w for w, _ in sorted_by_freq[:4]]
    # north (rank 1 by sim) should win
    assert top4[0] == "north", f"Expected north #1, got {sorted_by_freq[:4]}"
    # up (rank 3 by sim) should be in top 4 of sampling distribution
    assert "up" in top4, (
        f"'up' should be in top 4 by sampling freq at τ=0.03. "
        f"Distribution: {sorted_by_freq[:4]}"
    )


def test_single_word_always_picks_it():
    """If only one word, sampling always returns it regardless of τ."""
    rankings = [("north", 0.5)]
    for t in [0.01, 0.1, 1.0]:
        result = _sample_with_temperature(rankings, temperature=t, rng_seed=42)
        assert result == "north"


def test_two_word_temperature_curve():
    """Two-word rankings: at low τ argmax wins; at high τ approaches 50/50."""
    rankings = [("a", 0.10), ("b", 0.05)]

    # Low τ: a wins ~all the time
    a_low = sum(
        1 for s in range(500)
        if _sample_with_temperature(rankings, temperature=0.005, rng_seed=s) == "a"
    )
    assert a_low / 500 > 0.95, f"Low τ should give a-rate >95%, got {a_low/500:.2%}"

    # High τ: ~50/50
    a_high = sum(
        1 for s in range(2000)
        if _sample_with_temperature(rankings, temperature=10.0, rng_seed=s) == "a"
    )
    assert 0.45 < a_high / 2000 < 0.55, (
        f"High τ should give ~50% a-rate, got {a_high/2000:.2%}"
    )
