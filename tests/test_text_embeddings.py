"""Tests for sim.text_embeddings — token I/O for language regions."""
from __future__ import annotations

import numpy as np
import pytest


def test_embed_deterministic():
    """Same token must always produce the same vector across calls."""
    from sim.text_embeddings import embed

    a = embed("north")
    b = embed("north")
    assert np.allclose(a, b)


def test_embed_different_tokens_different_vectors():
    """Different tokens must produce distinct embeddings (non-collision)."""
    from sim.text_embeddings import embed

    n = embed("north")
    e = embed("east")
    assert not np.allclose(n, e)
    # And cosine similarity should be near zero (orthogonal in expectation)
    cos = float(np.dot(n, e))  # both L2-normalized
    assert abs(cos) < 0.5, f"north vs east cosine={cos:.3f} too similar"


def test_embed_shape_and_norm():
    """Embeddings are 256-dim and L2-normalized to unit length."""
    from sim.text_embeddings import embed

    v = embed("goal")
    assert v.shape == (256,)
    assert v.dtype == np.float32
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"norm={norm} not unit"


def test_embed_empty_string():
    """Empty string gets zero vector (avoids hash issues)."""
    from sim.text_embeddings import embed

    v = embed("")
    assert np.allclose(v, np.zeros(256))


def test_nearest_token_finds_self():
    """Embedding a token, then asking for nearest token to that vector,
    must return the original token."""
    from sim.text_embeddings import embed, nearest_token

    for token in ["north", "east", "goal", "agent"]:
        v = embed(token)
        got = nearest_token(v, k=1)
        assert got[0] == token, f"nearest({token}) = {got[0]} (expected {token})"


def test_nearest_token_top_k():
    """Top-k must return k items, ranked by similarity."""
    from sim.text_embeddings import embed, nearest_token, DEFAULT_VOCAB

    v = embed("north")
    top3 = nearest_token(v, k=3)
    assert len(top3) == 3
    assert top3[0] == "north"  # first must be exact match
    # The other two should also be in the vocabulary
    for t in top3:
        assert t in DEFAULT_VOCAB


def test_nearest_token_zero_activity_returns_some_tokens():
    """When activity is all-zeros, return arbitrary top-k (no crash)."""
    from sim.text_embeddings import nearest_token

    got = nearest_token(np.zeros(256, dtype=np.float32), k=2)
    assert len(got) == 2


def test_nearest_token_wrong_shape_raises():
    """Mismatched activity dim raises ValueError."""
    from sim.text_embeddings import nearest_token

    with pytest.raises(ValueError, match="shape"):
        nearest_token(np.zeros(128, dtype=np.float32))


def test_vocab_to_drive_pattern():
    """Drive pattern is sparse (~sparsity fraction active) and consistent
    across calls."""
    from sim.text_embeddings import vocab_to_drive_pattern

    drive = vocab_to_drive_pattern("north", n_neurons=256, sparsity=0.1)
    assert drive.shape == (256,)
    assert drive.dtype == np.float32
    n_active = int(np.sum(drive > 0))
    assert 20 <= n_active <= 30, f"n_active={n_active} not ~10% of 256"

    # Determinism: same token, same pattern
    drive2 = vocab_to_drive_pattern("north", n_neurons=256, sparsity=0.1)
    assert np.allclose(drive, drive2)


def test_text_io_regions_off_by_default():
    """Without --enable-text-io, build_bg_brain_regions does not emit
    language_input / language_output. Backward-compat check."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions()
    region_names = {r.name for r in regions}
    assert "language_input" not in region_names
    assert "language_output" not in region_names


def test_text_io_regions_on_adds_two_language_regions():
    """With --enable-text-io, language_input + language_output regions added
    with default 256-neuron sizes."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_text_io=True)
    by_name = {r.name: r for r in regions}

    assert "language_input" in by_name
    assert "language_output" in by_name
    assert by_name["language_input"].n_neurons == 256
    assert by_name["language_output"].n_neurons == 256


def test_text_io_pathways_wired_to_cortex():
    """language_input -> cortex_{N,E,S,W} pathways are added (plastic,
    gated 'language_input_to_cortex'). Non-zero default weight_mean
    (per Kandel ch 53 — developmental pruning starts from dense, not
    zero, connectivity)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_text_io=True)
    by_edge = {(p.from_region, p.to_region): p for p in pathways}

    for action in ["N", "E", "S", "W"]:
        key = ("language_input", f"cortex_{action}")
        assert key in by_edge, f"Missing language_input -> cortex_{action}"
        p = by_edge[key]
        assert p.plastic is True
        assert p.plasticity_gate == "language_input_to_cortex"
        # Non-zero baseline (developmental dense init) so STDP has activity
        # to refine; opt-in to zero via text_input_to_cortex_weight=0.0.
        assert p.weight_mean > 0.0


def test_text_io_with_visual_cortex_adds_it_to_language_output():
    """With both --enable-text-io and --enable-visual-cortex, the
    cortex_it -> language_output pathway is wired (image-to-word)."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(
        enable_text_io=True,
        enable_visual_cortex=True,
    )
    by_edge = {(p.from_region, p.to_region): p for p in pathways}

    key = ("cortex_it", "language_output")
    assert key in by_edge
    p = by_edge[key]
    assert p.plastic is True
    assert p.plasticity_gate == "it_to_language_output"
    assert p.weight_mean == 0.0


def test_text_io_action_to_language_output_pathways():
    """Verbal-of-action pathways (cortex_X -> language_output) are wired
    so the agent can learn to verbalize what it just did."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_text_io=True)
    by_edge = {(p.from_region, p.to_region): p for p in pathways}

    for action in ["N", "E", "S", "W"]:
        key = (f"cortex_{action}", "language_output")
        assert key in by_edge
        p = by_edge[key]
        assert p.plastic is True
        assert p.plasticity_gate == "cortex_to_language_output"
        assert p.weight_mean == 0.0


def test_vocab_to_drive_pattern_different_tokens_differ():
    """Different tokens produce different drive patterns (different
    active neuron sets in expectation)."""
    from sim.text_embeddings import vocab_to_drive_pattern

    d_north = vocab_to_drive_pattern("north")
    d_east = vocab_to_drive_pattern("east")
    # Active sets should be mostly disjoint (independent random patterns)
    overlap = int(np.sum((d_north > 0) & (d_east > 0)))
    n_active = int(np.sum(d_north > 0))
    # In expectation, overlap = sparsity^2 * n = 0.01 * 256 ≈ 2-3
    assert overlap < n_active // 2, (
        f"overlap={overlap} suggests too-similar drive patterns"
    )


def test_vocab_to_drive_pattern_sparsity_005_orthogonal():
    """At sparsity=0.05 (12 active per word at n=256), the four direction
    word codes are nearly orthogonal — pairwise overlap ~0-1. Verifies the
    architectural-pivot hypothesis that sparser codes give cleaner
    word-to-motor discrimination.
    """
    from sim.text_embeddings import vocab_to_drive_pattern

    words = ["north", "east", "south", "west"]
    codes = {w: vocab_to_drive_pattern(w, n_neurons=256, sparsity=0.05) > 0
             for w in words}
    for w in words:
        assert int(codes[w].sum()) == 13, (
            f"{w} active count {int(codes[w].sum())} != 13 (sparsity=0.05, "
            f"n=256: int(round(0.05 * 256)) = 13)"
        )
    max_pairwise_overlap = 0
    for i, w1 in enumerate(words):
        for w2 in words[i + 1:]:
            overlap = int((codes[w1] & codes[w2]).sum())
            max_pairwise_overlap = max(max_pairwise_overlap, overlap)
    # Orthogonal-ish: at most 1 overlapping neuron between any two words
    assert max_pairwise_overlap <= 1, (
        f"max pairwise overlap {max_pairwise_overlap} > 1; codes are not "
        "near-orthogonal at sparsity=0.05"
    )


def test_evaluate_word_to_action_accepts_token_sparsity():
    """evaluate_word_to_action exposes token_sparsity parameter so eval
    matches whatever sparsity training used. Smoke-checks the parameter
    is in the signature; full behavior validated in research runs."""
    import inspect
    from research.runners.text_eval import evaluate_word_to_action

    sig = inspect.signature(evaluate_word_to_action)
    assert "token_sparsity" in sig.parameters, (
        "evaluate_word_to_action must accept token_sparsity for "
        "architectural sweep eval (sparsity=0.05, 0.10, etc.)"
    )
    # Default should be 0.1 (matches historical behavior)
    assert sig.parameters["token_sparsity"].default == 0.1


def test_run_curriculum_training_accepts_token_sparsity():
    """run_curriculum_training must accept token_sparsity to wire it to
    both training (Phase 2 navigation, Phase 3 SWR replay) AND eval."""
    import inspect
    from research.runners.text_train_curriculum import run_curriculum_training

    sig = inspect.signature(run_curriculum_training)
    assert "token_sparsity" in sig.parameters, (
        "run_curriculum_training must accept token_sparsity"
    )
    assert sig.parameters["token_sparsity"].default == 0.1


def test_run_curriculum_training_accepts_hebbian_overrides():
    """Fundamentals sweep needs to override v2's enable_hebbian_learning=False
    + stdp_w_max=5 + hebbian_weight_decay defaults via kwargs."""
    import inspect
    from research.runners.text_train_curriculum import run_curriculum_training

    sig = inspect.signature(run_curriculum_training)
    for param in ("enable_hebbian_learning", "hebbian_weight_decay",
                  "hebbian_learning_rate", "stdp_w_max"):
        assert param in sig.parameters, (
            f"run_curriculum_training must accept {param} for fundamentals sweep"
        )
    # Defaults should match the v2 baseline
    assert sig.parameters["enable_hebbian_learning"].default is False
    assert sig.parameters["stdp_w_max"].default == 5.0
    # None means "use sim.config default" — preserves backward compat
    assert sig.parameters["hebbian_weight_decay"].default is None
    assert sig.parameters["hebbian_learning_rate"].default is None


def test_run_curriculum_training_accepts_dt_ms():
    """dt-ms override allows halving sub-step count for speedup tests."""
    import inspect
    from research.runners.text_train_curriculum import run_curriculum_training

    sig = inspect.signature(run_curriculum_training)
    assert "dt_ms" in sig.parameters
    assert sig.parameters["dt_ms"].default == 0.5  # match sim's tuned value
