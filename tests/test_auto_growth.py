"""Tests for sim.auto_growth — TierPromoter + weight transfer.

CPU-only (numpy + mock bridge). The real GPU integration (bridge.cp_connections
slicing + set_pathway_weights) is exercised by a smoke test on a real
bridge, not in this file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.auto_growth import (
    TIER_LADDER, TierLadder, TierPromoter, PromotionPlan, GROWING_PATHWAYS,
    TransferPlan, compute_transfer_plan, transfer_weights_dense,
    DEFAULT_THRESHOLD, DEFAULT_CONSECUTIVE,
)
from sim.lineage import BridgeLineage, LineageMetadata


# ──────────────────────────────────────────────────────────────────────
# TierLadder
# ──────────────────────────────────────────────────────────────────────


def test_tier_ladder_default_ordering():
    """Default ladder is sorted ascending."""
    ladder = TierLadder()
    assert ladder.tiers == tuple(sorted(ladder.tiers))
    assert ladder.tiers[0] == 4


def test_tier_ladder_next_tier_advances():
    """next_tier returns the next-larger tier."""
    ladder = TierLadder()
    assert ladder.next_tier(4) == 8
    assert ladder.next_tier(8) == 12
    assert ladder.next_tier(64) == 96


def test_tier_ladder_next_tier_at_top():
    """next_tier at the top of the ladder returns None."""
    ladder = TierLadder()
    top = ladder.tiers[-1]
    assert ladder.next_tier(top) is None


def test_tier_ladder_prev_tier():
    """prev_tier returns the immediately-smaller tier."""
    ladder = TierLadder()
    assert ladder.prev_tier(8) == 4
    assert ladder.prev_tier(12) == 8
    assert ladder.prev_tier(4) is None


def test_tier_ladder_arch_for_tier():
    """arch_for_tier returns expected n_lang / n_motor / n_motor_fs."""
    ladder = TierLadder()
    assert ladder.arch_for_tier(4) == {"n_lang": 2048, "n_motor": 500,
                                          "n_motor_fs": 60}
    assert ladder.arch_for_tier(8) == {"n_lang": 4096, "n_motor": 1000,
                                          "n_motor_fs": 120}
    # 64-word: the validated local-3090 ceiling
    assert ladder.arch_for_tier(64)["n_lang"] == 8192


def test_tier_ladder_arch_unknown_tier_raises():
    """Unknown tier raises ValueError with helpful message."""
    ladder = TierLadder()
    with pytest.raises(ValueError, match="Unknown tier"):
        ladder.arch_for_tier(7)


# ──────────────────────────────────────────────────────────────────────
# TierPromoter — step() trigger logic
# ──────────────────────────────────────────────────────────────────────


def test_promoter_initial_state():
    """Fresh promoter starts at tier 4 with 0 consecutive passes."""
    p = TierPromoter(initial_tier=4)
    assert p.current_tier == 4
    assert p.consecutive_passes == 0
    assert p.threshold == DEFAULT_THRESHOLD
    assert p.consecutive_required == DEFAULT_CONSECUTIVE


def test_promoter_invalid_initial_tier_raises():
    """Initial tier not on the ladder raises ValueError."""
    with pytest.raises(ValueError, match="not on ladder"):
        TierPromoter(initial_tier=7)


def test_promoter_step_below_threshold_does_not_promote():
    """Below-threshold accuracy never triggers promotion."""
    p = TierPromoter(initial_tier=4)
    for _ in range(10):
        plan = p.step(0.5)  # well below 0.90
        assert plan is None
    assert p.consecutive_passes == 0


def test_promoter_step_threshold_requires_consecutive():
    """Three consecutive passes >= 0.90 trigger promotion."""
    p = TierPromoter(initial_tier=4, consecutive_required=3)
    # First two passes: no promote yet (counter is 1, 2)
    assert p.step(0.95) is None
    assert p.consecutive_passes == 1
    assert p.step(0.91) is None
    assert p.consecutive_passes == 2
    # Third pass: promote!
    plan = p.step(0.92)
    assert plan is not None
    assert plan.from_tier == 4
    assert plan.to_tier == 8
    assert plan.from_arch == {"n_lang": 2048, "n_motor": 500, "n_motor_fs": 60}
    assert plan.to_arch == {"n_lang": 4096, "n_motor": 1000, "n_motor_fs": 120}


def test_promoter_step_resets_on_dip():
    """One below-threshold eval resets the consecutive counter."""
    p = TierPromoter(initial_tier=4, consecutive_required=3)
    p.step(0.95)
    p.step(0.93)
    assert p.consecutive_passes == 2
    p.step(0.40)  # dip
    assert p.consecutive_passes == 0
    # Needs 3 more consecutive
    assert p.step(0.95) is None
    assert p.step(0.95) is None
    plan = p.step(0.95)
    assert plan is not None
    assert plan.to_tier == 8


def test_promoter_at_top_of_ladder_returns_none():
    """At the top tier, even passing accuracy returns None (no further promotion)."""
    ladder = TierLadder()
    top = ladder.tiers[-1]
    p = TierPromoter(initial_tier=top, consecutive_required=2)
    p.step(0.95)
    plan = p.step(0.95)
    assert plan is None  # would-be promote but no next tier
    # History records the at_top decision
    assert any(decision == "at_top" for _, _, decision in p.eval_history)


def test_promoter_records_eval_history():
    """Each step() appends to eval_history with the decision."""
    p = TierPromoter(initial_tier=4)
    p.step(0.95)
    p.step(0.40)
    p.step(0.91)
    assert len(p.eval_history) == 3
    assert p.eval_history[0] == (4, 0.95, "wait")
    assert p.eval_history[1] == (4, 0.40, "wait")
    assert p.eval_history[2] == (4, 0.91, "wait")


# ──────────────────────────────────────────────────────────────────────
# TierPromoter — confirm_promotion + lineage integration
# ──────────────────────────────────────────────────────────────────────


def test_confirm_promotion_advances_current_tier():
    """confirm_promotion advances current_tier + resets counter."""
    p = TierPromoter(initial_tier=4, consecutive_required=1)
    plan = p.step(0.95)
    assert plan is not None
    assert p.current_tier == 4  # still 4 until confirmed
    p.confirm_promotion(plan)
    assert p.current_tier == 8
    assert p.consecutive_passes == 0


def test_confirm_promotion_wrong_from_tier_raises():
    """confirm_promotion with stale plan (from != current) raises."""
    p = TierPromoter(initial_tier=4)
    stale_plan = PromotionPlan(
        from_tier=8, to_tier=12,  # from doesn't match current 4
        from_arch={"n_lang": 4096, "n_motor": 1000, "n_motor_fs": 120},
        to_arch={"n_lang": 4096, "n_motor": 2000, "n_motor_fs": 240},
    )
    with pytest.raises(ValueError, match="plan.from_tier"):
        p.confirm_promotion(stale_plan)


def test_confirm_promotion_records_growth_event_in_lineage(tmp_path):
    """Promotion appends a growth event + updates arch in the lineage metadata."""
    # Seed a lineage at tier1
    lineage = BridgeLineage("main", root=tmp_path)
    meta = LineageMetadata(lineage_name="main", current_tier="4-word",
                             arch={"mode": "tier1", "n_lang_input": 2048,
                                    "n_motor_per_action": 500,
                                    "n_motor_fs_per_action": 60})
    # Make the lineage "exist" by writing both files
    lineage.root.mkdir(parents=True, exist_ok=True)
    lineage.current_path.write_text("fake-bridge-state", encoding="utf-8")
    lineage.write_metadata(meta)

    # Run a promotion
    p = TierPromoter(initial_tier=4, consecutive_required=1)
    plan = p.step(0.95)
    p.confirm_promotion(plan, lineage=lineage)

    # Reload metadata and verify the growth event landed
    meta2 = lineage.read_metadata()
    assert meta2.current_tier == "8-word"
    assert meta2.arch["n_lang_input"] == 4096
    assert meta2.arch["n_motor_per_action"] == 1000
    assert meta2.arch["n_motor_fs_per_action"] == 120
    assert any(e["kind"] == "tier_promotion" for e in meta2.growth_events)
    promote_event = next(
        e for e in meta2.growth_events if e["kind"] == "tier_promotion"
    )
    assert promote_event["metadata"]["from_tier"] == 4
    assert promote_event["metadata"]["to_tier"] == 8


# ──────────────────────────────────────────────────────────────────────
# Weight-transfer pure-Python
# ──────────────────────────────────────────────────────────────────────


def test_compute_transfer_plan_basic():
    """Plan reports expected mapped + new edge counts."""
    plan = compute_transfer_plan(
        pathway_name="language_input_to_motor_N",
        pre_old_size=2048, post_old_size=500,
        pre_new_size=4096, post_new_size=1000,
        density=1.0,
    )
    assert plan.pre_old_size == 2048
    assert plan.post_old_size == 500
    assert plan.expected_mapped_edges == 2048 * 500  # 1,024,000
    assert plan.expected_new_edges == 4096 * 1000 - 2048 * 500  # 3,072,000


def test_compute_transfer_plan_with_density():
    """Edge counts respect pathway density."""
    plan = compute_transfer_plan(
        pathway_name="test",
        pre_old_size=100, post_old_size=100,
        pre_new_size=200, post_new_size=200,
        density=0.1,  # only 10% of edges
    )
    assert plan.expected_mapped_edges == 100 * 100 * 0.1   # 1000
    assert plan.expected_new_edges == 200 * 200 * 0.1 - 1000  # 3000


def test_compute_transfer_plan_shrinking_raises():
    """compute_transfer_plan rejects pool shrinking."""
    with pytest.raises(ValueError, match="smaller than old"):
        compute_transfer_plan(
            pathway_name="bad",
            pre_old_size=4096, post_old_size=1000,
            pre_new_size=2048, post_new_size=500,
        )


def test_transfer_weights_dense_preserves_old_block():
    """Upper-left (post_old, pre_old) block of new_W matches old_W."""
    rng = np.random.default_rng(42)
    old_W = rng.normal(loc=0.5, scale=0.1, size=(10, 20)).astype(np.float32)
    new_W = transfer_weights_dense(
        old_W, new_pre_size=40, new_post_size=20,
        new_weight_mean=0.5, new_weight_jitter=0.1, rng=rng,
    )
    assert new_W.shape == (20, 40)
    # Upper-left block is preserved
    np.testing.assert_array_equal(new_W[:10, :20], old_W)


def test_transfer_weights_dense_random_init_outside_block():
    """Outside-block regions are sampled from N(mean, jitter)."""
    rng = np.random.default_rng(7)
    old_W = np.ones((4, 4), dtype=np.float32) * 5.0
    new_W = transfer_weights_dense(
        old_W, new_pre_size=8, new_post_size=8,
        new_weight_mean=0.0, new_weight_jitter=0.05, rng=rng,
    )
    # Upper-left is 5.0
    np.testing.assert_array_equal(new_W[:4, :4], 5.0)
    # Other quadrants should have values centered near 0, far from 5
    outside = np.concatenate([
        new_W[4:, :].flatten(),
        new_W[:4, 4:].flatten(),
    ])
    assert abs(outside.mean()) < 0.05  # near 0
    assert outside.max() < 1.0  # nowhere near 5


def test_transfer_weights_dense_shrink_raises():
    """transfer_weights_dense rejects pool shrinking."""
    rng = np.random.default_rng(0)
    old_W = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="smaller than old"):
        transfer_weights_dense(old_W, new_pre_size=5, new_post_size=5, rng=rng)


def test_transfer_weights_dense_preserves_dtype():
    """new_W has same dtype as old_W (typically float32 for GPU)."""
    old_W = np.ones((4, 4), dtype=np.float32)
    new_W = transfer_weights_dense(old_W, new_pre_size=8, new_post_size=8)
    assert new_W.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────
# GROWING_PATHWAYS contract
# ──────────────────────────────────────────────────────────────────────


def test_growing_pathways_covers_phase14_branch_a():
    """The pathway list covers all four motor regions + FS inhib + readout."""
    pathways = set(GROWING_PATHWAYS)
    # 4 actions × (language_input -> motor + motor -> language_output + ...)
    for action in ("N", "E", "S", "W"):
        assert f"language_input_to_motor_{action}" in pathways
        assert f"motor_{action}_to_language_output" in pathways
        assert f"motor_{action}_to_motor_FS_{action}" in pathways
        assert f"motor_FS_{action}_to_motor_{action}" in pathways
    # Total: 4 + 4 + 4 + 4 = 16 pathways
    assert len(pathways) == 16


# ──────────────────────────────────────────────────────────────────────
# End-to-end: TierPromoter + lineage in one mini-loop
# ──────────────────────────────────────────────────────────────────────


def test_end_to_end_tier1_to_tier21_promotion(tmp_path):
    """Simulate: train tier1 for 5 epochs, hit threshold, auto-promote to tier2.1."""
    # Seed a lineage at tier1
    lineage = BridgeLineage("main", root=tmp_path)
    meta = LineageMetadata(lineage_name="main", current_tier="4-word",
                             arch={"mode": "tier1", "n_lang_input": 2048,
                                    "n_motor_per_action": 500,
                                    "n_motor_fs_per_action": 60})
    lineage.root.mkdir(parents=True, exist_ok=True)
    lineage.current_path.write_text("v0", encoding="utf-8")
    lineage.write_metadata(meta)

    p = TierPromoter(initial_tier=4, threshold=0.90, consecutive_required=3)

    # Simulated training: accuracy ramps up over 5 evals
    accuracies = [0.30, 0.50, 0.85, 0.92, 0.94, 0.93, 0.95]
    plan = None
    for acc in accuracies:
        plan = p.step(acc)
        if plan is not None:
            break

    # Should have triggered after the 3rd consecutive >=0.90 (acc 0.93 at idx 5)
    assert plan is not None
    assert plan.from_tier == 4
    assert plan.to_tier == 8

    # Confirm promotion records to lineage
    p.confirm_promotion(plan, lineage=lineage)
    assert p.current_tier == 8

    # Lineage now reflects the new tier + arch
    meta2 = lineage.read_metadata()
    assert meta2.current_tier == "8-word"
    assert meta2.arch["n_motor_per_action"] == 1000
    # Growth event recorded
    assert any(e["kind"] == "tier_promotion"
               for e in meta2.growth_events)
