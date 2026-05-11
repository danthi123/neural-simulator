"""Tests for auto_grow_chat — Strategy B of Phase A2.

CPU-only. Tests the orchestration loop with mock train_fn / transfer_fn
so we can verify TierPromoter wiring without GPU.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.runners.auto_grow_chat import (
    run_auto_grow_demo,
    synthetic_train_fn,
    synthetic_transfer_fn,
    _MockBridge,
    AutoGrowResult,
)


# ──────────────────────────────────────────────────────────────────────
# synthetic_train_fn — accuracy ramp behavior
# ──────────────────────────────────────────────────────────────────────


def test_synthetic_train_fn_ramps_accuracy_over_epochs():
    """Accuracy climbs from 0.45 at epoch 0 to >=0.90 by epoch 6+."""
    arch = {"n_lang": 2048, "n_motor": 500, "n_motor_fs": 60}
    bridge = None
    accs = []
    for epoch in range(10):
        bridge, acc = synthetic_train_fn(tier=4, arch=arch,
                                            bridge=bridge, epoch=epoch)
        accs.append(acc)
    assert accs[0] < 0.6, f"early epoch accuracy too high: {accs[0]}"
    assert accs[-1] >= 0.90, f"late epoch accuracy too low: {accs[-1]}"


def test_synthetic_train_fn_returns_consistent_bridge():
    """Same bridge object passed back across epochs."""
    arch = {"n_lang": 2048, "n_motor": 500, "n_motor_fs": 60}
    bridge1, _ = synthetic_train_fn(tier=4, arch=arch, bridge=None, epoch=0)
    bridge2, _ = synthetic_train_fn(tier=4, arch=arch, bridge=bridge1, epoch=1)
    assert bridge1 is bridge2


# ──────────────────────────────────────────────────────────────────────
# Orchestration: promotion fires on consecutive passes
# ──────────────────────────────────────────────────────────────────────


def test_auto_grow_promotes_at_threshold():
    """With synthetic_train_fn, 3 promotions fire over the natural ramp."""
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=3, max_epochs_per_tier=20,
        verbose=False,
    )
    # 3 promotions should fire: 4 -> 8 -> 12 -> 16
    assert result.promotions_executed == 3
    assert result.final_tier == 16
    # Each tier should have used at least the consecutive_required pass epochs
    assert all(e >= 3 for e in result.epochs_at_each_tier.values()
                if result.epochs_at_each_tier)


def test_auto_grow_zero_promotions_when_max_is_zero():
    """max_promotions=0 trains one tier and stops without promotion."""
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=0, max_epochs_per_tier=20,
        verbose=False,
    )
    assert result.promotions_executed == 0
    assert result.final_tier == 4


def test_auto_grow_stops_if_accuracy_never_passes():
    """Low-acc train_fn → no promotions; loop ends at max_epochs_per_tier."""
    def low_acc_fn(tier, arch, bridge=None, epoch=0):
        if bridge is None:
            bridge = _MockBridge(tier=tier, arch=dict(arch))
        return bridge, 0.30  # well below 0.90
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=3, max_epochs_per_tier=5,
        train_fn=low_acc_fn,
        verbose=False,
    )
    assert result.promotions_executed == 0
    assert result.final_tier == 4
    assert result.epochs_total == 5


def test_auto_grow_resets_consecutive_on_dip():
    """Oscillating accuracy resets the consecutive-pass counter."""
    history = []
    def oscillating_fn(tier, arch, bridge=None, epoch=0):
        if bridge is None:
            bridge = _MockBridge(tier=tier, arch=dict(arch))
        # First 2 passes high, then dip, then 3 passes high → promotion
        seq = [0.95, 0.95, 0.40, 0.95, 0.95, 0.95]
        acc = seq[epoch] if epoch < len(seq) else 0.95
        history.append(acc)
        return bridge, acc
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=1, max_epochs_per_tier=20,
        train_fn=oscillating_fn,
        verbose=False,
    )
    # Promotion should fire on the 6th epoch (3rd consecutive after the dip)
    assert result.promotions_executed == 1
    assert result.final_tier == 8


def test_auto_grow_transfer_fn_is_called_per_promotion():
    """transfer_fn is invoked for each promotion."""
    call_count = [0]
    def counting_transfer(from_tier, to_tier, old_bridge, new_arch):
        call_count[0] += 1
        return _MockBridge(tier=to_tier, arch=dict(new_arch))
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=2, max_epochs_per_tier=20,
        transfer_fn=counting_transfer,
        verbose=False,
    )
    assert result.promotions_executed == 2
    assert call_count[0] == 2


def test_auto_grow_at_top_of_ladder():
    """Starting at the top tier → 0 promotions even if accuracy passes."""
    from sim.auto_growth import TierLadder
    top_tier = TierLadder().tiers[-1]
    result = run_auto_grow_demo(
        initial_tier=top_tier, threshold=0.90, consecutive_required=3,
        max_promotions=3, max_epochs_per_tier=10,
        verbose=False,
    )
    assert result.promotions_executed == 0
    assert result.final_tier == top_tier


# ──────────────────────────────────────────────────────────────────────
# Lineage growth-event integration
# ──────────────────────────────────────────────────────────────────────


def test_auto_grow_writes_growth_events_to_lineage(tmp_path):
    """When lineage_name is given, each promotion adds a growth event."""
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=2, max_epochs_per_tier=20,
        lineage_name="auto_grow_test",
        lineage_root=tmp_path,
        verbose=False,
    )
    assert result.promotions_executed == 2
    assert result.growth_event_count == 2

    # Lineage metadata should have the growth events
    from sim.lineage import BridgeLineage
    lineage = BridgeLineage("auto_grow_test", root=tmp_path)
    meta = lineage.read_metadata()
    promotion_events = [e for e in meta.growth_events
                          if e["kind"] == "tier_promotion"]
    assert len(promotion_events) == 2
    # First should be 4 -> 8
    assert promotion_events[0]["metadata"]["from_tier"] == 4
    assert promotion_events[0]["metadata"]["to_tier"] == 8


def test_auto_grow_no_lineage_no_growth_events():
    """Without lineage_name, growth_event_count stays 0 (still promotes)."""
    result = run_auto_grow_demo(
        initial_tier=4, threshold=0.90, consecutive_required=3,
        max_promotions=1, max_epochs_per_tier=20,
        verbose=False,
    )
    assert result.promotions_executed == 1
    assert result.growth_event_count == 0


# ──────────────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────────────


def test_auto_grow_chat_cli_help():
    """The CLI module's --help is well-formed."""
    import subprocess
    p = subprocess.run(
        [sys.executable, "-m", "research.runners.auto_grow_chat", "--help"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert p.returncode == 0
    assert "--initial-tier" in p.stdout
    assert "--max-promotions" in p.stdout
    assert "--lineage" in p.stdout


def test_auto_grow_chat_cli_run(tmp_path):
    """End-to-end CLI run produces expected JSON output."""
    import subprocess
    out_path = tmp_path / "result.json"
    p = subprocess.run(
        [sys.executable, "-m", "research.runners.auto_grow_chat",
         "--initial-tier", "4",
         "--max-promotions", "2",
         "--max-epochs-per-tier", "15",
         "--out", str(out_path)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert p.returncode == 0, p.stderr
    assert "PROMOTING" in p.stdout
    assert "Complete" in p.stdout

    import json
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["initial_tier"] == 4
    assert summary["promotions_executed"] == 2
    assert summary["final_tier"] == 12
