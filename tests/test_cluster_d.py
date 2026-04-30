"""Cluster D v2 — Sharp-Wave-Ripple replay for offline CA3 cleanup.

Covers the design from docs/plans/2026-04-30-cluster-d-v2-swr-design.md:
- _ca3_burst_active() detects population bursts in CA3 (rate > mu + 2*sigma)
- --enable-cluster-d-v2-swr CLI flag wires up the gate
- ca3_swr_burst plasticity gate: low (0.1) outside bursts during sleep,
  high (1.0) during burst windows or during wake.
"""
from __future__ import annotations

from collections import deque

import pytest

from research.runners.g11_bg_runner import (
    _ca3_burst_active,
    _swr_gate_value,
    _swr_gate_value_scheduled,
    build_bg_brain_regions,
)


def test_burst_detector_flat_rate_never_fires():
    """50 flat-rate samples (constant 5 Hz) should produce no burst at any
    step. Standard-deviation defense: a flat signal has sigma=0, and the
    detector should not return True even when current_rate equals the mean.
    """
    history: deque[float] = deque(maxlen=40)
    constant_rate = 5.0
    burst_count = 0
    for _ in range(50):
        if _ca3_burst_active(constant_rate, history):
            burst_count += 1
    assert burst_count == 0, (
        f"flat constant rate should never trigger a burst; got {burst_count} bursts"
    )


def test_burst_detector_2sigma_spike_fires_on_spike_step():
    """Inject a 5 Hz baseline for 30 steps, then a single 50 Hz spike at
    step 30 (~13σ above the noise-free baseline). Burst should fire on the
    spike step. Flat baseline before should produce no bursts."""
    history: deque[float] = deque(maxlen=40)
    burst_steps = []
    rates = []
    # Build up a baseline with small jitter so sigma is non-trivial.
    # Without jitter the floor σ=1e-6 kicks in but we still want a normal
    # case where the spike clearly crosses μ + 2σ.
    rng = [4.8, 5.1, 4.9, 5.2, 5.0, 4.7, 5.3, 5.0, 4.9, 5.1] * 3  # 30 samples
    for i, r in enumerate(rng):
        rates.append(r)
        if _ca3_burst_active(r, history):
            burst_steps.append(i)
    # Now spike at step 30.
    rates.append(50.0)
    if _ca3_burst_active(50.0, history):
        burst_steps.append(30)
    # And a return-to-baseline at step 31; should NOT fire (the spike
    # itself is now in the history and raises σ).
    rates.append(5.0)
    if _ca3_burst_active(5.0, history):
        burst_steps.append(31)
    assert burst_steps == [30], (
        f"expected burst exactly at step 30; got {burst_steps}; rates={rates}"
    )


def test_v2_flag_creates_swr_gate_on_ca3_recurrent():
    """With v2 enabled, CA3's automatic internal connectivity is replaced
    by an explicit ca3→ca3 pathway carrying the `ca3_swr_burst` plasticity
    gate. v1 alone keeps the implicit internal_density and has no explicit
    self-pathway."""
    # v1 only
    regions_v1, pathways_v1 = build_bg_brain_regions(
        enable_cluster_d_hippocampus=True,
    )
    ca3_v1 = next(r for r in regions_v1 if r.name == "ca3")
    self_paths_v1 = [
        p for p in pathways_v1
        if p.from_region == "ca3" and p.to_region == "ca3"
    ]
    assert ca3_v1.internal_density > 0, (
        "v1 should keep implicit CA3 internal_density"
    )
    assert len(self_paths_v1) == 0, (
        f"v1 should not have an explicit ca3→ca3 pathway; got {len(self_paths_v1)}"
    )

    # v1 + v2
    regions_v2, pathways_v2 = build_bg_brain_regions(
        enable_cluster_d_hippocampus=True,
        enable_cluster_d_v2_swr=True,
    )
    ca3_v2 = next(r for r in regions_v2 if r.name == "ca3")
    self_paths_v2 = [
        p for p in pathways_v2
        if p.from_region == "ca3" and p.to_region == "ca3"
    ]
    assert ca3_v2.internal_density == 0, (
        f"v2 should disable implicit CA3 internal_density; got {ca3_v2.internal_density}"
    )
    assert len(self_paths_v2) == 1, (
        f"v2 should add exactly one ca3→ca3 pathway; got {len(self_paths_v2)}"
    )
    p = self_paths_v2[0]
    assert p.plasticity_gate == "ca3_swr_burst", (
        f"ca3→ca3 pathway should carry the SWR plasticity gate; got "
        f"plasticity_gate={p.plasticity_gate!r}"
    )
    assert p.plastic is True, "ca3→ca3 should be plastic under v2"


def test_v2_off_no_swr_gate():
    """If v2 is OFF (regardless of v1), no pathway should carry the
    `ca3_swr_burst` gate."""
    for v1 in (False, True):
        _, pathways = build_bg_brain_regions(
            enable_cluster_d_hippocampus=v1,
        )
        gated = [p for p in pathways if p.plasticity_gate == "ca3_swr_burst"]
        assert len(gated) == 0, (
            f"with v1={v1}, v2=off: no pathway should carry ca3_swr_burst; "
            f"got {[(p.from_region, p.to_region) for p in gated]}"
        )


def test_v2_requires_v1():
    """v2 without v1 makes no sense (no CA3 region to gate). Should raise
    a ValueError with a clear message."""
    with pytest.raises(ValueError, match=r"cluster.*[Dd].*v[12]|hippocampus"):
        build_bg_brain_regions(
            enable_cluster_d_hippocampus=False,
            enable_cluster_d_v2_swr=True,
        )


def test_swr_gate_default_low_in_sleep_no_burst():
    """During sleep with no active burst, the gate should sit at the low
    baseline (0.1) so STDP on the CA3 recurrent stays mostly suppressed."""
    history: deque[float] = deque(maxlen=40)
    flat_rate = 5.0
    # Fill the history with the flat baseline (no bursts)
    for _ in range(15):
        gate = _swr_gate_value(in_sleep=True, current_rate_hz=flat_rate, history=history)
    # After enough samples to compute stats, gate should be the low value
    assert gate == pytest.approx(0.1), (
        f"sleep + no burst should give gate=0.1; got {gate}"
    )


def test_swr_gate_thaws_during_burst():
    """When a burst is detected during sleep, the gate should jump to 1.0
    so STDP fully fires for the duration of the burst."""
    history: deque[float] = deque(maxlen=40)
    # Build a noisy baseline so sigma is finite
    for r in [4.8, 5.1, 4.9, 5.2, 5.0, 4.7, 5.3, 5.0, 4.9, 5.1, 5.0, 4.8]:
        _swr_gate_value(in_sleep=True, current_rate_hz=r, history=history)
    # Now spike: should detect burst and return 1.0
    burst_gate = _swr_gate_value(in_sleep=True, current_rate_hz=50.0, history=history)
    assert burst_gate == pytest.approx(1.0), (
        f"sleep + burst should give gate=1.0; got {burst_gate}"
    )


def test_swr_gate_unchanged_during_wake():
    """During wake the gate should always be 1.0 regardless of CA3 firing
    rate. v2's gating only modulates plasticity offline; wake plasticity
    follows the standard cluster D v1 path."""
    history: deque[float] = deque(maxlen=40)
    # Even with rates that would normally trigger a burst during sleep,
    # wake mode should ignore them and keep the gate fully open.
    for r in [5.0, 5.0, 5.0, 5.0, 5.0, 50.0, 5.0, 100.0, 5.0]:
        gate = _swr_gate_value(in_sleep=False, current_rate_hz=r, history=history)
        assert gate == pytest.approx(1.0), (
            f"wake should always give gate=1.0; got {gate} at rate={r}"
        )


def test_scheduled_swr_gate_wake_always_open():
    """Scheduled-SWR gate during wake always returns 1.0 regardless of
    sleep_step_index. Wake = no v2 gating."""
    for idx in [0, 1, 5, 7, 14, 100]:
        assert _swr_gate_value_scheduled(False, idx, period=7) == 1.0


def test_scheduled_swr_gate_sleep_period_pattern():
    """During sleep, the gate is 1.0 on every period-th step and 0.1
    otherwise. With period=7: indices 0, 7, 14, 21 = burst windows."""
    period = 7
    burst_indices = []
    suppressed_indices = []
    for idx in range(50):
        gate = _swr_gate_value_scheduled(True, idx, period=period)
        if gate == 1.0:
            burst_indices.append(idx)
        else:
            assert gate == pytest.approx(0.1), (
                f"non-burst sleep step {idx} should be 0.1; got {gate}"
            )
            suppressed_indices.append(idx)
    # Expected: indices 0, 7, 14, 21, 28, 35, 42, 49 (8 burst windows in 50 steps)
    expected_bursts = list(range(0, 50, 7))
    assert burst_indices == expected_bursts, (
        f"expected burst indices {expected_bursts}; got {burst_indices}"
    )
    # Duty cycle: ~14% (8/50 ≈ 16%, close to biology's 10-15%)
    duty = len(burst_indices) / 50
    assert 0.10 <= duty <= 0.20, f"duty cycle out of biological range: {duty}"


def test_scheduled_swr_gate_default_period_is_seven():
    """Default period gives ~14% duty cycle which approximates biological
    SWR rates during NREM (Buzsaki 2015). Sanity check the default."""
    # Calling without period kwarg should match the period=7 result
    for idx in range(20):
        assert _swr_gate_value_scheduled(True, idx) == _swr_gate_value_scheduled(True, idx, period=7)
