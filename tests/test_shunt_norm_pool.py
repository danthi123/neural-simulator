"""Tests for the SHUNT-NORM POOL (Tier-2 linattn on-bridge divisive normalization, 2026-09-04).

The on-substrate realization of the num/den division the linattn own-voice mouth's read performs
(`read_t = num_t / (den_t + eps)`, research/findings/2026-09-03-linattn-spike-native-normalization-
DESIGN.md). Unlike `enable_dendritic_divisive_gain` (whose divisor is each presynaptic SOURCE's own
firing-rate EMA -- a per-source self-suppression) and `enable_input_divisive_norm`/`_2` (whose divisor is
the flagged READ set's own current mean -- the wrong axis, the already-refuted --dual-nonneg-divnorm
channel-pool NO-GO), this pool's divisor is a single shared EXTERNAL scalar `den_ema`: the firing-rate EMA
of a SEPARATE, designated norm-neuron region (BrainRegion.shunt_norm_source=True), read by every
shunt_norm_read region via `r_i = x_i / (shunt_norm_sigma + shunt_norm_gain*den_ema)` -- Carandini-Heeger's
single normalization pool, over the query's match-mass axis (DESIGN doc Sec 1-2).

NOTE on the region-pathway fixture: `sim.bridge`'s connectivity generator hits a pre-existing, UNRELATED
bug (`UnboundLocalError: profile_name_for_conn`) when a brain-region-framework config declares ZERO
region_pathways at all -- it falls back to a spatial generator that isn't reached in this file's tests.
Every build here therefore declares one real (non-empty) cross-region pathway with weight_mean=0.0: real
synapse objects exist (sidestepping the fallback), but their conductance contribution is EXACTLY zero, so
`read`'s dynamics are driven ONLY by (a) its own external current and (b) the shunt-norm divisor -- the
ONLY coupling between norm and read is the shared divisor, never a literal synapse. Not a shunt-norm-pool
bug; flagged as a residual (see the runner + this task's report) rather than fixed here (out of scope: an
unrelated pre-existing connectivity-fallback defect).

Load-bearing guarantees pinned here (CPU / SIM_BACKEND=numpy, deterministic, no GPU; all four thresholds
below were verified empirically before being pinned -- see the task's provenance):
  1. BYTE-IDENTITY WHEN OFF: with enable_shunt_norm_pool=False (the default), cp_shunt_norm_read_mask /
     _source_mask / _den_ema all stay None, the apply + EMA-update blocks are unreached, and two
     same-seed builds step bitwise-identically.
  2. ON allocates + den_ema updates: with the flag on, den_ema is allocated (starts at 0.0, matching the
     exact recursion's zden_0=0) and is nonzero after running (the substrate's OU background noise alone
     gives a nonzero floor; a driven source pushes it well above that floor -- test 3).
  3. THE DIVISOR TRACKS THE SOURCE, not a fixed gain: a driven norm-source (den_ema well above the noise
     floor) measurably SUPPRESSES the read region's own firing rate relative to a silent source (den_ema
     at the noise floor only) -- same seed, same read drive, no literal synapse between them (see the
     pathway note above). This is the shared-pool mechanism itself.
  4. SIGMA-DOMINATION anti-cheat (DESIGN doc Sec 4, "the clamp owned 97% of the effect" trap): for THIS
     primitive's shape (r = x/(sigma + gain*den)), sigma dominating means the silent-vs-driven CONTRAST
     collapses (both conditions converge to nearly the same rate, since gain*den becomes negligible next
     to a huge fixed sigma) -- NOT that behavior reverts to flag-off (dividing by a huge sigma actually
     crushes the drive toward zero, the opposite of "off"). Verified: at sigma=1e-6..0.05 the silent/driven
     ratio is far from 1 (den clearly matters); at sigma=1e6 the ratio is 1.00 (den stops mattering) --
     exactly the "is the effect den-driven or sigma-driven" question the design's anti-cheat asks.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def numpy_backend(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests, get_backend
    _reset_cache_for_tests()
    xp, name = get_backend("numpy")
    assert name == "numpy"
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])
    yield xp
    _reset_cache_for_tests()
    for modname in ("sim.kernels", "sim.connectivity", "sim.bridge"):
        if modname in sys.modules:
            importlib.reload(sys.modules[modname])


def _build(seed=42, enable_pool=False, sigma=0.05, gain=20.0,
           norm_n=6, read_n=40, source_drive=0.0, read_drive=180.0):
    """A small 2-region bridge: `norm` (the shunt_norm_source pool) driven by constant external current
    `source_drive`; `read` (the shunt_norm_read pool) driven independently by constant external current
    `read_drive`. The ONLY declared pathway is norm->read with weight_mean=0.0 (real synapses, zero
    conductance -- see the module docstring's fixture note), so any change in read's behavior as
    source_drive varies is attributable to the shared shunt-norm divisor alone."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="norm", n_neurons=norm_n, exc_fraction=1.0, internal_density=0.0,
                    shunt_norm_source=enable_pool),
        BrainRegion(name="read", n_neurons=read_n, exc_fraction=1.0, internal_density=0.0,
                    shunt_norm_read=enable_pool),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="norm", to_region="read", density=0.2,
                      weight_mean=0.0, weight_jitter=0.0),
    ]
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_shunt_norm_pool = enable_pool
    cfg.shunt_norm_sigma = sigma
    cfg.shunt_norm_gain = gain
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    norm_idx = bridge.region_manager.indices("norm")
    read_idx = bridge.region_manager.indices("read")
    bridge.cp_external_input_current[norm_idx] = source_drive
    bridge.cp_external_input_current[read_idx] = read_drive
    return bridge, norm_idx, read_idx


def _run(bridge, n_steps):
    import numpy as np
    fired = []
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        fired.append(np.asarray(bridge.cp_firing_states).copy())
    return np.stack(fired)


def _read_rate(bridge, read_idx, n_steps):
    """Mean fraction of read-pool neurons firing per step, over n_steps."""
    import numpy as np
    read_idx = np.asarray(read_idx)
    total = 0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        total += int(np.asarray(bridge.cp_firing_states)[read_idx].sum())
    return total / (n_steps * len(read_idx))


def test_off_is_none_and_deterministic(numpy_backend):
    """Flag off (default): all three shunt-norm arrays stay None, and two same-seed builds step
    bit-identically."""
    import numpy as np
    b1, _, _ = _build(seed=7, enable_pool=False, source_drive=200.0)
    assert b1.cp_shunt_norm_read_mask is None
    assert b1.cp_shunt_norm_source_mask is None
    assert b1.cp_shunt_norm_den_ema is None
    f1 = _run(b1, 60)
    b2, _, _ = _build(seed=7, enable_pool=False, source_drive=200.0)
    f2 = _run(b2, 60)
    assert np.array_equal(f1, f2), "flag-off run is not deterministic across builds"


def test_on_allocates_and_den_ema_grows(numpy_backend):
    """Flag on: den_ema is allocated, starts at 0.0 (matching the exact recursion's zden_0=0), and is
    nonzero after running with a driven source (well above the substrate's own OU-noise floor)."""
    import numpy as np
    b, norm_idx, read_idx = _build(seed=7, enable_pool=True, source_drive=200.0)
    assert b.cp_shunt_norm_read_mask is not None
    assert b.cp_shunt_norm_source_mask is not None
    assert b.cp_shunt_norm_den_ema is not None
    assert float(np.asarray(b.cp_shunt_norm_den_ema)[0]) == 0.0, "den_ema must start at 0 (zden_0=0)"
    _run(b, 300)
    den_ema_final = float(np.asarray(b.cp_shunt_norm_den_ema)[0])
    assert 0.0 < den_ema_final <= 1.0 + 1e-6, (
        f"den_ema must be a nonzero, bounded [0,1] mean-firing-fraction after a driven run (got {den_ema_final})")


def test_divisor_tracks_source_not_a_fixed_gain(numpy_backend):
    """THE MECHANISM: with the norm-source region DRIVEN (den_ema well above the noise floor), the read
    region's OWN firing rate is measurably LOWER than with the source SILENT (den_ema at the noise floor
    only) -- same seed, same read drive, no literal synapse between them (see module docstring). Verified
    empirically: silent~=0.28, driven~=0.04 (a >6x suppression) at sigma=0.05, gain=20."""
    n = 300
    b_silent, _, r_silent = _build(seed=11, enable_pool=True, sigma=0.05, gain=20.0, source_drive=0.0)
    rate_silent = _read_rate(b_silent, r_silent, n)
    b_driven, _, r_driven = _build(seed=11, enable_pool=True, sigma=0.05, gain=20.0, source_drive=200.0)
    rate_driven = _read_rate(b_driven, r_driven, n)
    print(f"read firing rate: source-silent={rate_silent:.4f}  source-driven={rate_driven:.4f}  "
          f"ratio={rate_driven / rate_silent if rate_silent else float('nan'):.3f}")
    assert rate_driven < 0.5 * rate_silent, (
        f"a driven norm-source did not suppress the read region's firing rate "
        f"(silent {rate_silent} vs driven {rate_driven}) -- the shared divisor is not tracking den")


def test_sigma_domination_collapses_the_contrast(numpy_backend):
    """SIGMA-DOMINATION anti-cheat (DESIGN doc Sec 4, "the clamp owned 97% of the effect" trap). For
    r = x/(sigma + gain*den), sigma DOMINATING means den stops mattering: the silent-vs-driven contrast
    (the previous test's mechanism) must COLLAPSE toward a ratio of 1.0 as sigma grows huge, even though
    at a den-sensitive sigma the SAME contrast is large. This is the correct signature for this formula's
    shape (unlike a multiplicative sigma/(sigma+a) gain, dividing by a huge sigma crushes the read toward
    silence -- NOT toward flag-off behavior -- so "matches off" is the wrong test; "stops tracking den" is
    the right one, and is exactly the design's own anti-cheat question)."""
    n = 300

    def contrast_ratio(sigma):
        b_s, _, r_s = _build(seed=23, enable_pool=True, sigma=sigma, gain=20.0, source_drive=0.0)
        rs = _read_rate(b_s, r_s, n)
        b_d, _, r_d = _build(seed=23, enable_pool=True, sigma=sigma, gain=20.0, source_drive=200.0)
        rd = _read_rate(b_d, r_d, n)
        return (rd / rs) if rs else float("nan"), rs, rd

    ratio_small, rs_small, rd_small = contrast_ratio(0.05)
    ratio_huge, rs_huge, rd_huge = contrast_ratio(1e6)
    print(f"sigma=0.05:  silent={rs_small:.4f} driven={rd_small:.4f} ratio={ratio_small:.3f}")
    print(f"sigma=1e6:   silent={rs_huge:.4f} driven={rd_huge:.4f} ratio={ratio_huge:.3f}")
    assert ratio_small < 0.5, (
        f"sigma=0.05 should be den-sensitive (large silent/driven contrast), got ratio={ratio_small}")
    assert 0.85 < ratio_huge < 1.15, (
        f"sigma=1e6 should be sigma-DOMINATED (silent/driven contrast collapses to ~1.0), "
        f"got ratio={ratio_huge} -- den is still mattering at an absurdly large sigma")
