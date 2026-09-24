"""Pins the gap#4 on-bridge feedforward-transmission defect found 2026-09-24 (plan step S19), with the review fixes.

At the legacy config the explicit feedforward synapses of the on-bridge BDSP net (OnBridgeBDSPNet -> Gap4InEngineNet ->
Gap4ReadoutNet) carry almost nothing, and it takes TWO changes together to make them transmit: bypassing the engine's
default Tsodyks-Markram short-term depression on those synapses (`--no-ff-stp`) AND the stronger feedforward gain
(ff_w_init 40, propagation 0.5). Either change alone leaves the input->H1 cut indistinguishable from run-to-run noise.
So the cause is two-factor: default STP at the legacy gain, not STP alone. (The first version of this test and of the
FAILURE_LOG row said "STP owns the block"; the full-size probe's own ff_stp_bypassed variant contradicts that.)

The statistic is per unit and has a noise reference (research/runners/_gap4_tc_transmission_noise_ref_diag.py):
effect/noise = mean |intact - input-cut| / mean |intact - intact|, and the across-item reliability of each unit's read.
Checked at tonic 0 AND at the 2026-09-15 operating point (tonic 450 / 500 pA). Dev seed 7, tiny net, numpy; seconds.
This test fails if any direction changes: a single factor starts transmitting (the two-factor diagnosis would be stale)
or the calibrated operating point stops transmitting.
"""
import os

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")


def _h1_stats(variant, tonic_scale):
    from research.runners._gap4_tc_transmission_noise_ref_diag import build, transmission_stats
    from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
    (Xtr, _ytr, _), _te, meta, _idx = make_task_semantic_inheritance(7, n_super=8, n_members=4, held_per_super=1,
                                                                     n_prop=2, member_id_dim=3, n_obs=4)
    net = build(variant, Xtr.shape[1], 8, 2, int(meta["k_classes"]), tonic_scale, "spikes", seed=7)
    st = transmission_stats(net, Xtr[:12])
    assert st["weights_restored"]
    return st["h1"]


@pytest.mark.parametrize("tonic_scale", [0.0, 1.0])
def test_feedforward_transmits_only_with_both_factors(tonic_scale):
    res = {v: _h1_stats(v, tonic_scale) for v in ("legacy", "stp_bypass", "gain", "stp_bypass_gain")}
    print({v: (round(h["effect_over_noise"], 2), round(h["item_reliability"], 2)) for v, h in res.items()})
    for v in ("legacy", "stp_bypass", "gain"):
        # a single factor (or neither): cutting the input weights looks like noise, and no unit's read is item-driven
        assert res[v]["effect_over_noise"] < 1.4, (v, res[v])
        assert res[v]["item_reliability"] < 0.3, (v, res[v])
    both = res["stp_bypass_gain"]
    # both factors: the cut is several times the noise and each unit's item-to-item pattern is reproducible
    assert both["effect_over_noise"] > 2.0, both
    assert both["item_reliability"] > 0.6, both
