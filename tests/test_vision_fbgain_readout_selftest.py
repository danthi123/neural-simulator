"""Local, pure-numpy selftest for the SPIKING FEEDBACK DIVISIVE GAIN-CONTROL readout
(research/runners/_vision_lindiscrim_readout_derisk.py's `--readout attention-gated-soft-fbgain` +
research/runners/_vision_hmax_spiking_derisk.py's `lif_spike_read_fbgain`).

Pre-registration: research/findings/2026-09-23-vision-configural-binding-spiking-feedback-divisive-gain-
control-readout-PREREGISTERED.md. Built after banking a 12/12-seed collapse of `--readout
attention-gated-soft` (research/findings/2026-09-23-vision-attention-gated-soft-readout-spiking-port-
collapse-NOGO-banked.md) across two front-end operating points -- LEARNED_spkwta_held pinned to exact
chance while LEARNED_linscore_held (a separate, never-gated pathway) retained real signal.

Every test is built to FAIL in its failing direction, not merely pass by construction:
  - `test_fbgain_disabled_reproduces_lif_spike_read_exactly` / `test_readout_disabled_reproduces_linear_
    exactly` pin BOTH independent byte-identical-off levers (fb_strength<=0 delegates to the plain LIF
    stepper; attn_gain_exponent<=0 short-circuits the gain template), including the COMBINATION.
  - `test_fbgain_actually_changes_output_vs_gain_only` would fail if the feedback loop were a no-op
    (e.g. `r_fb` never updated, or `I_eff` computed but not actually used) by comparing the SAME
    gain-shaped drive read WITH vs WITHOUT the feedback loop.
  - `test_feedback_trace_is_driven_by_real_spikes_not_a_constant` directly inspects `lif_spike_read_
    fbgain`'s r_fb machinery via a controlled two-population contrast (high-drive vs zero-drive rows
    must diverge in spike count once feedback couples them, ruling out an accidentally-disconnected
    r_fb that never influences `I_eff`).
  - `test_class_read_dispatcher_routes_fbgain` pins the `_class_read` dispatch table.
"""
from __future__ import annotations

import argparse

import numpy as np

from research.runners._vision_hmax_spiking_derisk import lif_spike_read, lif_spike_read_fbgain
from research.runners._vision_lindiscrim_readout_derisk import (
    _attention_gated_soft_fbgain_class_read,
    _class_read,
    _spiking_class_read,
)


def _make_args(**overrides):
    a = argparse.Namespace(
        class_pop=6,
        read_gain=2.5,
        read_bias=1.0,
        T_read=16,
        tau=8.0,
        v_thresh=1.0,
        t_ref=2,
        noise=0.06,
        attn_gain_exponent=1.0,
        fb_strength=0.0,
        fb_tau=8.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _synthetic_problem(seed):
    rng = np.random.default_rng(seed)
    n_classes, D, N = 4, 20, 24
    r = np.abs(rng.standard_normal((N, D))).astype(np.float32) + 0.05
    V = rng.standard_normal((n_classes, D)).astype(np.float32)
    b = rng.standard_normal(n_classes).astype(np.float32)
    mu = rng.standard_normal(D).astype(np.float32) * 0.1
    sd = (np.abs(rng.standard_normal(D)).astype(np.float32) + 0.5)
    return r, V, b, mu, sd, n_classes, D


# ------------------------------------------------------------------------------------------------
# lif_spike_read_fbgain in isolation
# ------------------------------------------------------------------------------------------------
def test_fbgain_disabled_reproduces_lif_spike_read_exactly():
    """fb_strength<=0 must delegate to lif_spike_read VERBATIM (identical RNG draw order too)."""
    rng = np.random.default_rng(9)
    drive = np.abs(rng.standard_normal((30, 8))).astype(np.float32)
    for fb_strength in (0.0, -1.0, -5.0):
        c1, f1 = lif_spike_read(drive, T=20, seed=123, tau=8.0, v_thresh=1.0, t_ref=2, noise=0.06, gain=1.0)
        c2, f2 = lif_spike_read_fbgain(drive, T=20, seed=123, tau=8.0, v_thresh=1.0, t_ref=2, noise=0.06,
                                        gain=1.0, fb_strength=fb_strength, fb_tau=8.0)
        assert np.array_equal(c1, c2), "fb_strength<=0 must reproduce lif_spike_read's counts exactly"
        assert np.array_equal(f1, f2), "fb_strength<=0 must reproduce lif_spike_read's first-spike times exactly"


def test_feedback_trace_is_driven_by_real_spikes_not_a_constant():
    """A row with strong drive in every column must accumulate a LARGER pooled r_fb trace than a row with
    weak drive, so a strong-drive row's own feedback-induced suppression must show up as a SMALLER extra
    boost from raising fb_strength than a weak-drive row gets (the strong row is already saturating, the
    weak row has more headroom) -- this fails if r_fb is computed but never actually coupled into I_eff,
    or if it is a constant that ignores the actual spikes."""
    T, C = 60, 5
    strong = np.full((1, C), 3.0, dtype=np.float32)
    weak = np.full((1, C), 0.3, dtype=np.float32)
    drive = np.concatenate([strong, weak], axis=0)   # (2, C): row 0 strong, row 1 weak

    c_off, _ = lif_spike_read_fbgain(drive, T=T, seed=7, fb_strength=0.0, fb_tau=8.0)
    c_on, _ = lif_spike_read_fbgain(drive, T=T, seed=7, fb_strength=3.0, fb_tau=8.0)

    assert c_off[0].mean() > c_off[1].mean(), "sanity: the strong row must out-spike the weak row with feedback off"
    # Feedback is POOLED PER ROW (per trial), so it must suppress the strong row's own high spike rate
    # (large r_fb from its own output) proportionally more than the weak row's low rate does -- i.e. the
    # RATIO strong/weak must shrink once feedback is on, not merely "some numbers changed".
    ratio_off = c_off[0].mean() / max(c_off[1].mean(), 1e-6)
    ratio_on = c_on[0].mean() / max(c_on[1].mean(), 1e-6)
    assert not np.array_equal(c_off, c_on), "fb_strength>0 must change spike counts vs fb_strength=0"
    assert ratio_on <= ratio_off, (
        "divisive feedback pooled per-row must compress the strong/weak spike-count ratio (self-"
        "normalizing), not widen it -- a ratio_on > ratio_off would mean the 'inhibition' is amplifying "
        "the already-stronger row instead of dividing it down")


# ------------------------------------------------------------------------------------------------
# _attention_gated_soft_fbgain_class_read (the full readout)
# ------------------------------------------------------------------------------------------------
def test_readout_disabled_reproduces_linear_exactly():
    """BOTH levers off (attn_gain_exponent<=0 AND fb_strength<=0) must reproduce _spiking_class_read
    (--readout linear) bit-for-bit -- the full byte-identical-off chain."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=1)
    a_off = _make_args(attn_gain_exponent=0.0, fb_strength=0.0)
    base_seed = 42

    pred_lin, sp_lin = _spiking_class_read(r, V, b, mu, sd, a_off, code="count", base_seed=base_seed)
    pred_fb, sp_fb = _attention_gated_soft_fbgain_class_read(
        r, V, b, mu, sd, a_off, code="count", base_seed=base_seed)

    assert np.array_equal(pred_lin, pred_fb)
    assert np.array_equal(sp_lin, sp_fb)

    # a negative exponent must take the identical short-circuit branch too
    a_neg = _make_args(attn_gain_exponent=-2.0, fb_strength=0.0)
    pred_neg, sp_neg = _attention_gated_soft_fbgain_class_read(
        r, V, b, mu, sd, a_neg, code="count", base_seed=base_seed)
    assert np.array_equal(pred_lin, pred_neg)
    assert np.array_equal(sp_lin, sp_neg)


def test_fbgain_actually_changes_output_vs_gain_only():
    """Gain ON + feedback OFF vs gain ON + feedback ON must differ (the feedback loop's own causal
    contribution, isolated with the gain template held fixed) -- this fails if the feedback machinery
    is wired but never actually reaches the LIF current (a no-op bug hiding behind the interface)."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=4)
    base_seed = 42
    a_gain_only = _make_args(attn_gain_exponent=1.0, fb_strength=0.0)
    a_gain_fb = _make_args(attn_gain_exponent=1.0, fb_strength=2.0)

    pred_a, sp_a = _attention_gated_soft_fbgain_class_read(
        r, V, b, mu, sd, a_gain_only, code="count", base_seed=base_seed)
    pred_b, sp_b = _attention_gated_soft_fbgain_class_read(
        r, V, b, mu, sd, a_gain_fb, code="count", base_seed=base_seed)

    assert not np.array_equal(sp_a, sp_b), "fb_strength>0 must change the class-spike-count read vs fb_strength=0"
    assert np.all(np.isfinite(sp_b)) and sp_b.shape == (r.shape[0], n_classes)


def test_class_read_dispatcher_routes_fbgain():
    """Pin the _class_read dispatcher: --readout attention-gated-soft-fbgain must route to
    _attention_gated_soft_fbgain_class_read, not silently fall back to another mode."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=5)
    base_seed = 42
    a = _make_args(attn_gain_exponent=1.0, fb_strength=2.0, readout="attention-gated-soft-fbgain")

    pred_direct, sp_direct = _attention_gated_soft_fbgain_class_read(
        r, V, b, mu, sd, a, code="count", base_seed=base_seed)
    pred_dispatch, sp_dispatch = _class_read(r, V, b, mu, sd, a, code="count", base_seed=base_seed)

    assert np.array_equal(pred_direct, pred_dispatch)
    assert np.array_equal(sp_direct, sp_dispatch)
