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


def _reference_fbgain_with_pinned_rfb(drive, T, seed, pinned_rfb, tau=8.0, v_thresh=1.0, t_ref=2,
                                       noise=0.06, gain=1.0, fb_strength=0.0, fb_tau=8.0):
    """Byte-for-byte the SAME stepping loop as `lif_spike_read_fbgain` (research/runners/
    _vision_hmax_spiking_derisk.py), with exactly ONE line changed: the feedback trace's update target is
    the CONSTANT `pinned_rfb` instead of the real population's own `spk.mean(axis=1)`. This is the
    reviewer's exact mutation (2026-09-24 re-review: 'spk.mean(axis=1) -> 0.5 passed all 5 tests'),
    reproduced here as an independent reference so the real test below can assert the PRODUCTION function
    does NOT match what a disconnected-constant r_fb would produce."""
    rng = np.random.default_rng(seed)
    M, C = drive.shape
    v = np.zeros((M, C), dtype=np.float32)
    ref = np.zeros((M, C), dtype=np.int32)
    counts = np.zeros((M, C), dtype=np.float32)
    first = np.full((M, C), float(T), dtype=np.float32)
    I0 = (gain * drive).astype(np.float32)
    r_fb = np.zeros(M, dtype=np.float32)
    for t in range(int(T)):
        can = ref <= 0
        I_eff = (I0 / (1.0 + fb_strength * r_fb[:, None])).astype(np.float32)
        v = np.where(can, v + (1.0 / tau) * (-v + I_eff) + rng.standard_normal((M, C)).astype(np.float32) * noise, v)
        spk = can & (v >= v_thresh)
        counts += spk
        newf = spk & (first >= T)
        first = np.where(newf, float(t), first)
        v = np.where(spk, 0.0, v)
        ref = np.where(spk, t_ref, ref - 1)
        r_fb = r_fb + (1.0 / fb_tau) * (-r_fb + pinned_rfb)          # <-- the mutation, isolated here
    return counts, first


def test_feedback_is_zero_until_the_rows_first_spike():
    """GENERAL guard against a disconnected r_fb (any constant, not only the 0.5 a prior review named). A
    spike-driven, row-pooled r_fb is exactly 0 until that row's first spike, and the RNG draws are identical with
    feedback on or off, so each row's FIRST spike time must be identical at fb_strength=3.0 and 0.0. Any positive
    constant r_fb reduces the drive from t=0 and delays it (re-review round 4 measured 0.1 -> [17,22,2],
    0.25 -> [55,60,3], 0.5 -> [60,60,3] vs the real [11,14,2]). The 0.0 constant is caught by the count-change
    assertion in the test below."""
    drive = np.array([[1.3] * 5, [1.1] * 5, [3.0] * 5], dtype=np.float32)
    T, seed, fb_tau = 60, 7, 8.0
    _, first_on = lif_spike_read_fbgain(drive, T=T, seed=seed, fb_strength=3.0, fb_tau=fb_tau)
    _, first_off = lif_spike_read_fbgain(drive, T=T, seed=seed, fb_strength=0.0, fb_tau=fb_tau)
    assert np.array_equal(first_on.min(axis=1), first_off.min(axis=1)), (
        f"a row's first spike moved when feedback was switched on ({first_on.min(axis=1)} vs "
        f"{first_off.min(axis=1)}): r_fb is non-zero before any spike, i.e. not driven by the row's own spikes")
    assert (first_off.min(axis=1) < T).all(), "fixture drift: every row must spike with feedback off"


def test_feedback_trace_is_driven_by_real_spikes_not_a_constant():
    """DIRECTLY catches the reviewer's exact mutation (r_fb's update target `spk.mean(axis=1)` replaced by
    a constant, e.g. 0.5) by comparing the PRODUCTION function against `_reference_fbgain_with_pinned_rfb`
    run with the SAME seed/params but r_fb pinned to that constant: if production's own r_fb were also a
    disconnected constant, the two would be BYTE-IDENTICAL (both take the identical RNG draws and the
    identical I_eff formula, differing only in what feeds r_fb). Uses two rows engineered to have real
    per-row spike fractions far from the pinned constant on BOTH sides (a saturating-strong row whose true
    mean spike fraction is near 1.0, and a near-silent weak row whose true mean spike fraction is near 0.0),
    so a real activity-coupled r_fb is GUARANTEED to diverge from a pinned 0.5 -- this is the 'compare r_fb
    against the actual spike counts across two inputs with different spiking' check."""
    # drive=3.0 keeps the strong row's own current close enough to v_thresh=1.0 that the EXACT r_fb value
    # feeding I_eff = I0/(1+fb_strength*r_fb) changes spike timing (unlike a saturating/floor drive, where
    # any r_fb in a modest range produces the identical spike train and the constant-vs-real contrast below
    # would be undiscriminating regardless of which mutation is present).
    T, C = 60, 5
    strong = np.full((1, C), 3.0, dtype=np.float32)
    weak = np.full((1, C), 0.3, dtype=np.float32)
    drive = np.concatenate([strong, weak], axis=0)   # (2, C): row 0 strong, row 1 weak
    fb_strength, fb_tau, seed = 3.0, 8.0, 7

    c_on, _ = lif_spike_read_fbgain(drive, T=T, seed=seed, fb_strength=fb_strength, fb_tau=fb_tau)
    c_off, _ = lif_spike_read_fbgain(drive, T=T, seed=seed, fb_strength=0.0, fb_tau=fb_tau)

    # Ground truth: what each row's OWN mean spike fraction actually is with feedback off, i.e. what a
    # REAL per-row r_fb converges toward. Confirm the strong row spikes at a real, non-trivial, non-0.5
    # rate and the weak row (below v_thresh even with noise) stays silent -- both far from a pinned 0.5.
    strong_rate = c_off[0].mean() / T
    weak_rate = c_off[1].mean() / T
    assert 0.05 < strong_rate < 0.45, f"fixture drift: strong row's real rate must be non-trivial and != 0.5 (got {strong_rate})"
    assert weak_rate == 0.0, f"fixture drift: weak row must stay silent so its real r_fb is exactly 0 (got {weak_rate})"

    # THE CATCH: pin r_fb to a constant chosen deliberately far from BOTH rows' real spike fractions. If
    # production's r_fb were that same disconnected constant (the reviewer's mutation), production's
    # counts would equal this reference's counts EXACTLY (identical RNG, identical I_eff algebra). A real,
    # activity-coupled r_fb must diverge from a constant fed with either row's own actual rate.
    for pinned in (0.5, strong_rate, weak_rate):
        c_pinned, _ = _reference_fbgain_with_pinned_rfb(drive, T=T, seed=seed, pinned_rfb=pinned,
                                                         fb_strength=fb_strength, fb_tau=fb_tau)
        assert not np.array_equal(c_on, c_pinned), (
            f"lif_spike_read_fbgain's counts are BYTE-IDENTICAL to a reference whose r_fb is pinned to the "
            f"constant {pinned} instead of tracking the row's own spikes -- r_fb is disconnected from the "
            f"actual spike train (this is exactly the 'spk.mean(axis=1) -> constant' mutation)")

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
