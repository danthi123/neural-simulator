"""Local, pure-numpy selftest for `_attention_gated_soft_class_read` (research/runners/
_vision_lindiscrim_readout_derisk.py's `--readout attention-gated-soft` graded/divisive-norm attention-gain
readout, built 2026-09-16 as the named next rung after the hard-k-WTA `--readout attention-gated` regressed --
research/findings/2026-09-09-vision-configural-binding-attention-gated-readout-NEXT-MECHANISM-PREREGISTERED.md).

Pre-registration: research/findings/2026-09-23-vision-configural-binding-attention-gated-soft-readout-
NEXT-MECHANISM-PREREGISTERED.md. This mechanism shipped on `main` (commit b7108ca04) with a manual
"byte-identical-off proof" written into the module docstring, but no automated test ever pinned it -- a
regression in `--attn-gain-exponent <= 0`'s short-circuit, or in the enabled path collapsing to a no-op,
would have shipped silently. This file makes both directions a CI-checked selftest, sub-second, no
SimulationBridge / CoreSimConfig / subprocess.

Each test is built so it FAILS in its failing direction (not just passes by construction):
  - `test_disabled_exponent_reproduces_plain_linear_read_exactly` would fail the moment the
    `attn_gain_exponent <= 0` short-circuit stopped being an exact `gated = r` (e.g. if someone
    "simplified" it to `A_c**0` composed with the satdiv step, which is NOT the identity -- the module
    docstring explicitly calls this out as the wrong, non-byte-identical shortcut).
  - `test_enabled_exponent_is_not_a_relabelled_linear_read` would fail if the gain/normalization path
    ever collapsed to returning `r` unchanged (a degenerate no-op that would silently pass every other
    check while doing nothing).
  - `test_class_read_dispatcher_routes_attention_gated_soft` pins the `_class_read` dispatch table itself,
    so a future refactor that drops the `"attention-gated-soft"` branch (or points it at the wrong
    function) is caught here rather than only in a 10-minute decisive run.
"""
from __future__ import annotations

import argparse

import numpy as np

from research.runners._vision_lindiscrim_readout_derisk import (
    _attention_gated_soft_class_read,
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
        attn_satdiv_n=2.0,
        attn_satdiv_sigma_frac=0.25,
        attn_satdiv_scale_mult=1.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _synthetic_problem(seed):
    """A small (N, D) spike-rate-like code r, an (n_classes, D) fitted discriminant V/b, and per-feature
    standardisation mu/sd -- the exact call contract every `*_class_read` function in this file shares."""
    rng = np.random.default_rng(seed)
    n_classes, D, N = 4, 20, 24
    r = np.abs(rng.standard_normal((N, D))).astype(np.float32) + 0.05   # nonneg "spike rate"-like drive
    V = rng.standard_normal((n_classes, D)).astype(np.float32)
    b = rng.standard_normal(n_classes).astype(np.float32)
    mu = rng.standard_normal(D).astype(np.float32) * 0.1
    sd = (np.abs(rng.standard_normal(D)).astype(np.float32) + 0.5)      # keep away from 0
    return r, V, b, mu, sd, n_classes, D


def test_disabled_exponent_reproduces_plain_linear_read_exactly():
    """BYTE-IDENTICAL-OFF: `attn_gain_exponent <= 0` must short-circuit to `gated = r` BEFORE either the
    gain multiply or the satdiv-normalize formula runs, so `--readout attention-gated-soft
    --attn-gain-exponent 0` must reproduce `_spiking_class_read` (`--readout linear`) bit-for-bit -- same
    base_seed, so the downstream LIF spike-read draws identically too."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=1)
    a_off = _make_args(attn_gain_exponent=0.0)
    base_seed = 42

    pred_lin, sp_lin = _spiking_class_read(r, V, b, mu, sd, a_off, code="count", base_seed=base_seed)
    pred_soft, sp_soft = _attention_gated_soft_class_read(
        r, V, b, mu, sd, a_off, code="count", base_seed=base_seed)

    assert np.array_equal(pred_lin, pred_soft), "disabled exponent must reproduce the SAME predictions"
    assert np.array_equal(sp_lin, sp_soft), "disabled exponent must reproduce the SAME class-spike counts"

    # ALSO check a negative exponent (not just exactly 0.0) takes the identical short-circuit branch.
    a_neg = _make_args(attn_gain_exponent=-3.0)
    pred_neg, sp_neg = _attention_gated_soft_class_read(
        r, V, b, mu, sd, a_neg, code="count", base_seed=base_seed)
    assert np.array_equal(pred_lin, pred_neg)
    assert np.array_equal(sp_lin, sp_neg)


def test_enabled_exponent_is_not_a_relabelled_linear_read():
    """NON-DEGENERATE: at the real default (`attn_gain_exponent=1.0`), the gain+satdiv path must actually
    transform the drive -- if it silently collapsed to `gated == r` (a no-op bug hiding behind the same
    interface), this must fail, not pass by construction."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=2)
    a_on = _make_args(attn_gain_exponent=1.0)
    base_seed = 42

    pred_lin, sp_lin = _spiking_class_read(r, V, b, mu, sd, a_on, code="count", base_seed=base_seed)
    pred_soft, sp_soft = _attention_gated_soft_class_read(
        r, V, b, mu, sd, a_on, code="count", base_seed=base_seed)

    assert not np.array_equal(sp_lin, sp_soft), (
        "enabled attention-gated-soft must produce a DIFFERENT class-spike-count read than the plain "
        "linear read on this synthetic non-degenerate problem -- identical output would mean the gate "
        "is a no-op, not that it is 'off'")
    assert np.all(np.isfinite(sp_soft)) and sp_soft.shape == (r.shape[0], n_classes)
    # a collapsed/degenerate read would predict the SAME class for every trial -- on this synthetic
    # problem (4 classes, random discriminant) that would be a striking, checkable failure mode.
    assert len(np.unique(pred_soft)) > 1, "predictions must not collapse to a single class"


def test_class_read_dispatcher_routes_attention_gated_soft():
    """Pin the `_class_read` dispatcher itself: `--readout attention-gated-soft` must route to
    `_attention_gated_soft_class_read`, not silently fall back to `linear` or the hard-gated function --
    the exact failure mode that would make a future refactor's decisive run silently re-measure the
    WRONG mechanism under the right flag name."""
    r, V, b, mu, sd, n_classes, D = _synthetic_problem(seed=3)
    base_seed = 42
    a = _make_args(attn_gain_exponent=1.0, readout="attention-gated-soft")

    pred_direct, sp_direct = _attention_gated_soft_class_read(
        r, V, b, mu, sd, a, code="count", base_seed=base_seed)
    pred_dispatch, sp_dispatch = _class_read(r, V, b, mu, sd, a, code="count", base_seed=base_seed)

    assert np.array_equal(pred_direct, pred_dispatch)
    assert np.array_equal(sp_direct, sp_dispatch)
