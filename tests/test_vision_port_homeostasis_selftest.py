"""Selftest for the readout-port HOMEOSTASIS (`--port-homeostasis ip`) in
research/runners/_vision_lindiscrim_readout_derisk.py.

Pre-registration: research/findings/2026-09-24-vision-readout-port-homeostasis-intrinsic-plasticity-PREREGISTERED.md.

Each test is built to fail in its failing direction:
  - identity-at-init: the IP path at the init state (and with ip=None) must reproduce the NEUTRAL fbgain
    read bit-for-bit -- fails if the new current path perturbs a single value or the RNG order.
  - lesion: `ip_lesion_update` must leave the state at init -- fails if an update leaks through.
  - rescue: on a synthetic port that is collapsed WITHOUT homeostasis (asserted, not assumed), the learned
    state must yield a non-constant, above-chance read -- fails if the update rule is a no-op or diverges.
  - specific state: rolling the learned state by one class population must collapse the read again --
    fails if any generic shrink of the drive (not the per-unit learned state) were doing the work.
"""
from __future__ import annotations

import argparse

import numpy as np

from research.runners._vision_lindiscrim_readout_derisk import (
    _attention_gated_soft_fbgain_class_read,
    _attention_gated_soft_fbgain_ip_class_read,
    _class_read,
    _fbgain_pre_port_drive,
    _ip_permuted,
    _ip_port_current,
    _learn_port_homeostasis,
    _pred_entropy_bits,
)


def _args(**kw):
    a = argparse.Namespace(class_pop=6, read_gain=2.5, read_bias=1.0, T_read=48, tau=8.0, v_thresh=1.0,
                           t_ref=2, noise=0.06, attn_gain_exponent=1.0, fb_strength=0.0, fb_tau=8.0,
                           readout="attention-gated-soft-fbgain", ip_target_mean=0.25, ip_epochs=400,
                           ip_eta_theta=0.05, ip_eta_gain=0.01, ip_max_log_step=0.5, ip_lesion_update=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _collapsed_problem(seed, n_per=24):
    """A class-discriminative code with a LARGE common mode (the lane's own regime): the discriminant is
    fit on the un-gained code, so the per-class top-down gain leaves large trial-independent class offsets."""
    rng = np.random.default_rng(seed)
    n_classes, D = 4, 40
    protos = rng.standard_normal((n_classes, D)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per)
    r = (5000.0 + 3.0 * protos[y] + rng.standard_normal((y.size, D))).astype(np.float32)
    mu = r.mean(axis=0)
    sd = r.std(axis=0) + 1e-6
    X = (r - mu) / sd
    Y = np.eye(n_classes)[y] - 1.0 / n_classes
    W = np.linalg.solve(X.T @ X + 1.0 * y.size * np.eye(D), X.T @ Y)
    V = W.T.astype(np.float32)
    b = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    return r, y, V, b, mu.astype(np.float32), sd.astype(np.float32)


def test_ip_current_identity_at_init_and_none():
    rng = np.random.default_rng(3)
    net = (rng.standard_normal((20, 4)) * 50).astype(np.float32)
    M = 5
    ref = np.clip(np.repeat(net, M, axis=1), 0.0, None)
    init = {"log_gain": np.zeros(4 * M), "theta": np.zeros(4 * M)}
    assert np.array_equal(_ip_port_current(net, None, M), ref)
    out = _ip_port_current(net, init, M)
    assert out.dtype == np.float32 and np.array_equal(out, ref)


def test_ip_read_at_init_reproduces_neutral_fbgain_read_exactly():
    r, _, V, b, mu, sd = _collapsed_problem(11)
    for fb in (0.0, 1.0):
        a = _args(fb_strength=fb)
        init = {"log_gain": np.zeros(V.shape[0] * a.class_pop), "theta": np.zeros(V.shape[0] * a.class_pop)}
        p0, s0 = _attention_gated_soft_fbgain_class_read(r, V, b, mu, sd, a, "count", 1234)
        p1, s1 = _attention_gated_soft_fbgain_ip_class_read(r, V, b, mu, sd, a, "count", 1234, init)
        p2, s2 = _class_read(r, V, b, mu, sd, a, "count", 1234)          # ip=None default path
        assert np.array_equal(p0, p1) and np.array_equal(s0, s1)
        assert np.array_equal(p0, p2) and np.array_equal(s0, s2)


def test_lesion_update_leaves_state_at_init():
    r, _, V, b, mu, sd = _collapsed_problem(5)
    ip = _learn_port_homeostasis(r, V, b, mu, sd, _args(ip_lesion_update=True, ip_epochs=20), 99)
    assert not ip["log_gain"].any() and not ip["theta"].any()
    assert ip["diag"]["lesion_update"] is True


def test_homeostasis_rescues_a_collapsed_port_and_permuted_state_does_not():
    r, y, V, b, mu, sd = _collapsed_problem(7)
    a = _args()
    net = _fbgain_pre_port_drive(r, V, b, mu, sd, a)
    # precondition, asserted: WITHOUT homeostasis the port is collapsed (constant output)
    p_off, _ = _class_read(r, V, b, mu, sd, a, "count", 77)
    assert _pred_entropy_bits(p_off, 4) == 0.0, (net.mean(0), net.std(0))
    ip = _learn_port_homeostasis(r, V, b, mu, sd, a, 500)
    assert np.abs(ip["log_gain"]).max() > 0.1 or np.abs(ip["theta"]).max() > 0.1   # the rule moved
    p_on, _ = _class_read(r, V, b, mu, sd, a, "count", 77, ip=ip)
    assert _pred_entropy_bits(p_on, 4) >= 1.0
    assert (p_on == y).mean() >= 0.5
    p_perm, _ = _class_read(r, V, b, mu, sd, a, "count", 77, ip=_ip_permuted(ip, 4, a.class_pop))
    assert (p_perm == y).mean() <= 0.35


def test_ip_requires_fbgain_readout():
    r, _, V, b, mu, sd = _collapsed_problem(2)
    a = _args(readout="linear")
    init = {"log_gain": np.zeros(4 * a.class_pop), "theta": np.zeros(4 * a.class_pop)}
    try:
        _class_read(r, V, b, mu, sd, a, "count", 1, ip=init)
    except ValueError:
        return
    raise AssertionError("ip state with a non-fbgain readout must raise")
