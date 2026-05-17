"""LOCAL Urbanczik-Senn somato-dendritic mismatch plasticity, apical-
gated. Delta-w ~ apical-gated (somatic_rate - phi(v_basal)) * pre.
When an apical_signal is supplied (top-down feedback delivered via the
neuron's FIXED-RANDOM B_apical -- NO weight transport), it sets the
local dendritic target the soma is pulled toward. Pure numpy;
biologically-LOCAL by construction; NO automatic differentiation, NO
reverse-mode, NO computational graph, NO imported oracle. ASCII
only."""
from __future__ import annotations
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def urbanczik_senn_update(pre_rate, soma_rate, v_basal,
                          apical_gate, apical_signal=None, lr=1.0):
    """pre_rate (n_pre,), soma_rate/v_basal/apical_gate (n_post,).
    apical_signal (n_post,) optional: the top-down teaching mismatch
    already projected through the FIXED-RANDOM apical feedback by the
    caller (no weight transport here). Returns dW (n_pre, n_post)."""
    pre = np.asarray(pre_rate, float)
    soma = np.asarray(soma_rate, float)
    vb = np.asarray(v_basal, float)
    gate = np.asarray(apical_gate, float)
    if apical_signal is None:
        mismatch = soma - _sig(vb)            # self-prediction error
    else:
        # apical-driven local target (GLR-2017): soma pulled toward
        # the FIXED-random-projected top-down signal. The top-down
        # signal is the positive output-error projected through the
        # FIXED-random apical feedback; the local descent step is the
        # negative of error * phi'(soma) * pre (still purely local --
        # only post-synaptic quantities + the random-projected
        # teaching signal; NO weight transport).
        mismatch = -np.asarray(apical_signal, float) * soma * (1.0 - soma)
    dw = np.outer(pre, lr * gate * mismatch)
    return dw
