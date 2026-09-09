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
        # the FIXED-random-projected top-down signal. apical_signal is
        # the positive output-error projected through the FIXED-random
        # apical feedback (caller side; NO weight transport here).
        # The rule returns the ASCENT-on-reward vector: under weight
        # transport dW1 == +g_true (the true gradient itself), so the
        # documented training convention W1 += lr*(-dW1) is exactly
        # gradient DESCENT. Still purely local -- only post-synaptic
        # quantities * the random-projected teaching signal.
        mismatch = np.asarray(apical_signal, float) * soma * (1.0 - soma)
    dw = np.outer(pre, lr * gate * mismatch)
    return dw


def selfpredicting_interneuron_update(int_rate, wpi, y_fixed, lr=1.0, xp=np):
    """Sacramento-Senn (2018) Eq.9 LOCAL self-prediction update for the SST-like
    interneuron's PLASTIC apical-cancellation weight W^PI (gap#4 RANK-1).

    int_rate (k,) = the interneuron drive (the network's own prediction / rate);
    wpi (k, H) = the plastic interneuron->apical weight the interneuron LEARNS;
    y_fixed (k, H) = the FIXED top-down feedback the interneuron learns to predict
    and cancel. The free-phase residual apical is v_free = int_rate @ (y_fixed -
    wpi) (H,), and the update is dW^PI = lr * outer(int_rate, v_free), which drives
    W^PI toward the self-predicting fixed point where int_rate @ wpi == int_rate @
    y_fixed -- i.e. the interneuron's cancellation matches the top-down and the
    RESIDUAL apical is SILENT when the network is already correct (Sacramento-Senn
    'apical silent when correct'). This is the in-engine analogue of the runner's
    former host-side W^PI update: the SAME local rule, now carried on the substrate
    arrays inside _run_one_simulation_step. TRANSPORT-FREE + LOCAL: reads ONLY
    int_rate (an activity), wpi and y_fixed -- NEVER a forward weight. Backend
    argument xp selects numpy (default) or the caller's cupy module so the update
    runs on whichever arrays the substrate holds. Returns dW^PI (k, H)."""
    r = xp.atleast_2d(int_rate)                   # (m, k), m>=1
    v_free = r @ (y_fixed - wpi)                  # (m, H) free-phase residual apical
    m = max(1, int(r.shape[0]))
    dwpi = lr * (r.T @ v_free) / float(m)         # (k, H) == wpi.shape
    return dwpi
