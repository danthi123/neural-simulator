"""PATHWAY TRACE for the N=5 wall: where does the salience differentiation vanish?

The motor-pool gain sweep (`_tone_bg_nchannel_motor_gain_sweep`) showed the motor AND commit pools are
BYTE-IDENTICAL across the salience gain (600 -> 38400, a 64x range): the salience bias is completely decoupled
from the decision pools. The earlier D1 diagnosis showed the striatal-D1 pools DO differentiate by salience at the
default gain. So the differentiation is lost somewhere in D1 -> GPi -> thalamus -> commit -> motor.

This trace measures the FULL-run firing at EVERY pathway step (D1, GPi, thalamus, commit, motor) at the
sub-threshold tonic (scale 0.05, gpi=50, thal=45), v_held=+0.9, default gain, so the per-channel spread at each
step is visible. The step where the spread collapses (from the D1's salience-tracking spread to a flat/noisy floor)
is the bottleneck synapse.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_pathway_trace
"""
from __future__ import annotations

import os
import dataclasses

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np

from sim.backend import get_backend, to_host
from research.runners._tone_bg_nchannel_selector import (
    build_nchannel_bridge,
    _set_nchannel_tonic,
    _step,
    congruence_saliences,
    host_argmin,
    CAND_VALENCES,
    N_CHANNELS,
    SALIENCE_GAIN_PA,
    selector_config,
)
from research.runners._vocal_action_selector_gate import _indices

DEV_SEED = 7
VH = +0.9
GPI_TONIC_PA = 50.0
THAL_TONIC_PA = 45.0
# pathway steps in order, with the region-name template (per channel)
PATHWAY = [
    ("str_d1_{ch}", "D1 (salience bias applied here)"),
    ("gpi_{ch}", "GPi (D1 inhibits GPi)"),
    ("thal_{ch}", "thalamus (GPi inhibits thal)"),
    ("commit_{ch}", "commit pool (thal drives commit)"),
    ("motor_{ch}", "motor pool (commit drives motor; the DECISION pool)"),
]


def _pathway_firing(bridge, config, saliences, channels, *, gain=SALIENCE_GAIN_PA, n_steps=None):
    """Full-run (no early break) per-channel firing at every pathway step."""
    xp, _ = get_backend()
    if n_steps is None:
        n_steps = int(config.action_steps)
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = xp.float32(config.practice_pA)
    for ch in channels:
        idx = xp.asarray(_indices(bridge, f"str_d1_{ch}"))
        bridge.cp_external_input_current[idx] = (
            bridge.cp_external_input_current[idx] + xp.float32(float(saliences[ch]) * gain)
        )
    counts = {name: np.zeros(len(channels), dtype=np.int64) for name, _ in PATHWAY}
    for step in range(n_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for i, ch in enumerate(channels):
            for name, _ in PATHWAY:
                counts[name][i] += int(firing[_indices(bridge, name.format(ch=ch))].sum())
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_nchannel_tonic(bridge, config, channels)
    _step(bridge, config.washout_steps)
    return {name: c.tolist() for name, c in counts.items()}


def _spread(vals):
    """max/min ratio (the differentiation) + the argmax channel."""
    v = np.asarray(vals, dtype=float)
    vmin = max(1.0, v.min())
    return v.max() / vmin, int(np.argmax(v))


def main():
    config = selector_config("v2")
    cfg = dataclasses.replace(config, gpi_tonic_pA=GPI_TONIC_PA, thalamus_tonic_pA=THAL_TONIC_PA)
    channels = tuple(range(N_CHANNELS))
    sal = congruence_saliences(CAND_VALENCES, VH)
    expected = host_argmin(CAND_VALENCES, VH)

    print(f"=== pathway trace (sub-threshold tonic), dev seed {DEV_SEED} ===", flush=True)
    print(f"  tonic: gpi={GPI_TONIC_PA}, thal={THAL_TONIC_PA}; v_held={VH:+.1f}; expected winner={expected}", flush=True)
    print(f"  saliences={[round(s, 2) for s in sal]}", flush=True)

    b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
    _set_nchannel_tonic(b, cfg, channels)
    _step(b, cfg.warmup_steps)
    firing = _pathway_firing(b, cfg, sal, channels)

    print(f"\n  {'step':<40} {'cand0':>6} {'cand1':>6} {'cand2':>6} {'cand3':>6} {'cand4':>6}  {'max/min':>8} {'argmax':>6}", flush=True)
    for name, label in PATHWAY:
        vals = firing[name]
        ratio, argmax = _spread(vals)
        marker = "  <-- expected winner" if argmax == expected else ""
        print(f"  {label:<40} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6} {vals[4]:>6}  "
              f"{ratio:>8.2f} {argmax:>6}{marker}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
