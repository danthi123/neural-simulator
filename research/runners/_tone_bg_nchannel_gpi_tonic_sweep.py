"""GPi-TONIC sweep for the N=5 wall: does raising the GPi's tonic rate propagate the D1 differentiation?

The pathway trace (`_tone_bg_nchannel_pathway_trace`) localised the wall: the salience bias differentiates the
striatal-D1 pool PERFECTLY (D1=[452,913,702,417,210], argmax = the expected winner), but the GPi sits near its
floor (8-28 spikes over 600 steps, nearly silent) at the sub-threshold tonic (gpi=50, thal=45), so its D1-driven
differentiation is too small to modulate the thalamus (which fires flat at its tonic rate). The differentiation
dies at the GPi->thalamus step.

This sweep tests the no-defer lever: hold the thalamic tonic FIXED at the sub-threshold value (45 pA) and raise
ONLY the GPi tonic (50 -> 1000 pA). In the real basal ganglia the GPi fires at a HIGH tonic rate (~20-50 Hz) and
the direct (D1) pathway REDUCES it for the selected action; the model's GPi at gpi=50 is unfaithfully silent. If
raising the GPi tonic gives the D1 inhibition "room" to work, the GPi's per-channel differentiation grows and
propagates to thalamus -> commit -> motor, making the motor pool track the saliences (argmax -> expected winner).

Each level is an independent fresh bridge (same dev seed -> the same network, only the tonic operating point
differs), warmed up at its tonic, then the FULL-run (no early break) firing is measured at every pathway step.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_gpi_tonic_sweep
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
THAL_TONIC_PA = 45.0  # FIXED at the sub-threshold window's thalamic value (the pathway trace's operating point)
GPI_SWEEP_PA = [50.0, 100.0, 200.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 800.0, 1000.0]
CLEAN_LOSER_RATIO = 0.25  # the config's clean-loser ratio (the commit read)
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
    # restore the resting tonic (no reset/washout needed: each level uses a fresh bridge)
    _set_nchannel_tonic(bridge, config, channels)
    return {name: c.tolist() for name, c in counts.items()}


def _spread(vals):
    """max/min ratio (the differentiation) + the argmax channel."""
    v = np.asarray(vals, dtype=float)
    vmin = max(1.0, v.min())
    return v.max() / vmin, int(np.argmax(v))


def _motor_commit_read(motor_vals, expected):
    """The commit read applied to the FULL-run motor counts: the argmax channel wins iff EVERY other channel is
    suppressed below the clean-loser ratio (the `_run_nchannel_trial` read, without the early break)."""
    v = np.asarray(motor_vals, dtype=float)
    winner = int(np.argmax(v))
    losers = [v[i] for i in range(len(v)) if i != winner]
    loser_ratio = max(losers) / max(1.0, v[winner]) if losers else 0.0
    clean = loser_ratio <= CLEAN_LOSER_RATIO
    return winner, float(loser_ratio), bool(clean), (winner == expected)


def main():
    base = selector_config("v2")
    channels = tuple(range(N_CHANNELS))
    sal = congruence_saliences(CAND_VALENCES, VH)
    expected = host_argmin(CAND_VALENCES, VH)

    print(f"=== GPi-tonic sweep (thal fixed at {THAL_TONIC_PA} pA), dev seed {DEV_SEED} ===", flush=True)
    print(f"  v_held={VH:+.1f}; expected winner={expected}; saliences={[round(s, 2) for s in sal]}", flush=True)
    print(f"  gpi_tonic sweep (pA): {GPI_SWEEP_PA}", flush=True)

    summary = []
    for gpi in GPI_SWEEP_PA:
        cfg = dataclasses.replace(base, gpi_tonic_pA=gpi, thalamus_tonic_pA=THAL_TONIC_PA)
        # fresh bridge per level: same dev seed -> same network, only the tonic operating point differs (controlled)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)
        firing = _pathway_firing(b, cfg, sal, channels)

        print(f"\n  --- gpi_tonic = {gpi:.0f} pA (thal={THAL_TONIC_PA:.0f}) ---", flush=True)
        print(f"    {'step':<40} {'cand0':>6} {'cand1':>6} {'cand2':>6} {'cand3':>6} {'cand4':>6}  {'max/min':>8} {'argmax':>6}", flush=True)
        for name, label in PATHWAY:
            vals = firing[name]
            ratio, argmax = _spread(vals)
            marker = "  <-- expected winner" if argmax == expected else ""
            print(f"    {label:<40} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6} {vals[4]:>6}  "
                  f"{ratio:>8.2f} {argmax:>6}{marker}", flush=True)

        motor_winner, loser_ratio, clean, matches = _motor_commit_read(firing["motor_{ch}"], expected)
        summary.append((gpi, firing["gpi_{ch}"], firing["motor_{ch}"], motor_winner, loser_ratio, clean, matches))

    # summary: does the motor pool start tracking the saliences as the GPi tonic rises?
    print(f"\n=== summary: motor-pool decision read as the GPi tonic rises (thal fixed at {THAL_TONIC_PA:.0f}) ===", flush=True)
    print(f"  {'gpi':>6} | {'GPi (cand0..4)':<38} | {'motor (cand0..4)':<38} | {'winner':>6} {'loser%':>7} {'clean':>6} {'==exp':>6}", flush=True)
    for gpi, gpi_vals, motor_vals, winner, loser_ratio, clean, matches in summary:
        gpi_str = "[" + ", ".join(f"{v}" for v in gpi_vals) + "]"
        motor_str = "[" + ", ".join(f"{v}" for v in motor_vals) + "]"
        print(f"  {gpi:>6.0f} | {gpi_str:<38} | {motor_str:<38} | {winner:>6} {100*loser_ratio:>6.1f}% {str(clean):>6} {str(matches):>6}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
