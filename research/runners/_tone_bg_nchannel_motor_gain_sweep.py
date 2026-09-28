"""MOTOR-POOL GAIN SWEEP for the N=5 wall: does a higher salience gain make the DECISION pool track the saliences?

The motor-pool tonic sweep showed the decision (motor) pools fire ~1100-2100 at the sub-threshold tonic (scale 0.05,
gpi=50, thal=45) but do NOT track the congruence saliences: at v_held=+0.9 the saliences are [0.55, 0.9, 0.7, 0.4,
0.2] (cand 1 expected to win) yet the motor pools are [2138, 1632, 717, 2035, 1136] -- cand 0 (salience 0.55) fires
the most, cand 2 (salience 0.7) the least. The salience bias (applied at the striatal-D1 pool) is swamped by the
proposal/tonic drive before it reaches the decision pool.

This sweep fixes the tonic at the sub-threshold window and raises the salience gain, measuring the FULL-run motor
pool (no early break) at v_held=+0.9. If a higher gain makes the motor pools track the saliences (the argmax motor
pool == the expected winner, and the max/second-max ratio grows), the gain is the lever. If the motor pools stay
un-correlated with the saliences even at 32x gain, the salience bias is structurally swamped and the suspect moves
to the proposal drive / the D1->GPi contrast.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_motor_gain_sweep
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
GAIN_SWEEP = (SALIENCE_GAIN_PA, 1200.0, 2400.0, 4800.0, 9600.0, 19200.0, 38400.0)


def _full_run_motor(bridge, config, saliences, channels, *, gain, n_steps=None):
    """Full-run (no early break) motor + commit pool firing, so the salience-tracking is visible over the whole run."""
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
    motor = np.zeros(len(channels), dtype=np.int64)
    commit = np.zeros(len(channels), dtype=np.int64)
    for step in range(n_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for i, ch in enumerate(channels):
            motor[i] += int(firing[_indices(bridge, f"motor_{ch}")].sum())
            commit[i] += int(firing[_indices(bridge, f"commit_{ch}")].sum())
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_nchannel_tonic(bridge, config, channels)
    _step(bridge, config.washout_steps)
    return motor.tolist(), commit.tolist()


def main():
    config = selector_config("v2")
    cfg = dataclasses.replace(config, gpi_tonic_pA=GPI_TONIC_PA, thalamus_tonic_pA=THAL_TONIC_PA)
    channels = tuple(range(N_CHANNELS))
    sal = congruence_saliences(CAND_VALENCES, VH)
    expected = host_argmin(CAND_VALENCES, VH)
    print(f"=== motor-pool gain sweep (sub-threshold tonic), dev seed {DEV_SEED} ===", flush=True)
    print(f"  tonic: gpi={GPI_TONIC_PA}, thal={THAL_TONIC_PA}; v_held={VH:+.1f}; expected winner={expected}", flush=True)
    print(f"  saliences={[round(s, 2) for s in sal]}", flush=True)

    for gain in GAIN_SWEEP:
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)
        motor, commit = _full_run_motor(b, cfg, sal, channels, gain=gain)
        m = np.asarray(motor, dtype=float)
        order = np.argsort(m)[::-1]
        top2 = m[order[0]] / max(1.0, m[order[1]])
        argmax = int(order[0])
        tracks = "TRACKS" if argmax == expected else "no"
        print(f"  gain={gain:<8} motor={list(motor)}  commit={list(commit)}  "
              f"argmax={argmax}({tracks})  max/2nd={top2:.2f}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
