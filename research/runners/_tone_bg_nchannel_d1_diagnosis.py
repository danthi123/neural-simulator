"""DIAGNOSIS for the N=5 wall: is the salience bias reaching the D1 pools and differentiating them?

The wall experiment (H1/H2/H3) showed the salience bias does NOT differentiate the commit pools. This
diagnosis measures the D1 pool's firing rate per channel (where the salience bias lands) over a FULL run
(no early break) to see whether the bias is reaching the D1 pools at all, and whether it differentiates them.

Also measures the commit pool's firing rate per channel (the floor) and the thalamic pool's firing rate, to
localise where the differentiation is lost.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_d1_diagnosis
"""
from __future__ import annotations

import os

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
    CAND_VALENCES,
    N_CHANNELS,
    SALIENCE_GAIN_PA,
    selector_config,
)
from research.runners._vocal_action_selector_gate import _indices

DEV_SEED = 7


def _full_run(bridge, config, saliences, channels, *, gain=SALIENCE_GAIN_PA, n_steps=None):
    """Run the N-way race for n_steps (default: config.action_steps) WITHOUT the early commit break; return
    per-region per-channel firing counts (D1, thal, commit, motor)."""
    xp, _ = get_backend()
    if n_steps is None:
        n_steps = int(config.action_steps)
    _set_nchannel_tonic(bridge, config, channels)
    if True:  # arousal on (the shared practice-arousal drive)
        bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = xp.float32(config.practice_pA)
        for ch in channels:
            idx = xp.asarray(_indices(bridge, f"str_d1_{ch}"))
            bridge.cp_external_input_current[idx] = (
                bridge.cp_external_input_current[idx] + xp.float32(float(saliences[ch]) * gain)
            )
    d1 = np.zeros(len(channels), dtype=np.int64)
    thal = np.zeros(len(channels), dtype=np.int64)
    commit = np.zeros(len(channels), dtype=np.int64)
    motor = np.zeros(len(channels), dtype=np.int64)
    for step in range(n_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for i, ch in enumerate(channels):
            d1[i] += int(firing[_indices(bridge, f"str_d1_{ch}")].sum())
            thal[i] += int(firing[_indices(bridge, f"thal_{ch}")].sum())
            commit[i] += int(firing[_indices(bridge, f"commit_{ch}")].sum())
            motor[i] += int(firing[_indices(bridge, f"motor_{ch}")].sum())
    # washout (return the cached bridge to the resting point)
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_nchannel_tonic(bridge, config, channels)
    _step(bridge, config.washout_steps)
    return {"d1": d1.tolist(), "thal": thal.tolist(), "commit": commit.tolist(), "motor": motor.tolist()}


def main():
    config = selector_config("v2")
    channels = tuple(range(N_CHANNELS))
    print(f"=== D1 diagnosis, dev seed {DEV_SEED}, {config.action_steps} steps ===", flush=True)
    print(f"  candidate valences: {CAND_VALENCES}", flush=True)

    for vh in (+0.9, -0.9, 0.0):
        sal = congruence_saliences(CAND_VALENCES, vh)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, config)
        _set_nchannel_tonic(b, config, channels)
        _step(b, config.warmup_steps)
        r = _full_run(b, config, sal, channels)
        print(f"\n  v_held={vh:+.1f}  saliences={[round(s, 2) for s in sal]}", flush=True)
        print(f"    D1 firing:    {r['d1']}", flush=True)
        print(f"    thal firing:  {r['thal']}", flush=True)
        print(f"    commit firing:{r['commit']}", flush=True)
        print(f"    motor firing: {r['motor']}", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
