"""MOTOR-POOL TONIC SWEEP for the N=5 wall: the DECISION pool (motor) response to the tonic floor.

The tonic-floor sweep (`_tone_bg_nchannel_tonic_sweep`) measured the COMMIT pool and found a sub-threshold window
at scale 0.05 (gpi=50, thal=45) where the commit pools differentiate (8055-8101). But the RACE decision reads the
MOTOR pool (`_run_nchannel_trial` counts `motor_{ch}`), and the combination probe showed the motor pools are
near-silent (7-16 spikes) at that window -- the commit->motor pathway (`commit_{ch} -> motor_{ch}`, weight
`commit_to_motor_weight`) does not propagate the commit pool's ~8000 firing to the motor pool. The sweep measured
the wrong pool.

This sweep measures the MOTOR pool (the decision pool) across the tonic scales, at the default salience gain,
alongside the commit pool, to answer the core question: is there a tonic where the motor pools fire sub-threshold
AND differentiated (so the salience bias can break the N-way tie)? If yes, the tonic is the lever. If the motor
pool is always either silent or saturated (no active+differentiated window), the suspect moves to the
commit->motor synapse / the motor pool's own operating point.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_motor_tonic_sweep
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
    _run_nchannel_trial,
    _step,
    congruence_saliences,
    CAND_VALENCES,
    N_CHANNELS,
    SALIENCE_GAIN_PA,
    selector_config,
)
from research.runners._vocal_action_selector_gate import _indices

DEV_SEED = 7
V_HELD_GRID = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]


def _motor_firing(bridge, config, saliences, channels, *, gain=SALIENCE_GAIN_PA, n_steps=None):
    """Count the MOTOR pool (the decision pool) AND the commit pool over the full run (no early break), so the
    commit->motor attenuation is visible directly at each tonic."""
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
    channels = tuple(range(N_CHANNELS))
    print(f"=== motor-pool tonic sweep (the DECISION pool), dev seed {DEV_SEED} ===", flush=True)
    print(f"  candidate valences: {CAND_VALENCES}", flush=True)
    print(f"  threshold={config.commit_threshold_spikes}, clean_loser_ratio={config.clean_loser_ratio}, "
          f"action_steps={config.action_steps}, gain={SALIENCE_GAIN_PA}", flush=True)

    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01):
        gpi = round(config.gpi_tonic_pA * scale, 1)
        thal = round(config.thalamus_tonic_pA * scale, 1)
        cfg = dataclasses.replace(config, gpi_tonic_pA=gpi, thalamus_tonic_pA=thal)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)

        # motor + commit firing at v_held=+0.9 (the salience that should favor cand 1), default gain
        sal = congruence_saliences(CAND_VALENCES, +0.9)
        motor, commit = _motor_firing(b, cfg, sal, channels)

        # race across the grid (default gain)
        results = [
            _run_nchannel_trial(b, cfg, congruence_saliences(CAND_VALENCES, vh), channels)
            for vh in V_HELD_GRID
        ]
        n_committed = sum(1 for r in results if r["committed"])
        winners = [r["winner"] for r in results]
        print(f"  scale={scale:<6} gpi={gpi:<7} thal={thal:<7} "
              f"motor(vh=+0.9)={motor}  commit(vh=+0.9)={commit}  "
              f"committed={n_committed}/7  winners={winners}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
