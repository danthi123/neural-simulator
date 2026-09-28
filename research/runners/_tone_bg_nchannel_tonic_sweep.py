"""TONIC-FLOOR SWEEP for the N=5 wall: find the tonic level where the commit pools fire sub-threshold
AND differentiated (so the salience bias can break the tie).

The commit saturation diagnosis showed the tonic floor (gpi_tonic + thalamus_tonic) is the driver:
tonic OFF -> commit pools fire 0; baseline (gpi=1000, thal=900) -> all saturated at 9000. This sweep
varies the tonic floor (gpi and thal scaled together) and reports the commit pool's per-channel firing
+ whether the race commits cleanly + the winner.

A tonic level where (a) the commit pools fire sub-threshold (not all at 9000) AND (b) the race commits
cleanly with the winner tracking the highest-salience candidate is the no-defer wiring fix.

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_tonic_sweep
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


def _commit_firing(bridge, config, saliences, channels, *, n_steps=None):
    xp, _ = get_backend()
    if n_steps is None:
        n_steps = int(config.action_steps)
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = xp.float32(config.practice_pA)
    for ch in channels:
        idx = xp.asarray(_indices(bridge, f"str_d1_{ch}"))
        bridge.cp_external_input_current[idx] = (
            bridge.cp_external_input_current[idx] + xp.float32(float(saliences[ch]) * SALIENCE_GAIN_PA)
        )
    commit = np.zeros(len(channels), dtype=np.int64)
    for step in range(n_steps):
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        for i, ch in enumerate(channels):
            commit[i] += int(firing[_indices(bridge, f"commit_{ch}")].sum())
    _set_nchannel_tonic(bridge, config, channels)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "selector_reset"))] = xp.float32(config.reset_pA)
    _step(bridge, config.reset_steps)
    _set_nchannel_tonic(bridge, config, channels)
    _step(bridge, config.washout_steps)
    return commit.tolist()


def main():
    config = selector_config("v2")
    channels = tuple(range(N_CHANNELS))
    print(f"=== tonic-floor sweep, dev seed {DEV_SEED} ===", flush=True)
    print(f"  candidate valences: {CAND_VALENCES}", flush=True)

    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01):
        gpi = round(config.gpi_tonic_pA * scale, 1)
        thal = round(config.thalamus_tonic_pA * scale, 1)
        cfg = dataclasses.replace(config, gpi_tonic_pA=gpi, thalamus_tonic_pA=thal)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)

        # commit firing at v_held=+0.9 (the salience that should favor cand 1)
        sal = congruence_saliences(CAND_VALENCES, +0.9)
        commit = _commit_firing(b, cfg, sal, channels)

        # race across the grid
        results = []
        for vh in V_HELD_GRID:
            s = congruence_saliences(CAND_VALENCES, vh)
            r = _run_nchannel_trial(b, cfg, s, channels)
            results.append(r)
        n_committed = sum(1 for r in results if r["committed"])
        winners = [r["winner"] for r in results]
        print(f"  scale={scale:<6} gpi={gpi:<7} thal={thal:<7} "
              f"commit(vh=+0.9)={commit}  committed={n_committed}/7  winners={winners}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
