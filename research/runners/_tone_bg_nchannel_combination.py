"""COMBINATION PROBE for the N=5 wall: sub-threshold tonic window (H3) x higher salience gain (H1).

The tonic-floor sweep (finding 2026-09-27-tone-bg-nchannel-tonic-firing-wall, the 2026-09-28 section) showed the
tonic floor is the saturation driver: at scale~0.05 (gpi=50, thal=45) the commit pools fire sub-threshold (~8000)
and differentiate, but only at the noise level (a 0.57% spread) and the race still commits 0/7. H3 (lower tonic)
alone fails. The untested lever is the COMBINATION: run the race in that sub-threshold window WITH a higher
salience gain, so the bias has contrast to break the N-way tie.

For each gain, run the race across the v_held grid (the same 7 levels the probe uses) and report how many commit
cleanly + whether the winner tracks host_argmin (the salience-favored candidate, the declared shortcut the race is
meant to match). A gain where the race commits cleanly AND the winner tracks host_argmin across the grid is the
no-defer wiring fix.

DEV seed 7 only. numpy/CPU (this probe is cheap; the 6-seed validation is the GPU step). NO sim//webapp/ edit: the
production selector is imported and wrapped, never changed.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_combination
"""
from __future__ import annotations

import os
import dataclasses

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

from research.runners._tone_bg_nchannel_selector import (
    build_nchannel_bridge,
    _set_nchannel_tonic,
    _run_nchannel_trial,
    _step,
    congruence_saliences,
    host_argmin,
    CAND_VALENCES,
    N_CHANNELS,
    SALIENCE_GAIN_PA,
    selector_config,
)

DEV_SEED = 7
V_HELD_GRID = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
# the sub-threshold tonic window from the sweep (scale 0.05): the only level where the pools fire sub-threshold
# AND differentiate (scale>=0.1 pinned at 9000, scale<=0.025 silent).
GPI_TONIC_PA = 50.0
THAL_TONIC_PA = 45.0
GAIN_SWEEP = (SALIENCE_GAIN_PA, 1200.0, 2400.0, 4800.0, 9600.0, 19200.0)


def main():
    config = selector_config("v2")
    cfg = dataclasses.replace(config, gpi_tonic_pA=GPI_TONIC_PA, thalamus_tonic_pA=THAL_TONIC_PA)
    channels = tuple(range(N_CHANNELS))
    expected = [host_argmin(CAND_VALENCES, vh) for vh in V_HELD_GRID]

    print(f"=== combination probe (sub-threshold tonic x salience gain), dev seed {DEV_SEED} ===", flush=True)
    print(f"  tonic: gpi={GPI_TONIC_PA}, thal={THAL_TONIC_PA} (the sweep's sub-threshold window)", flush=True)
    print(f"  expected winners (host_argmin): {expected}", flush=True)
    print(f"  threshold={config.commit_threshold_spikes}, clean_loser_ratio={config.clean_loser_ratio}, "
          f"action_steps={config.action_steps}", flush=True)

    # fresh bridge per gain (the wall-experiment pattern: avoids any cross-gain state contamination).
    for gain in GAIN_SWEEP:
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)
        results = [
            _run_nchannel_trial(b, cfg, congruence_saliences(CAND_VALENCES, vh), channels, gain=gain)
            for vh in V_HELD_GRID
        ]
        n_committed = sum(1 for r in results if r["committed"])
        winners = [r["winner"] for r in results]
        n_correct = sum(1 for w, e in zip(winners, expected) if w == e)
        motor_at_09 = results[-1]["motor_spikes"]  # v_held=+0.9 (should favor cand 1)
        print(f"  gain={gain:<8} committed={n_committed}/7  correct={n_correct}/7  winners={winners}  "
              f"motor(vh=+0.9)={motor_at_09}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
