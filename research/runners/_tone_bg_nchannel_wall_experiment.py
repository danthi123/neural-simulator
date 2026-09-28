"""NO-DEFER EXPERIMENT for the N=5 tonic-firing wall (finding 2026-09-27-tone-bg-nchannel-tonic-firing-wall).

The N-channel BG race commits cleanly at N=2 but NOT at N=5: every commit pool fires tonically
(~9000 spikes each) so the race never produces a clean winner. This experiment tests three wiring
fix hypotheses against the LIVE N=5 bridge (dev seed 7, numpy/CPU):

  H1: higher salience gain (the thalamic-burst analogue) -- the salience bias on the D1 pool is the
      declared cortical afferent; a stronger burst should break the tie.
  H2: stronger cross-inhibition (commit_fsi_cross_weight) -- the N-way mutual inhibition must suppress
      N-1 losers simultaneously.
  H3: lower tonic floor (gpi_tonic_pA / thalamus_tonic_pA) -- the tonic floor is what drives the pools
      to fire tonically; lowering it should let the salience bias differentiate.

For each hypothesis we run the race across the v_held grid (the same 7 levels the probe uses) and report
how many commit cleanly (winner is not None). A fix that makes >= 5/7 commit cleanly (and the winner
tracks the highest-salience candidate) is a candidate for the no-defer wiring fix.

DEV seed 7 only (runbook hard rule: never 42/43/44/100/101/102). numpy/CPU (cost-routed). NO sim//webapp/
edit: the production selector is imported and wrapped, never changed.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_wall_experiment
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import dataclasses

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
from research.runners._tone_bg_nchannel_selector import SelectorConfig

DEV_SEED = 7
V_HELD_GRID = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]


def _race_matrix(bridge, config, saliences_grid, *, gain=SALIENCE_GAIN_PA):
    """Run the race across the v_held grid; return per-level (winner, committed, motor_spikes, loser_ratio)."""
    out = []
    for sal in saliences_grid:
        r = _run_nchannel_trial(bridge, config, sal, tuple(range(N_CHANNELS)), gain=gain)
        out.append(r)
    return out


def _summary(results):
    n_committed = sum(1 for r in results if r["committed"])
    winners = [r["winner"] for r in results]
    return n_committed, winners


def main():
    config = selector_config("v2")
    sal_grid = [congruence_saliences(CAND_VALENCES, vh) for vh in V_HELD_GRID]

    print(f"=== N=5 wall experiment, dev seed {DEV_SEED} ===", flush=True)
    print(f"  candidate valences: {CAND_VALENCES}", flush=True)
    print(f"  v_held grid: {V_HELD_GRID}", flush=True)
    print(f"  default config: gain={SALIENCE_GAIN_PA}, cross_weight={config.commit_fsi_cross_weight}, "
          f"gpi={config.gpi_tonic_pA}, thal={config.thalamus_tonic_pA}, "
          f"threshold={config.commit_threshold_spikes}, clean_ratio={config.clean_loser_ratio}", flush=True)

    # Baseline (the wall): default config, default gain.
    b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, config)
    _set_nchannel_tonic(b, config, tuple(range(N_CHANNELS)))
    _step(b, config.warmup_steps)
    base = _race_matrix(b, config, sal_grid)
    n_base, winners_base = _summary(base)
    print(f"\n[BASELINE] committed {n_base}/{len(V_HELD_GRID)}  winners={winners_base}", flush=True)
    for vh, r in zip(V_HELD_GRID, base):
        print(f"  v_held={vh:+.1f}  winner={r['winner']}  committed={r['committed']}  "
              f"simultaneous={r['simultaneous']}  motor_spikes={r['motor_spikes']}  "
              f"loser_ratio={r['loser_ratio']}", flush=True)

    # H1: higher salience gain (the thalamic-burst analogue).
    for gain in (1200.0, 2400.0, 4800.0):
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, config)
        _set_nchannel_tonic(b, config, tuple(range(N_CHANNELS)))
        _step(b, config.warmup_steps)
        res = _race_matrix(b, config, sal_grid, gain=gain)
        n, winners = _summary(res)
        print(f"\n[H1 gain={gain}] committed {n}/{len(V_HELD_GRID)}  winners={winners}", flush=True)

    # H2: stronger cross-inhibition.
    for cw in (80.0, 160.0, 320.0):
        cfg2 = dataclasses.replace(config, commit_fsi_cross_weight=cw)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg2)
        _set_nchannel_tonic(b, cfg2, tuple(range(N_CHANNELS)))
        _step(b, cfg2.warmup_steps)
        res = _race_matrix(b, cfg2, sal_grid)
        n, winners = _summary(res)
        print(f"\n[H2 cross_weight={cw}] committed {n}/{len(V_HELD_GRID)}  winners={winners}", flush=True)

    # H3: lower tonic floor.
    for gpi, thal in ((500.0, 450.0), (250.0, 225.0), (100.0, 90.0)):
        cfg3 = dataclasses.replace(config, gpi_tonic_pA=gpi, thalamus_tonic_pA=thal)
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg3)
        _set_nchannel_tonic(b, cfg3, tuple(range(N_CHANNELS)))
        _step(b, cfg3.warmup_steps)
        res = _race_matrix(b, cfg3, sal_grid)
        n, winners = _summary(res)
        print(f"\n[H3 gpi={gpi} thal={thal}] committed {n}/{len(V_HELD_GRID)}  winners={winners}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
