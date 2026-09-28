"""DIAGNOSIS 2: what drives the commit pools to saturation (all channels at 9000 spikes)?

The D1 diagnosis showed the D1/thalamic pools differentiate but the commit pools saturate at exactly 9000
(30 neurons x 300 spikes = 50% rate) for EVERY channel, flattening the upstream differentiation. This
experiment toggles each candidate driver and measures the commit pool's per-channel firing:

  A: baseline (internal recurrence on, tonic floor on)
  B: internal recurrence OFF (commit_internal_density=0, commit_internal_weight=0)
  C: tonic floor OFF (gpi_tonic=0, thalamus_tonic=0)
  D: internal recurrence OFF + tonic floor OFF
  E: NMDA OFF (commit_enable_nmda=False)

For each, report the commit pool's per-channel firing (does it still saturate? does it differentiate?).

DEV seed 7 only. numpy/CPU. NO sim//webapp/ edit.

Usage:
  SIM_BACKEND=numpy python -m research.runners._tone_bg_nchannel_commit_saturation
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
    CAND_VALENCES,
    N_CHANNELS,
    SALIENCE_GAIN_PA,
    selector_config,
)
from research.runners._vocal_action_selector_gate import _indices

DEV_SEED = 7
VH = +0.9  # the v_held that should favor cand 1 (v=0.7)


def _full_run_commit(bridge, config, saliences, channels, *, gain=SALIENCE_GAIN_PA, n_steps=None):
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
    sal = congruence_saliences(CAND_VALENCES, VH)
    print(f"=== commit saturation diagnosis, dev seed {DEV_SEED}, v_held={VH:+.1f} ===", flush=True)
    print(f"  saliences={[round(s, 2) for s in sal]}", flush=True)

    variants = [
        ("A baseline", config, dict()),
        ("B internal OFF", config, dict(commit_internal_density=0.0, commit_internal_weight=0.0)),
        ("C tonic OFF", dataclasses.replace(config, gpi_tonic_pA=0.0, thalamus_tonic_pA=0.0), dict()),
        ("D internal+tonic OFF", dataclasses.replace(config, gpi_tonic_pA=0.0, thalamus_tonic_pA=0.0),
         dict(commit_internal_density=0.0, commit_internal_weight=0.0)),
    ]
    for name, cfg, build_kw in variants:
        b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, cfg, **build_kw)
        _set_nchannel_tonic(b, cfg, channels)
        _step(b, cfg.warmup_steps)
        commit = _full_run_commit(b, cfg, sal, channels)
        print(f"  {name:22s} commit={commit}", flush=True)

    # E: NMDA off
    b = build_nchannel_bridge(N_CHANNELS, DEV_SEED, config, commit_enable_nmda=False)
    _set_nchannel_tonic(b, config, channels)
    _step(b, config.warmup_steps)
    commit = _full_run_commit(b, config, sal, channels)
    print(f"  {'E NMDA OFF':22s} commit={commit}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
