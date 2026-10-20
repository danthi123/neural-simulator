#!/usr/bin/env python3
"""Step 0b Part A (docs/drafts/bg-tonic-loop/PACKET.md, RULING bg-tonic-loop-1020): the tonic
epoch, organ only. Modelled line for line on research/runners/_bg_tonic_calibration.py, but ONE
FRESH BRIDGE PER RUN (the calibration probe reused one bridge across grid points; this does not).

Run as a module (provenance sidecars are automatic):
  python -m research.runners.bg_tonic_loop_probe --seed 9 --stn 50 --arm A0 --out <json>

Arms: A0 (hybrid, today), A0p (BRAIN_BG_PRESET_CELLS=1), A5 (A0 + BRAIN_BG_BOUNDED_COND=1),
A5p (A0p + BRAIN_BG_BOUNDED_COND=1). GPe is held at 100 pA (the registered pick), STN 40 or 50 pA.
selector_config("v2"), _set_equal_tonic_current, BRAIN_BG_TONIC_DRIVE=1, no arousal, no stimulus,
200 equilibration steps discarded, then 1000 measured steps (dt 1 ms).

Per region — EVERY region the bridge actually builds (v2 builds 21: practice_arousal, selector_stn,
selector_reset and, for ch in CHANNELS, proposal_{ch}, str_d1_{ch}, str_d2_{ch}, gpe_{ch}, gpi_{ch},
thal_{ch}, commit_{ch}, commit_fs_{ch}, motor_{ch}; the packet also names str_fsi_{ch}, which is
registered only when enable_striatal_fsi and selector_config("v2") sets it False, so it does not
exist here — the list is derived from region_manager so it records exactly what is built): every
neuron's rate (Hz), ISI-CV and burst fraction (share of ISIs at or below 10 ms), the count of silent
neurons, the count of steps with v below E_inh - 5 mV and v_min.

Assert (hard, exit 2) for A0p: vs a fresh A0 reference bridge — both driven by
_set_equal_tonic_current first (the build-time init of cp_external_input_current is a
parameter-derived random and is overwritten identically by the tonic currents) — the arm differs
only in the four parameter arrays (cp_izh_a, cp_izh_b, cp_izh_d_increment, cp_izh_C). For A5/A5p
the same diff is RECORDED against A0 but not asserted (not specified in the packet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["BRAIN_BG_TONIC_DRIVE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from sim.backend import to_host  # noqa: E402
from research.runners._vocal_action_selector_gate import (  # noqa: E402
    _indices,
    _set_equal_tonic_current,
    _step,
    build_selector_bridge,
    selector_config,
)

WARMUP, MEASURE = 200, 1000
E_INH_MV = -75.0
SUB_THRESHOLD_MV = E_INH_MV - 5.0            # E_inh - 5 mV = -80 mV
BURST_ISI_MS = 10.0
GPE_PA = 100.0                                # registered pick (dev refile)
PARAM_ARRAYS = ("cp_izh_a", "cp_izh_b", "cp_izh_d_increment", "cp_izh_C")
PRESET_FLAG, BOUNDED_FLAG = "BRAIN_BG_PRESET_CELLS", "BRAIN_BG_BOUNDED_COND"
SWITCHES = {"A0": (), "A0p": (PRESET_FLAG,), "A5": (BOUNDED_FLAG,),
            "A5p": (PRESET_FLAG, BOUNDED_FLAG)}


def region_names(bridge):
    """Every region this bridge actually builds, in registration order (v2 has no str_fsi: see module
    docstring)."""
    return [r.name for r in bridge.region_manager.regions()]


def _bridge_arrays(bridge):
    out = {}
    for name in vars(bridge):
        v = vars(bridge)[name]
        if hasattr(v, "tobytes") and hasattr(v, "dtype"):
            out[name] = v
    return out


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def build(seed, cfg, flags_on):
    """One fresh bridge; only `flags_on` set (both switches popped first: ambient env must not leak)."""
    os.environ.pop(PRESET_FLAG, None)
    os.environ.pop(BOUNDED_FLAG, None)
    for f in flags_on:
        os.environ[f] = "1"
    bridge = build_selector_bridge(int(seed), cfg)
    return bridge


def param_diff(arm_arrays, ref_arrays):
    if set(arm_arrays) != set(ref_arrays):
        return None, sorted(set(arm_arrays) ^ set(ref_arrays))
    differing = sorted(k for k in ref_arrays if not _same(arm_arrays[k], ref_arrays[k]))
    return differing, []


def neuron_stats(spike_matrix, window_ms):
    """spike_matrix: (n_neurons, steps) bool -> per-neuron rate Hz, ISI-CV, burst fraction."""
    counts = spike_matrix.sum(axis=1)
    rate = (counts / float(window_ms) * 1000.0)
    isi_cv, burst = [], []
    for row in spike_matrix:
        t = np.flatnonzero(row)
        if t.size < 2:
            isi_cv.append(None)
            burst.append(None)
            continue
        isi = np.diff(t).astype(np.float64)
        m = float(isi.mean())
        isi_cv.append(round(float(isi.std() / m), 4) if m > 0 else None)
        burst.append(round(float((isi <= BURST_ISI_MS).mean()), 4))
    return rate, isi_cv, burst, int((counts == 0).sum())


def run_arm(seed, stn_pa, arm):
    cfg = selector_config("v2")
    os.environ["BRAIN_BG_TONIC_GPE_PA"] = str(GPE_PA)
    os.environ["BRAIN_BG_TONIC_STN_PA"] = str(float(stn_pa))
    flags = SWITCHES[arm]

    ref_arrays = None
    if arm != "A0":
        ref = build(seed, cfg, ())          # fresh A0 reference (both switches off)
        _set_equal_tonic_current(ref, cfg)
        ref_arrays = _bridge_arrays(ref)
        del ref
    bridge = build(seed, cfg, flags)
    _set_equal_tonic_current(bridge, cfg)
    # Both bridges are driven identically before the snapshot: at build time
    # cp_external_input_current is a parameter-derived random init (sim/bridge.py:2021), so an
    # undriven A0p/A0 comparison would flag it; _set_equal_tonic_current overwrites it identically
    # before any step, which is the state that actually enters measurement.
    arms_arr = _bridge_arrays(bridge)
    assert_block = {"arm": arm, "asserted": arm == "A0p"}
    if arm != "A0" and ref_arrays is not None:
        differing, sym = param_diff(arms_arr, ref_arrays)
        assert_block["differing_arrays"] = differing
        assert_block["symmetric_diff"] = sym
        assert_block["four_params_differing"] = [k for k in PARAM_ARRAYS
                                                 if k in (differing or [])]
        if arm == "A0p":
            if differing is None or sym:
                print("A0p vs A0: array key sets differ:", sym, file=sys.stderr)
                sys.exit(2)
            stray = [k for k in differing if k not in PARAM_ARRAYS]
            if stray:
                print("A0p vs A0 differs OUTSIDE the four parameter arrays:", stray,
                      file=sys.stderr)
                sys.exit(2)

    names = region_names(bridge)
    idx = {}
    for name in names:
        idx[name] = _indices(bridge, name)
    n_total = int(np.asarray(to_host(bridge.cp_firing_states)).size)

    for _ in range(WARMUP):                  # discarded
        _step(bridge)

    spikes = {name: np.zeros((idx[name].size, MEASURE), dtype=bool) for name in names}
    steps_below = {name: 0 for name in names}
    v_min = {name: float("inf") for name in names}
    for step in range(MEASURE):
        _step(bridge)
        f = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        v = np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=np.float64)
        for name in names:
            ii = idx[name]
            spikes[name][:, step] = f[ii]
            lo = float(v[ii].min())
            if lo < SUB_THRESHOLD_MV:
                steps_below[name] += 1
            if lo < v_min[name]:
                v_min[name] = lo

    regions = {}
    for name in names:
        rate, isi_cv, burst, silent = neuron_stats(spikes[name], MEASURE)
        r = np.asarray(rate)
        regions[name] = {
            "n": int(idx[name].size),
            "rate_hz": [round(float(x), 4) for x in rate],
            "isi_cv": isi_cv,
            "burst_frac": burst,
            "silent": silent,
            "rate_mean": round(float(r.mean()), 4),
            "rate_median": round(float(np.median(r)), 4),
            "rate_min": round(float(r.min()), 4),
            "rate_max": round(float(r.max()), 4),
            "v_steps_below": steps_below[name],
            "v_min": round(v_min[name], 4) if np.isfinite(v_min[name]) else None,
        }

    return {
        "packet": "docs/drafts/bg-tonic-loop/PACKET.md",
        "part": "A",
        "arm": arm,
        "seed": int(seed),
        "gpe_pa": GPE_PA,
        "stn_pa": float(stn_pa),
        "warmup_steps": WARMUP,
        "measure_steps": MEASURE,
        "dt_ms": float(bridge.core_config.dt_ms),
        "switches": {"BRAIN_BG_PRESET_CELLS": PRESET_FLAG in flags,
                     "BRAIN_BG_BOUNDED_COND": BOUNDED_FLAG in flags},
        "n_neurons_total": n_total,
        "param_assert": assert_block,
        "sub_threshold_mv": SUB_THRESHOLD_MV,
        "regions": regions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stn", type=float, required=True, choices=(40.0, 50.0))
    p.add_argument("--arm", required=True, choices=tuple(SWITCHES))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = run_arm(a.seed, a.stn, a.arm)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=False)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
