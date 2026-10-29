"""Would centring the trait-typed LearnedAssocGraph cells on their own trait preset remove the cells that sit below -110 mV
under SIM_STABLE_INTEGRATION=1? (research/stable-mechc; a diagnostic on dev seeds only, engine untouched.)

SIM_PRESET_CENTRED_HET changes none of those cells (research/findings/raw/_stable_mechc/lag_probe_s7.json): LearnedAssocGraph's
bridge has no region preset, its cells are typed by trait (cfg.num_traits default > 1) and the a/b/d/C draw that overwrites
each trait preset is centred on the global RS values. This probe builds the same LearnedAssocGraph on the stable
integrator, rewrites a, b, d, C (and the initial u = b (v - vr)) of every cell BEFORE the first step, and stores the same five
facts (30 cycles each, as the brain does), tracking each cell's lowest membrane potential:

  as_built      the engine as it is (trait preset overwritten by the RS-centred draw)
  trait_centred each cell's draw recentred on its OWN trait preset with the engine's own rule
                (SimulationBridge._recentre_het_draw_host: sample / centre, kept in the [0.1, 3] clip, times the preset)
  trait_exact   each cell exactly its trait preset (no heterogeneity)
  rs_exact      every cell exactly the RS preset (a 0.03, b -2, d 100, C 100; no heterogeneity)

The drive is identical in all four arms of a seed (same patterns, same facts); only a, b, d, C and the initial u differ.

  .venv/bin/python -m research.runners.stable_mechc_lag_trait_probe --seeds 7 1 2 3 \
      --out research/findings/raw/_stable_mechc/lag_trait_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["SIM_STABLE_INTEGRATION"] = "1"
os.environ.pop("SIM_PRESET_CENTRED_HET", None)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners.learned_assoc_graph import LearnedAssocGraph  # noqa: E402
from sim import SimulationBridge  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from sim.enums import DefaultIzhikevichParamsManager, NeuronType  # noqa: E402

DEV_SEEDS = tuple(range(1, 16))
CONCEPTS = ["lynx", "chase", "vole", "beetle", "mite", "dog", "cat", "eat", "fish", "fox", "hen", "bird", "worm", "owl"]
FACTS = [["lynx", "chase", "vole"], ["vole", "chase", "beetle"], ["beetle", "chase", "mite"],
         ["dog", "chase", "cat"], ["cat", "eat", "fish"]]
HET_KEYS = {"a": ("izh_a_val", "a"), "b": ("izh_b_val", "b"), "d_increment": ("izh_d_val", "d_increment"),
            "C": ("izh_C_val", "C")}
V_LOW = -110.0
RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL


def _preset_arrays(bridge):
    """Per cell, the a/b/d/C of the preset it was built from (the viz label written at trait assignment)."""
    labels = bridge.runtime_state.neuron_types_list_for_viz
    out = {k: np.zeros(len(labels)) for k in HET_KEYS}
    for i, lab in enumerate(labels):
        prm = DefaultIzhikevichParamsManager.PARAMS[NeuronType["IZH2007_" + lab.replace("Izh2007_", "")]]
        for k, (_hk, f) in HET_KEYS.items():
            out[k][i] = float(prm[f])
    return out, labels


def _rewrite(bridge, variant):
    xp, _ = get_backend()
    presets, labels = _preset_arrays(bridge)
    rs = DefaultIzhikevichParamsManager.PARAMS[RS]
    n = len(labels)
    dists = bridge.core_config.heterogeneity_distributions
    for k, (hk, f) in HET_KEYS.items():
        arr = getattr(bridge, "cp_izh_" + k)
        if variant == "as_built":
            continue
        if variant == "trait_exact":
            new = presets[k]
        elif variant == "rs_exact":
            new = np.full(n, float(rs[f]))
        elif variant == "trait_centred":
            draw = np.asarray(to_host(arr)).astype(np.float64)
            new = SimulationBridge._recentre_het_draw_host(draw, dists[hk], np.ones(n, dtype=bool), presets[k])
        else:
            raise ValueError(variant)
        arr[:] = xp.asarray(np.asarray(new, dtype=np.float32))
    if variant != "as_built":
        bridge.cp_recovery_variable_u[:] = bridge.cp_izh_b * (bridge.cp_membrane_potential_v - bridge.cp_izh_vr)
    return labels


def run_one(seed, variant):
    t0 = time.time()
    lag = LearnedAssocGraph(CONCEPTS, seed=int(seed))
    b = lag.bridge
    labels = _rewrite(b, variant)
    n = len(labels)
    vmin = np.full(n, np.inf)
    first = np.full(n, -1, dtype=np.int64)
    step = [0]
    spikes = [0]
    orig = b._run_one_simulation_step

    def wrapped(*a, **k):
        r = orig(*a, **k)
        step[0] += 1
        v = np.asarray(to_host(b.cp_membrane_potential_v)).astype(np.float64)
        np.minimum(vmin, v, out=vmin)
        low = (v < V_LOW) & (first < 0)
        first[low] = step[0]
        spikes[0] += int(np.count_nonzero(np.asarray(to_host(b.cp_firing_states))))
        return r

    b._run_one_simulation_step = wrapped
    for f in FACTS:
        lag.store_fact(f)
    bv = np.asarray(to_host(b.cp_izh_b)).astype(np.float64)
    below = vmin < V_LOW
    by_type = {}
    for lab in sorted(set(labels)):
        m = np.array([x == lab for x in labels])
        by_type[lab] = {"n": int(m.sum()), "n_below_110": int(np.count_nonzero(below & m)),
                        "n_b_negative": int(np.count_nonzero(bv[m] < 0))}
    return {"seed": int(seed), "variant": variant, "n_cells": n, "n_steps": step[0], "n_spikes": spikes[0],
            "n_below_110": int(np.count_nonzero(below)), "v_min": float(vmin.min()),
            "n_b_negative": int(np.count_nonzero(bv < 0)), "stable_integration_applied": bool(b.stable_integration_applied),
            "by_trait_type": by_type, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["as_built", "trait_centred", "trait_exact", "rs_exact"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bad = [s for s in a.seeds if s not in DEV_SEEDS]
    if bad:
        raise SystemExit("dev seeds 1-15 only; refused: %s" % bad)
    rows = []
    for s in a.seeds:
        for v in a.variants:
            r = run_one(s, v)
            rows.append(r)
            print("seed %2d %-13s below -110: %4d / %d | v_min %9.2f | b<0 %4d | spikes %d | %.0fs" % (
                s, v, r["n_below_110"], r["n_cells"], r["v_min"], r["n_b_negative"], r["n_spikes"], r["wall_s"]), flush=True)
    res = {"runner": "research/runners/stable_mechc_lag_trait_probe.py", "backend": "numpy",
           "env": {"SIM_STABLE_INTEGRATION": "1", "SIM_PRESET_CENTRED_HET": None}, "concepts": CONCEPTS, "facts": FACTS,
           "v_low_mV": V_LOW, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
