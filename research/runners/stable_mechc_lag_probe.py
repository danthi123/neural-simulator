"""Does SIM_PRESET_CENTRED_HET reach the LearnedAssocGraph cells that sit below -110 mV under SIM_STABLE_INTEGRATION?
(research/stable-mechc; build-only, no stepping, no brain turn.)

LearnedAssocGraph builds its bridge with research.runners._D_sparse_heteroassoc.build (regions `pool` and `fs`, neither with
a per-region neuron type; cfg.num_traits stays at the CoreSimConfig default). This builds that bridge at the production
size (n_pool 1500, n_fs 300) on the given seed with the switch unset and set to "1" and records:
  * whether the a/b/d/C/k/vr/vt/vpeak/c_reset arrays differ at all between the two builds (the claim: they do not);
  * the heterogeneity read-back of the ON build (neurons with a region preset);
  * the preset each cell was built from (identified by its never-jittered k, vr, vt, vpeak, c_reset), and how many
    cells in each region carry b < 0 after the draw although their preset has b >= 0 (the cells a trait-centred draw
    would change);
  * if --census is given, the indices of the census's out-of-range LearnedAssocGraph cells and their preset / b.
Observe-only.

  .venv/bin/python -m research.runners.stable_mechc_lag_probe --seed 7 \
      --census research/findings/raw/_stable_integration/census/on/s7/inner__intact_a.json \
      --out research/findings/raw/_stable_mechc/lag_probe_s7.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners.membrane_census import _preset_by_fixed_params  # noqa: E402
from sim.backend import to_host  # noqa: E402
from sim.enums import DefaultIzhikevichParamsManager, NeuronType  # noqa: E402

ARRAYS = ("a", "b", "d_increment", "C", "k", "vr", "vt", "vpeak", "c_reset")


def _build(seed, on, n_pool, n_fs):
    if on:
        os.environ["SIM_PRESET_CENTRED_HET"] = "1"
    else:
        os.environ.pop("SIM_PRESET_CENTRED_HET", None)
    from research.runners._D_sparse_heteroassoc import build
    return build(int(seed), n_pool=n_pool, n_fs=n_fs)


def _arrs(b):
    return {k: np.asarray(to_host(getattr(b, "cp_izh_" + k))).astype(np.float64) for k in ARRAYS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=1500)
    ap.add_argument("--n-fs", type=int, default=300)
    ap.add_argument("--census", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off = _build(a.seed, False, a.n_pool, a.n_fs)
    A_off = _arrs(off)
    rm = off.region_manager
    regions = {r.name: np.asarray(sorted(int(i) for i in rm.indices(r.name)), dtype=np.int64) for r in off.core_config.brain_regions}
    del off
    on = _build(a.seed, True, a.n_pool, a.n_fs)
    A_on = _arrs(on)
    differs = {k: int(np.count_nonzero(A_on[k] != A_off[k])) for k in ARRAYS}
    table = _preset_by_fixed_params()
    sig = np.round(np.stack([A_off[k] for k in ("k", "vr", "vt", "vpeak", "c_reset")], axis=1), 3)
    names = ["/".join(table.get(tuple(float(x) for x in sig[i]), ["?"])) for i in range(sig.shape[0])]
    preset_b = np.array([float(DefaultIzhikevichParamsManager.PARAMS[NeuronType["IZH2007_" + n.split("/")[0]]]["b"])
                         if n != "?" else np.nan for n in names])
    per_region = {}
    for r, ix in regions.items():
        per_region[r] = {"n": int(ix.size), "preset_counts": dict(collections.Counter(names[int(i)] for i in ix)),
                         "b_mean_off": float(A_off["b"][ix].mean()), "b_mean_on": float(A_on["b"][ix].mean()),
                         "n_b_negative_off": int(np.count_nonzero(A_off["b"][ix] < 0)),
                         "n_b_negative_on": int(np.count_nonzero(A_on["b"][ix] < 0)),
                         "n_b_negative_with_preset_b_nonneg": int(np.count_nonzero((A_off["b"][ix] < 0) & (preset_b[ix] >= 0)))}
    res = {"runner": "research/runners/stable_mechc_lag_probe.py", "seed": a.seed, "n_pool": a.n_pool, "n_fs": a.n_fs,
           "backend": "numpy", "arrays_differing_between_off_and_on": differs,
           "switch_applied_on_build": bool(on.preset_centred_het_applied),
           "heterogeneity_readback_on_build": on.preset_centred_het_readback, "regions": per_region}
    if a.census:
        rec = json.load(open(a.census))
        rows = []
        for b in rec["census"]["bridges"]:
            if b["organ"].endswith("LearnedAssocGraph"):
                rows = b["oor_neurons"]
                break
        cells = []
        for r in rows:
            i = int(r["idx"])
            cells.append({"idx": i, "region": r["region"], "preset": names[i], "preset_b": float(preset_b[i]),
                          "b_drawn": float(A_off["b"][i]), "b_census": r["b"], "C_drawn": float(A_off["C"][i]),
                          "same_value_on": bool(A_on["b"][i] == A_off["b"][i] and A_on["C"][i] == A_off["C"][i])})
        res["census_offender_cells"] = {"census": a.census, "n": len(cells),
                                        "n_b_matches_this_build": sum(1 for c in cells if abs(c["b_drawn"] - c["b_census"]) < 1e-3),
                                        "n_unchanged_by_switch": sum(1 for c in cells if c["same_value_on"]),
                                        "n_preset_b_nonneg": sum(1 for c in cells if c["preset_b"] >= 0),
                                        "presets": dict(collections.Counter(c["preset"] for c in cells)),
                                        "regions": dict(collections.Counter(c["region"] for c in cells))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("arrays_differing_between_off_and_on", "heterogeneity_readback_on_build")}))
    for r, v in per_region.items():
        print(r, v)
    if "census_offender_cells" in res:
        print(res["census_offender_cells"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
