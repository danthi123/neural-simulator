"""ONE-BRAIN MERGE — RUNG 1 verify: are the SURPRISE + WORLD-MODEL production organs' reads BYTE-IDENTICAL
merged (one shared spiking bridge) vs co-resident (each on its own bridge), with the two merge flags ON?

This is the load-bearing claim of the first production rung of the one-substrate merge: migrating the two most
compatible production organs onto ONE `SimulationBridge` must not change a single read. It exercises the SAME
production organ classes (`SurpriseProductionOrgan`, `WorldModelProductionOrgan`) and their SAME public read
APIs (`judge`/`read_surprise`, `expectation`/`read_surprise`) that `brain_chat` calls — the only difference is
whether they share one bridge (MERGED) or each build their own (CO-RESIDENT), both with
`per_region_threshold_heterogeneity` + `per_region_homeostasis_isolation` ON so the merge is exact.

Per seed it verifies:
  * ONE SHARED POOL (merged): both organs' `bridge` is the SAME object + the SAME `cp_membrane_potential_v`
    array; N_all >= sum(region sizes); every region index < N_all.
  * BYTE-IDENTITY: every production read of each organ (calibration numbers + a read battery) is identical
    merged-vs-co-resident (max abs delta printed; expect 0.0).
  * FACULTIES ALIVE on the merged bridge: surprise contradict/confirm separation >= 2x; world-model predicted
    valence signs opposite for +/- context, and violated surprise > expected surprise.
  * DETERMINISM: build the merged substrate twice at one seed; substrate hashes (membrane, thresholds,
    connections) identical.

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_merge_rung1_verify \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_merge_rung1_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners.onebrain_merge_production import MergedSubstrate
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from research.runners.worldmodel_production_organ import WorldModelProductionOrgan


def _host(a):
    try:
        return a.get()
    except AttributeError:
        return a


def _arr_hash(a):
    import numpy as np
    return hashlib.sha1(np.ascontiguousarray(_host(a)).tobytes()).hexdigest()


# A fixed word battery for the surprise organ — a deterministic sequence so merged + co-resident assign the
# same circuit blocks (block assignment is stateful/round-robin, but identical for an identical call order).
_SURPRISE_BATTERY = [
    ("alpha", "alpha"),     # confirm (same concept -> same block -> cancels)
    ("beta", "gamma"),      # contradict (distinct stored blocks)
    ("delta", "omega"),     # contradict / novel asserted
    ("kappa", "kappa"),     # confirm
]


def _surprise_reads(organ: SurpriseProductionOrgan) -> dict:
    """Every production number the surprise organ exposes: the calibration (threshold + confirm/contradict/novel
    Hz + homeostat gains) and a read battery of surprise Hz. Flat {label: float}."""
    organ.ensure_built()
    out = {}
    c = organ.calib
    for k in ("confirm_hz", "contradict_hz", "novel_hz"):
        out[f"calib.{k}"] = float(c[k])
    out["threshold"] = float(organ.threshold)
    if "pred_gain_min" in c:
        out["calib.pred_gain_min"] = float(c["pred_gain_min"])
        out["calib.pred_gain_max"] = float(c["pred_gain_max"])
        out["calib.confirm_after_max"] = float(c["confirm_after_max"])
    for (ps, pa) in _SURPRISE_BATTERY:
        j = organ.judge("agent", "acts", ps, pa)
        out[f"judge[{ps}->{pa}].hz"] = float(j["surprise_hz"])
    return out


def _worldmodel_reads(organ: WorldModelProductionOrgan) -> dict:
    """Every production number the world-model organ exposes: the calibration (threshold + selected states +
    expected/violated Hz), the queryable prediction for +/- context, and the surprise battery. Flat {label: float}."""
    organ.ensure_built()
    out = {}
    c = organ.calib
    out["threshold"] = float(organ.threshold)
    out["calib.state_pos"] = float(c["state_pos"])
    out["calib.state_neg"] = float(c["state_neg"])
    for i, v in enumerate(c["expected_hz"]):
        out[f"calib.expected_hz[{i}]"] = float(v)
    for i, v in enumerate(c["violated_hz"]):
        out[f"calib.violated_hz[{i}]"] = float(v)
    for ctx in (+1, -1):
        e = organ.expectation(ctx)
        out[f"expect[{ctx:+d}].pred_pos"] = float(e["pred_pos_rate"])
        out[f"expect[{ctx:+d}].pred_neg"] = float(e["pred_neg_rate"])
        out[f"expect[{ctx:+d}].margin"] = float(e["pred_margin"])
    for ctx, obs in ((+1, +1), (+1, -1), (-1, -1), (-1, +1)):
        r = organ.read_surprise(ctx, obs)
        out[f"surprise[ctx{ctx:+d},obs{obs:+d}].hz"] = float(r["surprise_hz"])
    return out


def _max_delta(a: dict, b: dict):
    """Max abs delta over the shared keys of two flat read dicts + the worst-offending key."""
    keys = sorted(set(a) & set(b))
    worst_k, worst = None, 0.0
    for k in keys:
        d = abs(a[k] - b[k])
        if d >= worst:
            worst, worst_k = d, k
    missing = sorted(set(a) ^ set(b))
    return worst, worst_k, missing


def run_seed(seed: int, verbose: bool = True) -> dict:
    # ── MERGED: both organs on ONE shared bridge (build surprise FIRST, then worldmodel — the production warm order). ──
    shared = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    m_surprise = SurpriseProductionOrgan(seed=seed, shared=shared)
    m_worldmodel = WorldModelProductionOrgan(seed=seed, shared=shared)
    m_surprise.ensure_built()
    m_worldmodel.ensure_built()

    # ── CO-RESIDENT baseline: each organ on its OWN bridge, both merge flags ON (single-organ MergedSubstrate,
    #    the identical construction path) -> merged-vs-solo isolates the merge itself. ──
    subS = MergedSubstrate(seed=seed, organs=("surprise",))
    subW = MergedSubstrate(seed=seed, organs=("worldmodel",))
    s_surprise = SurpriseProductionOrgan(seed=seed, shared=subS)
    s_worldmodel = WorldModelProductionOrgan(seed=seed, shared=subW)
    s_surprise.ensure_built()
    s_worldmodel.ensure_built()

    # ── ONE SHARED POOL (merged): both organs share ONE bridge + ONE cp_membrane_potential_v array. ──
    same_bridge = m_surprise.bridge is m_worldmodel._st["bridge"]
    same_array = m_surprise.bridge.cp_membrane_potential_v is m_worldmodel._st["bridge"].cp_membrane_potential_v
    n_all = int(m_surprise.bridge.cp_membrane_potential_v.shape[0])
    from research.runners._spiking_expectation_rpe_derisk import _idx as _sidx
    all_regions = ("cue", "patient_expected", "patient_asserted", "surprise",
                   "state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
    region_sizes = {r: int(len(_sidx(m_surprise.bridge, r))) for r in all_regions}
    one_pool = bool(same_bridge and same_array
                    and n_all >= sum(region_sizes.values())
                    and all(int(_host(_sidx(m_surprise.bridge, r)).max()) < n_all for r in all_regions))

    # ── BYTE-IDENTITY of the production reads (the load-bearing claim). ──
    ms = _surprise_reads(m_surprise); ss = _surprise_reads(s_surprise)
    mw = _worldmodel_reads(m_worldmodel); sw = _worldmodel_reads(s_worldmodel)
    surp_delta, surp_key, surp_missing = _max_delta(ms, ss)
    wm_delta, wm_key, wm_missing = _max_delta(mw, sw)
    surprise_byte_id = bool(surp_delta == 0.0 and not surp_missing)
    worldmodel_byte_id = bool(wm_delta == 0.0 and not wm_missing)

    # ── FACULTIES ALIVE on the merged bridge (functional read-outs). ──
    surp_sep = ms["calib.contradict_hz"] / max(ms["calib.confirm_hz"], 1e-6)
    surprise_alive = bool(surp_sep >= 2.0 and ms["calib.contradict_hz"] >= 5.0)
    pos_sign = m_worldmodel.expectation(+1)["pred_sign"]
    neg_sign = m_worldmodel.expectation(-1)["pred_sign"]
    exp_pos = mw["surprise[ctx+1,obs+1].hz"]; vio_pos = mw["surprise[ctx+1,obs-1].hz"]
    worldmodel_alive = bool(pos_sign > 0 and neg_sign < 0 and vio_pos > exp_pos)

    # ── DETERMINISM: build the merged substrate twice, hash the substrate (membrane, thresholds, connections). ──
    d1 = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel")); d1.ensure_built()
    d2 = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel")); d2.ensure_built()
    det_ok = bool(_arr_hash(d1.bridge.cp_membrane_potential_v) == _arr_hash(d2.bridge.cp_membrane_potential_v)
                  and _arr_hash(d1.bridge.cp_neuron_firing_thresholds) == _arr_hash(d2.bridge.cp_neuron_firing_thresholds)
                  and _arr_hash(d1.bridge.cp_connections.tocsr().data) == _arr_hash(d2.bridge.cp_connections.tocsr().data))

    rung_go = bool(one_pool and surprise_byte_id and worldmodel_byte_id
                   and surprise_alive and worldmodel_alive and det_ok)
    res = {
        "seed": seed, "one_shared_pool": one_pool, "same_bridge": same_bridge, "same_array": same_array,
        "n_all_neurons": n_all, "region_sizes": region_sizes,
        "surprise_read_maxdelta": surp_delta, "surprise_worst_key": surp_key, "surprise_missing_keys": surp_missing,
        "worldmodel_read_maxdelta": wm_delta, "worldmodel_worst_key": wm_key, "worldmodel_missing_keys": wm_missing,
        "surprise_byte_identical": surprise_byte_id, "worldmodel_byte_identical": worldmodel_byte_id,
        "surprise_separation_ratio": float(surp_sep), "surprise_alive": surprise_alive,
        "worldmodel_pred_pos_sign": int(pos_sign), "worldmodel_pred_neg_sign": int(neg_sign),
        "worldmodel_expected_hz": exp_pos, "worldmodel_violated_hz": vio_pos, "worldmodel_alive": worldmodel_alive,
        "determinism_ok": det_ok, "rung_go": rung_go,
    }
    if verbose:
        print(f"  [seed {seed}] pool={one_pool}(N={n_all}) det={det_ok} | "
              f"surprise byte-id={surprise_byte_id}(d={surp_delta:.2e}@{surp_key}) sep={surp_sep:.1f}x alive={surprise_alive} | "
              f"worldmodel byte-id={worldmodel_byte_id}(d={wm_delta:.2e}@{wm_key}) "
              f"pred±=({pos_sign:+d}/{neg_sign:+d}) vio>exp={vio_pos:.1f}>{exp_pos:.1f} alive={worldmodel_alive} | "
              f"RUNG-GO={rung_go}", flush=True)
    return res


def _gate(n_go, n):
    return "GO" if ((n >= 6 and n_go >= 5) or (n < 6 and n_go == n)) else "BOUNDARY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    print("=== ONE-BRAIN MERGE — RUNG 1: surprise + world-model production organs on ONE shared spiking bridge ===")
    print("    two merge flags ON (per_region_threshold_heterogeneity + per_region_homeostasis_isolation); no cross synapse")
    results = [run_seed(s) for s in seeds]
    n = len(results)
    n_pool = sum(r["one_shared_pool"] for r in results)
    n_sbi = sum(r["surprise_byte_identical"] for r in results)
    n_wbi = sum(r["worldmodel_byte_identical"] for r in results)
    n_salive = sum(r["surprise_alive"] for r in results)
    n_walive = sum(r["worldmodel_alive"] for r in results)
    n_det = sum(r["determinism_ok"] for r in results)
    n_go = sum(r["rung_go"] for r in results)
    max_sd = max(r["surprise_read_maxdelta"] for r in results)
    max_wd = max(r["worldmodel_read_maxdelta"] for r in results)
    print("\n=== VERDICT (rung 1) ===")
    print(f"  one shared neuron pool (both organs, one array): {n_pool}/{n}")
    print(f"  surprise reads BYTE-IDENTICAL merged-vs-coresident:   {n_sbi}/{n}  ->  {_gate(n_sbi, n)}  (max delta {max_sd:.2e})")
    print(f"  worldmodel reads BYTE-IDENTICAL merged-vs-coresident: {n_wbi}/{n}  ->  {_gate(n_wbi, n)}  (max delta {max_wd:.2e})")
    print(f"  surprise faculty alive (merged):   {n_salive}/{n}")
    print(f"  worldmodel faculty alive (merged): {n_walive}/{n}")
    print(f"  determinism (cfg.seed incl thresholds): {n_det}/{n}")
    print(f"  RUNG-1 MERGE (pool + byte-id both + alive both + det): {n_go}/{n}  ->  {_gate(n_go, n)}")
    payload = {"mode": "onebrain_merge_rung1", "organs": ["surprise", "worldmodel"],
               "merge_flags": ["per_region_threshold_heterogeneity", "per_region_homeostasis_isolation"],
               "n_seeds": n, "results": results,
               "n_one_shared_pool": n_pool, "n_surprise_byte_identical": n_sbi, "n_worldmodel_byte_identical": n_wbi,
               "n_surprise_alive": n_salive, "n_worldmodel_alive": n_walive, "n_determinism_ok": n_det,
               "n_rung_go": n_go, "max_surprise_read_delta": max_sd, "max_worldmodel_read_delta": max_wd,
               "surprise_byteid_verdict": _gate(n_sbi, n), "worldmodel_byteid_verdict": _gate(n_wbi, n),
               "rung_verdict": _gate(n_go, n)}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
