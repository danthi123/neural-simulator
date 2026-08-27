"""ONE-BRAIN MERGE — the BATCHED, REGISTRY-DRIVEN verify (the O(N)->O(1) migration gate).

DESIGN: research/findings/2026-08-27-onebrain-merge-framework-DESIGN.md §3. This is the generalization of the
bespoke per-organ verify (`_onebrain_twopool_merge_derisk.byte_identity` / `_onebrain_twopool_organread_verify`)
to a SINGLE code path that reads the `onebrain_merge_framework.REGISTRY` and gates EVERY registered organ in ONE
sweep. Adding an organ adds a REGISTRY row, not a runner.

WHAT IT GATES per registered organ, merged-vs-CORESIDENT (the organ ALONE on the merged pool's SUPERSET config,
so a non-zero delta isolates CO-RESIDENCE, not a config change):

  (1) SUBSTRATE-INIT byte-identity  — the MIGRATION-SAFETY gate. Every per-neuron init array of the organ's
      region slice (thresholds, v, u, the 8 Izhikevich params, the 2 gate masks) is byte-IDENTICAL merged vs
      coresident. This needs ONLY the descriptor's spec_fn+config — NO `shared=` plumbing — so it scales to any
      registered organ. It is the exact gate `2026-08-27-onebrain-twopool-merge-...-6seed-GO.md` used for 4 organs.
  (2) ORGAN-READ byte-identity      — the STRONGER gate, run ONLY for organs whose shipped class supports
      `shared=` (`descriptor.supports_shared`): construct the UNMODIFIED organ against the merged pool and against
      the coresident pool, run its real read_fn/answer_fn, require bit-identical reads + preserved answers.
  (3) LEGACY DISCRIMINATOR           — with the name-keyed seams OFF, at least one organ's slice must DIVERGE
      merged-vs-coresident, so the seam-ON byte-identity is NOT a vacuous all-zero compare.

HONEST SCOPE. Byte-identity-in-isolation is the MIGRATION gate, NOT the one-brain INTEGRATION goal (DESIGN §4):
it deliberately FORBIDS the cross-region interaction that IS the goal. A pool with zero cross-edges is MIGRATED,
not INTEGRATED. The functional-integration F-gate is the named next phase.

Run (CPU, bit-exact):
    SIM_BACKEND=numpy python -m research.runners.onebrain_merge_verify \
        --keys all --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_merge_groupA_6seed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    REGISTRY, GROUP_A_KEYS, GROUP_A_DEFERRED, merge_organs, substrate_byte_identity,
    _region_indices, _host,
)

_HET_PARAM_ARRAYS = ("cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_d_increment")


def _het_delta(merged, het_off, regions):
    """Max |delta| in the Izhikevich params of `regions` between the normal merged pool and the same pool with
    the per-region param-het mask cleared. > 0 == the reconciled param-het is genuinely doing work (not a vacuous
    all-zero het). (Raw std mixes RS/FS types, so this masked-vs-unmasked delta is the valid witness — twopool.)"""
    delta = 0.0
    for rname in regions:
        mi = _region_indices(merged.bridge, rname)
        oi = _region_indices(het_off.bridge, rname)
        if mi.size != oi.size:
            return float("inf")
        for a in _HET_PARAM_ARRAYS:
            x = _host(getattr(merged.bridge, a, None)); y = _host(getattr(het_off.bridge, a, None))
            if x is None or y is None:
                continue
            import numpy as _np
            d = float(_np.max(_np.abs(x[mi].astype(_np.float64) - y[oi].astype(_np.float64)))) if mi.size else 0.0
            delta = max(delta, d)
    return delta


def _max_delta(a: dict, b: dict):
    """Max |delta| across two numeric-read dicts + any missing keys. 0.0 & no-missing == bit-identical reads."""
    if a is None or b is None:
        return float("inf"), "one-none", []
    keys = set(a) | set(b)
    missing = sorted(k for k in keys if k not in a or k not in b)
    worst, worst_key = 0.0, None
    for k in sorted(set(a) & set(b)):
        try:
            d = abs(float(a[k]) - float(b[k]))
        except (TypeError, ValueError):
            d = 0.0 if a[k] == b[k] else float("inf")
        if d > worst:
            worst, worst_key = d, k
    return worst, worst_key, missing


def _resolve_keys(keys_arg: str):
    # 'all' == the GROUP A migration batch (NOT the already-merged pool-1 organs, which own a colliding "cue").
    if keys_arg is None or keys_arg.strip().lower() in ("all", "groupa", "group_a", "*", ""):
        return list(GROUP_A_KEYS)
    return [k.strip() for k in keys_arg.split(",") if k.strip()]


def verify_seed(keys, seed: int, verbose: bool = True) -> dict:
    descs = [REGISTRY[k] for k in keys]
    merged = merge_organs(descs, seed)
    leg_merged = merge_organs(descs, seed, legacy=True)
    # LOAD-BEARING control: same pool with the param-het mask cleared (only rebuilt if some organ is param-het).
    any_het = any(getattr(d, "param_het", False) for d in descs)
    het_off = merge_organs(descs, seed, force_het_off=True) if any_het else None

    organs = {}
    for d in descs:
        core = merge_organs([d], seed, config_descriptors=descs)               # organ alone, superset config
        regions = merged.organ_regions.get(d.key) or list(d.regions)           # build-discovered names
        sbi = substrate_byte_identity(merged, core, regions)
        leg_core = merge_organs([d], seed, config_descriptors=descs, legacy=True)
        lbi = substrate_byte_identity(leg_merged, leg_core, regions)

        entry = {
            "substrate_maxerr": sbi["maxerr"],
            "substrate_byte_identical": bool(sbi["maxerr"] == 0.0),
            "legacy_maxerr": lbi["maxerr"],
            "param_het": bool(getattr(d, "param_het", False)),
            "het_loadbearing_delta": None, "het_loadbearing_ok": None,
            "read_checked": False, "read_byte_identical": None, "read_maxerr": None,
            "answer_checked": False, "answer_same": None,
        }
        if entry["param_het"] and het_off is not None:
            hd = _het_delta(merged, het_off, regions)
            entry["het_loadbearing_delta"] = hd
            entry["het_loadbearing_ok"] = bool(hd > 0.0)

        # (2) ORGAN-READ byte-identity — only where the shipped class runs UNMODIFIED against a MergedPool.
        if d.supports_shared and d.organ_cls is not None and d.read_fn is not None:
            try:
                m_org = d.organ_cls(seed=seed, shared=merged)
                c_org = d.organ_cls(seed=seed, shared=core)
                rd, rk, miss = _max_delta(d.read_fn(m_org), d.read_fn(c_org))
                entry["read_checked"] = True
                entry["read_maxerr"] = rd
                entry["read_worst_key"] = rk
                entry["read_missing_keys"] = miss
                entry["read_byte_identical"] = bool(rd == 0.0 and not miss)
                if d.answer_fn is not None:
                    entry["answer_checked"] = True
                    entry["answer_same"] = bool(d.answer_fn(m_org) == d.answer_fn(c_org))
            except Exception as exc:                                            # honest: record, do not hide
                entry["read_checked"] = True
                entry["read_byte_identical"] = False
                entry["read_error"] = f"{type(exc).__name__}: {exc}"

        # per-organ GO: substrate byte-identical, AND (param-het organ) the reconciliation is load-bearing,
        # AND (if a read was checked) read byte-identical + answer preserved.
        go = entry["substrate_byte_identical"]
        if entry["param_het"]:
            go = go and bool(entry["het_loadbearing_ok"])
        if entry["read_checked"]:
            go = go and bool(entry["read_byte_identical"])
            if entry["answer_checked"]:
                go = go and bool(entry["answer_same"])
        entry["GO"] = bool(go)
        organs[d.key] = entry

    legacy_diverges = any(organs[d.key]["legacy_maxerr"] > 0.0 for d in descs)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    res = {"seed": seed, "n_all_neurons": n_all, "organs": organs, "legacy_diverges": legacy_diverges}
    if verbose:
        tag = " ".join(f"{k}={'GO' if organs[k]['GO'] else 'X'}" for k in keys)
        print(f"  [seed {seed}] N={n_all} legacy_diverges={legacy_diverges} | {tag}", flush=True)
    return res


def verify(keys, seeds, verbose: bool = True) -> dict:
    per_seed = [verify_seed(keys, s, verbose=verbose) for s in seeds]
    n = len(seeds)
    per_organ = {}
    for k in keys:
        n_sub = sum(ps["organs"][k]["substrate_byte_identical"] for ps in per_seed)
        n_go = sum(ps["organs"][k]["GO"] for ps in per_seed)
        n_legacy_organ = sum(ps["organs"][k]["legacy_maxerr"] > 0.0 for ps in per_seed)
        is_het = any(ps["organs"][k]["param_het"] for ps in per_seed)
        n_het = sum(bool(ps["organs"][k]["het_loadbearing_ok"]) for ps in per_seed) if is_het else None
        read_checked = any(ps["organs"][k]["read_checked"] for ps in per_seed)
        n_read = sum(bool(ps["organs"][k]["read_byte_identical"]) for ps in per_seed) if read_checked else None
        per_organ[k] = {
            "n_substrate_byte_identical": n_sub, "n_go": n_go, "n_seeds": n,
            "n_legacy_diverges": n_legacy_organ, "param_het": is_het, "n_het_loadbearing": n_het,
            "read_checked": read_checked, "n_read_byte_identical": n_read,
            "verdict": "GO" if n_go == n and n > 0 else "NO-GO",
        }
    n_legacy = sum(ps["legacy_diverges"] for ps in per_seed)
    # STRONGER non-vacuousness: every organ diverged under seams-off on every seed (not merely SOME organ).
    n_all_legacy = sum(all(ps["organs"][k]["legacy_maxerr"] > 0.0 for k in keys) for ps in per_seed)
    all_go = all(per_organ[k]["verdict"] == "GO" for k in keys) and (n_legacy == n) and n > 0
    return {"keys": list(keys), "seeds": list(seeds), "per_seed": per_seed,
            "per_organ": per_organ, "n_legacy_diverges": n_legacy,
            "n_all_organ_legacy_diverges": n_all_legacy, "all_go": all_go}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", type=str, default="all", help="comma list of registry keys, or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    keys = _resolve_keys(args.keys)

    print("=== ONE-BRAIN MERGE — BATCHED registry verify (substrate-init byte-identity + organ-read where wired) ===")
    print(f"    registry keys: {keys}")
    out = verify(keys, seeds)

    print("\n=== PER-ORGAN VERDICT (6-seed) ===")
    for k in keys:
        po = out["per_organ"][k]
        het = "" if not po["param_het"] else f" het_loadbearing={po['n_het_loadbearing']}/{po['n_seeds']}"
        rd = "" if not po["read_checked"] else f" read_byte={po['n_read_byte_identical']}/{po['n_seeds']}"
        print(f"  {k:26s} substrate_byte={po['n_substrate_byte_identical']}/{po['n_seeds']}"
              f" legacy_diverges={po['n_legacy_diverges']}/{po['n_seeds']}{het}{rd}  -> {po['verdict']}")
    print(f"\n  pool legacy discriminator diverges: {out['n_legacy_diverges']}/{len(seeds)}  "
          f"(EVERY organ diverges: {out['n_all_organ_legacy_diverges']}/{len(seeds)})")
    n_go = sum(out["per_organ"][k]["verdict"] == "GO" for k in keys)
    print(f"  ORGANS GO: {n_go}/{len(keys)}   POOL ALL-GO: {out['all_go']}")
    if GROUP_A_DEFERRED:
        print("\n  GROUP-B/C DEFERRED (honest, with the seam each needs):")
        for k, why in GROUP_A_DEFERRED.items():
            print(f"    - {k}: {why}")

    from tools.verdict import Verdict
    v = Verdict(f"one-brain merge batched migration byte-identity ({len(keys)} Group-A organs, N-organ pool)")
    n = len(seeds)
    het_keys = [k for k in keys if out["per_organ"][k]["param_het"]]
    v.require("every registered organ substrate-byte-identical merged-vs-coresident, every seed",
              sum(out["per_organ"][k]["n_substrate_byte_identical"] for k in keys),
              expect=len(keys) * n)
    v.require("legacy discriminator diverges (byte-identity NOT vacuous), every seed",
              out["n_legacy_diverges"], expect=n)
    if het_keys:
        v.require("param-het reconciliation is load-bearing for every param-het organ, every seed",
                  sum(out["per_organ"][k]["n_het_loadbearing"] for k in het_keys), expect=len(het_keys) * n)
    v.require("every organ's per-seed GO gate holds, every seed",
              sum(out["per_organ"][k]["n_go"] for k in keys), expect=len(keys) * n)
    v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
               why="MIGRATION gate: byte-identity-in-isolation forbids cross-synapses BY DEFINITION (DESIGN §4)")
    v.disabled("organ-read / answer equivalence (no shipped Group-A class takes shared= today)",
               why="the substrate-init co-residence-invariance gate; organ-read needs shared= plumbing (next rung)")
    decided = v.decide(go=out["all_go"])

    payload = {"mode": "onebrain_merge_batched_verify", **out}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
