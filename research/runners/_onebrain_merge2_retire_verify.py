"""ONE-BRAIN MERGE — pool #2 RETIREMENT regression gate (the migration-safety check that GATES repointing
`get_merged_substrate2()` from the bespoke `MergedSubstrate2.ensure_built()` build to the declarative
`onebrain_merge_framework.merge_organs()` engine).

WHY THIS RUNNER, GIVEN `onebrain_merge_framework.py` ALREADY HAS `--smoke2`. `_smoke2` (added
2026-08-27, `2026-08-27-onebrain-merge-framework-pool2-fold.md`) already proves the PRODUCTION 2-organ
combo (`organs=("metacog","pragmatic")`, the one `get_merged_substrate2()` builds) round-trips
byte-identically — per-region init arrays AND the real `judge()`/`interpret()` reads — against the
declarative registry pool, 6/6 seeds. That finding explicitly named the retirement as "de-risked... needs a
dedicated regression pass" and flagged the reason it did NOT flip the class: `MergedSubstrate2` ALSO has
SINGLE-organ constructor callers outside the production entry point
(`_metacog_robust_confidence_derisk.py`'s `MergedSubstrate2(organs=("metacog",))` / `(("pragmatic",))`, and
`_onebrain_production_flip2_verify.py`'s identical CORESIDENT-baseline pattern) whose exact behavior a
thin-wrapper refactor must ALSO preserve — a code path `_smoke2` never exercises. THIS runner is that
dedicated pass: it re-verifies the 2-organ production combo (the literal repoint target) AND separately
verifies the two 1-organ combos (the callers the repoint must not silently break), each with a STRONGER
whole-bridge fingerprint (SHA1 over every per-neuron init array + the full wired connectivity, not just the
organ's own region slice) on top of the real production-organ reads. GO on all three combos, 6 seeds each, is
the gate this file exists to compute.

WHAT "byte-identical" MEANS HERE (`docs/TERMS.md`): for each combo, build the SAME organs' pool two ways —
(a) the bespoke `MergedSubstrate2(seed, organs=...)` (`onebrain_merge_production2.py`) and (b) the framework
`merge_organs([...], seed, wire=True)` (`onebrain_merge_framework.py`) — and require EXACT equality (max
delta 0.0 on every numeric read, identical SHA1 hash on the whole bridge) between the two builds. `wire=True`
is required: pool #2's wiring inject (base pathways + metacog's assembly loops, per-region-seamed) is
ALWAYS-ON in the shipped class (unlike pool #1, whose wiring is a `post_build` add-on), so the framework must
run its own equivalent inject to reproduce it.

SEEDING (`cfg.seed`, NOT `actual_seed_used` — CLAUDE.md's seed trap): both build paths route the seed through
`cfg.seed = int(seed)` (`MergedSubstrate2.ensure_built` line ~174; `onebrain_merge_framework._base_config`
line ~105), so a seed genuinely controls both substrates' heterogeneity/threshold/wiring RNG draws.

Run (numpy CPU, bit-exact, foreground):
    SIM_BACKEND=numpy python -m research.runners._onebrain_merge2_retire_verify \\
        --seeds 42,43,44,100,101,102 \\
        --out research/findings/raw/2026-08-27-merged-substrate2-retirement-framework-backed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    merge_organs, METACOG, PRAGMATIC, _host, _METACOG_READ_EVIDENCE, _PRAGMATIC_READ_UTTS,
)
from research.runners.onebrain_merge_production2 import MergedSubstrate2
from research.runners.metacog_production_organ import MetacogProductionOrgan
from research.runners.pragmatic_production_organ import PragmaticProductionOrgan
from tools.verdict import Verdict


# ── the whole-bridge fingerprint: every per-neuron init array a merge seam could perturb, PLUS the full wired
#    connectivity (row, col, weight — sorted to a canonical order so two builds with the same edges hash equal
#    regardless of insertion order). Stronger than `onebrain_merge_framework._smoke2`'s per-region-slice
#    init-array compare: this catches ANY divergence anywhere on the bridge, including a wiring-plan mismatch
#    outside the two organs' own regions (there are none in a 2-region-family pool, but the hash proves it,
#    not assumes it).
_HASH_ARRAYS = (
    "cp_membrane_potential_v", "cp_recovery_variable_u", "cp_neuron_firing_thresholds",
    "cp_izh_a", "cp_izh_b", "cp_izh_C", "cp_izh_c_reset", "cp_izh_d_increment",
    "cp_izh_vpeak", "cp_izh_vt", "cp_izh_vr",
)


def _bridge_hash(bridge) -> str:
    """SHA1 over the whole bridge's per-neuron state arrays + its full wired connectivity (edges sorted
    row-major so the hash is insertion-order invariant). Identical hash == the two bridges are the SAME
    substrate, not just agreeing on the two organs' own region slices."""
    h = hashlib.sha1()
    for name in _HASH_ARRAYS:
        arr = _host(getattr(bridge, name, None))
        if arr is not None:
            h.update(np.ascontiguousarray(arr).astype(np.float64).tobytes())
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(_host(coo.row), dtype=np.int64)
    col = np.asarray(_host(coo.col), dtype=np.int64)
    data = np.asarray(_host(coo.data), dtype=np.float64)
    order = np.lexsort((col, row))
    h.update(row[order].tobytes())
    h.update(col[order].tobytes())
    h.update(data[order].tobytes())
    return h.hexdigest()


def _metacog_check(seed: int, shared_ship, shared_eng) -> dict:
    """Run the REAL production `MetacogProductionOrgan.judge()` path against both builds and require
    EXACT agreement on the confidence margin, the confident/uncertain decision, AND the self-calibrated
    threshold (the calibration battery runs on each build's own slice, so a threshold mismatch alone would
    flag a divergence the margin compare might miss)."""
    m_ship = MetacogProductionOrgan(seed=seed, shared=shared_ship)
    m_eng = MetacogProductionOrgan(seed=seed, shared=shared_eng)
    worst, worst_where = 0.0, None
    for e in _METACOG_READ_EVIDENCE:
        js, je = m_ship.judge(e), m_eng.judge(e)
        d = abs(float(js["balance"]) - float(je["balance"]))
        if d > worst:
            worst, worst_where = d, f"judge({e}).balance"
        if bool(js["confident"]) != bool(je["confident"]):
            worst, worst_where = float("inf"), f"judge({e}).confident {js['confident']}!={je['confident']}"
    d = abs(float(m_ship.threshold) - float(m_eng.threshold))
    if d > worst:
        worst, worst_where = d, "threshold"
    return {"identical": bool(worst == 0.0), "max_delta": worst, "worst_where": worst_where}


def _pragmatic_check(seed: int, shared_ship, shared_eng) -> dict:
    """Run the REAL production `PragmaticProductionOrgan.interpret()` path against both builds and require
    EXACT agreement on the belief distribution AND the rendered enriched-interpretation phrase."""
    p_ship = PragmaticProductionOrgan(seed=seed, shared=shared_ship)
    p_eng = PragmaticProductionOrgan(seed=seed, shared=shared_eng)
    worst, worst_where = 0.0, None
    for u in _PRAGMATIC_READ_UTTS:
        is_, ie = p_ship.interpret(u), p_eng.interpret(u)
        for i, (bs, be) in enumerate(zip(is_["belief"], ie["belief"])):
            d = abs(float(bs) - float(be))
            if d > worst:
                worst, worst_where = d, f"interpret({u!r}).belief[{i}]"
        if is_["enriched_interpretation"] != ie["enriched_interpretation"]:
            worst, worst_where = float("inf"), (
                f"interpret({u!r}).enriched_interpretation "
                f"{is_['enriched_interpretation']!r}!={ie['enriched_interpretation']!r}")
    return {"identical": bool(worst == 0.0), "max_delta": worst, "worst_where": worst_where}


def verify_seed(seed: int) -> dict:
    """The repoint's regression gate for ONE seed: the PRODUCTION 2-organ combo (what `get_merged_substrate2()`
    builds — the literal repoint target) PLUS the two 1-organ combos (`_metacog_robust_confidence_derisk.py`
    / `_onebrain_production_flip2_verify.py`'s CORESIDENT-baseline callers — the code paths a thin-wrapper
    refactor of `MergedSubstrate2` would ALSO route through the framework, so they must be proven too)."""
    out = {"seed": seed}

    # ── the PRODUCTION combo: both organs on one shared bridge (get_merged_substrate2's exact construction) ──
    ship2 = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    ship2.ensure_built()
    eng2 = merge_organs([METACOG, PRAGMATIC], seed=seed, wire=True)

    bridge_hash_identical = bool(_bridge_hash(ship2.bridge) == _bridge_hash(eng2.bridge))
    n_ship, n_eng = int(ship2.bridge.cp_membrane_potential_v.shape[0]), int(eng2.bridge.cp_membrane_potential_v.shape[0])
    mc = _metacog_check(seed, ship2, eng2)
    pr = _pragmatic_check(seed, ship2, eng2)

    out["bridge_hash_identical"] = bridge_hash_identical
    out["n_shipped"] = n_ship
    out["n_engine"] = n_eng
    out["metacog_identical"] = mc["identical"]
    out["metacog_max_delta"] = mc["max_delta"]
    out["metacog_worst_where"] = mc["worst_where"]
    out["pragmatic_identical"] = pr["identical"]
    out["pragmatic_max_delta"] = pr["max_delta"]
    out["pragmatic_worst_where"] = pr["worst_where"]
    out["production_combo_go"] = bool(bridge_hash_identical and n_ship == n_eng and mc["identical"] and pr["identical"])

    # ── the SOLO combos: the other constructor pattern the repoint must not silently break (the CORESIDENT
    #    baseline `_onebrain_production_flip2_verify.py` / `_metacog_robust_confidence_derisk.py` build) ──
    ship_m = MergedSubstrate2(seed=seed, organs=("metacog",))
    ship_m.ensure_built()
    eng_m = merge_organs([METACOG], seed=seed, wire=True)
    solo_metacog_hash_identical = bool(_bridge_hash(ship_m.bridge) == _bridge_hash(eng_m.bridge))
    solo_mc = _metacog_check(seed, ship_m, eng_m)

    ship_p = MergedSubstrate2(seed=seed, organs=("pragmatic",))
    ship_p.ensure_built()
    eng_p = merge_organs([PRAGMATIC], seed=seed, wire=True)
    solo_pragmatic_hash_identical = bool(_bridge_hash(ship_p.bridge) == _bridge_hash(eng_p.bridge))
    solo_pr = _pragmatic_check(seed, ship_p, eng_p)

    out["solo_metacog_hash_identical"] = solo_metacog_hash_identical
    out["solo_metacog_read_identical"] = solo_mc["identical"]
    out["solo_metacog_max_delta"] = solo_mc["max_delta"]
    out["solo_pragmatic_hash_identical"] = solo_pragmatic_hash_identical
    out["solo_pragmatic_read_identical"] = solo_pr["identical"]
    out["solo_pragmatic_max_delta"] = solo_pr["max_delta"]
    out["solo_combos_go"] = bool(solo_metacog_hash_identical and solo_mc["identical"]
                                 and solo_pragmatic_hash_identical and solo_pr["identical"])

    out["all_go"] = bool(out["production_combo_go"] and out["solo_combos_go"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/2026-08-27-merged-substrate2-retirement-framework-backed.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    per_seed = {}
    print("=== onebrain_merge2_retire_verify (MergedSubstrate2 vs framework merge_organs, "
          f"seeds={seeds}) ===")
    for s in seeds:
        r = verify_seed(s)
        per_seed[str(s)] = r
        print(f"  seed={s}  production: hash={r['bridge_hash_identical']} "
              f"metacog={r['metacog_identical']} pragmatic={r['pragmatic_identical']}  "
              f"solo: metacog={r['solo_metacog_hash_identical'] and r['solo_metacog_read_identical']} "
              f"pragmatic={r['solo_pragmatic_hash_identical'] and r['solo_pragmatic_read_identical']}  "
              f"-> {'GO' if r['all_go'] else 'NO-GO'}")

    n_go = sum(bool(r["all_go"]) for r in per_seed.values())
    n_production_go = sum(bool(r["production_combo_go"]) for r in per_seed.values())
    go = bool(n_go == len(seeds) and seeds)
    print(f"  PRODUCTION-COMBO GO: {n_production_go}/{len(seeds)}   OVERALL (production+solo) GO: {n_go}/{len(seeds)}")
    print(f"  VERDICT: {'GO' if go else 'NO-GO'} ({n_go}/{len(seeds)} == 6/6 required)")

    # ── the verdict must travel with what earned it (tools/gates/verdict_preconditions.py): every one of the
    #    18 (3 combos x 6 seeds) byte-identity checks above is registered as an EARNED precondition, not just
    #    summarized as a count -- an unmeasured or failed seed makes the verdict UNDEFINED, never a plain GO. ──
    v = Verdict("onebrain pool2 (MergedSubstrate2) retirement byte-identity gate, 6 seeds x 3 combos")
    for s in seeds:
        r = per_seed[str(s)]
        v.require(f"seed{s} production bridge_hash_identical", r["bridge_hash_identical"], expect=True)
        v.require(f"seed{s} production n_shipped==n_engine", r["n_shipped"] == r["n_engine"], expect=True)
        v.require(f"seed{s} production metacog judge() identical", r["metacog_identical"], expect=True)
        v.require(f"seed{s} production pragmatic interpret() identical", r["pragmatic_identical"], expect=True)
        v.require(f"seed{s} solo-metacog bridge_hash_identical", r["solo_metacog_hash_identical"], expect=True)
        v.require(f"seed{s} solo-metacog judge() identical", r["solo_metacog_read_identical"], expect=True)
        v.require(f"seed{s} solo-pragmatic bridge_hash_identical", r["solo_pragmatic_hash_identical"], expect=True)
        v.require(f"seed{s} solo-pragmatic interpret() identical", r["solo_pragmatic_read_identical"], expect=True)
    decided = v.decide(go=go, verbose=False)

    payload = {
        "mode": "onebrain_merge2_retire_verify",
        "seeds": seeds,
        "per_seed": per_seed,
        "n_go": n_go,
        "n_seeds": len(seeds),
        "n_production_combo_go": n_production_go,
        "go": go,
        "verdict": decided["status"],
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
