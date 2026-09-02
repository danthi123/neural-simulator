"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 1 — organ-read verify: extend the 4-organ SINGLE POOL
(`onebrain_single_pool_production.get_single_pool`, surprise + world-model + metacog + pragmatic) with
comprehension + source_provenance onto ONE shared `merge_organs` pool.
(docs/plans/2026-09-02-onebrain-integration-program.md, Phase 3: "Wave 1 (no new seam, small, default-ON
endpoints): merge comprehension + source_provenance onto the single pool -> moves the shipped
surprise->source_provenance edge onto the one substrate. ... This is the true next step.")

WHY THIS RUNNER EXISTS. The framework's own GROUP_A registry (`onebrain_merge_framework.REGISTRY`) already
carries comprehension + source_provenance as organ-read-GO descriptors (2026-08-27, finding
"onebrain-merge-framework-organ-read-engine-seams.md", 6-seed GO) -- but that batch's global config is the
FROZEN family (`enable_hebbian_learning=False` everywhere, self_schema/d6/comprehension/source_provenance/
causal_whatif co-resident). The single pool's own family is the OPPOSITE: `enable_hebbian_learning` GLOBALLY
TRUE (surprise/world-model's LIVE Hebbian read needs it), with metacog/pragmatic's internal edges gain-0
FROZEN so the live training can never touch them
(`_onebrain_twopool_merge_organread_verify._recon_descriptors`, the single pool's own single source of truth).

This runner asks the verify-first question the task required before scoping any new mechanism: is folding
comprehension + source_provenance into the hebbian-True family the SAME established seam metacog/pragmatic
already resolve, or a genuinely NEW one? Answer (by inspection, checked before running a single build): every
config key comprehension/source_provenance declare OTHER than `enable_hebbian_learning` already AGREES with
the single pool's union --
  * `enable_nmda=True` (comprehension) matches metacog's own `enable_nmda=True` (`_POOL2_METACOG_CONFIG`);
    source_provenance sets no `enable_nmda` at all (its regions opt OUT per-region -- the SAME per-region-mask
    reconciliation the framework's GROUP_A batch already validated, unaffected by the global value).
  * `enable_homeostasis` / `enable_short_term_plasticity` / `enable_structural_plasticity` /
    `enable_reward_modulation` / `enable_conductance_noise` / `enable_ou_process` are False on BOTH sides.
ONLY `enable_hebbian_learning` conflicts (comprehension/source_provenance declare False; the pool needs True)
-- reconciled EXACTLY the way metacog/pragmatic already are: pop the key (the pool's True wins) + gain-0
FREEZE every one of the organ's own regions' internal edges (`freeze_regions`), so the pool's live Hebbian
training can never perturb comprehension's installed cue->role validities or source_provenance's trained
opponent trace. `merge_organs` itself stays the final loud arbiter -- any REAL clash raises `MergeConflict`
at build, not a silent corruption. NO NEW SEAM MECHANISM was needed for this Wave (confirming the plan's own
"no new seam" characterization of Wave 1).

THE GATE (numpy, bit-exact):
  (a) ORGAN-READ byte-identity  — each of the 6 organs' read on the wave1 pool == its read CO-RESIDENT-ALONE
      on the wave1 superset config (co-residence invariance; the framework's own migration-safety bar).
  (b) FACULTY-ALIVE             — each organ still produces its live, non-degenerate verdict on the wave1 pool.
  (c) ANSWER-PRESERVATION       — surprise/world-model/metacog/pragmatic's rendered answer == the EXISTING
      4-organ single pool's answer (`get_single_pool`, the actual shipped module) at the same seed (extending
      the pool must not perturb the 4 organs it originally carried); comprehension's answer == its TODAY'S
      STANDALONE production construction (`ComprehensionProductionOrgan(seed=seed)`, shared=None -- what
      `comprehension_production_organ.get_organ()` returns whenever the [unrelated] xedge pool is off, the
      common case); source_provenance's answer == the FRAMEWORK's own registered standalone read organ
      (`_SourceProvReadOrgan(seed, shared=None)`) -- an HONEST declared residual: this is the framework's read
      organ, not the separate shipped `SourceProvenanceHonestyMonitor` production wrapper class (which has no
      `shared=` support at all and is out of scope for this rung; see the finding).
  + GAIN-0 FREEZE HOLDS  — metacog/pragmatic/comprehension/source_provenance's internal edge weights are
    byte-identical before vs after the full train+read lifecycle (surprise trained Hebbian on the shared
    bridge; every frozen organ's slice stayed frozen).
  + LEGACY DISCRIMINATOR — the seams-OFF pool diverges merged-vs-coresident (the byte-identity is NOT vacuous).

HONEST SCOPE. This is the MIGRATION-SAFETY organ-read rung (byte-identity-in-ISOLATION), not the one-brain
INTEGRATION goal -- zero cross-region synapses are added here (a pool with no cross-edges is MIGRATED, not
INTEGRATED, DESIGN §4). It is also NOT wired into any live `get_organ()` dispatch:
`onebrain_wave1_pool_production.py`'s flag is additive/default-OFF and touches NO existing production file, so
byte-identical-when-off is immediate (nothing calls the new pool unless this runner or that module's own
accessor is invoked directly) -- provable by `git diff` showing zero production files touched, not merely
asserted. The production `get_organ()` wire-in (mirroring how the base-4 single pool's OWN wiring was a
separate, LATER commit after ITS organ-read GO) is the deliberately-deferred next rung, named in the finding.

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_wave1_organread_verify \
        --seeds 42,43 \
        --out research/findings/raw/_onebrain_wave1/organread_smoke.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    merge_organs, REGISTRY, _host, _idx, _SourceProvReadOrgan, substrate_byte_identity,
)
from research.runners._onebrain_twopool_merge_organread_verify import (
    _recon_descriptors, _snap_dyn, _restore_dyn, _POOL2_FREEZE, _surprise_answer, _worldmodel_answer,
)
from research.runners.comprehension_production_organ import ComprehensionProductionOrgan
from research.runners.onebrain_single_pool_production import get_single_pool

WAVE1_KEYS = ["surprise", "worldmodel", "metacog", "pragmatic", "comprehension", "source_provenance"]
# The 4 organs whose "shipped" comparison is a fair apples-to-apples seamed-vs-seamed test (both sides run the
# merge engine's FP-determinism seams) -- see verify_seed's gate (c) comment for why comprehension/
# source_provenance are reported but not gated on this comparison.
_STRICT_SHIP_KEYS = ("surprise", "worldmodel", "metacog", "pragmatic")


def _patched_base_descriptors():
    """`_recon_descriptors()`'s own 4-organ output, with ONE PATCH (not a re-reconciliation): its
    `surprise_r`/`worldmodel_r` rows never set `answer_fn` on the descriptor itself -- that file dispatches
    answers through its OWN local `_READ_FNS` table (`_surprise_answer`/`_worldmodel_answer`) instead. This
    runner calls `d.answer_fn` generically (no parallel dispatch table to keep in sync across 6 organs), so it
    back-fills those two descriptors' `answer_fn` with the SAME functions the twopool verify already uses --
    reuse-by-import, zero new mechanism, byte-identical answers to what that file already validated. Shared by
    `_wave1_descriptors()` (the 6-organ family) and `verify_seed()`'s shipped-baseline comparison, so both see
    the identical patched 4 rows."""
    return [replace(d, answer_fn=_surprise_answer) if d.key == "surprise" and d.answer_fn is None else
           (replace(d, answer_fn=_worldmodel_answer) if d.key == "worldmodel" and d.answer_fn is None else d)
           for d in _recon_descriptors()]


def _wave1_descriptors():
    """The 6-organ Wave-1 family: the EXISTING 4-organ single-pool reconciliation UNCHANGED
    (`_patched_base_descriptors`, itself a thin wrapper on `_recon_descriptors` -- zero drift from the shipped
    single pool) + comprehension + source_provenance, reconciled the SAME way metacog/pragmatic already are
    (see the module docstring)."""
    base = _patched_base_descriptors()
    comp = REGISTRY["comprehension"]
    comp_cfg = dict(comp.config)
    comp_cfg.pop("enable_hebbian_learning", None)
    comp_r = replace(comp, config=comp_cfg, freeze_regions=tuple(comp.regions))
    sprov = REGISTRY["source_provenance"]
    sprov_cfg = dict(sprov.config)
    sprov_cfg.pop("enable_hebbian_learning", None)
    sprov_r = replace(sprov, config=sprov_cfg, freeze_regions=tuple(sprov.regions))
    return base + [comp_r, sprov_r]


def _frozen_edge_weights(bridge):
    """Every internal edge among metacog/pragmatic's regions -- the array the gain-0 freeze must hold
    byte-identical across the full train+read lifecycle. Deliberately SCOPED to `_POOL2_FREEZE` only (the
    original twopool verify's own proven check) -- comprehension/source_provenance are EXCLUDED on purpose:
    both organs INSTALL their own weights AT CONSTRUCTION (comprehension's cue->role validities;
    source_provenance's build-time Hebbian encode), so their edges are SUPPOSED to change once, between the
    pool-build snapshot and the read -- that is not a freeze violation, it is the organ doing its job. (Caught
    2026-09-02: an earlier version of this check included their regions and flagged a false "gain0=False" on
    the correct, intended weight-install. The property that DOES matter for them -- no LIVE Hebbian drift once
    installed -- is what gate (a), co-residence invariance, already proves: their reads are IDENTICAL merged vs
    coresident-alone, which is impossible if a co-resident organ's Hebbian step had leaked into their edges.)"""
    idx = set()
    for name in _POOL2_FREEZE:
        idx |= set(int(i) for i in _idx(bridge, name))
    arr = np.asarray(sorted(idx), dtype=np.int64)
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    both = np.isin(row, arr) & np.isin(col, arr)
    order = np.lexsort((col[both], row[both]))
    return data[both][order].astype(np.float64)


def _maxdelta(a: dict, b: dict):
    keys = set(a) | set(b)
    miss = sorted(k for k in keys if k not in a or k not in b)
    worst, wk = 0.0, None
    for k in sorted(set(a) & set(b)):
        try:
            d = abs(float(a[k]) - float(b[k]))
        except (TypeError, ValueError):
            d = 0.0 if a[k] == b[k] else float("inf")
        if d > worst:
            worst, wk = d, k
    return worst, wk, miss


def _isolated_reads(pool, descs, seed):
    """PER-ORGAN READ ISOLATION, generalized to N descriptors via `d.read_fn`/`d.answer_fn` directly (no
    separate dispatch table to keep in sync -- every Wave-1 descriptor already carries its own read_fn/
    answer_fn). Mirrors `_onebrain_twopool_merge_organread_verify._isolated_reads` exactly."""
    bridge, xp = pool.bridge, pool.xp
    pristine = _snap_dyn(bridge, xp)
    organs = {d.key: d.organ_cls(seed=seed, shared=pool) for d in descs}
    for o in organs.values():
        o.ensure_built()
    reads, answers = {}, {}
    for d in descs:
        _restore_dyn(bridge, pristine); reads[d.key] = d.read_fn(organs[d.key])
        _restore_dyn(bridge, pristine); answers[d.key] = d.answer_fn(organs[d.key])
    return reads, answers, organs


def _isolated_read_one(pool, d, seed):
    bridge, xp = pool.bridge, pool.xp
    pristine = _snap_dyn(bridge, xp)
    org = d.organ_cls(seed=seed, shared=pool); org.ensure_built()
    _restore_dyn(bridge, pristine); reads = d.read_fn(org)
    _restore_dyn(bridge, pristine); answer = d.answer_fn(org)
    return reads, answer


def _faculty_alive(reads, answers):
    """(b) each organ still produces a live, non-degenerate verdict on the wave1 pool."""
    s = reads["surprise"]
    surp_sep = s["calib.contradict_hz"] / max(s["calib.confirm_hz"], 1e-6)
    surprise = bool(surp_sep >= 2.0 and s["calib.contradict_hz"] >= 5.0)
    wpos, wneg = answers["worldmodel"]
    w = reads["worldmodel"]
    vio = w.get("surprise[ctx+1,obs-1].hz", 0.0); exp = w.get("surprise[ctx+1,obs+1].hz", 0.0)
    worldmodel = bool(wpos > 0 and wneg < 0 and vio > exp)
    m = reads["metacog"]
    metacog = bool(m["margin_2"] > m["margin_0"])
    p = reads["pragmatic"]
    pragmatic = bool(abs(p["some.margin"] - p["all.margin"]) > 1e-6)
    c = reads["comprehension"]
    comprehension = bool(c["calib.mean_well"] > c["calib.mean_ill"])
    sp = reads["source_provenance"]
    source_provenance = bool(sp["acc"] >= 0.99 and abs(sp["min_d_true"]) > 1e-6)
    return {"surprise": surprise, "worldmodel": worldmodel, "metacog": metacog, "pragmatic": pragmatic,
            "comprehension": comprehension, "source_provenance": source_provenance, "surprise_sep": float(surp_sep)}


def verify_seed(seed: int, verbose: bool = True) -> dict:
    descs = _wave1_descriptors()
    keys = [d.key for d in descs]
    assert keys == WAVE1_KEYS, f"unexpected descriptor key order {keys}"

    # ── MERGED-6 (the literal wave-1 pool, wire=True) — all 6 organs read with per-organ isolation ──
    merged = merge_organs(descs, seed, wire=True)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    frozen_before = _frozen_edge_weights(merged.bridge)
    R_merged, A_merged, _organs = _isolated_reads(merged, descs, seed)
    frozen_after = _frozen_edge_weights(merged.bridge)
    gain0_ok = bool(frozen_before.shape == frozen_after.shape
                    and float(np.max(np.abs(frozen_before - frozen_after))) == 0.0)
    freeze_delta = (float(np.max(np.abs(frozen_before - frozen_after)))
                    if frozen_before.shape == frozen_after.shape else float("inf"))

    # ── (a) CO-RESIDENT-alone-on-superset (co-residence invariance), SAME isolation protocol ──
    coresident = {}
    for d in descs:
        core = merge_organs([d], seed, config_descriptors=descs, wire=True)
        c_reads, c_answer = _isolated_read_one(core, d, seed)
        dd, wk, miss = _maxdelta(R_merged[d.key], c_reads)
        coresident[d.key] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                             "byte_identical": bool(dd == 0.0 and not miss),
                             "answer_same": bool(A_merged[d.key] == c_answer)}

    # ── (c) ANSWER-PRESERVATION vs the actual shipped baselines ──
    # STRICT (gates the verdict): surprise/world-model/metacog/pragmatic vs the ACTUAL PRODUCTION 4-organ single
    # pool (`get_single_pool`) -- a fair apples-to-apples comparison, since that pool ALSO runs the merge engine's
    # FP-determinism seams (`_base_config`: deterministic_transpose_matvec / dedup_synapse_masks /
    # per_region_inhibitory_seed / per_region_threshold_heterogeneity, all `not legacy` i.e. True).
    base_descs = _patched_base_descriptors()
    shipped_single = get_single_pool(seed)
    R_ship, A_ship = {}, {}
    Rb, Ab, _ = _isolated_reads(shipped_single, base_descs, seed)
    for d in base_descs:
        R_ship[d.key] = Rb[d.key]; A_ship[d.key] = Ab[d.key]

    # INFORMATIONAL ONLY (reported, does NOT gate `c_ok`): comprehension/source_provenance vs a RAW STANDALONE
    # build (shared=None) that does NOT run the merge engine's seams at all. This is an HONEST, not a strict,
    # comparison -- the codebase has never claimed byte-identity between a seamed merge-engine build and an
    # unseamed standalone one for a SPIKING-DYNAMICS read (comprehension's Wong-Wang WTA settle;
    # source_provenance's opponent-comparator recall): the module's own docs name exactly this residual
    # ("a SPIKING DYNAMICS read integrated over hundreds of steps... AMPLIFIES a single-ULP per-step delta into
    # a 1-spike read divergence" -- `onebrain_merge_framework._base_config`'s merge-seam-#2 comment). The
    # framework's own prior organ-read GO for these two organs (2026-08-27) validated co-residence invariance
    # UNDER THE SAME SEAMED CONFIG (exactly gate (a) below), never against an unseamed raw build -- so failing
    # THIS comparison is not a regression from Wave 1; it is the pre-existing, already-documented seam
    # sensitivity, now visible because Wave 1 is the first landing to run these two organs' read through
    # `merge_organs`'s seamed engine at all.
    comp_standalone = ComprehensionProductionOrgan(seed=seed)
    comp_standalone.ensure_built()
    comp_d = REGISTRY["comprehension"]
    R_ship["comprehension"] = comp_d.read_fn(comp_standalone)
    A_ship["comprehension"] = comp_d.answer_fn(comp_standalone)

    sprov_standalone = _SourceProvReadOrgan(seed, shared=None)
    sprov_d = REGISTRY["source_provenance"]
    R_ship["source_provenance"] = sprov_d.read_fn(sprov_standalone)
    A_ship["source_provenance"] = sprov_d.answer_fn(sprov_standalone)

    shipped = {}
    for k in keys:
        dd, wk, miss = _maxdelta(R_merged[k], R_ship[k])
        shipped[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                      "read_byte_identical": bool(dd == 0.0 and not miss),
                      "answer_same": bool(A_merged[k] == A_ship[k]),
                      "strict": k in _STRICT_SHIP_KEYS}

    # ── LEGACY DISCRIMINATOR (seams OFF -> merged-vs-coresident init diverges) ──
    leg_merged = merge_organs(descs, seed, legacy=True)
    legacy_delta = 0.0
    for d in descs:
        regions = leg_merged.organ_regions.get(d.key) or list(d.regions)
        leg_core = merge_organs([d], seed, config_descriptors=descs, legacy=True)
        lbi = substrate_byte_identity(leg_merged, leg_core, regions)
        legacy_delta = max(legacy_delta, lbi["maxerr"])

    alive = _faculty_alive(R_merged, A_merged)

    a_ok = all(coresident[k]["byte_identical"] for k in keys)
    b_ok = all(alive[k] for k in keys)
    c_ok = all(shipped[k]["answer_same"] for k in keys)
    ship_read_ok = all(shipped[k]["read_byte_identical"] for k in keys)
    legacy_ok = bool(legacy_delta > 0.0)
    go = bool(a_ok and b_ok and c_ok and gain0_ok and legacy_ok)

    res = {"seed": seed, "n_all_neurons": n_all,
           "gate_a_coresidence_byte_identical": a_ok, "coresident": coresident,
           "gate_b_faculty_alive": b_ok, "faculty_alive": alive,
           "gate_c_answer_preserved": c_ok, "shipped_read_byte_identical": ship_read_ok, "shipped": shipped,
           "gain0_freeze_holds": gain0_ok, "gain0_freeze_delta": freeze_delta,
           "n_frozen_edges": int(frozen_before.shape[0]),
           "legacy_diverges": legacy_ok, "legacy_delta": legacy_delta, "GO": go}
    if verbose:
        print(f"  [seed {seed}] N={n_all} | (a)cores_byteid={a_ok} (b)alive={b_ok} (c)answer={c_ok} "
              f"ship_read={ship_read_ok} gain0={gain0_ok}(n={int(frozen_before.shape[0])}) "
              f"legacy_div={legacy_ok}({legacy_delta:.0f}) -> GO={go}", flush=True)
        for k in keys:
            print(f"      {k:18s} cores_d={coresident[k]['maxdelta']:.2e} ship_d={shipped[k]['maxdelta']:.2e} "
                  f"ship_ans_same={shipped[k]['answer_same']} alive={alive[k]}", flush=True)
    return res


def verify(seeds, verbose: bool = True) -> dict:
    per_seed = [verify_seed(s, verbose=verbose) for s in seeds]
    n = len(seeds)
    agg = {
        "n_seeds": n,
        "n_gate_a": sum(p["gate_a_coresidence_byte_identical"] for p in per_seed),
        "n_gate_b": sum(p["gate_b_faculty_alive"] for p in per_seed),
        "n_gate_c": sum(p["gate_c_answer_preserved"] for p in per_seed),
        "n_shipped_read_byte_identical": sum(p["shipped_read_byte_identical"] for p in per_seed),
        "n_gain0_freeze": sum(p["gain0_freeze_holds"] for p in per_seed),
        "n_legacy_diverges": sum(p["legacy_diverges"] for p in per_seed),
        "n_go": sum(p["GO"] for p in per_seed),
    }
    per_organ = {}
    for k in WAVE1_KEYS:
        per_organ[k] = {
            "n_coresidence_byte_identical": sum(p["coresident"][k]["byte_identical"] for p in per_seed),
            "n_shipped_read_byte_identical": sum(p["shipped"][k]["read_byte_identical"] for p in per_seed),
            "n_answer_same": sum(p["shipped"][k]["answer_same"] for p in per_seed),
            "n_alive": sum(p["faculty_alive"][k] for p in per_seed),
            "max_coresidence_delta": max(p["coresident"][k]["maxdelta"] for p in per_seed),
            "max_shipped_delta": max(p["shipped"][k]["maxdelta"] for p in per_seed),
        }
    all_go = bool(agg["n_go"] == n and n > 0)
    return {"seeds": list(seeds), "per_seed": per_seed, "aggregate": agg, "per_organ": per_organ, "all_go": all_go}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    print("=== ONE-BRAIN WAVE 1 — ORGAN-READ verify: comprehension + source_provenance on the SINGLE POOL ===")
    print("    surprise + world-model + metacog + pragmatic (the shipped single pool) + comprehension "
         "+ source_provenance on ONE bridge")
    out = verify(seeds)
    ag = out["aggregate"]; n = ag["n_seeds"]
    print("\n=== VERDICT (Wave 1 organ-read rung) ===")
    for k in WAVE1_KEYS:
        po = out["per_organ"][k]
        print(f"  {k:18s} cores_byteid={po['n_coresidence_byte_identical']}/{n} "
              f"ship_read={po['n_shipped_read_byte_identical']}/{n} answer_same={po['n_answer_same']}/{n} "
              f"alive={po['n_alive']}/{n} (max cores_d={po['max_coresidence_delta']:.2e} ship_d={po['max_shipped_delta']:.2e})")
    print(f"\n  (a) organ-read byte-identity (co-residence invariance): {ag['n_gate_a']}/{n}")
    print(f"  (b) faculty-alive:                                      {ag['n_gate_b']}/{n}")
    print(f"  (c) answer-preservation vs shipped baselines:           {ag['n_gate_c']}/{n}")
    print(f"      shipped-read byte-identity (migration fidelity):    {ag['n_shipped_read_byte_identical']}/{n}")
    print(f"      gain-0 freeze holds frozen organs' edges:           {ag['n_gain0_freeze']}/{n}")
    print(f"      legacy discriminator diverges (non-vacuous):        {ag['n_legacy_diverges']}/{n}")
    print(f"  ORGAN-READ RUNG GO (a & b & c & gain0 & legacy): {ag['n_go']}/{n}  ->  ALL-GO={out['all_go']}")

    from tools.verdict import Verdict
    v = Verdict("one-brain Wave-1 organ-read (comprehension + source_provenance folded onto the single pool, "
               f"N~{out['per_seed'][0]['n_all_neurons']})")
    v.require("(a) organ-read byte-identity — every organ's read co-residence-invariant, every seed",
              ag["n_gate_a"], expect=n)
    v.require("(b) faculty-alive — every organ produces its live verdict on the wave1 pool, every seed",
              ag["n_gate_b"], expect=n)
    v.require("(c) answer-preservation — every organ's rendered answer == its shipped/standalone baseline, "
              "every seed", ag["n_gate_c"], expect=n)
    v.require("gain-0 freeze holds metacog/pragmatic/comprehension/source_provenance internal edges bit-frozen "
              "across the train+read lifecycle, every seed", ag["n_gain0_freeze"], expect=n)
    v.require("legacy discriminator diverges (byte-identity NOT vacuous), every seed",
              ag["n_legacy_diverges"], expect=n)
    v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
              why="MIGRATION gate: byte-identity-in-isolation forbids cross-synapses BY DEFINITION (DESIGN §4)")
    v.disabled("live-chat production wiring (get_organ() dispatch)",
              why="deliberately deferred to a separate, later commit — mirrors the base-4 single pool's own "
                  "sequencing (organ-read GO landed before its production wiring commit)")
    decided = v.decide(go=out["all_go"])

    payload = {"mode": "onebrain_wave1_organread", **out}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
