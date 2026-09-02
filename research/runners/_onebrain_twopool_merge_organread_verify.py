"""ONE-BRAIN TWO-POOL MERGE — the ORGAN-READ rung: run all FOUR core cortical organs' REAL production read
pipelines on ONE shared merged pool (the literal two-pool merge) and prove each reads correctly off the shared
substrate. Vikunja #171 organ-read rung.

WHY THIS RUNNER EXISTS (the gap on `main`, 2026-09-02). Production runs TWO separate merged pools:
  pool #1 (`onebrain_merge_production.MergedSubstrate`)  = D2 SURPRISE + E2 WORLD-MODEL  (hebbian ON, param-het OFF)
  pool #2 (`onebrain_merge_production2.MergedSubstrate2`) = E1 METACOG + D PRAGMATIC     (frozen, param-het ON)
The declarative `onebrain_merge_framework` verifies each pool's PAIR co-residence-invariant SEPARATELY
(`--keys surprise,worldmodel` and `--keys metacog,pragmatic`) -- but NEVER all four on ONE bridge, because their
GLOBAL configs conflict (`enable_hebbian_learning` T vs F, `enable_parameter_heterogeneity` F vs T,
`hebbian_max_weight` 45 vs 400). Substrate-init never exercised that conflict (no training at init); the
ORGAN-READ does (surprise's `judge` trains Hebbian on the shared `cp_connections`). The prior all-4 result
(`2026-08-27-onebrain-twopool-organ-read-byte-identity-6seed-GO.md`) was landed doc-only -- its bespoke runner is
NOT on `main` and cannot reproduce. This runner restores a RUNNABLE all-4 organ-read verify, built on the
declarative `merge_organs` engine (reuse, not a bespoke pool), so the rung is reproducible.

THE RECONCILIATION (matches the proven bespoke method, now declarative). The 3 config conflicts resolve to ONE
mutually-consistent superset config:
  * enable_hebbian_learning -> True GLOBALLY (pool #1 needs it) + a per-synapse `cp_plasticity_rate_gain = 0.0`
    freeze on every pool-#2 (metacog/pragmatic) INTERNAL edge (the framework's `freeze_regions`), so pool-1's
    Hebbian training can NEVER corrupt a pool-2 weight (the primitive `2026-08-14-onebrain-parser-on-pool-GO.md`
    + the deleted twopool runner used).
  * enable_parameter_heterogeneity -> reconciled the twopool way: global OFF + the name-keyed per-region MASK on
    metacog/pragmatic's regions (`param_het=True`), byte-identical to their global-het standalone (proven by
    `2026-08-27-onebrain-twopool-merge-substrate-byte-identity-6seed-GO.md`).
  * hebbian_max_weight -> pool #1's 45 GLOBALLY (pool-2 edges frozen => never clipped).

THE GATE (all 6 seeds 42/43/44/100/101/102, numpy CPU, bit-exact):
  (a) ORGAN-READ byte-identity  — each organ's read on the merged pool == its read CO-RESIDENT-ALONE on the
      merged superset config (co-residence invariance; the framework's own migration-safety bar). Also measured
      vs the SHIPPED 2 production pools (migration fidelity) and reported honestly.
  (b) FACULTY-ALIVE             — each organ still produces its live verdict on the merged pool (surprise fires
      harder on a contradiction than a confirmation; world-model predicts opposite valence signs for +/- context
      and fires more on a violated than an expected turn; metacog's confidence margin grows with evidence;
      pragmatic's implicature margin separates the scalar family).
  (c) ANSWER-PRESERVATION       — the rendered chat answer of each organ on the merged pool == its answer on the
      current 2-separate-pools production handler.
  + GAIN-0 FREEZE HOLDS  — pool-2 internal edge weights are byte-identical before vs after the full train+read
    lifecycle (surprise trained Hebbian on the shared bridge, pool-2 stayed frozen).
  + LEGACY DISCRIMINATOR — the seams-OFF pool diverges merged-vs-coresident (the byte-identity is NOT vacuous).

HONEST SCOPE. Byte-identity-in-ISOLATION is the MIGRATION-SAFETY gate (co-locating the 4 organs didn't change a
read), NOT the one-brain INTEGRATION goal (organs INTERACTING via cross-region synapses) -- a pool with zero
cross-edges is MIGRATED, not INTEGRATED. This rung earns the safe single-pool migration; the production
single-`BRAIN_ONEBRAIN_MERGE`-pool flip + the cross-region F-gate are the named next rungs. NO `sim/` edit; the
only framework change is 3 ADDITIVE MergedPool accessors (`meta_surprise`/`meta_worldmodel`/`worldmodel_idx_map`)
so the UNMODIFIED shipped surprise/world-model organs run against the pool (byte-identical when unused).

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_twopool_merge_organread_verify \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_twopool_merge_organread_6seed.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    SURPRISE, WORLDMODEL, METACOG, PRAGMATIC, merge_organs, _host, _idx,
)
from research.runners.onebrain_merge_framework import (
    _metacog_reads, _pragmatic_reads, _metacog_answer, _pragmatic_answer,
)
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from research.runners.worldmodel_production_organ import WorldModelProductionOrgan
from research.runners.metacog_production_organ import MetacogProductionOrgan
from research.runners.pragmatic_production_organ import PragmaticProductionOrgan
from research.runners.onebrain_merge_production import MergedSubstrate
from research.runners.onebrain_merge_production2 import MergedSubstrate2
from research.runners._onebrain_merge_rung1_verify import _surprise_reads, _worldmodel_reads

_POOL2_FREEZE = ("workspace", "workspace_fs", "meta_schema", "item", "item_fs")
_SURPRISE_BATTERY = [("alpha", "alpha"), ("beta", "gamma"), ("delta", "omega"), ("kappa", "kappa")]


# The pool-1 (hebbian) organs need HOMEOSTASIS ON (their standalone runs the CoreSimConfig default
# enable_homeostasis=True); the pool-2 (frozen) organs set enable_homeostasis=False GLOBALLY. Merged, pool-2's
# global False would silence pool-1 (world-model went completely dead: pred_pos=0, state_neg mis-selected).
# Reconcile the twopool way (`2026-08-27-onebrain-twopool-merge-substrate-byte-identity`'s homeostasis row):
# global OFF + per-region `enable_homeostasis=True` on EVERY surprise/world-model region (the diffbuilder mask),
# so pool-1 keeps its homeostasis and pool-2 stays frozen -- co-residence-invariant (each slice's flag is set the
# same alone or merged).
_SURPRISE_REGIONS = ("cue", "patient_expected", "patient_asserted", "surprise")
_WORLDMODEL_REGIONS = ("state", "pred_pos", "pred_neg", "obs_pos", "obs_neg", "surprise_pos", "surprise_neg")
_HOMEO_ON = {"enable_homeostasis": True}


def _recon_descriptors():
    """The 4 core organs as ONE mutually-consistent merged family (see the module docstring's reconciliation)."""
    surprise_r = replace(SURPRISE, organ_cls=SurpriseProductionOrgan, read_fn=_surprise_reads, supports_shared=True,
                         region_flags={r: dict(_HOMEO_ON) for r in _SURPRISE_REGIONS})
    worldmodel_r = replace(WORLDMODEL, organ_cls=WorldModelProductionOrgan, read_fn=_worldmodel_reads,
                           supports_shared=True, region_flags={r: dict(_HOMEO_ON) for r in _WORLDMODEL_REGIONS})
    mc = dict(METACOG.config)
    mc["enable_hebbian_learning"] = True                      # match pool-1 global; pool-2 edges gain-0 frozen
    for k in ("enable_parameter_heterogeneity", "per_region_parameter_heterogeneity", "hebbian_max_weight"):
        mc.pop(k, None)                                       # param_het=True path sets het; 45 (pool-1) wins
    metacog_r = replace(METACOG, config=mc, param_het=True,
                        freeze_regions=("workspace", "workspace_fs", "meta_schema"))
    pg = dict(PRAGMATIC.config)
    pg["enable_hebbian_learning"] = True
    for k in ("enable_parameter_heterogeneity", "per_region_parameter_heterogeneity", "hebbian_max_weight"):
        pg.pop(k, None)
    pragmatic_r = replace(PRAGMATIC, config=pg, param_het=True, freeze_regions=("item", "item_fs"))
    return [surprise_r, worldmodel_r, metacog_r, pragmatic_r]


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


def _pool2_edge_weights(bridge):
    """The weights of every pool-2 INTERNAL edge (both endpoints in a metacog/pragmatic region) — the array the
    gain-0 freeze must hold byte-identical across the full train+read lifecycle."""
    idx = set()
    for name in _POOL2_FREEZE:
        idx |= set(int(i) for i in _idx(bridge, name))
    arr = np.asarray(sorted(idx), dtype=np.int64)
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    both = np.isin(row, arr) & np.isin(col, arr)
    order = np.lexsort((col[both], row[both]))
    return data[both][order].astype(np.float64)


def _surprise_answer(organ):
    return tuple(bool(organ.judge("agent", "acts", ps, pa)["surprised"]) for ps, pa in _SURPRISE_BATTERY)


def _worldmodel_answer(organ):
    return (int(organ.expectation(+1)["pred_sign"]), int(organ.expectation(-1)["pred_sign"]))


_READ_FNS = {"surprise": (_surprise_reads, _surprise_answer), "worldmodel": (_worldmodel_reads, _worldmodel_answer),
             "metacog": (_metacog_reads, _metacog_answer), "pragmatic": (_pragmatic_reads, _pragmatic_answer)}


# Transient dynamical state a co-resident organ's CALIBRATION (fired at construction) or a prior organ's READ
# leaves on the SHARED bridge and that the organ's own per-CALL read_isolation (per-neuron only) + `_hard_reset`
# (v/u only) do NOT wash: the conductance rise/slow buffers (captured by _snapshot_state's _STATE_ARRAYS) + the
# homeostatic accumulators + the synapse pulse timers. World-model (a long-integration read under homeostasis)
# is the sensitive one — constructing metacog/pragmatic drifts its read ~0.7 Hz (bisected 2026-09-02).
_EXTRA_STATE = ("cp_neuron_firing_thresholds", "cp_neuron_activity_ema",
                "cp_synapse_pulse_timers", "cp_synapse_pulse_progress")


def _snap_dyn(bridge, xp):
    """Snapshot the pool's post-build PRISTINE dynamical state (all conductance/rise/slow buffers + homeostatic +
    pulse timers). NOT the connection WEIGHTS — those are trained at organ construction and MUST persist; the
    restore washes only the transient residue, so all 4 organs genuinely coexist on ONE trained substrate."""
    from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state
    dyn = _snapshot_state(bridge, xp)
    extra = {nm: getattr(bridge, nm).copy() for nm in _EXTRA_STATE if getattr(bridge, nm, None) is not None}
    return dyn, extra


def _restore_dyn(bridge, snap):
    from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state
    dyn, extra = snap
    _restore_state(bridge, dyn)
    for nm, v in extra.items():
        getattr(bridge, nm)[:] = v


def _isolated_reads(pool, descs, seed):
    """PER-ORGAN READ ISOLATION (Vikunja #171: one organ's read must not leak into another's). Snapshot the pool's
    post-build pristine dynamical state, construct EVERY organ (weights train on the shared bridge and persist),
    then restore the pristine state before EACH organ's read AND answer battery — so every organ reads from the
    SAME clean substrate, exactly as its coresident-alone baseline does under the identical protocol. The
    full-snapshot-restore the deleted bespoke runner used (its 'full-snapshot-restore' read isolation)."""
    bridge, xp = pool.bridge, pool.xp
    pristine = _snap_dyn(bridge, xp)
    organs = {d.key: d.organ_cls(seed=seed, shared=pool) for d in descs}
    for o in organs.values():
        o.ensure_built()
    reads, answers = {}, {}
    for d in descs:
        rf, af = _READ_FNS[d.key]
        _restore_dyn(bridge, pristine); reads[d.key] = rf(organs[d.key])
        _restore_dyn(bridge, pristine); answers[d.key] = af(organs[d.key])
    return reads, answers, organs, pristine


def _isolated_read_one(pool, d, seed):
    """A SINGLE organ on its own pool, the SAME isolation protocol — the co-residence-invariance baseline."""
    bridge, xp = pool.bridge, pool.xp
    pristine = _snap_dyn(bridge, xp)
    org = d.organ_cls(seed=seed, shared=pool); org.ensure_built()
    rf, af = _READ_FNS[d.key]
    _restore_dyn(bridge, pristine); reads = rf(org)
    _restore_dyn(bridge, pristine); answer = af(org)
    return reads, answer


def _isolated_reads_shipped(pool, key_cls_list, seed):
    """The SAME per-organ isolation protocol on a SHIPPED production pool (MergedSubstrate / MergedSubstrate2),
    which exposes `.bridge`/`.xp` but is not a framework MergedPool — so gate (c) compares merged-vs-shipped under
    the identical read protocol (a clean config-only comparison, not confounded by the shipped pool's own weaker
    per-call read isolation)."""
    pool.ensure_built()
    bridge = pool.bridge
    xp = getattr(pool, "xp", None)
    if xp is None:
        from sim.backend import get_backend
        xp, _ = get_backend()
    pristine = _snap_dyn(bridge, xp)
    organs = {k: cls(seed=seed, shared=pool) for k, cls in key_cls_list}
    for o in organs.values():
        o.ensure_built()
    reads, answers = {}, {}
    for k, _cls in key_cls_list:
        rf, af = _READ_FNS[k]
        _restore_dyn(bridge, pristine); reads[k] = rf(organs[k])
        _restore_dyn(bridge, pristine); answers[k] = af(organs[k])
    return reads, answers


def _faculty_alive(reads, answers):
    """(b) each organ still produces a live, non-degenerate verdict on the merged pool."""
    s = reads["surprise"]
    surp_sep = s["calib.contradict_hz"] / max(s["calib.confirm_hz"], 1e-6)
    surprise = bool(surp_sep >= 2.0 and s["calib.contradict_hz"] >= 5.0)
    wpos, wneg = answers["worldmodel"]
    w = reads["worldmodel"]
    vio = w.get("surprise[ctx+1,obs-1].hz", 0.0); exp = w.get("surprise[ctx+1,obs+1].hz", 0.0)
    worldmodel = bool(wpos > 0 and wneg < 0 and vio > exp)
    m = reads["metacog"]
    metacog = bool(m["margin_2"] > m["margin_0"])                       # confidence grows with evidence
    p = reads["pragmatic"]
    pragmatic = bool(abs(p["some.margin"] - p["all.margin"]) > 1e-6)    # implicature separates the scalar family
    return {"surprise": surprise, "worldmodel": worldmodel, "metacog": metacog, "pragmatic": pragmatic,
            "surprise_sep": float(surp_sep)}


def verify_seed(seed: int, verbose: bool = True) -> dict:
    descs = _recon_descriptors()
    keys = [d.key for d in descs]

    # ── MERGED-4 (the literal two-pool merge, wire=True) — all 4 organs read with per-organ isolation ──
    merged = merge_organs(descs, seed, wire=True)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    pool2_w_before = _pool2_edge_weights(merged.bridge)
    R_merged, A_merged, _m_organs, _ = _isolated_reads(merged, descs, seed)
    pool2_w_after = _pool2_edge_weights(merged.bridge)
    gain0_ok = bool(pool2_w_before.shape == pool2_w_after.shape
                    and float(np.max(np.abs(pool2_w_before - pool2_w_after))) == 0.0)
    freeze_delta = float(np.max(np.abs(pool2_w_before - pool2_w_after))) if pool2_w_before.shape == pool2_w_after.shape else float("inf")

    # ── (a) CO-RESIDENT-alone-on-superset (co-residence invariance), SAME isolation protocol ──
    coresident = {}
    for d in descs:
        core = merge_organs([d], seed, config_descriptors=descs, wire=True)
        c_reads, c_answer = _isolated_read_one(core, d, seed)
        dd, wk, miss = _maxdelta(R_merged[d.key], c_reads)
        coresident[d.key] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                             "byte_identical": bool(dd == 0.0 and not miss),
                             "answer_same": bool(A_merged[d.key] == c_answer)}

    # ── (c) SHIPPED 2-pool (migration fidelity + answer preservation), SAME isolation protocol ──
    shipP1 = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    shipP2 = MergedSubstrate2(seed=seed, organs=("metacog", "pragmatic"))
    Rs1, As1 = _isolated_reads_shipped(shipP1, [("surprise", SurpriseProductionOrgan),
                                               ("worldmodel", WorldModelProductionOrgan)], seed)
    Rs2, As2 = _isolated_reads_shipped(shipP2, [("metacog", MetacogProductionOrgan),
                                               ("pragmatic", PragmaticProductionOrgan)], seed)
    R_ship = {**Rs1, **Rs2}; A_ship = {**As1, **As2}
    shipped = {}
    for k in keys:
        dd, wk, miss = _maxdelta(R_merged[k], R_ship[k])
        shipped[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                      "read_byte_identical": bool(dd == 0.0 and not miss),
                      "answer_same": bool(A_merged[k] == A_ship[k])}

    # ── LEGACY DISCRIMINATOR (seams OFF -> merged-vs-coresident init diverges) ──
    from research.runners.onebrain_merge_framework import substrate_byte_identity
    leg_merged = merge_organs(descs, seed, legacy=True)
    legacy_delta = 0.0
    for d in descs:
        regions = leg_merged.organ_regions.get(d.key) or list(d.regions)
        leg_core = merge_organs([d], seed, config_descriptors=descs, legacy=True)
        lbi = substrate_byte_identity(leg_merged, leg_core, regions)
        legacy_delta = max(legacy_delta, lbi["maxerr"])

    alive = _faculty_alive(R_merged, A_merged)

    # ── per-seed GO ──
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
           "n_pool2_frozen_edges": int(pool2_w_before.shape[0]),
           "legacy_diverges": legacy_ok, "legacy_delta": legacy_delta, "GO": go}
    if verbose:
        print(f"  [seed {seed}] N={n_all} | (a)cores_byteid={a_ok} (b)alive={b_ok} (c)answer={c_ok} "
              f"ship_read={ship_read_ok} gain0={gain0_ok}(n={int(pool2_w_before.shape[0])}) "
              f"legacy_div={legacy_ok}({legacy_delta:.0f}) -> GO={go}", flush=True)
        for k in keys:
            print(f"      {k:11s} cores_d={coresident[k]['maxdelta']:.2e} ship_d={shipped[k]['maxdelta']:.2e} "
                  f"ship_ans_same={shipped[k]['answer_same']} alive={alive[k]}", flush=True)
    return res


def verify(seeds, verbose: bool = True) -> dict:
    per_seed = [verify_seed(s, verbose=verbose) for s in seeds]
    n = len(seeds)
    keys = ["surprise", "worldmodel", "metacog", "pragmatic"]
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
    for k in keys:
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

    print("=== ONE-BRAIN TWO-POOL MERGE — ORGAN-READ rung: 4 core organs on ONE shared merged pool ===")
    print("    surprise + world-model (pool #1, hebbian) + metacog + pragmatic (pool #2, frozen) on ONE bridge")
    out = verify(seeds)
    ag = out["aggregate"]; n = ag["n_seeds"]
    print("\n=== VERDICT (organ-read rung) ===")
    for k in ["surprise", "worldmodel", "metacog", "pragmatic"]:
        po = out["per_organ"][k]
        print(f"  {k:11s} cores_byteid={po['n_coresidence_byte_identical']}/{n} "
              f"ship_read={po['n_shipped_read_byte_identical']}/{n} answer_same={po['n_answer_same']}/{n} "
              f"alive={po['n_alive']}/{n} (max cores_d={po['max_coresidence_delta']:.2e} ship_d={po['max_shipped_delta']:.2e})")
    print(f"\n  (a) organ-read byte-identity (co-residence invariance): {ag['n_gate_a']}/{n}")
    print(f"  (b) faculty-alive:                                      {ag['n_gate_b']}/{n}")
    print(f"  (c) answer-preservation vs shipped 2-pool:              {ag['n_gate_c']}/{n}")
    print(f"      shipped-read byte-identity (migration fidelity):    {ag['n_shipped_read_byte_identical']}/{n}")
    print(f"      gain-0 freeze holds pool-2 edges:                   {ag['n_gain0_freeze']}/{n}")
    print(f"      legacy discriminator diverges (non-vacuous):        {ag['n_legacy_diverges']}/{n}")
    print(f"  ORGAN-READ RUNG GO (a & b & c & gain0 & legacy): {ag['n_go']}/{n}  ->  ALL-GO={out['all_go']}")

    from tools.verdict import Verdict
    v = Verdict("one-brain two-pool merge organ-read (4 core organs on ONE shared merged pool, N=2034)")
    v.require("(a) organ-read byte-identity — every organ's read co-residence-invariant, every seed",
              ag["n_gate_a"], expect=n)
    v.require("(b) faculty-alive — every organ produces its live verdict on the merged pool, every seed",
              ag["n_gate_b"], expect=n)
    v.require("(c) answer-preservation — every organ's rendered answer == shipped 2-pool, every seed",
              ag["n_gate_c"], expect=n)
    v.require("gain-0 freeze holds pool-2 edges bit-frozen across the train+read lifecycle, every seed",
              ag["n_gain0_freeze"], expect=n)
    v.require("legacy discriminator diverges (byte-identity NOT vacuous), every seed",
              ag["n_legacy_diverges"], expect=n)
    v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
               why="MIGRATION gate: byte-identity-in-isolation forbids cross-synapses BY DEFINITION (DESIGN §4)")
    decided = v.decide(go=out["all_go"])

    payload = {"mode": "onebrain_twopool_merge_organread", **out}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
