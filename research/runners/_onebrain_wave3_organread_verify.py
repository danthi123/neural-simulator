"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 3 (the FINAL merge wave) — organ-read verify: extend the WAVE-2
9-organ pool (`onebrain_wave2_pool_production.get_wave2_pool`: surprise + world-model + metacog + pragmatic +
comprehension + source_provenance + self_schema + curiosity + causal_whatif) with d6_multiref_wm +
prospective_memory onto ONE shared `merge_organs` pool.
(docs/plans/2026-09-02-onebrain-integration-program.md, Phase 3: "Wave 3 (post-Phase-2 scale): merge
d6-multiref-WM + prospective-memory (largest, 1720 neurons; pushes N->~4968) -> completes the d6->comprehension
template on the single pool.")

ALREADY-BUILT CHECK (done before writing a line of code): `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` has no
"wave3"/11-organ single-pool reference (only the UNRELATED earlier "wave-3 default-on FLIP" of 4 already-shipped
faculties -- spiking-mouth-recall/gnw-global-stop/activity-silent-wm/bg-action-selection -- a different naming
collision, not this merge). `research/findings/2026-09*.md` names only Wave 1 (comprehension + source_provenance)
and Wave 2 (self_schema + curiosity + causal_whatif) as landed. The RAG
(`tools/rag/rag_search.py "d6 multiref wm prospective memory single pool merge wave 3"`) surfaces only the
standalone D6/pmem de-risks + the 2026-08-27 dedup-synapse-masks finding (a DIFFERENT, frozen-hebbian-family
batch that never co-resided d6/pmem with metacog/self_schema/curiosity under a LIVE-hebbian pool). `git branch -a`
/ `git log --all --oneline | grep wave3` show only `wave3-flip*` branches (the unrelated flip). Confirmed NOT
already built.

WAVE-3-SPECIFIC SEAMS FOUND BY VERIFYING (not assuming) each new organ's region/wiring-key footprint + weight
scale against the wave-2 9-organ superset:
  (a) REGION-NAME COLLISIONS: NONE. d6_multiref_wm's regions are DISCOVERED at build (`w0..w29` + `fs`, from
      `_D6_REGION_FLAGS`'s own key list -- the 5 banks * 6 slots `_spec_d6_multiref_wm(seed) -> MultiSlotHold(seed,
      5, 6)` builds); prospective_memory's are `cortex_ctx`/`dlpfc_wm`/`rel_A`/`rel_B`. Enumerated against every
      wave-2 organ's region set (surprise: cue/patient_expected/patient_asserted/surprise; worldmodel: state/
      pred_pos/pred_neg/obs_pos/obs_neg/surprise_pos/surprise_neg; metacog: workspace/workspace_fs/meta_schema;
      pragmatic: item/item_fs; comprehension: sel_agent/sel_FS_agent/sel_patient/sel_FS_patient/cue_position_pos/
      cue_position_neg/cue_animacy_pos/cue_animacy_neg/cue_verbfit_pos/cue_verbfit_neg/cue_lexbias_pos/
      cue_lexbias_neg; source_provenance: episode/content_readout/ctx_perceived/ctx_generated/prov_perceived/
      prov_generated/inh_perceived/inh_generated; self_schema (wave2-renamed): ss_workspace/ss_workspace_fs/
      self_schema; curiosity (wave2-renamed): cur_cue/striosome_value/reward_us/snc/ask; causal_whatif: evt) --
      zero overlap either direction. NO rename wrapper needed (unlike Wave 2's self_schema/curiosity) -- both new
      organs' REGISTRY descriptors are reused UNCHANGED on `regions`/`spec_fn`/`idx_fn`.
  (b) WIRING-KEY COLLISIONS: NONE. prospective_memory's `explicit_wiring_fn` (`_pmem_wiring`) emits `c2d`/`d2c`/
      `cue_monitor` -- disjoint from metacog's `loop_0`/`loop_1`, self_schema's wave-2 `ss_loop_k`/
      `ss_member{k}_to_attend`, and causal_whatif's `xblock`. d6_multiref_wm declares NO `explicit_wiring_fn` at
      all (its recurrent bump-attractor connectivity is expressed as ordinary `RegionPathway` entries returned
      by `_spec_d6_multiref_wm`'s own `cfg.region_pathways`, consumed generically by the base
      `rm.build_wiring_plan` union) -- it cannot collide with any organ's wiring-key namespace by construction.
  (c) HEBBIAN POP+FREEZE (the standard Wave-1/Wave-2 seam, reused not re-derived): both new organs declare
      `enable_hebbian_learning: False` in their own REGISTRY config (`_D6_CONFIG` / `_PMEM_CONFIG`), conflicting
      with the pool's global `enable_hebbian_learning=True` (surprise/world-model's live Hebbian read needs it) --
      POPPED (pool's True wins) + every one of BOTH organs' OWN regions gain-0 FROZEN. NEITHER organ needs a
      causal_whatif-style freeze EXCLUSION: d6 is a pure frozen bump-attractor forward pass (no build-time weight
      training -- its `MultiSlotHold` never runs a plasticity rule, only current-driven write/hold/read); pmem's
      `SFANmdaProspectiveMemory` construction re-homes ONLY per-neuron state (the homeostat tonic bias + the
      NMDA-plateau calibration) onto the pmem slice -- NO synapse weight is ever trained at construction (the
      attractor/cue-monitor/rel-recurrent edges are FIXED `initial_weights` installed once by `_pmem_wiring`, never
      touched by any rule). Both organs' full region sets stay IN the before/after gain-0 array check below (no
      "evt"-style exclusion, matching self_schema/curiosity's Wave-2 precedent, not causal_whatif's).
  (d) THE "45 CEILING vs ATTRACTOR DESIGN-WEIGHT" CONCERN (task-flagged): VERIFIED, not assumed. d6's installed
      recurrent weight is `MultiSlotHold`'s own default `recur=25.0` (`_spec_d6_multiref_wm` calls
      `MultiSlotHold(int(seed), 5, 6)`, no override) -- well under the pool's `hebbian_max_weight=45.0` ceiling,
      same as Wave 2's organs. prospective_memory is the genuine NEW case: its POOL-GAINED attractor weight is
      `_PMEM_ATTRACTOR_W = 50.0 * _PMEM_POOL_GAIN(6.0) = 300.0` (plus `_PMEM_HOLD_W=19.2` / `_PMEM_CUE_W=25.2`) --
      300 is ~7x the 45 ceiling, the largest design-weight-vs-ceiling gap any wave has hit. Checked directly in
      `sim/bridge.py`: `inject_explicit_wiring` (~L4219) installs `initial_weights` VERBATIM into `cp_connections.
      data` with NO clip at install time; the ONLY clip site that could touch them (~L10014/10025) is applied
      EXCLUSIVELY to synapses with `cp_plasticity_rate_gain > 0` (the 2026-07-31 fix the Wave-2 finding already
      documented) -- and the merge engine's own build order (`onebrain_merge_framework.MergedPool.ensure_built`
      step 7, "GAIN-0 FREEZE") applies pmem's freeze IMMEDIATELY after wiring install/post_build and BEFORE any
      simulation step ever runs (step 8/9's settle-to-rest snapshot happens strictly AFTER). `_pmem_wiring`'s own
      `mk()` helper additionally marks every one of its edges `"plastic": False` at the wiring-plan level -- a
      SECOND, independent reason the clip never applies. So pmem's 300-weight attractor survives UNCLIPPED at 300,
      not silently truncated to 45 -- confirmed empirically below by the gain-0 before/after check (a truncation
      would show as a nonzero freeze-delta, and would also collapse pmem's `_faculty_alive`/answer-preservation
      gates, since a clipped-to-45 attractor could not self-sustain the multi-turn hold at all).

THE GATE (numpy, bit-exact) -- the SAME 3-part rung Wave 1/2 used, extended to 11 organs:
  (a) ORGAN-READ byte-identity -- each of the 11 organs' read on the wave3 pool == its read CO-RESIDENT-ALONE on
      the wave3 superset config (co-residence invariance).
  (b) FACULTY-ALIVE -- each organ still produces its live, non-degenerate verdict on the wave3 pool.
  (c) ANSWER-PRESERVATION -- the 9 wave-2-carried organs' rendered answer == the ACTUAL SHIPPED
      `get_wave2_pool(seed)`'s answer (strict, apples-to-apples: both run the merge engine's seams);
      d6_multiref_wm/prospective_memory's answer == their TODAY'S STANDALONE construction (shared=None). The
      categorical ANSWER is gated for every organ (matching Wave 1/2's own precedent); the CONTINUOUS margin for
      the 2 new organs is reported but not gated (an honest declared residual, same reasoning Wave 1/2 gave for
      their own new organs -- a spiking-dynamics read integrated over hundreds of steps amplifies a single-ULP
      per-step delta into a 1-spike divergence vs an UNSEAMED standalone build).
  + GAIN-0 FREEZE HOLDS -- d6_multiref_wm/prospective_memory's internal edge weights are byte-identical before vs
    after the full train+read lifecycle (both organs' FULL region sets are checked -- neither needs an "evt"-style
    exclusion, see seam (c) above).
  + LEGACY DISCRIMINATOR -- the seams-OFF pool diverges merged-vs-coresident (byte-identity is NOT vacuous).

HONEST SCOPE. MIGRATION-SAFETY organ-read rung only (byte-identity-in-isolation), NOT the one-brain INTEGRATION
goal -- zero cross-region synapses are added here (a pool with no cross-edges is MIGRATED, not INTEGRATED). NOT
wired into any live `get_organ()` dispatch: `onebrain_wave3_pool_production.py`'s flag is additive/default-OFF
and touches NO existing production file (verify with `git diff`). prospective_memory's "shipped" comparison is
against the FRAMEWORK's own `_PMemReadOrgan(seed, shared=None)` read organ, NOT the separate production wrapper
`prospective_memory_production_organ.py` (which has no `shared=` support) -- the same honest declared residual
Wave 1 named for source_provenance's `_SourceProvReadOrgan`.

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_wave3_organread_verify \
        --seeds 42,43,44 \
        --out research/findings/raw/_onebrain_wave3/organread_3seed_smoke.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    merge_organs, REGISTRY, substrate_byte_identity,
    _d6_reads, _d6_answer,
    _pmem_reads, _pmem_answer, _PMemReadOrgan,
    _D6_REGION_FLAGS,
)
from research.runners._onebrain_wave2_organread_verify import (
    _wave2_descriptors, WAVE2_KEYS, _frozen_edge_weights as _frozen_edge_weights_wave2,
)
from research.runners._onebrain_wave1_organread_verify import (
    _isolated_reads, _isolated_read_one, _maxdelta,
)
from research.runners.onebrain_wave2_pool_production import get_wave2_pool
from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan

WAVE2_CARRIED_KEYS = tuple(WAVE2_KEYS)
NEW_KEYS = ("prospective_memory", "d6_multiref_wm")
WAVE3_KEYS = list(WAVE2_CARRIED_KEYS) + list(NEW_KEYS)

# d6's regions are discovered at build (`regions=()` on its REGISTRY descriptor -- see the module docstring's
# seam (a)); `_D6_REGION_FLAGS`'s own key list (w0..w29 + fs, the framework's own NMDA-mask seam construct) IS
# the exact 31-name region set every d6 build produces, reused-by-import rather than re-derived.
_D6_REGIONS = tuple(_D6_REGION_FLAGS.keys())
_PMEM_REGIONS = ("cortex_ctx", "dlpfc_wm", "rel_A", "rel_B")


def _wave3_descriptors():
    """The 11-organ Wave-3 family: the EXISTING 9-organ wave-2 reconciliation UNCHANGED (`_wave2_descriptors`,
    reuse-by-import -- zero drift from the shipped wave-2 pool) + prospective_memory + d6_multiref_wm, reconciled
    the SAME "pop enable_hebbian_learning + freeze own regions" seam metacog/pragmatic/comprehension/
    source_provenance/self_schema/curiosity/causal_whatif already use (see the module docstring seam (c)). NO
    rename wrapper needed for either new organ (seam (a): zero region-name collisions)."""
    base = _wave2_descriptors()

    # SEAM (e), found empirically by actually building the pool (NOT predicted by the plan doc -- this is the
    # FIRST wave d6_multiref_wm co-resides with metacog, so it is the first time this key pair could collide):
    # metacog's config declares `nmda_recurrent_tau_decay_ms=150.0` (`_POOL2_METACOG_CONFIG`, = `DEFAULT_NMDA_TAU`
    # -- a VESTIGIAL re-statement of the same constant its own `nmda_tau_decay` key uses). metacog never sets
    # `enable_nmda_recurrent` (absent from its config; sim/config.py's own engine default is False), so this value
    # is functionally INERT for metacog -- the slow-NMDA-recurrent pathway never turns on for its regions either
    # way. d6_multiref_wm is the first organ in this program to set `enable_nmda_recurrent=True`: its whole
    # slow-NMDA bump-attractor HOLD mechanism depends on `nmda_recurrent_tau_decay_ms=100.0` (which also happens
    # to equal the ENGINE's own default, sim/config.py:402) -- the value that must win the union. Popped from
    # metacog's LOCAL descriptor copy ONLY (this function's own `base` list, NOT the shared REGISTRY row Wave
    # 1/2's shipped pools still use elsewhere) -- a config key with zero functional effect on metacog either way,
    # so this cannot perturb metacog's already-verified Wave 1/2 read.
    base = [replace(d, config={k: v for k, v in d.config.items() if k != "nmda_recurrent_tau_decay_ms"})
            if d.key == "metacog" else d for d in base]

    pmem = REGISTRY["prospective_memory"]
    pmem_cfg = dict(pmem.config)
    pmem_cfg.pop("enable_hebbian_learning", None)
    pmem_r = replace(pmem, config=pmem_cfg, freeze_regions=_PMEM_REGIONS)

    d6 = REGISTRY["d6_multiref_wm"]
    d6_cfg = dict(d6.config)
    d6_cfg.pop("enable_hebbian_learning", None)
    d6_r = replace(d6, config=d6_cfg, freeze_regions=_D6_REGIONS)

    return base + [pmem_r, d6_r]


# Both new organs' FULL region sets stay in this check (neither needs an "evt"-style exclusion -- see seam (c)).
_WAVE3_FREEZE = _PMEM_REGIONS + _D6_REGIONS


def _frozen_edge_weights(bridge, frozen_regions):
    """Reuse Wave 2's already-generalized helper (arbitrary region list -> the array the gain-0 freeze must hold
    byte-identical across the full train+read lifecycle) rather than re-deriving it a third time."""
    return _frozen_edge_weights_wave2(bridge, frozen_regions)


def _faculty_alive(reads, answers):
    """(b) each organ still produces a live, non-degenerate verdict on the wave3 pool."""
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
    ss = reads["self_schema"]
    self_schema = bool(ss["author_rate_self"] > ss["author_rate_heard"])
    cur = reads["curiosity"]
    curiosity = bool(cur["want_novel_hz"] > cur["want_familiar_hz"])
    ca = reads["causal_whatif"]
    causal_whatif = bool(ca["fwd_acc"] >= 0.5 and ca["directed_fwd_BtoD"] > ca["directed_rev_DtoB"])
    pm = reads["prospective_memory"]
    prospective_memory = bool(pm["fire_A_on_cueA"] >= max(2.0 * pm["same_pool_silent"], 0.03))
    d6 = reads["d6_multiref_wm"]
    d6_multiref_wm = bool(d6["all_recovered"] >= 1.0 and d6["hold_alive_min"] > 0.0)
    return {"surprise": surprise, "worldmodel": worldmodel, "metacog": metacog, "pragmatic": pragmatic,
            "comprehension": comprehension, "source_provenance": source_provenance,
            "self_schema": self_schema, "curiosity": curiosity, "causal_whatif": causal_whatif,
            "prospective_memory": prospective_memory, "d6_multiref_wm": d6_multiref_wm,
            "surprise_sep": float(surp_sep)}


def verify_seed(seed: int, verbose: bool = True) -> dict:
    descs = _wave3_descriptors()
    keys = [d.key for d in descs]
    assert keys == WAVE3_KEYS, f"unexpected descriptor key order {keys}"

    # ── MERGED-11 (the literal wave-3 pool, wire=True) — all 11 organs read with per-organ isolation ──
    merged = merge_organs(descs, seed, wire=True)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    frozen_before = _frozen_edge_weights(merged.bridge, _WAVE3_FREEZE)
    R_merged, A_merged, _organs = _isolated_reads(merged, descs, seed)
    frozen_after = _frozen_edge_weights(merged.bridge, _WAVE3_FREEZE)
    gain0_ok = bool(frozen_before.shape == frozen_after.shape
                    and float(np.max(np.abs(frozen_before - frozen_after))) == 0.0)
    freeze_delta = (float(np.max(np.abs(frozen_before - frozen_after)))
                    if frozen_before.shape == frozen_after.shape else float("inf"))
    # explicit "no silent 300->45 truncation" witness (seam (d)): the max frozen edge weight actually present.
    max_frozen_weight = float(np.max(np.abs(frozen_after))) if frozen_after.size else 0.0

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
    # STRICT (gates the verdict): the 9 wave-2-carried organs vs the ACTUAL PRODUCTION wave-2 pool
    # (`get_wave2_pool`) — a fair apples-to-apples comparison, since that pool ALSO runs the merge engine's
    # FP-determinism seams. Extending it with 2 more organs must not perturb the 9 it originally carried.
    wave2_descs = _wave2_descriptors()
    shipped_wave2 = get_wave2_pool(seed)
    R_ship, A_ship = {}, {}
    Rb, Ab, _ = _isolated_reads(shipped_wave2, wave2_descs, seed)
    for d in wave2_descs:
        R_ship[d.key] = Rb[d.key]; A_ship[d.key] = Ab[d.key]

    # INFORMATIONAL for the CONTINUOUS margin only (ANSWER itself IS gated): prospective_memory/d6_multiref_wm vs
    # a RAW STANDALONE build (shared=None) that does NOT run the merge engine's seams at all — the same honest,
    # not strict, comparison Wave 1/2 made for their own new organs.
    pmem_standalone = _PMemReadOrgan(seed, shared=None)
    R_ship["prospective_memory"] = _pmem_reads(pmem_standalone)
    A_ship["prospective_memory"] = _pmem_answer(pmem_standalone)

    d6_standalone = MultiReferentWMOrgan(seed=seed, shared=None)
    R_ship["d6_multiref_wm"] = _d6_reads(d6_standalone)
    A_ship["d6_multiref_wm"] = _d6_answer(d6_standalone)

    shipped = {}
    for k in keys:
        dd, wk, miss = _maxdelta(R_merged[k], R_ship[k])
        shipped[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                      "read_byte_identical": bool(dd == 0.0 and not miss),
                      "answer_same": bool(A_merged[k] == A_ship[k]),
                      "strict": k in WAVE2_CARRIED_KEYS}

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
    ship_read_ok_strict = all(shipped[k]["read_byte_identical"] for k in WAVE2_CARRIED_KEYS)
    legacy_ok = bool(legacy_delta > 0.0)
    go = bool(a_ok and b_ok and c_ok and gain0_ok and legacy_ok)

    res = {"seed": seed, "n_all_neurons": n_all,
           "gate_a_coresidence_byte_identical": a_ok, "coresident": coresident,
           "gate_b_faculty_alive": b_ok, "faculty_alive": alive,
           "gate_c_answer_preserved": c_ok, "wave2_carried_read_byte_identical": ship_read_ok_strict,
           "shipped": shipped,
           "gain0_freeze_holds": gain0_ok, "gain0_freeze_delta": freeze_delta,
           "n_frozen_edges": int(frozen_before.shape[0]), "max_frozen_weight": max_frozen_weight,
           "legacy_diverges": legacy_ok, "legacy_delta": legacy_delta, "GO": go}
    if verbose:
        print(f"  [seed {seed}] N={n_all} | (a)cores_byteid={a_ok} (b)alive={b_ok} (c)answer={c_ok} "
              f"wave2_read={ship_read_ok_strict} gain0={gain0_ok}(n={int(frozen_before.shape[0])},"
              f"max_w={max_frozen_weight:.1f}) legacy_div={legacy_ok}({legacy_delta:.0f}) -> GO={go}", flush=True)
        for k in keys:
            print(f"      {k:20s} cores_d={coresident[k]['maxdelta']:.2e} ship_d={shipped[k]['maxdelta']:.2e} "
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
        "n_wave2_carried_read_byte_identical": sum(p["wave2_carried_read_byte_identical"] for p in per_seed),
        "n_gain0_freeze": sum(p["gain0_freeze_holds"] for p in per_seed),
        "n_legacy_diverges": sum(p["legacy_diverges"] for p in per_seed),
        "n_go": sum(p["GO"] for p in per_seed),
    }
    per_organ = {}
    for k in WAVE3_KEYS:
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

    print("=== ONE-BRAIN WAVE 3 (FINAL) — ORGAN-READ verify: d6_multiref_wm + prospective_memory on the WAVE-2 POOL ===")
    print("    surprise + world-model + metacog + pragmatic + comprehension + source_provenance + self_schema + "
         "curiosity + causal_whatif (the shipped wave-2 pool) + prospective_memory + d6_multiref_wm on ONE bridge")
    out = verify(seeds)
    ag = out["aggregate"]; n = ag["n_seeds"]
    print("\n=== VERDICT (Wave 3 organ-read rung) ===")
    for k in WAVE3_KEYS:
        po = out["per_organ"][k]
        print(f"  {k:20s} cores_byteid={po['n_coresidence_byte_identical']}/{n} "
              f"ship_read={po['n_shipped_read_byte_identical']}/{n} answer_same={po['n_answer_same']}/{n} "
              f"alive={po['n_alive']}/{n} (max cores_d={po['max_coresidence_delta']:.2e} ship_d={po['max_shipped_delta']:.2e})")
    print(f"\n  (a) organ-read byte-identity (co-residence invariance): {ag['n_gate_a']}/{n}")
    print(f"  (b) faculty-alive:                                      {ag['n_gate_b']}/{n}")
    print(f"  (c) answer-preservation vs shipped baselines:           {ag['n_gate_c']}/{n}")
    print(f"      wave2-carried read byte-identity (migration fidelity): {ag['n_wave2_carried_read_byte_identical']}/{n}")
    print(f"      gain-0 freeze holds frozen organs' edges:           {ag['n_gain0_freeze']}/{n}")
    print(f"      legacy discriminator diverges (non-vacuous):        {ag['n_legacy_diverges']}/{n}")
    print(f"  ORGAN-READ RUNG GO (a & b & c & gain0 & legacy): {ag['n_go']}/{n}  ->  ALL-GO={out['all_go']}")

    from tools.verdict import Verdict
    v = Verdict("one-brain Wave-3 (FINAL) organ-read (d6_multiref_wm + prospective_memory folded onto the "
               f"wave-2 pool, N~{out['per_seed'][0]['n_all_neurons']})")
    v.require("(a) organ-read byte-identity — every organ's read co-residence-invariant, every seed",
              ag["n_gate_a"], expect=n)
    v.require("(b) faculty-alive — every organ produces its live verdict on the wave3 pool, every seed",
              ag["n_gate_b"], expect=n)
    v.require("(c) answer-preservation — every organ's rendered answer == its shipped/standalone baseline, "
              "every seed", ag["n_gate_c"], expect=n)
    v.require("gain-0 freeze holds prospective_memory/d6_multiref_wm internal edges bit-frozen across the "
              "train+read lifecycle (incl. pmem's 300-weight attractor surviving unclipped vs the pool's 45 "
              "ceiling), every seed", ag["n_gain0_freeze"], expect=n)
    v.require("legacy discriminator diverges (byte-identity NOT vacuous), every seed",
              ag["n_legacy_diverges"], expect=n)
    v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
              why="MIGRATION gate: byte-identity-in-isolation forbids cross-synapses BY DEFINITION")
    v.disabled("live-chat production wiring (get_organ() dispatch)",
              why="deliberately deferred to a separate, later commit — mirrors the wave-1/wave-2 pools' own "
                  "sequencing (organ-read GO landed before its production wiring commit)")
    decided = v.decide(go=out["all_go"])

    payload = {"mode": "onebrain_wave3_organread", **out}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
