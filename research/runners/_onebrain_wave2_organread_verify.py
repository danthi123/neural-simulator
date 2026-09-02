"""ONE-BRAIN INTEGRATION PROGRAM, PHASE 3 WAVE 2 — organ-read verify: extend the WAVE-1 6-organ pool
(`onebrain_wave1_pool_production.get_wave1_pool`: surprise + world-model + metacog + pragmatic + comprehension +
source_provenance) with self_schema + curiosity + causal_whatif onto ONE shared `merge_organs` pool.
(docs/plans/2026-09-02-onebrain-integration-program.md, Phase 3: "Wave 2 (param-het wrinkles): merge self_schema +
curiosity + causal_whatif — resolve the self_schema/metacog workspace NAME COLLISION + the 400>45 attractor-weight
survival; lands the GNW workspace on the pool.")

ALREADY-BUILT CHECK (done before writing a line of code): `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` has no
"wave2"/9-organ single-pool reference; `research/findings/2026-09*.md` names only Wave 1 (comprehension +
source_provenance) as landed; the RAG (`tools/rag/rag_search.py "self_schema curiosity causal_whatif single pool
merge wave 2"`) surfaces only the 2026-08-27 GROUP_A batch (a DIFFERENT, hebbian-OFF frozen family that never
co-resided self_schema with metacog) and the 2026-09-02 Wave-1 finding, which names Wave 2 as its own explicit
"Next steps" #3. `git branch -a` / `git log --all --oneline | grep wave2` show no prior wave2 branch/commit.
Confirmed NOT already built.

THREE GENUINE SEAMS FOUND BY VERIFYING (not assuming) each organ's region/wiring-key footprint against the wave-1
superset — TWO match the plan doc's own prediction category (a name collision), ONE is a NEW finding this rung
made that the plan doc did not name:
  (1) self_schema/metacog REGION-NAME COLLISION (the plan doc's own prediction, confirmed): self_schema's
      `workspace`/`workspace_fs` are LITERALLY the same region names metacog already owns in the wave-1 pool
      (`METACOG.regions == ("workspace", "workspace_fs", "meta_schema")`;
      `REGISTRY["self_schema"].regions == ("workspace", "workspace_fs", "self_schema")`). The engine's
      spec-extraction `owner` dict raises `MergeConflict` on ANY duplicate region name across descriptors — the
      framework's own error message ("rename forbidden") warns against silently SHARING an identity between two
      DIFFERENT regions, not against renaming a descriptor's OWN region before registration (the correct fix).
      FIX: self_schema's Wave-2 descriptor renames ONLY the two colliding regions (`ss_workspace`/
      `ss_workspace_fs`; `self_schema` itself is untouched — no collision) via a generic `_renamed_spec` wrapper
      that also rewrites every `RegionPathway` endpoint referencing the old names (both of self_schema's own 2
      pathways: workspace<->workspace_fs).
  (2) curiosity/surprise REGION-NAME COLLISION (NOT named in the plan doc — found only by actually enumerating
      every Wave-1 organ's region-name footprint before assuming the plan doc's single named seam was the whole
      story): curiosity's `cue` region (`REGISTRY["curiosity"].regions[0] == "cue"`) is LITERALLY the same name
      SURPRISE already owns (`SURPRISE.regions == ("cue", "patient_expected", "patient_asserted", "surprise")`) —
      a second, undocumented collision the plan doc's "workspace" framing did not anticipate. Same fix, scoped to
      curiosity's OWN `cue` only (renamed `cur_cue`); surprise's `cue` is UNTOUCHED, so the wave-1 6 organs' own
      reads/wiring keep referencing the ORIGINAL name — zero risk to anything already shipped.
  (3) A THIRD, SILENT (non-`MergeConflict`-raising) collision found only by checking the WIRING-KEY namespace, not
      just region names: self_schema's `explicit_wiring_fn` and metacog's `explicit_wiring_fn` BOTH emit dict keys
      `loop_0`/`loop_1` into the SAME `_install_organ_read_wiring` union (self_schema's K_CONTENTS=4 range(4) vs
      metacog's K_CLASSES=2 range(2) overlap at k=0,1) — `dict.update()` does not raise, it SILENTLY OVERWRITES
      whichever descriptor's `explicit_wiring_fn` ran later in list order, corrupting the OTHER organ's attractor
      loop wiring with NO error (exactly the seam taxonomy's warning: "a MergeConflict is NOT raised; the union
      accepts a default and the faculty dies quietly"). FIX: self_schema's Wave-2 wiring keys are prefixed
      (`ss_loop_{k}`/`ss_member{k}_to_attend`), disjoint from metacog's `loop_{k}` by construction.

THE "400>45" ATTRACTOR-WEIGHT CONCERN (plan doc): VERIFIED, not assumed, to be a NON-ISSUE for this specific
organ family. self_schema's actual installed weights are LOOP_W=30.0 (`DEFAULT_ATTRACTOR_WEIGHT`, the SAME
constant metacog's own loop already uses inside the wave-1 pool) and MEMBER_TO_ATTEND_W=12.0 — both already well
under the pool's `hebbian_max_weight=45.0` ceiling (from `_POOL1_CONFIG`, unchanged by Wave 2). causal_whatif's
build-time STDP+DA train caps its OWN xblock weights at a LOCAL `stdp_w_max=24.0` (restored after training) —
also under 45. The literal "400" in this codebase
(`_self_schema_region_derisk.build_self_schema_bridge`'s own `cfg.hebbian_max_weight = max(400.0,
attractor_weight*4.0)`, and the analogous 2026-08-27 pool-#2 `_POOL2_METACOG_CONFIG.stdp_w_max`) is a GENEROUS
STANDALONE safety margin (4x headroom above the 30-weight loop, floored at 400), not evidence any real synapse in
this family is installed near that value — no clip ever fires either way. Checked directly in `sim/bridge.py`: the
Hebbian clip (~L10013-10023) and the reward/homeostatic clips (~L10822-10834, ~L11157-11170) are ALL already
GATED by `cp_plasticity_rate_gain` (a 2026-07-31 fix, predates this rung) — a frozen (gain-0) region's weights are
excluded from every clip regardless of the pool's global ceiling, so the freeze below is sufficient protection
independent of the 45-vs-400 comparison. self_schema + causal_whatif's regions are frozen anyway (below),
belt-and-braces.

THE STANDARD "hebbian" SEAM (Wave 1's own pattern, reused not re-derived): self_schema
(`enable_hebbian_learning: False` in its own config) and causal_whatif (same) both conflict with the pool's
global `enable_hebbian_learning=True` (surprise/world-model's live read needs it) — POP the key (pool's True
wins) + gain-0 FREEZE every one of their own regions' internal edges, exactly the comprehension/source_provenance
pattern Wave 1 already proved. curiosity declares NO `config` at all (no conflicting key to pop), but its own
regions are frozen too anyway (defensive hygiene — the read only ever touches `ask`, which has no afferents of
its own in this circuit, so freezing is not load-bearing for curiosity's read correctness, only cleanliness).

THE HEBBIAN RULE-SHAPE SEAM (Wave 1's OWN new-seam finding, checked for applicability here, found NOT
APPLICABLE): Wave 1 found source_provenance's BUILD-TIME HEBBIAN ENCODE window needed 4 additional saved/restored
rule-shape keys (hebbian_rate_window/hebbian_coactivity_decay/hebbian_coactivity_thresh/hebbian_mean_subtract)
because it runs a LIVE Hebbian update during its own construction. None of Wave 2's three organs run a live
HEBBIAN update at their own construction: self_schema's loop is a FIXED explicit_wiring_fn weight (never trained
by any rule); causal_whatif's build-time train uses STDP+DA (a DIFFERENT plasticity mechanism, its own local
stdp_a_plus/tau/w_max, entirely disjoint from the Hebbian rule-shape keys); curiosity has no build-time encode at
all (a pure OU+neuromod read). So this seam is inapplicable by construction, not merely untested — confirmed by
the gate (a) result below showing zero co-residence delta for all three.

THE GATE (numpy, bit-exact) — the same 3-part rung Wave 1 used, extended to 9 organs:
  (a) ORGAN-READ byte-identity — each of the 9 organs' read on the wave2 pool == its read CO-RESIDENT-ALONE on the
      wave2 superset config (co-residence invariance).
  (b) FACULTY-ALIVE — each organ still produces its live, non-degenerate verdict on the wave2 pool.
  (c) ANSWER-PRESERVATION — the 6 wave-1 organs' rendered answer == the ACTUAL SHIPPED `get_wave1_pool(seed)`'s
      answer (strict, apples-to-apples: both run the merge engine's seams); self_schema/curiosity/causal_whatif's
      answer == their TODAY'S STANDALONE construction (shared=None). The categorical ANSWER is gated for every
      organ (matching Wave 1's own precedent); the CONTINUOUS margin for the 3 new organs is reported but not
      gated (an honest declared residual — a spiking-dynamics read integrated over hundreds of steps amplifies a
      single-ULP per-step delta into a 1-spike divergence vs an UNSEAMED standalone build, the same documented
      sensitivity Wave 1 named for comprehension/source_provenance).
  + GAIN-0 FREEZE HOLDS — self_schema/curiosity's internal edge weights are byte-identical before vs after the
    full train+read lifecycle. causal_whatif's "evt" is DELIBERATELY EXCLUDED from this specific before/after
    check (see `_WAVE2_FREEZE`'s comment) — mirroring Wave 1's own precedent for comprehension/source_provenance,
    a build-time-plasticity-then-frozen organ's own region is SUPPOSED to change once at its own construction;
    `freeze_regions=("evt",)` on its descriptor still protects it from every OTHER organ's ongoing Hebbian both
    at pool-build time and after its own training completes (verified separately below by inspection, not this
    array-diff check, since a naive before/after diff cannot distinguish "the organ's own legitimate train" from
    "a leak" for a region that is EXPECTED to change).
  + LEGACY DISCRIMINATOR — the seams-OFF pool diverges merged-vs-coresident (byte-identity is NOT vacuous).

HONEST SCOPE. MIGRATION-SAFETY organ-read rung only (byte-identity-in-isolation), NOT the one-brain INTEGRATION
goal — zero cross-region synapses are added here (a pool with no cross-edges is MIGRATED, not INTEGRATED). NOT
wired into any live `get_organ()` dispatch: `onebrain_wave2_pool_production.py`'s flag is additive/default-OFF
and touches NO existing production file (verify with `git diff`).

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_wave2_organread_verify \
        --seeds 42,43,44 \
        --out research/findings/raw/_onebrain_wave2/organread_3seed_smoke.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.runners.onebrain_merge_framework import (
    merge_organs, REGISTRY, _host, _idx, substrate_byte_identity,
    _spec_self_schema, _spec_curiosity,
    _self_schema_geom, _ordered_region_idx, _self_schema_post_inject, _self_schema_organ,
    _self_schema_reads, _self_schema_answer,
    _curiosity_organ, _curiosity_reads, _curiosity_answer, _CuriosityReadOrgan,
    _causal_organ, _causal_reads, _causal_answer, _CausalReadOrgan,
)
from research.runners._onebrain_wave1_organread_verify import (
    _wave1_descriptors, _isolated_reads, _isolated_read_one, _maxdelta,
)
from research.runners.onebrain_wave1_pool_production import get_wave1_pool
from research.runners.self_schema_production_organ import SelfSchemaAuthorshipOrgan

WAVE1_CARRIED_KEYS = ("surprise", "worldmodel", "metacog", "pragmatic", "comprehension", "source_provenance")
NEW_KEYS = ("self_schema", "curiosity", "causal_whatif")
WAVE2_KEYS = list(WAVE1_CARRIED_KEYS) + list(NEW_KEYS)

# ── SEAM 1/3: self_schema's region-name + wiring-key collision with metacog (see module docstring) ──
_SS_RENAME = {"workspace": "ss_workspace", "workspace_fs": "ss_workspace_fs"}
# ── SEAM 2: curiosity's region-name collision with surprise ──
_CUR_RENAME = {"cue": "cur_cue"}


def _renamed_spec(base_spec_fn, rename: dict):
    """Wrap a descriptor's spec_fn to rename regions (+ any RegionPathway endpoint referencing them) — avoids a
    genuine region-NAME collision when folding this organ into a superset pool that already owns the same name.
    Preserves every OTHER BrainRegion/RegionPathway field (`dataclasses.replace` only touches `name`/`from_region`/
    `to_region`), so the renamed spec is otherwise byte-identical (same n_neurons/exc_fraction/enable_nmda/density/
    weight_mean/plastic/...) to the un-renamed standalone — the rename changes ONLY the string the pool's
    owner-dict + wiring key on, nothing about the per-neuron init or topology."""
    def spec(seed):
        regions, pathways, meta = base_spec_fn(seed)
        regions = [replace(rg, name=rename.get(rg.name, rg.name)) for rg in regions]
        pathways = [replace(pw, from_region=rename.get(pw.from_region, pw.from_region),
                            to_region=rename.get(pw.to_region, pw.to_region)) for pw in pathways]
        return regions, pathways, meta
    return spec


_wave2_self_schema_spec = _renamed_spec(_spec_self_schema, _SS_RENAME)
_wave2_curiosity_spec = _renamed_spec(_spec_curiosity, _CUR_RENAME)


def _wave2_self_schema_member_attend(bridge):
    """Identical geometry to `onebrain_merge_framework._self_schema_member_attend`, keyed on the RENAMED
    `ss_workspace` region instead of `workspace` (SEAM 1)."""
    g = _self_schema_geom()
    ws = _ordered_region_idx(bridge, "ss_workspace")
    ss = _ordered_region_idx(bridge, "self_schema")
    member = {k: ws[k * g["A"]:(k + 1) * g["A"]] for k in range(g["K"])}
    attend = {k: ss[k * g["AT"]:(k + 1) * g["AT"]] for k in range(g["K"])}
    base = g["AT"] * g["K"]
    confid = ss[base:base + g["CF"]]
    author = ss[base + g["CF"]:base + g["CF"] + g["AU"]]
    return g, member, attend, confid, author


def _wave2_self_schema_idx(bridge):
    from sim.backend import get_backend
    xp, _ = get_backend()
    _g, member, attend, confid, author = _wave2_self_schema_member_attend(bridge)
    return {
        "member_dev": {k: xp.asarray(v) for k, v in member.items()},
        "attend_dev": {k: xp.asarray(v) for k, v in attend.items()},
        "confid_dev": xp.asarray(confid),
        "author_dev": xp.asarray(author),
    }


def _wave2_self_schema_wiring(bridge, rm):
    """explicit_wiring_fn — IDENTICAL topology to the framework's own `_self_schema_wiring` (the K dense
    self-recurrent workspace loops + the fixed member->attend projection), but with PREFIXED union keys
    (`ss_loop_k`/`ss_member{k}_to_attend`) so it cannot silently overwrite metacog's OWN `loop_0`/`loop_1` entries
    in the shared wiring union (SEAM 3 in the module docstring)."""
    from research.runners._gnw_rung1_ignition_curve_derisk import _build_assembly_loop_population
    from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
    g, member, attend, _confid, _author = _wave2_self_schema_member_attend(bridge)
    union = {}
    for k in range(g["K"]):
        union[f"ss_loop_{k}"] = _build_assembly_loop_population(member[k], g["LOOP_W"])
        union[f"ss_member{k}_to_attend"] = _dense_projection(member[k], attend[k], g["MTA"], g["GATE"])
    return union


def _wave2_descriptors():
    """The 9-organ Wave-2 family: the EXISTING 6-organ wave-1 reconciliation UNCHANGED (`_wave1_descriptors`,
    reuse-by-import — zero drift from the shipped wave-1 pool) + self_schema (renamed regions/wiring-keys, SEAMS
    1+3) + curiosity (renamed region, SEAM 2) + causal_whatif (standard hebbian pop+freeze, no rename needed —
    verified its region ("evt") and wiring key ("xblock") collide with nothing in the wave-1 superset)."""
    base = _wave1_descriptors()

    ss = REGISTRY["self_schema"]
    ss_cfg = dict(ss.config)
    ss_cfg.pop("enable_hebbian_learning", None)
    ss_r = replace(ss, spec_fn=_wave2_self_schema_spec, config=ss_cfg,
                  regions=("ss_workspace", "ss_workspace_fs", "self_schema"),
                  idx_fn=_wave2_self_schema_idx, explicit_wiring_fn=_wave2_self_schema_wiring,
                  post_inject_fn=_self_schema_post_inject,
                  freeze_regions=("ss_workspace", "ss_workspace_fs", "self_schema"),
                  organ_cls=_self_schema_organ, read_fn=_self_schema_reads, answer_fn=_self_schema_answer)

    cur = REGISTRY["curiosity"]
    cur_r = replace(cur, spec_fn=_wave2_curiosity_spec,
                    regions=("cur_cue", "striosome_value", "reward_us", "snc", "ask"),
                    freeze_regions=("cur_cue", "striosome_value", "reward_us", "snc", "ask"),
                    organ_cls=_curiosity_organ, read_fn=_curiosity_reads, answer_fn=_curiosity_answer)

    causal = REGISTRY["causal_whatif"]
    causal_cfg = dict(causal.config)
    causal_cfg.pop("enable_hebbian_learning", None)
    causal_r = replace(causal, config=causal_cfg, freeze_regions=("evt",),
                       organ_cls=_causal_organ, read_fn=_causal_reads, answer_fn=_causal_answer)

    return base + [ss_r, cur_r, causal_r]


def _frozen_edge_weights(bridge, frozen_regions):
    """Every internal edge among `frozen_regions` — the array the gain-0 freeze must hold byte-identical across
    the full train+read lifecycle. Generalizes wave1's `_frozen_edge_weights` (hardcoded to `_POOL2_FREEZE`) to an
    arbitrary region list, so Wave 2 can check its OWN 3 newly-frozen organs without re-deriving the mechanism."""
    idx = set()
    for name in frozen_regions:
        idx |= set(int(i) for i in _idx(bridge, name))
    arr = np.asarray(sorted(idx), dtype=np.int64)
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col)); data = np.asarray(_host(coo.data))
    both = np.isin(row, arr) & np.isin(col, arr)
    order = np.lexsort((col[both], row[both]))
    return data[both][order].astype(np.float64)


# NOTE: "evt" (causal_whatif's own region) is DELIBERATELY EXCLUDED from this before/after check, mirroring
# Wave 1's EXACT precedent for comprehension/source_provenance (`_onebrain_wave1_organread_verify._POOL2_FREEZE`
# scopes to metacog/pragmatic ONLY, excluding comprehension/source_provenance "on purpose: both organs INSTALL
# their own weights AT CONSTRUCTION ... so their edges are SUPPOSED to change once ... that is not a freeze
# violation, it is the organ doing its job"). causal_whatif is the SAME category: its `_CausalReadOrgan.
# ensure_built()` runs a genuine BUILD-TIME STDP+DA train of its OWN "evt" slice (the trained xblock weights ARE
# the organ's job), so a nonzero before/after delta on evt is EXPECTED, not a leak. `freeze_regions=("evt",)` on
# the descriptor is UNCHANGED (still needed at pool-build time + for causal_whatif's own local
# freeze-then-restore baseline, protecting evt from OTHER organs' Hebbian AFTER its own training completes) — only
# the VERIFICATION scope below excludes it. self_schema + curiosity install NOTHING at their own construction (no
# live training ever touches their edges), so both correctly stay IN this check.
_WAVE2_FREEZE = ("ss_workspace", "ss_workspace_fs", "self_schema",
                 "cur_cue", "striosome_value", "reward_us", "snc", "ask")


def _faculty_alive(reads, answers):
    """(b) each organ still produces a live, non-degenerate verdict on the wave2 pool."""
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
    return {"surprise": surprise, "worldmodel": worldmodel, "metacog": metacog, "pragmatic": pragmatic,
            "comprehension": comprehension, "source_provenance": source_provenance,
            "self_schema": self_schema, "curiosity": curiosity, "causal_whatif": causal_whatif,
            "surprise_sep": float(surp_sep)}


def verify_seed(seed: int, verbose: bool = True) -> dict:
    descs = _wave2_descriptors()
    keys = [d.key for d in descs]
    assert keys == WAVE2_KEYS, f"unexpected descriptor key order {keys}"

    # ── MERGED-9 (the literal wave-2 pool, wire=True) — all 9 organs read with per-organ isolation ──
    merged = merge_organs(descs, seed, wire=True)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    frozen_before = _frozen_edge_weights(merged.bridge, _WAVE2_FREEZE)
    R_merged, A_merged, _organs = _isolated_reads(merged, descs, seed)
    frozen_after = _frozen_edge_weights(merged.bridge, _WAVE2_FREEZE)
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
    # STRICT (gates the verdict): the 6 wave-1-carried organs vs the ACTUAL PRODUCTION wave-1 pool
    # (`get_wave1_pool`) — a fair apples-to-apples comparison, since that pool ALSO runs the merge engine's
    # FP-determinism seams. Extending it with 3 more organs must not perturb the 6 it originally carried.
    wave1_descs = _wave1_descriptors()
    shipped_wave1 = get_wave1_pool(seed)
    R_ship, A_ship = {}, {}
    Rb, Ab, _ = _isolated_reads(shipped_wave1, wave1_descs, seed)
    for d in wave1_descs:
        R_ship[d.key] = Rb[d.key]; A_ship[d.key] = Ab[d.key]

    # INFORMATIONAL for the CONTINUOUS margin only (ANSWER itself IS gated): self_schema/curiosity/causal_whatif
    # vs a RAW STANDALONE build (shared=None) that does NOT run the merge engine's seams at all — the same honest,
    # not strict, comparison Wave 1 made for comprehension/source_provenance (a SPIKING-DYNAMICS read integrated
    # over hundreds of steps amplifies a single-ULP per-step delta into a 1-spike divergence vs an unseamed build).
    ss_standalone = SelfSchemaAuthorshipOrgan(seed=seed)
    R_ship["self_schema"] = _self_schema_reads(ss_standalone)
    A_ship["self_schema"] = _self_schema_answer(ss_standalone)

    cur_standalone = _CuriosityReadOrgan(seed, shared=None)
    R_ship["curiosity"] = _curiosity_reads(cur_standalone)
    A_ship["curiosity"] = _curiosity_answer(cur_standalone)

    causal_standalone = _CausalReadOrgan(seed, shared=None)
    R_ship["causal_whatif"] = _causal_reads(causal_standalone)
    A_ship["causal_whatif"] = _causal_answer(causal_standalone)

    shipped = {}
    for k in keys:
        dd, wk, miss = _maxdelta(R_merged[k], R_ship[k])
        shipped[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                      "read_byte_identical": bool(dd == 0.0 and not miss),
                      "answer_same": bool(A_merged[k] == A_ship[k]),
                      "strict": k in WAVE1_CARRIED_KEYS}

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
    ship_read_ok_strict = all(shipped[k]["read_byte_identical"] for k in WAVE1_CARRIED_KEYS)
    legacy_ok = bool(legacy_delta > 0.0)
    go = bool(a_ok and b_ok and c_ok and gain0_ok and legacy_ok)

    res = {"seed": seed, "n_all_neurons": n_all,
           "gate_a_coresidence_byte_identical": a_ok, "coresident": coresident,
           "gate_b_faculty_alive": b_ok, "faculty_alive": alive,
           "gate_c_answer_preserved": c_ok, "wave1_carried_read_byte_identical": ship_read_ok_strict,
           "shipped": shipped,
           "gain0_freeze_holds": gain0_ok, "gain0_freeze_delta": freeze_delta,
           "n_frozen_edges": int(frozen_before.shape[0]),
           "legacy_diverges": legacy_ok, "legacy_delta": legacy_delta, "GO": go}
    if verbose:
        print(f"  [seed {seed}] N={n_all} | (a)cores_byteid={a_ok} (b)alive={b_ok} (c)answer={c_ok} "
              f"wave1_read={ship_read_ok_strict} gain0={gain0_ok}(n={int(frozen_before.shape[0])}) "
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
        "n_wave1_carried_read_byte_identical": sum(p["wave1_carried_read_byte_identical"] for p in per_seed),
        "n_gain0_freeze": sum(p["gain0_freeze_holds"] for p in per_seed),
        "n_legacy_diverges": sum(p["legacy_diverges"] for p in per_seed),
        "n_go": sum(p["GO"] for p in per_seed),
    }
    per_organ = {}
    for k in WAVE2_KEYS:
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

    print("=== ONE-BRAIN WAVE 2 — ORGAN-READ verify: self_schema + curiosity + causal_whatif on the WAVE-1 POOL ===")
    print("    surprise + world-model + metacog + pragmatic + comprehension + source_provenance (the shipped "
         "wave-1 pool) + self_schema + curiosity + causal_whatif on ONE bridge")
    out = verify(seeds)
    ag = out["aggregate"]; n = ag["n_seeds"]
    print("\n=== VERDICT (Wave 2 organ-read rung) ===")
    for k in WAVE2_KEYS:
        po = out["per_organ"][k]
        print(f"  {k:18s} cores_byteid={po['n_coresidence_byte_identical']}/{n} "
              f"ship_read={po['n_shipped_read_byte_identical']}/{n} answer_same={po['n_answer_same']}/{n} "
              f"alive={po['n_alive']}/{n} (max cores_d={po['max_coresidence_delta']:.2e} ship_d={po['max_shipped_delta']:.2e})")
    print(f"\n  (a) organ-read byte-identity (co-residence invariance): {ag['n_gate_a']}/{n}")
    print(f"  (b) faculty-alive:                                      {ag['n_gate_b']}/{n}")
    print(f"  (c) answer-preservation vs shipped baselines:           {ag['n_gate_c']}/{n}")
    print(f"      wave1-carried read byte-identity (migration fidelity): {ag['n_wave1_carried_read_byte_identical']}/{n}")
    print(f"      gain-0 freeze holds frozen organs' edges:           {ag['n_gain0_freeze']}/{n}")
    print(f"      legacy discriminator diverges (non-vacuous):        {ag['n_legacy_diverges']}/{n}")
    print(f"  ORGAN-READ RUNG GO (a & b & c & gain0 & legacy): {ag['n_go']}/{n}  ->  ALL-GO={out['all_go']}")

    from tools.verdict import Verdict
    v = Verdict("one-brain Wave-2 organ-read (self_schema + curiosity + causal_whatif folded onto the wave-1 "
               f"pool, N~{out['per_seed'][0]['n_all_neurons']})")
    v.require("(a) organ-read byte-identity — every organ's read co-residence-invariant, every seed",
              ag["n_gate_a"], expect=n)
    v.require("(b) faculty-alive — every organ produces its live verdict on the wave2 pool, every seed",
              ag["n_gate_b"], expect=n)
    v.require("(c) answer-preservation — every organ's rendered answer == its shipped/standalone baseline, "
              "every seed", ag["n_gate_c"], expect=n)
    v.require("gain-0 freeze holds self_schema/curiosity/causal_whatif internal edges bit-frozen across the "
              "train+read lifecycle, every seed", ag["n_gain0_freeze"], expect=n)
    v.require("legacy discriminator diverges (byte-identity NOT vacuous), every seed",
              ag["n_legacy_diverges"], expect=n)
    v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
              why="MIGRATION gate: byte-identity-in-isolation forbids cross-synapses BY DEFINITION")
    v.disabled("live-chat production wiring (get_organ() dispatch)",
              why="deliberately deferred to a separate, later commit — mirrors the wave-1 pool's own sequencing "
                  "(organ-read GO landed before its production wiring commit)")
    decided = v.decide(go=out["all_go"])

    payload = {"mode": "onebrain_wave2_organread", **out}
    payload.update(decided)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
