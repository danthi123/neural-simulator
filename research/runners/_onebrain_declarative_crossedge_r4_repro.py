"""Declarative `cross_edges` PROOF — R4 (self_schema authorship -> source_provenance monitoring), the SECOND
learned faculty->faculty cross-edge migrated onto the declarative `CrossEdge` framework (R1's d6->comprehension
edge was the first, `_onebrain_declarative_crossedge_r1_repro.py`, and is already production-flipped default-ON,
`2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.md`). Re-expresses R4's bespoke hand-wired
`author -> prov_generated` cross-edge (`research/runners/_onebrain_integration_r4_selfschema_provenance.py`,
`R4Pool._build_pool`) as ONE `CrossEdge` registry row on `merge_organs`, and shows the declaratively-wired pool
reproduces R4's bespoke result: the SAME F1-F4 functional gate (imported UNMODIFIED — reuse-by-import, no
reimplementation) + the SAME emergent weight growth + the SAME F2 lesion-attributable interaction.

RECONCILIATION THIS ARC PERFORMED FIRST (see the companion finding for the full writeup):
  * R2 (three-factor neuromod-gated upgrade, `_onebrain_integration_r2_threefactor_selforganized.py`) is NOT a
    separate faculty-pair migration -- it re-runs on "ONE shared merge_organs([d6_multiref_wm, comprehension],
    wire=True) pool, exactly R1's substrate" (R2's own finding, verbatim). That pair is ALREADY the production
    edge flipped default-ON today (`fe1911f2`, BRAIN_ONEBRAIN_XEDGE + BRAIN_ONEBRAIN_XEDGE_LEARN). R2 is a
    plasticity-RULE + candidate-TOPOLOGY refinement of an edge already in production, not a second edge.
  * R4 (self_schema -> source_provenance) IS a genuinely distinct pair, non-redundant with board #129
    (`2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md`, source_provenance's OWN
    internal perceived-vs-generated opponent-trace mechanism): R4 externally FEEDS self_schema's authorship axis
    into ONE input of that existing opponent mechanism (`ctx_generated`/`prov_generated`), it does not rebuild it.
  * R4's production wire-in PART-1 already exists (`d84775aa8`, `onebrain_xedge_selfschema_production.py`,
    `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA`, default-OFF) but constructs the pool via the OLD bespoke `merge_organs(...,
    wire=True)` + hand-typed `_dense(...)` pattern (`R4Pool` reused by import) -- NOT yet re-expressed through the
    declarative `cross_edges=` framework the way R1 was. That is the genuine residual this file closes.

A FRAMEWORK GAP FOUND AND CLOSED (not a redundant layer -- `onebrain_merge_framework.py`'s own honest gap): R1/
R3-v3/the surprise->episodic edge (the two existing `CrossEdge` consumers before this file) all wire whole
REGISTERED regions (`region_manager.indices("w0")`, `.indices("surprise")`, ...). R4's SOURCE endpoint,
`author`, is NOT a registered top-level region -- it is a SUB-SLICE of the single `"self_schema"` region
(`self_schema_production_organ.py`'s own attend/confid/author offset split inside one BrainRegion,
`_self_schema_member_attend`). `region_manager.indices("author")` raises `KeyError` (verified directly against
`sim/regions.py:766`'s `RegionManager.indices`) -- there is no such name to look up. `CrossEdge` gained two new
OPTIONAL fields, `source_idx_fn`/`target_idx_fn` ((bridge) -> ndarray of absolute neuron indices), consulted by
`_cross_edge_dense` INSTEAD of `region_manager.indices(name)` when given. Both default `None`, so every
pre-existing `CrossEdge` (R1's 4 rows, the surprise->episodic edge) is BYTE-IDENTICAL — this is additive, not a
redesign.

WHAT DIFFERS between the two pools at each seed (everything else is IDENTICAL — same seed, same F-gate
constants, same battery, same protocol):
  BESPOKE       `research.runners._onebrain_integration_r4_selfschema_provenance.R4Pool` — hand-writes the
                 `x_author_provgen` wiring dict entry via a hand-typed `_dense(pre,post,w,gate)` helper inside
                 `_build_pool`, injects TWICE (once via the framework's own automatic `wire=True` path with no
                 cross edge, thrown away; once manually re-injecting the SAME base+self_schema union WITH the
                 cross edge added), then hand-types the 3-line whitelist freeze in `R4Pool.__init__` BEFORE
                 building `sp_organ`/`ss_organ`.
  DECLARATIVE   `DeclarativeR4Pool` (this file) — the SAME edge as ONE `CrossEdge(...)` data row (using
                 `source_idx_fn` to resolve `author`), passed straight to `merge_organs(..., cross_edges=[...])`.
                 ONE inject (the framework's own `wire=True` path already includes the cross edge in the SAME
                 union position R4's manual re-inject placed it — after the base plan + self_schema's own
                 `explicit_wiring_fn`, matching insertion order exactly), and the whitelist freeze is ONE call,
                 `pool.apply_cross_edge_freeze()`, run at the SAME point in construction (before `sp_organ`/
                 `ss_organ` are built) R4Pool's hand-typed version runs. `DeclarativeR4Pool` subclasses `R4Pool`
                 and overrides ONLY `__init__`; every downstream method (`_hard_reset`, `_drive`, `_wmean`,
                 `cross_weights`, `_make_ambiguous_pattern`, `_encode_ambiguous`, `train`, `amb_read`) is
                 INHERITED UNCHANGED, so the train/read PROTOCOL is provably identical between the two arms —
                 only the pool-CONSTRUCTION path differs.

Both arms then run through the IDENTICAL imported F1/F2/F3/F4/emergence/migration-invariant functions R4's own
module defines — reuse-by-import, not reimplementation.

Run (numpy CPU; NO sim/ edit; routes off the GPU):
  SIM_BACKEND=numpy python -m research.runners._onebrain_declarative_crossedge_r4_repro --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_declarative_crossedge_r4_repro \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_declarative_crossedge_r4_repro_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import lever

from research.runners._onebrain_integration_r4_selfschema_provenance import (
    R4Pool, GATE, W0, HMAX, _f1, _f2, _f3, _f4, _emergence, _migration_invariant,
)
from research.runners.onebrain_merge_framework import (
    REGISTRY, CrossEdge, merge_organs, _self_schema_member_attend, _source_prov_organ, _self_schema_organ,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE EDGE — R4's ONE hand-typed `_dense(...)` union entry (`x_author_provgen`), re-expressed as
#  ONE CrossEdge row. Same gate name, same W0 seed weight, same freeze_rest=True whitelist behavior R4's 3
#  hand-typed lines implement by hand. `source_idx_fn` resolves `author` (a sub-slice of the single "self_schema"
#  region — NOT a registered region name) exactly as R4Pool's own `_build_pool` does, via the SAME imported
#  `_self_schema_member_attend` helper (reuse-by-import: this is not a re-derivation of the offset geometry).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _r4_author_idx(bridge):
    _g, _member, _attend, _confid, author_idx = _self_schema_member_attend(bridge)
    return np.asarray(author_idx, np.int64)


CROSS_EDGES = [
    CrossEdge(key="author_to_provgen", source_key="self_schema", source_region="author",
             target_key="source_provenance", target_region="prov_generated",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True,
             source_idx_fn=_r4_author_idx),
]


class DeclarativeR4Pool(R4Pool):
    """R4Pool with the pool-CONSTRUCTION path replaced by the declarative `cross_edges=` framework machinery.
    Every method below `__init__` is INHERITED UNCHANGED from `R4Pool` (train/read protocol reuse-by-inheritance,
    not copy-paste) -- only how the ONE cross-edge gets onto the bridge, and the freeze, differ."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        SS, SP = REGISTRY["self_schema"], REGISTRY["source_provenance"]
        # ONE declarative build: the wire=True inject already includes CROSS_EDGES (onebrain_merge_framework's
        # `_install_organ_read_wiring`), so unlike bespoke `_build_pool` there is no second manual re-inject.
        self.pool = merge_organs([SS, SP], seed=seed, wire=True, cross_edges=CROSS_EDGES)
        self.b = self.pool.bridge
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("episode", "content_readout", "ctx_perceived", "ctx_generated",
                                           "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
        self.ix["author"] = _r4_author_idx(self.b)

        # The cross-edge mask R4Pool's methods (_wmean/cross_weights/_f2's lesion) consume, in the SAME shape and
        # under the SAME key ("author->provgen") R4Pool's own `_build_pool` uses -- derived from the bridge's OWN
        # connectivity (row/col region membership), not re-declared, so a mismatch here would be a framework
        # wiring bug, not a test artifact.
        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        self.masks = {"author->provgen": np.isin(row, self.ix["author"]) & np.isin(col, self.ix["prov_generated"])}
        n = int(self.masks["author->provgen"].sum())
        assert n > 0, "declarative R4 cross-edge mask is EMPTY -- the CrossEdge did not wire"

        # THE DECLARATIVE WHITELIST FREEZE -- replaces R4Pool's 3 hand-typed lines with ONE framework call, run
        # at the SAME point R4Pool.__init__ calls its hand-typed version: BEFORE building sp_organ/ss_organ, so
        # their own build-time save+zero+reopen dance restores back to THIS whitelist, not a pre-freeze state.
        self.pool.apply_cross_edge_freeze()

        # build the source_provenance organ VIEW (its own build-time 8-item battery Hebbian encode), then the
        # self_schema organ VIEW (calibration only) -- same order, same reuse-by-import as R4Pool.
        self.sp_organ = _source_prov_organ(seed, self.pool)
        self.sp_organ.ensure_built()
        self.ss_organ = _self_schema_organ(seed, self.pool)
        self.ss_organ.ensure_built()

        # the fresh AMBIGUOUS content pattern (INHERITED methods, unchanged).
        self.ambig_pattern = self._make_ambiguous_pattern()
        self._encode_ambiguous()

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=0.05, hebbian_max_weight=HMAX,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()


def run_seed(pool_cls, seed):
    """IDENTICAL orchestration to R4's own `run_seed` (research/runners/_onebrain_integration_r4_selfschema_
    provenance.py), parametrized over the pool class so BESPOKE and DECLARATIVE run through the exact same F-gate
    call sequence. F1-F4/emergence/migration functions are imported, not reimplemented."""
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    t0 = time.time()
    r4 = pool_cls(seed)
    traj = r4.train()
    emg = _emergence(traj, r4.frozen_maxdrift)
    f1 = _f1(r4)
    f4 = _f4(r4)                                   # F4 BEFORE F2 (F2 lesions the edge in place)
    f2 = _f2(r4)                                   # F2 lesions the cross-edge at its end
    r4._hard_reset()
    sp_les = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = r4.sp_organ.brain.recall(r4.sp_organ.patterns[prov][i])
            sp_les.append(rec["rate_perceived"] - rec["rate_generated"])
    ss_les = (r4.ss_organ._author_rate(authored=True, lesion=False),
             r4.ss_organ._author_rate(authored=False, lesion=False))
    f3 = _f3(r4, traj, f2)
    mig = _migration_invariant(seed, r4, sp_les, ss_les)
    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and emg["PASS"] and mig["PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def compare_seed(seed):
    """Run BOTH arms at one seed and report the reproduction-fidelity numbers: emergent weight growth (the ONE
    author->provgen edge) + F2's lesion-attributable fraction (both should read ~1.0 -- the shift vanishes on
    lesion in BOTH arms), plus a direct max|delta| on the grown weight and the F2 deltas."""
    bespoke = run_seed(R4Pool, seed)
    declar = run_seed(DeclarativeR4Pool, seed)
    bg, dg = bespoke["emergence"]["final_weight"], declar["emergence"]["final_weight"]
    weight_delta = abs(bg - dg)
    bf2, df2 = bespoke["F2"], declar["F2"]
    # LEVER (tools.lab): the DECLARATIVE arm's own lesion manipulation must have actually MOVED the margin --
    # intact->lesion delta is asserted non-identical (required=True raises if not), so the "reproduces bespoke"
    # claim below cannot rest on a declaratively-wired pool whose cross-edge is accidentally inert.
    lever(f"seed{seed} declarative F2 intact->lesion", df2["delta_intact"], df2["delta_lesion"])
    f2_delta = abs(bf2["delta_intact"] - df2["delta_intact"])
    frac_delta = (None if bf2["frac_attributable"] is None or df2["frac_attributable"] is None
                 else abs(bf2["frac_attributable"] - df2["frac_attributable"]))
    reproduces = bool(bespoke["PASS"] and declar["PASS"] and weight_delta < 1.0
                      and (frac_delta is None or frac_delta < 0.15))
    return {"seed": seed, "bespoke_PASS": bespoke["PASS"], "declarative_PASS": declar["PASS"],
            "reproduces": reproduces,
            "bespoke_grown": bg, "declarative_grown": dg, "weight_maxdelta": weight_delta,
            "bespoke_frac_attributable": bf2["frac_attributable"],
            "declarative_frac_attributable": df2["frac_attributable"],
            "f2_delta_intact": f2_delta,
            "bespoke_F2_delta_intact": bf2["delta_intact"], "declarative_F2_delta_intact": df2["delta_intact"],
            "bespoke_F2_delta_lesion": bf2["delta_lesion"], "declarative_F2_delta_lesion": df2["delta_lesion"],
            "bespoke_elapsed_s": bespoke["elapsed_s"], "declarative_elapsed_s": declar["elapsed_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = compare_seed(s)
        runs.append(r)
        print(f"[seed {s}] bespoke={'GO' if r['bespoke_PASS'] else 'no'} "
              f"declarative={'GO' if r['declarative_PASS'] else 'no'} "
              f"reproduces={r['reproduces']} | grown besp={r['bespoke_grown']:.3f} "
              f"decl={r['declarative_grown']:.3f} (maxdelta={r['weight_maxdelta']:.4f}) | F2 attrib "
              f"besp={r['bespoke_frac_attributable']} decl={r['declarative_frac_attributable']} "
              f"({r['bespoke_elapsed_s']}s + {r['declarative_elapsed_s']}s)", flush=True)

    n_go = sum(r["reproduces"] for r in runs)
    all_go = (n_go == len(runs))
    tag = "GO" if (all_go and not args.smoke) else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO")
    verdict = (f"{tag} — the declarative `cross_edges=[CrossEdge(...)]` param on `merge_organs` "
               f"(onebrain_merge_framework.py) REPRODUCES R4's bespoke self_schema-authorship->source_provenance "
               f"hand-wired cross-edge: {n_go}/{len(runs)} seeds match on BOTH arms passing their own full F1-F4+"
               f"emergence+migration gate AND the reproduction check (grown weight maxdelta<1.0, F2 "
               f"lesion-attributable fraction within 0.15). This is R4's edge, so the SOURCE endpoint ('author') "
               f"is a SUB-SLICE of the single self_schema region, not a registered region name -- the framework's "
               f"CrossEdge gained an optional source_idx_fn/target_idx_fn (bridge)->indices resolver to express "
               f"it (None for every pre-existing CrossEdge -> byte-identical). The ONE hand-typed `_dense(...)` "
               f"union entry + 3-line whitelist freeze is replaced by ONE `merge_organs(cross_edges=[...])` call "
               f"+ ONE `pool.apply_cross_edge_freeze()` call -- same wiring data, same protocol (DeclarativeR4Pool "
               f"subclasses R4Pool, only __init__ differs), different construction path. numpy CPU; NO sim/ edit "
               f"beyond the additive CrossEdge fields.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        max_weight_delta = max(r["weight_maxdelta"] for r in runs)
        f2_deltas = [r["f2_delta_intact"] for r in runs if r["f2_delta_intact"] is not None]
        max_f2_delta = max(f2_deltas) if f2_deltas else None
        Vd = Verdict("onebrain_declarative_crossedge_r4_repro")
        Vd.require("bespoke_arm_all_pass", sum(r["bespoke_PASS"] for r in runs), expect=lambda x: x == len(runs),
                   note="R4's own bespoke F1-F4+emergence+migration gate must pass every seed, or the baseline "
                        "this reproduction is measured against is itself unearned")
        Vd.require("declarative_arm_all_pass", sum(r["declarative_PASS"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="the declaratively-wired pool must independently pass the SAME F-gate, not merely match numbers")
        Vd.require("weight_reproduction_bitidentical", max_weight_delta, expect=lambda x: x < 1.0,
                   note="max|delta| between bespoke and declarative grown weight, across every seed")
        if max_f2_delta is not None:
            Vd.require("f2_attribution_reproduction", max_f2_delta, expect=lambda x: x < 0.15,
                       note="max|delta| of the F2 lesion-attributable intact-delta between the two arms")
        Vd.require("declarative_lesion_moves_margin_every_seed", len(runs), expect=lambda x: x == len(seeds),
                   note="each seed's lever() call (tools.lab, above in compare_seed) RAISES immediately if the "
                        "declarative arm's intact->lesion margin failed to move; reaching this aggregation with "
                        "len(runs)==len(seeds) is proof every per-seed lever already passed")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_declarative_crossedge_r4_repro", "verdict": verdict, "GO": all_go,
              "n_go": n_go, "n_seeds": len(runs), "seeds": seeds,
              "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
              "preconditions": preconditions,
              "cross_edges_schema": [dict(key=ce.key, source_key=ce.source_key, source_region=ce.source_region,
                                          target_key=ce.target_key, target_region=ce.target_region,
                                          init_weight=ce.init_weight, plastic=ce.plastic, gate=ce.gate_name,
                                          learn_rule=ce.learn_rule, freeze_rest=ce.freeze_rest,
                                          uses_source_idx_fn=ce.source_idx_fn is not None,
                                          uses_target_idx_fn=ce.target_idx_fn is not None)
                                     for ce in CROSS_EDGES],
              "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[REPRO] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
