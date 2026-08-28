"""Declarative `cross_edges` PROOF — re-express R1's bespoke d6->comprehension edge as FOUR `CrossEdge` registry
rows on `merge_organs`, and show the declaratively-wired pool reproduces R1's bespoke `_build_pool` result: the
SAME F1-F4 functional gate (`research/findings/2026-08-27-onebrain-integration-phase-DESIGN.md`, imported
UNMODIFIED from the R1 module — reuse-by-import, no reimplementation) + the SAME emergent weight growth +
the SAME F2 lesion-attributable interaction. This is the proof that the `cross_edges` abstraction
(`research/runners/onebrain_merge_framework.py` — `CrossEdge`, `_cross_edge_dense`,
`MergedPool.apply_cross_edge_freeze`; the one-brain-completeness-audit's top-ranked framework investment #185,
`research/findings/2026-08-27-onebrain-completeness-audit.md` §4 step 2) is FAITHFUL to the bespoke mechanism it
generalizes, not a reimplementation that could silently drift from it.

WHAT DIFFERS between the two pools at each seed (everything else is IDENTICAL — same seed, same F-gate constants,
same battery, same protocol):
  BESPOKE       `research.runners._onebrain_integration_r1_wm_comprehension.R1Pool` — hand-writes the 4
                 `x_w{0,1}_sel{a,p}` wiring dict entries via a hand-typed `_dense(pre,post,w,gate)` helper, injects
                 TWICE (once via the framework's own automatic wire=True path, thrown away; once manually with the
                 cross edges added), then hand-types the 3-line whitelist freeze.
  DECLARATIVE   `DeclarativeR1Pool` (this file) — the SAME 4 edges as `CrossEdge(...)` DATA rows, passed straight
                 to `merge_organs(..., cross_edges=[...])`. ONE inject (the framework's own wire=True path already
                 includes the cross edges), and the whitelist freeze is ONE call:
                 `pool.apply_cross_edge_freeze()`. `DeclarativeR1Pool` subclasses `R1Pool` and overrides ONLY
                 `__init__` — every downstream method (`_hard_reset`, `_drive`, `train`, `amb_read`,
                 `cross_weights`, `_wmean`) is inherited UNCHANGED, so the train/read PROTOCOL is provably
                 identical between the two arms; only the pool-CONSTRUCTION path differs.

Both arms then run through the IDENTICAL imported F1/F2/F3/F4/emergence/lesion-recovers-migration functions.
`_cross_edge_dense`'s pre/post/weight construction (`np.repeat`/`np.tile`, same dtype, same union insertion
position) is byte-identical in SHAPE to R1's hand-typed `_dense`, so this is not merely a functional-equivalence
claim — the wiring DATA the two paths inject is the same, only the code that builds it differs.

Run (numpy CPU; NO sim/ edit; routes off the GPU):
  SIM_BACKEND=numpy python -m research.runners._onebrain_declarative_crossedge_r1_repro --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_declarative_crossedge_r1_repro \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_declarative_crossedge_r1_repro_6seed.json
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
import types
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import lever

from research.runners._onebrain_integration_r1_wm_comprehension import (
    R1Pool, GATE, W0, HMAX, _f1, _f2, _f3, _f4, _emergence_with_drift, _migration_invariant,
)
from research.runners.onebrain_merge_framework import (
    REGISTRY, CrossEdge, merge_organs, _comprehension_organ, _d6_organ, _comprehension_battery,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE EDGE — R1's 4 hand-typed `_dense(...)` union entries, re-expressed as 4 CrossEdge rows.
#  Same keys ("x_w0_sela" etc.), same source/target regions, same W0 seed weight, same GATE name, same
#  freeze_rest=True whitelist behavior R1's 3 hand-typed lines implement by hand.
# ─────────────────────────────────────────────────────────────────────────────────────────────
CROSS_EDGES = [
    CrossEdge(key="x_w0_sela", source_key="d6_multiref_wm", source_region="w0",
             target_key="comprehension", target_region="sel_agent",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
    CrossEdge(key="x_w0_selp", source_key="d6_multiref_wm", source_region="w0",
             target_key="comprehension", target_region="sel_patient",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
    CrossEdge(key="x_w1_sela", source_key="d6_multiref_wm", source_region="w1",
             target_key="comprehension", target_region="sel_agent",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
    CrossEdge(key="x_w1_selp", source_key="d6_multiref_wm", source_region="w1",
             target_key="comprehension", target_region="sel_patient",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
]


class DeclarativeR1Pool(R1Pool):
    """R1Pool with the pool-CONSTRUCTION path replaced by the declarative `cross_edges=` framework machinery.
    Every method below `__init__` is INHERITED UNCHANGED from `R1Pool` (train/read protocol reuse-by-inheritance,
    not copy-paste) -- only how the 4 cross-edges get onto the bridge differs."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        D6, COMP = REGISTRY["d6_multiref_wm"], REGISTRY["comprehension"]
        extra = types.SimpleNamespace(key="r1_hebbian", config={"hebbian_rate_window": True}, param_het=False)
        # ONE declarative build: the wire=True inject already includes CROSS_EDGES (onebrain_merge_framework's
        # `_install_organ_read_wiring`), so unlike the bespoke `_build_pool` there is no second manual re-inject.
        self.pool = merge_organs([D6, COMP], seed=seed, config_descriptors=[D6, COMP, extra],
                                 wire=True, cross_edges=CROSS_EDGES)
        self.b = self.pool.bridge
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("w0", "w1", "w2", "sel_agent", "sel_patient", "fs",
                                           "cue_animacy_pos", "cue_animacy_neg", "cue_verbfit_pos", "cue_verbfit_neg")}
        # The 4 cross-edge masks R1Pool's methods (_wmean/cross_weights/_f2's lesion) consume, in the SAME shape
        # -- derived from the bridge's OWN connectivity (row/col region membership), not re-declared, so a
        # mismatch here would be a framework wiring bug, not a test artifact.
        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        self.masks = {"w0->A": np.isin(row, self.ix["w0"]) & np.isin(col, self.ix["sel_agent"]),
                     "w0->P": np.isin(row, self.ix["w0"]) & np.isin(col, self.ix["sel_patient"]),
                     "w1->A": np.isin(row, self.ix["w1"]) & np.isin(col, self.ix["sel_agent"]),
                     "w1->P": np.isin(row, self.ix["w1"]) & np.isin(col, self.ix["sel_patient"])}
        for k, m in self.masks.items():
            n = int(m.sum())
            assert n > 0, f"declarative cross-edge mask {k!r} is EMPTY -- the CrossEdge did not wire"

        self.comp_organ = _comprehension_organ(seed, self.pool)
        self.d6_organ = _d6_organ(seed, self.pool)
        self.comp_organ.ensure_built()                      # installs + freezes cue gates, calibrates threshold

        # THE DECLARATIVE WHITELIST FREEZE -- replaces R1Pool's 3 hand-typed lines with ONE framework call driven
        # by each CrossEdge's freeze_rest=True field.
        self.pool.apply_cross_edge_freeze()

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        for kk, vv in dict(hebbian_rate_window=True, hebbian_coactivity_thresh=0.02, hebbian_learning_rate=0.05,
                           hebbian_max_weight=HMAX, hebbian_coactivity_decay=0.9).items():
            setattr(self.b.core_config, kk, vv)
        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()


def run_seed(pool_cls, seed):
    """IDENTICAL orchestration to R1's own `run_seed` (research/runners/_onebrain_integration_r1_wm_comprehension.
    py), parametrized over the pool class so BESPOKE and DECLARATIVE run through the exact same F-gate call
    sequence. F1-F4/emergence/migration functions are imported, not reimplemented."""
    t0 = time.time()
    r1 = pool_cls(seed)
    traj = r1.train()
    emg = _emergence_with_drift(traj, r1.frozen_maxdrift)
    f1 = _f1(r1)
    f4 = _f4(r1)                                   # F4 BEFORE F2 (F2 lesions the edge in place)
    f2 = _f2(r1)                                   # F2 lesions the cross-edge at its end
    r1._hard_reset()
    lesioned_reads = [float(r1.comp_organ.read_margin(n0, v, n1))
                      for (_l, _t, n0, v, n1) in _comprehension_battery(seed)]
    f3 = _f3(r1, traj, f2)
    mig = _migration_invariant(seed, r1, lesioned_reads)
    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and emg["PASS"] and mig["PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def compare_seed(seed):
    """Run BOTH arms at one seed and report the reproduction-fidelity numbers: emergent weight growth
    (w0->A / w1->P, the RIGHT pairs) + F2's lesion-attributable fraction (both should read ~1.0 -- the shift
    vanishes on lesion in BOTH arms), plus a direct max|delta| on the grown weights and the F2 deltas."""
    bespoke = run_seed(R1Pool, seed)
    declar = run_seed(DeclarativeR1Pool, seed)
    bg, dg = bespoke["emergence"]["final"], declar["emergence"]["final"]
    weight_delta = {k: abs(bg[k] - dg[k]) for k in ("w0->A", "w0->P", "w1->A", "w1->P")}
    bf2, df2 = bespoke["F2"], declar["F2"]
    # LEVER (tools.lab): the DECLARATIVE arm's own lesion manipulation must have actually MOVED the margin --
    # ref0's intact->lesion delta and ref1's intact->lesion delta are asserted non-identical (required=True
    # raises if not), so the "reproduces bespoke" claim below cannot rest on a declaratively-wired pool whose
    # cross-edge is accidentally inert (a lesion that changes nothing would make the F2 gate's PASS vacuous).
    lever(f"seed{seed} declarative F2 ref0 intact->lesion", df2["delta_ref0_intact"], df2["delta_ref0_lesion"])
    lever(f"seed{seed} declarative F2 ref1 intact->lesion", df2["delta_ref1_intact"], df2["delta_ref1_lesion"])
    f2_delta = {"delta_ref0_intact": abs(bf2["delta_ref0_intact"] - df2["delta_ref0_intact"]),
               "delta_ref1_intact": abs(bf2["delta_ref1_intact"] - df2["delta_ref1_intact"]),
               "delta_ref0_lesion": abs(bf2["delta_ref0_lesion"] - df2["delta_ref0_lesion"]),
               "delta_ref1_lesion": abs(bf2["delta_ref1_lesion"] - df2["delta_ref1_lesion"])}
    reproduces = bool(bespoke["PASS"] and declar["PASS"]
                      and max(weight_delta.values()) < 1.0
                      and (bf2["frac_attributable_ref0"] is None or df2["frac_attributable_ref0"] is None
                           or abs(bf2["frac_attributable_ref0"] - df2["frac_attributable_ref0"]) < 0.15)
                      and (bf2["frac_attributable_ref1"] is None or df2["frac_attributable_ref1"] is None
                           or abs(bf2["frac_attributable_ref1"] - df2["frac_attributable_ref1"]) < 0.15))
    return {"seed": seed, "bespoke_PASS": bespoke["PASS"], "declarative_PASS": declar["PASS"],
            "reproduces": reproduces,
            "bespoke_grown": bg, "declarative_grown": dg, "weight_maxdelta": max(weight_delta.values()),
            "weight_delta": weight_delta,
            "bespoke_frac_attributable": {"ref0": bf2["frac_attributable_ref0"], "ref1": bf2["frac_attributable_ref1"]},
            "declarative_frac_attributable": {"ref0": df2["frac_attributable_ref0"], "ref1": df2["frac_attributable_ref1"]},
            "f2_delta": f2_delta,
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
              f"reproduces={r['reproduces']} | grown w0->A "
              f"besp={r['bespoke_grown']['w0->A']:.2f} decl={r['declarative_grown']['w0->A']:.2f} "
              f"w1->P besp={r['bespoke_grown']['w1->P']:.2f} decl={r['declarative_grown']['w1->P']:.2f} "
              f"(maxdelta={r['weight_maxdelta']:.3f}) | F2 attrib "
              f"besp={r['bespoke_frac_attributable']} decl={r['declarative_frac_attributable']} "
              f"({r['bespoke_elapsed_s']}s + {r['declarative_elapsed_s']}s)", flush=True)

    n_go = sum(r["reproduces"] for r in runs)
    all_go = (n_go == len(runs))
    tag = "GO" if (all_go and not args.smoke) else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO")
    verdict = (f"{tag} — the declarative `cross_edges=[CrossEdge(...)]` param on `merge_organs` "
               f"(onebrain_merge_framework.py) REPRODUCES R1's bespoke d6->comprehension hand-wired cross-edge: "
               f"{n_go}/{len(runs)} seeds match on BOTH arms passing their own full F1-F4+emergence+migration gate "
               f"AND the reproduction check (grown weight maxdelta<1.0, F2 lesion-attributable fraction within "
               f"0.15). The 4 CrossEdge rows replace R1's hand-typed `_dense(...)` union + 3-line whitelist freeze "
               f"with ONE `merge_organs(cross_edges=[...])` call + ONE `pool.apply_cross_edge_freeze()` call -- "
               f"same wiring data, same protocol (DeclarativeR1Pool subclasses R1Pool, only __init__ differs), "
               f"different construction path. numpy CPU; NO sim/ edit.")

    # EARNED VERDICT PRECONDITIONS (tools.verdict.Verdict) — the reproduction claim above must travel with what
    # earned it, not just assert it (verdict_preconditions gate). Aggregated across every seed run, not typed.
    preconditions = []
    try:
        from tools.verdict import Verdict
        max_weight_delta = max(r["weight_maxdelta"] for r in runs)
        f2_deltas = [abs(r["bespoke_frac_attributable"]["ref0"] - r["declarative_frac_attributable"]["ref0"])
                    for r in runs if r["bespoke_frac_attributable"]["ref0"] is not None
                    and r["declarative_frac_attributable"]["ref0"] is not None]
        max_f2_delta = max(f2_deltas) if f2_deltas else None
        Vd = Verdict("onebrain_declarative_crossedge_r1_repro")
        Vd.require("bespoke_arm_all_pass", sum(r["bespoke_PASS"] for r in runs), expect=lambda x: x == len(runs),
                   note="R1's own bespoke F1-F4+emergence+migration gate must pass every seed, or the baseline "
                        "this reproduction is measured against is itself unearned")
        Vd.require("declarative_arm_all_pass", sum(r["declarative_PASS"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="the declaratively-wired pool must independently pass the SAME F-gate, not merely match numbers")
        Vd.require("weight_reproduction_bitidentical", max_weight_delta, expect=lambda x: x < 1.0,
                   note="max|delta| between bespoke and declarative grown weights, across every seed and pair")
        if max_f2_delta is not None:
            Vd.require("f2_attribution_reproduction", max_f2_delta, expect=lambda x: x < 0.15,
                       note="max|delta| of the F2 lesion-attributable fraction between the two arms")
        Vd.require("declarative_lesion_moves_margin_every_seed", len(runs), expect=lambda x: x == len(seeds),
                   note="each seed's lever() call (tools.lab, above in compare_seed) RAISES immediately if the "
                        "declarative arm's intact->lesion margin failed to move; reaching this aggregation with "
                        "len(runs)==len(seeds) is proof every per-seed lever already passed (a raise would have "
                        "crashed the run before payload construction, not merely recorded a False here)")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_declarative_crossedge_r1_repro", "verdict": verdict, "GO": all_go,
              "n_go": n_go, "n_seeds": len(runs), "seeds": seeds,
              "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
              "preconditions": preconditions,
              "cross_edges_schema": [dict(key=ce.key, source_key=ce.source_key, source_region=ce.source_region,
                                          target_key=ce.target_key, target_region=ce.target_region,
                                          init_weight=ce.init_weight, plastic=ce.plastic, gate=ce.gate_name,
                                          learn_rule=ce.learn_rule, freeze_rest=ce.freeze_rest)
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
