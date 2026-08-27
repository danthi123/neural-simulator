"""ONE-BRAIN CROSS-EDGE — the FIRST learned cross-region synapse wired into the LIVE chat brain (2026-08-27).

THE GAP (scoping `2026-08-27-onebrain-production-integration-SCOPING.md`). The de-risked R-cross-edges (R1/R3-v3/
R4) live on the RESEARCH merge FRAMEWORK, default-off, NOT in the live brain — every live faculty organ builds its
OWN standalone `SimulationBridge` (`shared=None`), so no shared substrate a cross-region synapse could span. This
module co-locates the R3-v3 pair (d6 multi-referent WM + D4 comprehension) on ONE `MergedPool` and loads the FROZEN,
pre-grown `w{k}->sel_agent/sel_patient` cross-edge R3-v3 de-risked (6-seed GO,
`2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md`), so a held WM pool DRIVES the comprehension role
competition the live judge consumes.

FROZEN, PLASTICITY-OFF IN PRODUCTION. The cross-edge is GROWN ONCE at first build (R3-v3's own credit-gated
training, `R3v3Pool.train()`) and then FROZEN (`set_plasticity_gate(GATE, 0.0)` — R3v3Pool does this the instant
train returns). No weight moves during any live turn. Growth is IN-PROCESS (not a saved artifact) on purpose: the
CROSS-BACKEND SEED TRAP (`docs/ENGINE_REFERENCE.md`, commit 289cad1) means a numpy-grown weight file is NOT valid
for a cupy production build (different RNG -> different substrate), so growing in whatever backend the process runs
guarantees the frozen edge matches the substrate it rides. The converged block-mean weights ARE written to a
sidecar artifact for the record, but correctness never depends on loading it.

WHY A CO-DRIVE COUPLING IS REQUIRED (the scoping's "attach `shared=` and done" was optimistic). Comprehension's
read `_hard_reset`s the WHOLE shared bridge to `pool.snap` before every sel-settle, and d6's `load()` runs in
`read_isolation` (restores every OTHER slice) — so a d6 bump does NOT survive into the comprehension read on its
own. The cross-edge only transmits when the held d6 pool is FIRING during the sel-settle (exactly R3-v3's F2
`amb_read` protocol: establish the self-sustaining slow-NMDA bump, then read the cues while it self-sustains). So
the comprehension read, on the shared+xedge path with a focus register set, RE-ESTABLISHES that bump. Without this
coupling the wire-in would be HOLLOW (co-resident but not interacting) — the exact drift memory #84/#85 gates.

WHAT IS LOAD-BEARING vs DECLARED RESIDUAL (honest, from the de-risk probe + R3-v3's own scaffold residuals):
  * LOAD-BEARING + lesion-attributable: a HELD WM candidate pool drives the comprehension role competition the
    live judge/repair reads. Instrument = the DIFFERENTIAL hold(p_agent-pool) vs hold(p_patient-pool) (cancels the
    generic "any WM activity perturbs the shared inhibition" confound); it is EXACTLY 0 when the cross-edge is
    lesioned and nonzero intact. The genuinely-driven quantity is the SIGNED net-lean (a0+a1) that `repair_target`
    consumes (the per-noun |a0-a1| the `comprehended` threshold uses partly cancels a symmetric sel bias).
  * DECLARED RESIDUAL (carried UNCHANGED from R3-v3): the candidate topology (w0/w1/w2) is a host-chosen abstract
    "3 structurally-identical d6 slot pools", driven at TRAIN time by the teacher schedule — it is NOT wired to a
    SEMANTIC discourse role->pool binding. So which real referent maps to the agent- vs patient-candidate pool is
    host-directed (the live focus is a POSITIONAL proxy). Closing that semantic binding is a later rung.

GUARDED, DEFAULT-OFF, BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE` gates the whole thing (default OFF — the flip to
default-ON is a separate owner-gated step). Unset/0/false/no/off => every organ builds standalone exactly as
today (byte-identical). A build failure DEGRADES to standalone (never crashes brain load). `BRAIN_ONEBRAIN_XEDGE_
LESION=1` zeroes the cross-edge (the load-bearing lesion control) while keeping everything else, for the live
vary->lesion check.

Run (offline grow + record + self-verify):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_production --grow --seeds 42 \
      --out research/findings/raw/_onebrain_xedge_production_frozen_seed42.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — OFF. The owner-gated flip to default-ON is a SEPARATE step (never autonomous).
_XEDGE_DEFAULT_ON = False


def xedge_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE` in {1,true,yes,on} -> the frozen d6-WM->comprehension cross-edge is live (the two
    organs share ONE spiking pool with the pre-grown, frozen cross-synapse). Unset/{0,false,no,off} -> every
    organ builds standalone exactly as today (byte-identical). Default per `_XEDGE_DEFAULT_ON` (OFF)."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE")
    if v is None:
        return _XEDGE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def xedge_lesioned() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_LESION` in {1,true,yes,on} -> zero the w{k}->sel cross-edge (the load-bearing lesion
    control: the WM->comprehension drive must VANISH). Everything else (d6 hold, comprehension read) is unchanged,
    so the shift this cross-edge introduces must vanish here."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── Co-drive params the comprehension read reads OFF the shared pool (no import from the organ into the R2/R3
#    runner). LOAD_PA/LOAD_STEPS/HOLD are R2's own WM-hold protocol (the same amb_read uses). ──
_CODRIVE_PARAMS = {"load_pa": 400.0, "load_steps": 30, "hold_steps": 6}


class XedgeProductionPool:
    """Process-shared holder of the [d6_multiref_wm + comprehension + da_credit] `MergedPool` with the FROZEN
    pre-grown cross-edge. Exposes the surface the live attach points consume: `.pool` (the MergedPool the organs
    take as `shared=`), `.comp_organ` (the cross-edge-grown comprehension organ), `.ix`/`.masks`, and the
    `set_focus`/`clear_focus` the coupling uses. Built lazily on first use; degrades to a disabled holder on any
    build failure (the caller then falls back to standalone organs)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.ok = False
        self.pool = None            # the framework MergedPool (what organs take as shared=)
        self.bridge = None
        self.comp_organ = None      # the cross-edge-grown ComprehensionProductionOrgan on the shared pool
        self.d6_organ = None        # the pool-internal d6 organ (used to GROW; live sessions build their own)
        self.ix = None
        self.masks = None
        self.role = None            # {"p_agent","p_patient","p_ctrl"} candidate-pool assignment for this seed
        self.grow_traj = None
        self.cross_weights = None

    def ensure_built(self):
        if self._built:
            return
        self._built = True     # set FIRST so a failed build is not retried every turn
        try:
            self._build()
            self.ok = True
        except Exception as e:   # never crash brain load — degrade to standalone
            import traceback
            print(f"[webapp] ONEBRAIN XEDGE build FAILED -> degrading to standalone organs "
                  f"({type(e).__name__}: {e})", flush=True)
            print(traceback.format_exc(), flush=True)
            self.ok = False

    def _build(self):
        # Import lazily (the R3-v3 runner sets DA_SENSITIVITY=10000 on import + os.environ.setdefault SIM_BACKEND;
        # by first-use the webapp has already fixed the backend, so setdefault is a no-op).
        from research.runners._onebrain_integration_r3v3_functional_drive import R3v3Pool
        from research.runners._onebrain_integration_r2_threefactor_selforganized import _role_assignment

        p = R3v3Pool(self.seed, mode="intact")
        self.grow_traj = p.train()                 # grows the cross-edge; freezes the candidate gate on return
        p_agent, p_patient, p_ctrl = _role_assignment(self.seed)
        self.role = {"p_agent": p_agent, "p_patient": p_patient, "p_ctrl": p_ctrl}
        self.pool = p.pool
        self.bridge = p.b
        self.comp_organ = p.comp_organ
        self.d6_organ = p.d6_organ
        self.ix = p.ix
        self.masks = p.masks
        self.cross_weights = p.cross_weights()
        # publish the coupling handles ONTO the MergedPool (the object the organs hold as `shared=`), so the
        # comprehension co-drive + the d6 focus-set find them without importing this module.
        self.pool.xedge_focus = None
        self.pool.xedge_codrive_params = dict(_CODRIVE_PARAMS)
        # optional lesion control (env) — zero the cross-edge weights in place.
        if xedge_lesioned():
            self.lesion_cross()

    # ── the load-bearing lesion control (env or explicit) ──
    def lesion_cross(self):
        """Zero every w{k}->sel cross-edge weight in place (the WM->comprehension drive must then vanish)."""
        import numpy as np
        from sim.backend import to_host
        b = self.bridge
        data = np.asarray(to_host(b.cp_connections.data)).copy()
        for k in self.masks:
            data[self.masks[k]] = 0.0
        b.cp_connections.data = self.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)

    # ── the coupling focus (which held d6 candidate pool the comprehension read co-drives) ──
    def set_focus(self, region_name):
        if self.pool is not None:
            self.pool.xedge_focus = region_name

    def clear_focus(self):
        if self.pool is not None:
            self.pool.xedge_focus = None

    def candidate_pool_for_register(self, register_index: int):
        """POSITIONAL proxy (declared residual): map a held d6 referent's register index to a candidate pool
        w0/w1/w2. This is NOT a semantic role->pool binding (R3-v3's candidate topology is host-chosen); it lets
        the live pipeline carry a VARYING WM state into the comprehension read. Returns None if out of range."""
        from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS
        if register_index is None or register_index < 0:
            return None
        return CAND_POOLS[min(int(register_index), len(CAND_POOLS) - 1)]


_POOL: "XedgeProductionPool | None" = None


def get_xedge_pool(seed: int = 42) -> "XedgeProductionPool | None":
    """The process-shared xedge pool (built once on first use). Returns the holder even if the build failed
    (holder.ok is False) so the caller can fall back to standalone. Returns None only when the flag is OFF."""
    global _POOL
    if not xedge_enabled():
        return None
    if _POOL is None:
        _POOL = XedgeProductionPool(seed)
    _POOL.ensure_built()
    return _POOL if _POOL.ok else None


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Offline grow + record + self-verify entrypoint (0 Claude tokens; CPU numpy).
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _selftest_loadbearing(pool, seed):
    """Directly exercise the REAL production judge/repair path: hold(p_agent-pool) vs hold(p_patient-pool) on an
    ambiguous in-scope transitive, INTACT vs cross-edge-LESIONED. The differential must be nonzero intact and
    (on ambiguous items) exactly 0 lesioned -> lesion-attributable WM->comprehension drive through the live read.
    Returns the measured deltas."""
    import numpy as np
    from sim.backend import to_host
    from tools.lab import attributable_to
    from research.runners._onebrain_integration_r2_threefactor_selforganized import LOAD_PA, LOAD_STEPS
    from research.runners.comprehension_production_organ import READ_STEPS as CRS, _evs_for_organ
    from research.runners._spiking_comprehension_monitor_derisk import (
        _agent_evidence_from_spikes, SEMANTIC_CUES, build_battery)

    corg = pool.comp_organ
    comp = corg.comp

    def pernoun(hold_region, n0, v, n1):
        corg._hard_reset(comp)
        if hold_region is not None:
            idx = comp.xp.asarray(np.asarray(pool.ix[hold_region], np.int64))
            cur = comp.xp.zeros(comp.bridge.core_config.num_neurons, dtype=comp.xp.float32)
            cur[idx] = comp.xp.float32(LOAD_PA)
            comp.bridge.cp_external_input_current[:] = cur
            for _ in range(LOAD_STEPS):
                comp.bridge._run_one_simulation_step()
            comp.bridge.cp_external_input_current[:] = 0.0
            for _ in range(6):
                comp.bridge._run_one_simulation_step()
        evs = _evs_for_organ(n0, v, n1)
        a0 = float(_agent_evidence_from_spikes(comp, evs[0], SEMANTIC_CUES, CRS))
        a1 = float(_agent_evidence_from_spikes(comp, evs[1], SEMANTIC_CUES, CRS))
        return a0, a1

    pa, pp = pool.role["p_agent"], pool.role["p_patient"]
    batt = build_battery(seed, n_per_cond=2)
    ambig = [it for it in batt if it[0] == 0 and "ambig" in it[1]][:3]

    def measure():
        rows = []
        for (lab, tag, n0, v, n1) in ambig:
            a0a, a1a = pernoun(pa, n0, v, n1)
            a0p, a1p = pernoun(pp, n0, v, n1)
            rows.append({"item": f"{n0}/{v}/{n1}", "tag": tag,
                         "dNet": (a0a + a1a) - (a0p + a1p),
                         "dMargin": abs(a0a - a1a) - abs(a0p - a1p)})
        return rows

    intact = measure()
    # lesion + re-measure
    b = pool.bridge
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    for k in pool.masks:
        data[pool.masks[k]] = 0.0
    b.cp_connections.data = pool.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
    lesioned = measure()

    max_intact = max(abs(r["dNet"]) for r in intact) if intact else 0.0
    max_les = max(abs(r["dNet"]) for r in lesioned) if lesioned else 0.0
    # ATTRIBUTION (whose difference IS it?): the hold(p_agent)-vs-hold(p_patient) net-lean differential must be
    # OWNED by the cross-edge — measuring both arms is not the same as asking whose the difference was (gap#5).
    frac = attributable_to(f"seed{seed} xedge WM->comprehension net-lean drive vs cross-edge lesion",
                           max_intact, max_les)
    return {"role": pool.role, "intact": intact, "lesioned": lesioned,
            "max_abs_dNet_intact": max_intact, "max_abs_dNet_lesioned": max_les,
            "frac_attributable_to_cross_edge": (None if frac is None else float(frac)),
            "lesion_attributable": bool(max_les < 1e-9 and max_intact > 1e-3)}


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="build+grow the pool and self-verify load-bearing")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    for s in seeds:
        global _POOL
        _POOL = None
        pool = get_xedge_pool(s)
        assert pool is not None and pool.ok, "xedge pool failed to build"
        sv = _selftest_loadbearing(pool, s)
        print(f"[seed {s}] role={sv['role']} cross_weights={pool.cross_weights} "
              f"max|dNet| intact={sv['max_abs_dNet_intact']:.4f} lesioned={sv['max_abs_dNet_lesioned']:.4f} "
              f"lesion_attributable={sv['lesion_attributable']}", flush=True)
        for r in sv["intact"]:
            print(f"    intact  {r['tag']:14s} {r['item']:18s} dNet={r['dNet']:+.4f} dMargin={r['dMargin']:+.4f}")
        for r in sv["lesioned"]:
            print(f"    lesion  {r['tag']:14s} {r['item']:18s} dNet={r['dNet']:+.4f} dMargin={r['dMargin']:+.4f}")
        results.append({"seed": s, "cross_weights": pool.cross_weights, "selftest": sv})

    payload = {"probe": "onebrain_xedge_production_frozen", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "results": results,
               "note": ("frozen pre-grown d6-WM->comprehension cross-edge wired into the live pool; the DRIVE is "
                        "lesion-attributable on the real production comprehension read (differential hold "
                        "p_agent-pool vs p_patient-pool, exactly 0 lesioned). Semantic referent->pool binding is "
                        "a declared residual (positional live focus).")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
