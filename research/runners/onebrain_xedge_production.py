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


def xedge_learn_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_LEARN` in {1,true,yes,on} -> PART 2 LIVE-LEARNING: the cross-edge starts near-zero
    (W0=0.05) and GROWS from an IN-BRAIN, self-supervised credit signal (comprehension's OWN confident sel
    resolution drives teach_*, three-factor DA-gated, bounded by stdp_w_max) over a multi-turn sequence -- NOT a
    frozen pre-grown host-schedule weight. Default OFF (unset) -> the PART-1 FROZEN host-schedule edge. Only takes
    effect when `BRAIN_ONEBRAIN_XEDGE` is also on."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_LEARN")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def grow_live_selfsupervised(p, n_turns: int = 80, conf: float = 0.02):
    """PART 2 LIVE-LEARNING. Grow the cross-edge from W0=0.05 via an IN-BRAIN, SELF-SUPERVISED credit signal: no
    host ground-truth label anywhere. Per turn: hold a WM candidate pool, present a role-resolving discourse
    (agent- or patient-leaning content), READ the brain's OWN sel resolution (`amb_read` margin), and IFF the
    comprehension is CONFIDENT (|margin| > `conf`) drive teach_{the-role-the-brain-resolved} -- the DA-coincidence
    machinery then grows w{held}->sel_{resolved}. Interleaves agent-discourse-while-holding-p_agent and
    patient-discourse-while-holding-p_patient so BOTH role edges learn from use. Freezes the candidate gate at the
    end (no further growth in production reads). Returns the weight trajectory. `p` is an R3Pool (gate open, edge
    at 0.05). Credit VALUE = comprehension's own spiking verdict; the WM->role association is the LEARNED content."""
    from research.runners._onebrain_integration_r2_threefactor_selforganized import CUE_PA, GATE, _role_assignment
    pa, pp, pc = _role_assignment(p.seed)
    ix = p.ix
    AG_KEYS = [("cue_animacy_pos", CUE_PA), ("cue_verbfit_pos", CUE_PA)]           # amb_read: string keys
    PA_KEYS = [("cue_animacy_neg", CUE_PA), ("cue_verbfit_neg", CUE_PA)]
    AG_IX = [(ix["cue_animacy_pos"], CUE_PA), (ix["cue_verbfit_pos"], CUE_PA)]     # _episode: index arrays
    PA_IX = [(ix["cue_animacy_neg"], CUE_PA), (ix["cue_verbfit_neg"], CUE_PA)]
    traj = [dict(turn=0, **p.cross_weights())]
    n_credited = 0
    for t in range(n_turns):
        # alternate the discourse: even turns = agent-content while holding p_agent; odd = patient while holding p_patient
        if t % 2 == 0:
            hold, cue_keys, cue_ix = pa, AG_KEYS, AG_IX
        else:
            hold, cue_keys, cue_ix = pp, PA_KEYS, PA_IX
        margin = float(p.amb_read(hold, cue_keys, band=True)["margin"])            # the brain's OWN resolution
        if abs(margin) > conf:                                                     # confident comprehension = credit
            teach = "teach_agent" if margin > 0 else "teach_patient"
            p._episode(hold, cue_ix, credited=True, teach_pool=teach)              # self-supervised credited episode
            n_credited += 1
        if (t + 1) % 20 == 0 or t == n_turns - 1:
            traj.append(dict(turn=t + 1, **p.cross_weights()))
    p.b.set_plasticity_gate(GATE, 0.0)                                             # freeze after live learning
    traj[-1]["n_credited"] = n_credited
    return traj


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
        self.learned = False        # True when the edge was GROWN LIVE (self-supervised, PART 2) vs frozen host-grown

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
        # by first-use the webapp has already fixed the backend, so setdefault is a no-op). Importing it also
        # calibrates the DA gain the LIVE-LEARNING path below relies on (it uses the base R3Pool).
        from research.runners._onebrain_integration_r3v3_functional_drive import R3v3Pool
        from research.runners._onebrain_integration_r2_threefactor_selforganized import _role_assignment

        if xedge_learn_enabled():
            # PART 2 LIVE-LEARNING: start the edge at W0=0.05 (R3Pool, gate OPEN) and GROW it from the in-brain
            # self-supervised credit signal over a multi-turn sequence, then freeze. Emergent, not pre-grown.
            from research.runners._onebrain_integration_r3_spiking_dopamine_credit import R3Pool
            p = R3Pool(self.seed, mode="intact")
            self.grow_traj = grow_live_selfsupervised(p)
            self.learned = True
        else:
            # PART 1 FROZEN: grow via R3-v3's host-schedule credit-gated training, freeze on return.
            p = R3v3Pool(self.seed, mode="intact")
            self.grow_traj = p.train()             # grows the cross-edge; freezes the candidate gate on return
            self.learned = False
        p_agent, p_patient, p_ctrl = _role_assignment(self.seed)
        self.role = {"p_agent": p_agent, "p_patient": p_patient, "p_ctrl": p_ctrl}
        self._r3pool = p            # keep the R3(v3)Pool: its VALIDATED amb_read is the WM-resolved role read
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
        # WM-RESOLVED-ROLE read (closes the sub-decision caveat): bind R3-v3's VALIDATED balanced `amb_read` (the F2
        # instrument) + the balanced (content-cancelled) cue spec + the control-hold pool. The comprehension organ
        # calls these off `self._shared` to resolve an ambiguous role from the held WM referent -- reusing the proven
        # read rather than reimplementing it (a hand-rolled balanced read was NOT actually balanced).
        from research.runners._onebrain_integration_r2_threefactor_selforganized import AMBIG_PA, BASE_POOL
        self.pool.xedge_amb_read = p.amb_read
        self.pool.xedge_balanced_cues = [("cue_animacy_pos", AMBIG_PA), ("cue_animacy_neg", AMBIG_PA)]
        self.pool.xedge_base_pool = BASE_POOL
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


def _selftest_livelearn(pool, seed):
    """PART 2 end-to-end: the LIVE-LEARNED edge (grown from in-brain self-supervised credit) (1) grew from W0=0.05,
    selectively, bounded by stdp_w_max (F3); (2) CLOSES the caveat -- the real production `repair_target` role
    DECISION flips with the held WM referent (p_agent vs p_patient candidate pool) and reverts under cross-edge
    lesion. Returns the growth + decision-flip measurements."""
    import numpy as np
    from sim.backend import to_host
    from research.runners._onebrain_integration_r2_threefactor_selforganized import HMAX
    from research.runners._spiking_comprehension_monitor_derisk import build_battery

    pa, pp = pool.role["p_agent"], pool.role["p_patient"]
    w = pool.cross_weights
    w0 = pool.grow_traj[0]
    grew_agent = w[f"{pa}->A"] > 0.5 and w0[f"{pa}->A"] <= 0.06
    grew_patient = w[f"{pp}->P"] > 0.5 and w0[f"{pp}->P"] <= 0.06
    bounded = all(v <= HMAX + 1e-6 for v in w.values())

    corg = pool.comp_organ
    corg.ensure_built()
    sh = pool.pool
    ambig = [it for it in build_battery(seed, n_per_cond=3) if it[0] == 0 and "ambig" in it[1]][:5]

    def roles():
        rows = []
        for (lab, tag, n0, v, n1) in ambig:
            sh.xedge_focus = pa
            ra = corg.repair_target(f"{n0} {v} {n1}")
            sh.xedge_focus = pp
            rp = corg.repair_target(f"{n0} {v} {n1}")
            ra_r = ra and ra.get("role"); rp_r = rp and rp.get("role")
            rows.append({"item": f"{n0}/{v}/{n1}", "tag": tag, "role_p_agent": ra_r, "role_p_patient": rp_r,
                         "flip": bool(ra_r != rp_r and ra_r in ("agent", "patient") and rp_r in ("agent", "patient"))})
        return rows

    intact = roles()
    b = pool.bridge
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    for k in pool.masks:
        data[pool.masks[k]] = 0.0
    b.cp_connections.data = pool.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
    lesioned = roles()

    flips_i = sum(r["flip"] for r in intact)
    flips_l = sum(r["flip"] for r in lesioned)
    return {"role": pool.role, "learned": pool.learned, "grow_traj": pool.grow_traj,
            "final_weights": w, "grew_both": bool(grew_agent and grew_patient), "bounded_F3": bool(bounded),
            "intact_roles": intact, "lesioned_roles": lesioned,
            "decision_flips_intact": flips_i, "decision_flips_lesioned": flips_l,
            "caveat_closed": bool(flips_i > 0 and flips_l == 0),
            "GO": bool(grew_agent and grew_patient and bounded and flips_i > 0 and flips_l == 0)}


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="build+grow the FROZEN pool and self-verify load-bearing")
    ap.add_argument("--verify-live", action="store_true",
                    help="PART 2: build the LIVE-LEARNED pool (edge grows from in-brain credit) + verify caveat close")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    global _POOL
    if args.verify_live:
        os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"
        results = []
        for s in seeds:
            _POOL = None
            from research.runners import comprehension_production_organ as _CO
            _CO._ORGAN = None
            pool = get_xedge_pool(s)
            assert pool is not None and pool.ok, "live-learn pool failed to build"
            sv = _selftest_livelearn(pool, s)
            traj = [{k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} for d in sv["grow_traj"]]
            print(f"[seed {s}] learned={sv['learned']} role={sv['role']} grew_both={sv['grew_both']} "
                  f"bounded_F3={sv['bounded_F3']} flips(intact={sv['decision_flips_intact']}/5 "
                  f"lesioned={sv['decision_flips_lesioned']}/5) caveat_closed={sv['caveat_closed']} GO={sv['GO']}",
                  flush=True)
            print(f"    grow: {traj}")
            for r in sv["intact_roles"]:
                print(f"    intact  {r['tag']:14s} {r['item']:18s} p_agent->{r['role_p_agent']} "
                      f"p_patient->{r['role_p_patient']} {'FLIP' if r['flip'] else ''}")
            results.append({"seed": s, "selftest": sv})
        payload = {"probe": "onebrain_xedge_production_live_learning", "seeds": seeds,
                   "backend": os.environ.get("SIM_BACKEND", "numpy"), "results": results,
                   "n_go": sum(r["selftest"]["GO"] for r in results),
                   "note": ("PART 2: the cross-edge GROWS from W0=0.05 via an IN-BRAIN self-supervised credit signal "
                            "(comprehension's own confident sel resolution drives teach_*, three-factor DA-gated, "
                            "bounded by stdp_w_max) -- NOT a frozen pre-grown weight -- and CLOSES the sub-decision "
                            "caveat: the real production repair role DECISION flips with the held WM referent, "
                            "lesion-attributable. Semantic referent->pool binding stays a declared residual.")}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
            print(f"wrote {args.out}", flush=True)
        return 0

    results = []
    for s in seeds:
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
