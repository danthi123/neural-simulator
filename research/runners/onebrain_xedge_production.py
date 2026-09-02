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

GUARDED, DEFAULT-ON since 2026-08-28 (commit fe1911f2f; corrected -- this paragraph previously said "default OFF
— the flip to default-ON is a separate owner-gated step", stale). `BRAIN_ONEBRAIN_XEDGE` gates the whole thing;
explicit 0/false/no/off => every organ builds standalone exactly as today (byte-identical-off). A build failure
DEGRADES to standalone (never crashes brain load). `BRAIN_ONEBRAIN_XEDGE_LESION=1` zeroes the cross-edge (the
load-bearing lesion control) while keeping everything else, for the live vary->lesion check.

Run (offline grow + record + self-verify):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_production --grow --seeds 42 \
      --out research/findings/raw/_onebrain_xedge_production_frozen_seed42.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — ON (flipped 2026-08-28). The owner PRE-AUTHORIZED this autonomous flip on a genuine
# non-hollow GO; `_xedge_flip_production_verify` returned FLIP_VERIFY_GO=True (arm_A byte-identical-off 4/4,
# arm_B n_visible_grown_focus=4 + n_hollow=0 + all_seeds_lesion_revert, arm_C no-regression PASS). The
# `BRAIN_ONEBRAIN_XEDGE=0` env escape hatch is preserved (explicit-off => byte-identical to pre-flip). Revert
# = set these back to False. See finding 2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.
_XEDGE_DEFAULT_ON = True
_XEDGE_LEARN_DEFAULT_ON = True   # same flip: PART-2 per-turn live-learning cross-edge default-ON (arm_B on_learn GO)


def xedge_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE` in {1,true,yes,on} -> the frozen d6-WM->comprehension cross-edge is live (the two
    organs share ONE spiking pool with the pre-grown, frozen cross-synapse). Unset/{0,false,no,off} -> every
    organ builds standalone exactly as today (byte-identical). Default per `_XEDGE_DEFAULT_ON` (ON, since the
    2026-08-28 flip; corrected -- previously documented as OFF here)."""
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
    frozen pre-grown host-schedule weight. Default ON since the 2026-08-28 flip (`_XEDGE_LEARN_DEFAULT_ON`, above;
    corrected -- previously documented as "Default OFF (unset) -> the PART-1 FROZEN host-schedule edge", stale);
    `BRAIN_ONEBRAIN_XEDGE_LEARN=0` is the byte-identical escape to the PART-1 FROZEN host-schedule edge. Only
    takes effect when `BRAIN_ONEBRAIN_XEDGE` is also on."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_LEARN")
    if v is None:
        return _XEDGE_LEARN_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


# PART 3 PER-TURN LIVE PLASTICITY — module toggle. When True (the LIVE production intent), the learn build LEAVES
# the cross-edge at W0=0.05 with the gate OPEN (no build-time curriculum, no freeze), so it grows PER-TURN from
# in-brain credit during real chat. When False, the learn build pre-grows over the PART-2 build curriculum then
# freezes (reproduces `2026-08-27-...-live-learning-GO`); the PART-2 `--verify-live` entrypoint sets it False.
_LIVE_PER_TURN = True


def live_per_turn_enabled() -> bool:
    """Whether the LEARN build grows the cross-edge PER-TURN during live chat (True, production intent) vs the
    PART-2 build-time curriculum (False). Only meaningful when `BRAIN_ONEBRAIN_XEDGE_LEARN` is also on."""
    return bool(_LIVE_PER_TURN)


def set_live_per_turn(on: bool) -> None:
    """Select per-turn live growth (True) vs the PART-2 build curriculum (False) for the LEARN build. Must be set
    BEFORE the first `get_xedge_pool` (the pool is built once + cached)."""
    global _LIVE_PER_TURN
    _LIVE_PER_TURN = bool(on)


# discourse cue directions (host/teacher-scaffold, declared residual #1 -- WHICH discourse is presented). The
# in-brain part is the credit VALUE + teach DIRECTION, read off the brain's OWN confident amb_read below.
def _cue_dirs(ix):
    from research.runners._onebrain_integration_r2_threefactor_selforganized import CUE_PA
    AG_KEYS = [("cue_animacy_pos", CUE_PA), ("cue_verbfit_pos", CUE_PA)]           # amb_read: string keys
    PA_KEYS = [("cue_animacy_neg", CUE_PA), ("cue_verbfit_neg", CUE_PA)]
    AG_IX = [(ix["cue_animacy_pos"], CUE_PA), (ix["cue_verbfit_pos"], CUE_PA)]     # _episode: index arrays
    PA_IX = [(ix["cue_animacy_neg"], CUE_PA), (ix["cue_verbfit_neg"], CUE_PA)]
    return AG_KEYS, PA_KEYS, AG_IX, PA_IX


def _credit_turn_step(p, hold, cue_keys, cue_ix, conf: float, gate=None):
    """ONE in-brain self-supervised credit step (the atom of PART 2's curriculum AND PART 3's per-turn chat). Hold
    `hold`, present the role-resolving discourse `cue_keys`, READ the brain's OWN sel resolution (`amb_read`
    margin), and IFF CONFIDENT (|margin| > `conf`) drive teach_{the-role-the-brain-resolved} for ONE credited
    `_episode` -- the DA-coincidence machinery then grows w{hold}->sel_{resolved} (three-factor, bounded by
    stdp_w_max). NO host ground-truth label: the credit VALUE + DIRECTION are both the substrate's own spikes.
    When `gate` is given (PART 3 per-turn), OPEN it for exactly this one credited episode then RE-FREEZE, so every
    production READ is a frozen forward pass (the sel-WTA margin a live read consumes is unreliable while the gate
    is open). `gate=None` (PART 2 curriculum) leaves the gate as the caller manages it. The confidence `amb_read`
    is always a frozen read (it drives no teach pool -> no credit). Returns (credited: bool, teach: str|None,
    margin: float)."""
    margin = float(p.amb_read(hold, cue_keys, band=True)["margin"])                # the brain's OWN resolution (frozen)
    if abs(margin) > conf:                                                         # confident comprehension = credit
        teach = "teach_agent" if margin > 0 else "teach_patient"
        if gate is not None:
            p.b.set_plasticity_gate(gate, 1.0)                                     # open for exactly ONE credited step
        p._episode(hold, cue_ix, credited=True, teach_pool=teach)                  # self-supervised credited episode
        if gate is not None:
            p.b.set_plasticity_gate(gate, 0.0)                                     # re-freeze -> reads stay frozen
        return True, teach, margin
    return False, None, margin


def grow_live_selfsupervised(p, n_turns: int = 80, conf: float = 0.02):
    """PART 2 LIVE-LEARNING (build-time curriculum). Grow the cross-edge from W0=0.05 via an IN-BRAIN,
    SELF-SUPERVISED credit signal: no host ground-truth label anywhere. Per turn: hold a WM candidate pool,
    present a role-resolving discourse (agent- or patient-leaning content), and run `_credit_turn_step` (READ the
    brain's OWN sel resolution, IFF confident credit teach_{resolved}). Interleaves agent-discourse-while-holding-
    p_agent and patient-discourse-while-holding-p_patient so BOTH role edges learn from use. Freezes the candidate
    gate at the end (no further growth in production reads). Returns the weight trajectory. `p` is an R3Pool (gate
    open, edge at 0.05). PART 3 fires the SAME `_credit_turn_step` ONCE per LIVE chat turn instead (no freeze)."""
    from research.runners._onebrain_integration_r2_threefactor_selforganized import GATE, _role_assignment
    pa, pp, pc = _role_assignment(p.seed)
    AG_KEYS, PA_KEYS, AG_IX, PA_IX = _cue_dirs(p.ix)
    traj = [dict(turn=0, **p.cross_weights())]
    n_credited = 0
    for t in range(n_turns):
        # alternate the discourse: even turns = agent-content while holding p_agent; odd = patient while holding p_patient
        if t % 2 == 0:
            hold, cue_keys, cue_ix = pa, AG_KEYS, AG_IX
        else:
            hold, cue_keys, cue_ix = pp, PA_KEYS, PA_IX
        credited, _teach, _margin = _credit_turn_step(p, hold, cue_keys, cue_ix, conf)
        if credited:
            n_credited += 1
        if (t + 1) % 20 == 0 or t == n_turns - 1:
            traj.append(dict(turn=t + 1, **p.cross_weights()))
    p.b.set_plasticity_gate(GATE, 0.0)                                             # freeze after live learning
    traj[-1]["n_credited"] = n_credited
    return traj


# ── Co-drive params the comprehension read reads OFF the shared pool (no import from the organ into the R2/R3
#    runner). LOAD_PA/LOAD_STEPS/HOLD are R2's own WM-hold protocol (the same amb_read uses). ──
_CODRIVE_PARAMS = {"load_pa": 400.0, "load_steps": 30, "hold_steps": 6}

# 2026-08-27 CROSS-SESSION xedge_focus LEAK FIX (research/FAILURE_LOG.md). `credit_live_turn`/
# `credit_live_turn_from_comprehension` used to read `self.pool.xedge_focus`, a single mutable attribute on the
# ONE process-shared MergedPool -- written by whichever session's d6 organ last held >=2 referents, read (and
# credited against!) by EVERY session's live-plasticity hook thereafter. `focus`/`wm_focus` below are now EXPLICIT
# per-call arguments: the caller (webapp/server.py) resolves the value from the REQUESTING session's own
# `MultiReferentWMOrgan.current_focus()` and passes it down. `_FOCUS_UNSET` keeps the OLD ambient-pool-attribute
# read as a fallback ONLY for callers that omit the kwarg (the offline self-tests below), so their byte-identical
# behaviour is untouched; production now always passes an explicit value (even None), so it never consults the
# ambient global.
_FOCUS_UNSET = object()


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
        self.live_per_turn = False  # PART 3: the edge starts at W0=0.05, gate OPEN, and grows PER real chat turn
        self.n_live_credited = 0    # PART 3: count of live turns that produced a credited plasticity step

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
            # LIVE-LEARNING: start the edge at W0=0.05 (R3Pool, gate OPEN). Emergent, not pre-grown.
            from research.runners._onebrain_integration_r3_spiking_dopamine_credit import R3Pool
            p = R3Pool(self.seed, mode="intact")
            self.learned = True
            if live_per_turn_enabled():
                # PART 3 PER-TURN: leave the edge at W0=0.05, but FREEZE the candidate gate now (R3Pool.__init__
                # opened it) -- do NOT run a build curriculum. The cross-edge GROWS one credited step per LIVE chat
                # turn via `credit_live_turn`, which OPENS the gate for exactly that one step then RE-FREEZES, so
                # every production READ is a frozen forward pass. The trajectory starts at the near-zero baseline.
                from research.runners._onebrain_integration_r2_threefactor_selforganized import GATE as _GATE
                p.b.set_plasticity_gate(_GATE, 0.0)
                self.live_per_turn = True
                self.grow_traj = [dict(turn=0, **p.cross_weights())]
            else:
                # PART 2 BUILD-CURRICULUM: grow over a multi-turn build-time sequence, then freeze (reproduces the
                # 2026-08-27 live-learning-GO; the `--verify-live` entrypoint sets `set_live_per_turn(False)`).
                self.grow_traj = grow_live_selfsupervised(p)
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
        # `xedge_focus` (LEGACY, self-tests only, 2026-08-27): production no longer reads or writes this attribute
        # -- every real session's focus lives on ITS OWN `MultiReferentWMOrgan._own_focus` and is threaded
        # explicitly through `wm_focus`/`focus` kwargs (see `_FOCUS_UNSET` above). Kept here ONLY so the offline
        # `_selftest_livelearn`/`_selftest_perturn` below (which manipulate it directly, single-session, sequential,
        # never exercising cross-session isolation) keep running byte-identical to before this fix.
        self.pool.xedge_focus = None
        self.pool.xedge_codrive_params = dict(_CODRIVE_PARAMS)   # NOT session state -- a constant marker/config dict
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
    # LEGACY (self-tests only, 2026-08-27): production never calls these -- a real session's focus lives on its
    # own `MultiReferentWMOrgan` and is passed explicitly per call (see `_FOCUS_UNSET` above). `clear_focus` had
    # ZERO live callers even before this fix (the finding's own observation) -- kept for the offline self-tests.
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

    # ── PART 3 PER-TURN LIVE PLASTICITY: grow the cross-edge ONE credited step per real chat turn ──
    def credit_live_turn(self, content_direction: str, conf: float = 0.02, lesion_plasticity: bool = False,
                         focus=_FOCUS_UNSET):
        """Apply ONE in-brain self-supervised credited plasticity step to the cross-edge from the HELD focus pool
        (`focus`, the CALLING session's own xedge focus -- resolved by the caller from ITS OWN d6 organ; see
        `credit_live_turn_from_comprehension` / `_FOCUS_UNSET`), during a LIVE chat turn -- the SAME
        `_credit_turn_step` PART 2 fires over a build curriculum, now fired ONCE per turn. The gate is OPENED for
        exactly this one credited step then RE-FROZEN (every production READ is a frozen forward pass).
        `content_direction` in {"agent","patient"} is the turn's discourse content (host/teacher scaffold, declared
        residual #1 -- WHICH discourse); the credit VALUE + the teach DIRECTION are read off the brain's OWN
        confident `amb_read` (|margin|>conf), so NO host label writes the weight. Bounded by stdp_w_max (F3, the
        R3Pool gate). `lesion_plasticity=True` runs the SAME credit path but LEAVES THE GATE FROZEN (the
        load-bearing lesion: the credited episode drives, yet no weight can accumulate). No-op (byte-identical,
        returns None) unless `live_per_turn` is active, a focus is held, and the content resolves. Appends the new
        weights to `grow_traj` and returns a small trace dict."""
        if not self.ok or not self.live_per_turn:
            return None
        foc = focus
        if foc is _FOCUS_UNSET:    # legacy fallback (offline self-tests only, via set_focus/set on self.pool)
            foc = getattr(self.pool, "xedge_focus", None)
        if foc is None or content_direction not in ("agent", "patient"):
            return None
        p = self._r3pool
        from research.runners._onebrain_integration_r2_threefactor_selforganized import GATE
        AG_KEYS, PA_KEYS, AG_IX, PA_IX = _cue_dirs(p.ix)
        if content_direction == "agent":
            cue_keys, cue_ix = AG_KEYS, AG_IX
        else:
            cue_keys, cue_ix = PA_KEYS, PA_IX
        w_before = dict(self.cross_weights) if self.cross_weights else {}
        # OPEN the gate for exactly this one credited step, then RE-FREEZE (reads stay a frozen forward pass). The
        # lesion runs the identical credit path with the gate LEFT FROZEN (gate=None) -> no weight accumulates.
        credited, teach, margin = _credit_turn_step(p, foc, cue_keys, cue_ix, conf,
                                                    gate=(None if lesion_plasticity else GATE))
        w_after = p.cross_weights()
        self.cross_weights = w_after
        if credited:
            self.n_live_credited += 1
        self.grow_traj.append(dict(turn=len(self.grow_traj), focus=foc, content=content_direction,
                                   credited=bool(credited), teach=teach, margin=round(float(margin), 4), **w_after))
        return {"credited": bool(credited), "teach": teach, "margin": float(margin), "focus": foc,
                "content_direction": content_direction, "w_before": w_before, "w_after": w_after,
                "n_live_credited": self.n_live_credited}


def credit_live_turn_from_comprehension(comp_organ, svo, conf: float = 0.02, wm_focus=_FOCUS_UNSET):
    """LIVE reply-path hook (PART 3). Called from `webapp.server.brain_reply` on a real chat turn where a WM
    referent is HELD BY THIS SESSION (`wm_focus`, resolved by the caller from the REQUESTING session's own d6
    organ via `MultiReferentWMOrgan.current_focus()` -- never read off the shared process pool's ambient global;
    see `_FOCUS_UNSET` and the 2026-08-27 cross-session xedge_focus leak fix) AND comprehension resolved (a
    competent transitive). Derives the held referent's DISCOURSE role IN-BRAIN -- the sign of the first noun's
    per-noun agent-evidence a0 off `cp_firing_states` (positional focus = the primary held referent ~ n0,
    declared residual) -- and applies ONE credited plasticity step via `pool.credit_live_turn`. No-op
    (byte-identical, returns None) unless the flags are on, the learn build is per-turn, a focus is held, and the
    content commits (|a0| clears the calibrated role floor). Never raises into the turn (best-effort)."""
    try:
        if not (xedge_enabled() and xedge_learn_enabled()):
            return None
        pool = get_xedge_pool()
        if pool is None or not getattr(pool, "live_per_turn", False):
            return None
        foc = wm_focus
        if foc is _FOCUS_UNSET:    # legacy fallback (offline self-tests only)
            foc = getattr(pool.pool, "xedge_focus", None)
        if foc is None:
            return None
        n0, v, n1 = svo
        comp_organ.ensure_built()
        # in-brain content read of the held (first) referent: a0 = sel_agent-sel_patient evidence for n0.
        with comp_organ._guard():
            a0, a1 = comp_organ._read_per_noun(comp_organ.comp, n0, v, n1, wm_focus=foc)
        floor = float(getattr(comp_organ, "role_floor", 0.0) or 0.0)
        if abs(a0) <= floor:
            return None                                  # content did not commit a role -> no self-supervised credit
        direction = "agent" if a0 > 0 else "patient"
        trace = pool.credit_live_turn(direction, conf=conf, focus=foc)
        if trace is not None:
            trace["a0"] = float(a0)
            trace["a1"] = float(a1)
            trace["role_floor"] = floor
        return trace
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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


def _selftest_perturn(seed, n_turns=24, conf=0.02):
    """PART 3 end-to-end (offline, exercises the SAME per-turn credit path the live handler calls): over a
    sequence of turns on the SAME held focus pool (w0 = the positional focus), the cross-edge (1) RISES from
    W0=0.05 one credited step at a time (emergent per-turn, NOT a build curriculum); (2) stays BOUNDED by
    stdp_w_max (F3); (3) is LOAD-BEARING across the session -- a session that TEACHES the held referent the AGENT
    role makes a LATER balanced read resolve it AGENT, a session that teaches PATIENT resolves it PATIENT (the
    later resolution reflects what earlier turns taught); (4) LESION the per-turn plasticity (freeze the gate ->
    no weight moves) and the edge does NOT accumulate -> the later read reverts to baseline (no resolved role)."""
    import numpy as np
    from research.runners import comprehension_production_organ as _CO
    from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS, HMAX
    from research.runners._spiking_comprehension_monitor_derisk import build_battery
    global _POOL
    foc = CAND_POOLS[0]                                    # the positional live focus (declared residual)
    ambig = [it for it in build_battery(seed, n_per_cond=3) if it[0] == 0 and "ambig" in it[1]][:5]

    def run_session(direction, lesion_plasticity):
        global _POOL
        _POOL = None                                      # FRESH pool per session (module global, not a local)
        _CO._ORGAN = None
        pool = get_xedge_pool(seed)                       # per-turn build: edge at 0.05, gate FROZEN between turns
        assert pool is not None and pool.ok and pool.live_per_turn, "per-turn pool failed to build"
        pool.set_focus(foc)
        role_key = f"{foc}->{'A' if direction == 'agent' else 'P'}"
        traj = [float(pool.cross_weights[role_key])]
        for _t in range(n_turns):
            # ONE in-brain self-supervised credited step per turn (opens the gate for that step then re-freezes).
            # LESION: the identical credit path with the gate LEFT FROZEN -> no weight accumulates.
            pool.credit_live_turn(direction, conf=conf, lesion_plasticity=lesion_plasticity)
            traj.append(float(pool.cross_weights[role_key]))
        # LATER read #1 (headline, content-cancelled + edge-attributable): the WM-RESOLVED balanced margin for the
        # held focus -- EXACTLY the quantity `_wm_resolved_role` thresholds. delta = margin(foc) - baseline(no-edge
        # control hold) is signed ONLY by the grown cross-edge, so it isolates what the per-turn turns taught.
        amb = pool.pool.xedge_amb_read
        cues = pool.pool.xedge_balanced_cues
        base_pool = pool.pool.xedge_base_pool
        baseline = float(amb(base_pool, cues)["margin"])
        wm_margins = [float(amb(foc, cues)["margin"]) for _ in range(3)]
        delta = float(np.mean(wm_margins)) - baseline
        # LATER read #2 (decision-level, through the REAL production repair path): the role + whether the edge
        # SIGNED it (wm_resolved). Content-ambiguous items -> the cross-edge is the tiebreaker the content lacked.
        corg = pool.comp_organ
        corg.ensure_built()
        pool.pool.xedge_focus = foc
        reads = []
        for (lab, tag, n0, v, n1) in ambig:
            r = corg.repair_target(f"{n0} {v} {n1}")
            reads.append({"item": f"{n0}/{v}/{n1}", "role": (r.get("role") if r else None),
                          "wm_resolved": bool(r.get("wm_resolved")) if r else False})
        wmax = max(pool.cross_weights.values())
        return {"direction": direction, "lesion": lesion_plasticity, "w_focus_start": traj[0],
                "w_focus_end": traj[-1], "w_traj": [round(x, 4) for x in traj], "wmax": round(float(wmax), 4),
                "n_live_credited": pool.n_live_credited, "baseline_margin": round(baseline, 4),
                "wm_delta_margin": round(delta, 4), "reads": reads,
                "wm_resolved_reads": sum(rd["wm_resolved"] for rd in reads)}

    agent_sess = run_session("agent", lesion_plasticity=False)
    patient_sess = run_session("patient", lesion_plasticity=False)
    lesion_sess = run_session("agent", lesion_plasticity=True)

    eps = max(0.004, 3.0 * abs(agent_sess["baseline_margin"]))
    grew = agent_sess["w_focus_end"] > 0.5 and agent_sess["w_focus_start"] <= 0.06 and \
        patient_sess["w_focus_end"] > 0.5 and patient_sess["w_focus_start"] <= 0.06
    bounded = agent_sess["wmax"] <= HMAX + 1e-6 and patient_sess["wmax"] <= HMAX + 1e-6
    # LOAD-BEARING (headline): the taught role SIGNS the later content-cancelled read -- agent-taught -> delta>+eps
    # (resolves AGENT), patient-taught -> delta<-eps (resolves PATIENT); vary the taught role -> the later
    # resolution DIFFERS in sign. Plus the decision-level read: the edge SIGNS >=1 real repair role per session.
    da, dp = agent_sess["wm_delta_margin"], patient_sess["wm_delta_margin"]
    taught_signs_read = da > eps and dp < -eps
    differs = bool(da > eps and dp < -eps)                # opposite-signed -> the later resolution differs by taught role
    decision_load_bearing = agent_sess["wm_resolved_reads"] > 0 and patient_sess["wm_resolved_reads"] > 0
    # LESION (per-turn plasticity frozen): the edge does NOT accumulate (stays ~0.05) -> the later content-cancelled
    # read stays at baseline (|delta|<eps) and the edge signs NO repair role.
    lesion_no_accum = (lesion_sess["w_focus_end"] <= 0.06 and abs(lesion_sess["wm_delta_margin"]) < eps
                       and lesion_sess["wm_resolved_reads"] == 0)
    # ATTRIBUTION (whose difference IS the later-read shift?): the content-cancelled wm_delta_margin the taught turns
    # produce must be OWNED by the per-turn plasticity, not by anything the credit path does with a frozen gate.
    # max|delta| over the two intact taught sessions vs the frozen-gate lesion -- forces the subtraction (gap#5).
    from tools.lab import attributable_to
    max_intact_delta = max(abs(da), abs(dp))
    frac = attributable_to(f"seed{seed} per-turn xedge plasticity: taught wm_delta vs frozen-gate lesion",
                           max_intact_delta, abs(lesion_sess["wm_delta_margin"]))
    GO = bool(grew and bounded and taught_signs_read and differs and decision_load_bearing and lesion_no_accum)
    return {"seed": seed, "n_turns": n_turns, "focus": foc, "eps": round(eps, 4),
            "agent_session": agent_sess, "patient_session": patient_sess, "lesion_session": lesion_sess,
            "grew_from_0.05_per_turn": bool(grew), "bounded_F3": bool(bounded),
            "taught_role_signs_later_read": bool(taught_signs_read), "later_resolution_differs": bool(differs),
            "decision_load_bearing": bool(decision_load_bearing),
            "frac_attributable_to_per_turn_plasticity": (None if frac is None else float(frac)),
            "lesion_no_accumulation": bool(lesion_no_accum), "GO": GO}


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="build+grow the FROZEN pool and self-verify load-bearing")
    ap.add_argument("--verify-live", action="store_true",
                    help="PART 2: build the LIVE-LEARNED pool (edge grows from in-brain credit) + verify caveat close")
    ap.add_argument("--verify-per-turn", action="store_true",
                    help="PART 3: grow the edge ONE credited step per LIVE turn (0.05->), load-bearing + lesion")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    global _POOL
    if args.verify_per_turn:
        os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"
        set_live_per_turn(True)
        results = []
        for s in seeds:
            sv = _selftest_perturn(s)
            a = sv["agent_session"]; p = sv["patient_session"]; l = sv["lesion_session"]
            print(f"[seed {s}] grew_0.05_per_turn={sv['grew_from_0.05_per_turn']} bounded_F3={sv['bounded_F3']} "
                  f"taught_signs_read={sv['taught_role_signs_later_read']} differs={sv['later_resolution_differs']} "
                  f"decision_LB={sv['decision_load_bearing']} lesion_no_accum={sv['lesion_no_accumulation']} "
                  f"eps={sv['eps']} GO={sv['GO']}", flush=True)
            print(f"    agent-taught  w{sv['focus']}->A {a['w_focus_start']}->{a['w_focus_end']:.3f} "
                  f"(credited {a['n_live_credited']}) wm_delta={a['wm_delta_margin']:+.4f} "
                  f"wm_resolved_reads={a['wm_resolved_reads']}/{len(a['reads'])}")
            print(f"    agent traj: {a['w_traj']}")
            print(f"    patient-taught w{sv['focus']}->P {p['w_focus_start']}->{p['w_focus_end']:.3f} "
                  f"(credited {p['n_live_credited']}) wm_delta={p['wm_delta_margin']:+.4f} "
                  f"wm_resolved_reads={p['wm_resolved_reads']}/{len(p['reads'])}")
            print(f"    LESION(agent) w{sv['focus']}->A {l['w_focus_start']}->{l['w_focus_end']:.3f} "
                  f"(credited {l['n_live_credited']}) wm_delta={l['wm_delta_margin']:+.4f} "
                  f"wm_resolved_reads={l['wm_resolved_reads']}/{len(l['reads'])}")
            results.append({"seed": s, "selftest": sv})
        payload = {"probe": "onebrain_xedge_production_per_turn_live_plasticity", "seeds": seeds,
                   "backend": os.environ.get("SIM_BACKEND", "numpy"), "results": results,
                   "n_go": sum(r["selftest"]["GO"] for r in results),
                   "note": ("PART 3: the cross-edge grows ONE in-brain self-supervised credited step PER LIVE turn "
                            "(via pool.credit_live_turn, the SAME atom PART 2's build curriculum runs) -- rises from "
                            "W0=0.05 emergently, bounded by stdp_w_max (F3), LOAD-BEARING across the session (the "
                            "taught role shows in a LATER balanced read; varying it flips the later resolution), and "
                            "LESIONING the per-turn plasticity (frozen gate) yields no accumulation. Semantic "
                            "referent->pool binding + WHICH discourse are declared residuals (host/teacher scaffold).")}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
            print(f"wrote {args.out}", flush=True)
        return 0

    if args.verify_live:
        os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"
        set_live_per_turn(False)                          # PART 2 uses the build-time curriculum (not per-turn)
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
