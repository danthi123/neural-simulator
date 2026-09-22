"""ON-SUBSTRATE episodic-dialogue memory = the standing 6/6-GO gap#5 dendritic-dAP READOUT completion
(ab9f7dbe, research/runners/_gap5_dendritic_dap_readout_completion_derisk.py) wired as a per-topic episodic STORE.

WHAT THIS REPLACES: the conversation eval's turn-7 recall was a host dict lookup over a per-turn `episode_mem`
buffer (a DECLARED SCAFFOLD -- host bookkeeping, the brain did no memory). Here the recall is SPIKING
pattern-completion: each spoken dialogue TOPIC BTSP-forms a CA3 assembly on a dedicated spiking CA3 readout bridge;
a later referential cue COMPLETES that assembly cue-specifically via the two-compartment dendritic dAP apical read
(the ~23-cell EMERGENT assembly is what frees the per-cell read; a pre-assigned 0.18*N assembly does NOT complete --
so the scale + emergent membership are load-bearing, NOT tunable away).

MECHANISM (reuse-by-import, NO re-derive, NO sim/ edit):
  * membership  : emergent_assemblies (DG sparse-detonator SELECTION, n_ca3=2000)   [emergent_end_to_end runner]
  * readout     : _build_dap_readout (coincidence ON, two-compartment apical dAP)   [gap5 dapB runner]
  * store       : _form_one_assembly (BTSP one-shot, isolated encode episode) -> copy WITHIN weights onto readout
  * recall      : _apical_up_read (fraction of held cells whose cp_v_apical is UP)  [gap5 dapB runner]
  * geometry    : _held_cue_perm (make_readout's eval_assembly cue/held/perm split) [gap5 dapB runner]

HONESTY: recall reports the SPIKING completion. A topic never spoken -> its assembly is never BTSP-formed ->
its cue does NOT complete (apical UP ~0) -> "not in memory" is a GENUINE completion failure, not a host flag.
LOAD-BEARING: `lesion=True` restores the UNFORMED (baseline) recurrent weights before the read -> the dendritic
completion collapses -> recall breaks. That is the teeth that prove the recall is carried by the spiking assembly,
not by the co-kept host oracle.

The store's default scale is the GO scale (n_ca3=2000, cupy). On the numpy substrate this is faithful but SLOW
(speed is secondary, per the mission). A `precompute -> .npz cache` of (assembly geometry + formed readout weights)
lets a live eval pay the store cost once. See the conversation eval's `--spiking-episodic` wiring.
"""
from __future__ import annotations

import os
import numpy as np

from sim.backend import get_backend
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (
    _build_dap_readout, _apical_up_read, _apical_dual_read, _held_cue_perm, GRADED_READS)
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (
    make_readout, _form_one_assembly, _build_bridge as _formation_build_bridge)
from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import emergent_assemblies


# GO defaults, copied from the standing 6/6-GO dapB runner's argparse (ab9f7dbe) -- EXCEPT kthresh.
# ⛔ kthresh CORRECTED 30 -> 8 (2026-08-10, direct 6-seed test in FRESH isolated builds = the production path). The
# apical dAP UP-fraction read has a NARROW per-assembly operating window in kthresh: kt=30 (the value this dict shipped
# with) is far ABOVE it -> fires NOTHING on either backend (the #2b "cupy-GO" mis-verification); kt>=10 SILENCES the
# smallest emergent assemblies (~13 cells: 0.57 @kt8 -> 0.0-0.43 @kt10, on the cliff); kt<=6 lets some emergent
# memberships SELF-IGNITE (the nocue read goes UP with NO cue -> specificity fails: a fresh-build s102 dog reads
# nocue=1.0 @kt6). kt=8 THREADS the window: the smallest 13-14-cell assembly fires (0.57-0.86) AND no self-ignition,
# cue-specific on 6/6 seeds with perm=nocue=lesion(baseline)=0 (research/findings/raw/_episodic_dap_kthresh/
# clean_verify_kt8.json). NB (i) the dapB runner SWEEPS kthresh {15,30} scoring the MEAN apical-UP over patterns (kt=15
# wins, never kt=30) -- but this module reads ONE topic/recall and that mean MASKED the per-topic size failure; (ii) the
# reuse-heavy per-topic sweep (sixseed_kt_sweep.json) is UNRELIABLE (state contamination across many live-mutated reads
# on one bridge) and produced TWO artifacts -- it MISSED kt=6's self-ignition AND FABRICATED a kt=8 teeth-fail (s102 cat
# lesion=1.0) -- both disproven by the fresh isolated builds (fresh s102@kt8 = clean BOTH-PASS); (iii) emergent assembly
# membership is non-deterministic at the firing threshold (FMA/summation reorder, sim/kernels.py) so exact per-seed
# reads vary build-to-build, but kt=8 passes across builds where kt=6 does not. gap#5 assembly-SIZE residual corrected
# by the operating point, not a wall. The apical-UP read is NON-monotonic in kthresh; 8 is chosen empirically.
GO_DEFAULTS = dict(
    density=0.5, wmax=100.0, kthresh=8.0, plateau_strength=30.0, apical_R=0.15, self_regen=2.0,
    v_hold=-35.0, apical_kir_g=1.0, apical_gc=0.3, apical_gc_read=0.3, up_thresh=-20.0, ca3_fb_inhib=60.0,
    btsp_lr=0.05, encode_drive=700.0, encode_plateau_pA=250.0, train_events=40, drive_steps=48, reset_steps=15,
    assembly_frac=0.18, cue_frac=0.5, drive_pA=300.0, warm_steps=100, read_steps=100, silence_steps=50,
)

# GO cue-specific completion criterion (dapB `_go`): apical UP fraction on held cells.
COMPLETE_MIN = 0.20        # held_cue >= 0.20
CUE_OVER_CTRL = 3.0        # held_cue >= 3 * (perm|nocue)
CTRL_MAX = 0.10            # held_nocue <= 0.10


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# STORE-VERIFY (ENCODE→VERIFY→RE-ENCODE) — env-gated, DEFAULT-OFF -> byte-identical single-shot store.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# WHY (load-bearing #1 metric, finding 2026-09-22-borderline-separability-stabilizer-is-buildable):
# episodic-memory is load-bearing on 5/6 seeds; at s44 the INTACT store->recall sometimes fails to read the memory
# back (`in_memory=False` == the lesion -> a 0-diff, not load-bearing). The failure is FLAKY across build contexts:
# the emergent DG-selected 'dog' ensemble is drawn ONCE at organ-build time (`emergent_assemblies`) and is subject to
# GPU firing-threshold non-determinism (FMA/summation reorder, per this module's kt=8 note) — most draws form a
# completable attractor (a strong cue read ~0.86), but an occasional draw is degenerate and does NOT complete. A
# single-shot store banks whatever draw it got.
#
# THE MECHANISM (biology, not a tuned constant): a memory whose FIRST encoding did not form a completable attractor is
# RE-ENCODED. In hippocampus, BTSP place-field induction is PROBABILISTIC per dendritic plateau (Bittner/Milstein/
# Magee, Science 2017; Grienberger & Magee 2022) and a stable field is established over repeated plateaus/laps, each
# re-recruiting a (stochastically) different sparse ensemble; immediate post-encoding sharp-wave-ripple replay
# re-instates and strengthens a labile trace until it reactivates (Girardeau/Zugaro; Roux et al. 2017). The natural
# stopping condition is the circuit's OWN read-back — does the memory reactivate from a PARTIAL cue — i.e. the exact
# dendritic-dAP completion gate the recall uses. So the store runs ENCODE→VERIFY→RE-ENCODE: encode, drive the partial
# cue and read the completion (the recall gate), and if it does not read back, RE-RECRUIT a fresh DG-selected ensemble
# (an independent emergent draw on the SAME CA3 microcircuit) and encode again — until the substrate's own completion
# gate confirms the trace, bounded by a safety lap cap. It is EMERGENT (the loop exits on the substrate's own read,
# never on a target count) and applies uniformly to every topic/seed. OFF (default) -> single-shot -> byte-identical.
def _store_verify_enabled() -> bool:
    """`BRAIN_EPISODIC_STORE_VERIFY` in {1,true,yes,on} -> the store runs the ENCODE→VERIFY→RE-ENCODE loop (make the
    intact store read back reliably across seeds). Default (unset/anything else) -> single-shot store, byte-identical
    to HEAD."""
    v = os.environ.get("BRAIN_EPISODIC_STORE_VERIFY")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _store_verify_max_laps() -> int:
    """Safety bound on re-encoding laps. The loop EXITS on the FIRST successful read-back, so this only caps a
    persistently-failing encode (it never forces extra laps on a memory that already completed). Env override
    `BRAIN_EPISODIC_STORE_VERIFY_MAX_LAPS`; default 8 (with per-draw success ~0.8, 8 independent re-recruitments miss
    with probability ~1e-5)."""
    try:
        return max(1, int(os.environ.get("BRAIN_EPISODIC_STORE_VERIFY_MAX_LAPS", "8")))
    except Exception:
        return 8


class EpisodicDapMemory:
    """Per-topic spiking episodic-dialogue store on a CA3 dendritic-dAP readout bridge.

    topics: the toy-world agent topics (e.g. ['cat','dog']); each is pre-allocated an assembly SLOT so that a
    referent never spoken still reads through the SPIKING completion (unformed -> no completion), not a host flag.
    """

    def __init__(self, seed, topics, *, verbose=False, sep_bias=0.0, **overrides):
        self.seed = int(seed)
        self.topics = sorted(set(topics))
        self.p = dict(GO_DEFAULTS); self.p.update(overrides)
        self.verbose = verbose
        self.sep_bias = float(sep_bias)
        self.cp, _ = get_backend()
        self.backend = os.environ.get("SIM_BACKEND", "(unset)")

        n_slots = max(len(self.topics), 1)
        # ---- emergent DG-selected membership (anti-cheat #1 of the GO) at the GO scale --------------------------
        # D5 PATTERN-SEPARATION SET-POINT (board #73, knob-1 of the learn-through-use default-ON flip): when
        # sep_bias > 0, form the assemblies through a DG winner-fatigue set-point (a per-CA3 intrinsic-excitability
        # depression accrued as each pattern recruits cells; Turrigiano 2011 / Marr 1971 / O'Reilly-McClelland 1994)
        # so later patterns recruit DISJOINT cells. Disjoint membership removes the shared-cell path that lets the
        # D5 consolidation of one memory shift a NEIGHBOR's surfaced recall strength (the crosstalk residual that
        # blocked the flip). ADDITIVE: sep_bias <= 0 (the default) calls the UNMODIFIED emergent_assemblies ->
        # byte-identical to HEAD. HOST-SCAFFOLD RESIDUAL (declared): the winner-fatigue bias is a host-applied
        # intrinsic current; the SELECTION (which cells cross theta) stays on-substrate spiking. The on-substrate
        # spiking intrinsic-plasticity form (a real conductance on the granule region) is the tracked next step.
        # Lazy import: keep the de-risk module's import-time SIM_BACKEND default out of the load path (it only runs
        # when the separator is actually armed, by which point SIM_BACKEND is already set).
        if self.sep_bias > 0.0:
            from research.runners._d5_pattern_separation_setpoint_derisk import _emergent_assemblies_setpoint
            self.assemblies, r1 = _emergent_assemblies_setpoint(self.seed, n_slots, self.sep_bias)
        else:
            self.assemblies, r1 = emergent_assemblies(self.seed, n_patterns=n_slots)
        self.n_ca3 = int(r1[2])
        self.assembly_sizes = [int(len(a)) for a in self.assemblies]

        # ---- the dAP readout bridge (coincidence ON = the dendritic completion read) ---------------------------
        self.bridge = _build_dap_readout(
            self.seed, n_ca3=self.n_ca3, ca3_density=self.p["density"], ca3_fb_inhib=self.p["ca3_fb_inhib"],
            k_thresh=self.p["kthresh"], plateau_strength=self.p["plateau_strength"], apical_R=self.p["apical_R"],
            self_regen=self.p["self_regen"], v_hold=self.p["v_hold"], apical_kir_g=self.p["apical_kir_g"],
            apical_gc=self.p["apical_gc"], apical_gc_read=self.p["apical_gc_read"], coincidence=True)
        self._read_kwargs = dict(assembly_frac=self.p["assembly_frac"], cue_frac=self.p["cue_frac"],
                                 drive_pA=self.p["drive_pA"], warm_steps=self.p["warm_steps"],
                                 read_steps=self.p["read_steps"], silence_steps=self.p["silence_steps"],
                                 assemblies_ext=self.assemblies)
        self.R = make_readout(self.bridge, self.seed, **self._read_kwargs)
        self.baseline_weights = self.R.C.data.copy()     # UNFORMED recurrent weights (the lesion target)
        self.held_pos_by_asm, self.cue_by_asm, self.perm_by_asm = _held_cue_perm(self.R, self.seed)

        self._form_build_kwargs = dict(n_ca3=self.n_ca3, ca3_density=self.p["density"],
                                       ca3_fb_inhib=self.p["ca3_fb_inhib"], ca3_ff_inhib=None, nmda_tau=100.0,
                                       nmda_ratio=1.0, enable_ou=False, element="nmda_slow")
        self.topic_slot = {t: i for i, t in enumerate(self.topics)}   # pre-alloc ALL toy topics to slots
        self.formed = set()                                            # slot indices BTSP-formed (topic spoken)
        self.store_log = []                                            # ordered topics stored (for the record)
        if self.verbose:
            print(f"[episodic-dap] n_ca3={self.n_ca3} slots={self.topic_slot} sizes={self.assembly_sizes} "
                  f"sep_bias={self.sep_bias} backend={self.backend}", flush=True)

    # ---- one BTSP encoding episode (a 'lap'): form slot on a FRESH isolated bridge + copy within-slot weights -----
    def _form_slot_onto_readout(self, slot):
        """Encode assembly `slot` once: BTSP-form it on a fresh, isolated formation bridge (only that assembly driven,
        so cross-assembly dW==0 by construction) and copy its WITHIN-assembly potentiated weights onto the readout
        self.R. Extracted VERBATIM from the original store() body so the default (verify-OFF) path is byte-identical.
        Rebuilds the formation readout from self._read_kwargs, whose `assemblies_ext` is a live reference to
        self.assemblies -> a re-recruited ensemble (see _reselect_slot) is picked up here automatically."""
        bi = _formation_build_bridge(self.seed, **self._form_build_kwargs)
        Ri = make_readout(bi, self.seed, **self._read_kwargs)
        _form_one_assembly(bi, Ri, slot, btsp_w_max=self.p["wmax"], btsp_lr=self.p["btsp_lr"],
                           encode_drive=self.p["encode_drive"], encode_plateau_pA=self.p["encode_plateau_pA"],
                           train_events=self.p["train_events"], drive_steps=self.p["drive_steps"],
                           reset_steps=self.p["reset_steps"], plateau=True)
        m = Ri.withinA_masks[slot]
        self.R.C.data[m] = bi.cp_connections.data[m]      # copy ONLY the within-slot BTSP-formed weights
        del bi, Ri

    def _geom_for_slot(self, A):
        """The (held_pos, cue, perm) geometry for a single assembly A on self.R — mirrors `_held_cue_perm` exactly
        (same seed*131+se[0] permutation, same cue_frac split, same non-member perm draw), for one re-recruited slot."""
        R = self.R
        se = np.asarray(A, dtype=np.int64)
        r = np.random.default_rng(self.seed * 131 + int(se[0]))
        se = se[r.permutation(len(se))]
        n_cue = max(2, int(R._cue_frac * len(se)))
        cue, held = se[:n_cue], se[n_cue:]
        held_pos = [R.ca3_pos[int(g)] for g in held]
        member = set(int(g) for g in se)
        nonA = np.asarray([g for g in R.ca3_idx if int(g) not in member], dtype=np.int64)
        perm_cue = r.choice(nonA, size=len(cue), replace=False)
        return held_pos, cue, perm_cue

    def _reselect_slot(self, slot, lap):
        """RE-RECRUIT the memory's sparse ensemble: on the SAME CA3 microcircuit (self.seed — so the ensemble's
        cells are the seed's own recurrently-connected cells, a completable attractor on self.R), drive a FRESH DG
        pattern to co-activate a DIFFERENT sparse ensemble, return the prior ensemble's within-connections to the
        UNFORMED baseline (no lingering stale trace), and rebuild this slot's within-mask + cue/held/perm geometry.
        The subsequent _form_slot_onto_readout encodes the new ensemble. Biology: a memory whose first encoding did
        not form a completable attractor is re-encoded by the DG pattern-separator recruiting a stochastically
        different sparse ensemble across plateau laps (Bittner/Magee 2017; post-encoding SWR re-instatement). The DG
        pattern is varied (NOT the network seed): re-drawing on a DIFFERENT network would select cells that are not
        recurrently connected in this readout, so the trace would not complete."""
        cp = self.cp
        R = self.R
        n_slots = max(len(self.topics), 1)
        # A FRESH co-active ensemble on the SAME seed network: request one extra DG pattern beyond the per-topic
        # patterns (each pattern m uses a distinct drive seed self.seed*100+m) and take it. Each lap advances to a new
        # pattern; the verify loop keeps going until one forms a completable attractor.
        asm, _r1 = emergent_assemblies(self.seed, n_patterns=n_slots + lap + 1)
        new_A = np.asarray(asm[n_slots + lap], dtype=np.int64)
        # 1) return the OLD slot's within-connections to the unformed baseline
        old_m = R.withinA_masks[slot]
        R.C.data[old_m] = self.baseline_weights[old_m]
        # 2) install the fresh ensemble + recompute this slot's within-mask and the union across all slots
        self.assemblies[slot] = new_A
        R.assemblies[slot] = new_A
        self.assembly_sizes[slot] = int(len(new_A))
        is_A = cp.zeros(R.n, dtype=cp.bool_)
        if len(new_A) > 0:
            is_A[cp.asarray(new_A)] = True
        R.withinA_masks[slot] = is_A[R.rows] & is_A[R.cols]
        wu = cp.zeros(len(R.rows), dtype=cp.bool_)
        for mm in R.withinA_masks:
            wu |= mm
        R.within_union = wu
        # 3) recompute this slot's cue/held/perm geometry for the fresh ensemble
        self.held_pos_by_asm[slot], self.cue_by_asm[slot], self.perm_by_asm[slot] = self._geom_for_slot(new_A)

    # ---- STORE (episodic WRITE): a spoken topic BTSP-forms its assembly on the readout bridge -------------------
    def store(self, topic):
        slot = self.topic_slot.get(topic)
        if slot is None or slot in self.formed:
            return False
        self._form_slot_onto_readout(slot)
        self.formed.add(slot); self.store_log.append(topic)
        # ENCODE→VERIFY→RE-ENCODE (BRAIN_EPISODIC_STORE_VERIFY, default-OFF -> byte-identical single-shot store above).
        # Verify the trace reads back from its OWN partial cue (the recall completion gate); if not, re-recruit a fresh
        # DG ensemble and re-encode, until the substrate's own read-back confirms it (or the safety lap cap is hit).
        n_laps = 0
        if _store_verify_enabled():
            for lap in range(_store_verify_max_laps()):
                if bool(self.recall(topic).get("in_memory")):
                    break
                self._reselect_slot(slot, lap)
                self._form_slot_onto_readout(slot)
                n_laps = lap + 1
        w_within = float(self.cp.mean(self.R.C.data[self.R.withinA_masks[slot]]))
        if self.verbose:
            _extra = f" reencode_laps={n_laps}" if _store_verify_enabled() else ""
            print(f"[episodic-dap] STORE topic={topic!r} slot={slot} w_within={w_within:.1f}{_extra}", flush=True)
        return True

    # ---- RECALL (episodic READ): drive the topic-slot cue, read the dendritic dAP apical completion -------------
    def _apical(self, slot, cue_kind, lesion):
        """The BINARY-only UP-fraction read (the historical moat gate). Retained as the byte-identity oracle the
        graded read is verified against; `recall` now uses `_apical_dual` (which returns the identical `up`)."""
        cue = {"cue": self.cue_by_asm, "perm": self.perm_by_asm}.get(cue_kind)
        drive = [cue[slot]] if cue is not None else [None]
        if lesion:
            saved = self.R.C.data.copy(); self.R.C.data[:] = self.baseline_weights
        try:
            return _apical_up_read(self.bridge, self.R, [self.held_pos_by_asm[slot]], drive, self.p["up_thresh"])
        finally:
            if lesion:
                self.R.C.data[:] = saved

    def _apical_dual(self, slot, cue_kind, lesion):
        """One-pass dual read: the BINARY UP-fraction (`up`, byte-identical to `_apical_up_read` = the moat gate) AND
        the three GRADED magnitudes (depth_rest / depth_hold / soft) from the SAME cp_v_apical. Mirrors the
        6-seed-validated `GradedEpisodicDapMemory._apical_dual` (finding
        2026-08-20-d5-graded-apical-read-makes-learn-through-use-reliably-conversation-visible)."""
        cue = {"cue": self.cue_by_asm, "perm": self.perm_by_asm}.get(cue_kind)
        drive = [cue[slot]] if cue is not None else [None]
        if lesion:
            saved = self.R.C.data.copy(); self.R.C.data[:] = self.baseline_weights
        try:
            return _apical_dual_read(self.bridge, self.R, [self.held_pos_by_asm[slot]], drive,
                                     self.p["up_thresh"], self.p["v_hold"])
        finally:
            if lesion:
                self.R.C.data[:] = saved

    def recall(self, topic, *, lesion=False):
        """Return the SPIKING recall record for `topic`: apical UP completion for cue/perm/nocue + a cue-specific
        completion verdict, PLUS the GRADED apical magnitude (the conversation-visible recall STRENGTH that rises with
        learn-through-use, where the quantised binary UP-fraction is flat). lesion=True reads through the UNFORMED
        baseline weights (the load-bearing teeth). The BINARY UP-fraction + specificity gates STILL decide `in_memory`
        (the moat is unchanged, byte-identical to `_apical_up_read`); the graded magnitude is surfaced BESIDE it and is
        only meaningful when in_memory=True."""
        slot = self.topic_slot.get(topic)
        if slot is None:
            return {"topic": topic, "slot": None, "formed": False, "in_memory": False,
                    "apical_cue": 0.0, "apical_perm": 0.0, "apical_nocue": 0.0,
                    "graded_cue": {r: 0.0 for r in GRADED_READS},
                    "graded_perm": {r: 0.0 for r in GRADED_READS},
                    "graded_nocue": {r: 0.0 for r in GRADED_READS}, "reason": "no-slot"}
        c = self._apical_dual(slot, "cue", lesion)
        p = self._apical_dual(slot, "perm", lesion)
        n = self._apical_dual(slot, "nocue", lesion)
        cue, perm, nocue = c["up"], p["up"], n["up"]
        completes = bool(cue >= COMPLETE_MIN and cue >= CUE_OVER_CTRL * (perm + 1e-6)
                         and cue >= CUE_OVER_CTRL * (nocue + 1e-6) and nocue <= CTRL_MAX)
        return {"topic": topic, "slot": slot, "formed": bool(slot in self.formed and not lesion),
                "in_memory": completes, "apical_cue": float(cue), "apical_perm": float(perm),
                "apical_nocue": float(nocue), "lesioned": bool(lesion), "reason": "spiking-dap-completion",
                "graded_cue": {r: float(c[r]) for r in GRADED_READS},
                "graded_perm": {r: float(p[r]) for r in GRADED_READS},
                "graded_nocue": {r: float(n[r]) for r in GRADED_READS}}

    def discussed_topics(self, *, lesion=False):
        """Topics whose CA3 assembly COMPLETES via the dendritic dAP read = what the brain spiking-recalls as
        discussed (order = store order for the ones that complete)."""
        recalled = {t: self.recall(t, lesion=lesion) for t in self.topics}
        done = [t for t in self.store_log if recalled[t]["in_memory"]]
        # include any completing topic not in store_log (should be none) for completeness
        for t in self.topics:
            if recalled[t]["in_memory"] and t not in done:
                done.append(t)
        return done, recalled
