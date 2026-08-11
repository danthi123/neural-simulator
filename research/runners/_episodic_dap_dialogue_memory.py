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
    _build_dap_readout, _apical_up_read, _held_cue_perm)
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


class EpisodicDapMemory:
    """Per-topic spiking episodic-dialogue store on a CA3 dendritic-dAP readout bridge.

    topics: the toy-world agent topics (e.g. ['cat','dog']); each is pre-allocated an assembly SLOT so that a
    referent never spoken still reads through the SPIKING completion (unformed -> no completion), not a host flag.
    """

    def __init__(self, seed, topics, *, verbose=False, **overrides):
        self.seed = int(seed)
        self.topics = sorted(set(topics))
        self.p = dict(GO_DEFAULTS); self.p.update(overrides)
        self.verbose = verbose
        self.cp, _ = get_backend()
        self.backend = os.environ.get("SIM_BACKEND", "(unset)")

        n_slots = max(len(self.topics), 1)
        # ---- emergent DG-selected membership (anti-cheat #1 of the GO) at the GO scale --------------------------
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
                  f"backend={self.backend}", flush=True)

    # ---- STORE (episodic WRITE): a spoken topic BTSP-forms its assembly on the readout bridge -------------------
    def store(self, topic):
        slot = self.topic_slot.get(topic)
        if slot is None or slot in self.formed:
            return False
        bi = _formation_build_bridge(self.seed, **self._form_build_kwargs)
        Ri = make_readout(bi, self.seed, **self._read_kwargs)
        _form_one_assembly(bi, Ri, slot, btsp_w_max=self.p["wmax"], btsp_lr=self.p["btsp_lr"],
                           encode_drive=self.p["encode_drive"], encode_plateau_pA=self.p["encode_plateau_pA"],
                           train_events=self.p["train_events"], drive_steps=self.p["drive_steps"],
                           reset_steps=self.p["reset_steps"], plateau=True)
        m = Ri.withinA_masks[slot]
        self.R.C.data[m] = bi.cp_connections.data[m]      # copy ONLY the within-slot BTSP-formed weights
        self.formed.add(slot); self.store_log.append(topic)
        w_within = float(self.cp.mean(self.R.C.data[self.R.withinA_masks[slot]]))
        del bi, Ri
        if self.verbose:
            print(f"[episodic-dap] STORE topic={topic!r} slot={slot} w_within={w_within:.1f}", flush=True)
        return True

    # ---- RECALL (episodic READ): drive the topic-slot cue, read the dendritic dAP apical completion -------------
    def _apical(self, slot, cue_kind, lesion):
        cue = {"cue": self.cue_by_asm, "perm": self.perm_by_asm}.get(cue_kind)
        drive = [cue[slot]] if cue is not None else [None]
        if lesion:
            saved = self.R.C.data.copy(); self.R.C.data[:] = self.baseline_weights
        try:
            return _apical_up_read(self.bridge, self.R, [self.held_pos_by_asm[slot]], drive, self.p["up_thresh"])
        finally:
            if lesion:
                self.R.C.data[:] = saved

    def recall(self, topic, *, lesion=False):
        """Return the SPIKING recall record for `topic`: apical UP completion for cue/perm/nocue + a cue-specific
        completion verdict. lesion=True reads through the UNFORMED baseline weights (the load-bearing teeth)."""
        slot = self.topic_slot.get(topic)
        if slot is None:
            return {"topic": topic, "slot": None, "formed": False, "in_memory": False,
                    "apical_cue": 0.0, "apical_perm": 0.0, "apical_nocue": 0.0, "reason": "no-slot"}
        cue = self._apical(slot, "cue", lesion)
        perm = self._apical(slot, "perm", lesion)
        nocue = self._apical(slot, "nocue", lesion)
        completes = bool(cue >= COMPLETE_MIN and cue >= CUE_OVER_CTRL * (perm + 1e-6)
                         and cue >= CUE_OVER_CTRL * (nocue + 1e-6) and nocue <= CTRL_MAX)
        return {"topic": topic, "slot": slot, "formed": bool(slot in self.formed and not lesion),
                "in_memory": completes, "apical_cue": float(cue), "apical_perm": float(perm),
                "apical_nocue": float(nocue), "lesioned": bool(lesion), "reason": "spiking-dap-completion"}

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
