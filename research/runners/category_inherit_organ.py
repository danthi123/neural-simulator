"""CATEGORY-INHERITANCE ORGAN -- rung-3c's LEARNED member->context pathway, generalized to the live chat's runtime
vocabulary, with rung-3c's hand-placed context->property projection replaced by a LEARNED one, so a never-described
member inherits a property of the category it was DISCOVERED to share.

WHY (Stage-1 exit criterion C1, `research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md` section 4.1; this
wiring's prereg `research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md`). After chat
teaching builds a bird context (`the robin eats the worm`, `the robin flies in the sky`, ...) and a fish context, a
`canary` given ONLY the bird motion context (`the canary flies in the sky`) should answer "worm" to "what does the
canary eat". The live handler's strict route is an exact (agent, action) recall, so it abstains (dev seed 7).

THE MECHANISM. Two excitatory Izhikevich populations on ONE standalone SimulationBridge: `member` (one assembly slot
per heard agent) and `context` (one assembly slot per heard predicate FRAME, the verb + its object: `eat worm`,
`fly sky` -- the Stage-1 prereg's "eat-patient + motion context"). The only synapses are two all-to-all pathways, both
FLOOR-initialised (0.01: they drive nothing until learned), each on its own plasticity gate:
  * member -> context: rung-3c's learned member->context pathway (`_gnw_rung3c_learned_inheritance_derisk.py`);
  * context -> context between DIFFERENT frame slots: rung-3c's context->property projection, which rung-3c
    hand-placed (`CONTEXT_TO_REASON_W`, fixed); here it is LEARNED by the same rule from frames that are co-active.
Rung-3c's two fixes are kept: the on-bridge rate-Hebbian (pre(t-1) AND post(t)) with ZERO every-step decay, and a
strong co-drive. OBSERVE (a fact the chat just ACQUIRED): the member slot and the frame slot are co-driven for a
within-turn rehearsal (`REPS` x `STREAM_STEPS` steps, each from the clean quiescent state) with the gate open. When
the robin's second fact (`fly sky`) is rehearsed, the robin's learned member->frame drive re-activates its first frame
(`eat worm`) while `fly sky` is driven, so `eat worm` <-> `fly sky` grows: frames that the same members carry become
associated (the emergent superordinate lives in that frame-frame structure; EMERGE-30's "those cells are the
emergent superordinate"). When the canary's only fact (`fly sky`) is rehearsed, the learned `fly sky` -> `eat worm`
fires `eat worm` while the canary is driven, so canary -> `eat worm` can grow too (pattern completion during encoding).
INFER (only when the strict route missed): drive the asked member's slot for `QUERY_STEPS`; activity spreads
member -> its frames -> (lateral) the frames associated with them. The frames whose verb is the asked verb are
ADMITTED (host); the admitted frame with the most spikes wins if it clears `FLOOR_RATE` and beats the runner-up by
`TIE_MARGIN`, else the organ ABSTAINS (moat). The winner's object is the answer. No fact table is read at read time.

WHAT IS HOST (declared; the seam audit in the prereg section 5):
  * slot recruitment (a new agent word -> the next free member slot; a new (verb, object) frame -> the next free
    context slot): the same host slot index as the composer's recruit-an-assembly;
  * the frame token itself comes from the host SVO extraction + verb lemmatization (declared scaffolds, Stage-1
    prereg section 4.2);
  * candidate ADMISSION by the asked verb, the floor and the tie margin (a k-select over spiking reads, the same
    seam class as the strict route's first match);
  * which facts reach the organ: only facts the chat ACQUIRED this session with AFFIRM polarity;
  * the organ is its own co-resident bridge, not the one-brain merged pool (the named next rung).

LESIONS (env `BRAIN_CATEGORY_INHERIT_LESION`; all READ-TIME transmission lesions: encoding is identical to the intact
arm, the weights are zeroed for the read and restored after, a CNQX/TTX-style retrieval block):
  * `1` / `all`: both learned pathways silenced -> the asked member reaches no frame -> abstain (the gate's lesion arm);
  * `lateral`: only frame->frame silenced at the read (mechanism dissection: does the read need the lateral
    association, or was the inherited link already written into member->frame during encoding?).
and one ENCODING-side ablation (a plasticity blockade, AP5-style):
  * `lateral_learn`: the frame->frame gate is never opened, so no frame-frame association forms (is the learned
    lateral association NECESSARY for inheritance at all?).

HOP ATTENUATION (`BRAIN_CATEGORY_INHERIT_HOP=1`, default off; the constants block below says what and why): the
frame -> frame pathway transmits at HOP_LATERAL_GAIN and the read lasts HOP_QUERY_STEPS, so a bridging member no
longer carries a property two category hops away. Off builds no transmission gate: byte-identical to the 09-30 organ.

Parameters were set on dev seeds 1-8 only (no gate seed), recorded in the prereg section 11. Deterministic per seed
(per-neuron heterogeneity seeded by cfg.seed; no OU noise: 300 pA OU made every link grow at encoding, dev seeds 1-3).
No `sim/` edit.
"""
from __future__ import annotations

import os

import numpy as np

# ── frozen organ parameters ──────────────────────────────────────────────────────────────────────────────────────
ASSEMBLY = 20                  # neurons per slot
N_MEMBER_SLOTS = 24            # capacity: distinct agents one session can teach (overflow -> not encoded, recorded)
N_CONTEXT_SLOTS = 24           # capacity: distinct (verb, object) frames
FLOOR_W = 0.01                 # rung-3c MEMBER_TO_CONTEXT_INIT: the all-to-all drives nothing until learned
HEBB_RATE = 0.05               # rung-3c HEBB_RATE
HEBB_MAX = 40.0                # soft bound of the learned weight (rung-3c 20; dev sweep: 40 and 60 both clean)
MEMBER_DRIVE_PA = 2500.0       # rung-3c MEMBER_DRIVE_PA
CONTEXT_DRIVE_PA = 2200.0      # rung-3c CONTEXT_DRIVE_PA
QUERY_DRIVE_PA = 2500.0        # the asked member's drive at read time (the stream's member drive)
SETTLE_STEPS = 40              # quiescent settle before the snapshot (rung-1 SETTLE_STEPS)
STREAM_STEPS = 20              # rung-3c STREAM_STEPS: co-fire steps per presentation
REPS = 12                      # within-turn rehearsal of ONE heard fact (rung-3c streamed 45 epochs of the whole set)
QUERY_STEPS = 60               # read window: the asked member driven, spikes counted per frame slot
FLOOR_RATE = 0.02              # moat: an admitted frame must fire above this (spikes per neuron per step)
TIE_MARGIN = 1.25              # the winner must exceed the runner-up by this factor, else abstain (no decision)
HETEROGENEITY = True           # per-neuron parameter heterogeneity, seeded by cfg.seed (identical neurons mode-lock)
GATE_MF = "category_learn_member_frame"   # member -> frame plasticity gate; open only while observing
GATE_LAT = "category_learn_lateral"       # frame -> frame plasticity gate; open only while observing
# ── hop attenuation (BRAIN_CATEGORY_INHERIT_HOP; default OFF; set on dev seeds 1-8 only, 2026-10-06) ─────────────
# The bridging-member defect (a bat that eats the bug and flies in the sky made "what does the canary swim" reach
# lake, dev 8/8) has two parts, both measured (research/findings/2026-10-06-c1-hop-attenuation-bridging-member.md):
# (1) at ENCODING, a frame two lateral hops from the heard one was reactivated (0.05-0.06 vs 0.09 for a first-order
# associate) and the LTP-only rule wrote a direct link between frames that no member carried together (fly sky ->
# swim lake 5.7-9.9 after the bat); (2) at the READ, the 60-step window lets the activity wave walk the merged graph
# (each lateral hop arrives ~10 steps later; the 2-hop frame fills the second half of the window).
# The option: the intrinsic frame -> frame pathway transmits at HOP_LATERAL_GAIN of its weight, at encoding and at
# the read (a recurrent associative synapse carries less current than the afferent one, so the drive decays with each
# hop: a first-order associate is still reactivated and integrative encoding still writes the canary -> eat worm
# link, a second-order one stays subthreshold), and the read lasts one retrieval cycle of HOP_QUERY_STEPS.
# Off = no transmission gate exists and the read is QUERY_STEPS: the organ is byte-identical to the 09-30 build.
TX_LAT = "category_lateral_transmission"  # transmission gate on frame -> frame (exists only when the option is on)
HOP_LATERAL_GAIN = 0.5         # frame -> frame current gain (1.0 = the 09-30 organ)
HOP_QUERY_STEPS = 35           # read window with the option on (one retrieval cycle; 60 = the 09-30 organ). Dev 1-8:
                               # 30 left the permuted read 1.25x the floor, 40 put the bridge lake read at 0.8x it;
                               # 35 balances them (permuted >= 1.4x floor, bridge lake <= 0.57x floor, worm >= 2.9x)


def category_inherit_hop_enabled() -> bool:
    """`BRAIN_CATEGORY_INHERIT_HOP`: truthy (1/true/yes/on) -> hop attenuation on (the frame -> frame pathway at
    HOP_LATERAL_GAIN, a HOP_QUERY_STEPS read). Unset or anything else -> off (the 09-30 organ)."""
    return os.environ.get("BRAIN_CATEGORY_INHERIT_HOP", "").strip().lower() in ("1", "true", "yes", "on")


def category_inherit_lesion_mode() -> str | None:
    """`BRAIN_CATEGORY_INHERIT_LESION`: unset/0/false/no/off -> None; `lateral` / `lateral_learn` -> that mode; any
    other truthy value (1/true/yes/on/all) -> 'all'."""
    v = os.environ.get("BRAIN_CATEGORY_INHERIT_LESION", "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return None
    return v if v in ("lateral", "lateral_learn") else "all"


class CategoryInheritOrgan:
    """member -> frame + frame -> frame Hebbian organ. `observe(a, v, p)` learns; `infer(a, v)` reads or abstains."""

    def __init__(self, seed: int = 42, lesion: str | None = "env", n_member_slots: int = N_MEMBER_SLOTS,
                 n_context_slots: int = N_CONTEXT_SLOTS, assembly: int = ASSEMBLY, hop="env"):
        self.seed = int(seed)
        self.hop = category_inherit_hop_enabled() if hop == "env" else bool(hop)
        self.query_steps = HOP_QUERY_STEPS if self.hop else QUERY_STEPS
        self.lesion = category_inherit_lesion_mode() if lesion == "env" else lesion
        if self.lesion not in (None, "all", "lateral", "lateral_learn"):
            raise ValueError("lesion must be None, 'all', 'lateral' or 'lateral_learn': %r" % (self.lesion,))
        self.n_member_slots = int(n_member_slots)
        self.n_context_slots = int(n_context_slots)
        self.assembly = int(assembly)
        self.member_slot = {}          # agent word -> slot index (host recruit, in order of first appearance)
        self.context_slot = {}         # (verb, object) -> slot index
        self.n_observed = 0
        self.n_overflow = 0
        self.n_reads = 0
        self.n_lesioned_reads = 0
        self.last_lesion_held = None
        self._build()

    # ── the substrate ────────────────────────────────────────────────────────────────────────────────────────────
    def _build(self):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.enums import NeuronModel
        from sim.regions import BrainRegion, RegionPathway
        from sim.backend import get_backend, to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state
        self.xp, _ = get_backend()
        n_m = self.n_member_slots * self.assembly
        n_c = self.n_context_slots * self.assembly
        regions = [
            BrainRegion(name="member", n_neurons=n_m, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
            BrainRegion(name="context", n_neurons=n_c, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        ]
        pathways = [RegionPathway(from_region="member", to_region="context", density=1.0, weight_mean=FLOOR_W,
                                  weight_jitter=0.0, plastic=True, plasticity_gate=GATE_MF)]
        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions
        cfg.region_pathways = pathways
        cfg.dt_ms = 1.0
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        cfg.seed = self.seed
        cfg.enable_nmda = True
        cfg.nmda_ratio = 0.5
        cfg.enable_hebbian_learning = True
        cfg.hebbian_learning_rate = HEBB_RATE
        cfg.hebbian_max_weight = HEBB_MAX
        cfg.hebbian_weight_decay = 0.0            # rung-3c fix (1): decay would collapse the sparse potentiation
        for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis",
                  "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
            setattr(cfg, f, False)
        cfg.enable_parameter_heterogeneity = bool(HETEROGENEITY)
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        rm = self.bridge.region_manager
        self._mem_idx = np.asarray(rm.indices("member"), dtype=np.int64)
        self._ctx_idx = np.asarray(rm.indices("context"), dtype=np.int64)
        union = dict(rm.build_wiring_plan(seed=self.seed))
        # frame -> frame between DIFFERENT slots (no within-slot loops: a slot is not its own association)
        slot_of = np.repeat(np.arange(self.n_context_slots), self.assembly)
        pre = np.repeat(self._ctx_idx, len(self._ctx_idx))
        post = np.tile(self._ctx_idx, len(self._ctx_idx))
        keep = np.repeat(slot_of, len(slot_of)) != np.tile(slot_of, len(slot_of))
        union["category_frame_lateral"] = {
            "pre_indices": pre[keep].astype(np.int64), "post_indices": post[keep].astype(np.int64),
            "initial_weights": np.full(int(keep.sum()), float(FLOOR_W), dtype=np.float32),
            "plastic": True, "plasticity_gate": GATE_LAT, "conn_type": "E_TO_E", "count": int(keep.sum())}
        if self.hop:
            union["category_frame_lateral"]["transmission_gate"] = TX_LAT
        self.bridge.inject_explicit_wiring(union, output_inhibitory_indices=None)
        self.bridge.set_plasticity_gate(GATE_MF, 0.0)             # both closed except while observing
        self.bridge.set_plasticity_gate(GATE_LAT, 0.0)
        if self.hop:
            self.bridge.set_transmission_gate(TX_LAT, float(HOP_LATERAL_GAIN))   # fixed: encoding and read
        # positions (in the CSR data array) of each learned pathway, for the read-time lesions and the audit
        coo = self.bridge.cp_connections.tocoo()
        rows, cols = np.asarray(to_host(coo.row)), np.asarray(to_host(coo.col))
        m0, m1 = int(self._mem_idx[0]), int(self._mem_idx[-1]) + 1
        c0, c1 = int(self._ctx_idx[0]), int(self._ctx_idx[-1]) + 1
        in_m = (rows >= m0) & (rows < m1)
        pre_c, post_c = (rows >= c0) & (rows < c1), (cols >= c0) & (cols < c1)
        self._pos = {"member_frame": np.nonzero(in_m & post_c)[0], "lateral": np.nonzero(pre_c & post_c)[0]}
        self._pos_dev = {k: self.xp.asarray(v) for k, v in self._pos.items()}
        self._rows, self._cols = rows, cols
        self.n_learn_synapses = {k: int(v.size) for k, v in self._pos.items()}
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE_STEPS):
            self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0
        self._q0 = _snapshot_state(self.bridge, self.xp)         # the clean quiescent dynamical state
        self._dev_cache = {}

    def _slot_idx(self, kind, slot):
        key = (kind, slot)
        if key not in self._dev_cache:
            base = self._mem_idx if kind == "m" else self._ctx_idx
            self._dev_cache[key] = self.xp.asarray(base[slot * self.assembly:(slot + 1) * self.assembly])
        return self._dev_cache[key]

    @staticmethod
    def _recruit(table, key, cap):
        if key in table:
            return table[key]
        if len(table) >= cap:
            return None
        table[key] = len(table)
        return table[key]

    def _restore(self):
        from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state
        self.bridge.cp_external_input_current[:] = 0.0
        _restore_state(self.bridge, self._q0)
        self.bridge.cp_external_input_current[:] = 0.0

    # ── learning: one heard fact ─────────────────────────────────────────────────────────────────────────────────
    def observe(self, agent, action, patient) -> dict:
        """Co-drive the agent's member slot and the (action, patient) frame slot with the plasticity gate open
        (`REPS` presentations of `STREAM_STEPS` steps, each from the clean quiescent state). Returns a trace dict."""
        a, v, p = (str(agent).lower(), str(action).lower(), str(patient).lower())
        ms = self._recruit(self.member_slot, a, self.n_member_slots)
        cs = self._recruit(self.context_slot, (v, p), self.n_context_slots)
        if ms is None or cs is None:
            self.n_overflow += 1
            return {"observed": False, "reason": "capacity", "member_slot": ms, "context_slot": cs}
        m_dev, c_dev = self._slot_idx("m", ms), self._slot_idx("c", cs)
        f32 = self.xp.float32
        self.bridge.set_plasticity_gate(GATE_MF, 1.0)
        self.bridge.set_plasticity_gate(GATE_LAT, 0.0 if self.lesion == "lateral_learn" else 1.0)
        try:
            for _ in range(REPS):
                self._restore()
                for _ in range(STREAM_STEPS):
                    self.bridge.cp_external_input_current[:] = 0.0
                    self.bridge.cp_external_input_current[m_dev] = f32(MEMBER_DRIVE_PA)
                    self.bridge.cp_external_input_current[c_dev] = f32(CONTEXT_DRIVE_PA)
                    self.bridge._run_one_simulation_step()
        finally:
            self.bridge.set_plasticity_gate(GATE_MF, 0.0)
            self.bridge.set_plasticity_gate(GATE_LAT, 0.0)
            self._restore()
        self.n_observed += 1
        return {"observed": True, "member_slot": ms, "context_slot": cs}

    # ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────
    def read(self, agent) -> dict | None:
        """Drive the agent's member slot for `QUERY_STEPS` and count spikes per frame slot. Returns {(verb, object):
        rate} (spikes per neuron per step) for every recruited frame, or None if the agent was never heard."""
        a = str(agent).lower()
        if a not in self.member_slot:
            return None
        from sim.backend import to_host
        xp = self.xp
        m_dev = self._slot_idx("m", self.member_slot[a])
        saved = []
        if self.lesion in ("all", "lateral"):              # read-time transmission lesion (encoding untouched)
            parts = ("member_frame", "lateral") if self.lesion == "all" else ("lateral",)
            w = self.bridge.cp_connections.data
            for part in parts:
                pos = self._pos_dev[part]
                saved.append((pos, w[pos].copy()))
                w[pos] = xp.float32(0.0)
            self.bridge.mark_weights_edited()
            self.n_lesioned_reads += 1
        try:
            self._restore()
            ctx_dev = xp.asarray(self._ctx_idx)
            acc = xp.zeros(len(self._ctx_idx), dtype=xp.float64)
            for _ in range(self.query_steps):
                self.bridge.cp_external_input_current[:] = 0.0
                self.bridge.cp_external_input_current[m_dev] = xp.float32(QUERY_DRIVE_PA)
                self.bridge._run_one_simulation_step()
                acc += self.bridge.cp_firing_states[ctx_dev].astype(xp.float64)
            acc_h = np.asarray(to_host(acc))
            # docs/TERMS.md "lesion": verify the cut STILL HOLDS at the end of the measured window
            self.last_lesion_held = (None if not saved else all(
                float(to_host(xp.abs(self.bridge.cp_connections.data[pos]).max())) == 0.0 for pos, _v in saved))
        finally:
            if saved:
                for pos, vals in saved:
                    self.bridge.cp_connections.data[pos] = vals
                self.bridge.mark_weights_edited()
            self._restore()
        self.n_reads += 1
        denom = float(self.query_steps * self.assembly)
        return {frame: float(acc_h[s * self.assembly:(s + 1) * self.assembly].sum()) / denom
                for frame, s in self.context_slot.items()}

    def infer(self, agent, action) -> dict:
        """The strict-route-miss read: which object does the asked (agent, action) reach through the learned
        pathways? Returns {"patient": str|None, "reason": ..., "admitted": [[object, rate], ...], "rates": {...}}."""
        v = str(action).lower()
        rates = self.read(agent)
        if rates is None:
            out = {"patient": None, "reason": "agent_never_observed", "rates": None, "admitted": [],
                   "lesion": self.lesion}
            if self.hop:
                out.update(hop=True, hop_lateral_gain=HOP_LATERAL_GAIN, query_steps=self.query_steps)
            return out
        admitted = sorted(((r, fr[1]) for fr, r in rates.items() if fr[0] == v), key=lambda t: (-t[0], t[1]))
        out = {"rates": {"%s %s" % fr: round(r, 4) for fr, r in rates.items()},
               "admitted": [[p, round(r, 4)] for r, p in admitted], "lesion": self.lesion,
               "lesion_held": self.last_lesion_held,
               "floor_rate": FLOOR_RATE, "tie_margin": TIE_MARGIN}
        if self.hop:                                       # recorded only when on (off: the 09-30 trace, unchanged)
            out.update(hop=True, hop_lateral_gain=HOP_LATERAL_GAIN, query_steps=self.query_steps)
        if not admitted or admitted[0][0] < FLOOR_RATE:
            out.update(patient=None, reason="below_floor")
        elif len(admitted) > 1 and admitted[1][0] >= FLOOR_RATE and admitted[0][0] < TIE_MARGIN * admitted[1][0]:
            out.update(patient=None, reason="tie")
        else:
            out.update(patient=admitted[0][1], reason="reached")
            if self.lesion is None:
                out["attribution"] = self.pathway_attribution(agent, (v, admitted[0][1]), admitted[0][0])
        return out

    def pathway_attribution(self, agent, frame, rate_intact) -> dict:
        """WHOSE are the winner's spikes? Re-read the same member with each read-time lesion (deterministic: every
        read starts from the same quiescent state and restores the weights) and ask `tools.lab.attributable_to`
        what fraction of the winning frame's rate vanishes when a pathway is cut (treatment = the intact rate, control
        = the rate with the pathway silenced): `learned_pathways` (both silenced; 1.0 = all of it is theirs),
        `lateral_at_read` (frame->frame only; small = the inherited link was already written into member->frame at
        encoding)."""
        from tools.lab import attributable_to
        out = {"rate_intact": round(float(rate_intact), 4)}
        saved = self.lesion
        try:
            for mode, key in (("all", "learned_pathways"), ("lateral", "lateral_at_read")):
                self.lesion = mode
                r = (self.read(agent) or {}).get(frame, 0.0)
                out["rate_" + key + "_cut"] = round(float(r), 4)
                out[key] = attributable_to("category organ %s -> %s %s, %s cut" % (agent, frame[0], frame[1], key),
                                           float(rate_intact), float(r))
        finally:
            self.lesion = saved
        return out

    def weight_audit(self) -> dict:
        """Mean learned weight per block (rung-3c's floor-before / selective-after audit). Read-only.
        {"member_frame": {agent: {"verb object": w}}, "lateral": {"verb object": {"verb object": w}}}."""
        from sim.backend import to_host
        w = np.asarray(to_host(self.bridge.cp_connections.data))
        A = self.assembly
        m0, c0 = int(self._mem_idx[0]), int(self._ctx_idx[0])
        out = {"member_frame": {}, "lateral": {}}
        pmf, plat = self._pos["member_frame"], self._pos["lateral"]
        ms, cs, wmf = (self._rows[pmf] - m0) // A, (self._cols[pmf] - c0) // A, w[pmf]
        for a, s in self.member_slot.items():
            out["member_frame"][a] = {"%s %s" % fr: round(float(wmf[(ms == s) & (cs == t)].mean()), 3)
                                      for fr, t in self.context_slot.items()}
        ls, lt, wl = (self._rows[plat] - c0) // A, (self._cols[plat] - c0) // A, w[plat]
        for fr1, s in self.context_slot.items():
            out["lateral"]["%s %s" % fr1] = {"%s %s" % fr2: round(float(wl[(ls == s) & (lt == t)].mean()), 3)
                                             for fr2, t in self.context_slot.items() if t != s}
        return out
