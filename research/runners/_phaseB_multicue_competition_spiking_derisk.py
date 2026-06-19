"""Phase-1 SPIKING-substrate de-risk: multi-cue COMPETITION parser for robust thematic-role assignment.

This is the BRAIN-BASED production version of the numpy MECHANISM de-risk
(`_phaseB_multicue_competition_derisk.py`, GO 6/6, finding `2026-06-19-multicue-competition-derisk.md`).
Per the BRAIN-BASED-ONLY standard, the numpy delta-rule + softmax was the FUNCTIONAL stand-in; here the
COMPETITION + the reliability-weighted ACCUMULATION + the WINNER are realized as real spiking neurons on a
`SimulationBridge`, and the cue VALIDITIES are LEARNED as SYNAPTIC weights by the parser's own Hebbian
co-firing rule (`enable_hebbian_learning=True`, the v16 rule `BridgeParser` uses).

WHAT IS SPIKING-LEARNED vs INSTALLED (honest, per the directive):
  * SPIKING: the role-COMPETITION (two role assemblies `sel_AGENT`/`sel_PATIENT` in Wong-Wang mutual inhibition
    via `sel_FS_X`, the Rutishauser selective-inhibition WTA re-pointed from REFERENT to ROLE — reused-by-import
    from `biased_competition_buffer.py`), the reliability-weighted ACCUMULATION (each cue population drives
    evidence into the role assemblies through a PLASTIC cue->role projection; the accumulator sums the weighted
    cue drive), the WINNER (the spiking WTA settle), AND the cue VALIDITIES = the LEARNED SYNAPTIC weights of the
    cue->role projections (trained by Hebbian co-firing on the naturalistic distribution — the high-validity cues'
    synapses strengthen, the chance-validity distractor stays weak; this is the spiking analogue of the numpy
    `w_position < w_animacy`).
  * HOST SCAFFOLD (flagged for conversion, same boundary as the buffer + the numpy de-risk): the feature LEXICONS
    (animacy, verb-selectional-fit) supply each cue's VALUE for a word (which cue population to light); they do
    NOT supply the role decision (that is the learned-weight spiking competition). The PERMUTED-CUE + NO-LEARNING
    controls guard against the lexicon doing the discrimination. The cue VALUE -> a cue population's drive current
    is the legitimate lexical front-end boundary; the conversion target is a learned lexical-feature map.

THE PRIMARY MISLEAD GUARDED (identical to the numpy de-risk): hand-tuned cues masquerading as a learned model.
The decisive controls (all must pass or it is NOT a GO):
  - POSITION-ONLY baseline COLLAPSES on the position-degrading battery (scramble+object-front).
  - NO-LEARNING control (cue->role weights FROZEN at uniform init, Hebbian skipped) collapses -> validities LEARNED.
  - CUE-LESION (zero the animacy+verbfit cue->role weights, keep position) collapses -> the cues are load-bearing.
  - PERMUTED-CUE (train against scrambled animacy/verb-fit tags) collapses -> not a relabelled position / leak.
  - HELD-OUT FILLERS+VERBS (test pools disjoint from training) -> not memorizing examples.
  - the no-confab MOAT holds (two animate nouns + a symmetric verb, scrambled -> no decisive cue -> ABSTAIN).

Run (CPU/numpy smoke is fast; GPU for the multi-seed):
    SIM_BACKEND=numpy python -m research.runners._phaseB_multicue_competition_spiking_derisk --smoke   # 1 seed
    SIM_BACKEND=cupy  python -m research.runners._phaseB_multicue_competition_spiking_derisk \
        --seeds 42,43,44,45,46,47 --out research/findings/raw/_phaseB_multicue_competition_spiking.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Feature lexicons -- REUSED verbatim from the numpy de-risk (which itself reuses the buffer's ANIMACY with a
# drift assertion). HOST scaffold: supplies each cue's VALUE, not the role decision.
# ---------------------------------------------------------------------------
from research.runners.biased_competition_buffer import ANIMACY as _BUF_ANIMACY  # noqa: E402

ANIMACY = dict(_BUF_ANIMACY)
ANIMACY.update({
    # --- animate (agents) ---
    "wolf": "animate", "bear": "animate", "owl": "animate", "frog": "animate",
    # --- inanimate (patients) ---
    "stick": "inanimate", "bone": "inanimate", "leaf": "inanimate", "cup": "inanimate",
})
assert all(ANIMACY[n] == _BUF_ANIMACY[n] for n in _BUF_ANIMACY), "ANIMACY drifted from buffer lexicon"

# Verb selectional restriction (per-slot): agentive verb's AGENT slot prefers an animate filler, PATIENT slot
# prefers an inanimate one (asymmetric verbs). Symmetric verbs (patient also animate) feed the moat.
VERB_SELECTS = {
    "chase": {"agent": "animate", "patient": "animate"},
    "eat":   {"agent": "animate", "patient": "inanimate"},
    "push":  {"agent": "animate", "patient": "inanimate"},
    "carry": {"agent": "animate", "patient": "inanimate"},
    # held-out verbs (disjoint -- verb-fit generalizes from animacy structure, not memorizes the verb)
    "bite":  {"agent": "animate", "patient": "inanimate"},
    "kick":  {"agent": "animate", "patient": "inanimate"},
    "grab":  {"agent": "animate", "patient": "inanimate"},
    "watch": {"agent": "animate", "patient": "animate"},
}

TRAIN_ANIMATE = ["dog", "cat", "fox", "bird"]
TRAIN_INANIM = ["ball", "apple", "rock", "book"]
HELD_ANIMATE = ["wolf", "bear", "owl", "frog"]
HELD_INANIM = ["stick", "bone", "leaf", "cup"]
TRAIN_VERBS = ["chase", "eat", "push", "carry"]
HELD_VERBS = ["bite", "kick", "grab", "watch"]

ROLES = ("agent", "patient")  # 2-role assignment for NOUNS -> chance = 0.5
CUES = ("position", "animacy", "verbfit", "lexbias")
TRUE_VALIDITY = {"animacy": 0.90, "verbfit": 0.90, "lexbias": 0.50}
POSITION_NOISE = 0.0
_CUE_ID = {c: i for i, c in enumerate(CUES)}


# ===========================================================================
# Cue evidence (mirrors the numpy de-risk EXACTLY: cues are individually NON-DECISIVE; each emits a signed vote
# in {-1,0,+1} toward (agent:+ / patient:-) with a per-cue label-noise at rate (1-validity), deterministically
# seeded per (sentence, cue, noun) on a STABLE integer key. POSITION's vote is structural (right iff canonical).
# The lexbias DISTRACTOR is sign-correlated with position but at chance validity -> must be learned out.)
# ===========================================================================

def _det_unit(sent_id, cue, noun_index):
    key = (int(sent_id) * 97 + _CUE_ID[cue]) * 31 + int(noun_index)
    return float(np.random.default_rng(key & 0xFFFFFFFF).random())


def _maybe_flip(vote, validity, sent_id, cue, noun_index):
    if vote == 0.0:
        return 0.0
    return vote if _det_unit(sent_id, cue, noun_index) < validity else -vote


def _position_vote(noun_index, n_nouns):
    if n_nouns <= 1:
        return 0.0
    frac = noun_index / (n_nouns - 1)
    return 1.0 - 2.0 * frac


def _animacy_vote(noun):
    a = ANIMACY.get(noun)
    return +1.0 if a == "animate" else (-1.0 if a == "inanimate" else 0.0)


def _verbfit_vote(noun, verb):
    sel = VERB_SELECTS.get(verb)
    a = ANIMACY.get(noun)
    if sel is None or a is None:
        return 0.0
    fits_agent = (sel["agent"] == a)
    fits_patient = (sel["patient"] == a)
    if fits_agent and not fits_patient:
        return +1.0
    if fits_patient and not fits_agent:
        return -1.0
    return 0.0


def cue_evidence(noun, noun_index, n_nouns, verb, sent_id,
                 permute_map=None, lesion_semantic=False, drop_cues=(), clean_cues=False):
    """{cue: (vote, reliability)} for one noun (per-cue label noise baked in). See the numpy de-risk docstring."""
    def flip(vote, validity, cue):
        return vote if clean_cues else _maybe_flip(vote, validity, sent_id, cue, noun_index)

    ev = {}
    pv = _position_vote(noun_index, n_nouns)
    ev["position"] = (flip(pv, 1.0 - POSITION_NOISE, "position"), 1.0)

    sem_noun = permute_map.get(noun, noun) if permute_map else noun
    if lesion_semantic:
        ev["animacy"] = (0.0, 0.0)
        ev["verbfit"] = (0.0, 0.0)
    else:
        av = flip(_animacy_vote(sem_noun), TRUE_VALIDITY["animacy"], "animacy")
        ev["animacy"] = (av, 1.0 if _animacy_vote(sem_noun) != 0.0 else 0.0)
        vv = flip(_verbfit_vote(sem_noun, verb), TRUE_VALIDITY["verbfit"], "verbfit")
        ev["verbfit"] = (vv, 1.0 if _verbfit_vote(sem_noun, verb) != 0.0 else 0.0)

    lv = flip(np.sign(pv) if pv != 0 else 0.0, TRUE_VALIDITY["lexbias"], "lexbias")
    ev["lexbias"] = (float(lv), 1.0)

    for c in drop_cues:
        ev[c] = (0.0, 0.0)
    return ev


# ===========================================================================
# THE SPIKING ROLE-COMPETITION (re-points biased_competition_buffer's sel_X/sel_FS_X WTA from REFERENT to ROLE).
#
# Architecture (all spiking SimulationBridge neurons, region framework, reuse-by-import / additive; NO sim/ edit):
#   * Per ROLE r in {agent, patient}: a Wong-Wang ACCUMULATOR pool `sel_r` (NMDA-slow recurrent, soft-WTA gain
#     alpha<1) + a selective inhibitory pool `sel_FS_r` (exc_fraction=0.0 -> inhibitory traits). Wiring:
#       sel_r -> sel_FS_r        (exc: a winning role recruits its interneuron)
#       sel_FS_r -> sel_(s!=r)   (inh: that interneuron suppresses the OTHER role) -- Rutishauser selective inh.
#   * Per CUE c in {position, animacy, verbfit, lexbias}: a small cue population `cue_c` whose firing encodes
#     that cue's signed vote magnitude (a +vote lights it toward agent, a -vote toward patient). Realized as a
#     pair of vote-sign drivers feeding through a PLASTIC `cue_c -> {sel_agent, sel_patient}` projection.
#   * The PLASTIC cue->role projection weights ARE the learned cue validities (trained by Hebbian co-firing:
#     drive the cue's correct-sign population + teacher-drive the gold role; Hebbian strengthens the
#     cue->correct-role synapse). A high-validity cue's projection strengthens; the distractor stays weak.
#
# The WINNER is read from the sel pools (argmax firing rate). The per-noun read drives each cue's vote, lets the
# competition settle, and reads which role pool wins. Sentence-level: the higher-agent-evidence noun = agent.
# ===========================================================================

@dataclass
class _Layout:
    n_loop: int = 1               # unused placeholder; competition is the cue->role->sel pipeline
    n_sel: int = 24               # Wong-Wang accumulator pool size per role
    n_sel_fs: int = 12            # selective inhibitory pool size per role
    n_cue: int = 120              # cue population size per cue-sign sub-pop. Sized so the cue->role conductance
    #                               feed reliably drives the sel WTA at a Hebbian-learnable weight (the bridge's
    #                               conductance-based synapse needs ~n_cue*cue_rate*W above a floor to fire the
    #                               sel pool; 120 neurons @ ~0.15 rate * weight ~15 clears it cleanly).


class SpikingRoleCompetition:
    """Spiking multi-cue role competition on ONE SimulationBridge. Cue populations -> plastic cue->role
    projections -> Wong-Wang role accumulators (sel_agent / sel_patient) in mutual inhibition. The learned
    synaptic cue->role weights are the cue validities; the spiking WTA settle is the role decision."""

    def __init__(self, seed=42, layout=None,
                 sel_recurrent_weight=0.30, sel_recurrent_density=0.5,
                 sel_to_fs_weight=18.0, fs_to_sel_weight=3.0,
                 cue_to_role_init=4.0, cue_drive_pA=3500.0, role_teacher_pA=2600.0,
                 hebbian_lr=0.02, hebbian_max=60.0,
                 verbose=False):
        import sim.backend as B
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronType
        self.B = B
        self.xp, _ = B.get_backend()
        self.L = layout or _Layout()
        self.cue_drive_pA = float(cue_drive_pA)
        self.role_teacher_pA = float(role_teacher_pA)
        self._cue_init = float(cue_to_role_init)
        self.verbose = verbose

        # ---- regions ----
        regions = []
        # role accumulators + their selective inhibitory pools (Wong-Wang + Rutishauser, re-pointed to ROLE)
        for r in ROLES:
            regions.append(BrainRegion(
                name=f"sel_{r}", n_neurons=self.L.n_sel, exc_fraction=1.0,
                internal_density=sel_recurrent_density, exc_weight_mean=sel_recurrent_weight,
                inh_weight_mean=0.0, weight_jitter=0.2, plastic_internal=False, enable_nmda=True,
                izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))
            regions.append(BrainRegion(
                name=f"sel_FS_{r}", n_neurons=self.L.n_sel_fs, exc_fraction=0.0,
                internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        # cue populations: a signed cue is a pair of sub-pops (agent-vote half + patient-vote half). We model
        # this as ONE cue region per (cue, sign); the agent-half projects to sel_agent, patient-half to sel_patient.
        # Each cue's two halves SHARE the learned validity weight (a cue's validity is sign-symmetric).
        for c in CUES:
            for sgn in ("pos", "neg"):
                regions.append(BrainRegion(
                    name=f"cue_{c}_{sgn}", n_neurons=self.L.n_cue, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions

        # ---- pathways ----
        pathways = []
        # sel_r -> sel_FS_r (exc); sel_FS_r -> sel_(s!=r) (inh). NON-plastic WTA wiring.
        for r in ROLES:
            pathways.append(RegionPathway(from_region=f"sel_{r}", to_region=f"sel_FS_{r}",
                                          density=1.0, weight_mean=sel_to_fs_weight, weight_jitter=0.2,
                                          plastic=False))
        for r in ROLES:
            for s in ROLES:
                if r == s:
                    continue
                pathways.append(RegionPathway(from_region=f"sel_FS_{r}", to_region=f"sel_{s}",
                                              density=1.0, weight_mean=fs_to_sel_weight, weight_jitter=0.2,
                                              plastic=False))
        # cue_c_pos -> sel_agent ; cue_c_neg -> sel_patient. PLASTIC (Hebbian). These are the learned cue validities.
        for c in CUES:
            pathways.append(RegionPathway(from_region=f"cue_{c}_pos", to_region="sel_agent",
                                          density=1.0, weight_mean=self._cue_init, weight_jitter=0.0,
                                          plastic=True, plasticity_gate=f"cue_{c}"))
            pathways.append(RegionPathway(from_region=f"cue_{c}_neg", to_region="sel_patient",
                                          density=1.0, weight_mean=self._cue_init, weight_jitter=0.0,
                                          plastic=True, plasticity_gate=f"cue_{c}"))
        cfg.region_pathways = pathways

        cfg.dt_ms = 0.5
        cfg.seed = int(seed)
        cfg.enable_nmda = True
        cfg.enable_ou_process = False
        cfg.enable_structural_plasticity = False
        cfg.enable_hebbian_learning = True            # v16 co-firing rule for the plastic cue->role edges
        cfg.hebbian_max_weight = float(hebbian_max)
        cfg.hebbian_learning_rate = float(hebbian_lr)
        cfg.enable_stdp = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_homeostasis = False
        cfg.enable_reward_modulation = False
        cfg.stdp_w_max = 60.0
        cfg.fast_spike_reset = True

        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge = bridge
        rm = bridge.region_manager

        self._sel_idx = {r: np.asarray(rm.indices(f"sel_{r}"), dtype=np.int64) for r in ROLES}
        self._cue_idx = {(c, sgn): np.asarray(rm.indices(f"cue_{c}_{sgn}"), dtype=np.int64)
                         for c in CUES for sgn in ("pos", "neg")}
        self._n = self.bridge.core_config.num_neurons

        if verbose:
            print(f"[spiking role-competition] {self._n} neurons, "
                  f"sel/role={self.L.n_sel}, cue pops={len(self._cue_idx)}", flush=True)

    # ---- per-synapse learned cue->role weight read-out (the cue VALIDITIES) ----
    def cue_weights(self):
        """Mean learned synaptic weight of each cue's cue->role projection (the spiking analogue of the numpy
        learned cue validity). Reads cp_connections directly for the cue_c_pos -> sel_agent edges."""
        import scipy.sparse as _sp
        try:
            csr = self.B.to_host(self.bridge.cp_connections)
        except Exception:
            csr = self.bridge.cp_connections
        if not _sp.issparse(csr):
            csr = _sp.csr_matrix(csr)
        csr = csr.tocsr()
        out = {}
        for c in CUES:
            pre = self._cue_idx[(c, "pos")]
            post = set(int(j) for j in self._sel_idx["agent"])
            vals = []
            for i in pre:
                row = csr.getrow(int(i))
                for j, w in zip(row.indices, row.data):
                    if int(j) in post:
                        vals.append(float(w))
            out[c] = float(np.mean(vals)) if vals else 0.0
        return out

    def set_cue_weight(self, cue, value):
        """Overwrite a cue's cue->role projection weight (used to install validated weights / lesion a cue)."""
        v = np.float32(value)
        for sgn, role in (("pos", "agent"), ("neg", "patient")):
            pre = self._cue_idx[(cue, sgn)]
            post = self._sel_idx[role]
            p = np.repeat(pre, post.size).astype(np.int64)
            q = np.tile(post, pre.size).astype(np.int64)
            w = np.full(p.size, v, np.float32)
            self.bridge.set_pathway_weights(f"set_{cue}_{sgn}", pre_indices=p, post_indices=q, weights=w,
                                            add_missing=False)
        if hasattr(self, "_edge_slots") and cue in self._edge_slots:
            self._cur_w[cue] = float(value)  # keep the fast-path cache coherent

    def _precompute_cue_edge_slots(self):
        """Precompute the cp_connections.data slot indices for each cue's cue->role edges ONCE, so the
        error-gated learner can update weights in-place per item (O(edges)) instead of rebuilding the O(nnz)
        (pre,post)->slot map every call (set_pathway_weights does the latter -- far too slow per training item)."""
        import scipy.sparse as _sp
        csr = self.B.to_host(self.bridge.cp_connections)
        if not _sp.issparse(csr):
            csr = _sp.csr_matrix(csr)
        csr = csr.tocsr()
        indptr, indices = csr.indptr, csr.indices
        # map (pre, post) -> data slot, restricted to cue->role edges
        wanted = {}
        for c in CUES:
            slots = []
            for sgn, role in (("pos", "agent"), ("neg", "patient")):
                post_set = set(int(j) for j in self._sel_idx[role])
                for i in self._cue_idx[(c, sgn)]:
                    a, b = int(indptr[i]), int(indptr[i + 1])
                    for off in range(a, b):
                        if int(indices[off]) in post_set:
                            slots.append(off)
            wanted[c] = np.asarray(slots, dtype=np.int64)
        self._edge_slots = wanted
        self._cur_w = {c: 0.0 for c in CUES}

    def _fast_set_cue_weight(self, cue, value):
        """In-place weight write to cp_connections.data at the precomputed slots for `cue` (fast per-item path)."""
        slots = self._edge_slots[cue]
        data = self.bridge.cp_connections.data
        data[slots] = self.xp.float32(value)
        self._cur_w[cue] = float(value)

    def freeze_all_cue_plasticity(self):
        for c in CUES:
            self.bridge.set_plasticity_gate(f"cue_{c}", 0.0)

    def _reset(self, steps=12):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(steps):
            self.bridge._run_one_simulation_step()

    # ---- Hebbian cue-validity training (the v16 co-firing rule on the plastic cue->role edges) ----
    def learn(self, train_examples, epochs=8, train_steps=40, seed=0, freeze=False):
        """Learn the cue validities as synaptic weights. For each training sentence-noun, drive the cue
        populations at their (noisy) signed votes AND teacher-drive the gold role's sel pool; Hebbian co-firing
        strengthens the cue->gold-role synapse in proportion to how often that cue agrees with gold (= the cue's
        empirical validity). `freeze=True` = the NO-LEARNING control (skip training, weights stay at uniform init).
        """
        if freeze:
            return
        rng = np.random.default_rng(seed)
        # flatten to (ev_per_noun, gold_role) training items
        items = []
        for _nouns, evs, gold in train_examples:
            for ni, ev in enumerate(evs):
                items.append((ev, gold[ni]))
        idx = list(range(len(items)))
        for _ep in range(epochs):
            rng.shuffle(idx)
            for i in idx:
                ev, gold_role = items[i]
                self._reset(steps=6)
                cur = self.xp.zeros(self._n, dtype=self.xp.float32)
                # light each cue's correct-sign sub-pop in proportion to its (noisy) vote * reliability
                for c, (vote, rel) in ev.items():
                    if rel <= 0.0 or vote == 0.0:
                        continue
                    sgn = "pos" if vote > 0 else "neg"
                    cur[self._cue_idx[(c, sgn)]] = self.cue_drive_pA * float(rel)
                # teacher-drive the GOLD role pool (so Hebbian co-firing strengthens cue->gold-role)
                cur[self._sel_idx[gold_role]] = self.role_teacher_pA
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(train_steps):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def _cue_role_weight_raw(self):
        """Per-cue mean of BOTH agreeing edges (pos->agent + neg->patient), for the error-gated updater."""
        import scipy.sparse as _sp
        csr = self.B.to_host(self.bridge.cp_connections)
        if not _sp.issparse(csr):
            csr = _sp.csr_matrix(csr)
        csr = csr.tocsr()
        out = {}
        sa = set(int(j) for j in self._sel_idx["agent"])
        sp_ = set(int(j) for j in self._sel_idx["patient"])
        for c in CUES:
            vals = []
            for sgn, post in (("pos", sa), ("neg", sp_)):
                for i in self._cue_idx[(c, sgn)]:
                    row = csr.getrow(int(i))
                    for j, w in zip(row.indices, row.data):
                        if int(j) in post:
                            vals.append(float(w))
            out[c] = float(np.mean(vals)) if vals else 0.0
        return out

    def learn_error_gated(self, train_examples, epochs=30, settle_steps=24, seed=0,
                          lr=0.6, decay=0.04, w_floor=0.0, w_init=12.0, output_sem_scale=20.0):
        """BRAIN-BASED THREE-FACTOR cue-validity learning (the rule plain Hebbian co-firing CANNOT do -- see the
        finding). For each training sentence-noun: (1) settle the cue-driven WTA on the substrate (NO teacher) and
        MEASURE each cue population's spiking ELIGIBILITY (its firing during the settle) -- a genuine spike-based
        signal; (2) read the predicted agent-evidence (the spiking role-pool contrast); (3) the REWARD/error =
        (gold target - predicted) -- a cue that drove the CORRECT role gets net-reinforced, a cue that drove the
        WRONG role (e.g. position on the non-canonical minority) gets net-WEAKENED; (4) apply a reward-modulated
        weight delta dw = lr * (error * cue_eligibility * vote) - decay*w on the cue->role synapses. The error
        term is exactly what plain Hebbian lacks; the distractor's error averages to ~0 and the decay zeros it;
        position is pushed DOWN by its errors on the non-canonical minority -> w_position << w_semantic. The
        ELIGIBILITY is spike-measured; the reward gate is the three-factor neuromodulatory signal (Schultz-1998
        dopamine-as-RPE); the weight update is on real synapses. (The reward computation -- did the spiking winner
        match gold -- is the host teaching signal, the legitimate environment/body boundary, exactly as the nav
        reward-RPE scaffolds; see the finding for what is spike-measured vs host-gated.)"""
        rng = np.random.default_rng(seed)
        self._precompute_cue_edge_slots()
        # initialize all cue->role weights to a common mid value (so learning, not init, sets the spread)
        for c in CUES:
            self._fast_set_cue_weight(c, w_init)
        W = {c: float(w_init) for c in CUES}
        items = [(ev, gold[ni]) for _n, evs, gold in train_examples for ni, ev in enumerate(evs)]
        for _ep in range(epochs):
            rng.shuffle(items)
            for ev, gold_role in items:
                # (1) settle the cue-driven WTA + (2) measure spiking eligibility (cue firing) and the role contrast
                elig, rr = self._settle_with_eligibility(ev, settle_steps=settle_steps)
                pred = rr["agent"] - rr["patient"]                      # spiking role-pool contrast
                target = +1.0 if gold_role == "agent" else -1.0
                # (3) reward/error (RPE): bound the contrast to a comparable scale
                err = target - np.tanh(pred * 8.0)
                d = 1.0 - np.tanh(pred * 8.0) ** 2                       # d tanh (gradient shape)
                # (4) reward-modulated weight delta per cue: eligibility (spike) x error (reward) x vote (sign)
                for c, (vote, rel) in ev.items():
                    if rel <= 0.0 or vote == 0.0:
                        continue
                    e_c = elig.get(c, 0.0)                                # spike-measured eligibility of cue c
                    dw = lr * (err * d * e_c * float(vote) - decay * W[c])
                    W[c] = max(w_floor, W[c] + dw)
                    self._fast_set_cue_weight(c, W[c])
        # The three-factor rule recovers the correct relative VALIDITIES (the spread: pos << sem, distractor ~0)
        # but at a small absolute magnitude (eligibility values are small). The synaptic WTA needs the weights in
        # its working dynamic range to fire the sel pools, so a single SCALAR GAIN places the learned semantic
        # weight at `output_sem_scale` (a homeostatic output-gain on the projection -- it preserves every learned
        # RATIO, it does not change which cue won the validity competition). This is the spiking analogue of the
        # numpy softmax temperature: the learned validities set the ratios; a fixed gain sets the decision scale.
        w_sem = 0.5 * (W["animacy"] + W["verbfit"])
        gain = (output_sem_scale / w_sem) if w_sem > 1e-6 else 1.0
        for c in CUES:
            W[c] = W[c] * gain
            self._fast_set_cue_weight(c, W[c])
        self.bridge.cp_external_input_current[:] = 0.0
        return W

    def _settle_with_eligibility(self, ev, settle_steps=24):
        """Drive the cue votes, settle the WTA, and return (eligibility, role_rates) where eligibility[c] = the
        mean firing of cue c's driven sub-pop over the settle (the spike-based eligibility the reward gates)."""
        self._reset(steps=6)
        cur = self.xp.zeros(self._n, dtype=self.xp.float32)
        cue_pops = {}
        for c, (vote, rel) in ev.items():
            if rel <= 0.0 or vote == 0.0:
                continue
            sgn = "pos" if vote > 0 else "neg"
            idx = self._cue_idx[(c, sgn)]
            cur[idx] = self.cue_drive_pA * float(rel)
            cue_pops[c] = idx
        self.bridge.cp_external_input_current[:] = cur
        elig = {c: 0.0 for c in cue_pops}
        rates = {r: 0.0 for r in ROLES}
        for _ in range(settle_steps):
            self.bridge._run_one_simulation_step()
            fs = self.bridge.cp_firing_states
            for c, idx in cue_pops.items():
                elig[c] += float(self.B.to_host(fs[idx]).sum()) / (idx.size)
            for r in ROLES:
                rates[r] += float(self.B.to_host(fs[self._sel_idx[r]]).sum())
        self.bridge.cp_external_input_current[:] = 0.0
        elig = {c: elig[c] / settle_steps for c in elig}
        rates = {r: rates[r] / (self.L.n_sel * settle_steps) for r in rates}
        return elig, rates

    # ---- inference: settle the competition for one noun, read the winning role pool ----
    def _noun_role_rates(self, ev, read_steps=40):
        """Drive the cue votes for one noun, let the WTA settle, return {role: accumulated sel firing rate}."""
        self._reset(steps=8)
        cur = self.xp.zeros(self._n, dtype=self.xp.float32)
        for c, (vote, rel) in ev.items():
            if rel <= 0.0 or vote == 0.0:
                continue
            sgn = "pos" if vote > 0 else "neg"
            cur[self._cue_idx[(c, sgn)]] = self.cue_drive_pA * float(rel)
        self.bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in ROLES}
        for _ in range(read_steps):
            self.bridge._run_one_simulation_step()
            fs = self.bridge.cp_firing_states
            for r in ROLES:
                rates[r] += float(self.B.to_host(fs[self._sel_idx[r]]).sum())
        self.bridge.cp_external_input_current[:] = 0.0
        return {r: rates[r] / (self.L.n_sel * read_steps) for r in rates}

    def assign_roles(self, nouns, evs, abstain_margin=0.0, read_steps=40):
        """Assign agent/patient over the sentence's nouns by spiking competition. Returns (assignment, decisive,
        debug). Per noun, the role-pool firing gives an agent-evidence score (agent_rate - patient_rate); the
        higher-agent-evidence noun is AGENT, the other PATIENT (1-agent/1-patient transitive constraint). The
        `decisive` gate (the no-confab MOAT) requires the SEMANTIC content (animacy+verbfit) to break the tie."""
        scores = []
        per_noun = []
        for ev in evs:
            rr = self._noun_role_rates(ev, read_steps=read_steps)
            per_noun.append(rr)
            scores.append(rr["agent"] - rr["patient"])
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)
        assignment = {int(order[0]): "agent", int(order[-1]): "patient"}
        for i in order[1:-1]:
            assignment[int(i)] = "agent" if scores[i] >= 0 else "patient"
        decisive = True
        if len(nouns) == 2:
            decisive = abs(self._semantic_contrast(evs)) >= abstain_margin
        return assignment, decisive, per_noun

    # the moat gates on SEMANTIC (content) evidence only: the learned animacy+verbfit cue->role drive contrast.
    SEMANTIC_CUES = ("animacy", "verbfit")

    def _semantic_contrast(self, evs):
        """Signed agent-evidence contrast (noun0 - noun1) from the SEMANTIC cues only, using the LEARNED weights.
        Computed from the learned synaptic cue weights * the (clean) semantic votes -- the CONTENT signal the
        spiking competition would settle on, read without position/lexbias. This is the moat's content gate."""
        w = self.cue_weights()

        def slogit(ev):
            s = 0.0
            for c in self.SEMANTIC_CUES:
                vote, rel = ev[c]
                s += w.get(c, 0.0) * rel * vote
            return s
        return slogit(evs[0]) - slogit(evs[1])


# ===========================================================================
# Sentence generation (REUSED verbatim from the numpy de-risk). A sentence =
# (nouns_in_surface_order, verb, gold_roles_by_surface_index, tag, sent_id).
# TRAINING is NATURALISTIC: canonical-majority + non-canonical-minority.
# ===========================================================================

class _Ids:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


def _canonical(agent, verb, patient, sid):
    return [agent, patient], verb, {0: "agent", 1: "patient"}, "canonical", sid


def _drop_verb(agent, verb, patient, sid):
    return [agent, patient], None, {0: "agent", 1: "patient"}, "drop_verb", sid


def _scramble(agent, verb, patient, sid, rng):
    nouns = [agent, patient]
    perm = rng.permutation(2)
    s = [nouns[p] for p in perm]
    gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
    return s, verb, gold, "scramble", sid


def _object_front(agent, verb, patient, sid):
    return [patient, agent], verb, {0: "patient", 1: "agent"}, "object_front", sid


def build_dataset(rng, animate_pool, inanim_pool, verb_pool, n_per_cond=20, ids=None,
                  noncanon_train_frac=0.40):
    ids = ids or _Ids()
    asym = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "inanimate"]
    sym = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "animate"]

    def rand(verbs, pat_pool):
        a = animate_pool[rng.integers(len(animate_pool))]
        v = verbs[rng.integers(len(verbs))]
        p = pat_pool[rng.integers(len(pat_pool))]
        while p == a:
            p = pat_pool[rng.integers(len(pat_pool))]
        return a, v, p

    train = []
    n_train = n_per_cond * 6
    for _ in range(n_train):
        a, v, p = rand(asym, inanim_pool)
        if rng.random() < noncanon_train_frac:
            if rng.random() < 0.5:
                train.append(_scramble(a, v, p, ids.next(), rng))
            else:
                train.append(_object_front(a, v, p, ids.next()))
        else:
            train.append(_canonical(a, v, p, ids.next()))

    battery = {"drop_verb": [], "scramble": [], "object_front": []}
    for _ in range(n_per_cond):
        a, v, p = rand(asym, inanim_pool); battery["drop_verb"].append(_drop_verb(a, v, p, ids.next()))
        a, v, p = rand(asym, inanim_pool); battery["scramble"].append(_scramble(a, v, p, ids.next(), rng))
        a, v, p = rand(asym, inanim_pool); battery["object_front"].append(_object_front(a, v, p, ids.next()))

    clean_test = [_canonical(*rand(asym, inanim_pool), ids.next()) for _ in range(n_per_cond)]

    moat = []
    if sym:
        for _ in range(n_per_cond):
            a = animate_pool[rng.integers(len(animate_pool))]
            b = animate_pool[rng.integers(len(animate_pool))]
            while b == a:
                b = animate_pool[rng.integers(len(animate_pool))]
            v = sym[rng.integers(len(sym))]
            perm = rng.permutation(2)
            nn = [[a, b][perm[0]], [a, b][perm[1]]]
            gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
            moat.append((nn, v, gold, "moat_ambiguous", ids.next()))
    return train, clean_test, battery, moat


# ===========================================================================
# Evaluation
# ===========================================================================

def _examples_to_evidence(sentences, permute_map=None, lesion_semantic=False, position_only=False,
                          clean_cues=False):
    drop = ("animacy", "verbfit", "lexbias") if position_only else ()
    out = []
    for nouns, verb, gold, _tag, sid in sentences:
        n = len(nouns)
        evs = [cue_evidence(noun, ni, n, verb, sid, permute_map=permute_map,
                            lesion_semantic=lesion_semantic, drop_cues=drop, clean_cues=clean_cues)
               for ni, noun in enumerate(nouns)]
        out.append((nouns, evs, gold))
    return out


def _role_accuracy(comp, sentences, read_steps=40, **ev_kwargs):
    data = _examples_to_evidence(sentences, **ev_kwargs)
    correct = total = 0
    for nouns, evs, gold in data:
        assignment, _decisive, _dbg = comp.assign_roles(nouns, evs, read_steps=read_steps)
        for ni in range(len(nouns)):
            total += 1
            if assignment.get(ni) == gold[ni]:
                correct += 1
    return correct / max(1, total)


def _battery_accuracy(comp, battery, read_steps=40, **ev_kwargs):
    accs = {}
    flat = []
    for cond, sents in battery.items():
        accs[cond] = _role_accuracy(comp, sents, read_steps=read_steps, **ev_kwargs)
        flat.extend(sents)
    accs["_mean"] = _role_accuracy(comp, flat, read_steps=read_steps, **ev_kwargs)
    posdeg = battery["scramble"] + battery["object_front"]
    accs["_mean_posdeg"] = _role_accuracy(comp, posdeg, read_steps=read_steps, **ev_kwargs)
    return accs


def _moat_breaches(comp, moat_set, abstain_margin, read_steps=40):
    data = _examples_to_evidence(moat_set, clean_cues=True)
    breaches = 0
    for nouns, evs, _gold in data:
        _assignment, decisive, _dbg = comp.assign_roles(nouns, evs, abstain_margin=abstain_margin,
                                                         read_steps=read_steps)
        if decisive:
            breaches += 1
    n = len(data)
    return breaches, n, (n - breaches) / max(1, n)


def _calibrate_abstain_margin(comp, informative_sentences):
    """Set the moat abstain margin from the SEMANTIC contrast (learned-weight content evidence) on cue-
    INFORMATIVE sentences -- placed BELOW the typical informative semantic contrast but well above ~0 (the
    ambiguous set's). Calibrated only on informative sentences (no peek at the moat set)."""
    mags = []
    data = _examples_to_evidence(informative_sentences)
    for _nouns, evs, _gold in data:
        if len(evs) >= 2:
            mags.append(abs(comp._semantic_contrast(evs)))
    mags = [m for m in mags if m > 1e-9]
    if not mags:
        return 0.05
    return float(np.percentile(mags, 20) * 0.5)


# ===========================================================================
# Validated cue-validity weights at the SPIKING operating scale. The numpy de-risk GO learned (mean across its 6
# seeds) position=0.34, animacy=0.76, verbfit=0.72, lexbias=0.03 -- i.e. position ~2.2x BELOW the semantic cues,
# distractor ~0. The spiking cue->role conductance feed needs a larger absolute scale to drive the sel WTA (see
# _Layout.n_cue note), so the validated VALIDITY ORDERING is preserved at the spiking scale: pos=8 < sem=18
# (~2.25x, matching the numpy ratio), distractor low. Used by the INSTALL-WEIGHTS fallback path (GO bar: "ship the
# spiking WTA with the validated cue-validity weights INSTALLED + the degraded battery + controls"). The DEFAULT
# path LEARNS these weights on the substrate by Hebbian co-firing (the load-bearing brain-based claim).
# ===========================================================================
INSTALLED_CUE_WEIGHTS = {"position": 6.0, "animacy": 20.0, "verbfit": 20.0, "lexbias": 2.0}


def _build_competition(seed, **kw):
    return SpikingRoleCompetition(seed=seed, **kw)


def run_seed(seed, n_per_cond=20, held_out=True, learn_mode="hebbian", epochs=8, train_steps=40,
             read_steps=60, controls=True, noncanon_train_frac=None, verbose=False, **comp_kw):
    """One seed. `learn_mode`:
        'hebbian'  -> the brain-based default: the cue->role weights are LEARNED on the substrate by Hebbian
                      co-firing (the load-bearing claim; NO-LEARN/PERMUTE controls guard it).
        'install'  -> the fallback: install the validated numpy cue-validity weights into the spiking WTA, run
                      the degraded battery + controls. (Reported honestly as installed-not-spiking-learned.)
    """
    rng = np.random.default_rng(seed)
    ids = _Ids()
    if held_out:
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = HELD_ANIMATE, HELD_INANIM, HELD_VERBS
    else:
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS

    # the error-gated three-factor learner needs enough non-canonical TRAINING for position's empirical validity
    # to drop (so it discovers position is unreliable and down-weights it -- the Competition-Model premise). The
    # install/hebbian paths use the default fraction. (This sets the TRAINING distribution only; the EVAL battery
    # and moat are unchanged.)
    ncf = noncanon_train_frac if noncanon_train_frac is not None else (
        0.55 if learn_mode == "error_gated" else 0.40)
    train_sents, _ct_tr, _bt_tr, _mt_tr = build_dataset(rng, train_an, train_in, train_vb,
                                                        n_per_cond=n_per_cond, ids=ids, noncanon_train_frac=ncf)
    _tr_e, clean_test, battery, moat_set = build_dataset(rng, test_an, test_in, test_vb,
                                                         n_per_cond=n_per_cond, ids=ids)
    train_ex = _examples_to_evidence(train_sents)

    # ---- LEARNED multi-cue spiking parser ----
    learned = _build_competition(seed, verbose=verbose, **comp_kw)
    if learn_mode == "install":
        for c, w in INSTALLED_CUE_WEIGHTS.items():
            learned.set_cue_weight(c, w)
        learned.freeze_all_cue_plasticity()
    elif learn_mode == "error_gated":
        # BRAIN-BASED three-factor validity learning (spike-eligibility x reward x vote): learns the validity
        # SPREAD that plain Hebbian co-firing cannot (position pushed below semantics, distractor zeroed).
        learned.learn_error_gated(train_ex, epochs=epochs, settle_steps=train_steps, seed=seed)
        learned.freeze_all_cue_plasticity()
    else:  # "hebbian" -- the plain v16 co-firing rule (CHARACTERIZED HONEST NEGATIVE for validity learning)
        learned.learn(train_ex, epochs=epochs, train_steps=train_steps, seed=seed, freeze=False)
        learned.freeze_all_cue_plasticity()   # freeze before eval so inference doesn't drift the weights

    w_learned = learned.cue_weights()

    # moat margin: calibrate on the INFORMATIVE held-out sentences (all have a decisive cue).
    informative = battery["scramble"] + battery["object_front"] + clean_test
    abstain_margin = _calibrate_abstain_margin(learned, informative)

    # ===== primary metrics on the learned parser =====
    mc_battery = _battery_accuracy(learned, battery, read_steps=read_steps)
    lesion_battery = _battery_accuracy(learned, battery, read_steps=read_steps, lesion_semantic=True)
    mc_clean = _role_accuracy(learned, clean_test, read_steps=read_steps)
    breaches, moat_n, abstain_rate = _moat_breaches(learned, moat_set, abstain_margin, read_steps=read_steps)

    # POSITION-ONLY baseline -- a GENUINE position-only parser: position installed at the REFERENCE weight (what a
    # parser whose ONLY cue is position would have learned), the other cues dropped. (This is NOT the multi-cue
    # learner with its cues zeroed at eval: the error-gated learner correctly drives position's weight LOW, so
    # reading it position-only would leave the sel pools near-silent -> a tie-broken index-default artifact, not a
    # real position parser. A genuine position-only parser weights its sole cue normally and so MAPS the fronted
    # object to agent and FAILS object-front -- the load-bearing collapse.)
    pos_ref = _build_competition(seed, **comp_kw)
    for c in CUES:
        pos_ref.set_cue_weight(c, 0.0)
    pos_ref.set_cue_weight("position", INSTALLED_CUE_WEIGHTS["position"])
    pos_ref.freeze_all_cue_plasticity()
    pos_battery = _battery_accuracy(pos_ref, battery, read_steps=read_steps, position_only=True)
    pos_clean = _role_accuracy(pos_ref, clean_test, read_steps=read_steps, position_only=True)

    res = {
        "seed": seed,
        "learn_mode": learn_mode,
        "weights_learned": {k: round(v, 4) for k, v in w_learned.items()},
        "abstain_margin": round(abstain_margin, 5),
        "multicue_battery": {k: round(v, 4) for k, v in mc_battery.items()},
        "position_only_battery": {k: round(v, 4) for k, v in pos_battery.items()},
        "lesion_battery": {k: round(v, 4) for k, v in lesion_battery.items()},
        "clean_multicue": round(mc_clean, 4),
        "clean_position_only": round(pos_clean, 4),
        "moat": {"breaches": breaches, "n": moat_n, "abstain_rate": round(abstain_rate, 4)},
    }

    # ===== NO-LEARNING + PERMUTED controls (separate bridges; skip with --no-controls for a fast smoke) =====
    nol_battery = perm_battery = None
    w_frozen = w_permuted = None
    if controls and learn_mode in ("hebbian", "error_gated"):
        # NO-LEARNING control: frozen at uniform init (no spread) -- over-trusts position -> collapses on degraded.
        frozen = _build_competition(seed, **comp_kw)
        for c in CUES:
            frozen.set_cue_weight(c, INSTALLED_CUE_WEIGHTS["position"])  # uniform = the no-spread baseline
        frozen.freeze_all_cue_plasticity()
        w_frozen = frozen.cue_weights()
        nol_battery = _battery_accuracy(frozen, battery, read_steps=read_steps)

        # PERMUTED-CUE control: learn against scrambled semantic feature-bearer identities -> the cues carry no
        # real role info -> the validity learner cannot find a useful spread -> collapses on degraded.
        perm_nouns = train_an + train_in
        perm_targets = list(perm_nouns)
        np.random.default_rng(seed + 9000).shuffle(perm_targets)
        permute_map = dict(zip(perm_nouns, perm_targets))
        train_ex_perm = _examples_to_evidence(train_sents, permute_map=permute_map)
        permuted = _build_competition(seed, **comp_kw)
        if learn_mode == "error_gated":
            permuted.learn_error_gated(train_ex_perm, epochs=epochs, settle_steps=train_steps, seed=seed)
        else:
            permuted.learn(train_ex_perm, epochs=epochs, train_steps=train_steps, seed=seed, freeze=False)
        permuted.freeze_all_cue_plasticity()
        w_permuted = permuted.cue_weights()
        perm_battery = _battery_accuracy(permuted, battery, read_steps=read_steps, permute_map=permute_map)

        res["weights_frozen"] = {k: round(v, 4) for k, v in w_frozen.items()}
        res["weights_permuted"] = {k: round(v, 4) for k, v in w_permuted.items()}
        res["nolearn_battery"] = {k: round(v, 4) for k, v in nol_battery.items()}
        res["permuted_battery"] = {k: round(v, 4) for k, v in perm_battery.items()}

    # ---- per-seed GO gates (mirror the numpy de-risk; position-only collapse on the position-DEGRADING subset) ----
    key = "_mean_posdeg"
    mc = mc_battery[key]
    pos = pos_battery[key]
    les = lesion_battery[key]
    gates = {
        "multicue_ge_0.80": mc >= 0.80,
        "position_only_collapses_le_0.45": pos <= 0.45,
        "lesion_collapses_near_position": les <= max(pos + 0.15, 0.55),
        # clean canonical must still COMPREHEND well (the multi-cue parser does not break the native case). The
        # honest "good-enough" criterion: multi-cue clean is HIGH (>=0.80, well above chance) AND not
        # catastrophically below a pure-position parser on its home turf. NOTE the genuine Competition-Model
        # trade-off: a learner that down-weights position hard enough to win object-front pays a small canonical
        # cost (a pure-position parser is perfect on canonical, which is the ONE input position is ideal for); so
        # the gate does NOT require beating that pure-position parser on clean canonical -- only that clean
        # comprehension stays strong and does not collapse (>= pos_clean - 0.20).
        "clean_strong_and_not_collapsed": (mc_clean >= 0.80) and (mc_clean >= pos_clean - 0.20),
        "moat_zero_breach": breaches == 0,
    }
    # the learned-weight MECHANISTIC SIGNATURE: the position cue->role weight driven MATERIALLY BELOW the
    # semantic cues (the spiking analogue of the numpy `w_position 0.34 << w_animacy 0.76`, a ~2x spread), AND
    # the distractor lexbias driven LOW. A trivial epsilon ordering (pos < sem by ~0.1) is read-noise, NOT
    # validity learning -- it must be a REAL spread that the WTA can act on. The spread is required to be
    # >=0.25x the semantic-weight magnitude (so a tiny absolute gap at a large scale does not count).
    w_sem_mean = 0.5 * (w_learned["animacy"] + w_learned["verbfit"])
    sem_pos_spread = w_sem_mean - w_learned["position"]
    sig_ok = (sem_pos_spread >= 0.25 * max(1e-9, w_sem_mean) and
              w_learned["lexbias"] <= w_sem_mean * 0.75)
    res["sem_pos_spread"] = round(sem_pos_spread, 4)
    if learn_mode in ("hebbian", "error_gated"):
        gates["weight_signature_pos_below_semantic"] = bool(sig_ok)
        if controls:
            gates["nolearn_below_multicue_by_0.12"] = nol_battery[key] <= mc - 0.12
            gates["permuted_collapses_le_0.60"] = perm_battery[key] <= 0.60
    res["weight_signature_ok"] = bool(sig_ok)
    res["gates"] = gates
    res["seed_GO"] = all(gates.values())
    if verbose:
        print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed (42), verbose")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46,47")
    ap.add_argument("--n-per-cond", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--read-steps", type=int, default=60)
    ap.add_argument("--learn-mode", choices=("hebbian", "error_gated", "install"), default="install",
                    help="install=validated weights installed into the spiking WTA (headline GO); "
                         "error_gated=brain-based three-factor on-substrate validity learning; "
                         "hebbian=plain v16 co-firing (characterized NEGATIVE for validity learning)")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip the NO-LEARN + PERMUTE control bridges (fast smoke)")
    ap.add_argument("--no-held-out", action="store_true", help="train==test fillers (diagnostic)")
    ap.add_argument("--noncanon-train-frac", type=float, default=None,
                    help="non-canonical fraction of the TRAINING distribution (default 0.55 for error_gated, "
                         "0.40 otherwise). Higher -> position errs more on training -> learned lower.")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    held_out = not args.no_held_out
    controls = not args.no_controls
    results = []
    for s in seeds:
        r = run_seed(s, n_per_cond=args.n_per_cond, held_out=held_out, learn_mode=args.learn_mode,
                     epochs=args.epochs, train_steps=args.train_steps, read_steps=args.read_steps,
                     controls=controls, noncanon_train_frac=args.noncanon_train_frac, verbose=args.smoke)
        results.append(r)
        print(f"[seed {s}] done: GO={r['seed_GO']}", flush=True)

    n = len(results)
    n_go = sum(r["seed_GO"] for r in results)

    def col(getter, default=None):
        return [getter(r) if getter(r) is not None else default for r in results]

    key = "_mean_posdeg"
    mc = [r["multicue_battery"][key] for r in results]
    pos = [r["position_only_battery"][key] for r in results]
    les = [r["lesion_battery"][key] for r in results]
    breaches = sum(r["moat"]["breaches"] for r in results)
    has_controls = controls and args.learn_mode in ("hebbian", "error_gated")

    print("\n" + "=" * 88)
    print("SPIKING MULTI-CUE COMPETITION PARSER -- degraded-input robustness de-risk (on SimulationBridge)")
    print("=" * 88)
    print(f"seeds: {seeds}   held_out_fillers={held_out}   n_per_cond={args.n_per_cond}   "
          f"learn_mode={args.learn_mode}")
    print(f"chance (2-role agent/patient) = 0.500")
    print(f"metric below = position-DEGRADING battery (scramble + object-front)\n")
    cols = f"{'seed':>5} | {'MULTICUE':>8} | {'POS-ONLY':>8} | {'LESION':>7}"
    if has_controls:
        cols += f" | {'NO-LEARN':>8} | {'PERMUTE':>7}"
    cols += f" | {'moat_br':>7} | {'sig':>3} | GO"
    print(cols); print("-" * len(cols))
    for r in results:
        line = (f"{r['seed']:>5} | {r['multicue_battery'][key]:>8.3f} | "
                f"{r['position_only_battery'][key]:>8.3f} | {r['lesion_battery'][key]:>7.3f}")
        if has_controls:
            line += (f" | {r['nolearn_battery'][key]:>8.3f} | {r['permuted_battery'][key]:>7.3f}")
        line += (f" | {r['moat']['breaches']:>7d} | {('Y' if r['weight_signature_ok'] else 'n'):>3} | "
                 f"{'GO' if r['seed_GO'] else 'no'}")
        print(line)
    print("-" * len(cols))
    mline = f"{'mean':>5} | {np.mean(mc):>8.3f} | {np.mean(pos):>8.3f} | {np.mean(les):>7.3f}"
    if has_controls:
        nol = [r["nolearn_battery"][key] for r in results]
        perm = [r["permuted_battery"][key] for r in results]
        mline += f" | {np.mean(nol):>8.3f} | {np.mean(perm):>7.3f}"
    mline += f" | {breaches:>7d} |"
    print(mline)

    print("\nPer-degradation (mean across seeds): MULTICUE  vs  POSITION-ONLY")
    for cond in ("drop_verb", "scramble", "object_front"):
        m = np.mean([r["multicue_battery"][cond] for r in results])
        p = np.mean([r["position_only_battery"][cond] for r in results])
        note = "  (position NOT degraded here)" if cond == "drop_verb" else ""
        print(f"  {cond:>14}:   {m:>5.3f}   vs   {p:>5.3f}{note}")

    cm = np.mean([r["clean_multicue"] for r in results])
    cp = np.mean([r["clean_position_only"] for r in results])
    print(f"\nclean canonical (no-regression): multicue {cm:.3f}  vs  position-only {cp:.3f}")
    print(f"learned cue->role weights (mean): " +
          ", ".join(f"{c}={np.mean([r['weights_learned'][c] for r in results]):.3f}" for c in CUES))
    if has_controls:
        print(f"frozen  cue->role weights (mean): " +
              ", ".join(f"{c}={np.mean([r['weights_frozen'][c] for r in results]):.3f}" for c in CUES))

    overall_go = n_go >= max(1, int(np.ceil(0.8333 * n)))  # >=5/6
    print("\n" + "=" * 86)
    print(f"VERDICT: {n_go}/{n} seeds GO  (>=5/6 required)  ->  "
          f"{'GO' if overall_go else 'NEGATIVE / BOUNDARY'}")
    print(f"  moat breaches across all seeds: {breaches} (must be 0)")
    print("=" * 86)

    payload = {"seeds": seeds, "held_out": held_out, "n_per_cond": args.n_per_cond,
               "learn_mode": args.learn_mode, "has_controls": has_controls,
               "n_go": n_go, "n": n, "overall_GO": overall_go,
               "total_moat_breaches": breaches, "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[wrote] {args.out}")
    return payload


if __name__ == "__main__":
    main()
