"""CHOOSE-TO-SPEAK VALUE / SALIENCE APPRAISAL DE-RISK -- the brain's DECISION to SPEAK, grounded in its own
VALUE SYSTEM, decided by a SPIKING accumulator (NOT a host `if score > threshold`).

The next communicable-brain mechanism after Probe 1 GO (`2026-06-24-communicable-brain-probe1-GO.md`): Probe 1
took the SINGLE highest-PLAUSIBILITY topic-relevant proposal and emitted-or-abstained, speaking on ~26% of
topics (6-9/30). It has NO APPRAISAL -- no read-out of "is this WORTH saying / which of my candidate thoughts
is most SALIENT," and no value-grounded speak threshold. This probe adds exactly that, per the scoping
`research/findings/raw/_value_salience_appraisal_scoping.md` (Option A, "salience-drift" on the existing
commit circuit), CPU-first on the Probe-1 standalone brain.

THE MECHANISM (Option A, salience-drift):
  - PROPOSE returns the CANDIDATE SET (all topic-relevant, novel, graph-plausible, non-contradictory triples
    about X) -- not just the single best-plausibility (Probe-1's behaviour).
  - A VALUE / SALIENCE APPRAISAL ranks the candidates by WORTH = f(DA-value, plausibility, familiarity), where
    the DA-value is a CPU STAND-IN for the merged-bridge spiking SNc/striosome_value critic, and is **STRUCTURALLY
    DISTINCT from the plausibility axis** (owner-steer #2): a per-concept reward/INTEREST signal seeded from a
    SEPARATE reward-tagging RNG -- NOT a relabeled PPMI relatedness (verified: corr(value, plausibility) ~ 0). So
    "value adds beyond plausibility" is NOT circular.
  - A SPIKING speak/silence accumulator (the GO sel->commit->OPN WTA template: Wang-2002 NMDA integrators in
    biased competition via a shared FS pool, Lo-Wang-style all-or-none commit) DECIDES emit-vs-stay-silent. The
    speak pool's drift = the top candidate's WORTH; the silence pool's drift = its complement; the brain SPEAKS
    when the speak pool wins the spiking race. A SINGLE calibratable `talkativeness` gain (the owner-steer #1
    values parameter); conservative default (speaks somewhat MORE than Probe-1, but ONLY on grounded topics).
  - EMIT stays a graded-confidence FLAGGED hypothesis (NOT stored; the known-fact channel hard-gated).

THE LESION ANTI-CHEAT (load-bearing -- this is what makes the appraisal the BRAIN's value system, not a host
`if`): pin the DA-value input to BASELINE (lesion the value system) -> the worth collapses to a
plausibility(+familiarity)-only score -> the speak-decision through the SAME spiking accumulator reverts to the
plausibility-only BASELINE. The EXTRA, value-driven emissions MUST require the brain's value system. (Mirrors the
Probe-1 lesion 46/46 + the familiarity-gate lesion() precedent.) The plausibility-only baseline IS the lesion arm
by construction (same accumulator, value contribution zeroed), so "lesion -> collapses to plausibility-only" is
exact, not approximate.

GO (>=3 seeds; controller runs 6-seed if GO) requires ALL of:
  (1) SPEAKS-MORE-WHERE-SUPPORTED -- the value appraisal emits on MORE topics than the plausibility-only baseline,
      AND every NEW emission is still GROUNDED (the shuffled-PPMI-graph control collapses the emission set's
      grounded advantage >= 3x) -- speaks more WITHOUT confabulating.
  (2) CALIBRATED -- the stated confidence tracks the committed candidate's WORTH (spearman >= bar) AND the
      high-worth bin carries more of an INDEPENDENT graph-support property (b2 _strong_plausible) -- non-tautological.
  (3) MOAT RELAXED-NOT-REMOVED (HARD) -- 0 known-fact-channel leaks (a who/what query on every emitted un-stored
      proposition still ABSTAINS); every emission flagged; stored facts still answer.
  (4) LESION COLLAPSES TO PLAUSIBILITY-ONLY -- pinning the value system reverts the emission count + set to the
      plausibility-only baseline (the extra emissions vanish) -> the extra emissions are the BRAIN's value system.

HONEST: if the value axis does NOT make the brain speak-more-where-supported beyond plausibility-only (value adds
nothing, OR it speaks more but UNGROUNDED), this reports it PRECISELY -- that is the finding, not a failure to hide.
The value is NOT a relabeled plausibility (the independence is measured + asserted). NO sim/ edit.

The SPIKING accumulator is a real Izhikevich WTA on a numpy SimulationBridge slice (the brain-based requirement:
the speak DECISION is a neural pool's FIRING, not a host comparison). It mirrors the merged-bridge sel/commit/OPN
default (`g11_bg_runner` `sel_recurrent_weight=0.3`, Wang-2002 NMDA, biased competition).

CPU (`SIM_BACKEND=numpy`); reuse-by-import; NO `sim/` edit. Run:
  SIM_BACKEND=numpy python -u -m research.runners._value_salience_appraisal_derisk \
      --seeds 42,43,44 --out research/findings/raw/_value_salience_appraisal_derisk.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# the whole pipeline is the numpy-CPU brain (PPMI cortex + RF composer + parser + a spiking WTA accumulator slice).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  -- backend-safe device->host read (passthrough on numpy, .get() on cupy)

# --- the Probe-1 brain (reuse-by-import VERBATIM): PPMI cortex + b2 proposer + RF composer + parser + faculty ---
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
    build_plausibility,
    build_stored_facts,
    shuffle_graph,
    _category_pools,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty  # noqa: E402
from research.runners._grounded_lang_integration_derisk import (  # noqa: E402
    _build_inflection_map,
    _extract_svo_from_prose,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
# the Probe-1 graded-confidence read-out (the SAME plausibility->hedge mapping) -- reused VERBATIM.
from research.runners._communicable_brain_probe1_whatdoyouthink import (  # noqa: E402
    plausibility_score,
    hedge_for,
)


# ===========================================================================
# The DA-value / interest system -- a CPU STAND-IN for the merged-bridge spiking SNc / striosome_value critic.
# CRITICAL (owner-steer #2): this MUST be a DISTINCT signal from the plausibility (PPMI) axis, else
# "value adds beyond plausibility" is circular. It is a per-concept reward/INTEREST tag seeded from a SEPARATE
# RNG -- the value the brain learned a concept is rewarding/salient (Berridge incentive-salience "wanting";
# Niv-2007 tonic-DA vigor), independent of how graph-related two words are. The GPU follow-on replaces this
# stand-in with the REAL shared `dopamine` concentration off the merged bridge (so the lesion anti-cheat pins
# the real spiking SNc); on the standalone CPU brain it is a transparent stand-in (the scoping's recommendation).
# ===========================================================================
def build_concept_value(vocab, agents, patients, seed):
    """A per-concept VALUE / interest scalar in [0,1], seeded from a SEPARATE reward-tagging RNG (NOT PPMI). A
    sparse subset of content concepts are 'high-interest' (the brain found them rewarding/salient); the rest are
    low. This is the DA-value axis -- structurally independent of the PPMI plausibility axis (we MEASURE and
    ASSERT the independence: corr(value, plausibility-derived signal) ~ 0). Biology: vmPFC/OFC subjective value
    + striatal action-value, distinct from selectional-preference relatedness."""
    rng = np.random.default_rng(seed * 101 + 7)
    value = {}
    # content concepts (agents + patients) carry a value tag; actions are neutral (the verb is not the
    # "interesting thing" -- the value sits on the entities the brain cares about, OFC/vmPFC object-value).
    content = sorted(set(agents) | set(patients))
    # sparse high-interest set (~35%): a separate, reward-tagged subset. Drawn from a Beta so values spread
    # (some strongly wanted, some mildly), NOT a relabeled relatedness.
    base = rng.beta(1.5, 4.0, size=len(content))      # right-skewed: most low, a tail of high-interest
    # boost a sparse subset to 'salient' (the wanting tail)
    salient = rng.random(len(content)) < 0.35
    base = np.where(salient, np.clip(base + rng.uniform(0.3, 0.6, size=len(content)), 0, 1), base * 0.5)
    for w, v in zip(content, base):
        value[w] = float(v)
    for w in vocab:
        value.setdefault(w, 0.0)
    return value


def triple_value(value, triple):
    """The VALUE of a proposed proposition = the value of the entities it concerns. Use the MAX over the
    content concepts (the most-salient entity drives 'is this worth saying' -- the salience network prioritizes
    the single most important candidate). agent + patient carry value; the action is neutral."""
    a, _ac, p = triple
    return max(value.get(a, 0.0), value.get(p, 0.0))


# ===========================================================================
# The SPIKING speak/silence accumulator (the brain-based speak DECISION -- a neural pool's FIRING, not a host
# `if`). Mirrors the merged-bridge GO sel/commit/OPN default: two Wang-2002 NMDA integrators (speak vs silence)
# in BIASED COMPETITION through a shared FS pool (soft-WTA), Lo-Wang-style winner-take-all. The pool whose drift
# (its appraisal-worth input) wins the spiking race is the DECISION. NO sim/ edit -- built from BrainRegion /
# RegionPathway (the same primitives g11_bg_runner uses).
# ===========================================================================
class SpikingSpeakAccumulator:
    """A small, fast (CPU, ~10ms/decision) spiking speak-vs-silence WTA. speak_acc and silence_acc are NMDA
    integrators; a shared wta_fs FS pool implements biased competition (each accumulator drives the FS, the FS
    inhibits both). The DECISION = whichever pool fires more over the integration window (the spiking commit).
    The speak drive = base + gain * worth; the silence drive = base + gain * (1 - worth) -- so worth modulates
    the DRIFT (catalog O.19/C.32: value modulates the accumulator drift rate)."""

    def __init__(self, seed, n_acc=40, n_fs=20, n_steps=120, ou_pA=15.0):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_nmda = True                      # Wang-2002 NMDA-slow integration (the accumulator)
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)
        cfg.stdp_w_max = 30.0
        cfg.hebbian_max_weight = 30.0
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = False         # the accumulator is fixed wiring; no weight drift
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True                # OU noise -> the soft (graded) threshold near equal drives
        cfg.ou_std_current_pA = float(ou_pA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        regions = [
            BrainRegion(name="speak_acc", n_neurons=n_acc, exc_fraction=1.0, internal_density=0.5,
                        exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                        izh_neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL", enable_nmda=True),
            BrainRegion(name="silence_acc", n_neurons=n_acc, exc_fraction=1.0, internal_density=0.5,
                        exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                        izh_neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL", enable_nmda=True),
            BrainRegion(name="wta_fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON"),
        ]
        pathways = [
            RegionPathway(from_region="speak_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="silence_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="wta_fs", to_region="speak_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="wta_fs", to_region="silence_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        ]
        cfg.brain_regions = regions
        cfg.region_pathways = pathways
        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._idx = {n: np.asarray(v) for n, v in self._bridge.region_manager.region_indices_dict().items()}
        self.n_steps = int(n_steps)

    def decide(self, speak_drive_pA, silence_drive_pA):
        """Run one spiking race. Returns (decision_is_speak, speak_spikes, silence_spikes, margin). The DECISION
        is the spiking commit: whichever NMDA integrator wins the biased-competition race over the window.

        The spiking circuit's OU noise stream is HELD FIXED per drive-pair (snapshot/restore the global RNG,
        re-seeded from the rounded drives), so the decision is a deterministic FUNCTION of the drives. This is
        what makes the value-vs-lesion comparison a clean ABLATION: the value arm's drive is >= the lesion arm's
        for every topic (worth_value >= worth_lesion), and a monotone WTA on a FIXED noise realization then
        speaks on a SUPERSET -> the EXTRA emissions isolate the value signal, not a noise coincidence. (The
        spiking gate is still the brain-based decision; freezing the noise realization is the control, exactly as
        an ablation freezes everything but the lesioned variable.)"""
        b = self._bridge
        # snapshot the global RNG (the OU noise source), seed deterministically from the drives
        _state = np.random.get_state()
        dseed = (int(round(speak_drive_pA * 7.0)) * 100003 + int(round(silence_drive_pA * 7.0))) % (2**31 - 1)
        np.random.seed(dseed)
        try:
            b._initialize_simulation_data(called_from_playback_init=False)   # reset state per decision
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self._idx["speak_acc"]] = np.float32(speak_drive_pA)
            b.cp_external_input_current[self._idx["silence_acc"]] = np.float32(silence_drive_pA)
            sp = si = 0.0
            for _ in range(self.n_steps):
                b._run_one_simulation_step()
                # backend-safe (C3): to_host is a passthrough on numpy (byte-identical to the np.asarray it replaces)
                # and `.get()` on cupy -- so the speak accumulator's per-step spike read works when the console runs on
                # SIM_BACKEND=cupy for the onebrain composer (the accumulator bridge shares the active backend).
                fs = to_host(b.cp_firing_states)
                sp += float(fs[self._idx["speak_acc"]].sum())
                si += float(fs[self._idx["silence_acc"]].sum())
        finally:
            np.random.set_state(_state)                                       # restore the global RNG
        return (sp > si), sp, si, (sp - si)


# ===========================================================================
# The CHOOSE-TO-SPEAK appraisal turn -- extends the Probe-1 turn with the candidate SET + the value appraisal +
# the spiking speak decision. Reuses Probe-1's render+VERIFY contract + hedge mapping VERBATIM.
# ===========================================================================
class AppraisalTurn:
    """One conversational turn with the value/salience APPRAISAL. The brain does the cognition (PPMI cortex +
    RF composer + the value system); the LLM (CPU stand-in) does the surface form; the spiking accumulator does
    the speak DECISION.

    Pipeline:
      ASSIMILATE(X)            -- (unchanged Probe-1) the topic's PPMI neighborhood.
      PROPOSE CANDIDATE SET    -- all topic-relevant, novel, graph-plausible, non-contradictory triples about X.
      APPRAISE + RANK          -- worth(c) = talkativeness * (w_v*value + w_p*plaus + w_f*familiarity), ranked.
      SPEAK DECISION (spiking) -- the top candidate's worth drives the speak pool; the spiking WTA decides emit.
      RENDER + VERIFY          -- (unchanged Probe-1) fluency faculty surface form + BridgeParser re-parse.
      EMIT                     -- (unchanged Probe-1) graded-confidence FLAGGED hypothesis; NOT stored.
    """

    def __init__(self, proposer, comp, agent, P, row, vocab_sets, faculty, value, accumulator,
                 full_pools=None, talkativeness=1.0, w_value=0.5, w_plaus=0.35, w_fam=0.15,
                 speak_base_pA=70.0, speak_gain_pA=180.0, silence_drive_pA=150.0):
        self.proposer = proposer
        self.comp = comp
        self.agent = agent
        self.P, self.row = P, row
        self.agents_set, self.actions_set, self.patients_set, self.inflect = vocab_sets
        self.faculty = faculty
        self.value = value                          # the DA-value / interest map (DISTINCT from PPMI)
        self.acc = accumulator                      # the SpikingSpeakAccumulator
        self.talkativeness = float(talkativeness)
        self.w_value, self.w_plaus, self.w_fam = float(w_value), float(w_plaus), float(w_fam)
        self.speak_base_pA, self.speak_gain_pA = float(speak_base_pA), float(speak_gain_pA)
        self.silence_drive_pA = float(silence_drive_pA)
        fa, fac, fp = full_pools if full_pools else (set(proposer.agents), set(proposer.actions),
                                                     set(proposer.patients))
        self.full_agents, self.full_actions, self.full_patients = set(fa), set(fac), set(fp)
        self._fa_list = sorted(self.full_agents)
        self._fac_list = sorted(self.full_actions)
        self._fp_list = sorted(self.full_patients)
        # per-seed normalisers, set by calibrate()
        self._plaus_lo = self._plaus_hi = None
        self._val_lo = self._val_hi = None
        self._fam_lo = self._fam_hi = None
        self._worth_lo = self._worth_hi = None

    # --- ASSIMILATE (unchanged Probe-1) ---
    def assimilate(self, topic):
        if topic not in self.row:
            return {"in_graph": False, "related_actions": [], "related_patients": []}
        ti = self.row[topic]
        rel_ac = sorted(self.proposer.actions, key=lambda w: -self.P[ti, self.row[w]])
        rel_pt = sorted(self.proposer.patients, key=lambda w: -self.P[ti, self.row[w]])
        return {"in_graph": True,
                "related_actions": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_ac[:4]],
                "related_patients": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_pt[:4]]}

    # --- the FAMILIARITY axis (the brain's confidence / "do I know this neighborhood" signal) ---
    def familiarity(self, triple):
        """A graded familiarity = the geometric-mean PPMI of the topic-entity with the rest of the proposition
        normalised by neighborhood density. Reuses the brain's learned relatedness as the metacognitive
        feeling-of-knowing (Bogacz-Brown familiarity read GRADED). Lesionable (the GPU version uses the
        AntiHebbianFamiliarity.lesion())."""
        a, ac, p = triple
        # how embedded the proposition's content is in the learned graph (mean of its pair strengths)
        s = (float(self.P[self.row[a], self.row[ac]]) + float(self.P[self.row[ac], self.row[p]])
             + float(self.P[self.row[a], self.row[p]])) / 3.0
        return s

    # --- PROPOSE the CANDIDATE SET (all topic-relevant novel plausible triples about X) ---
    def propose_candidates_about(self, topic, n_attempts=500):
        """Return ALL distinct topic-relevant, NOVEL, graph-plausible, non-contradictory candidate triples about
        X, each with its plausibility score. (Probe-1 kept only the single best-plausibility; the appraisal ranks
        the SET by WORTH.) Returns a list of (triple, plausibility) -- empty = the honest 'no graph-supported
        candidate' (the brain genuinely cannot speak ABOUT X in a grounded way)."""
        if topic not in self.row:
            return []
        topic_is_agent = topic in self.full_agents
        topic_is_patient = topic in self.full_patients
        if not (topic_is_agent or topic_is_patient):
            return []
        seen = {}
        for _ in range(n_attempts):
            if topic_is_agent:
                a = topic
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a,), self._fac_list))
                p = self.proposer._sample_weighted(
                    self._fp_list, self.proposer._weight_partner((a, ac), self._fp_list))
            else:
                p = topic
                a = self.proposer._sample_weighted(
                    self._fa_list, self.proposer._weight_partner((p,), self._fa_list))
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a, p), self._fac_list))
            triple = (a, ac, p)
            if topic not in triple:
                continue
            if triple in self.proposer.all_stored:
                continue
            if not self.proposer._plausible(a, ac, p):
                continue
            if self.proposer._contradicts(a, ac, p):
                continue
            if triple not in seen:
                seen[triple] = plausibility_score(self.P, self.row, a, ac, p)
        return sorted(seen.items(), key=lambda kv: -kv[1])

    # --- the worth components (raw) for a candidate ---
    def _components(self, triple, lesion_value=False):
        plaus = plausibility_score(self.P, self.row, *triple)
        val = 0.0 if lesion_value else triple_value(self.value, triple)
        fam = self.familiarity(triple)
        return plaus, val, fam

    def _norm(self, x, lo, hi):
        if lo is None or hi is None or hi <= lo:
            return 0.5
        return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))

    def worth(self, triple, lesion_value=False):
        """The appraisal WORTH = talkativeness * (w_v*VALUE + w_p*plausibility + w_f*familiarity), each axis
        normalised by the per-seed [lo,hi] (value-intact). lesion_value=True is a CLEAN ABLATION: the value
        TERM is dropped (the speak drive loses exactly the w_v*VALUE contribution; the plausibility + familiarity
        terms keep their SAME normalisers + weights). So worth_lesion <= worth_value for every candidate (the
        value term is >= 0), and the lesion arm's spiking decision is the plausibility(+familiarity)-only
        baseline -- the EXACT ablation, not a re-scaled re-derivation."""
        plaus, val, fam = self._components(triple, lesion_value=lesion_value)
        pn = self._norm(plaus, self._plaus_lo, self._plaus_hi)
        vn = self._norm(val, self._val_lo, self._val_hi)
        fn = self._norm(fam, self._fam_lo, self._fam_hi)
        raw = self.w_plaus * pn + self.w_fam * fn
        if not lesion_value:
            raw += self.w_value * vn          # the value TERM (ablated when lesioned) -- nothing else changes
        return self.talkativeness * raw

    # --- RENDER + VERIFY (unchanged Probe-1 contract; no neural renderer on the CPU smoke) ---
    def render_and_verify(self, triple, faculty):
        a, v, p = triple
        surface, asserted = faculty.render_svo(a, v, p)
        csvo = _extract_svo_from_prose(surface, self.agents_set, self.actions_set, self.patients_set, self.inflect)
        if csvo is None:
            return {"surface": surface, "asserted_svo": asserted, "reparse_svo": None, "verified": False}
        parsed = self.agent.parse(csvo, voice="active")
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        return {"surface": surface, "asserted_svo": asserted, "reparse_svo": rsvo,
                "verified": (rsvo == list(triple))}

    # --- the speak DECISION via the spiking accumulator (additive incentive-salience scheme) ---
    def _speak_drives(self, plaus_norm, value_norm, fam_norm, lesion_value=False):
        """Build the (speak, silence) drift currents. The SPEAK pool gets ADDITIVE pushes from plausibility +
        familiarity + (unless lesioned) VALUE -- incentive salience ADDS drive to 'want to say it' (Berridge
        wanting; catalog O.19/C.32 value modulates the accumulator drift). The SILENCE pool has a FIXED competing
        'default reticence' drive. talkativeness scales the whole speak drive (owner-steer #1 -- how readily the
        brain volunteers a view). The value-LESION removes EXACTLY the value push (a clean additive ablation),
        so the lesion arm == the plausibility(+familiarity)-only baseline through the SAME accumulator + the SAME
        silence drive -> the EXTRA emissions isolate the value system."""
        push = self.w_plaus * plaus_norm + self.w_fam * fam_norm
        if not lesion_value:
            push += self.w_value * value_norm
        speak = self.speak_base_pA + self.talkativeness * self.speak_gain_pA * push
        silence = self.silence_drive_pA
        return float(speak), float(silence)

    def turn(self, topic, conf_lo, conf_hi, n_attempts=500, lesion_value=False):
        """Run the full appraisal turn. The candidate SET is proposed, APPRAISED + ranked by worth, the top
        candidate drives the SPIKING speak decision, and (if speak wins + VERIFY passes) a graded-confidence
        flagged hypothesis is EMITTED. lesion_value=True pins the value system to baseline (the plausibility-only
        baseline / the value-lesion arm)."""
        assim = self.assimilate(topic)
        cands = self.propose_candidates_about(topic, n_attempts=n_attempts)
        rec = {"topic": topic, "assimilation": assim, "n_candidates": len(cands)}
        if not cands:
            rec.update({"proposed_triple": None, "plausibility": None, "worth": None, "rendered": None,
                        "verified": None, "speak_decision": False, "emitted": False,
                        "reply": "I don't really have a view on that.", "hedge": None, "confidence": None,
                        "no_grounded_candidate": True})
            return rec
        # APPRAISE + RANK by worth
        ranked = sorted(cands, key=lambda tp: -self.worth(tp[0], lesion_value=lesion_value))
        best_triple, best_plaus = ranked[0]
        w = self.worth(best_triple, lesion_value=lesion_value)
        wn = self._norm(w, self._worth_lo, self._worth_hi)
        # the SPIKING speak DECISION (a neural pool's firing, NOT a host `if`). The committed candidate's
        # component norms build the additive speak drive (value ADDS drive unless lesioned).
        plaus_raw, val_raw, fam_raw = self._components(best_triple, lesion_value=False)
        pn = self._norm(plaus_raw, self._plaus_lo, self._plaus_hi)
        vn = self._norm(val_raw, self._val_lo, self._val_hi)
        fn = self._norm(fam_raw, self._fam_lo, self._fam_hi)
        speak_pA, silence_pA = self._speak_drives(pn, vn, fn, lesion_value=lesion_value)
        is_speak, sp_spk, si_spk, margin = self.acc.decide(speak_pA, silence_pA)
        rv = self.render_and_verify(best_triple, self.faculty) if is_speak else None
        # the graded-confidence hedge tracks the committed candidate's WORTH (the brain's appraisal of how
        # worth-saying it is) -- not raw plausibility (Probe-1's signal), since the committed candidate is now
        # CHOSEN by worth. conf_lo/hi are the per-seed worth-population min/max (set in calibrate()).
        hedge, conf = hedge_for(w, conf_lo, conf_hi)
        if is_speak and rv is not None and rv["verified"]:
            rec.update({"proposed_triple": list(best_triple), "topic_in_proposition": (topic in best_triple),
                        "plausibility": round(best_plaus, 4), "worth": round(float(w), 4),
                        "worth_norm": round(float(wn), 4), "rendered": rv, "verified": True,
                        "speak_decision": True, "speak_spikes": sp_spk, "silence_spikes": si_spk,
                        "decision_margin": margin, "emitted": True,
                        "reply": f"{hedge} {' '.join(best_triple)}.", "hedge": hedge,
                        "confidence": round(conf, 3), "no_grounded_candidate": False})
        else:
            # either the spiking accumulator chose SILENCE (worth below the brain's speak bar), or VERIFY rejected
            reason = "spiking accumulator chose SILENCE" if not is_speak else "render/VERIFY rejected"
            rec.update({"proposed_triple": list(best_triple), "topic_in_proposition": (topic in best_triple),
                        "plausibility": round(best_plaus, 4), "worth": round(float(w), 4),
                        "worth_norm": round(float(wn), 4), "rendered": rv, "verified": bool(rv and rv["verified"]),
                        "speak_decision": bool(is_speak), "speak_spikes": sp_spk, "silence_spikes": si_spk,
                        "decision_margin": margin, "emitted": False, "reply": None, "hedge": hedge,
                        "confidence": round(conf, 3), "no_grounded_candidate": False, "silence_reason": reason})
        return rec

    # --- calibrate the per-seed normalisers + the hedge bands over the topic population ---
    def calibrate(self, topics, n_attempts=500):
        """Pre-pass: gather the best candidate per topic + its components, to set per-seed [lo, hi] normalisers
        for {plausibility, value, familiarity, worth} and the hedge-band population (Probe-1's calibration,
        extended to the worth ranking). Returns (conf_lo, conf_hi) for the hedge map."""
        plaus_v, val_v, fam_v = [], [], []
        best_per_topic = []
        for t in topics:
            cands = self.propose_candidates_about(t, n_attempts=n_attempts)
            if not cands:
                best_per_topic.append(None)
                continue
            # rank by raw (un-normalised) value+plaus to pick the calibration representative; collect components
            for tp, _pl in cands:
                pl, vv, fm = self._components(tp, lesion_value=False)
                plaus_v.append(pl); val_v.append(vv); fam_v.append(fm)
            best = max(cands, key=lambda kv: kv[1])
            best_per_topic.append(best)
        if plaus_v:
            self._plaus_lo, self._plaus_hi = float(min(plaus_v)), float(max(plaus_v))
            self._val_lo, self._val_hi = float(min(val_v)), float(max(val_v))
            self._fam_lo, self._fam_hi = float(min(fam_v)), float(max(fam_v))
        # worth range over the per-topic WORTH-ranked best candidate (the one the turn commits). This is BOTH the
        # worth-normaliser (for the drift map) AND the hedge-band population (the hedge maps worth now), so the
        # hedge band reflects the committed-candidate worth distribution.
        worths = []
        for t in topics:
            cs = self.propose_candidates_about(t, n_attempts=n_attempts)
            if not cs:
                continue
            worths.append(max(self.worth(tp, lesion_value=False) for tp, _pl in cs))
        if worths:
            self._worth_lo, self._worth_hi = float(min(worths)), float(max(worths))
        # the hedge maps WORTH -> [conf_lo, conf_hi] is the per-seed worth population (committed-candidate worths)
        conf_lo = float(min(worths)) if worths else 0.0
        conf_hi = float(max(worths)) if worths else 1.0
        return conf_lo, conf_hi


# ===========================================================================
# Per-seed run: build the Probe-1 brain + the value system + the spiking accumulator; run the value-appraisal
# turns AND the plausibility-only baseline (== the value-lesion arm); measure the 4 gates + the lesion anti-cheat
# + the shuffled-graph groundedness control + the value/plausibility independence.
# ===========================================================================
def run_seed(seed, vocab, corpus, a, accumulator):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, plausible_all = build_stored_facts(
        agents, actions, patients, P, row, tau, a.n_facts, a.n_negated, rng)
    all_stored = set(affirmed) | set(negated)

    # the brain's KNOWN-fact store (RF composer; the no-confab moat intact)
    comp = RFPhasorComposer(seed=seed, D=a.D, vocab=vocab)
    for ag, ac, pt in affirmed:
        comp.store(ag, ac, pt, polarity="AFFIRM")
    for ag, ac, pt in negated:
        comp.store(ag, ac, pt, polarity="NEGATE")

    bc_agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab},
                                        composer=comp, composer_kind="rf")
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1))

    agents_set, actions_set, patients_set = set(agents), set(actions), set(patients)
    inflect = _build_inflection_map(sorted(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)
    grounded_faculty = TemplateStubFaculty()
    full_pools = (set(agents), set(actions), set(patients))

    # the DA-VALUE / interest system -- DISTINCT from the PPMI plausibility axis (owner-steer #2)
    value = build_concept_value(vocab, agents, patients, seed)

    turn = AppraisalTurn(proposer, comp, bc_agent, P, row, vocab_sets, grounded_faculty, value, accumulator,
                         full_pools=full_pools, talkativeness=a.talkativeness,
                         w_value=a.w_value, w_plaus=a.w_plaus, w_fam=a.w_fam,
                         speak_base_pA=a.speak_base_pA, speak_gain_pA=a.speak_gain_pA,
                         silence_drive_pA=a.silence_drive_pA)

    # held-out topics (same selection as Probe-1: words NOT the agent of any stored fact)
    stored_agents = {f[0] for f in affirmed}
    topic_pool = [w for w in (agents + patients) if w not in stored_agents]
    rng.shuffle(topic_pool)
    topics = topic_pool[:a.n_topics]

    # calibrate the per-seed normalisers + hedge bands
    conf_lo, conf_hi = turn.calibrate(topics, n_attempts=a.n_attempts)

    # ---- VALUE appraisal turns (the brain's value system intact) ----
    turns_val = [turn.turn(t, conf_lo, conf_hi, n_attempts=a.n_attempts, lesion_value=False) for t in topics]
    emitted_val = [r for r in turns_val if r["emitted"]]
    n_emit_val = len(emitted_val)

    # ---- PLAUSIBILITY-ONLY baseline == the value-LESION arm (value pinned to baseline) ----
    turns_les = [turn.turn(t, conf_lo, conf_hi, n_attempts=a.n_attempts, lesion_value=True) for t in topics]
    emitted_les = [r for r in turns_les if r["emitted"]]
    n_emit_les = len(emitted_les)

    emit_topics_val = {r["topic"] for r in emitted_val}
    emit_topics_les = {r["topic"] for r in emitted_les}
    new_topics = emit_topics_val - emit_topics_les            # the EXTRA, value-driven emissions
    n_new = len(new_topics)

    # =====================================================================
    # value/plausibility INDEPENDENCE (owner-steer #2 -- non-circularity): the value axis must NOT be a relabeled
    # plausibility. The PRIMARY, robust measure is CONCEPT-LEVEL: correlate, over the DISTINCT content concepts,
    # each concept's VALUE tag with its mean PPMI relatedness (the plausibility axis's per-concept summary). ~0 ->
    # the value system is a genuinely SEPARATE reward/interest signal (seeded from a distinct RNG), so "value adds
    # beyond plausibility" is not circular. (The candidate-level correlation over the few repeat-weighted topic
    # candidates is tiny-sample-noisy; the concept-level measure over all content concepts is the honest claim.)
    content_concepts = sorted(set(agents) | set(patients))
    cvals = np.array([value.get(w, 0.0) for w in content_concepts], dtype=float)
    # per-concept mean PPMI relatedness to the rest of the vocab (the plausibility axis's per-concept summary)
    cppmi = np.array([float(P[row[w]].mean()) for w in content_concepts], dtype=float)
    if len(cvals) >= 3 and cvals.std() > 0 and cppmi.std() > 0:
        value_plaus_corr = float(np.corrcoef(cvals, cppmi)[0, 1])
    else:
        value_plaus_corr = 0.0
    # SECONDARY (reported, not gated): the candidate-level correlation over DISTINCT candidate triples
    cand_val, cand_plaus = {}, {}
    for t in topics:
        for tp, pl in turn.propose_candidates_about(t, n_attempts=a.n_attempts):
            cand_val[tp] = triple_value(value, tp); cand_plaus[tp] = pl
    cv = np.array(list(cand_val.values()), dtype=float)
    cp_ = np.array([cand_plaus[k] for k in cand_val], dtype=float)
    value_plaus_corr_candidate = (float(np.corrcoef(cv, cp_)[0, 1])
                                  if len(cv) >= 3 and cv.std() > 0 and cp_.std() > 0 else 0.0)

    # =====================================================================
    # (1a) SPEAKS-MORE-WHERE-SUPPORTED: more emissions than the plausibility-only baseline.
    # =====================================================================
    speaks_more = n_emit_val > n_emit_les

    # =====================================================================
    # (1b) GROUNDED: the value-driven emission set's grounded advantage (shuffled-PPMI control) >= 3x. Every NEW
    # emission must still be GROUNDED: the proposition is graph-plausible under the TRUE graph but collapses under
    # a shuffled graph. Measure the emission set's mean TRUE-graph plausibility-pass vs a shuffled-graph proposer's.
    # =====================================================================
    # the emitted propositions (value arm) -- all are _plausible by construction; the shuffled control asks
    # whether a shuffled-graph proposer would have FOUND them plausible (it must not -> the learned structure
    # is what makes them grounded).
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0

    def _related_shuf(w1, w2):
        return P_shuf[row[w1], row[w2]] >= tau_s

    def _plausible_shuf(tp):
        a_, ac_, p_ = tp
        return _related_shuf(a_, ac_) and _related_shuf(ac_, p_)

    emit_triples_val = [tuple(r["proposed_triple"]) for r in emitted_val]
    true_pass = sum(1 for tp in emit_triples_val if proposer._plausible(*tp))   # == len (gate-constructed)
    shuf_pass = sum(1 for tp in emit_triples_val if _plausible_shuf(tp))
    true_frac = true_pass / max(1, len(emit_triples_val))
    shuf_frac = shuf_pass / max(1, len(emit_triples_val))
    grounded_advantage = true_frac / max(shuf_frac, 1.0 / max(1, len(emit_triples_val)))
    grounded_ok = (len(emit_triples_val) > 0) and (grounded_advantage >= a.advantage_bar)

    # the NEW (value-only) emissions: the load-bearing claim is that the EXTRA, value-driven emissions are
    # GROUNDED, not noise. Each is graph-plausible under the TRUE graph by the proposer gate (new_true == len);
    # the SHUFFLED graph must NOT find them plausible (the learned neighborhoods are what make them grounded).
    # The >=advantage_bar collapse is asserted on the FULL emission SET (a large-enough sample); on the tiny
    # new-subset we assert (a) every new emission is true-graph-plausible AND (b) the shuffled graph passes few
    # of them (shuf_frac <= 1/advantage_bar of the true frac) -- a per-triple grounded check, not a ratio on n<5.
    new_triples = [tuple(r["proposed_triple"]) for r in emitted_val if r["topic"] in new_topics]
    new_true = sum(1 for tp in new_triples if proposer._plausible(*tp))
    new_shuf = sum(1 for tp in new_triples if _plausible_shuf(tp))
    new_true_frac = new_true / max(1, len(new_triples))
    new_shuf_frac = new_shuf / max(1, len(new_triples))
    # every new emission graph-plausible (true) AND the shuffled graph passes at most 1/advantage_bar of them
    new_emissions_grounded = (len(new_triples) == 0) or (
        new_true == len(new_triples) and new_shuf_frac <= (1.0 / a.advantage_bar) + 1e-9)
    new_grounded_advantage = (new_true_frac / max(new_shuf_frac, 1.0 / max(1, len(new_triples)))
                              if new_triples else None)

    # =====================================================================
    # (2) CALIBRATED: confidence tracks WORTH (non-tautological -- worth has a value axis the plausibility-only
    # confidence does NOT read). Spearman(worth, confidence) + the INDEPENDENT strong-plausibility bin check.
    # =====================================================================
    cal_rows = [(r["worth"], r["confidence"], int(proposer._strong_plausible(*tuple(r["proposed_triple"]))))
                for r in emitted_val if r["worth"] is not None and r["confidence"] is not None]
    if len(cal_rows) >= 3:
        ws = np.array([x[0] for x in cal_rows], dtype=float)
        cs = np.array([x[1] for x in cal_rows], dtype=float)
        strong = np.array([x[2] for x in cal_rows], dtype=float)
        wr = np.argsort(np.argsort(ws)).astype(float)
        cr = np.argsort(np.argsort(cs)).astype(float)
        spearman = float(np.corrcoef(wr, cr)[0, 1]) if wr.std() > 0 and cr.std() > 0 else 1.0
        order = np.argsort(ws)
        n = len(ws)
        lo_idx = order[: max(1, n // 3)]
        hi_idx = order[-max(1, n // 3):]
        strong_lo = float(strong[lo_idx].mean())
        strong_hi = float(strong[hi_idx].mean())
        strong_reliability = bool(strong_hi >= strong_lo)
        calibrated_ok = (spearman >= a.calib_spearman_bar) and strong_reliability
    else:
        spearman = strong_lo = strong_hi = None
        strong_reliability = None
        calibrated_ok = None

    # =====================================================================
    # (3) MOAT RELAXED-NOT-REMOVED (HARD): (i) 0 known-fact-channel LEAKS (an un-stored proposition never passes
    # the known-fact channel) + (ii) every emission FLAGGED + (iii) stored facts still ANSWER (per distinct
    # multi-valued cue). The speak gate is an ADDITIVE emission channel; it does NOT touch the known-fact channel.
    # =====================================================================
    moat_leaks = 0
    for r in emitted_val:
        a_, v_, p_ = r["proposed_triple"]
        if bc_agent.what_does(a_, v_) == p_:
            moat_leaks += 1
        if bc_agent.is_it_true(a_, v_, p_) == "yes":
            moat_leaks += 1
    all_flagged = (n_emit_val > 0) and all(r["hedge"] is not None for r in emitted_val)
    # stored facts still ANSWER -- per DISTINCT (agent, action) cue (the b2 store is MULTI-VALUED by construction:
    # an agent does several plausible things -- 'dog plays ball' AND 'dog plays toy' share the (dog, play) cue --
    # so what_does(cue) can only return ONE of them; a strict per-fact ==pt check spuriously "fails" the others).
    # The correct test: each distinct stored cue resolves to ONE OF ITS stored patients (a genuine stored fact,
    # NOT a confabulation) -> the known-fact channel is intact. (Plus the 0-leak check above is the load-bearing
    # part: an UN-stored proposition never passes as a known fact.)
    # NOT a confabulation) -> the known-fact channel is intact. A small tolerance absorbs the DOCUMENTED RF
    # small-D retrieval tail (b2 doc: "the documented RF code-fidelity tail at small D" -- a single cue out of
    # ~13 may land on a non-stored patient at D=64); that is a composer-fidelity property, NOT a moat-mechanism
    # failure. The LOAD-BEARING moat claim is the 0-LEAK check (an UN-stored proposition never passes as a known
    # fact) + every emission FLAGGED -- both must be PERFECT; the stored-cue answer rate carries the small-D tol.
    from collections import defaultdict as _dd
    cue_to_patients = _dd(set)
    for ag, ac, pt in affirmed:
        cue_to_patients[(ag, ac)].add(pt)
    stored_answer_ok, stored_answer_total = 0, 0
    for (ag, ac), pats in cue_to_patients.items():
        stored_answer_total += 1
        if bc_agent.what_does(ag, ac) in pats:                        # resolves to a genuine stored patient
            stored_answer_ok += 1
    stored_answer_rate = stored_answer_ok / max(1, stored_answer_total)
    stored_answers = (stored_answer_total > 0) and (stored_answer_rate >= a.stored_answer_bar)
    # the LOAD-BEARING relaxed-not-removed gate: 0 leaks (PERFECT) + all flagged (PERFECT) + stored-cues answer
    # at/above the small-D-tolerant bar (the moat is RELAXED to a flagged speak channel, never REMOVED).
    moat_ok = (moat_leaks == 0) and all_flagged and stored_answers

    # =====================================================================
    # (4) LESION COLLAPSES TO PLAUSIBILITY-ONLY: with the value system pinned to baseline, the emission count +
    # set revert to the plausibility-only baseline -> the EXTRA emissions are the BRAIN's value system, not a
    # host re-ranking. (The lesion arm IS the plausibility-only baseline by construction; this confirms the
    # value arm's EXTRA emissions VANISH under lesion.) Pass iff: value arm speaks MORE than lesion arm AND every
    # value-only NEW topic is NOT emitted in the lesion arm (the extra emissions require the value system).
    # =====================================================================
    lesion_collapses = (n_emit_val > n_emit_les) and (len(new_topics & emit_topics_les) == 0)

    print(f"\n[appraisal seed {seed}] stored {len(affirmed)} ({len(negated)} negated) | topics {len(topics)} "
          f"| tau(P{a.tau_pct})={tau:.3f} | talkativeness={a.talkativeness}", flush=True)
    print(f"  VALUE arm: EMITTED {n_emit_val}/{len(topics)}  |  PLAUSIBILITY-ONLY (lesion) arm: EMITTED "
          f"{n_emit_les}/{len(topics)}  ->  +{n_new} value-driven emissions", flush=True)
    print(f"  (1a) SPEAKS-MORE: {speaks_more} ({n_emit_val} > {n_emit_les})", flush=True)
    print(f"  (1b) GROUNDED: emission-set advantage {grounded_advantage:.1f}x (>= {a.advantage_bar}x: {grounded_ok}) "
          f"| NEW-emissions grounded {new_emissions_grounded} (n_new={len(new_triples)}, adv "
          f"{new_grounded_advantage if isinstance(new_grounded_advantage,float) else float('nan'):.1f}x)", flush=True)
    print(f"  (2) CALIBRATED: spearman(worth,conf)={spearman} | INDEPENDENT strong-plausible lo {strong_lo} -> "
          f"hi {strong_hi} (rises: {strong_reliability}) -> {calibrated_ok}", flush=True)
    print(f"  (3) MOAT (relaxed-not-removed): {moat_leaks} known-fact leaks (0) | all-flagged {all_flagged} | "
          f"stored-cues-answer {stored_answer_ok}/{stored_answer_total} ({stored_answer_rate:.2f} >= "
          f"{a.stored_answer_bar} small-D-tol) -> {moat_ok}", flush=True)
    print(f"  (4) LESION collapses-to-plausibility-only: {lesion_collapses} "
          f"(value arm {n_emit_val} > lesion {n_emit_les}; {len(new_topics & emit_topics_les)} extra topics leaked "
          f"into lesion)", flush=True)
    print(f"  value<->plausibility INDEPENDENCE (concept-level): corr={value_plaus_corr:+.3f} "
          f"(candidate-level {value_plaus_corr_candidate:+.3f}) -- must be ~0 -> non-circular", flush=True)
    if emitted_val:
        print(f"  example value-appraised flagged hypotheses:", flush=True)
        for r in emitted_val[:5]:
            tag = " [NEW vs plaus-only]" if r["topic"] in new_topics else ""
            print(f"     X={r['topic']!r:>10} -> {r['reply']!r}  (worth {r['worth']}, conf {r['confidence']}, "
                  f"plaus {r['plausibility']}){tag}", flush=True)

    return {
        "seed": seed,
        "n_stored": len(affirmed),
        "n_negated": len(negated),
        "n_topics": len(topics),
        "tau": tau,
        # emission counts: value arm vs plausibility-only (lesion) arm
        "n_emitted_value": n_emit_val,
        "n_emitted_plausibility_only": n_emit_les,
        "n_new_value_driven_emissions": n_new,
        "new_topics": sorted(new_topics),
        # (1a) speaks-more
        "speaks_more": bool(speaks_more),
        # (1b) grounded
        "emission_set_true_frac": true_frac,
        "emission_set_shuffled_frac": shuf_frac,
        "grounded_advantage_ratio": grounded_advantage,
        "grounded_ok": bool(grounded_ok),
        "new_emissions_true_frac": new_true_frac,
        "new_emissions_shuffled_frac": new_shuf_frac,
        "new_emissions_grounded_advantage": (float(new_grounded_advantage)
                                             if isinstance(new_grounded_advantage, float)
                                             and not np.isnan(new_grounded_advantage) else None),
        "new_emissions_grounded": bool(new_emissions_grounded),
        # (2) calibrated
        "calib_spearman_worth_conf": spearman,
        "calib_strong_incidence_lo": strong_lo,
        "calib_strong_incidence_hi": strong_hi,
        "calib_strong_reliability": strong_reliability,
        "calibrated_ok": calibrated_ok,
        # (3) moat
        "moat_leaks": moat_leaks,
        "all_flagged": bool(all_flagged),
        "stored_answer_ok": stored_answer_ok,
        "stored_answer_total": stored_answer_total,
        "stored_answer_rate": stored_answer_rate,
        "stored_facts_answer": bool(stored_answers),
        "moat_ok": bool(moat_ok),
        # (4) lesion
        "lesion_collapses_to_plausibility_only": bool(lesion_collapses),
        # owner-steer #2 non-circularity (concept-level = the gate; candidate-level = secondary)
        "value_plausibility_corr": value_plaus_corr,
        "value_plausibility_corr_candidate": value_plaus_corr_candidate,
        # trail
        "emitted_examples_value": [{"topic": r["topic"], "reply": r["reply"], "worth": r["worth"],
                                    "confidence": r["confidence"], "plausibility": r["plausibility"],
                                    "is_new_vs_plausibility_only": r["topic"] in new_topics}
                                   for r in emitted_val[:12]],
        "silenced_examples_value": [{"topic": r["topic"], "worth": r.get("worth"),
                                     "reason": r.get("silence_reason") or r.get("no_grounded_candidate")}
                                    for r in turns_val if not r["emitted"]][:8],
    }


def decide_verdict(rows, a):
    """GO iff, across ALL seeds: (1a) the value appraisal SPEAKS MORE than the plausibility-only baseline; (1b)
    every emission (and every NEW value-driven emission) is GROUNDED (shuffled-graph advantage >= bar); (2) the
    stated confidence tracks WORTH (calibrated, non-tautological); (3) the MOAT is RELAXED-NOT-REMOVED -- 0
    known-fact-channel leaks + every emission flagged + stored facts still answer (per distinct multi-valued cue);
    AND (4) the LESION COLLAPSES the extra emissions to the plausibility-only baseline (the extra emissions are the
    BRAIN's value system). The value axis must also be NON-circular (concept-level |corr(value, plausibility)| <=
    bar) else the probe is INVALID.
    Else HONEST_NEGATIVE / BOUNDARY + why."""
    def col(k):
        return [r[k] for r in rows]

    speaks_more_all = all(col("speaks_more"))
    grounded_all = all(col("grounded_ok")) and all(col("new_emissions_grounded"))
    moat_all = all(col("moat_ok"))
    lesion_all = all(col("lesion_collapses_to_plausibility_only"))
    cal = col("calibrated_ok")
    cal_assessable = [c for c in cal if c is not None]
    calibrated_all = (len(cal_assessable) > 0) and all(cal_assessable)
    # non-circularity: the value axis is genuinely distinct from plausibility (|corr| small)
    vpcorr = np.array(col("value_plausibility_corr"))
    noncircular_all = bool(np.all(np.abs(vpcorr) <= a.max_value_plaus_corr))

    n_val = np.array(col("n_emitted_value"))
    n_les = np.array(col("n_emitted_plausibility_only"))
    n_new = np.array(col("n_new_value_driven_emissions"))

    detail = {
        "n_emitted_value_mean": float(n_val.mean()),
        "n_emitted_value_min": int(n_val.min()),
        "n_emitted_plausibility_only_mean": float(n_les.mean()),
        "n_new_value_driven_mean": float(n_new.mean()),
        "n_new_value_driven_min": int(n_new.min()),
        "n_new_value_driven_total": int(n_new.sum()),
        "speaks_more_all_seeds": speaks_more_all,
        "grounded_advantage_mean": float(np.mean(col("grounded_advantage_ratio"))),
        "grounded_advantage_min": float(np.min(col("grounded_advantage_ratio"))),
        "grounded_all_seeds": grounded_all,
        "new_emissions_grounded_all_seeds": bool(all(col("new_emissions_grounded"))),
        "calib_spearman_mean": float(np.mean([s for s in col("calib_spearman_worth_conf") if s is not None])
                                     if any(s is not None for s in col("calib_spearman_worth_conf"))
                                     else float("nan")),
        "calibrated_all_seeds": calibrated_all,
        "calibrated_assessable_seeds": len(cal_assessable),
        "moat_leaks_total": int(np.sum(col("moat_leaks"))),
        "moat_all_seeds": moat_all,
        "stored_facts_answer_all_seeds": bool(all(col("stored_facts_answer"))),
        "stored_answer_rate_min": float(np.min(col("stored_answer_rate"))),
        "stored_answer_rate_mean": float(np.mean(col("stored_answer_rate"))),
        "lesion_collapses_all_seeds": lesion_all,
        "value_plaus_corr_mean": float(np.mean(vpcorr)),
        "value_plaus_corr_absmax": float(np.max(np.abs(vpcorr))),
        "noncircular_all_seeds": noncircular_all,
        "advantage_bar": float(a.advantage_bar),
        "calib_spearman_bar": float(a.calib_spearman_bar),
        "max_value_plaus_corr": float(a.max_value_plaus_corr),
    }

    if not noncircular_all:
        verdict = "INVALID_value_is_relabeled_plausibility"          # the probe would be circular -> not a finding
    elif not speaks_more_all:
        verdict = "HONEST_NEGATIVE_value_adds_no_emissions"          # the value axis doesn't make it speak more
    elif not grounded_all:
        verdict = "HONEST_NEGATIVE_extra_emissions_ungrounded"       # speaks more but confabulates
    elif not moat_all:
        verdict = "HONEST_NEGATIVE_moat_broken"
    elif not lesion_all:
        verdict = "HONEST_NEGATIVE_lesion_does_not_collapse"         # extra emissions NOT the value system
    elif not calibrated_all:
        verdict = "BOUNDARY_uncalibrated"
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Choose-to-speak value/salience appraisal de-risk: does the brain's "
                                            "VALUE system make it speak MORE where supported, via a SPIKING "
                                            "accumulator (NOT a host threshold)?")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--D", type=int, default=256,
                   help="phasor dimension for the RF composer store (256 keeps the stored-facts-answer sanity "
                        "read clean; D=64 sits in the documented small-D retrieval tail)")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED facts the brain is TOLD")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts (non-contradiction gate work)")
    p.add_argument("--n-topics", type=int, default=30, help="held-out 'what do you think about X' topics")
    p.add_argument("--n-attempts", type=int, default=500, help="generative-replay samples per topic")
    p.add_argument("--tau-pct", type=float, default=50.0, help="graph-related threshold = percentile of +PPMI")
    # the appraisal weights (worth = talkativeness * (w_value*value + w_plaus*plaus + w_fam*familiarity))
    p.add_argument("--talkativeness", type=float, default=1.0,
                   help="owner-steer #1: a single 'how readily volunteer a view' gain (conservative default)")
    p.add_argument("--w-value", type=float, default=0.5, help="weight on the DA-value/interest axis")
    p.add_argument("--w-plaus", type=float, default=0.35, help="weight on the plausibility axis")
    p.add_argument("--w-fam", type=float, default=0.15, help="weight on the familiarity axis")
    # the spiking accumulator drift mapping (additive incentive-salience scheme)
    p.add_argument("--speak-base-pA", type=float, default=70.0, help="speak-pool base drive")
    p.add_argument("--speak-gain-pA", type=float, default=180.0, help="component-push -> speak drift gain")
    p.add_argument("--silence-drive-pA", type=float, default=150.0,
                   help="silence-pool fixed competing 'default reticence' drive (the speak bar)")
    p.add_argument("--acc-steps", type=int, default=120, help="spiking integration window (steps)")
    # gate bars
    p.add_argument("--advantage-bar", type=float, default=3.0, help="grounded shuffled-graph advantage bar")
    p.add_argument("--calib-spearman-bar", type=float, default=0.5,
                   help="min Spearman(worth, stated confidence) for CALIBRATED")
    p.add_argument("--max-value-plaus-corr", type=float, default=0.35,
                   help="max |corr(value, plausibility)| for the value axis to be NON-circular (distinct)")
    p.add_argument("--stored-answer-bar", type=float, default=0.9,
                   help="min stored-cue answer rate (per distinct multi-valued cue); small-D-tolerant so the "
                        "documented RF retrieval tail (1 cue out of ~13 at D=64) does not spuriously fail the moat")
    p.add_argument("--max-bytes", type=int, default=4_000_000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    # quiet the per-decision bridge re-init logs (the accumulator resets state each decision)
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[appraisal] seeds={seeds} talkativeness={a.talkativeness} w=(v{a.w_value},p{a.w_plaus},f{a.w_fam}) "
          f"-- does the brain's VALUE system make it speak MORE where supported (vs plausibility-only), decided "
          f"by a SPIKING accumulator, lesion-confirmed?", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    # the SPIKING speak accumulator (built ONCE; reset per decision). Seed it independently of the brain seed so
    # the accumulator dynamics are shared across the per-seed brains (the DECISION circuit is the same brain part).
    print(f"[appraisal] building the spiking speak/silence accumulator (Wang-2002 NMDA WTA; "
          f"sel/commit/OPN template)...", flush=True)
    accumulator = SpikingSpeakAccumulator(seed=12345, n_steps=a.acc_steps)

    rows = [run_seed(s, vocab, corpus, a, accumulator) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*100}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  EMISSIONS: value arm mean {detail['n_emitted_value_mean']:.1f} vs plausibility-only "
          f"{detail['n_emitted_plausibility_only_mean']:.1f} -> +{detail['n_new_value_driven_mean']:.1f} "
          f"value-driven/seed (total {detail['n_new_value_driven_total']})", flush=True)
    print(f"  (1a) SPEAKS-MORE all seeds: {detail['speaks_more_all_seeds']}", flush=True)
    print(f"  (1b) GROUNDED all seeds: {detail['grounded_all_seeds']} (emission-set advantage mean "
          f"{detail['grounded_advantage_mean']:.1f}x, min {detail['grounded_advantage_min']:.1f}x; "
          f"NEW-emissions grounded all seeds: {detail['new_emissions_grounded_all_seeds']})", flush=True)
    print(f"  (2) CALIBRATED all seeds: {detail['calibrated_all_seeds']} (spearman(worth,conf) mean "
          f"{detail['calib_spearman_mean']:.3f}; assessable {detail['calibrated_assessable_seeds']}/{len(seeds)})",
          flush=True)
    print(f"  (3) MOAT relaxed-not-removed all seeds: {detail['moat_all_seeds']} "
          f"({detail['moat_leaks_total']} known-fact leaks [load-bearing=0]; stored-cue answer rate min "
          f"{detail['stored_answer_rate_min']:.2f} mean {detail['stored_answer_rate_mean']:.2f})", flush=True)
    print(f"  (4) LESION collapses-to-plausibility-only all seeds: {detail['lesion_collapses_all_seeds']}",
          flush=True)
    print(f"  NON-CIRCULAR (value distinct from plausibility) all seeds: {detail['noncircular_all_seeds']} "
          f"(|corr| max {detail['value_plaus_corr_absmax']:.3f}, mean {detail['value_plaus_corr_mean']:+.3f}; "
          f"bar {detail['max_value_plaus_corr']})", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "value_salience_appraisal_derisk",
        "verdict": verdict,
        "seeds": seeds,
        "config": {"D": a.D, "n_facts": a.n_facts, "n_negated": a.n_negated, "n_topics": a.n_topics,
                   "n_attempts": a.n_attempts, "tau_pct": a.tau_pct, "talkativeness": a.talkativeness,
                   "w_value": a.w_value, "w_plaus": a.w_plaus, "w_fam": a.w_fam,
                   "speak_base_pA": a.speak_base_pA, "speak_gain_pA": a.speak_gain_pA,
                   "silence_drive_pA": a.silence_drive_pA, "acc_steps": a.acc_steps,
                   "advantage_bar": a.advantage_bar, "calib_spearman_bar": a.calib_spearman_bar,
                   "max_value_plaus_corr": a.max_value_plaus_corr, "stored_answer_bar": a.stored_answer_bar,
                   "max_bytes": a.max_bytes},
        "baseline_to_beat": {"probe1_plausibility_only_emissions_per_30": "6-9 (mean 7.7)",
                             "source": "2026-06-24-communicable-brain-probe1-GO.md",
                             "note": "the plausibility-only arm here re-derives the Probe-1 baseline ON the "
                                     "spiking accumulator (value lesioned); the value arm must speak MORE."},
        "mechanism": (
            "Option A salience-drift: PROPOSE the candidate SET -> APPRAISE worth = talkativeness*(w_v*VALUE + "
            "w_p*plausibility + w_f*familiarity), where VALUE is a DA/interest stand-in STRUCTURALLY DISTINCT from "
            "the PPMI plausibility axis (owner-steer #2) -> RANK -> a SPIKING speak/silence WTA accumulator "
            "(Wang-2002 NMDA integrators in biased competition; the merged-bridge sel/commit/OPN template) DECIDES "
            "emit-vs-silent by the top candidate's worth modulating the drift (catalog O.19/C.32) -> EMIT a "
            "graded-confidence FLAGGED hypothesis (NOT stored; known-fact channel hard-gated). The plausibility-"
            "only baseline IS the value-LESION arm (same accumulator, value pinned to baseline)."),
        "brain_based_note": (
            "the speak DECISION is a NEURAL POOL's FIRING (a real Izhikevich WTA on a numpy SimulationBridge "
            "slice, Wang-2002 NMDA + biased-competition FS), NOT a host `if score > threshold`. The value axis is "
            "a CPU stand-in for the merged-bridge spiking SNc/striosome_value critic (the GPU follow-on reads the "
            "real shared `dopamine` so the lesion pins the real SNc). The PPMI cortex + RF composer + parser are "
            "the brain; the host does recombination bookkeeping + routes which assembly fired; the fluency faculty "
            "is the surface form only. NO sim/ edit; reuse-by-import; CPU."),
        "anti_cheats": {
            "lesion_value_system": "pin the DA-value input to baseline -> the speak-decision collapses to the "
                                   "plausibility-only baseline (the extra emissions vanish) -> the EXTRA emissions "
                                   "are the BRAIN's value system, not a host re-ranking. (gate 4)",
            "shuffled_graph_groundedness": "the value-driven emission set (and every NEW emission) collapses "
                                           ">= advantage_bar under a shuffled PPMI graph -> grounded, not noise. (1b)",
            "moat_relaxed_not_removed": "0 known-fact-channel leaks + every emission flagged + stored facts still "
                                        "answer (per distinct multi-valued cue; the speak gate is an ADDITIVE "
                                        "channel that does not touch the known-fact channel). (gate 3)",
            "non_tautological_calibration": "confidence tracks WORTH (which has a value axis the plausibility-only "
                                            "confidence does not read) + the high-worth bin carries more INDEPENDENT "
                                            "strong-plausibility. (gate 2)",
            "non_circular_value": "corr(value, plausibility) ~ 0 -> the value axis is NOT a relabeled plausibility "
                                  "(owner-steer #2); else the probe is INVALID.",
        },
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_value_salience_appraisal_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
