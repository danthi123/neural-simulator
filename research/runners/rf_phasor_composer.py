"""FHRR-on-bridge layer (b): a PARALLEL RF phasor composer running the conversational composition on the bridge's
resonate-and-fire neurons + complex synapses -- so the opponency (the rate-coded composer's SNR wall) is GONE (the
phasor algebra has no common mode). Same conversational API as core_sim_composition.CoreSimComposer; validated at
parity before the BrainConversationalAgent switches (layer c). Design:
docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md.

Reuse-by-import the RF + complex-synapse substrate already on the bridge (NeuronModel.RESONATE_AND_FIRE +
rf_kick / rf_read_phases / rf_set_complex_weights, layers RF-on-bridge + layer-a). NO sim/ edits here.

Representation: each concept/role is a PHASOR vector (phases in [0,1)^D, deterministic per seed). bind = role (x)
filler via a DIAGONAL complex synapse (weight = the role phasor); bundle = unit complex synapses (the sum -- NO
opponency); unbind = conj diagonal synapse; cleanup = phase-cosine argmax. Abstention (the no-confab moat): the
relational query returns None when no stored fact's cue roles match (architecture-preserved).
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge

ROLES = ("agent", "action", "patient", "polarity", "attribute")
DEFAULT_VOCAB = ["dog", "cat", "go", "run", "stop", "look", "north", "south", "east", "west", "apple", "river",
                 "big", "small", "hot", "cold"]


def _build_rf_bridge(n, seed=42):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return bridge


class RFPhasorComposer:
    def __init__(self, seed=42, D=64, vocab=None, period=400):
        self.seed = int(seed)
        self.D = int(D)
        self.period = int(period)
        self.words = sorted(vocab) if vocab is not None else sorted(DEFAULT_VOCAB)
        rng = np.random.default_rng(seed)
        # phasor codes: phases in [0,1)^D per concept + per role (deterministic per seed)
        self.concepts = {w: rng.uniform(0.0, 1.0, self.D) for w in self.words}
        # AFFIRM/NEGATE polarity fillers (phasor codes; cleaned up only against pol_words, not the main vocab)
        self.pol_words = ["AFFIRM", "NEGATE"]
        for tag in self.pol_words:
            self.concepts[tag] = rng.uniform(0.0, 1.0, self.D)
        self.roles = {r: rng.uniform(0.0, 1.0, self.D) for r in ROLES}
        self.kb = []  # (fact_dict, composite_phases)

    # --- RF complex-synapse ops (each op a per-op RF bridge; reuse-by-import the substrate) ---
    def _resonate(self, n, conns, kick):
        b = _build_rf_bridge(n, self.seed)
        b.rf_set_complex_weights(conns)
        b.rf_kick(kick, period=self.period, lam=0.0)
        for _ in range(self.period + 8):
            b._run_one_simulation_step()
        return np.asarray(b.rf_read_phases())

    @staticmethod
    def _to_phasor(phases):
        return np.exp(2j * np.pi * np.asarray(phases))

    def _bind(self, role_phases, filler_phases):
        """bound = role_phasor (x) filler_phasor, via a diagonal complex synapse (filler pre -> bound post,
        weight = the role phasor)."""
        D = self.D
        zf = self._to_phasor(filler_phases)
        zr = self._to_phasor(role_phases)
        conns = [(D + k, k, zr[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zf
        return self._resonate(2 * D, conns, kick)[D:]

    def _bundle(self, phase_list):
        """composite[k] = sum_l phase_list[l][k] via unit complex synapses (NO opponency)."""
        L = len(phase_list)
        D = self.D
        conns = [(L * D + k, l * D + k, 1.0) for l in range(L) for k in range(D)]
        kick = np.zeros((L + 1) * D, dtype=np.complex128)
        for l in range(L):
            kick[l * D:(l + 1) * D] = self._to_phasor(phase_list[l])
        return self._resonate((L + 1) * D, conns, kick)[L * D:]

    def _encode(self, fact):
        bounds = [self._bind(self.roles[r], self.concepts[fact[r]]) for r in ROLES if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _unbind_phases(self, composite_phases, role):
        """recovered = conj(role_phasor) (x) composite, via a conj diagonal complex synapse."""
        D = self.D
        zc = self._to_phasor(composite_phases)
        zr_conj = np.conj(self._to_phasor(self.roles[role]))
        conns = [(D + k, k, zr_conj[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zc
        return self._resonate(2 * D, conns, kick)[D:]

    def _cleanup(self, rec_phases, words=None):
        words = words if words is not None else self.words
        sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - self.concepts[w])))) for w in words]
        return words[int(np.argmax(sims))]

    def unbind(self, composite_phases, role, words=None):
        return self._cleanup(self._unbind_phases(composite_phases, role), words)

    # --- conversational API (mirrors CoreSimComposer; the no-confab moat preserved) ---
    def store(self, agent, action, patient, polarity=None):
        fact = {"agent": agent, "action": action}
        if isinstance(patient, tuple):            # ('big', 'apple') -- an attributed entity (1-attribute)
            adj, noun = patient
            fact["patient"] = noun
            fact["attribute"] = adj
        else:
            fact["patient"] = patient
        if polarity is not None:
            fact["polarity"] = polarity      # a bound AFFIRM/NEGATE tag (extra binding -> more load)
        self.kb.append((fact, self._encode(fact)))

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> the agent of the matching fact; None if no fact matches (abstention)."""
        for fact, comp in self.kb:
            if self.unbind(comp, "action") == action and self.unbind(comp, "patient") == patient:
                return self.unbind(comp, "agent")
        return None

    def query_patient(self, agent, action):
        """'what does <agent> <action>?' -> the patient of the matching fact (an attributed entity 'big apple' if
        the fact bound an ATTRIBUTE); None if no match (abstention). The stored structure only routes the rendering;
        the words are decoded from the RF unbind."""
        for fact, comp in self.kb:
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                noun = self.unbind(comp, "patient")
                if "attribute" in fact:
                    return f"{self.unbind(comp, 'attribute')} {noun}"
                return noun
        return None

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown' via the bound AFFIRM/NEGATE polarity tag.
        Matches the full SVO; 'unknown' (abstention) when no stored fact matches."""
        for fact, comp in self.kb:
            if (self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action
                    and self.unbind(comp, "patient") == patient):
                return "yes" if self.unbind(comp, "polarity", self.pol_words) == "AFFIRM" else "no"
        return "unknown"
