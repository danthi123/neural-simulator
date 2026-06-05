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
from collections import namedtuple

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge

ROLES = ("agent", "action", "patient", "polarity", "attribute")
DEFAULT_VOCAB = ["dog", "cat", "go", "run", "stop", "look", "north", "south", "east", "west", "apple", "river",
                 "big", "small", "hot", "cold"]
# A recursive SVO clause that can be a filler ('dog look (cat go north)'). Mirrors core_sim_composition.Clause.
Clause = namedtuple("Clause", ["agent", "action", "patient"])


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
    def __init__(self, seed=42, D=64, vocab=None, period=200):
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
        self._dlpfc = None       # dialogue-planning Control (lazy; rebuilt only when the association graph changes)
        self._dlpfc_key = None
        self._bridge_cache = {}  # (c-opt) reuse RF bridges by neuron count -> avoid _initialize_simulation_data per op

    # --- RF complex-synapse ops (each op a per-op RF bridge; reuse-by-import the substrate) ---
    def _resonate(self, n, conns, kick):
        # (c-opt) reuse a cached bridge per neuron count; zero its complex weights (rf_set_complex_weights appends)
        # and rf_kick resets the RF state -> each op is clean. Avoids _initialize_simulation_data per op.
        b = self._bridge_cache.get(n)
        if b is None:
            b = _build_rf_bridge(n, self.seed)
            self._bridge_cache[n] = b
        b.rf_set_complex_weights(conns)   # (c-opt) builds the sparse complex weights FRESH each op -> replaces; no reset needed
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

    def _filler_phases(self, filler):
        """The phasor phases to bind for a filler: a concept's code, OR (recursively) a Clause's bound composite."""
        if isinstance(filler, Clause):
            return self._encode({"agent": filler.agent, "action": filler.action, "patient": filler.patient})
        return self.concepts[filler]

    def _encode(self, fact):
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ROLES if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _render(self, comp_phases, role, stored):
        """Render `role`'s filler from a composite, FROM THE RF UNBIND. `stored` (a word or Clause) ROUTES
        flat-cleanup vs recursive clause-decode; the content is decoded from the substrate, not the stored labels."""
        rec = self._unbind_phases(comp_phases, role)
        if isinstance(stored, Clause):
            a = self._cleanup(self._unbind_phases(rec, "agent"))
            ac = self._cleanup(self._unbind_phases(rec, "action"))
            pt = self._cleanup(self._unbind_phases(rec, "patient"))
            return f"{a} {ac} {pt}"
        return self._cleanup(rec)

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
        if isinstance(patient, Clause):            # a recursive clause filler (check BEFORE tuple: Clause IS a tuple)
            fact["patient"] = patient
        elif isinstance(patient, tuple):           # ('big', 'apple') -- an attributed entity (1-attribute)
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
                noun = self._render(comp, "patient", fact["patient"])   # a word OR a recursive Clause
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

    def render_fact(self, agent):
        """Generation: render a full stored sentence whose agent matches `agent` -- e.g. 'dog go north' (an
        attributed patient 'big apple' or a nested clause renders too). The action + patient are DECODED from the
        RF unbind (not the stored labels); None if no fact's agent matches (the no-confab moat -- no invented
        sentence about an unknown subject)."""
        for fact, comp in self.kb:
            if self.unbind(comp, "agent") == agent:
                ac = self.unbind(comp, "action")
                pt = self._render(comp, "patient", fact["patient"])
                if "attribute" in fact:
                    pt = f"{self.unbind(comp, 'attribute')} {pt}"
                return f"{agent} {ac} {pt}"
        return None

    # --- dialogue planning (the dlPFC content-selection Control; architecture-independent: operates on the graph) ---
    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) from the stored facts (agent/action/patient co-occur;
        clause patients are skipped -- their inner concepts are structural). The graph the dlPFC spreads over."""
        graph = {}
        for fact, _ in self.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient") if isinstance(fact.get(r), str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning: the next on-topic concept about `topic`, chosen by the dlPFC spiking content-selection
        Control (loop-attractor working memory + spreading activation) over the agent's own association graph -- the
        same validated SpikingSpreadingController the rate-coded agent uses (it operates on the GRAPH, so it is
        substrate-independent). None if `topic` is unconnected."""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        graph = self._assoc_graph()
        if topic not in graph:
            return None
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc is None or self._dlpfc_key != key:
            self._dlpfc = SpikingSpreadingController(graph, seed=self.seed)
            self._dlpfc_key = key
        return self._dlpfc.turn_latency([topic])
