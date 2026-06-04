"""Phase 2 of the consolidation: the conversational loop on the core sim, assembling the validated core-sim
pieces into ONE agent -- a Hebbian-learned syntactic PARSER (comprehension) + the `CoreSimComposer`
(composition / SVO fact-memory / who-what Q&A / abstention / negation / clauses). Everything is a genuine spiking
computation on `SimulationBridge` neurons; no bolted-on numpy phasor simulator in the path.

Comprehension: the parser learns the conjunctive (word-position x voice) -> role mapping by Hebbian co-firing
(active: 1st->agent 2nd->action 3rd->patient; passive flips 1st<->3rd), validated in-substrate
(`_insubstrate_parser_stdp_probe`). At comprehension time, driving each word's (position, voice) conjunction reads
out its role on the bridge, so "dog go north" (active) and "north is reached by dog" (passive frame) assign the
same agent. The parsed roles feed the composer, which binds + stores the fact in spiking on its own bridge.

Provenance (ported faithfully): `_insubstrate_parser_stdp_probe.py` (the Hebbian parser core) + the validated
CoreSimComposer. Operating point as validated. Reuse-by-import; no protected-module modification.
"""
from __future__ import annotations
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import CoreSimComposer, Clause

# parser ground truth: conjunction index k = position*2 + voice (voice 0=active, 1=passive)
_GT = {0: "agent", 1: "patient", 2: "action", 3: "action", 4: "patient", 5: "agent"}


class BridgeParser:
    """Learned (word-position x voice) -> role mapping realized on a SimulationBridge: 6 conjunction units ->
    3 role ensembles, plastic (Hebbian co-firing). Trained once at construction (the validated v16 rule)."""

    ROLES = ["agent", "action", "patient"]

    def __init__(self, seed=42, R=40, n_epochs=30, train_steps=120, test_steps=80, drive=2500.0):
        self.R = R; self.test_steps = test_steps; self.drive = drive
        cfg = CoreSimConfig()
        cfg.num_neurons = 6 + 3 * R
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = int(seed); cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0; cfg.num_traits = 1
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True       # v16 embodied-Hebbian CO-FIRING rule (pre&post-gated -> selective)
        cfg.hebbian_max_weight = 400.0
        cfg.hebbian_learning_rate = 0.005
        for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
                  "enable_reward_modulation", "enable_watts_strogatz"):
            setattr(cfg, f, False)
        cfg.ou_std_current_pA = 20.0
        self.conj = list(range(6))
        self.role_idx = {r: list(range(6 + i * R, 6 + (i + 1) * R)) for i, r in enumerate(self.ROLES)}
        pre, post, w = [], [], []
        for k in self.conj:
            for r in self.ROLES:
                for j in self.role_idx[r]:
                    pre.append(k); post.append(j); w.append(0.5)
        plan = {"parse": {"pre_indices": pre, "post_indices": post,
                          "initial_weights": np.array(w, dtype=np.float32),
                          "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}}
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge.inject_explicit_wiring(plan)
        xp, _ = get_backend()
        self.conj_arr = xp.asarray(self.conj, dtype=xp.int64)
        self.role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in self.role_idx.items()}
        self._n = 6 + 3 * R
        self._train(n_epochs, train_steps)

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self, n_epochs, train_steps):
        xp, _ = get_backend()
        for _ in range(n_epochs):
            for k in range(6):
                self._step_reset()
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[self.conj_arr[k]] = self.drive
                cur[self.role_arr[_GT[k]]] = self.drive       # teacher-drive the correct role
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(train_steps):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def role_of(self, position, voice=0):
        """Drive the (position, voice) conjunction ALONE; read which role ensemble fires most -> the learned role."""
        xp, _ = get_backend()
        k = position * 2 + (0 if voice in (0, "active") else 1)
        self._step_reset()
        cur = xp.zeros(self._n, dtype=xp.float32)
        cur[self.conj_arr[k]] = self.drive
        self.bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in self.ROLES}
        for _ in range(self.test_steps):
            self.bridge._run_one_simulation_step()
            for r in self.ROLES:
                rates[r] += float(to_host(self.bridge.cp_firing_states[self.role_arr[r]].astype(xp.float64).mean()))
        self.bridge.cp_external_input_current[:] = 0.0
        return max(rates, key=rates.get)

    def parse(self, words, voice="active"):
        """Comprehend a 3-word SVO sentence -> {role: word}, assigning each word to the role its (position, voice)
        conjunction reads out on the bridge."""
        assert len(words) == 3, "this minimal parser handles 3-word SVO sentences"
        return {self.role_of(pos, voice): words[pos] for pos in range(3)}


class BrainConversationalAgent:
    """The conversational loop on the core sim: comprehend (parser) -> store/recall/compose (composer). Hear SVO
    statements, answer who/what, abstain on the unknown, negate, and handle embedded clauses -- all spiking on
    SimulationBridge neurons, the substrate's own concept codes, no bolted-on numpy simulator."""

    def __init__(self, seed=42, proj_dim=800, concepts=None):
        """`concepts` (optional) = a {word: code} dict to construct the composer at an arbitrary vocabulary
        (e.g. the production 320-concept code scheme) instead of the default V=16 `denoise64` cache. The parser is
        vocabulary-agnostic (it assigns roles by word position x voice), so the same parser serves any vocab."""
        self.seed = int(seed)
        self.parser = BridgeParser(seed=seed)
        self.composer = CoreSimComposer(seed=seed, proj_dim=proj_dim, concepts=concepts)
        self._dlpfc = None              # dialogue-planning Control: built lazily, cached, rebuilt only when the graph changes
        self._dlpfc_key = None

    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO statement and store it. `sentence` is 'agent action patient' (or its passive frame)."""
        roles = self.parser.parse(sentence.split(), voice)
        self.composer.store(roles["agent"], roles["action"], roles["patient"], polarity=polarity)
        return roles

    def hear_clause_fact(self, agent, action, clause, polarity=None):
        """Store a fact whose patient is an embedded clause (the parser handles flat SVO; nested input parsing is
        future work, so the clause is provided structurally here)."""
        self.composer.store(agent, action, clause, polarity=polarity)

    def what_does(self, agent, action):
        """'what does <agent> <action>?' -> patient (concept or rendered clause) or None (abstain)."""
        return self.composer.query_patient(agent, action)

    def who_does(self, action, patient):
        return self.composer.query_agent(action, patient)

    def is_it_true(self, agent, action, patient):
        return self.composer.ask_yes_no(agent, action, patient)

    def describe(self, agent):
        """Generation: produce a sentence about `agent` from the spiking memory ('dog go north'), or None if the
        agent knows no fact about it (no confabulation)."""
        return self.composer.render_fact(agent)

    # --- dialogue planning (what to say next) ---
    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) built from the agent's OWN stored facts: the
        agent/action/patient of each fact co-occur. Clause patients are skipped (their inner concepts are
        structural). This is the graph the dialogue-planning Control spreads over."""
        graph = {}
        for fact, _ in self.composer.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient")]
            cs = [c for c in cs if isinstance(c, str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning: bring up the next on-topic concept about `topic`, chosen by the dlPFC spiking
        content-selection Control (loop-attractor working memory + spreading activation) over the agent's own
        association graph -- all on a SimulationBridge. Returns an associate concept, or None if `topic` is
        unconnected. (Builds the dlPFC bridge on demand from the current facts.)"""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        graph = self._assoc_graph()
        if topic not in graph:
            return None
        # cache key = the graph CONTENT (not kb length: different fact sets can share a length -> stale Control)
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc is None or self._dlpfc_key != key:
            self._dlpfc = SpikingSpreadingController(graph, seed=self.seed)   # first-class: rebuild only when the graph changes
            self._dlpfc_key = key
        return self._dlpfc.turn_latency([topic])
