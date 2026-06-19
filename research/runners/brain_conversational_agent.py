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

    def __init__(self, seed=42, R=40, n_epochs=30, train_steps=120, test_steps=80, drive=2500.0,
                 shared_bridge=None, index_offset=0, defer_train=False):
        """Build a 6-conjunction → 3-role-ensemble Hebbian parser.

        Default (standalone) path — `shared_bridge=None`: build a private SimulationBridge of `6 + 3*R`
        neurons, wire + train it (unchanged from before).

        Shared-bridge path — `shared_bridge` given: do NOT build/init a bridge; use the provided one. Every
        conjunction unit (0..5) and role-ensemble neuron lives at `index_offset + local_index` on the shared
        bridge, so the `"parse"` wiring plan AND the drive/readout index arrays are all shifted by
        `index_offset`. The `"parse"` population is added to the shared bridge via
        `merge_population_into_shared_bridge` (which re-injects the running union of every region wired so
        far — see that helper's docstring). When `defer_train=True`, training is skipped in `__init__` and
        the caller invokes `train()` AFTER all other populations are wired (a later re-injection would
        otherwise reset the trained `"parse"` weights to their initial design values); the unified builder
        uses this. When `defer_train=False` (default), training runs in `__init__` as before."""
        self.R = R; self.test_steps = test_steps; self.drive = drive
        self.index_offset = int(index_offset)
        # Local layout (pre-offset): conjunction units 0..5, then 3 role ensembles of R neurons each.
        self.conj = [self.index_offset + k for k in range(6)]
        self.role_idx = {r: [self.index_offset + 6 + i * R + j for j in range(R)]
                         for i, r in enumerate(self.ROLES)}
        pre, post, w = [], [], []
        for k in self.conj:
            for r in self.ROLES:
                for j in self.role_idx[r]:
                    pre.append(k); post.append(j); w.append(0.5)
        plan = {"parse": {"pre_indices": pre, "post_indices": post,
                          "initial_weights": np.array(w, dtype=np.float32),
                          "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}}

        if shared_bridge is None:
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
            self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                           runtime_state=RuntimeState(), gpu_config=GPUConfig())
            self.bridge._initialize_simulation_data(called_from_playback_init=False)
            self.bridge.inject_explicit_wiring(plan)
        else:
            self.bridge = shared_bridge
            from research.runners.unified_brain_bridge import merge_population_into_shared_bridge
            merge_population_into_shared_bridge(self.bridge, plan)   # parser pop is plastic → no gate

        xp, _ = get_backend()
        self.conj_arr = xp.asarray(self.conj, dtype=xp.int64)
        self.role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in self.role_idx.items()}
        # `self._n` is the size of the *current-state* arrays we zero/index = the whole bridge.
        self._n = self.bridge.core_config.num_neurons
        self._n_epochs = n_epochs
        self._train_steps = train_steps
        if not defer_train:
            self._train(n_epochs, train_steps)

    def train(self):
        """Run the (deferred) Hebbian training. Used by the unified builder, which wires ALL regions onto the
        shared bridge first and trains the parser LAST — so a later population re-injection cannot reset the
        trained `"parse"` weights. Uses the epoch/step counts passed at construction."""
        self._train(self._n_epochs, self._train_steps)

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

    def __init__(self, seed=42, proj_dim=800, concepts=None, composer=None, composer_kind="rf",
                 enable_spiking_cleanup=False, enable_substrate_store=False, grounded_codes=None,
                 enable_learned_assoc=False, enable_neural_render=True, enable_rf_cudagraph=False,
                 enable_attributed=True, enable_multiframe=True,
                 enable_multicue_competition=False, multicue_verbs=None):
        """`concepts` (optional) = a {word: code} dict to set the vocabulary instead of the defaults. The parser is
        vocabulary-agnostic (it assigns roles by word position x voice), so the same parser serves any vocab.

        `composer_kind` (default **'rf'** as of the 2026-06-05 production switch): the DEFAULT composer substrate
        when no explicit `composer` is passed --
          - 'rf'   = the FHRR-on-bridge `RFPhasorComposer` (resonate-and-fire phasor neurons + complex synapses) --
                     OPPONENCY-FREE; the production default. Validated end-to-end (full capability matrix,
                     320-correctness, optimized, zero regression).
          - 'rate' = the legacy rate-coded `CoreSimComposer` (the +-1 Hadamard; opponency-bounded) -- kept as an
                     explicit opt-in (`composer_kind='rate'`) + needs the denoise64 cache.
        `composer` (optional) = an externally-constructed composer instance, overriding `composer_kind`. The agent
        delegates all fact storage/retrieval to the composer; the parser + dialogue-planning are composer-agnostic."""
        self.seed = int(seed)
        if composer is not None:
            self.composer = composer
        elif composer_kind == "rate":
            self.composer = CoreSimComposer(seed=seed, proj_dim=proj_dim, concepts=concepts)
        elif composer_kind == "onebrain":
            # the production OneBrainComposer: the WHOLE who/what pipeline (comprehend -> store -> query -> abstain) on
            # ONE persistent co-resident bridge, no host round-trips between ops. It carries its OWN on-bridge parser,
            # so `hear()` below delegates comprehension to it (one parser on the one brain). Affirmative-fact scope
            # (negation = a follow-on). See 2026-06-18-one-brain-composer-A3-GO.md.
            from research.runners.one_brain_composer import OneBrainComposer
            vocab = sorted(concepts.keys()) if isinstance(concepts, dict) else None
            # enable_attributed / enable_multiframe (richer-syntax consolidation, default OFF = byte-identical) pass
            # through to the production OneBrainComposer: single-attribute entities ('big apple', the 2-factor path
            # validated 100% on the learned 320 codes) + auto-selected multi-frame comprehension. The F=3 two-attribute
            # path is deliberately NOT wired (it degrades to ~29% on the correlated learned codes -- the documented
            # boundary, 2026-06-19-resonator-on-learned-codes-derisk.md).
            self.composer = OneBrainComposer(seed=seed, D=128, vocab=vocab, grounded_codes=grounded_codes,
                                             enable_attributed=enable_attributed,
                                             enable_multiframe=enable_multiframe)
        else:
            from research.runners.rf_phasor_composer import RFPhasorComposer
            vocab = sorted(concepts.keys()) if isinstance(concepts, dict) else None
            # enable_spiking_cleanup (opt-in): route cleanup through the fully-on-bridge spiking path (matched filter
            # on the complex synapse + Izhikevich WTA). == numpy at parity multi-seed; default OFF = numpy fast path.
            # period STAYS 200: the resonate-window shortening (2026-06-17-resonate-period-free-speedup.md) is real
            # for FLAT who/what (full accuracy at period>=32), but period=48 BREAKS embedded clauses
            # (test_embedded_clause fails) -- the recursive-clause unbind (a clause bound as a filler -> deeper
            # nesting + more bundle cross-talk) needs more phase resolution than flat queries. A period adoption is
            # gated on the FULL conversational suite (the clause test is the binding constraint); the clause-safe
            # threshold (likely ~100-128, still a ~1.6-2x win) is a bounded follow-on sweep.
            # enable_rf_cudagraph (opt-in, GPU-only): route the resonate step through the fused RF megakernel
            # (1 CUDA launch/step instead of ~15). Default OFF = byte-identical loop path; validated answer-identical
            # across the full conversational stack incl. embedded clauses (2026-06-17-rf-megakernel-resonate-GO.md).
            self.composer = RFPhasorComposer(seed=seed, D=128, vocab=vocab, period=200,
                                             enable_spiking_cleanup=enable_spiking_cleanup,
                                             enable_substrate_store=enable_substrate_store,
                                             grounded_codes=grounded_codes,
                                             enable_rf_cudagraph=enable_rf_cudagraph)
        # The agent's own comprehension parser -- built ONLY when the composer does not carry its own. The
        # OneBrainComposer carries an on-bridge parser (it has `hear`), so for it there is ONE parser on the one brain
        # and the agent's separate parser is skipped; the rf / rate / external paths build the agent parser as before.
        self.parser = None if hasattr(self.composer, "hear") else BridgeParser(seed=seed)
        self._dlpfc = None              # dialogue-planning Control: built lazily, cached, rebuilt only when the graph changes
        self._dlpfc_key = None
        # (cheat-D conversion, opt-in) the dialogue-planning association graph LEARNED in the substrate (a sparse
        # Hebbian recurrent, the CA3-autoassociator mechanism) instead of recomputed from the Python kb. hear() updates
        # it as facts arrive; _assoc_graph reads it. Validated multi-seed (24/24 edges, 9/9 top associate).
        self._learned_assoc = None
        if enable_learned_assoc:
            from research.runners.learned_assoc_graph import LearnedAssocGraph
            vocab = self.composer.words if hasattr(self.composer, "words") else (
                sorted(concepts.keys()) if isinstance(concepts, dict) else None)
            self._learned_assoc = LearnedAssocGraph(list(vocab), seed=seed)
        # (sentence-generation de-templating, opt-in) route describe()'s word ORDERING through the de-risked spiking
        # competitive-queuing serial-order generator instead of the host f-string. Default OFF = the f-string (the
        # production suite byte-preserved). The no-confab moat is unaffected (render_fact abstains BEFORE ordering).
        self._neural_render = None
        if enable_neural_render:
            from research.runners.neural_serial_order_renderer import NeuralSerialOrderRenderer
            self._neural_render = NeuralSerialOrderRenderer(seed=seed)
        # (richer-syntax #1, opt-in) attributed-entity comprehension ('dog eat big apple'): a neural
        # AttributedBridgeParser (from-start x from-END x voice conjunction, parse-in-spikes) parses 'S V adj* N'
        # on its own bridge; `hear_attributed` routes the parsed (adjs, noun) to the composer's attribute role.
        # Default OFF = byte-identical. Validated end-to-end 6/6 (2026-06-18-neural-attributed-endtoend-GO.md).
        # On the onebrain path the composer is also built with enable_attributed=True (above), so its store()/query
        # bind + read the single-attribute role -- single-attribute ('big apple') HOLDS 100% on the learned 320 codes;
        # the F=3 two-attribute path is NOT wired (it degrades to ~29% on the correlated learned codes -- the
        # documented boundary, 2026-06-19-resonator-on-learned-codes-derisk.md).
        self.enable_attributed = bool(enable_attributed)
        self._attr_parser = None
        if enable_attributed:
            from research.runners.attributed_parser import AttributedBridgeParser
            self._attr_parser = AttributedBridgeParser(seed=seed)
        # (richer-syntax #2, opt-in) multi-frame comprehension: a neural FrameParser (verb-position -> frame selection +
        # position x frame -> role) comprehends a sentence in an AUTO-SELECTED word-order frame (SVO/VSO/OSV).
        # `hear_multiframe(sentence, verbs)` routes through it; default OFF = byte-identical (the native BridgeParser /
        # onebrain SVO path is unchanged). Validated GO 6/6 (2026-06-18-frame-selection-GO.md). Built lazily.
        self.enable_multiframe = bool(enable_multiframe)
        self._frame_parser = None
        # (robust-comprehension wire-in, opt-in) multi-cue role COMPETITION: route hear()'s AGENT/PATIENT decision
        # through the validated SPIKING multi-cue role-competition (MultiCueRoleParser -- de-risk
        # 2026-06-19-multicue-competition-spiking-derisk.md, GO), so DEGRADED English (object-fronted / scrambled
        # word order) still assigns roles correctly where the position-only BridgeParser collapses. Default OFF =
        # byte-identical (the parser is never even constructed; the existing tests pass verbatim). Requires
        # `multicue_verbs` (the known-verb set the lexical front-end uses to find the sentence's verb). The
        # no-confab moat is preserved end-to-end (the composer's Q&A still abstains on any unstored fact, and the
        # parser's parse_decisive exposes the content gate for an ambiguous sentence). Built lazily/cached.
        # WIRED: the validated spiking role-competition INFERENCE (install-path validities). DEFERRED: continual
        # on-substrate validity LEARNING (seed-variable, documented) + neuralizing the learner's reward.
        self.enable_multicue_competition = bool(enable_multicue_competition)
        self._multicue_verbs = set(multicue_verbs) if multicue_verbs else None
        self._multicue_parser = None
        if enable_multicue_competition and self._multicue_verbs is None:
            raise ValueError("enable_multicue_competition=True needs multicue_verbs=<known-verb set> "
                             "(the lexical front-end that finds the sentence's verb)")

    def _ensure_multicue_parser(self):
        """Lazily build + cache the spiking MultiCueRoleParser (one bridge build, install-path validities)."""
        if self._multicue_parser is None:
            from research.runners.multicue_role_parser import MultiCueRoleParser
            self._multicue_parser = MultiCueRoleParser(known_verbs=self._multicue_verbs, seed=self.seed)
        return self._multicue_parser

    def hear_multicue(self, sentence, voice="active", polarity=None):
        """Comprehend a (possibly DEGRADED-order) transitive sentence with the SPIKING multi-cue role-competition
        and store the resolved fact, so an object-fronted 'apple eat dog' assigns the SAME agent (dog) / patient
        (apple) as canonical 'dog eat apple' -- where the position-only parser would invert them. The verb is
        identified lexically from `multicue_verbs`; the noun roles are the spiking WTA decision. Returns the parsed
        {role: word}. Requires enable_multicue_competition=True. The no-confab moat is unaffected (composer Q&A
        abstains on any unstored fact)."""
        assert self.enable_multicue_competition, \
            "hear_multicue needs BrainConversationalAgent(enable_multicue_competition=True, multicue_verbs=...)"
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = self._ensure_multicue_parser().parse(words, voice)
        self.composer.store(roles.get("agent"), roles.get("action"), roles.get("patient"), polarity=polarity)
        if self._learned_assoc is not None:
            self._learned_assoc.store_fact([roles.get("agent"), roles.get("action"), roles.get("patient")])
        return roles

    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO statement and store it. `sentence` is 'agent action patient' (or its passive frame).

        When the composer carries its OWN on-bridge parser (the OneBrainComposer: comprehension + storage on ONE
        persistent bridge), `hear()` DELEGATES comprehension to it -- one parser on the one brain, the parse result
        flowing operand->bind as spikes, not via the agent's separate parser. Otherwise the agent's parser comprehends
        and the composer stores the resolved roles (the rf / rate default path, byte-unchanged).

        When enable_multicue_competition is ON, hear() routes the AGENT/PATIENT decision through the SPIKING
        multi-cue role-competition (robust to degraded word order) instead of the position-only parser -- so the
        production turn entry point comprehends scrambled / object-fronted input correctly. Default OFF =
        byte-identical (the multicue parser is never built; this branch is skipped)."""
        if self.enable_multicue_competition:
            return self.hear_multicue(sentence, voice, polarity=polarity)
        if hasattr(self.composer, "hear"):
            roles = self.composer.hear(sentence, voice, polarity=polarity)
        else:
            roles = self.parser.parse(sentence.split(), voice)
            self.composer.store(roles["agent"], roles["action"], roles["patient"], polarity=polarity)
        if self._learned_assoc is not None:                  # learn the concept co-occurrence in the substrate
            self._learned_assoc.store_fact([roles["agent"], roles["action"], roles["patient"]])
        return roles

    def hear_clause_fact(self, agent, action, clause, polarity=None):
        """Store a fact whose patient is an embedded clause (the parser handles flat SVO; nested input parsing is
        future work, so the clause is provided structurally here)."""
        self.composer.store(agent, action, clause, polarity=polarity)

    def hear_attributed(self, sentence, voice="active", polarity=None):
        """Comprehend an attributed-entity sentence ('dog eat big red apple') with the NEURAL attributed parser
        (parse-in-spikes) and store it -- the parsed (adjective(s), noun) is routed to the composer's ready
        attribute/attribute2 roles, so `what_does('dog','eat')` -> 'big red apple'. Requires enable_attributed=True.
        Returns the parsed {role: word}. (Richer-syntax #1; the production hear() auto-routing is a follow-on.)"""
        assert self._attr_parser is not None, "hear_attributed needs BrainConversationalAgent(enable_attributed=True)"
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = self._attr_parser.parse(words, voice)
        adjs = [roles[r] for r in ("attribute", "attribute2") if r in roles]
        noun = roles.get("patient")
        patient = (adjs, noun) if adjs else noun
        self.composer.store(roles.get("agent"), roles.get("action"), patient, polarity=polarity)
        return roles

    def hear_multiframe(self, sentence, verbs, polarity=None):
        """Comprehend a sentence in an AUTO-SELECTED word-order frame (SVO/VSO/OSV) with the NEURAL FrameParser
        (verb-position -> frame selection + position x frame -> role, both spiking) and store the resolved fact, so
        'ran dog north' (VSO) and 'north dog ran' (OSV) answer who/what just like the native 'dog ran north' (SVO).
        `verbs` is the agent's known-verb set (the lexical front end the frame selector uses to find the verb).
        Requires enable_multiframe=True. Returns the parsed {role: word}. (Richer-syntax #2.)"""
        assert self.enable_multiframe, "hear_multiframe needs BrainConversationalAgent(enable_multiframe=True)"
        if self._frame_parser is None:
            from research.runners.frame_parser import FrameParser
            self._frame_parser = FrameParser(seed=self.seed)
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = self._frame_parser.parse(words, verbs)
        self.composer.store(roles.get("agent"), roles.get("action"), roles.get("patient"), polarity=polarity)
        if self._learned_assoc is not None:
            self._learned_assoc.store_fact([roles.get("agent"), roles.get("action"), roles.get("patient")])
        return roles

    def parse(self, words, voice="active"):
        """Comprehend an SVO into {agent, action, patient}, using whichever parser the agent has: its OWN parser (the
        rf / rate / external paths) OR the composer's on-bridge parser (the OneBrainComposer carries the one parser on
        the one brain). A single comprehension entry point so callers (e.g. a correction turn) don't depend on which
        composer is wired -- `self.parser` is None on the onebrain path."""
        parser = self.parser if self.parser is not None else getattr(self.composer, "parser", None)
        if parser is None:
            raise RuntimeError("BrainConversationalAgent has no parser (composer carries neither a parser nor hear)")
        return parser.parse(list(words), voice)

    def what_does(self, agent, action):
        """'what does <agent> <action>?' -> patient (concept or rendered clause) or None (abstain). With
        enable_neural_render, an inner-clause patient's word ORDER is produced by the spiking serial-order
        generator (the moat is unaffected: abstention returns None before any rendering)."""
        if self._neural_render is not None:
            return self.composer.query_patient(agent, action, order_fn=lambda n: self._neural_render.order(list(range(n))))
        return self.composer.query_patient(agent, action)

    def who_does(self, action, patient):
        return self.composer.query_agent(action, patient)

    def is_it_true(self, agent, action, patient):
        return self.composer.ask_yes_no(agent, action, patient)

    def reason_chain(self, cue, actions):
        """Multi-hop relational reasoning: chain stored facts, each hop's patient becoming the next hop's agent
        cue. reason_chain('dog', ['eat', 'eat']) -> 'mouse' over {dog eat cat, cat eat mouse}; None (abstain) the
        moment any hop has no matching fact (the no-confab moat holds at EVERY hop). Delegates to the composer's
        query_chain -- de-risked GO 3 seeds x 3 D, every anti-cheat collapsing (2026-06-17-multihop-query-chain-GO.md)."""
        return self.composer.query_chain(cue, actions)

    def describe(self, agent):
        """Generation: produce a sentence about `agent` from the spiking memory ('dog go north'), or None if the
        agent knows no fact about it (no confabulation). With enable_neural_render, the word ORDER is produced by
        the de-risked spiking competitive-queuing serial-order generator instead of the host f-string."""
        if self._neural_render is not None:
            return self.composer.render_fact(agent, order_fn=lambda n: self._neural_render.order(list(range(n))))
        return self.composer.render_fact(agent)

    # --- dialogue planning (what to say next) ---
    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) built from the agent's OWN stored facts: the
        agent/action/patient of each fact co-occur. Clause patients are skipped (their inner concepts are
        structural). This is the graph the dialogue-planning Control spreads over.

        With enable_learned_assoc, the graph is read from the SUBSTRATE-LEARNED sparse Hebbian recurrent (cheat-D
        resolution) instead of recomputed from the Python kb -- so dialogue planning spreads over a learned
        association memory, not a Python dict."""
        if self._learned_assoc is not None:
            return self._learned_assoc.graph()
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
