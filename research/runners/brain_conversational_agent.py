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

        xp = self._bridge_xp()
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

    def _bridge_xp(self):
        """Return the array module (cupy or numpy) the BRIDGE'S OWN state arrays
        actually use — NOT the process-global ``get_backend()``.

        Why: ``get_backend()`` is a sticky/global cache that a numpy-CPU code
        path elsewhere in a long-running process (e.g. the webapp) can flip to
        numpy AFTER ``sim.bridge`` bound its module-level ``cp`` to cupy. The
        bridge then builds cupy state arrays, but a later ``get_backend()`` here
        returns numpy → a numpy ``cur`` assigned into the cupy
        ``cp_external_input_current[:]`` raises cupy's
        "non-scalar numpy.ndarray cannot be used for fill". Deriving xp from the
        bridge's own array keeps ``cur`` matched to the bridge regardless of the
        global cache state (the live webapp single-fact bug, 2026-06-24)."""
        arr = getattr(self.bridge, "cp_external_input_current", None)
        if arr is not None:
            try:
                import cupy as _cp  # noqa: PLC0415
                return _cp.get_array_module(arr)  # cupy for a device array, numpy otherwise
            except Exception:
                return np  # cupy unavailable → arrays are numpy
        # No state array yet (shouldn't happen post-init): fall back to the global.
        xp, _ = get_backend()
        return xp

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self, n_epochs, train_steps):
        xp = self._bridge_xp()
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
        xp = self._bridge_xp()
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
                 enable_spiking_cleanup=None, enable_substrate_store=False, grounded_codes=None,
                 enable_learned_assoc=None, enable_neural_render=True, enable_rf_cudagraph=False,
                 enable_attributed=True, enable_multiframe=True, integrated_loop=False,
                 enable_multicue_competition=False, multicue_verbs=None,
                 enable_case_competition=False, case_verbs=None, case_lexicon=None,
                 enable_embedded_clause=False, embedded_nouns=None, embedded_verbs=None,
                 embedded_relativizers=None, embedded_readout_redundancy=3,
                 defer_parser=False, communicable_mode=False, communicable_draw="spiking",
                 communicable_config=None, speak_value_Q=None, D=128):
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
        delegates all fact storage/retrieval to the composer; the parser + dialogue-planning are composer-agnostic.

        SPIKING DEFAULT-FLIP (burndown 1A, C-3/C-5): `enable_spiking_cleanup` + `enable_learned_assoc` default to a
        sentinel (None) that resolves to **the production fully-spiking config on the onebrain path** (the brain's own
        spiking ops do the cognition) and **OFF on the rf/rate test-oracle + numpy-CPU path** (byte-identical). So a
        plain `BrainConversationalAgent(composer_kind="onebrain", ...)` is fully-spiking-cleanup + learned-assoc by
        default, == the production demo, == the host oracle on the who/what matrix with the no-confab moat at 0 false-
        accepts (the spiking ops are answer-identical in their validated config). C-4 (local_reciprocal_unbind) is
        ALREADY ON in OneBrainComposer's own default; H-7 (on-bridge complex-synapse store) is on-by-design for the
        OneBrainComposer. C-2 (`integrated_loop`, the spiking K-way cue-match sequencer) STAYS OFF at this library
        default: at the SMALL test vocab / fresh random codes the divnorm-WTA agent-line decode falls below firing
        (over-abstention, the SAFE direction, moat 0-FA -- the de-risk `_burndown_1A_c2_smallvocab_derisk.json`), so
        the host first-match _scan stays the byte-identical oracle here; the PRODUCTION demo (V=320 stream-learned
        codes, where it is GO 4/4) opts it ON explicitly. Pass the flags explicitly (True/False) to override the auto.

        BRAIN-LOAD SPEEDUP (`defer_parser`, default OFF = byte-identical): when True, the comprehension parsers
        (`BridgeParser` + the optional `AttributedBridgeParser`) are NOT built/trained in `__init__` -- they are
        constructed LAZILY on the FIRST runtime `hear()` / `parse()` / `hear_attributed()` (the only places a parser is
        used). A LOADED brain restores its facts via `composer.store()` directly (bypassing the parser entirely --
        `developed_brain_io._restore_facts`), so a pure Q&A session NEVER pays the ~75K-step Hebbian parser training.
        The lazy build trains EXACTLY as the eager one would, so a deferred agent's first teach is identical to a
        never-deferred agent's. DEFAULT-OFF preserves the standalone build path byte-for-byte; `load_developed_brain`
        passes True. On the onebrain path the composer carries its own on-bridge parser (`hasattr(composer, 'hear')`),
        so the agent's separate parser is None regardless and `defer_parser` only affects the rf/rate/external paths.
        See research/findings/2026-06-24-brain-load-speedup-scoping.md (option 2).

        `D` (default **128** = byte-identical) is the composer's phasor dimension when this agent constructs its OWN
        composer (the `composer is None` paths below). It is threaded through to the `OneBrainComposer`/`RFPhasorComposer`
        D so a caller (e.g. the longitudinal develop loop) can raise the recall/abstention margin at 100s of concepts
        (FHRR capacity ~sqrt(D)). When an EXTERNAL `composer` is passed, that composer's own D wins and this is ignored.
        The default 128 reproduces the prior hardcoded literal exactly. See
        research/findings/2026-06-27-develop-knowledge-scaling-arc-scoping.md (§3 option a).
        """
        # resolve the onebrain-aware spiking defaults (None = auto: ON for onebrain production, OFF for rf/rate oracle).
        _is_onebrain = (composer is None) and (composer_kind == "onebrain")
        if enable_spiking_cleanup is None:
            enable_spiking_cleanup = _is_onebrain        # C-3: spiking NEF/WTA cleanup-select (== host argmax)
        if enable_learned_assoc is None:
            enable_learned_assoc = _is_onebrain          # C-5: substrate-learned Hebbian CA3 assoc graph for elaborate
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
            # enable_spiking_cleanup (burndown 1A, C-3): the cleanup SELECTION (the winner-pick over the matched-filter
            # membrane) is a fully-on-substrate spiking Izhikevich WTA instead of a host argmax. DEFAULT-ON on this
            # onebrain production path (the None-sentinel auto-resolves True above) == host argmax + moat 0-FA
            # (test_onebrain_spiking_cleanup); the rf/rate test-oracle + numpy-CPU path keeps it OFF (byte-identical).
            # integrated_loop (shortcut #3, C-2; STAYS default-OFF at this library default = the byte-identical host-
            # _scan oracle): route the (agent, action) cue-match-and-first-match SELECTION through the validated spiking
            # K-way sequencer (gated-disinhibition match cascade + BG first-match priority WTA). At PRODUCTION scale
            # (V=320 stream-learned codes) it is GO 4/4 and the production demo opts it ON; but at the SMALL test vocab /
            # fresh random codes the divnorm-WTA agent-line decode falls below firing (over-abstention, the SAFE
            # direction, moat 0-FA -- the de-risk _burndown_1A_c2_smallvocab_derisk.json: a code-MARGIN boundary, NOT a
            # match_thresh re-cal), so the host read stays the byte-identical oracle here. See the #3 fold plan.
            self.composer = OneBrainComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded_codes,
                                             enable_attributed=enable_attributed,
                                             enable_multiframe=enable_multiframe,
                                             enable_spiking_cleanup=enable_spiking_cleanup,
                                             integrated_loop=integrated_loop)
        elif composer_kind == "slotbinder":
            # the gap-#2 SlotBinderComposer: a fully-spiking competitive-slot binder (each (fact, role) -> its own
            # slot = the win over the FHRR superposition cap) with content-addressable multi-fact recall by a neural
            # scan + the no-confab moat. Replaces the exact-inverse FHRR/VSA algebra with a learned slot->filler
            # associative store (6-seed GO, adversarially verified: no-teach->chance, scramble-teach->0.00). Flat SVO
            # facts (embedded-clause / attributed patients = a named follow-on). See
            # 2026-07-17-gap2-adversarial-verify-CONFIRMED-and-content-addressable-wire-in-GO.md.
            from research.runners.slotbinder_composer import SlotBinderComposer
            vocab = sorted(concepts.keys()) if isinstance(concepts, dict) else None
            self.composer = SlotBinderComposer(seed=seed, D=D, vocab=vocab, grounded_codes=grounded_codes)
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
            self.composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=200,
                                             enable_spiking_cleanup=enable_spiking_cleanup,
                                             enable_substrate_store=enable_substrate_store,
                                             grounded_codes=grounded_codes,
                                             enable_rf_cudagraph=enable_rf_cudagraph)
        # The agent's own comprehension parser -- built ONLY when the composer does not carry its own. The
        # OneBrainComposer carries an on-bridge parser (it has `hear`), so for it there is ONE parser on the one brain
        # and the agent's separate parser is skipped; the rf / rate / external paths build the agent parser as before.
        #
        # BRAIN-LOAD SPEEDUP (defer_parser): a parser is only USED on a runtime teach (hear/parse). With
        # `defer_parser=True` (a LOADED brain) the BridgeParser is NOT built/trained here -- `_ensure_parser()` builds
        # it lazily on the first hear()/parse() (the same trained parser the eager path would have). Default-OFF keeps
        # the standalone path byte-identical (the parser is constructed + trained eagerly, as before). `_composer_has_hear`
        # caches whether the composer carries its own parser (then the agent parser stays None regardless of the flag).
        self._defer_parser = bool(defer_parser)
        self._composer_has_hear = hasattr(self.composer, "hear")
        # counts how many TIMES a parser was actually TRAINED (eager build, or a lazy build) -- 0 on a loaded Q&A-only
        # session, proving the deferred parser never paid its ~75K-step training. (Diagnostic; read by the validators.)
        self._parser_trained_count = 0
        if self._composer_has_hear:
            self.parser = None
        elif self._defer_parser:
            self.parser = None        # built lazily by _ensure_parser() on the first hear()/parse()
        else:
            self.parser = BridgeParser(seed=seed)
            self._parser_trained_count += 1
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
        # BRAIN-LOAD SPEEDUP (defer_parser): the AttributedBridgeParser (~50K-step Hebbian training) is ALSO only used
        # on a runtime hear_attributed(). With `defer_parser=True` it is built lazily by `_ensure_attr_parser()` on the
        # first hear_attributed(); default-OFF it is constructed + trained eagerly here whenever enable_attributed
        # (byte-identical to before -- the original built it for the rf/rate AND onebrain paths alike, so a default
        # agent's `_attr_parser is not None` test is preserved).
        if enable_attributed and not self._defer_parser:
            from research.runners.attributed_parser import AttributedBridgeParser
            self._attr_parser = AttributedBridgeParser(seed=seed)
            self._parser_trained_count += 1
        # (richer-syntax #2, opt-in) multi-frame comprehension: a neural FrameParser (verb-position -> frame selection +
        # position x frame -> role) comprehends a sentence in an AUTO-SELECTED word-order frame (SVO/VSO/OSV).
        # `hear_multiframe(sentence, verbs)` routes through it; default OFF = byte-identical (the native BridgeParser /
        # onebrain SVO path is unchanged). Validated GO 6/6 (2026-06-18-frame-selection-GO.md). Built lazily.
        self.enable_multiframe = bool(enable_multiframe)
        self._frame_parser = None
        # (richer-syntax #3, opt-in) EMBEDDED-CLAUSE parsing: a two-pass parser SEGMENTS a depth-1 embedded relative
        # clause from a FLAT token stream ('dog that chase cat run') + role-assigns BOTH the embedded clause AND the
        # matrix clause with the SAME neural conjunctive position-code read-out, holding the suspended matrix head in
        # the spiking WM latch. `hear_nested(flat_sentence)` parses + stores the matrix fact with the parsed embedded
        # Clause as its patient -- replacing the host-constructed Clause that `hear_clause_fact` required. Default OFF
        # = byte-identical (the parser is never constructed; hear_nested asserts-off). Validated GO 6/6 @ 1.000 with
        # the population-redundancy read-out (2026-06-19-embedded-clause-{parse,redundancy}-derisk.md). Reuse-by-import
        # (EmbeddedClauseParser + RedundantEmbeddedReadout from _phaseB_embedded_clause_parse_derisk); NO sim/ edit.
        # The closed-class lexicon (which token is a relativizer/verb/noun) is the legitimate environment/lexicon
        # front end (same as FrameParser's known-verb set); default = the agent's own vocab (verbs auto-default to the
        # NOUNS/VERBS probe sets when not supplied). embedded_readout_redundancy>1 wraps the embedded read-out in R
        # majority-voting phasor replicas (the validated lever lifting the 0.88 marginal seeds to 1.000); the matrix
        # parse + the moat are unchanged. Built lazily/cached.
        self.enable_embedded_clause = bool(enable_embedded_clause)
        self._embedded_nouns = list(embedded_nouns) if embedded_nouns else None
        self._embedded_verbs = set(embedded_verbs) if embedded_verbs else None
        self._embedded_relativizers = set(embedded_relativizers) if embedded_relativizers else None
        self._embedded_readout_redundancy = max(1, int(embedded_readout_redundancy))
        self._embedded_parser = None
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
        # (cross-language wire-in, Phase-2, opt-in) case-marking role COMPETITION: route hear()'s AGENT/PATIENT
        # decision through the validated CASE-aware spiking multi-cue competition (CaseAwareRoleParser -- de-risk
        # 2026-06-19-case-cue-crosslanguage-derisk.md, GO), so a FREE-word-order CASE-MARKED sentence (Japanese-
        # style ga/wo) reads thematic roles by the case PARTICLE where the position-only BridgeParser (and even the
        # Phase-1 multicue parser on a case-free toy) cannot. EXTENDS the multicue path with a fifth `case` cue;
        # its install-path validity is HIGH in a case language (case dominant, position low). Default OFF =
        # byte-identical (the parser is never constructed). Requires `case_verbs` (the known-verb set the lexical
        # front-end uses to find the verb); `case_lexicon` (optional) overrides the default isolating-particle map
        # (ga->nom, wo->acc). The no-confab moat is preserved end-to-end (an UNMARKED ambiguous sentence -- case
        # silent + animacy ties + symmetric verb -- is reported non-decisive by parse_decisive so the caller
        # ABSTAINS). WIRED: the validated CASE-aware spiking competition INFERENCE (install-path case-language
        # validities). DEFERRED: continual on-substrate cue-validity LEARNING (the cross-linguistic dissociation,
        # seed-variable -- Tier 1 item 2) + neuralizing the learner's reward; fused/portmanteau case (Phase 3).
        # Built lazily/cached. enable_multicue_competition takes precedence if BOTH are set (they are alternative
        # comprehension front-ends; case is the case-language one).
        self.enable_case_competition = bool(enable_case_competition)
        self._case_verbs = set(case_verbs) if case_verbs else None
        self._case_lexicon = dict(case_lexicon) if case_lexicon else None
        self._case_parser = None
        if enable_case_competition and self._case_verbs is None:
            raise ValueError("enable_case_competition=True needs case_verbs=<known-verb set> "
                             "(the lexical front-end that finds the sentence's verb)")
        # (COMMUNICABLE BRAIN, Stage B wire-in, opt-in, default OFF = BYTE-IDENTICAL) -- route a conversational
        # TURN through the validated `CommunicableTurn` (the fused GENERATE / DECIDE-to-speak / LEARN-talkativeness
        # + known-fact + phatic orchestrator, Stage A GO 3-seed). When ON, `converse(msg, ...)` classifies intent
        # and routes to the known-fact (hard-gated, the no-confab moat) / novel-generative (a FLAGGED hypothesis,
        # never stored) / phatic / teaching channel; `communicable_feedback(topic, polarity)` runs the brain's
        # three-factor talkativeness update. Default OFF = the orchestrator is NEVER constructed (no corpus pass,
        # no proposer/accumulator build, no behaviour change to the existing what_does/who_does/is_it_true/hear
        # path -- the production suites pass verbatim). The orchestrator REUSES this agent's OWN composer (so its
        # known-fact channel reads the agent's facts + the same moat) and is built LAZILY on the first converse()
        # so even `communicable_mode=True` pays the ~corpus+brain build only when a turn actually runs.
        #   communicable_draw (owner-steer #3): 'spiking' (DEFAULT, the production generative draw -- each filler a
        #     spiking soft-WTA sample; ~40s/topic on CPU in the fused turn, the megakernel perf lever is Stage C) or
        #     'host' (the fast-interactive / numpy-CPU / test oracle -- the SAME PPMI likelihood, fast). The
        #     load-bearing SPIKING speak DECISION stays spiking regardless.
        #   speak_value_Q (optional): a {topic: float} dict to SEED the learned talkativeness Q (e.g. restored from a
        #     developed-brain bundle, so the talkativeness learned across sessions carries forward).
        #   communicable_config (optional): a dict of extra build_communicable_brain kwargs (D, weights, ...).
        self.communicable_mode = bool(communicable_mode)
        self._communicable_draw = str(communicable_draw)
        self._communicable_config = dict(communicable_config) if communicable_config else {}
        self._communicable_seed_Q = dict(speak_value_Q) if speak_value_Q else None
        self._communicable = None        # the built CommunicableTurn (lazy); None until first converse()
        self._communicable_brain = None  # the build_communicable_brain() result dict (the value object lives here)

    def _ensure_parser(self):
        """Lazily build + train the comprehension `BridgeParser` (BRAIN-LOAD SPEEDUP: deferred so a LOADED Q&A-only
        brain never pays the ~75K-step training). Returns the trained parser. A no-op (returns the existing parser)
        when one is already built -- so the FIRST hear()/parse() pays the one-time training, identical to a
        never-deferred agent. On the onebrain path the composer carries the parser, so this is never reached (hear()
        delegates to the composer); callers that need a parser without a composer hear() use this."""
        if self.parser is None and not self._composer_has_hear:
            self.parser = BridgeParser(seed=self.seed)   # trains in __init__ (defer_train default False) == the eager build
            self._parser_trained_count += 1
        return self.parser

    def _ensure_embedded_parser(self):
        """Lazily build + cache the `EmbeddedClauseParser` (the two-pass depth-1 relative-clause parser; its neural
        sub-parsers train once on first use). The lexicon defaults to the de-risk's validated NOUNS/VERBS probe sets
        unless the caller supplied embedded_nouns/verbs/relativizers (the environment/lexicon front end). When
        embedded_readout_redundancy>1, a `RedundantEmbeddedReadout` (R majority-voting phasor replicas) is also built
        so the redundant embedded decode is available via query_nested -- the validated lever that lifts the marginal
        seeds to 1.000; what_does reads the agent's single persistent composer (the canonical store)."""
        if self._embedded_parser is None:
            from research.runners._phaseB_embedded_clause_parse_derisk import (
                EmbeddedClauseParser, RedundantEmbeddedReadout)
            self._embedded_parser = EmbeddedClauseParser(
                seed=self.seed, nouns=self._embedded_nouns, verbs=self._embedded_verbs,
                relativizers=self._embedded_relativizers)
            self._embedded_redundant = None
            if self._embedded_readout_redundancy > 1:
                # the redundant embedded read-out: R independent phasor codebooks over the composer's vocab, voted
                # per slot (the population-redundancy robustness lever; reuse-by-import). query_nested uses it.
                vocab = getattr(self.composer, "words", None)
                D = getattr(self.composer, "D", 128)
                self._embedded_redundant = RedundantEmbeddedReadout(
                    seed=self.seed, D=D, vocab=list(vocab) if vocab is not None else None,
                    n_replicas=self._embedded_readout_redundancy)
        return self._embedded_parser

    def hear_nested(self, flat_sentence, voice="active", polarity=None):
        """Comprehend a FLAT token stream that may contain a depth-1 embedded relative clause ('dog that chase cat
        run') -- the two-pass parser SEGMENTS the embedded clause + role-assigns BOTH clauses NEURALLY (the spiking
        conjunctive position-code read-out + the spiking WM-latch hold of the suspended matrix head), then stores the
        matrix fact with the PARSED embedded `Clause` as its patient (replacing the host-constructed `Clause` that
        `hear_clause_fact` required). A non-nested SVO is stored as a plain flat fact (nested=False). Returns the
        parse dict ({'matrix','embedded','nested'}) or None on an unparseable / garbled stream (the no-confab moat:
        store nothing, return None). Requires enable_embedded_clause=True. After a nested store,
        what_does(matrix_agent, matrix_action) decodes the embedded clause. (Richer-syntax #3.)"""
        assert self.enable_embedded_clause, \
            "hear_nested needs BrainConversationalAgent(enable_embedded_clause=True)"
        parser = self._ensure_embedded_parser()
        parsed = parser.parse_nested(flat_sentence)
        if parsed is None:
            return None                                       # unparseable / garbled -> abstain (moat), store nothing
        m_agent, m_action, m_patient = parsed["matrix"]
        if parsed["nested"]:
            emb = parsed["embedded"]                          # the parsed embedded Clause (was host-constructed)
            self.composer.store(m_agent, m_action, emb, polarity=polarity)
            if self._embedded_redundant is not None:          # mirror the nested fact into the redundant read-out
                self._embedded_redundant.store(m_agent, m_action, emb, polarity=polarity)
        else:
            self.composer.store(m_agent, m_action, m_patient, polarity=polarity)
            if self._learned_assoc is not None:
                self._learned_assoc.store_fact([m_agent, m_action, m_patient])
        return parsed

    def query_nested(self, agent, action):
        """Decode an embedded-clause patient via the population-redundancy read-out (R majority-voting phasor
        replicas) -- the validated lever that lifts the marginal-seed embedded decode to 1.000. Falls back to the
        agent's single composer when redundancy is off (R=1). The no-confab moat is preserved (all replicas / the
        composer abstain -> None). Requires enable_embedded_clause=True + a prior hear_nested store."""
        assert self.enable_embedded_clause, \
            "query_nested needs BrainConversationalAgent(enable_embedded_clause=True)"
        if getattr(self, "_embedded_redundant", None) is not None:
            return self._embedded_redundant.query_patient(agent, action)
        return self.composer.query_patient(agent, action)

    def _ensure_attr_parser(self):
        """Lazily build + train the `AttributedBridgeParser` (BRAIN-LOAD SPEEDUP: deferred so a loaded Q&A-only brain
        never pays its ~50K-step training). The first hear_attributed() pays the one-time training; identical to the
        eager build."""
        if self._attr_parser is None:
            from research.runners.attributed_parser import AttributedBridgeParser
            self._attr_parser = AttributedBridgeParser(seed=self.seed)
            self._parser_trained_count += 1
        return self._attr_parser

    def _ensure_case_parser(self):
        """Lazily build + cache the CASE-aware spiking CaseAwareRoleParser (one bridge build, install-path
        case-language validities + the isolating-particle case lexicon)."""
        if self._case_parser is None:
            from research.runners.case_aware_role_parser import CaseAwareRoleParser
            self._case_parser = CaseAwareRoleParser(known_verbs=self._case_verbs,
                                                    case_lexicon=self._case_lexicon, seed=self.seed)
        return self._case_parser

    def hear_case(self, sentence, voice="active", polarity=None, markers=None):
        """Comprehend a (possibly FREE-word-order) CASE-MARKED transitive sentence with the CASE-aware spiking
        multi-cue role-competition and store the resolved fact, so an object-fronted 'wolf wo dog ga chase'
        assigns the SAME agent (dog) / patient (wolf) as canonical 'dog ga wolf wo chase' -- the case PARTICLE
        overrides word position. The verb is identified lexically from `case_verbs`; each noun's case particle is
        pulled from the surface tokens (or supplied via `markers`). Returns the parsed {role: word}. Requires
        enable_case_competition=True. The no-confab moat is unaffected (composer Q&A abstains on any unstored
        fact)."""
        assert self.enable_case_competition, \
            "hear_case needs BrainConversationalAgent(enable_case_competition=True, case_verbs=...)"
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        # GAP-1 (the comprehension no-confab moat): route through parse_decisive -- an UNMARKED content-ambiguous
        # sentence (no case particle + animacy tie + symmetric verb) is NON-decisive -> ABSTAIN (store nothing,
        # return None) rather than confabulate. Same gate as hear_multicue (CaseAwareRoleParser.parse_decisive, the
        # case content gate); the multicue path is de-risked, the case path follows the identical pattern.
        roles, decisive = self._ensure_case_parser().parse_decisive(words, voice, markers=markers)
        if not decisive:
            return None
        self.composer.store(roles.get("agent"), roles.get("action"), roles.get("patient"), polarity=polarity)
        if self._learned_assoc is not None:
            self._learned_assoc.store_fact([roles.get("agent"), roles.get("action"), roles.get("patient")])
        return roles

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
        # GAP-1 (the comprehension no-confab moat): route through parse_decisive -- a content-ambiguous degraded
        # sentence (two animate nouns + a symmetric verb -> no decisive content cue) is NON-decisive, so ABSTAIN at
        # comprehension (store NOTHING, return None) rather than confabulate a role assignment the query-time moat
        # could not un-store. De-risk GO (_phaseB_multicue_comprehension_moat_derisk.py): ambiguous abstain 1.00 /
        # 0 confab, the margin-lesion reproduces the confab, decisive + canonical unregressed.
        roles, decisive = self._ensure_multicue_parser().parse_decisive(words, voice)
        if not decisive:
            return None
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
        production turn entry point comprehends scrambled / object-fronted input correctly. When
        enable_case_competition is ON (and multicue is OFF), hear() routes through the CASE-aware competition so a
        free-word-order CASE-MARKED sentence (ga/wo) reads roles by the case particle. Default OFF (both) =
        byte-identical (neither parser is built; both branches are skipped)."""
        if self.enable_multicue_competition:
            return self.hear_multicue(sentence, voice, polarity=polarity)
        if self.enable_case_competition:
            return self.hear_case(sentence, voice, polarity=polarity)
        if hasattr(self.composer, "hear"):
            roles = self.composer.hear(sentence, voice, polarity=polarity)
        else:
            roles = self._ensure_parser().parse(sentence.split(), voice)   # builds+trains the parser lazily if deferred
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
        assert self.enable_attributed, "hear_attributed needs BrainConversationalAgent(enable_attributed=True)"
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = self._ensure_attr_parser().parse(words, voice)   # builds+trains the attributed parser lazily if deferred
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
        composer is wired -- `self.parser` is None on the onebrain path. (BRAIN-LOAD SPEEDUP: when the agent's own
        parser was deferred, `_ensure_parser()` builds+trains it lazily on first use here.)"""
        parser = None if self._composer_has_hear else self._ensure_parser()
        if parser is None:
            parser = getattr(self.composer, "parser", None)
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

    def chain_of_thought(self, start, goal=None, max_hops=4, return_path=False):
        """SELF-CUED associative chain-of-thought (Tier 2.2 -- the structural heart of 'thinking'): from `start`,
        the agent itself SELECTS each next relation to chase by LEARNED association strength over its own stored
        facts (NOT a caller-supplied action list, unlike reason_chain), then chases it via the validated single
        hop, re-cleaning between hops so error does not compound. Stops at `goal` (if reached) or a dead end ->
        ABSTAIN (the no-confab moat holds at EVERY hop; an unstored start or a dead end returns None / no fabricated
        hop). Delegates to the composer's chain_of_thought -- de-risked GO numpy 3 seeds x 3 D (self-cued 2-hop
        1.00 vs spreading floor 0.08; lesion-the-association/permuted/re-cue all collapse; no compounding to 4
        hops): 2026-06-27-tier2.2-chain-of-thought-GO.md. With return_path=True returns (terminal, [start, ...])."""
        if not hasattr(self.composer, "chain_of_thought"):
            raise NotImplementedError("the active composer does not support self-cued chain_of_thought "
                                      "(needs RFPhasorComposer / OneBrainComposer)")
        return self.composer.chain_of_thought(start, goal=goal, max_hops=max_hops, return_path=return_path)

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

    # --- COMMUNICABLE BRAIN (Stage B wire-in, opt-in) -----------------------------------------------------------
    def _ensure_communicable(self):
        """Lazily build the `CommunicableTurn` orchestrator (the Stage A fusion) over THIS agent's composer (so the
        known-fact channel reads the agent's OWN facts + the same no-confab moat). Built on the first converse() /
        feedback() / speak_value_Q() so even communicable_mode=True pays the corpus+brain build only when used.
        Returns the CommunicableTurn. Raises if communicable_mode is OFF (the orchestrator is never constructed)."""
        if not self.communicable_mode:
            raise RuntimeError("communicable_mode is OFF -- construct the agent with communicable_mode=True "
                               "(or call enable_communicable_mode()) to route a turn through the CommunicableTurn")
        if self._communicable is None:
            from research.runners._communicable_turn_stageA_derisk import build_communicable_brain
            cfg = dict(self._communicable_config)
            cfg.setdefault("host_oracle_sampler", self._communicable_draw == "host")
            brain = build_communicable_brain(seed=self.seed, composer=self.composer, bc_agent=self,
                                             speak_value_Q=self._communicable_seed_Q, **cfg)
            self._communicable_brain = brain
            self._communicable = brain["turn"]
        return self._communicable

    def enable_communicable_mode(self, *, draw=None, speak_value_Q=None, **config):
        """Turn communicable-mode ON at runtime (mirrors the constructor flag). `draw` ('spiking'|'host') selects
        the generative draw; `speak_value_Q` seeds the learned talkativeness; extra kwargs pass to
        build_communicable_brain. The orchestrator is still built lazily on the first converse()."""
        self.communicable_mode = True
        if draw is not None:
            self._communicable_draw = str(draw)
        if speak_value_Q is not None:
            self._communicable_seed_Q = dict(speak_value_Q)
        if config:
            self._communicable_config.update(config)
        self._communicable = None          # force a rebuild with the new config on next use
        self._communicable_brain = None
        return self

    def converse(self, msg, *, cue=None, topic=None, n_attempts=500):
        """Route ONE user message through the fused communicable turn (intent -> known-fact / novel-generative /
        phatic / teaching channel). Returns the CommunicableTurn's structured channel record. Requires
        communicable_mode=True. The no-confab moat is preserved (the known-fact channel hard-gates + abstains; the
        novel channel emits only FLAGGED hypotheses and NEVER stores). `cue` = an (agent, action[, patient]) tuple
        for a structured known-fact question; `topic` = the opinion topic. (The agent's existing what_does/who_does/
        is_it_true/hear API is unchanged and remains the structured entry point.)"""
        return self._ensure_communicable().turn(msg, cue=cue, topic=topic, n_attempts=n_attempts)

    def communicable_feedback(self, topic, polarity, *, lesion_DA=False, decorrelate=False):
        """Deliver a perceived conversational feedback on `topic` (+1 'elaborate' -> a DA burst raises the learned
        talkativeness there + at PPMI-similar contexts; -1 'stop' -> a DA dip lowers it). The brain's three-factor
        plasticity, NOT a host counter (the DA-lesion abolishes it). Requires communicable_mode=True."""
        self._ensure_communicable().feedback(topic, polarity, lesion_DA=lesion_DA, decorrelate=decorrelate)

    def speak_value_Q(self):
        """The LEARNED per-topic talkativeness Q ({topic: float}) -- the persistable talkativeness state (saved into
        a developed-brain bundle so it carries across sessions). Returns {} if communicable-mode was never built."""
        if self._communicable_brain is None:
            # never built -> return the seed Q (if any) so a save before any turn still round-trips a restored Q.
            return dict(self._communicable_seed_Q) if self._communicable_seed_Q else {}
        return {t: float(q) for t, q in self._communicable_brain["value"].Q.items()}
