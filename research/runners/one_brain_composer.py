"""The production `OneBrainComposer` (roadmap phase 2, the real "one brain"): the whole who/what conversational
pipeline on ONE persistent co-resident `SimulationBridge`, with no host round-trips between operations. An
`RFPhasorComposer` API-sibling the conversational agent can use via `composer_kind="onebrain"`.

Assembled from the validated GO pieces of the Phase-2 arc (each de-risked multi-seed this session):
  - the PARSER front-end (GAP B, `2026-06-18-one-brain-parser-frontend-GO.md`): a `BridgeParser` on slice [0:P]
    comprehends a sentence; the role it FIRES for each word selects that word's bind (no host {role:word} dict;
    voice-invariant).
  - the persistent multi-fact STORE (GAP A, `2026-06-18-one-brain-multifact-store-GAP-A-GO.md`): each fact = a 3-role
    composite written into a (1+D) trigger->readout block in the bridge's complex weights (register-reset-safe; GO to
    K=32).
  - the CUE-matching SCAN + on-bridge cleanup + the no-confab moat (`2026-06-18-one-brain-composer-A3-GO.md`): a
    who/what question reconstructs each stored block, unbinds all three roles IN PARALLEL (one reconstruction, no phase
    drift), cleans up, and the first block whose cue roles match answers; an absent cue / unstored fact abstains.

The parser (Izhikevich, voltage in v/u) and the resonate-and-fire composer registers (a complex phasor in v/u)
co-reside as disjoint slices on ONE bridge (the merged-bridge regime), the resonate-and-fire ops masked to their slice.

SCOPE (the A5 cleanup arc brings the rf composer's features to parity here so onebrain can be the documented default
and the legacy numpy production runtime can retire, numpy kept as the test oracle): who / what / affirmative & negated
yes-no (a bound polarity tag = a 4th role) / generation (`render_fact`) / multi-hop (`query_chain`) / recursive
embedded CLAUSES (a fact whose patient is an SVO clause -> a 2-level unbind). Bounded follow-ons still on the numpy
oracle only: reconsolidation (`update_on_mismatch`), multi-turn anaphora, attributed entities (adj+noun).

NO sim/ edit (reuse-by-import: BridgeParser + RFPhasorComposer + the masked rf_kick). GPU for real use (the parser
trains on the bridge); numpy is the test oracle.
"""
from __future__ import annotations

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import to_host, get_backend, get_sparse_module
from research.runners.brain_conversational_agent import BridgeParser
from research.runners.rf_phasor_composer import RFPhasorComposer, _is_clause


def _seq_imports():
    """Lazy import of the validated spiking K-way sequencer fabric (shortcut #3). Deferred so an integrated_loop=OFF
    composer (the byte-identical default + the numpy-CPU + test-oracle path) never imports the sequencer de-risk
    runners. Reuse-by-import (NO sim/ edit): the K-way sequencer builder + run + decode (S0), the divnorm score bridge
    + per-block decoded-line drive (S2/S5), all already-shipped. Returns the functions _ensure_sequencer/_seq_block use."""
    from research.runners._phaseB_onebrain_sequencerK_derisk import (
        build_sequencerK_bridge, decision_to_block)
    from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores
    from research.runners._phaseC_S5_divnorm_derisk import build_divnorm_score_bridge
    from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import make_block_drives
    from research.runners._phaseB_onebrain_sequencerK_divnorm_derisk import run_sequencerK_with_drive
    return dict(build_sequencerK_bridge=build_sequencerK_bridge, decision_to_block=decision_to_block,
                block_cleanup_scores=block_cleanup_scores, build_divnorm_score_bridge=build_divnorm_score_bridge,
                make_block_drives=make_block_drives, run_sequencerK_with_drive=run_sequencerK_with_drive)

ROLES3 = ["agent", "action", "patient"]


def _build_complex_csr(n_total, connections):
    """Build the (cp_rf_w_re, cp_rf_w_im) device CSR pair from a `(post, pre, complex_w)` connection list -- the SAME
    construction `SimulationBridge.rf_set_complex_weights` performs (np.fromiter -> backend sparse csr_matrix), pulled
    out so the OneBrainComposer can build a QUERY-INVARIANT operator ONCE and reuse the device handles across queries
    instead of rebuilding from a fresh tuple list every read (the measured 72%-of-a-query weight-rebuild cost; the
    latency-arc scoping). Backend-agnostic (cupy on GPU, scipy on numpy) so the A/B + test parity holds on both paths.
    Returns (W_re, W_im) ready to assign to b.cp_rf_w_re / b.cp_rf_w_im."""
    xp, _name = get_backend()
    csp = get_sparse_module()
    m = len(connections)
    rows = np.fromiter((int(post) for (post, pre, w) in connections), dtype=np.int32, count=m)
    cols = np.fromiter((int(pre) for (post, pre, w) in connections), dtype=np.int32, count=m)
    w_re = np.fromiter((float(complex(w).real) for (post, pre, w) in connections), dtype=np.float64, count=m)
    w_im = np.fromiter((float(complex(w).imag) for (post, pre, w) in connections), dtype=np.float64, count=m)
    r = xp.asarray(rows); c = xp.asarray(cols)
    W_re = csp.csr_matrix((xp.asarray(w_re), (r, c)), shape=(n_total, n_total))
    W_im = csp.csr_matrix((xp.asarray(w_im), (r, c)), shape=(n_total, n_total))
    return W_re, W_im


def build_coresident_bridge(seed, n_total, enable_rf_cudagraph=False):
    """An Izhikevich bridge (Hebbian ON for the parser); the RF region has no cp_connections wiring (its memory is in
    cp_rf_w_re/im), so global Hebbian has nothing to touch there. `enable_rf_cudagraph` (A5 lever 3): route the RF
    resonate through the masked megakernel (one CUDA launch/step instead of ~15-20) -- the resonate is ~83% of a query
    (the profile), so this closes the residual gap vs the rf reference. Default off = the loop (byte-identical)."""
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.enable_rf_cudagraph = bool(enable_rf_cudagraph)
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


class OneBrainComposer:
    """The who/what pipeline on ONE persistent co-resident bridge. Parser [0:P]; RF region from P: fill_0..2,
    bound_0..2, acc (7 blocks), the persistent store (k_max (1+D) blocks), 3 parallel-read Q registers, 3 V-cleanup
    blocks. API mirrors `RFPhasorComposer` for the conversational agent (`store`/`hear`/`query_patient`/`query_agent`/
    `ask_yes_no`; `kb` bookkeeping)."""

    def __init__(self, seed=42, D=128, vocab=None, k_max=32, period=200, enable_batched=True,
                 enable_rf_cudagraph=True, grounded_codes=None, confidence_gate=0.0, enable_csr_cache=True,
                 enable_attributed=False, enable_multiframe=False, enable_spiking_cleanup=False,
                 encoding_gain_fn=None, local_reciprocal_unbind=False, integrated_loop=False,
                 sequencer_match_thresh=0.06, sequencer_gain=0.11, sequencer_sigma=1.0, sequencer_input_gain=1.0):
        self.seed = int(seed); self.D = int(D); self.period = int(period)
        # integrated_loop (shortcut #3, default OFF = byte-identical = the host-_scan oracle + numpy-CPU + test-oracle
        # path): make the CUE-MATCH ROUTING fully on-substrate. The per-block reconstruction (_read_blocks) is ALREADY
        # spiking; the residual host op is the Python first-match loop that picks WHICH stored block answers a who/what
        # query (and answer vs abstain). When ON, that loop is replaced by the validated K-way sequencer (gated-
        # disinhibition match cascade + BG first-match priority WTA): the cue + each block's cleanup scores drive a
        # spiking control fabric whose winning channel IS the selected block (the legitimate body read), ==
        # host_scan_block multi-seed at match_thresh 0.06 (2026-06-21-shortcut3-K32-capability-surpass.md). The no-confab
        # moat is preserved by construction: the abstain channel maps to the same None/"unknown" the host returned, 0
        # false-accept on absent/cross cues (an absent cue WORD is caught before the sequencer). BUILD-1 SCOPE: the
        # (agent, action) hot-path sites (_scan / query_patient / ask_yes_no / _find_cued_block) route through spikes;
        # the (action, patient) `query_agent` + agent-only `render_fact`/`describe` stay on the host read (still
        # abstaining via the oracle) as named bounded follow-ons (a swapped-cue + a 1-role cascade). See the plan.
        self.integrated_loop = bool(integrated_loop)
        self.sequencer_match_thresh = float(sequencer_match_thresh)
        self.sequencer_gain = float(sequencer_gain)
        self.sequencer_sigma = float(sequencer_sigma)
        self.sequencer_input_gain = float(sequencer_input_gain)
        self._seq = None            # (sb, meta) -- the sequencer control bridge, built lazily on first query
        self._seq_score = None      # the divnorm score bridge
        self._seq_K = None          # the store size the current sequencer/drives were built for
        self._seq_drives = None     # the per-block decoded-line drives (recomputed when the store changes)
        self._seq_dirty = True      # the store changed since the drives were built -> rebuild the drives
        # local_reciprocal_unbind (FHRR-B mechanism 1, opt-in, DEFAULT-OFF = byte-identical): derive the UNBIND
        # synapse weights from the BIND (role) phasor by the one-time LOCAL reciprocal-conjugate rule (a per-component
        # quadrature flip via comp._local_conj) instead of the host np.conj over the role code. Closes the same host
        # residual on the PRODUCTION-default one-brain path that the rf composer's flag closes on its `_unbind_phases`:
        # conj(role) becomes a local wiring rule the construction step applies, so the bind structure is host-free at
        # runtime (the neuromorphic-port property). Applies to the 6 UNBIND-structure sites (comp.roles[...]); the
        # cleanup-codebook conj (comp.concepts[...]) is a SEPARATE residual (reducible-to-learned) left untouched.
        # Byte-identical (the local conj == host conj bit-for-bit for a unit phasor). See
        # research/findings/2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md.
        self.local_reciprocal_unbind = bool(local_reciprocal_unbind)
        # encoding_gain_fn (Tier-2 #6, opt-in, DEFAULT-OFF = byte-identical): the one-brain mirror of the RF composer's
        # DOPAMINE-GATED ENCODING STRENGTH (Lisman-Grace hippocampal-VTA loop; Kandel D.16 -- dopamine gates the entry of
        # information into LONG-TERM memory, making a trace STABLE vs degradable). An optional callable () -> float read
        # AT STORE TIME (the shared `dopamine` concentration in deployment; a probe value in the de-risk). When set, the
        # fact's composite phasor written into the persistent trigger->readout store block (_write_block) is multiplied
        # by the per-fact gain `g`. The RF phase read-out has a hard MAGNITUDE FLOOR (sim/bridge.py:5589 `_rf_mag2 >
        # _rf_floor2` -- a readout neuron whose |Z| decays below the floor never spikes -> reads phase 0 = garbage), so a
        # higher-gain (rewarded) fact reconstructs ABOVE the floor under common read damage where a unit-gain (neutral)
        # fact degrades BELOW it -> the rewarded fact wins the cue-match scan. NOT a vacuous global gain: the floor is the
        # nonlinearity that makes it differential. None -> g=1.0 for every fact -> the byte-identical unit-magnitude write
        # (exactly RFPhasorComposer._store_substrate's semantics). The no-confab moat is preserved by construction: the
        # gain only scales the stored magnitude; the cue-match abstention + the cleanup winner-pick are unchanged.
        self.encoding_gain_fn = encoding_gain_fn
        # enable_spiking_cleanup (burndown #1, default OFF = byte-identical = the numpy-CPU + test-oracle path): make
        # the cleanup SELECTION fully on-substrate. The matched FILTER is ALREADY on the co-resident bridge (the
        # complex-synapse `clean` matvec -> the rectified membrane `scores`); the residual host op was the WINNER-PICK
        # (`self.words[int(np.argmax(scores))]`). When ON, `_select` routes each role's scores through a spiking
        # Izhikevich WTA (input-normalized drive -> firing -> argmax-over-FIRING = a readout of the spiking competition,
        # NOT a host argmax over the membrane) -- the SAME validated NEF-cleanup Stage-2 as RFPhasorComposer._spiking_
        # cleanup (Stewart-Tang-Eliasmith; == numpy argmax multi-seed @ D=2048, 2026-06-05-composer-cleanup-NEF-GO.md).
        # The no-confab moat is preserved by construction: the confidence_gate margin + the cue-match abstention read
        # the SAME `scores`, and the WTA picks the same winner the argmax did, so every abstention is unchanged.
        self.enable_spiking_cleanup = bool(enable_spiking_cleanup)
        # enable_multiframe (richer-syntax #2, default OFF = byte-identical): build a FrameParser (verb-position ->
        # frame selection + position x frame -> role, both neural) so `hear_multiframe(sentence, verbs)` comprehends a
        # sentence in an AUTO-SELECTED word-order frame (SVO/VSO/OSV). The default `hear` (the on-bridge SVO/passive
        # BridgeParser) is untouched. Lazily built on first use to keep construction byte-identical when unused.
        self.enable_multiframe = bool(enable_multiframe)
        self._frame_parser = None
        self.enable_batched = bool(enable_batched)       # A5 lever 1: read ALL blocks in 3 windows (7.3x); per-block=oracle
        self.enable_rf_cudagraph = bool(enable_rf_cudagraph)   # A5 lever 3: masked megakernel for the resonate (GPU only)
        # enable_csr_cache (default ON, A5 lever 4 / the latency-arc top increment): cache the QUERY-INVARIANT unbind +
        # cleanup complex-weight CSRs (keyed by n_facts + the fixed block layout) and the store CSR (keyed by a store-
        # dirty flag), so the batched read reuses the device matrices instead of rebuilding ~100k-240k tuples + two
        # fresh csr_matrix constructions + H2D EVERY query (the measured ~72%-of-a-query cost). ANSWER-IDENTICAL (the
        # reused CSR VALUES are the same; only WHEN they're built changes -- the matvec/dynamics are byte-unchanged).
        # Invalidated on exactly the layout-changing ops: a `store` grows n_facts (new unbind/clean cache key) and a
        # `store`/reconsolidation rewrites store_conns (store CSR dirty). Toggle off for the A/B + numpy parity.
        self.enable_csr_cache = bool(enable_csr_cache)
        self._csr_cache = {}          # n_facts -> ((Ure,Uim), (Cre,Cim)) for the batched unbind + cleanup operators
        self._store_csr = None        # (Sre, Sim) for store_conns; rebuilt only when _store_dirty
        self._store_dirty = True      # store_conns changed since the last build (a write happened) -> rebuild the CSR
        # confidence_gate (default 0.0 = OFF = byte-identical): a familiarity/confidence gate on the cue read-out. The
        # cleanup is a matched filter; a CONFIDENT block's winner dominates (a large normalized margin), a noise-
        # dominated (heavily-damaged) block's cleanup is flat (a small margin). When > 0, a block whose CUE-role
        # (agent/action) margin falls below the gate is BLANKED in the read path, so every consumer naturally ABSTAINS
        # on it -- converting the extreme-damage confabulation/moat-leak tail (the cue-match abstention's boundary,
        # 2026-06-18-emergent-graceful-degradation-derisk.md) into abstention = a CALIBRATED moat, no broad refactor.
        self.confidence_gate = float(confidence_gate)
        # grounded_codes (optional word->phases): the learned-from-conversation concept codes (e.g. the 320 stream-learned
        # cortex). Passed to the inner RFPhasorComposer, which overrides its random codes for those words -> the cleanup
        # codebook + the binding both use the learned codes (production parity with the rf composer's grounded path).
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=period, grounded_codes=grounded_codes,
                                     local_reciprocal_unbind=local_reciprocal_unbind)
        self.words = list(self.comp.words)              # the cleanup codebook = the composer's ACTUAL vocab
        self.V = len(self.words)
        self.R = 40; self.P = 6 + 3 * self.R; self.k_max = int(k_max)
        self.pol_words = list(self.comp.pol_words)                   # ["AFFIRM","NEGATE"] -- cleaned up SEPARATELY from
        self.NP = len(self.pol_words)                                # the main vocab (a 2-word polarity codebook)
        # The fillable roles ALWAYS bound. Default = 4 (agent, action, patient, polarity[AFFIRM]) -> yes/no/negation;
        # the 4-role coherence is GO, so the 4th bind is within the substrate's per-fact capacity. With
        # `enable_attributed` (richer-syntax #1, default OFF = byte-identical), a 5th ATTRIBUTE role is bound so a
        # single-attribute entity ("big apple") stores + recalls -- the 2-factor (one bind / one unbind) path, which
        # HOLDS 100% on the production LEARNED 320 codes (2026-06-19-resonator-on-learned-codes-derisk.md). The TWO-
        # attribute (F=3 resonator) path is DELIBERATELY NOT added: it degrades to ~29% on the correlated learned
        # codes (same de-risk) and stays the documented boundary. `bind_roles` is the binding order (polarity LAST so
        # the existing flat layout is preserved when n_roles=4); `main_roles` is the subset cleaned against the main
        # vocab (every role except polarity, which uses the 2-word polarity codebook).
        self.enable_attributed = bool(enable_attributed)
        self.bind_roles = (["agent", "action", "patient", "attribute", "polarity"] if self.enable_attributed
                           else ["agent", "action", "patient", "polarity"])
        self.n_roles = len(self.bind_roles)
        self.main_roles = [r for r in self.bind_roles if r != "polarity"]   # cleaned vs the main vocab (3 or 4)
        self.n_main = len(self.main_roles)
        # work registers: fill_0..n-1 (n) + bound_0..n-1 (n) + acc (1) = 2*n+1 D-blocks. Default n=4 -> 9*D (byte-equal).
        self.store_base = self.P + (2 * self.n_roles + 1) * D
        self.block = 1 + D
        self.q_base = self.store_base + self.k_max * self.block      # PER-BLOCK (oracle): n_roles Q regs (one per role)
        self.c_base = self.q_base + self.n_roles * D                 # PER-BLOCK cleanup: n_main V-blocks + 1 NP-block
        self.cb = self.n_main * self.V + self.NP                    # cleanup neurons per block (main roles + polarity)
        # BATCHED region (A5 lever 1): K_max x (n_roles Q regs + cb cleanup) so all blocks read in one pass (additive --
        # the per-block region above is unchanged = the correctness oracle).
        self.bat_q_base = self.c_base + self.cb
        self.bat_c_base = self.bat_q_base + self.k_max * self.n_roles * D
        self.n_total = self.bat_c_base + self.k_max * self.cb
        self.b = build_coresident_bridge(seed, self.n_total, enable_rf_cudagraph=self.enable_rf_cudagraph)
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=0)   # wires+trains [0:P]
        self.rf_mask = np.zeros(self.n_total, dtype=bool); self.rf_mask[self.P:self.n_total] = True
        self.kb = []          # bookkeeping: list of (fact_dict, None) -- the agent's _assoc_graph reads fact dicts;
        #                       the bound VECTOR is on-substrate (the None placeholder keeps the (fact, vec) shape)
        self.store_conns = []
        self._word_index = {w: i for i, w in enumerate(self.words)}   # word -> codebook index (the sequencer cue idx)

    # --- comprehend + store ---
    def _pol(self, polarity):
        return polarity if polarity in self.pol_words else "AFFIRM"

    def _resolve_patient(self, patient):
        """Split a patient operand into (noun, attribute). A bare concept word -> (word, None). An attributed entity
        (adjs, noun) tuple -> (noun, the FIRST adjective) when the composer is attribute-enabled; the single-attribute
        (2-factor) path is the HOLDING one on the learned codes -- a 2nd adjective is dropped (the documented F=3 two-
        attribute boundary, ~29% on learned codes, deliberately not bound here). A Clause patient is returned as-is
        (noun=the Clause, attribute=None) so the recursive-clause path is unchanged."""
        if _is_clause(patient) or not isinstance(patient, tuple):
            return patient, None
        adjs, noun = patient                                    # (adj(s), noun)
        adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
        return noun, (adjs[0] if adjs else None)

    def _store_fact(self, agent, action, patient, polarity):
        """Compose + store one fact (the single _store_composite path for hear()/store()). When attribute-enabled and
        the patient is an attributed entity, the attribute role is bound (single-attribute); otherwise the flat 4-role
        path (byte-identical). Only the roles the fact ACTUALLY has are bound (a plain fact stays a 4-way bundle even
        on the attribute-enabled composer -> no extra crosstalk), in self.bind_roles order. The read path always
        unbinds the full bind_roles set; an un-bound role's unbind is noise the kb dict ignores (no "attribute" key ->
        the attribute is not joined into the answer). Returns the fact dict appended to kb."""
        pol = self._pol(polarity)
        noun, attr = self._resolve_patient(patient) if self.enable_attributed else (patient, None)
        fact = {"agent": agent, "action": action, "patient": noun, "polarity": pol}
        if self.enable_attributed and attr is not None:
            fact["attribute"] = attr
        roles = [r for r in self.bind_roles if r in fact]       # bind only present roles, in canonical order
        self._store_composite([fact[r] for r in roles], roles)
        return fact

    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO sentence with the on-bridge parser (its role firing selects each bind) + store the fact.
        `polarity` (AFFIRM default / NEGATE) is bound as a role -> `ask_yes_no` returns yes/no/unknown."""
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = [self.parser.role_of(pos, voice) for pos in range(3)]
        rmap = {roles[i]: words[i] for i in range(3)}
        fact = self._store_fact(rmap.get("agent"), rmap.get("action"), rmap.get("patient"), polarity)
        self.kb.append((fact, None))
        return fact

    def hear_multiframe(self, sentence, verbs, polarity=None):
        """Comprehend a sentence in an AUTO-SELECTED word-order frame (SVO/VSO/OSV) via the neural FrameParser, then
        store the resolved fact. `verbs` is the known-verb set (the lexical front end the frame selector uses to find
        the verb position). Requires enable_multiframe=True. Returns the parsed fact dict. Same store path as hear()
        (so it also handles an attributed patient when attribute-enabled)."""
        assert self.enable_multiframe, "hear_multiframe needs OneBrainComposer(enable_multiframe=True)"
        if self._frame_parser is None:
            from research.runners.frame_parser import FrameParser
            self._frame_parser = FrameParser(seed=self.seed)
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        rmap = self._frame_parser.parse(words, verbs)
        fact = self._store_fact(rmap.get("agent"), rmap.get("action"), rmap.get("patient"), polarity)
        self.kb.append((fact, None))
        return fact

    def store(self, agent, action, patient, polarity=None):
        """Store a fact whose roles are already resolved (API parity with RFPhasorComposer; used when the caller's
        parser comprehends). Binds agent/action/patient + the polarity tag (AFFIRM default). When attribute-enabled,
        an attributed-entity patient `(adjs, noun)` binds the single-attribute role too -> 'big apple'."""
        fact = self._store_fact(agent, action, patient, polarity)
        self.kb.append((fact, None))

    def _compose_phases(self, fillers, roles):
        """Bind each (role, filler) + bundle -> the composite phasor PHASES, via the work registers (fill_* -> bound_*
        -> acc). `_filler_phases` handles BOTH a concept word (its code) AND a recursive Clause (its bound composite),
        so a clause patient is the same path -- the patient role binds the clause's composite. Shared by the initial
        store AND the reconsolidation in-place rewrite (the only difference is which block the result is written to).
        The work layout is fill_0..n-1 (blocks 0..n-1), bound_0..n-1 (blocks n..2n-1), acc (block 2n) -- n = the number
        of roles passed (4 default, 5 with the attribute role); for n=4 the block math is identical to before."""
        comp, b, D, P, Pd = self.comp, self.b, self.D, self.P, self.period
        n = len(roles); acc = 2 * n                                                               # acc at block 2n
        binds, bundle = [], []
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(n):
            zr = comp._to_phasor(comp.roles[roles[i]]); zf = comp._to_phasor(comp._filler_phases(fillers[i]))
            kick[P + i * D:P + (i + 1) * D] = zf                                                  # fill_i at block i
            binds += [(P + (n + i) * D + k, P + i * D + k, complex(zr[k])) for k in range(D)]     # bound_i at block n+i
            bundle += [(P + acc * D + k, P + (n + i) * D + k, 1.0) for k in range(D)]             # acc at block 2n
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(Pd + 8)
        return comp._to_phasor(np.asarray(b.rf_read_phases())[P + acc * D:P + (acc + 1) * D])

    def _write_block(self, i, zc):
        """Write block i's persistent trigger->readout store weights (the composite `zc`). store_conns is block-major
        (block i = the i-th D-run), so an existing block is REPLACED in place (reconsolidation) and a new one is
        APPENDED (initial store) -- the slice math is exact either way."""
        D = self.D
        trig = self.store_base + i * self.block
        # (Tier-2 #6) DA-gated encoding strength: scale the stored composite magnitude by the per-fact gain `g` read from
        # the dopamine signal at store time. g=1.0 (encoding_gain_fn=None) -> the byte-identical unit-mag write (this
        # mirrors RFPhasorComposer._store_substrate). The RF read floor (sim/bridge.py:5589) makes the gain differential.
        g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())
        block_conns = [(trig + 1 + k, trig, complex(g) * zc[k]) for k in range(D)]
        if i * D < len(self.store_conns):
            self.store_conns[i * D:(i + 1) * D] = block_conns       # in-place rewrite (reconsolidation)
        else:
            self.store_conns += block_conns                         # append (a new fact)
        self._store_dirty = True       # store_conns changed -> the cached store CSR is stale (both store + reconsolidation)
        if self.integrated_loop:
            self._seq_dirty = True     # (shortcut #3) the store changed -> the per-block sequencer drives are stale

    def _store_composite(self, fillers, roles):
        i = len(self.kb)
        if i >= self.k_max:
            raise RuntimeError(f"OneBrainComposer store full: k_max={self.k_max} reached (shard or raise k_max)")
        self._write_block(i, self._compose_phases(fillers, roles))

    def _unbind_conj(self, role):
        """The UNBIND synapse weight phasor for `role` = conj(role phasor). DEFAULT (local_reciprocal_unbind=False):
        the host np.conj (the legacy path -- the genuine host residual). With the flag ON: the LOCAL reciprocal-
        conjugate rule (comp._local_conj, a per-component quadrature flip of the role phasor) -- no host np.conj, the
        unbind structure derived locally from the bind (role) phasor. Byte-identical (== conj for a unit phasor).
        Used at every unbind-structure site so the production one-brain bind structure becomes host-free at runtime."""
        comp = self.comp
        zr = comp._to_phasor(comp.roles[role])
        return comp._local_conj(zr) if self.local_reciprocal_unbind else np.conj(zr)

    def _cleanup_conj(self, concept_word):
        """The CLEANUP / matched-filter codebook synapse weight phasor for `concept_word` = conj(concept phasor) --
        so the recovered phasor correlates against each concept's CONJUGATE (the matched filter = the transpose/
        reciprocal of the encoder). DEFAULT (local_reciprocal_unbind=False): the host np.conj (the legacy residual).
        With the flag ON: the SAME one-time LOCAL reciprocal-conjugate rule already used for the unbind (per-component
        quadrature flip via comp._cleanup_conj/_local_conj) -- no host np.conj over the concept code, the cleanup
        codebook derived locally from the (learned/developmental) concept phasor. Byte-identical (== conj for a unit
        phasor). Routed at every cleanup-codebook site so the WHOLE bind+cleanup structure is host-free at runtime (the
        neuromorphic-port property). See 2026-06-20-FHRR-B-cleanup-codebook-local-conj.md."""
        comp = self.comp
        return comp._cleanup_conj(comp._to_phasor(comp.concepts[concept_word]))

    # --- query (cue-matching scan; reconstruct ONCE per block, read all 4 roles in PARALLEL) ---
    @staticmethod
    def _margin(scores):
        """Normalized decisiveness of a cleanup read-out = (peak - runner_up) / (peak + eps). ~1 when one concept
        dominates (a confident, familiar read), ~0 when the scores are flat (a noise-dominated, unfamiliar read).
        The confidence_gate compares the min of the agent+action cue-role margins against it."""
        s = np.sort(np.maximum(np.asarray(scores, dtype=float), 0.0))[::-1]
        return float((s[0] - s[1]) / (s[0] + 1e-9)) if s.size >= 2 and s[0] > 0.0 else 0.0

    def _spiking_select(self, scores, words):
        """Burndown #1 -- the cleanup SELECTION in SPIKES. `scores` are the rectified matched-filter membrane values
        (one per candidate in `words`), ALREADY computed on the co-resident bridge's complex-synapse cleanup. Stage 2
        (the SELECTION) is the validated NEF spiking WTA (== RFPhasorComposer._spiking_cleanup's Stage 2): input-
        normalize the scores -> drive a cached Izhikevich concept bank (reused from the inner RFPhasorComposer, keyed by
        candidate count) -> integrate firing over the cleanup window -> winner = argmax-over-FIRING (a readout of the
        spiking competition, the body-read of which neuron won, NOT a host argmax over the membrane). Off-target
        concepts get ZERO normalized drive (rectified scores) so they stay silent -> a clean WTA ('off-target emits zero
        spikes', Stewart-Tang-Eliasmith). Degenerate fallbacks (zero peak / zero firing) read the argmax of the same
        non-negative scores -- the same value the host path would return -- so a silent competition never confabulates."""
        comp = self.comp
        scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
        V = len(words)
        peak = float(scores.max()) if V else 0.0
        if peak <= 1e-9:
            return words[int(np.argmax(scores))]
        drive = (scores / peak) * comp._cleanup_drive_pA
        bank = comp._izh_bank(V)
        bank.cp_membrane_potential_v[:] = bank._cleanup_v0     # reset to resting -> each cleanup is independent
        bank.cp_recovery_variable_u[:] = bank._cleanup_u0
        import sim.backend as _b
        xp, _ = _b.get_backend()
        bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(V)
        for _ in range(comp._cleanup_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        if float(firing.max()) <= 0.0:
            return words[int(np.argmax(scores))]
        return words[int(np.argmax(firing))]

    def _select(self, scores, words):
        """Pick the winning concept from a role's matched-filter scores. Default (enable_spiking_cleanup=False): the
        byte-identical host argmax (the numpy-CPU + test-oracle path). When ON: the fully-on-substrate spiking WTA
        (`_spiking_select`). The single dispatch the three cleanup read sites share (per-block, batched, clause)."""
        if self.enable_spiking_cleanup:
            return self._spiking_select(scores, words)
        return words[int(np.argmax(np.asarray(scores, dtype=float)))]

    def _read_block(self, block_idx):
        """Reconstruct block_idx + unbind all roles IN PARALLEL (one settle, no phase drift). The main roles (agent,
        action, patient, +attribute when enabled) clean up against the main vocab; the polarity role cleans up against
        the 2-word polarity codebook. Returns a dict {role: word} for the bind_roles (attribute present only on the
        attribute-enabled composer; its value is noise for a plain fact and the caller ignores it via the kb dict)."""
        comp, b, D, Pd, V, NP = self.comp, self.b, self.D, self.period, self.V, self.NP
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for ri, role in enumerate(self.bind_roles):
            zc = self._unbind_conj(role)
            unbind += [(self.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        clean = []
        for ri, role in enumerate(self.main_roles):                     # main roles -> the main vocab codebook
            for j in range(V):
                cc = self._cleanup_conj(self.words[j])                   # local reciprocal rule when ON; conj when OFF
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        pol_ri = self.bind_roles.index("polarity")                      # polarity role -> the 2-word polarity codebook
        for j in range(NP):
            cc = self._cleanup_conj(self.pol_words[j])                   # local reciprocal rule when ON; conj when OFF
            clean += [(self.c_base + self.n_main * V + j, self.q_base + pol_ri * D + k, complex(cc[k]))
                      for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        scores = [np.maximum(mem[self.c_base + ri * V:self.c_base + (ri + 1) * V], 0.0) for ri in range(self.n_main)]
        out = {role: self._select(scores[ri], self.words) for ri, role in enumerate(self.main_roles)}
        pol_scores = np.maximum(mem[self.c_base + self.n_main * V:self.c_base + self.n_main * V + NP], 0.0)
        out["polarity"] = self._select(pol_scores, self.pol_words)
        if self.confidence_gate > 0.0 and min(self._margin(scores[0]), self._margin(scores[1])) < self.confidence_gate:
            return {role: None for role in self.bind_roles}   # an unfamiliar (noise-dominated) block -> blank -> abstain
        return out

    def _build_batched_unbind_clean(self, n):
        """Build the QUERY-INVARIANT batched unbind + cleanup connection lists for `n` blocks and convert them to device
        CSR pairs. These depend ONLY on (n, the role/concept codebooks, the fixed block layout) -- never on the stored
        fact content (that lives in store_conns) -- so for a fixed store size they are byte-IDENTICAL every query. Built
        once per n and cached in self._csr_cache[n]. Returns ((Ure,Uim),(Cre,Cim)). Iterates self.bind_roles /
        self.main_roles so the layout follows n_roles (4 default, 5 with the attribute role)."""
        comp, D, V, NP = self.comp, self.D, self.V, self.NP
        nr, nm = self.n_roles, self.n_main
        pol_ri = self.bind_roles.index("polarity")
        unbind = []
        for i in range(n):
            trig = self.store_base + i * self.block
            for ri, role in enumerate(self.bind_roles):
                zc = self._unbind_conj(role)
                qreg = self.bat_q_base + (i * nr + ri) * D
                unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        clean = []
        for i in range(n):
            cblk = self.bat_c_base + i * self.cb
            for ri in range(nm):                                          # main roles -> the main vocab codebook
                qreg = self.bat_q_base + (i * nr + ri) * D
                for j in range(V):
                    cc = self._cleanup_conj(self.words[j])                 # local reciprocal rule when ON; conj when OFF
                    clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
            qreg_p = self.bat_q_base + (i * nr + pol_ri) * D              # polarity role -> the polarity codebook
            for j in range(NP):
                cc = self._cleanup_conj(self.pol_words[j])                 # local reciprocal rule when ON; conj when OFF
                clean += [(cblk + nm * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
        return (_build_complex_csr(self.n_total, unbind), _build_complex_csr(self.n_total, clean))

    def _store_csr_cached(self):
        """The store_conns CSR pair, (re)built only when _store_dirty (a write since the last build). A query never
        changes store_conns, so this is built once per store/reconsolidation and reused across all subsequent reads."""
        if self.enable_csr_cache and not self._store_dirty and self._store_csr is not None:
            return self._store_csr
        self._store_csr = _build_complex_csr(self.n_total, self.store_conns)
        self._store_dirty = False
        return self._store_csr

    def _read_all_blocks(self):
        """A5 lever 1 (BATCHED): read ALL stored blocks in 3 resonate windows -- fire EVERY trigger (the readouts
        reconstruct in parallel, the validated per-block isolation, zero cross-talk) -> block-diagonal unbind (each
        block's 4 roles into the batched Q region) -> block-diagonal cleanup -> read all. == the per-block loop
        (de-risk `_phaseB_onebrain_batched_scan_derisk.py`: 6/6 answer-identical, 7.3x). Returns [(a,v,p,pol)] per block.

        A5 lever 4 (CSR cache, default on): the store CSR is reused across queries (rebuilt only on a write), and the
        unbind + cleanup CSRs (query-INVARIANT, keyed by n) are built once and installed by direct cp_rf_w_re/im
        assignment instead of rebuilt from fresh tuple lists per query. ANSWER-IDENTICAL -- the reused CSRs hold the
        same values; the dynamics + the megakernel matvec are byte-unchanged. enable_csr_cache=False = the stock path."""
        comp, b, D, Pd, V, NP = self.comp, self.b, self.D, self.period, self.V, self.NP
        n = len(self.kb)
        if n == 0:
            return []
        if self.enable_csr_cache:
            if n not in self._csr_cache:
                self._csr_cache[n] = self._build_batched_unbind_clean(n)       # query-invariant: build once per n
            (Ure, Uim), (Cre, Cim) = self._csr_cache[n]
            Sre, Sim = self._store_csr_cached()                                # rebuilt only when store changed
            b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
            kick = np.zeros(self.n_total, dtype=np.complex128)
            for i in range(n):
                kick[self.store_base + i * self.block] = 1.0                   # fire EVERY stored trigger
            b.cp_rf_w_re, b.cp_rf_w_im = Sre, Sim                              # install the cached store operator
            b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
            b.rf_resonate_steps(Pd + 8)
            b.cp_rf_w_re, b.cp_rf_w_im = Ure, Uim; b.rf_resonate_steps(Pd + 8)  # cached unbind
            b.cp_rf_w_re, b.cp_rf_w_im = Cre, Cim; b.rf_resonate_steps(1)       # cached cleanup
            mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
            return self._decode_batched_mem(mem, n)
        # --- stock path (cache off): rebuild every CSR from fresh tuple lists each query ---
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(n):
            kick[self.store_base + i * self.block] = 1.0                       # fire EVERY stored trigger
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        nr, nm = self.n_roles, self.n_main
        pol_ri = self.bind_roles.index("polarity")
        unbind = []
        for i in range(n):
            trig = self.store_base + i * self.block
            for ri, role in enumerate(self.bind_roles):
                zc = self._unbind_conj(role)
                qreg = self.bat_q_base + (i * nr + ri) * D
                unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        clean = []
        for i in range(n):
            cblk = self.bat_c_base + i * self.cb
            for ri in range(nm):
                qreg = self.bat_q_base + (i * nr + ri) * D
                for j in range(V):
                    cc = self._cleanup_conj(self.words[j])                 # local reciprocal rule when ON; conj when OFF
                    clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
            qreg_p = self.bat_q_base + (i * nr + pol_ri) * D
            for j in range(NP):
                cc = self._cleanup_conj(self.pol_words[j])                 # local reciprocal rule when ON; conj when OFF
                clean += [(cblk + nm * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        return self._decode_batched_mem(mem, n)

    def _decode_batched_mem(self, mem, n):
        """Decode the batched cleanup membrane read-out into a list of {role: word} dicts per block (the argmax +
        confidence-gate logic, shared by the cached + stock batched paths so they are answer-identical by
        construction). The main roles read off their V-block; polarity reads the NP-block after them. The agent +
        action margins drive the confidence gate."""
        V, NP, nm = self.V, self.NP, self.n_main
        out = []
        for i in range(n):
            cblk = self.bat_c_base + i * self.cb
            scores = [np.maximum(mem[cblk + ri * V:cblk + (ri + 1) * V], 0.0) for ri in range(nm)]
            row = {role: self._select(scores[ri], self.words) for ri, role in enumerate(self.main_roles)}
            ps = np.maximum(mem[cblk + nm * V:cblk + nm * V + NP], 0.0)
            row["polarity"] = self._select(ps, self.pol_words)
            if self.confidence_gate > 0.0 and min(self._margin(scores[0]), self._margin(scores[1])) < self.confidence_gate:
                row = {role: None for role in self.bind_roles}   # an unfamiliar (noise-dominated) block -> blank -> abstain
            out.append(row)
        return out

    def _read_blocks(self):
        """All stored blocks as {role: word} dicts: the BATCHED read (default, A5 lever 1) or the per-block loop (the
        oracle). Each dict has agent/action/patient/polarity (+attribute on the attribute-enabled composer)."""
        if self.enable_batched:
            return self._read_all_blocks()
        return [self._read_block(i) for i in range(len(self.kb))]

    def _ensure_sequencer(self, K):
        """Lazily build (and cache) the spiking K-way sequencer control fabric + the divnorm score bridge for store
        size K, and (re)compute the per-block decoded-line drives when the store grew or a write dirtied them. Reuse-
        by-import (NO sim/ edit): `build_sequencerK_bridge` (the gated-disinhibition match cascade + BG first-match
        priority WTA, S0) + `build_divnorm_score_bridge` (the on-bridge divisive normalization, S5) + `make_block_drives`
        (the divnorm-normalized decoded-line drive per block, S2) -- all at the validated op-point (gain/sigma/input_gain
        from __init__). The sequencer + score bridges depend only on (seed, V, K), so they are rebuilt only when K
        changes; the drives depend on the stored content, so they are rebuilt when _seq_dirty (a write happened) or K
        changed. The drives are derived from the composer's OWN on-bridge cleanup scores (`block_cleanup_scores`)."""
        fns = _seq_imports()
        if self._seq is None or self._seq_K != K:
            sb, meta = fns["build_sequencerK_bridge"](seed=self.seed, V=self.V, K=K)
            score_sb = fns["build_divnorm_score_bridge"](seed=self.seed, V=self.V, enable_divnorm=True,
                                                         sigma=self.sequencer_sigma, gain=self.sequencer_gain)
            self._seq = (sb, meta); self._seq_score = score_sb; self._seq_K = K
            self._seq_dirty = True                                 # a new K -> the drives must be (re)built
        if self._seq_dirty or self._seq_drives is None:
            bscores = [fns["block_cleanup_scores"](self, b) for b in range(K)]   # the composer's own op result per block
            drives, _lit = fns["make_block_drives"](self._seq_score, self.V, bscores,
                                                    input_gain=self.sequencer_input_gain, retreat="divnorm",
                                                    peak_mult=1.0)
            self._seq_drives = drives
            self._seq_dirty = False

    def _seq_block(self, agent, action):
        """The SELECTED block index for cue (agent, action) -- the spiking K-way sequencer decision (or None = abstain),
        replacing the host first-match loop. integrated_loop OFF -> the host read (byte-identical, the test oracle).
        Built lazily; the sequencer + drives are (re)built only when the store size changes or a write dirtied them
        (shortcut #3, the plan). The (agent, action) hot-path sites delegate here."""
        if not self.integrated_loop:
            # the host path: the EXACT same first-match loop the (agent, action) sites used (read here once so all
            # callers share it). == host_scan_block (the de-risk's `first_block_where(agent==., action==.)`).
            for i, got in enumerate(self._read_blocks()):
                if got.get("agent") == agent and got.get("action") == action:
                    return i
            return None
        # the spiking path (lazy build; rebuild drives on a dirtied/grown store).
        K = len(self.kb)
        if K == 0:
            return None
        if agent not in self._word_index or action not in self._word_index:
            return None                                           # an absent cue WORD -> no block -> abstain (the moat)
        self._ensure_sequencer(K)
        fns = _seq_imports()
        sb, meta = self._seq
        dec, _rates = fns["run_sequencerK_with_drive"](sb, meta, self._word_index[agent], self._word_index[action],
                                                       self._seq_drives, match_thresh=self.sequencer_match_thresh)
        return fns["decision_to_block"](dec, K)

    def _scan(self, cue, answer_role):
        for got in self._read_blocks():
            if all(got.get(role) == want for role, want in cue.items()):
                return got.get(answer_role)
        return None

    def _decode_clause(self, block_idx, order_fn=None):
        """Recursive clause decode (== the rf composer's `_render`): reconstruct the outer fact, unbind the OUTER
        patient role to recover the embedded CLAUSE composite, then unbind the clause's 3 roles + cleanup ->
        'agent action patient'. The decode is TWO unbind hops; like the numpy oracle (`_unbind_phases` kicks a fresh
        unit phasor each hop), the intermediate clause composite is READ OUT and RE-KICKED as a clean unit phasor
        before the 2nd hop -- chaining the resonate through an unbind-DRIVEN register (instead of a kicked one)
        degrades its magnitude and the deeper unbind reads the wrong filler (the agent slot fails first)."""
        comp, b, D, Pd, V = self.comp, self.b, self.D, self.period, self.V
        # Q register holding the recovered outer patient (= the clause composite). Reuse the POLARITY Q slot as scratch
        # (clause decode never reads polarity), which is valid for both the 4-role default (pol at index 3, == the old
        # hardcoded Q[3]) and the 5-role attribute layout (pol at index 4) -- always inside the per-block Q region, so
        # it never clobbers the cleanup region at c_base.
        pq = self.bind_roles.index("polarity")                             # the polarity Q slot, reused as scratch
        # hop 1: reconstruct the outer block (kick) + unbind the OUTER patient -> the embedded clause composite in Q[pq]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        zc = self._unbind_conj("patient")
        outer = [(self.q_base + pq * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(outer); b.rf_resonate_steps(Pd + 8)
        clause_phases = np.asarray(b.rf_read_phases())[self.q_base + pq * D:self.q_base + (pq + 1) * D]
        # hop 2: RE-KICK the clause composite as a clean unit phasor (== the oracle's fresh per-hop kick), then unbind
        # the 3 clause roles IN PARALLEL from Q[pq] -> Q[0..2] + cleanup against the main vocab
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        kick2 = np.zeros(self.n_total, dtype=np.complex128)
        kick2[self.q_base + pq * D:self.q_base + (pq + 1) * D] = comp._to_phasor(clause_phases)
        inner = []
        for ri, role in enumerate(ROLES3):
            zcr = self._unbind_conj(role)
            inner += [(self.q_base + ri * D + k, self.q_base + pq * D + k, complex(zcr[k])) for k in range(D)]
        b.rf_set_complex_weights(inner); b.rf_kick(kick2, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        clean = []
        for ri in range(3):
            for j in range(V):
                cc = self._cleanup_conj(self.words[j])                     # local reciprocal rule when ON; conj when OFF
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        words = [self._select(np.maximum(mem[self.c_base + ri * V:self.c_base + (ri + 1) * V], 0.0), self.words)
                 for ri in range(3)]
        order = order_fn(3) if order_fn is not None else [0, 1, 2]
        return " ".join(words[o] for o in order)

    def _attributed_patient(self, i, wp, got):
        """The patient word with its (single) attribute prepended, when this fact stored one -- 'big apple'. The
        attribute word is DECODED from the on-bridge unbind (got["attribute"], already in the read row passed by the
        caller); the kb dict only ROUTES whether to join it (a plain fact has no 'attribute' key -> the bare noun).
        Single-attribute only (the 2-factor path the de-risk validated 100% on the learned codes)."""
        if not self.enable_attributed or i >= len(self.kb) or "attribute" not in self.kb[i][0]:
            return wp
        adj = got.get("attribute")
        return f"{adj} {wp}" if adj is not None else wp

    def query_patient(self, agent, action, order_fn=None):
        """patient (a concept word) OR, when the stored fact's patient is an embedded CLAUSE, the recursively-decoded
        clause sentence; an attributed patient ('big apple') prepends the decoded attribute. Matches agent+action via
        the batched read, then routes on the kb-stored patient type."""
        for i, got in enumerate(self._read_blocks()):
            if got.get("agent") == agent and got.get("action") == action:
                stored = self.kb[i][0].get("patient") if i < len(self.kb) else None
                if _is_clause(stored):
                    return self._decode_clause(i, order_fn=order_fn)
                return self._attributed_patient(i, got.get("patient"), got)
        return None

    def query_agent(self, action, patient):
        return self._scan({"action": action, "patient": patient}, "agent")

    def ask_yes_no(self, agent, action, patient):
        """yes / no / unknown: the first fact matching the full SVO answers by its polarity tag (AFFIRM -> yes,
        NEGATE -> no); no matching fact -> 'unknown' (the no-confab moat)."""
        for got in self._read_blocks():
            if got.get("agent") == agent and got.get("action") == action and got.get("patient") == patient:
                return "yes" if got.get("polarity") == "AFFIRM" else "no"
        return "unknown"

    def render_fact(self, agent, order_fn=None):
        """Generation (for the agent's `describe`): 'agent action patient' decoded from the first stored fact whose
        agent matches, or None (the no-confab moat -- no invented sentence about an unknown subject). The action +
        patient are DECODED from the on-bridge unbind (not the stored labels). When the matched fact's patient is an
        embedded CLAUSE, the patient slot is the recursively-decoded clause ('dog see cat go south'); an attributed
        patient renders as 'big apple'. `order_fn` (opt-in) -> the word order (the spiking serial-order renderer);
        default = subject-verb-object."""
        for i, got in enumerate(self._read_blocks()):
            if got.get("agent") == agent:
                stored = self.kb[i][0].get("patient") if i < len(self.kb) else None
                wp = got.get("patient")
                pt = self._decode_clause(i, order_fn=order_fn) if _is_clause(stored) else self._attributed_patient(i, wp, got)
                words = [got.get("agent"), got.get("action"), pt]
                order = order_fn(3) if order_fn is not None else [0, 1, 2]
                return " ".join(words[o] for o in order)
        return None

    def query_chain(self, cue, actions):
        """Multi-hop relational reasoning (for the agent's `reason_chain`): `cue` is the starting agent; each action's
        patient becomes the next hop's agent cue. None (abstain) the moment any hop has no matching fact -- the
        no-confab moat holds at EVERY hop (it iterates query_patient, which already abstains on a miss)."""
        current = cue
        for action in actions:
            current = self.query_patient(current, action)
            if current is None:
                return None
        return current

    # --- reconsolidation: prediction-error-gated in-place fact update (== the rf composer's update_on_mismatch) ---
    def _recovered_patient_phases(self, block_idx):
        """Reconstruct block_idx + unbind the patient role -> the RAW recovered patient phases (NOT cleaned up to a
        word). The reconsolidation prediction error compares these against an asserted patient's code."""
        comp, b, D, Pd = self.comp, self.b, self.D, self.period
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        zc = self._unbind_conj("patient")
        unbind = [(self.q_base + 2 * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]      # patient -> Q[2]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        return np.asarray(b.rf_read_phases())[self.q_base + 2 * D:self.q_base + 3 * D]

    def _patient_prediction_error(self, block_idx, patient_word):
        """PE = 1 - phase-cos(recovered patient phasor, the asserted patient's code). ~0 when the asserted filler
        matches the stored one (a re-statement); ~1 on a mismatch (a correction). == the rf composer's measure."""
        rec = self._recovered_patient_phases(block_idx)
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.comp.concepts[patient_word]))))

    def _calibrate_pe_labile(self):
        """Frozen labilization gate = the midpoint of the same-vs-different prediction-error distributions over the
        CURRENT facts (each fact's PE against its OWN stored patient = 'same'; against other facts' patients =
        'different'). The data's own separation point -- NOT tuned to a downstream probe. 0.5 fallback when too few
        distinct facts exist to calibrate. == the rf composer's _calibrate_pe_labile (string-patient facts only)."""
        idxs = [i for i, (fact, _) in enumerate(self.kb) if isinstance(fact.get("patient"), str)]
        recs = {i: self._recovered_patient_phases(i) for i in idxs}
        pats = {i: self.kb[i][0]["patient"] for i in idxs}

        def pe(rec, word):
            return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.comp.concepts[word]))))
        same, diff = [], []
        for i in idxs:
            same.append(pe(recs[i], pats[i]))
            for j in idxs:
                if pats[j] != pats[i]:
                    diff.append(pe(recs[i], pats[j]))
        if not same or not diff:
            return 0.5
        return 0.5 * (float(np.mean(same)) + float(np.mean(diff)))

    def _find_cued_block(self, agent, action):
        """The FIRST stored block whose cue roles (agent+action) match (the batched read), or None (no trace to
        reactivate -> abstain). Returns the block/kb index."""
        for i, got in enumerate(self._read_blocks()):
            if got.get("agent") == agent and got.get("action") == action:
                return i
        return None

    def update_on_mismatch(self, agent, action, new_patient, pe_labile=None):
        """RECONSOLIDATION: a corrective utterance ('actually, <agent> <action> <new_patient>') reactivates the cued
        fact and -- ONLY if the new filler carries a prediction error above the labilization gate -- rewrites that
        fact's patient IN PLACE (no contradictory duplicate). A fully-predicted re-statement re-stabilizes unchanged;
        a NEVER-stored cue ABSTAINS (the no-confab moat: a reactivated trace is updated, a missing one is not
        fabricated). The in-place rewrite re-composes the fact (new patient) and OVERWRITES the same store block.
        ADDITIVE -- store/query are unchanged. pe_labile=None -> auto-calibrate from the current facts. Returns
        {action: abstain|rewrite|restabilize, wrote: bool, pe: float|None}. == the rf composer (Nader 2000;
        Osan-Tort-Amaral 2011; de-risked 6/6: 2026-06-17-reconsolidation-update-derisk-GO.md)."""
        idx = self._find_cued_block(agent, action)
        if idx is None:
            return {"action": "abstain", "wrote": False, "pe": None}    # no trace -> no update, no fabrication
        gate = self._calibrate_pe_labile() if pe_labile is None else float(pe_labile)
        pe = self._patient_prediction_error(idx, new_patient)
        if pe >= gate:
            f2 = dict(self.kb[idx][0]); f2["patient"] = new_patient
            f2.setdefault("polarity", "AFFIRM")
            roles = [r for r in self.bind_roles if r in f2]              # recompose only the roles the fact has
            self._write_block(idx, self._compose_phases([f2[r] for r in roles], roles))
            self.kb[idx] = (f2, None)
            return {"action": "rewrite", "wrote": True, "pe": pe}
        return {"action": "restabilize", "wrote": False, "pe": pe}      # PE below the gate -> re-stabilize unchanged

    def count_facts(self, agent, action):
        """Number of stored facts whose cue roles (agent+action) match -- 1 after a reconsolidation update, 2 if a
        correction was naively appended. Used by the reconsolidation tests + the correction-turn hook."""
        return sum(1 for got in self._read_blocks() if got.get("agent") == agent and got.get("action") == action)
