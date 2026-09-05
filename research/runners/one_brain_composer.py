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
    from research.runners._seq_vocab_shrink_derisk import (
        build_sequencerK_reduced_bridge, reduced_cue_vocab, run_sequencerK_reduced_with_drive)
    return dict(build_sequencerK_bridge=build_sequencerK_bridge, decision_to_block=decision_to_block,
                block_cleanup_scores=block_cleanup_scores, build_divnorm_score_bridge=build_divnorm_score_bridge,
                make_block_drives=make_block_drives, run_sequencerK_with_drive=run_sequencerK_with_drive,
                build_sequencerK_reduced_bridge=build_sequencerK_reduced_bridge, reduced_cue_vocab=reduced_cue_vocab,
                run_sequencerK_reduced_with_drive=run_sequencerK_reduced_with_drive)

ROLES3 = ["agent", "action", "patient"]

# Sentinel for the fact-shard fast path: "this cue cannot be routed by the index -> the caller uses the full scan"
# (distinct from None, which the fast path uses for an honest ABSTAIN). See `_fact_shard_first_match`.
_FS_ESCALATE = object()


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
                 enable_attributed=False, enable_multiframe=False, enable_spiking_cleanup=True,
                 encoding_gain_fn=None, local_reciprocal_unbind=True, integrated_loop=False,
                 sequencer_match_thresh=0.06, sequencer_gain=0.11, sequencer_sigma=1.0, sequencer_input_gain=1.0,
                 enable_seq_vocab_shrink=True, persistent_loop=True, typed_roles=None, framecq_seed=None,
                 use_spiking_cq=None, frame_lexicon=None, trace=False, persistent_store=False,
                 vocab_headroom=0, homeostatic_scaling=False, homeo_beta_down=0.25,
                 homeo_s_min=0.34, homeo_s_max=4.0,
                 enable_sparse_index=False, sparse_index_g=3, sparse_index_G=16,
                 sparse_index_c=8, sparse_index_conf_floor=0.5, no_batched_region=False,
                 enable_fact_shard=False, fact_shard_g=2, fact_shard_G=4, fact_shard_c=8):
        self.seed = int(seed); self.D = int(D); self.period = int(period)
        # PERSISTENT STORE (2026-07-20, fact-store on the substrate, opt-in DEFAULT-OFF = byte-identical): when True the
        # fact composites live IN the device synapses (cp_rf_store_re/im via rf_set_store_weights) and PERSIST across
        # per-op binds, instead of the host store_conns list being re-installed onto cp_rf_w_* per read. The read is
        # IDENTICAL (de-risked: the RF read is phase-based + magnitude-invariant, staged vs persistent |Δphase|=0.0000).
        # False (default) => the staged per-read install path is byte-unchanged (the rf/numpy oracle + all tests).
        self.persistent_store = bool(persistent_store)
        self._persistent_dirty = True                          # store changed since the last rf_set_store_weights
        # trace (B3 per-turn "brain activity", opt-in, DEFAULT-OFF = byte-identical): READ-ONLY trace of what the brain
        # DID on the LAST query -- the decoded role-words + their cleanup match-confidence (per role), which stored
        # fact-block matched + how many were scanned, and a scalar RF activity gauge (the fraction of the rf-slice
        # readout neurons that crossed `cp_rf_fired[rf_mask].mean()` + the mean recovery magnitude |Z| over the rf
        # slice). Populated from the ALREADY-COMPUTED decoded {role: word} rows + the matched-filter membrane `scores`
        # `_read_blocks` produces + two `.mean()` reads of the shared bridge's `cp_rf_fired`/v/u over the rf slice --
        # NO extra resonate, strictly observational. The no-confab moat is UNCHANGED (an abstain records
        # matched_fact_index=None + scanned=N WITHOUT a fallback answer). trace=False (default) -> the dict is never
        # built (byte-identical numpy-CPU + test-oracle path). See research/findings/raw/_b3_activity_viz_scoping.md.
        self.trace = bool(trace)
        self.last_trace = None
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
        # integrated_loop="fused" (R1 close, opt-in, DEFAULT False = byte-identical): fold the divnorm-score pool + the
        # K-way sequencer onto ONE Izhikevich fabric bridge and route the cleanup membrane -> score pool DEVICE-RESIDENT
        # (NO `to_host` of the cleanup score), closing the LAST host DATA seam in the integrated who/what query path.
        # `integrated_loop=True` keeps the legacy SEPARATE-bridge spiking path (the revertible escape, byte-unchanged);
        # `integrated_loop=False` keeps the host `_scan` oracle. `self._fused` selects the fused fabric inside
        # `_seq_block` (a `bool(integrated_loop)` is still True for "fused", so the spiking branch is taken; the
        # `self._fused` sub-branch swaps the separate-bridge sequencer for the folded device-resident one). The fused
        # fabric + its per-block device-resident decoded-line drives are lazily built + cached on first query
        # (`_fused_seq`/`_fused_drives`), rebuilt when the store grows or a write dirties them. See
        # research/runners/_seq_fused_fabric.py + research/findings/2026-06-30-tier2-integrated-spiking-loop-scoping.md R1.
        self._fused = (integrated_loop == "fused")
        self._fused_seq = None        # (sb, meta) -- the fused fabric bridge, built lazily on first fused query
        self._fused_K = None          # the store size the current fused fabric/drives were built for
        self._fused_drives = None     # the per-block device-resident decoded-line drives (recomputed on a store change)
        self._fused_dirty = True      # the store changed since the fused drives were built -> rebuild them
        # persistent_loop (Tier-2 TRUE one brain / I-1-a op-handoff-as-spikes, DEFAULT ON since 2026-06-24 close-out
        # Closure 2 -- the I-1-a clean-unit-phasor op-handoff is ANSWER-IDENTICAL + cleanup-membrane BYTE-IDENTICAL to
        # the legacy carry-live-Z handoff [maxabs 0.0, _persistent_loop_flat_derisk GO], so flipping it on is a
        # behaviorally-INVISIBLE formalization that makes the flat who/what read the canonical "each register holds a
        # clean unit phasor between ops" form by default; pass persistent_loop=False for the legacy carry-live-Z path,
        # the byte-identical revertible escape): make the FLAT who/what query path a PERSISTENT INTERACTING SPIKING LOOP -- the
        # composite is handed off BETWEEN ops as a clean unit phasor held ON THE BRIDGE (register->register), with NO
        # host round-trip. Today the flat read (`_read_block` / `_read_all_blocks`) carries the LIVE register Z forward
        # across the unbind->cleanup handoff (the resonate matvec reads the live unbound Q register). When ON, that
        # handoff instead RE-KICKS a clean unit phasor into the unbound Q register(s) via `_dev_rekick_into` -- the I-1-a
        # mechanism: recover the phase from the device spike-step trackers + install exp(2pi i phi) into each register's
        # v/u + reset the RF trackers, ALL device ops, NO `to_host` of the phasor value. This is the canonical "each
        # register holds a clean unit phasor between ops" form (WM-attractor / reentrant-loop biology, catalog A.05 /
        # G.06-G.08) -- the brain's own op result drives the next op as spikes, not a host buffer copy. It is BYTE-
        # IDENTICAL (cleanup membrane atol 0) to the HOST ROUND-TRIP reference (`to_host(rf_read_phases) -> exp ->
        # rf_kick` the Q registers before cleanup) -- the I-1-a de-risk's exact GO (`_burndown_I1a_op_handoff_probe`,
        # max|dphase|=0; reproduced over the flat per-block AND batched reads in `_persistent_loop_flat_derisk`). It is
        # ANSWER-IDENTICAL to the carry-live-Z default (the cleanup argmax is invariant to the common |Z| scale the
        # carry path leaves inflated). The no-confab moat is preserved by construction: the cleanup winner-pick + the
        # cue-match abstention read the SAME relative cleanup pattern (only the register's magnitude is normalized to
        # unit before cleanup -- the argmax + the confidence-gate margin are unchanged).
        #
        # SCOPE (CLOSURE 5, purity backlog #5 -- extend the persistent spiking loop to ALL ops): persistent_loop now
        # also gates the RECONSOLIDATION prediction-error op. Audit of the non-flat ops found that almost all of them
        # were ALREADY spike-resident: the recursive CLAUSE path's hop-1->hop-2 handoff uses `_dev_rekick_into`
        # unconditionally (it inherited this I-1-a GO); negation/yes-no reads the polarity role IN PARALLEL within the
        # SAME flat reconstruction (no separate sub-op -- it inherits the flat handoff); query_chain's hop-to-hop
        # handoff is a DECODED WORD (the cleanup body read), not a phasor crossing an op boundary (the validated
        # "re-discretize between hops" design, 2026-06-17). The ONE remaining genuine host seam was RECONSOLIDATION's
        # PE: `_recovered_patient_phases` read the recovered patient phasor TO HOST (rf_read_phases) and
        # `_patient_prediction_error` computed `1 - mean(cos(...))` as a HOST numpy cos. When persistent_loop is ON,
        # the PE is now SPIKE-RESIDENT: re-kick the recovered patient (Q[2]) as a clean unit phasor (`_dev_rekick_into`,
        # no host phasor copy) and read PE_w = 1 - score_w/D off the on-substrate matched-filter membrane
        # (`_patient_cleanup_scores`). Decision-identical to the host cos (residual ~2.5e-8 float32 << the gate margin;
        # the rewrite/restabilize/abstain decision is invariant, as the flat argmax is). The STORE-side composite
        # read-out (`_compose_phases` -> `_write_block`) is a legitimate "consolidate the composed result into the
        # synaptic store" step, not a between-op cognitive handoff, and is left as-is. persistent_loop=False keeps the
        # legacy host-cos PE (the revertible escape). See research/findings/raw/_persistent_loop_flat_derisk.json +
        # _closure5_persistent_loop_all_ops.json + _closure5_reconsolidation_onsub_pe_derisk.json.
        self.persistent_loop = bool(persistent_loop)
        self.sequencer_match_thresh = float(sequencer_match_thresh)
        self.sequencer_gain = float(sequencer_gain)
        self.sequencer_sigma = float(sequencer_sigma)
        self.sequencer_input_gain = float(sequencer_input_gain)
        self._seq = None            # (sb, meta) -- the sequencer control bridge, built lazily on first query
        self._seq_score = None      # the divnorm score bridge
        self._seq_K = None          # the store size the current sequencer/drives were built for
        self._seq_drives = None     # the per-block decoded-line drives (recomputed when the store changes)
        self._seq_dirty = True      # the store changed since the drives were built -> rebuild the drives
        # enable_seq_vocab_shrink (audit #2, DEFAULT ON; only active on the integrated_loop spiking path): build the
        # K-way sequencer over only the DISTINCT stored agents (role A = V'_A) / actions (role X = V'_X) instead of the
        # full V word-lines, since a who/what cue can only ever be a stored agent/action (else the moat abstains BEFORE
        # the sequencer). Byte-identical decisions (cue + decoded lines remapped global->reduced; a spurious near-tie
        # decoded word outside the reduced vocab is dropped == a closed/absent line no battery cue drives) at ~34.6x
        # fewer sequencer neurons at production V=320/K=32 (_seq_vocab_shrink_derisk.py, GO 2026-06-21).
        self.enable_seq_vocab_shrink = bool(enable_seq_vocab_shrink)
        self._seq_mapA = None        # word -> reduced-index (role A) for the shrunk sequencer
        self._seq_mapX = None        # word -> reduced-index (role X)
        self._seq_cuevocab_sig = None  # (tuple(V'_A), tuple(V'_X)) -- rebuild the reduced fabric when this changes
        self._seq_cleanup_conns_cache = None  # opt #4: the block-invariant sequencer drive-seed cleanup conns (per rebuild)
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
        # homeostatic_scaling (2026-08-25, opt-in, DEFAULT-OFF = byte-identical): Turrigiano 2008 multiplicative
        # homeostatic SYNAPTIC SCALING on the substrate store synapses -- the on-substrate SYNAPTIC realization of the
        # host-proxy DA-encoding homeostat (webapp/da_encoding_drives_chat.homeostatic_step), which was a documented
        # PROXY (a feed-forward multiply+clip on the DA scalar at write time). This is instead a FEEDBACK rule on the
        # synaptic STATE: `apply_homeostatic_scaling()` resonates each stored engram, SENSES its readout-neuron activity
        # (a genuine neural read: mean |Z| over the block's D readout neurons -- constant for a unit write, linear in the
        # encoding strength), and multiplicatively rescales that engram's store synapses toward a homeostatic set-point
        # A* = the unit-write readout activity. Weak engrams (a low-DA fact the DA gate halved) are scaled UP to the
        # functional set-point (the recall-safe FLOOR, now EMERGENT from measured activity, not a host clip); strong
        # engrams are partially down-regulated by (A*/A_i)**homeo_beta_down (beta_down<1 PRESERVES the relative DA-salience
        # ORDER while regulating the extreme). The sensed variable is postsynaptic activity; the actuator is the synaptic
        # weight -> a faithful synaptic-scaling rule, NOT host arithmetic on the DA reading. DEFAULT-OFF (never called) ->
        # store_conns byte-identical. See research/findings 2026-08-25 (da-encoding substrate scaling).
        self.homeostatic_scaling = bool(homeostatic_scaling)
        self.homeo_beta_down = float(homeo_beta_down)    # <1 preserves DA-salience order under down-regulation
        self.homeo_s_min = float(homeo_s_min)            # cap on the strongest down-scale (a strong engram floor)
        self.homeo_s_max = float(homeo_s_max)            # cap on the strongest up-scale (a near-dead engram ceiling)
        self._homeo_scales = None                        # the last applied per-engram scale vector (diagnostic)
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
        # enable_sparse_index (KNOWLEDGE-SCALE fast path, board #150/#66; ADDITIVE + DEFAULT-OFF = BYTE-IDENTICAL to
        # today when off, incl. the numpy-CPU + test-oracle path): route the who/what recall's V-wide matched-filter
        # cleanup through a DG-like SPARSE INDEX so it runs O(shard) instead of O(V) as the concept codebook grows to
        # LLM scale (500k-1M concepts). The wall today is the cleanup: each role's recovered phasor is matched against
        # the codebook of ALL V concept codes (a V x D matvec) then argmax (`_select`) -- LINEAR in vocabulary, ~1.1 s
        # / recall at ~37k vocab and intractable at small-LLM scale. The brain does NOT linearly scan memory: the
        # dentate gyrus does sparse PATTERN SEPARATION (Kandel: 'pattern separation results from the divergence of
        # entorhinal inputs onto a larger number of granule cells') then CA3 auto-associative COMPLETION restricted to
        # the routed ensemble ('the recurrent excitatory connections of CA3 ... reactivation of a subset ... sufficient
        # to activate the entire original neural ensemble'). When ON, a DG expansion + hard k-WTA + CA3-conjunction
        # routing (reuse-by-import of the 6-seed-GO de-risk `_sparse_indexed_retrieval_derisk.DGSparseIndex`) routes the
        # recovered role phasor to a SMALL candidate SHARD of the codebook, and the SAME matched-filter cleanup + argmax
        # runs only over the shard rows. NO-REGRESSION is preserved two ways: (1) the SHARD is a SUBSET of the codebook,
        # so its peak score <= the full peak -- if the full scan's decode/abstain holds, the shard's does too (the moat
        # is intact by construction: an out-of-store cue whose full decode does not match any stored (agent, action)
        # cue still does not match under the shard, so query_patient/query_agent/ask_yes_no abstain identically); and
        # (2) a per-role CONFIDENCE FALLBACK -- when the shard's peak is not decisive (< conf_floor*D), the role
        # ESCALATES to the full-codebook host cleanup, so the decoded word is IDENTICAL to the full scan (a stored
        # fact's own code scores ~D and dominates -> the fast path fires on it; an ambiguous/degraded role escalates).
        # BRAIN-BASED: the in-shard matched-filter cleanup IS the composer's existing on-substrate op (the same complex-
        # synapse cleanup + argmax the full path runs, over fewer rows -- the composer's own oracle form, rf_phasor_
        # composer.py:662, sum_k cos(2pi(rec - code))). The DG sparse PROJECTION is a DECLARED host-rate stand-in; its
        # named spiking burn-down is the granule-cell WTA in the trisynaptic-loop probes (_riii_ca3_completion_
        # specificity_derisk.py, cortex_dg_ca3_cleanup_probe.py, _gap5_emergent_dg_selection_derisk.py). Determinism:
        # the index seeds from cfg.seed (self.seed). Biology binding: research/biology/dg-ca3-sparse-index.md. Env
        # BRAIN_SPARSE_INDEX_RETRIEVAL=1 flips it on without a code change (owner reviews the default-on flip
        # separately -- leave it OFF). The index does NOT engage when confidence_gate>0 (its shard-local margins are not
        # byte-identical to full-V margins) -> the full read runs, no regression; clause-patient RENDERING also stays
        # on the full bridge cleanup (only the flat who/what SELECTION + role decode route through the shard).
        import os as _os
        self.enable_sparse_index = bool(enable_sparse_index) or (
            _os.environ.get("BRAIN_SPARSE_INDEX_RETRIEVAL", "").strip().lower() in ("1", "true", "on", "yes"))
        self._dg_g = int(sparse_index_g); self._dg_G = int(sparse_index_G); self._dg_c = int(sparse_index_c)
        self._dg_conf_floor = float(sparse_index_conf_floor)   # shard peak < floor*D -> escalate to the full scan
        self._dg_index = None          # the DGSparseIndex over the concept codebook (lazy; reuse-by-import)
        self._dg_codebook = None       # (V, D) concept phase-matrix aligned to self.words (fractional-cycle phases)
        self._dg_built_V = -1          # len(self.words) the current index/codebook were built for (rebuild on change)
        # enable_fact_shard (SUBLINEAR RETRIEVAL over the FACT-COUNT axis k_max; board rank-1 composer-latency
        # residual; ADDITIVE + DEFAULT-OFF = BYTE-IDENTICAL to today when off, incl. the numpy-CPU + test-oracle
        # path): route the who/what recall's (agent,action)/(action,patient) cue-match-and-first-match through a
        # DG-CA3 sparse index over the stored FACT BLOCKS, so a query decodes O(shard) blocks instead of O(k_max)
        # -- the missing axis `enable_sparse_index` does NOT close (that index shards the VOCABULARY axis V; its
        # `_read_blocks_indexed` STILL loops every fact block). This is the lever that retires the O(k_max) linear
        # scan (~149 s / recall @ 404 co-resident facts, why k_max was pinned at 32). MECHANISM (de-risked GO 6/6
        # @ 404 facts, 2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md): a per-role
        # DGSparseIndex over the CONCEPT CODES of the blocks' role fillers (encoding-time role knowledge -- the
        # fact's roles are known when stored). A query routes each asserted cue role's CLEAN word code to its DG
        # shard of candidate blocks and INTERSECTS the per-role shards (conjunctive cue); the intersection tightens
        # the shard to ~1 even with reused fillers AND gives the moat for free (an out-of-store valid-word combo
        # intersects to the empty set -> abstain). Only the shard blocks are decoded, via the composer's EXISTING
        # per-block spiking read (`_read_one_block` -> `_read_block`/`_read_block_indexed`: reconstruct + unbind +
        # cleanup on FIRING NEURONS -- the CA3 completion, restricted to the routed ensemble), first-match ascending
        # (== the full scan's first-match), answer read OFF the spiking decode (never off `kb`). The SHARD is a
        # SUPERSET of the true matches BY CONSTRUCTION (block i with the cued filler in role r routes to the SAME
        # bucket its filler was stored in), so parity with the full O(k_max) scan is exact (measured 6/6, 540/540).
        # BRAIN-BASED: the in-shard cleanup IS the composer's on-substrate op (unchanged, over fewer blocks); the DG
        # sparse PROJECTION is the SAME declared host-rate stand-in `enable_sparse_index` uses (research/biology/
        # dg-ca3-sparse-index.md; named spiking burn-down = the granule-cell WTA in the trisynaptic-loop probes).
        # Env BRAIN_FACT_SHARD_RETRIEVAL=1 flips it on without a code change (owner reviews the default-on flip
        # separately -- leave it OFF). GATED (like `enable_sparse_index`) to the host first-match regime
        # (integrated_loop off) + confidence_gate==0 (shard-local margins out of scope) -> otherwise the full path
        # runs, no regression. Determinism: the per-role indices seed from cfg.seed (self.seed). A fact-shard
        # composer never batches, so enabling it ALSO right-sizes the bridge (no_batched_region below), reclaiming
        # the k_max*(n_roles*D+cb) batched region so every per-block read runs at ~its small-store cost.
        self.enable_fact_shard = bool(enable_fact_shard) or (
            _os.environ.get("BRAIN_FACT_SHARD_RETRIEVAL", "").strip().lower() in ("1", "true", "on", "yes"))
        self._fs_g = int(fact_shard_g); self._fs_G = int(fact_shard_G); self._fs_c = int(fact_shard_c)
        self._fact_shard = None         # (per-role DGSparseIndex dict, per-role block-id map); lazy, reuse-by-import
        self._fact_shard_built_K = -1   # len(self.kb) the current fact-shard was built for (rebuild on a store change)
        if self.enable_fact_shard:
            no_batched_region = True    # a fact-shard composer never batches -> drop the dead batched region
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
        # --- runtime vocabulary growth via a RESERVE of uncommitted cleanup slots (recruit-an-assembly) ---
        # A cortex holds a pool of UNCOMMITTED assemblies (adult-born granule cells / silent synapses) that get
        # RECRUITED when a new concept is learned -- it does NOT re-architect on every new word. So reserve
        # `vocab_headroom` blank codebook slots HERE (sized into the layout + bridge below, since cb/n_total cascade
        # from V); runtime word-learning then RECRUITS a free slot (binds the new word to that slot's fixed code)
        # with NO layout/bridge change. Each reserved slot's code is used by BOTH the bind (self.comp.concepts[w])
        # and the cleanup (self.words) once recruited, so decode is consistent. DEFAULT vocab_headroom=0 => V/cb/
        # n_total/the bridge are byte-identical to before (the rf/numpy oracle + every existing test unchanged); the
        # PRODUCTION onebrain chat sets a headroom so a fact taught mid-conversation is laid down + recalled on the
        # spiking store. See 2026-08-12-onebrain-spiking-store... (the wrap-vs-inner codebook bug this closes).
        self._recruit_rng = np.random.default_rng(int(seed) + 90210)
        self._free_slots = []
        for _k in range(max(0, int(vocab_headroom))):
            _ph = f"__free{_k}__"
            if _ph in self.comp.concepts:                # avoid the (astronomically unlikely) name collision
                continue
            self.comp.concepts[_ph] = self._recruit_rng.uniform(0.0, 1.0, self.D)
            self._free_slots.append(len(self.words))
            self.words.append(_ph)
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
        # typed_roles (BURNDOWN C4, the LAST Bucket-A conversion -- the TYPED verb-frame argument-structure surface on the
        # spiking substrate, default None = byte-identical = the flat who/what path). A tuple of TYPED OBLIQUE roles
        # (GOAL/THEME/RECIPIENT/LOCATION/SOURCE/INSTRUMENT/TIME) the bare (agent, action, patient) alphabet cannot
        # express -- the MUC-Memory story (Hagoort: the verb stores its frame; Broca binds the fillers in). When set,
        # each typed role is (a) given a phasor code on the inner composer's `self.comp.roles` from a DISJOINT rng stream
        # (seed+2000, the SAME disjoint-stream discipline ArgStructureComposer + OrderedPositionWM use, so the parent's
        # concept/role codes stay byte-identical), and (b) inserted into `bind_roles` (and so `main_roles`) BEFORE polarity
        # -- preserving the polarity-LAST layout invariant -- so the on-bridge bind/store/unbind/cleanup machinery (which
        # iterates self.bind_roles / self.main_roles uniformly) carries them for free. The binding is role-AGNOSTIC, so
        # adding roles costs only more codebook entries + a wider per-block cleanup region; the per-fact BUNDLE never
        # exceeds the few roles a single verb frame actually realizes (go->agent+action+GOAL=3; give->agent+action+THEME+
        # RECIPIENT=4 -- the same density the flat+attribute path already validates), since _store_composite binds only
        # the roles a fact ACTUALLY has. The typed-role API (`store_fact`/`query_role`/`render`) mirrors the numpy
        # `ArgStructureComposer` oracle; the no-confab moat is the parent's (a cue matching no stored fact -> None).
        # `framecq_seed`/`use_spiking_cq` configure the verb-frame render's serial-order engine (the spiking C1 CQ
        # renderer on GPU / the numpy FrameCQ oracle on CPU -- the consolidated_320 default pattern; only used by render).
        self.typed_roles = tuple(typed_roles) if typed_roles else ()
        base_roles = (["agent", "action", "patient", "attribute"] if self.enable_attributed
                      else ["agent", "action", "patient"])
        self.bind_roles = base_roles + list(self.typed_roles) + ["polarity"]
        self.n_roles = len(self.bind_roles)
        self.main_roles = [r for r in self.bind_roles if r != "polarity"]   # cleaned vs the main vocab
        self.n_main = len(self.main_roles)
        # work registers: fill_0..n-1 (n) + bound_0..n-1 (n) + acc (1) = 2*n+1 D-blocks. Default n=4 -> 9*D (byte-equal).
        self.store_base = self.P + (2 * self.n_roles + 1) * D
        self.block = 1 + D
        self.q_base = self.store_base + self.k_max * self.block      # PER-BLOCK (oracle): n_roles Q regs (one per role)
        self.c_base = self.q_base + self.n_roles * D                 # PER-BLOCK cleanup: n_main V-blocks + 1 NP-block
        self.cb = self.n_main * self.V + self.NP                    # cleanup neurons per block (main roles + polarity)
        # BATCHED region (A5 lever 1): K_max x (n_roles Q regs + cb cleanup) so all blocks read in one pass (additive --
        # the per-block region above is unchanged = the correctness oracle).
        # no_batched_region (SUBLINEAR-RETRIEVAL de-risk, ADDITIVE + DEFAULT-OFF = byte-identical layout when False):
        # skip SIZING the k_max*(n_roles*D + cb) batched region entirely. That region exists ONLY for `_read_all_blocks`
        # (the batched O(k_max) full scan); a SHARDED / sparse-index retrieval that only ever decodes a small candidate
        # shard PER-BLOCK never touches it -- yet its ~k_max*cb neurons dominate n_total (~1.3M @ k_max=420/V=250),
        # inflating the per-step resonate cost of EVERY per-block read (and every store) O(n_total). Dropping it shrinks
        # the bridge to the store region + one per-block op region (~57k @ the same scale, ~23x smaller), so per-block
        # reads run at ~their small-store cost -> a shard of a few blocks recalls in ~FHRR's ~0.9s. When True, the
        # batched read is UNAVAILABLE (enable_batched is forced off below) and `_read_blocks` uses the per-block loop /
        # the sparse-index path; the per-block region [q_base:c_base+cb] is BYTE-UNCHANGED, so every per-block read
        # (`_read_block`, `_read_block_indexed`, `_decode_clause`, `_block_role_scores`) is bit-for-bit identical.
        self.no_batched_region = bool(no_batched_region)
        if self.no_batched_region:
            self.enable_batched = False                              # the batched region is not built -> never batch
            self.bat_q_base = self.bat_c_base = self.c_base + self.cb  # (defined for reference; no batched neurons follow)
            self.n_total = self.c_base + self.cb
        else:
            self.bat_q_base = self.c_base + self.cb
            self.bat_c_base = self.bat_q_base + self.k_max * self.n_roles * D
            self.n_total = self.bat_c_base + self.k_max * self.cb
        self.b = build_coresident_bridge(seed, self.n_total, enable_rf_cudagraph=self.enable_rf_cudagraph)
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=0)   # wires+trains [0:P]
        self.rf_mask = np.zeros(self.n_total, dtype=bool); self.rf_mask[self.P:self.n_total] = True
        # _rf_reset_mask (consolidation / co-residence, DEFAULT None = byte-identical = the full-array zero the private-
        # bridge path uses): the slice the per-op `v/u <- 0` reset is restricted to. None -> the whole bridge is zeroed
        # (the standalone composer owns its bridge, so a full reset is correct + byte-identical). A CO-RESIDENT subclass
        # (CoResidentOneBrainComposer) sets this to its rf slice so a composer op zeroes ONLY the rf slice and leaves a
        # co-resident Izhikevich (nav) slice's v/u byte-untouched (the same masked-rf-kick discipline). Routed through
        # `_zero_rf_v_u()` at every reset site.
        self._rf_reset_mask = None
        self.kb = []          # bookkeeping: list of (fact_dict, None) -- the agent's _assoc_graph reads fact dicts;
        #                       the bound VECTOR is on-substrate (the None placeholder keeps the (fact, vec) shape)
        self.store_conns = []
        self._word_index = {w: i for i, w in enumerate(self.words)}   # word -> codebook index (the sequencer cue idx)
        # (C4) register the TYPED-ROLE phasors on the inner composer's role codebook, from a DISJOINT rng stream
        # (seed+2000 -- == ArgStructureComposer, so the parent's concept/role codes are byte-identical). The read path
        # (_read_block / _read_all_blocks) unbinds + cleans up every role in self.bind_roles, so the typed roles need a
        # code on self.comp.roles. A typed role's filler is decoded against the SAME main vocab as the patient.
        if self.typed_roles:
            prng = np.random.default_rng(int(seed) + 2000)
            for r in self.typed_roles:
                self.comp.roles[r] = prng.uniform(0.0, 1.0, self.D)
        # (C4) the verb-frame render's serial-order engine config (only used by render(); lazily built). framecq_seed
        # defaults to seed; use_spiking_cq follows the consolidated_320 pattern (spiking CQ on GPU / numpy FrameCQ oracle
        # on CPU) when None. Imported lazily in render() so the flat-only path never touches argstructure_composer.
        self._framecq_seed = int(seed) if framecq_seed is None else int(framecq_seed)
        if use_spiking_cq is None:
            from sim.backend import is_gpu_backend
            use_spiking_cq = bool(is_gpu_backend())
        self.use_spiking_cq = bool(use_spiking_cq)
        # (B-mine-1 deploy) the verb-frame lexicon render() recalls/orders through. DEFAULT None -> the module-level
        # hand FRAME_LEXICON (byte-identical to the prior behaviour, so every existing caller + the C4 default path is
        # unchanged). Pass a same-shaped dict (the CORPUS-MINED frame lexicon) to render/recall through ACQUIRED frames;
        # frame_for/frame_id/realized_units + the numpy FrameCQ all take a `lexicon=` (the spiking SpikingFrameCQ is
        # frame-agnostic -- it orders the realized-index list -- so it needs no lexicon). The typed roles bind via
        # bind_roles regardless; only render's frame SHAPE (which roles + lead scaffold) follows _frames.
        self._frames = frame_lexicon
        self._frame_cq = None          # numpy FrameCQ oracle (lazy)
        self._spiking_cq = None        # spiking CQ renderer (lazy)

    def _zero_rf_v_u(self):
        """Reset the RF complex state v/u to 0 before a kick. DEFAULT (_rf_reset_mask=None): the whole bridge (the
        standalone composer owns its bridge -> byte-identical to the prior `b.cp_*[:] = 0.0`). CO-RESIDENT
        (_rf_reset_mask = the rf slice): zero ONLY the rf slice, so a composer op leaves a co-resident Izhikevich (nav)
        slice's v/u byte-untouched (the masked-rf-kick co-residence guarantee). The single dispatch every per-op reset
        site shares so co-residence isolation holds across all of them at once."""
        b = self.b
        m = self._rf_reset_mask
        if m is None:
            b.cp_membrane_potential_v[:] = 0.0
            b.cp_recovery_variable_u[:] = 0.0
        else:
            b.cp_membrane_potential_v[m] = 0.0
            b.cp_recovery_variable_u[m] = 0.0

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
        # RECRUIT any never-seen main-role filler into a reserved cleanup slot BEFORE binding, so the bind and the
        # cleanup use the SAME code. Without this the new word's code lives only on the inner comp while the outer
        # cleanup codebook (self.words, copied once at construction) stays blind to it -> the taught fact STORES but
        # never RECALLS (the wrap-vs-inner codebook bug). No-op when vocab_headroom=0 (the pool is empty).
        recruited = False
        for _r in self.main_roles:
            _v = fact.get(_r)
            if isinstance(_v, str):
                recruited = self._recruit_word(_v) or recruited
        if recruited:
            self._csr_cache.clear()                          # the cleanup operator changed -> rebuild on next batched read
        roles = [r for r in self.bind_roles if r in fact]       # bind only present roles, in canonical order
        self._store_composite([fact[r] for r in roles], roles)
        return fact

    def _recruit_word(self, w):
        """Recruit a never-seen concept into a reserved (uncommitted) cleanup slot -- the biological recruit-an-
        assembly (a pool of uncommitted assemblies is claimed when a new concept is learned). The word takes a free
        slot's FIXED code as BOTH its bind code (self.comp.concepts[w]) and its cleanup code (self.words[slot]); if
        the inner comp already allocated a code for w, that code is reused (so any fact already bound with it stays
        consistent). No layout/bridge change (the slot was pre-allocated by vocab_headroom). Returns True iff a
        recruit happened (caller clears the cleanup cache). A word already in the cleanup codebook, or an exhausted
        pool, is a no-op (False)."""
        if not isinstance(w, str) or w in self._word_index:
            return False                                     # already decodable by the outer cleanup
        if not self._free_slots:
            return False                                     # pool exhausted (headroom too small) -> stays a decode miss
        s = self._free_slots.pop(0)
        ph = self.words[s]
        reserve_code = self.comp.concepts.pop(ph, None)
        code = self.comp.concepts.get(w)
        if code is None:
            code = reserve_code if reserve_code is not None else self._recruit_rng.uniform(0.0, 1.0, self.D)
            self.comp.concepts[w] = code
        if w not in self.comp.words:
            import bisect; bisect.insort(self.comp.words, w)
        self.words[s] = w                                    # the cleanup slot now decodes to the real word
        self._word_index.pop(ph, None)
        self._word_index[w] = s
        self._dg_index = None; self._dg_built_V = -1         # codebook mutated -> rebuild the DG index on next read
        return True

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

    # --- (C4) TYPED VERB-FRAME ARGUMENT-STRUCTURE API on the spiking substrate (== ArgStructureComposer numpy oracle) ---
    def store_fact(self, fact):
        """Store a TYPED argument-structure fact dict on the spiking substrate, e.g.
        {'agent':'boy','action':'go','GOAL':'park'} or {'agent':'girl','action':'give','THEME':'ball','RECIPIENT':'dog'}.
        Binds + bundles + writes the persistent store block exactly like store()/hear() -- but over the TYPED roles
        present in the fact (the few-role bundle a verb frame realizes). Requires the composer to have been built with
        the relevant typed_roles. Mirrors ArgStructureComposer.store_fact; the no-confab moat is the parent's. The bind
        binds ONLY the roles the fact ACTUALLY has (in self.bind_roles canonical order), so a go-fact is a 3-way bundle
        and a give-fact a 4-way bundle -- never the full 10-role density. polarity defaults to AFFIRM (so ask_yes_no /
        the flat read still work on a typed fact)."""
        f = dict(fact)
        f.setdefault("polarity", "AFFIRM")
        f["polarity"] = self._pol(f["polarity"])
        roles = [r for r in self.bind_roles if r in f]                  # bind only present roles, canonical order
        unknown = [r for r in f if r not in self.bind_roles and r != "polarity"]
        if unknown:
            raise KeyError(f"store_fact: role(s) {unknown} not in bind_roles {self.bind_roles}; "
                           f"build OneBrainComposer(typed_roles=(...)) with them")
        self._store_composite([f[r] for r in roles], roles)
        self.kb.append((f, None))
        return f

    def query_role(self, role, **cue_roles):
        """Recall the filler of `role` (any typed role, agent/action/patient too) from the FIRST stored fact whose cue
        roles ALL match; None = abstain (the no-confab moat). The on-bridge spiking read (`_read_blocks`) reconstructs
        every block + unbinds + cleans up ALL roles in PARALLEL; this scans the decoded {role: word} rows for the first
        whose cue roles match and returns the requested role's decoded word. Generalizes query_patient/query_agent to
        ANY typed role. == ArgStructureComposer.query_role; the SELECTION + decode are on FIRING NEURONS (the substrate
        store + the resonate scan/unbind/cleanup). An unanswerable role (None decoded / not in any matching fact's
        bound roles) abstains."""
        for i, got in enumerate(self._read_blocks()):
            if all(got.get(cr) == cv for cr, cv in cue_roles.items()):
                # the role the caller wants -- but only if THIS fact actually bound it (else its decoded word is the
                # unbind of an unbound role = noise -> abstain, never confabulate a role the fact does not have).
                if role in self.bind_roles and (i >= len(self.kb) or role in self.kb[i][0]):
                    return got.get(role)
                return None
        return None

    def _composite_for_typed(self, fact):
        """The kb index of the stored fact whose agent (+ action) matches `fact` -- for render(). The composite itself
        lives on the substrate (read via _read_blocks); we return the BLOCK INDEX (the spiking read decodes it)."""
        for i, (f, _) in enumerate(self.kb):
            if f.get("agent") == fact.get("agent") and f.get("action") == fact.get("action"):
                return i
        return None

    def _ordering_engine(self):
        """The verb-frame render's serial-order engine: the validated SPIKING competitive-queuing renderer (C1
        SpikingFrameCQ, real firing rates) when use_spiking_cq, else the numpy FrameCQ oracle. Lazily built (each
        constructs/loads its mechanism) + cached. Imported here so the flat-only path never imports argstructure."""
        from research.runners.argstructure_composer import FrameCQ, SpikingFrameCQ
        if self.use_spiking_cq:
            if self._spiking_cq is None:
                self._spiking_cq = SpikingFrameCQ(seed=self._framecq_seed)   # frame-agnostic (orders the index list)
            return self._spiking_cq
        if self._frame_cq is None:
            # (B-mine-1) the numpy FrameCQ teaches a per-frame primacy gradient -> it must see the ACTIVE lexicon
            # (mined or hand) so a mined frame's slot order is taught; lexicon=None reuses the hand FRAME_LEXICON.
            self._frame_cq = FrameCQ(seed=self._framecq_seed, lexicon=self._frames)
        return self._frame_cq

    def render(self, fact, comp=None, ablate_closed_class=False, use_framecq=True):
        """Render a TYPED fact as prose via its verb frame -- e.g. {'agent':'boy','action':'go','GOAL':'park'} ->
        'the boy goes to the park'. The frame's closed-class scaffold (determiner 'the' + preposition 'to'/'on') comes
        from the FRAME LEXICON; the CONTENT words are DECODED FROM THE ON-BRIDGE UNBIND (the spiking read, not the
        stored labels); the content slots are ordered by the validated serial-order engine (the spiking C1 CQ renderer
        on GPU / the numpy FrameCQ oracle on CPU). `ablate_closed_class=True` -> telegraphic 'boy go park' (the Broca's
        agrammatism anti-cheat). The no-confab moat: a fact with no matching stored block -> None (no fabricated
        sentence). == ArgStructureComposer.render, on FIRING NEURONS. `comp` is ignored on the substrate path (the
        composite lives on the bridge) -- accepted for API parity with the numpy oracle's render(fact, comp)."""
        from research.runners.argstructure_composer import (
            frame_for, frame_id, realized_units, TENSE_3SG)
        idx = self._composite_for_typed(fact)
        if idx is None:
            return None                                    # moat: no stored composite -> no fabricated sentence
        decoded = self._read_blocks()[idx]                 # the on-bridge spiking decode of every role
        verb = fact["action"]
        # (B-mine-1) render through the ACTIVE frame lexicon (mined or hand); lexicon=None reuses the hand FRAME_LEXICON.
        units = realized_units(verb, fact, lexicon=self._frames)   # only the units whose role is present in the fact
        full_frame = frame_for(verb, lexicon=self._frames)
        if use_framecq:
            engine = self._ordering_engine()
            unit_to_idx = {id(u): i for i, u in enumerate(full_frame)}
            realized_idx = [unit_to_idx[id(u)] for u in units]
            order = engine.emit_order(frame_id(verb, lexicon=self._frames), realized_idx)
            idx_to_unit = {unit_to_idx[id(u)]: u for u in units}
            ordered_units = [idx_to_unit[j] for j in order]
        else:
            ordered_units = units
        toks = []
        for kind, role, lead in ordered_units:
            if not ablate_closed_class:
                toks.extend(lead)                          # the unit's closed-class scaffold (det / prep)
            if kind == "TENSE":
                bare = decoded.get("action")               # decoded bare verb (from the on-bridge unbind)
                toks.append(bare if ablate_closed_class else TENSE_3SG.get(bare, bare))
            else:
                toks.append(decoded.get(role))             # the role's decoded filler
        return " ".join(toks)

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
        self._zero_rf_v_u()
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
        self._persistent_dirty = True  # (persistent_store) store_conns changed -> the device store synapses are stale
        if self.integrated_loop:
            self._seq_dirty = True     # (shortcut #3) the store changed -> the per-block sequencer drives are stale
            if self._fused:
                self._fused_dirty = True   # (R1) the store changed -> the per-block FUSED device-resident drives are stale

    # --- Turrigiano homeostatic synaptic scaling on the substrate store (2026-08-25, opt-in, byte-identical-off) ------
    def _measure_block_readout(self, block_idx):
        """SENSE an engram's postsynaptic readout activity ON THE SUBSTRATE: kick the block's trigger, resonate, and read
        the mean |Z| over its D readout neurons (trig+1..trig+D) off the bridge membrane (cp_membrane_potential_v /
        cp_recovery_variable_u). This is a genuine neural read of the engram's synaptic drive -- CONSTANT for a unit
        write (independent of the fact's phase pattern) and LINEAR in the encoding strength (a g-scaled block reads g*A).
        The quantity Turrigiano homeostasis regulates. Read-only w.r.t. the store (restores nothing; the caller owns the
        store_conns it measured)."""
        b, Pd, D = self.b, self.period, self.D
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        re = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        im = np.asarray(to_host(b.cp_recovery_variable_u)).astype(float)
        idx = slice(trig + 1, trig + 1 + D)
        return float(np.sqrt(re[idx] * re[idx] + im[idx] * im[idx]).mean())

    def _homeo_setpoint(self):
        """The homeostatic ACTIVITY SET-POINT A* = the readout activity of a UNIT-magnitude engram (the intrinsic
        functional level: a unit composite kicked through a unit trigger->readout fan-out). Measured on the substrate
        from a unit-NORMALIZED reference of block 0's store synapses (so A* is independent of the stored DA distribution
        -> a tonic/unit fact maps to s=1 -> byte-safe). Temporarily unit-normalizes block 0, resonates, restores."""
        if not self.store_conns:
            return 0.0
        D = self.D
        saved = list(self.store_conns[0:D])
        mags = [abs(complex(w)) for (_p, _q, w) in saved]
        mean_mag = (sum(mags) / len(mags)) if mags else 0.0
        if mean_mag <= 0.0:
            return 0.0
        self.store_conns[0:D] = [(p, q, complex(w) / mean_mag) for (p, q, w) in saved]
        self._store_dirty = True; self._store_csr = None
        a_star = self._measure_block_readout(0)
        self.store_conns[0:D] = saved                              # restore the real (DA-gated) block 0
        self._store_dirty = True; self._store_csr = None
        return a_star

    def apply_homeostatic_scaling(self):
        """TURRIGIANO 2008 multiplicative homeostatic synaptic scaling on the substrate store synapses -- the
        on-substrate SYNAPTIC realization that REPLACES the host-proxy DA-encoding homeostat. For each stored engram:
        SENSE its readout activity A_i on the substrate (`_measure_block_readout`), then multiplicatively rescale its
        store synapses toward the set-point A* (`_homeo_setpoint`):
          * WEAK engram (A_i <= A*, e.g. a low-DA fact the DA gate wrote at g<1): s = min(s_max, A*/A_i) -> scaled UP to
            the functional set-point == the recall-safe FLOOR, now EMERGENT from measured activity (no host g_floor clip).
          * STRONG engram (A_i > A*): s = max(s_min, (A*/A_i)**beta_down), beta_down<1 -> a PARTIAL down-regulation that
            PRESERVES the relative DA-salience ORDER (all strong engrams keep their ranking) while pulling the extreme
            toward A* (Turrigiano's runaway-prevention half).
        The sensed variable is postsynaptic activity; the actuator is the synaptic weight -> a faithful homeostatic
        synaptic-scaling rule (multiplicative, activity-set-point, relative-strength-preserving), NOT host arithmetic on
        the DA reading. Rewrites store_conns in place (a real synaptic-weight change), busts the store CSR. Idempotent
        input-wise (call once after the fact battery is stored). Guarded by the caller (only when self.homeostatic_scaling)
        -> never called == store byte-identical. Returns the applied per-engram scale vector."""
        m = len(self.kb)
        if m == 0 or not self.store_conns:
            self._homeo_scales = []
            return []
        D = self.D
        a_star = self._homeo_setpoint()
        scales = []
        for i in range(m):
            a_i = self._measure_block_readout(i)
            if a_i <= 0.0 or a_star <= 0.0:
                s = 1.0
            else:
                ratio = a_star / a_i
                if ratio >= 1.0:                                  # weak engram -> full homeostatic restoration (floor)
                    s = min(self.homeo_s_max, ratio)
                else:                                             # strong engram -> partial, order-preserving down-reg
                    s = max(self.homeo_s_min, ratio ** self.homeo_beta_down)
            scales.append(float(s))
            self.store_conns[i * D:(i + 1) * D] = [
                (p, q, complex(s) * complex(w)) for (p, q, w) in self.store_conns[i * D:(i + 1) * D]]
        self._store_dirty = True; self._store_csr = None; self._persistent_dirty = True
        if getattr(self, "_csr_cache", None) is not None:
            self._csr_cache = {}
        if self.integrated_loop:
            self._seq_dirty = True
            if getattr(self, "_fused", False):
                self._fused_dirty = True
        self._homeo_scales = scales
        return scales

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

    def _dev_rekick_into(self, dst_slices):
        """BURNDOWN I-1 (op-handoff-as-spikes): the ON-SUBSTRATE read-phase + re-kick that REPLACES a host round-trip
        `to_host(rf_read_phases()) -> np.exp -> rf_kick`. Recover each RF neuron's phase from the device spike-step
        tracker with the SAME integer formula `rf_read_phases` uses (`((period - spike_step) % period)/period`), install
        a clean unit phasor `exp(2pi i phi)` into each register in `dst_slices`, and reset the RF trackers exactly as
        `rf_kick` does (counter=0, prev_im=u, fired=False, spike_step=period) -- ALL device ops, with NO `to_host` of the
        phasor value. Byte-identical to the host round-trip (I-1-a de-risk `_burndown_I1a_op_handoff_probe`: max|dphase|
        = 0.0 over 9 cases): the phase computation, the complex exp, and the writeback are device ops on the SAME float32
        membrane the host path casts to, so the quantize-to-spike-grid + the unit-normalize match the host path exactly.
        The caller then proceeds with `rf_set_complex_weights(...)` + `rf_resonate_steps(...)` (the re-kicked op),
        replacing the host `rf_kick` whose only job was to normalize+quantize+reset before that op."""
        xp, _name = get_backend()
        b, period, n = self.b, int(self.period), self.n_total
        ss = b.cp_rf_spike_step                                       # device int (per neuron, set by the prior resonate)
        phi_dev = ((period - ss) % period) / float(period)           # device phases (the rf_read_phases formula)
        zc = xp.exp(2j * np.pi * phi_dev)                            # device clean unit phasor (the np.exp host uses)
        self._zero_rf_v_u()        # full reset (private bridge) OR rf-slice-only (co-resident) -- byte-identical default
        for sl in dst_slices:
            b.cp_membrane_potential_v[sl] = xp.real(zc[sl]).astype(b.cp_membrane_potential_v.dtype)
            b.cp_recovery_variable_u[sl] = xp.imag(zc[sl]).astype(b.cp_recovery_variable_u.dtype)
        # rf_kick's global tracker resets (counter=0, prev_im=u, fired=False, spike_step=period):
        b._rf_counter = 0
        b.cp_rf_prev_im = b.cp_recovery_variable_u.copy()
        b.cp_rf_fired = xp.zeros(n, dtype=bool)
        b.cp_rf_spike_step = xp.full(n, period, dtype=xp.int64)

    def _loop_rekick(self, dst_slices):
        """The persistent-loop op-handoff hook: when `persistent_loop` is ON, re-kick a CLEAN UNIT PHASOR into the
        unbound Q register(s) `dst_slices` via `_dev_rekick_into` (the I-1-a on-substrate read-phase + re-kick, no host
        round-trip) BEFORE the cleanup op; when OFF, a no-op (the carry-live-Z default = byte-identical to today). The
        single dispatch the flat read sites (`_read_block`, both `_read_all_blocks` sub-paths) share so the flag toggles
        all of them at once. Byte-identical to a host round-trip on the cleanup membrane; answer-identical to the carry
        default (the cleanup argmax is invariant to the common register magnitude this normalizes)."""
        if self.persistent_loop:
            self._dev_rekick_into(dst_slices)

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
            w = self._spiking_select(scores, words)
        else:
            w = words[int(np.argmax(np.asarray(scores, dtype=float)))]
        # A reserved (unrecruited) cleanup slot must never surface as a decoded role word: an UNBOUND role's noise can
        # argmax onto a "__free*__" placeholder -> report None (the render already ignores an unbound role; this also
        # keeps the owner-visible activity trace clean). A recruited word was renamed to the real word, so it is never
        # masked. No-op at vocab_headroom=0 (no "__free*__" words exist) => byte-identical.
        return None if (isinstance(w, str) and w.startswith("__free")) else w

    def _read_block(self, block_idx):
        """Reconstruct block_idx + unbind all roles IN PARALLEL (one settle, no phase drift). The main roles (agent,
        action, patient, +attribute when enabled) clean up against the main vocab; the polarity role cleans up against
        the 2-word polarity codebook. Returns a dict {role: word} for the bind_roles (attribute present only on the
        attribute-enabled composer; its value is noise for a plain fact and the caller ignores it via the kb dict)."""
        comp, b, D, Pd, V, NP = self.comp, self.b, self.D, self.period, self.V, self.NP
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for ri, role in enumerate(self.bind_roles):
            zc = self._unbind_conj(role)
            unbind += [(self.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        # (persistent_loop) op-handoff-as-spikes: re-kick the unbound Q registers as clean unit phasors before cleanup
        # (== a host round-trip, no `to_host`; no-op when OFF = the carry-live-Z default). The Q regs are the contiguous
        # run [q_base : q_base + n_roles*D].
        self._loop_rekick([slice(self.q_base, self.q_base + self.n_roles * D)])
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

    def _sync_persistent_store(self):
        """(persistent_store) install the fact store into the DEVICE synapses (cp_rf_store_re/im) via
        rf_set_store_weights, once per store mutation. The store's readout rows (store_base+i*block+1..+D) are DISJOINT
        from the op operators' rows (Q at bat_q_base.., cleanup at bat_c_base..), so it never clobbers the per-op bind.
        A per-op rf_set_complex_weights/rf_kick never touches cp_rf_store_* -> the store persists across binds."""
        if self.persistent_store and self._persistent_dirty and self.store_conns:
            self.b.rf_set_store_weights(self.store_conns)
            self._persistent_dirty = False

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
            self._zero_rf_v_u()
            kick = np.zeros(self.n_total, dtype=np.complex128)
            for i in range(n):
                kick[self.store_base + i * self.block] = 1.0                   # fire EVERY stored trigger
            if self.persistent_store:
                self._sync_persistent_store()                                 # store lives in cp_rf_store_* (device)
                b.cp_rf_w_re = b.cp_rf_w_im = None                             # settle: only the persistent store drives
            else:
                b.cp_rf_w_re, b.cp_rf_w_im = Sre, Sim                          # install the cached store operator (staged)
            b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
            b.rf_resonate_steps(Pd + 8)
            b.cp_rf_w_re, b.cp_rf_w_im = Ure, Uim; b.rf_resonate_steps(Pd + 8)  # unbind (persistent store keeps driving readouts)
            # (persistent_loop) op-handoff-as-spikes: re-kick the active batched Q run as clean unit phasors before the
            # cleanup operator install (no-op when OFF). _dev_rekick_into touches only v/u + the RF trackers, never the
            # cleanup CSR installed on the next line.
            self._loop_rekick([slice(self.bat_q_base, self.bat_q_base + n * self.n_roles * D)])
            b.cp_rf_w_re, b.cp_rf_w_im = Cre, Cim; b.rf_resonate_steps(1)       # cached cleanup
            mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
            return self._decode_batched_mem(mem, n)
        # --- stock path (cache off): rebuild every CSR from fresh tuple lists each query ---
        self._zero_rf_v_u()
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(n):
            kick[self.store_base + i * self.block] = 1.0                       # fire EVERY stored trigger
        if self.persistent_store:
            self._sync_persistent_store()                                      # store lives in cp_rf_store_* (device)
            b.cp_rf_w_re = b.cp_rf_w_im = None                                  # settle: only the persistent store drives
        else:
            b.rf_set_complex_weights(self.store_conns)                          # staged: re-install store onto cp_rf_w_*
        b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
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
        # (persistent_loop) op-handoff-as-spikes: re-kick the active batched Q run as clean unit phasors before cleanup
        # (no-op when OFF = the carry-live-Z default).
        self._loop_rekick([slice(self.bat_q_base, self.bat_q_base + n * nr * D)])
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

    # --- (#150 knowledge-scale) DG-INDEXED cleanup fast path: route the cue -> shard -> cleanup over shard rows -----
    def _ensure_dg_index(self):
        """Build (lazily, once per codebook mutation) the DG-like sparse index + the concept phase-matrix over the
        CURRENT cleanup codebook `self.words`. Reuse-by-import of the validated de-risk (research/runners/
        _sparse_indexed_retrieval_derisk.py, 6-seed GO; import deferred so a default-off composer never imports it).
        The index KEY is computed from the concept feature vector via the DG sparse projection -- NEVER from an answer
        id (content-addressable, anti-cheat a). Rebuilt only when len(self.words) changed (a recruit/grow), so the moat
        is never served a stale index. m ~ V^(1/g) keeps bucket occupancy O(1) -> the shard stays ~constant as V
        grows. Seeds from self.seed (== cfg.seed determinism)."""
        V = len(self.words)
        if self._dg_index is not None and self._dg_built_V == V:
            return
        from research.runners._sparse_indexed_retrieval_derisk import DGSparseIndex
        # concept codebook aligned to self.words, in FRACTIONAL-CYCLE phases (the composer's convention; the phasor is
        # exp(2pi i phase)). The de-risk index's feature convention is [cos(phase_rad), sin(phase_rad)], so build/query
        # in RADIANS (phase * 2pi).
        cb = np.stack([np.asarray(self.comp.concepts[w], dtype=float) for w in self.words])   # (V, D) fractional-cycle
        m = max(2, int(np.ceil(V ** (1.0 / self._dg_g))))
        idx = DGSparseIndex(D=self.D, m=m, g=self._dg_g, G=self._dg_G, c=self._dg_c, seed=self.seed)
        idx.build(cb * (2.0 * np.pi))       # store each concept's band-winner conjunctive key -> bucket (id = word idx)
        self._dg_index = idx
        self._dg_codebook = cb
        self._dg_built_V = V

    def _dg_shard_select(self, rec_phases):
        """Route the recovered role phasor (fractional-cycle phases) to its DG shard and decode it by the matched
        filter over ONLY the shard concepts. Returns (word, peak_score), or (None, peak) to signal ESCALATE-to-full
        when the shard is empty or its peak is not decisive (< conf_floor*D) -- the no-regression fallback. The score
        is the composer's own on-substrate matched filter over fewer rows: score_w = sum_k cos(2pi(rec - code_w))
        (== rf_phasor_composer.py:662; equals Re(conj(code)*rec) for unit phasors, the bridge cleanup argmax under the
        default persistent_loop unit-normalization)."""
        self._ensure_dg_index()
        rec = np.asarray(rec_phases, dtype=float)
        shard = self._dg_index.query(rec * (2.0 * np.pi))          # candidate word indices (the routed CA3 ensemble)
        if shard.size == 0:
            return None, 0.0
        cb = self._dg_codebook[shard]                              # (s, D) shard concept phases
        sc = np.cos(2.0 * np.pi * (rec[None, :] - cb)).sum(axis=1)  # (s,) matched-filter cleanup over the shard
        t = int(np.argmax(sc)); peak = float(sc[t])
        if peak < self._dg_conf_floor * self.D:
            return None, peak                                     # not decisive -> escalate to the full scan (parity)
        return self.words[int(shard[t])], peak

    def _full_host_select(self, rec_phases):
        """Full-codebook host matched-filter argmax over the recovered role phasor (the escalation path == the full
        scan). Same operator as `_dg_shard_select`, over ALL V rows. Used only when a role's shard read is not
        decisive, so the decoded word is IDENTICAL to the full cleanup."""
        rec = np.asarray(rec_phases, dtype=float)
        sc = np.cos(2.0 * np.pi * (rec[None, :] - self._dg_codebook)).sum(axis=1)
        return self.words[int(np.argmax(sc))]

    def _read_block_indexed(self, block_idx):
        """DG-indexed variant of `_read_block`: reconstruct + unbind (IDENTICAL machinery), read each role's RECOVERED
        phasor from its Q register (rf_read_phases, == `_recovered_patient_phases`), and decode each MAIN role by
        ROUTING to a DG shard + matched-filter cleanup over the shard rows -- falling back to the full-codebook host
        cleanup on a non-decisive role, so the decode is identical to the full scan. Polarity (a 2-word codebook) is
        decoded directly (no index needed). Only reached when enable_sparse_index (and confidence_gate==0); the full
        read path stays byte-unchanged. NO V-wide cleanup wiring/resonate is built -> O(shard) not O(V) at scale."""
        b, D, Pd = self.b, self.D, self.period
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for ri, role in enumerate(self.bind_roles):
            zc = self._unbind_conj(role)
            unbind += [(self.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        recovered = np.asarray(b.rf_read_phases())                 # fractional-cycle phases over the whole bridge
        out = {}
        for ri, role in enumerate(self.main_roles):
            rec = recovered[self.q_base + ri * D: self.q_base + (ri + 1) * D]
            w, _pk = self._dg_shard_select(rec)
            out[role] = w if w is not None else self._full_host_select(rec)   # escalate on a non-decisive role
        pol_ri = self.bind_roles.index("polarity")
        rec_p = recovered[self.q_base + pol_ri * D: self.q_base + (pol_ri + 1) * D]
        pol_cb = np.stack([np.asarray(self.comp.concepts[w], dtype=float) for w in self.pol_words])   # (NP, D)
        psc = np.cos(2.0 * np.pi * (rec_p[None, :] - pol_cb)).sum(axis=1)
        out["polarity"] = self.pol_words[int(np.argmax(psc))]
        return out

    def _read_blocks_indexed(self):
        """All stored blocks decoded via the DG-indexed per-block read (the knowledge-scale fast path)."""
        return [self._read_block_indexed(i) for i in range(len(self.kb))]

    def _read_blocks(self):
        """All stored blocks as {role: word} dicts. DEFAULT: the BATCHED read (A5 lever 1) or the per-block loop (the
        oracle). When enable_sparse_index (and confidence_gate==0, whose shard-local margins are out of scope), the
        DG-INDEXED per-block read routes each role's cleanup through its sparse shard (O(shard) not O(V)). Each dict
        has agent/action/patient/polarity (+attribute on the attribute-enabled composer)."""
        if self.enable_sparse_index and self.confidence_gate == 0.0:
            return self._read_blocks_indexed()
        if self.enable_batched:
            return self._read_all_blocks()
        return [self._read_block(i) for i in range(len(self.kb))]

    # --- (FACT-COUNT-axis sparse retrieval) DG-CA3 shard over the stored fact blocks; default OFF = byte-identical ---
    def _fact_shard_active(self):
        """True when the FACT-COUNT-axis sharded fast path should serve a cue-known recall: enabled, in the host
        first-match regime (integrated_loop off -- the de-risk's validated envelope; the spiking K-way sequencer is
        a separate, orthogonal selection mechanism), confidence_gate off (its shard-local margins are out of scope,
        the `enable_sparse_index` precedent), and a non-empty store. Default-off -> always False -> byte-identical."""
        return (self.enable_fact_shard and not self.integrated_loop
                and self.confidence_gate == 0.0 and len(self.kb) > 0)

    def _ensure_fact_shard(self):
        """Build (lazily, once per store mutation) a per-role DG-CA3 sparse index over the stored FACT BLOCKS: for
        each main role r in (agent, action, patient), a `DGSparseIndex` over the concept codes of the blocks whose
        role-r filler is a stored WORD (a clause / non-word filler is SKIPPED for that role -- it can never match a
        word cue, so skipping it changes no answer). The bucket-member id maps back to the block index via a per-role
        id list. Reuse-by-import of the validated `DGSparseIndex` (the SAME class + math `enable_sparse_index` and
        the `FactShardIndex` de-risk use, 6-seed GO); on the flat-SVO store the de-risk validated (every block has a
        word filler in each role) this is BYTE-IDENTICAL to `FactShardIndex` (same m, radians convention, per-role
        seed offset, intersection), so its 6/6 parity/moat transfers. m ~ K^(1/g) keeps bucket occupancy O(1) -> the
        shard stays ~constant as the store grows. Rebuilt only when len(self.kb) changed. Seeds from self.seed."""
        K = len(self.kb)
        if self._fact_shard is not None and self._fact_shard_built_K == K:
            return
        from research.runners._sparse_indexed_retrieval_derisk import DGSparseIndex   # reuse-by-import (deferred)
        roles = tuple(r for r in ("agent", "action", "patient") if r in self.main_roles)
        m = max(2, int(np.ceil(max(1, K) ** (1.0 / self._fs_g))))
        idxs, blockids = {}, {}
        for ri, r in enumerate(roles):
            codes, bids = [], []
            for i in range(K):
                filler = self.kb[i][0].get(r)
                if isinstance(filler, str) and filler in self.comp.concepts:      # word filler only (skip clauses)
                    codes.append(np.asarray(self.comp.concepts[filler], dtype=float))
                    bids.append(i)
            if not codes:
                idxs[r] = None; blockids[r] = []
                continue
            idx = DGSparseIndex(D=self.D, m=m, g=self._fs_g, G=self._fs_G, c=self._fs_c,
                                seed=self.seed + 101 * (ri + 1))                   # == FactShardIndex's per-role seed
            idx.build(np.stack(codes) * (2.0 * np.pi))                            # radians convention (== the de-risk)
            idxs[r] = idx; blockids[r] = bids
        self._fact_shard = (idxs, blockids)
        self._fact_shard_built_K = K

    def _fact_shard_candidates(self, cue_roles):
        """Route a CLEAN cue {role: word} to the intersected shard of candidate block indices (ascending). Returns
        [] for an absent cue word or an empty per-role index (moat: no block -> abstain), and None to signal 'cannot
        route -> escalate to the full scan' (an unindexed cue role). Content-addressable: the key is the cue WORD's
        concept code, never an answer id. == the `FactShardIndex` de-risk's `candidates`, block-id-mapped."""
        self._ensure_fact_shard()
        idxs, blockids = self._fact_shard
        sets = []
        for r, w in cue_roles.items():
            if r not in idxs:
                return None                                        # cue role not indexed (typed/attribute) -> escalate
            if w not in self.comp.concepts:
                return []                                          # absent cue word -> empty shard -> abstain (moat)
            idx = idxs[r]
            if idx is None:
                return []                                          # no block has a word filler in this role -> abstain
            code = np.asarray(self.comp.concepts[w], dtype=float)
            cand = idx.query(code * (2.0 * np.pi))
            bids = blockids[r]
            sets.append(set(int(bids[int(x)]) for x in cand.tolist()))
        if not sets:
            return None
        shard = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return sorted(shard)

    def _read_one_block(self, i):
        """Decode ONE block by the SAME per-block method the full read path uses, so a sharded read is parity-exact
        with the full scan under every flag combo: the DG vocab-shard per-block decode when `enable_sparse_index`
        (+ confidence_gate==0, the two indices compose), else the plain per-block spiking decode `_read_block`
        (== the de-risk's `comp._read_block`)."""
        if self.enable_sparse_index and self.confidence_gate == 0.0:
            return self._read_block_indexed(i)
        return self._read_block(i)

    def _fact_shard_first_match(self, cue_roles):
        """(idx, got) for the FIRST shard block (ascending == the full scan's first-match) whose DECODED cue roles
        all match `cue_roles` (a {role: clean_word} cue); (None, None) to ABSTAIN (empty shard or no shard block
        matches); the sentinel (_FS_ESCALATE, None) when the cue cannot be routed -> the caller uses the full path.
        Decodes ONLY the shard blocks via `_read_one_block` (the on-substrate CA3 completion, restricted to the
        routed ensemble) and reads the answer OFF the spiking decode -- the de-risk's `sharded_query_*` consolidated
        (a fast-but-wrong index is caught: parity vs the full scan was the de-risk's hard GO gate)."""
        shard = self._fact_shard_candidates(cue_roles)
        if shard is None:
            return _FS_ESCALATE, None
        for i in shard:
            got = self._read_one_block(i)
            if all(got.get(r) == w for r, w in cue_roles.items()):
                return i, got
        return None, None

    # --- (B3) READ-ONLY per-turn trace helpers (only invoked on the trace path; default OFF = byte-identical) ---
    def _rf_gauge(self):
        """A scalar RF activity gauge over the rf SLICE of the shared bridge (read after the last query's resonate):
        the fraction of rf-slice readout neurons that crossed (`cp_rf_fired[rf_mask].mean()`) + the mean recovery
        magnitude |Z| = mean(sqrt(re²+im²)) over the rf slice. The parser slice [0:P] is EXCLUDED (rf_mask is the
        composer's readout region). All guarded -> None on absence. Strictly read-only of state the resonate already
        produced (no extra GPU work)."""
        b = getattr(self, "b", None)
        mask = getattr(self, "rf_mask", None)
        out = {"n_readout_neurons": (int(mask.sum()) if mask is not None else None),
               "frac_fired": None, "mean_magnitude": None}
        if b is None or mask is None:
            return out
        try:
            fired = getattr(b, "cp_rf_fired", None)
            if fired is not None:
                fh = np.asarray(to_host(fired)).astype(float)
                out["frac_fired"] = float(fh[mask].mean()) if fh.shape[0] == mask.shape[0] else float(fh.mean())
        except Exception:
            pass
        try:
            re = getattr(b, "cp_membrane_potential_v", None)
            im = getattr(b, "cp_recovery_variable_u", None)
            if re is not None and im is not None:
                re_h = np.asarray(to_host(re)).astype(float); im_h = np.asarray(to_host(im)).astype(float)
                magn = np.sqrt(re_h * re_h + im_h * im_h)
                out["mean_magnitude"] = float(magn[mask].mean()) if magn.shape[0] == mask.shape[0] else float(magn.mean())
        except Exception:
            pass
        return out

    def _block_role_scores(self, block_idx):
        """Read block_idx once + return {role: (decoded_word, confidence, margin, margin_spiking)} for the main
        roles (+polarity). `confidence` = the role's top matched-filter membrane score normalized into [0,1] by
        its row peak (kept for backward compat: this is `s[argmax]/max(s)`, i.e. ALWAYS 1.0 at a non-degenerate
        decode by construction -- it can never discriminate a clean recall from a genuinely ambiguous one, issue
        #181's root cause).
        `margin` = `self._margin(scores)`, the SAME normalized-decisiveness read the `confidence_gate` familiarity
        gate already uses ((peak-runner_up)/peak, 2026-06-18-emergent-graceful-degradation-derisk, multi-seed
        validated: ~0 on a noise-dominated/damaged read, ~0.5+ on an intact confident one, g=0.15 the validated
        clean/noise separator) -- reused here (2026-08-27, issue #181) rather than a new formula, so the metacog
        honesty hedge reads the SAME already-tested decisiveness signal the composer's own moat uses, not a
        fresh untested one. Read-only (mirrors `_read_block`'s per-block matched-filter, but ALSO returns the
        winner's normalized score + margin). Trace-only -- never on the answer path."""
        comp, b, D, Pd, V, NP = self.comp, self.b, self.D, self.period, self.V, self.NP
        self._zero_rf_v_u()
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
        for ri, role in enumerate(self.main_roles):
            for j in range(V):
                cc = self._cleanup_conj(self.words[j])
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        pol_ri = self.bind_roles.index("polarity")
        for j in range(NP):
            cc = self._cleanup_conj(self.pol_words[j])
            clean += [(self.c_base + self.n_main * V + j, self.q_base + pol_ri * D + k, complex(cc[k]))
                      for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)

        def _winner(scores, vocab):
            s = np.maximum(np.asarray(scores, dtype=float), 0.0)
            if s.size == 0:
                return (None, None, None, None)
            j = int(np.argmax(s)); peak = float(s[j])
            # legacy `confidence`: s[j]/peak is IDENTICALLY 1.0 at j=argmax(s) by construction (peak==s[j]) --
            # kept byte-identical as the backward-compat field (see the docstring); it is NOT used for the metacog
            # discrimination anymore (issue #181).
            conf = float(np.clip(s[j] / peak, 0.0, 1.0)) if peak > 0.0 else 0.0
            margin = self._margin(s)   # the composer's own validated decisiveness read (peak-runner_up)/peak
            # margin_spiking (scaffold-retirement backlog rank 9, opt-in via self.comp.spiking_recall_margin /
            # BRAIN_METACOG_SPIKING_MARGIN, default None = byte-identical): the SAME winner-vs-runner-up
            # decisiveness read, off the recall circuit's OWN Izhikevich WTA spike counts (`_spiking_margin`,
            # the SAME bank `_spiking_select` drives for the on-substrate winner-pick) instead of a host
            # comparison of `s`. See research/runners/rf_phasor_composer.py:_spiking_margin.
            margin_spiking = self.comp._spiking_margin(s) if self.comp.spiking_recall_margin else None
            w = vocab[j]
            if isinstance(w, str) and w.startswith("__free"):    # a reserved (unrecruited) slot -> not a real decode
                return (None, conf, margin, margin_spiking)
            return (w, conf, margin, margin_spiking)
        out = {}
        for ri, role in enumerate(self.main_roles):
            out[role] = _winner(mem[self.c_base + ri * V:self.c_base + (ri + 1) * V], self.words)
        out["polarity"] = _winner(mem[self.c_base + self.n_main * V:self.c_base + self.n_main * V + NP], self.pol_words)
        return out

    def _trace_query(self, cue_roles, idx, decoded_extra=None):
        """Build self.last_trace for the LAST query (read-only). `cue_roles` = the {role: asserted_value} matched on;
        `idx` = the selected fact-block index (or None = abstain); `decoded_extra` optionally overrides/adds answer
        chips (e.g. a rendered clause patient). Records the per-role chips (cue + answer roles), which engram block
        matched + how many were scanned, and the post-resonate RF gauge over the rf slice. An abstain records
        matched_fact_index=None + scanned=N WITHOUT a fabricated answer (the moat made visible). Never affects the
        return value."""
        if not self.trace:
            return
        n_scanned = len(self.kb)
        roles_out = []
        block_scores = self._block_role_scores(idx) if idx is not None else {}
        cue_set = set(cue_roles)
        for role, asserted in cue_roles.items():
            word, conf, margin, margin_spiking = block_scores.get(role, (asserted, None, None, None))
            roles_out.append({"role": role, "word": word, "confidence": conf, "margin": margin,
                              "margin_spiking": margin_spiking, "cue": True, "asserted": asserted})
        # answer/decoded roles = the non-cue main roles (+ polarity) of the selected block + any explicit extras
        for role, (word, conf, margin, margin_spiking) in block_scores.items():
            if role in cue_set:
                continue
            roles_out.append({"role": role, "word": word, "confidence": conf, "margin": margin,
                              "margin_spiking": margin_spiking, "cue": False})
        for role, extra in (decoded_extra or {}).items():
            # decoded_extra chips (e.g. a recursively-rendered clause patient) are host-composed, not a genuine
            # winner/runner-up VSA read -- no margin, no margin_spiking to report (2-tuple (word, conf) callers
            # stay valid).
            word, conf = extra[0], extra[1]
            margin = extra[2] if len(extra) > 2 else None
            roles_out.append({"role": role, "word": word, "confidence": conf, "margin": margin,
                              "margin_spiking": None, "cue": False})
        self.last_trace = {
            "roles": roles_out,
            "matched_fact_index": (int(idx) if idx is not None else None),
            "n_facts_scanned": int(n_scanned),
            "abstained": idx is None,
            "rf": self._rf_gauge(),
            "composer": "onebrain",
        }

    def _seq_cleanup_conns(self):
        """opt #4 (the audit's sequencer drive-seed lever): the cleanup-codebook connections that `block_cleanup_scores`
        installs are BLOCK-INVARIANT -- they depend only on the concept codebook + the fixed single-block c_base/q_base
        layout (the per-block trigger only changes the UNBIND wiring), so they are identical for every one of the K
        per-block drive-seed reads. Build them ONCE and reuse across the K reads. The cache is invalidated each drive
        rebuild in `_ensure_sequencer`, so a store / reconsolidation / regrounded concept is always picked up (the moat
        is never served a stale cleanup). Saves ~K x the V*main_roles*D tuple construction (the audit's ~3.9M-tuples-
        at-K=32 drive-seed cost)."""
        if self._seq_cleanup_conns_cache is not None:
            return self._seq_cleanup_conns_cache
        comp, D, V = self.comp, self.D, self.V
        clean = []
        for ri in range(len(self.main_roles)):
            for j in range(V):
                cc = np.conj(comp._to_phasor(comp.concepts[self.words[j]]))
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        self._seq_cleanup_conns_cache = clean
        return clean

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
        if self.enable_seq_vocab_shrink:
            # reduced fabric: role A over V'_A (distinct stored agents), role X over V'_X (distinct stored actions);
            # rebuilt when K grows OR the cue-vocab signature changes (e.g. reconsolidation rewrites an agent/action).
            facts = [(f.get("agent"), f.get("action"), f.get("patient")) for (f, _) in self.kb[:K]]
            agentsA, actionsX, mapA, mapX = fns["reduced_cue_vocab"](facts, K)
            sig = (tuple(agentsA), tuple(actionsX))
            if self._seq is None or self._seq_K != K or self._seq_cuevocab_sig != sig:
                sb, meta = fns["build_sequencerK_reduced_bridge"](seed=self.seed, VA=len(agentsA),
                                                                 VX=len(actionsX), K=K)
                score_sb = fns["build_divnorm_score_bridge"](seed=self.seed, V=self.V, enable_divnorm=True,
                                                             sigma=self.sequencer_sigma, gain=self.sequencer_gain)
                self._seq = (sb, meta); self._seq_score = score_sb; self._seq_K = K
                self._seq_mapA = mapA; self._seq_mapX = mapX; self._seq_cuevocab_sig = sig
                self._seq_dirty = True                             # a new reduced fabric -> the drives must be rebuilt
        elif self._seq is None or self._seq_K != K:
            sb, meta = fns["build_sequencerK_bridge"](seed=self.seed, V=self.V, K=K)
            score_sb = fns["build_divnorm_score_bridge"](seed=self.seed, V=self.V, enable_divnorm=True,
                                                         sigma=self.sequencer_sigma, gain=self.sequencer_gain)
            self._seq = (sb, meta); self._seq_score = score_sb; self._seq_K = K
            self._seq_dirty = True                                 # a new K -> the drives must be (re)built
        if self._seq_dirty or self._seq_drives is None:
            self._seq_cleanup_conns_cache = None              # opt #4: rebuild the block-invariant cleanup conns once for this drive rebuild
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
        if self._fused:
            # the R1 FUSED path: the folded one-bridge sequencer with the cleanup->score handoff DEVICE-RESIDENT
            # (no `to_host` of the cleanup score). == the separate-bridge spiking decision; only WHERE the cleanup
            # score lives (host array vs device) differs. Lazily built + cached (the fused fabric is a separate
            # runner module so the OFF/True paths never import it).
            from research.runners._seq_fused_fabric import fused_seq_block
            return fused_seq_block(self, agent, action)
        # the spiking path (lazy build; rebuild drives on a dirtied/grown store).
        K = len(self.kb)
        if K == 0:
            return None
        if agent not in self._word_index or action not in self._word_index:
            return None                                           # an absent cue WORD -> no block -> abstain (the moat)
        self._ensure_sequencer(K)
        fns = _seq_imports()
        sb, meta = self._seq
        if self.enable_seq_vocab_shrink:
            if agent not in self._seq_mapA or action not in self._seq_mapX:
                return None                                        # cue not a stored agent/action -> abstain (== no
                                                                   # block matches in the full-V build; moat-preserving)
            dec, _rates = fns["run_sequencerK_reduced_with_drive"](sb, meta, self.words, self._seq_mapA, self._seq_mapX,
                                                                   agent, action, self._seq_drives,
                                                                   match_thresh=self.sequencer_match_thresh)
        else:
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
        b, D, Pd, V = self.b, self.D, self.period, self.V
        # Q register holding the recovered outer patient (= the clause composite). Reuse the POLARITY Q slot as scratch
        # (clause decode never reads polarity), which is valid for both the 4-role default (pol at index 3, == the old
        # hardcoded Q[3]) and the 5-role attribute layout (pol at index 4) -- always inside the per-block Q region, so
        # it never clobbers the cleanup region at c_base.
        pq = self.bind_roles.index("polarity")                             # the polarity Q slot, reused as scratch
        # hop 1: reconstruct the outer block (kick) + unbind the OUTER patient -> the embedded clause composite in Q[pq]
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        zc = self._unbind_conj("patient")
        outer = [(self.q_base + pq * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(outer); b.rf_resonate_steps(Pd + 8)
        # hop 2: RE-KICK the clause composite as a clean unit phasor (== the oracle's fresh per-hop kick), then unbind
        # the 3 clause roles IN PARALLEL from Q[pq] -> Q[0..2] + cleanup against the main vocab.
        # BURNDOWN I-1: the hop-1 -> hop-2 handoff is ON-SUBSTRATE (the host `to_host(rf_read_phases()) -> _to_phasor ->
        # rf_kick` round-trip is gone): `_dev_rekick_into` recovers Q[pq]'s phase from the device spike trackers, installs
        # a clean unit phasor back into Q[pq], and resets the RF trackers (== rf_kick) -- all device ops, no host phasor
        # copy. Byte-identical to the round-trip (I-1-a de-risk). The unbind operator + the resonate window are unchanged.
        inner = []
        for ri, role in enumerate(ROLES3):
            zcr = self._unbind_conj(role)
            inner += [(self.q_base + ri * D + k, self.q_base + pq * D + k, complex(zcr[k])) for k in range(D)]
        self._dev_rekick_into([slice(self.q_base + pq * D, self.q_base + (pq + 1) * D)])
        b.rf_set_complex_weights(inner); b.rf_resonate_steps(Pd + 8)
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
        clause sentence; an attributed patient ('big apple') prepends the decoded attribute. The (agent, action)
        cue-match-and-first-match SELECTION routes through `_seq_block` (the spiking K-way sequencer when
        integrated_loop, else the host first-match -- byte-identical); the downstream patient-type routing + decode read
        the SAME block on both paths (only WHICH block is selected moves from host to spikes). When
        `enable_fact_shard` (default-off), the (agent, action) selection routes through the DG-CA3 fact-block shard
        (O(shard) blocks decoded, not O(k_max)); the SAME tail runs on the selected block -> answer-identical."""
        if self.trace:
            self.last_trace = None
        if self._fact_shard_active():                          # FACT-COUNT-axis sublinear fast path (default-off)
            idx, got = self._fact_shard_first_match({"agent": agent, "action": action})
            if idx is not _FS_ESCALATE:
                return self._finish_query_patient(agent, action, idx, got, order_fn)
        idx = self._seq_block(agent, action)                   # full path (byte-identical when the fast path is off)
        got = self._read_blocks()[idx] if idx is not None else None
        return self._finish_query_patient(agent, action, idx, got, order_fn)

    def _finish_query_patient(self, agent, action, idx, got, order_fn):
        """Shared tail for `query_patient`: patient-type routing (embedded-clause / attributed / plain word) + trace,
        over the already-selected (idx, got). Called by BOTH the fact-shard fast path and the full path with the same
        (idx, got), so the answer is identical -- the ONLY thing the fast path changes is WHICH blocks were decoded to
        select idx (a shard vs the full k_max scan)."""
        if idx is None:
            if self.trace:
                self._trace_query({"agent": agent, "action": action}, None)
            return None
        stored = self.kb[idx][0].get("patient") if idx < len(self.kb) else None
        if _is_clause(stored):
            ans = self._decode_clause(idx, order_fn=order_fn)
            if self.trace:
                self._trace_query({"agent": agent, "action": action}, idx,
                                  decoded_extra={"patient": (ans, None)})
            return ans
        ans = self._attributed_patient(idx, got.get("patient"), got)
        if self.trace:
            self._trace_query({"agent": agent, "action": action}, idx)
        return ans

    def query_agent(self, action, patient):
        if self.trace:
            self.last_trace = None
        if self._fact_shard_active():                          # FACT-COUNT-axis sublinear fast path (default-off)
            idx, got = self._fact_shard_first_match({"action": action, "patient": patient})
            if idx is not _FS_ESCALATE:                        # reverse lookup: cue = (action, patient) -> agent
                ans = got.get("agent") if idx is not None else None
                if self.trace:
                    self._trace_query({"action": action, "patient": patient}, idx)
                return ans
        ans = self._scan({"action": action, "patient": patient}, "agent")
        if self.trace:
            # find the block index (the first matching), for the trace's matched-engram line
            idx = None
            for i, got in enumerate(self._read_blocks()):
                if got.get("action") == action and got.get("patient") == patient:
                    idx = i; break
            self._trace_query({"action": action, "patient": patient}, idx)
        return ans

    def ask_yes_no(self, agent, action, patient):
        """yes / no / unknown: the first fact matching the full SVO answers by its polarity tag (AFFIRM -> yes,
        NEGATE -> no); no matching fact -> 'unknown' (the no-confab moat). The (agent, action) cue-match-and-first-match
        SELECTION routes through `_seq_block` (the spiking K-way sequencer when integrated_loop, else the host first-
        match -- byte-identical); the patient equality + polarity are the body read over the selected block (identical
        on both paths). NOTE: the host first-match scans for the first block matching the FULL SVO, whereas the
        sequencer matches (agent, action) then checks patient on the selected block -- equivalent for the production
        unique-(agent, action) store (each (agent, action) selects one block, and the patient check then decides
        yes/no/unknown); a degenerate same-(agent, action) different-patient pair is outside the production regime.
        When `enable_fact_shard` (default-off), the (agent, action) selection routes through the DG-CA3 fact-block
        shard (O(shard) not O(k_max)); the patient-equality + polarity tail is identical -> answer-identical."""
        if self.trace:
            self.last_trace = None
        if self._fact_shard_active():                          # FACT-COUNT-axis sublinear fast path (default-off)
            idx, got = self._fact_shard_first_match({"agent": agent, "action": action})
            if idx is not _FS_ESCALATE:
                return self._finish_ask_yes_no(agent, action, patient, idx, got)
        idx = self._seq_block(agent, action)                   # full path (byte-identical when the fast path is off)
        got = self._read_blocks()[idx] if idx is not None else None
        return self._finish_ask_yes_no(agent, action, patient, idx, got)

    def _finish_ask_yes_no(self, agent, action, patient, idx, got):
        """Shared tail for `ask_yes_no`: patient-equality + polarity read over the already-selected (idx, got).
        Called by BOTH the fact-shard fast path and the full path with the same (idx, got) -> yes/no/unknown is
        identical; the fast path only changes WHICH blocks were decoded to select idx."""
        if idx is None:
            if self.trace:
                self._trace_query({"agent": agent, "action": action, "patient": patient}, None)
            return "unknown"
        if got.get("patient") != patient:
            if self.trace:
                self._trace_query({"agent": agent, "action": action, "patient": patient}, None)
            return "unknown"                                   # the (agent, action) block's patient != the asserted one
        if self.trace:
            self._trace_query({"agent": agent, "action": action, "patient": patient}, idx)
        return "yes" if got.get("polarity") == "AFFIRM" else "no"

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

    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) from the stored facts (agent/action/patient co-occur;
        clause patients skipped -- their inner concepts are structural). The graph the dlPFC dialogue planner spreads
        over (the rich_answer_composer's `ordered_associates` + the agent's `elaborate`). A pure function of `self.kb`
        (the (fact_dict, None) bookkeeping), so it is BYTE-IDENTICAL to RFPhasorComposer._assoc_graph on the same kb --
        making the OneBrainComposer a complete RFPhasorComposer API-sibling for the console's dialogue-planning path
        (C3). Read-only (no resonate, no bridge step); the no-confab moat is untouched."""
        graph = {}
        for fact, _ in self.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient") if isinstance(fact.get(r), str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    # --- Tier 2.2: SELF-CUED associative chain-of-thought (== the rf composer's chain_of_thought) ---------------
    def _relation_assoc(self):
        """The agent's OWN learned RELATION-KEYED association strengths from its stored facts:
        assoc[(agent, action)] = co-occurrence count over the kb (keyed by the RELATION so the selector picks WHICH
        relation to chase). Mirrors RFPhasorComposer._relation_assoc."""
        assoc = {}
        for fact, _ in self.kb:
            a, act = fact.get("agent"), fact.get("action")
            if isinstance(a, str) and isinstance(act, str):
                assoc[(a, act)] = assoc.get((a, act), 0.0) + 1.0
        return assoc

    def _select_next_relation(self, x, assoc, lesion=None, lesion_rng=None):
        """SELF-CUE: among the relations available from concept `x` (as agent), pick the highest learned-association
        relation; None if no associate (dead end -> abstain). lesion='zero' -> None; 'scramble' -> random ordering.
        Mirrors RFPhasorComposer._select_next_relation."""
        cands = {rel: w for (a, rel), w in assoc.items() if a == x}
        if not cands:
            return None
        if lesion == "zero":
            return None
        if lesion == "scramble":
            cands = {rel: float(lesion_rng.random()) for rel in cands}
        return max(sorted(cands), key=cands.get)

    def chain_of_thought(self, start, goal=None, max_hops=4, lesion=None, lesion_rng=None, return_path=False):
        """SELF-CUED associative chain-of-thought (Tier 2.2) on the ONE persistent brain: from `start`, the agent
        SELECTS each next relation by LEARNED association over its own facts (NOT a caller plan), then chases it via
        the validated on-bridge `query_patient`; cleanup re-discretizes between hops (no compounding). Stops at
        `goal` or a dead end -> ABSTAIN (the no-confab moat at EVERY hop). == RFPhasorComposer.chain_of_thought;
        de-risked GO (2026-06-27-tier2.2-chain-of-thought-GO.md). Returns terminal (or None); return_path=True ->
        (terminal, [start, ...])."""
        assoc = self._relation_assoc()
        x = start
        path = [x]
        terminal = None
        for _ in range(int(max_hops)):
            rel = self._select_next_relation(x, assoc, lesion=lesion, lesion_rng=lesion_rng)
            if rel is None:
                break
            nxt = self.query_patient(x, rel)
            if nxt is None:
                break
            path.append(nxt)
            x = nxt
            terminal = x
            if goal is not None and x == goal:
                break
        return (terminal, path) if return_path else terminal

    # --- reconsolidation: prediction-error-gated in-place fact update (== the rf composer's update_on_mismatch) ---
    def _recovered_patient_phases(self, block_idx):
        """Reconstruct block_idx + unbind the patient role -> the RAW recovered patient phases (NOT cleaned up to a
        word), READ TO HOST via rf_read_phases. The reconsolidation prediction error compares these against an asserted
        patient's code. LEGACY HOST-SEAM read used ONLY when persistent_loop=False (the on-substrate PE path below
        avoids this host round-trip)."""
        comp, b, D, Pd = self.comp, self.b, self.D, self.period
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        zc = self._unbind_conj("patient")
        unbind = [(self.q_base + 2 * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]      # patient -> Q[2]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        return np.asarray(b.rf_read_phases())[self.q_base + 2 * D:self.q_base + 3 * D]

    def _patient_cleanup_scores(self, block_idx):
        """CLOSURE 5 (purity backlog #5 -- extend the persistent spiking loop to the RECONSOLIDATION op): the
        on-substrate prediction-error read. Reconstruct block_idx + unbind the patient role into Q[2], RE-KICK Q[2] as
        a CLEAN UNIT PHASOR (the Closure-2 `_dev_rekick_into` register->register op-handoff -- NO `to_host` of the
        phasor; the recovered patient composite is held on the bridge as a clean unit phasor), then run the matched-
        filter cleanup matvec (Q[2] -> a cleanup neuron per vocab word via `_cleanup_conj`) and read the per-word
        membrane SCORES off the body (cp_membrane_potential_v[c_base:c_base+V]). The reconsolidation PE then derives
        from the SPIKING score: PE_w = 1 - score_w/D, where score_w = Re(conj(code_w).clean_Q2) = sum_k
        cos(2pi(code_w-rec)) -- the EXACT on-substrate analog of the host numpy cos `1 - mean(cos(...))`. This replaces
        the host `rf_read_phases -> numpy cos` round-trip (the last non-flat op with a genuine host seam) with an
        on-device read-phase + re-kick + matched filter. Decision-identical to the host cos (the residual is float32
        membrane rounding ~2.5e-8 << the gate margins; the rewrite/restabilize/abstain decision is invariant, exactly
        as the flat cleanup argmax is). De-risk: research/findings/raw/_closure5_reconsolidation_onsub_pe_derisk.json."""
        b, D, Pd, V = self.b, self.D, self.period, self.V
        self._zero_rf_v_u()
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        zc = self._unbind_conj("patient")
        unbind = [(self.q_base + 2 * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]      # patient -> Q[2]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        # the Closure-2 clean-unit-phasor op-handoff: normalize+quantize Q[2] on-device (register->register), no host
        # phasor copy. == a host round-trip on the cleanup membrane (the I-1-a byte-identity GO); makes the recovered
        # patient a clean unit phasor held ON THE BRIDGE between the unbind op and the matched-filter PE op.
        self._dev_rekick_into([slice(self.q_base + 2 * D, self.q_base + 3 * D)])
        clean = []
        for j in range(V):
            cc = self._cleanup_conj(self.words[j])
            clean += [(self.c_base + j, self.q_base + 2 * D + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        return mem[self.c_base:self.c_base + V]

    def _patient_prediction_error(self, block_idx, patient_word):
        """PE = 1 - phase-cos(recovered patient phasor, the asserted patient's code). ~0 when the asserted filler
        matches the stored one (a re-statement); ~1 on a mismatch (a correction). == the rf composer's measure.
        persistent_loop ON (default, Closure 5): the SPIKE-RESIDENT read -- PE_w = 1 - score_w/D from the on-substrate
        matched-filter cleanup (`_patient_cleanup_scores`), NO host rf_read_phases/cos round-trip. OFF: the legacy host
        cos over the host-read recovered phases (the revertible byte-comparable escape)."""
        if self.persistent_loop:
            scores = self._patient_cleanup_scores(block_idx)
            j = self._word_index.get(patient_word)
            if j is None:                       # an out-of-vocab asserted patient: fall back to the host cos (no score)
                rec = self._recovered_patient_phases(block_idx)
                return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.comp.concepts[patient_word]))))
            return 1.0 - float(scores[j]) / float(self.D)
        rec = self._recovered_patient_phases(block_idx)
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.comp.concepts[patient_word]))))

    def _calibrate_pe_labile(self):
        """Frozen labilization gate = the midpoint of the same-vs-different prediction-error distributions over the
        CURRENT facts (each fact's PE against its OWN stored patient = 'same'; against other facts' patients =
        'different'). The data's own separation point -- NOT tuned to a downstream probe. 0.5 fallback when too few
        distinct facts exist to calibrate. == the rf composer's _calibrate_pe_labile (string-patient facts only).
        persistent_loop ON (default, Closure 5): each fact's PE is read from the SPIKE-RESIDENT matched-filter scores
        (PE = 1 - score/D), so the gate calibration is also host-round-trip-free; OFF: the legacy host cos."""
        idxs = [i for i, (fact, _) in enumerate(self.kb) if isinstance(fact.get("patient"), str)]
        pats = {i: self.kb[i][0]["patient"] for i in idxs}
        if self.persistent_loop:
            scores = {i: self._patient_cleanup_scores(i) for i in idxs}   # one matched-filter read per fact (full vocab)

            def pe(i, word):                                              # PE = 1 - score_word/D (the on-substrate read)
                j = self._word_index.get(word)
                return (1.0 - float(scores[i][j]) / float(self.D)) if j is not None else 1.0
            same, diff = [], []
            for i in idxs:
                same.append(pe(i, pats[i]))
                for j in idxs:
                    if pats[j] != pats[i]:
                        diff.append(pe(i, pats[j]))
            if not same or not diff:
                return 0.5
            return 0.5 * (float(np.mean(same)) + float(np.mean(diff)))
        recs = {i: self._recovered_patient_phases(i) for i in idxs}

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
        """The FIRST stored block whose cue roles (agent+action) match, or None (no trace to reactivate -> abstain).
        Returns the block/kb index. Routes through `_seq_block` (the spiking K-way sequencer when integrated_loop, else
        the host first-match -- byte-identical), so reconsolidation (`update_on_mismatch`) inherits the spiking decision
        for free."""
        return self._seq_block(agent, action)

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
