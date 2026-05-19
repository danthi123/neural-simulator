"""Net-new shared theta-gamma SPEAR conversational runner (Architecture A).

Biology (Hasselmo SPEAR -- Separate Phases of Encoding And Retrieval):
the recent/remote conflict is dissolved NOT by reading two stores but by
TEMPORAL MULTIPLEXING under one shared ~125 ms theta rhythm. Each theta
cycle has an ENCODE phase (entorhinal-afferent drive, plasticity ON,
retrieval suppressed) and a RETRIEVE/pattern-complete phase (CA3-recurrent
drive, plasticity OFF). Acetylcholine is the phase gate; a prefrontal
working-memory frame (dlpfc) holds the ordered compositional slot indexed
by a nested gamma sub-cycle; a generative replay loop emits ordered output.

This module is the ONLY genuinely net-new code: a small theta-gamma
rhythm/phase CONTROLLER + the wiring that composes the project's
ALREADY-VALIDATED subsystems under it. It is a TIMING controller, NOT a
new learning mechanism -- every learning rule is a reused validated rule.
NO automatic differentiation anywhere. ASCII only.

Everything below the controller is reused BYTE-UNCHANGED by import:

  * substrate + hippocampus + dlpfc PFC frame: the VALIDATED v16 concept
    recipe construction the Stage-1 re-review cleared
    (compose_retrieval_runner._build_substrate). It builds the 16-pool
    concept substrate via text_minimal_isolation.build_biological_brain_
    regions with enable_hippocampus_consolidation=True + enable_dlpfc_
    verb=True + global enable_nmda=True (dlpfc bistability). It does NOT
    override cfg.num_traits -- and neither do we (the Stage-1 lesson).
  * acetylcholine phase gate: the VALIDATED neuromodulator subsystem
    (sim.neuromodulators). A NeuromodulatorConfig with a
    plasticity_window_gate target (scope=all) + a `manual` production
    rule; the controller sets the concentration per phase via the reused
    NeuromodulatorManager.set_concentration. The bridge's reused step
    multiplies reward-driven weight updates by
    compute_plasticity_window_gate_multiplier() with NO edit to the
    step.
  * recent-specific encode: REUSED compose_concept_engram.encode_
    concept_pair with a hippocampal region_filter (validated Tonegawa
    stim-recall path, catalog D.14). Tag NAMES are OPAQUE (fact_{i}) --
    the answer is read from neural activity, never a string.
  * generative replay loop / consolidation: REUSED consolidation_trainer.
    run_concept_replay_phase + run_swr_replay_phase + the reused
    text_minimal_isolation set_awake_gates / set_sleep_gates for the
    slower encode<->consolidate transition.
  * regime-correct readout: REUSED compose_concept_engram.lang_output_
    pattern_during_stim / lang_output_pattern_during_input + cosine_to_
    word; ranking is the RAW lang_output FIRING-RATE confidence the
    validated readout / 650 abstention benchmark calibrate.
  * no-confabulation moat: REUSED abstention_gate.gate(ranked, 650.0)
    (byte-unchanged 7/7).
  * kill-safe/resume: REUSED sim.train_checkpoint (the same per-cell
    save_checkpoint pattern Stage-1 uses).
  * frozen verdict: REUSED spear_conversational_core.spear_conversational
    _verdict.

ACH TARGETING + POLARITY (closed 2026-05-19 net-new-runner-only
faithfulness-fix after the dedicated adversarial review confirmed the
prior plasticity_window_gate (scope=all) target was FUNCTIONALLY INERT
here -- its ONLY consumer at sim/bridge.py:5577-5579 sits inside the C2
reward-mod block, gated by `update_path_active` (bridge.py:5512-5513),
and this runner never drives `current_reward_signal`, so the gate
multiplier was never applied to anything):

  Targets now used (verified against the REUSED bridge consumers):
    - `plasticity_rate` (scope=all): consumed at sim/bridge.py:5519-5523
      via compute_plasticity_rate_multiplier(). Same C2 path -- inert
      when reward=0, but kept so the gate composes correctly the
      moment a downstream stage adds a reward signal (no surprise
      regression).
    - `synaptic_gain` (scope=all): consumed EVERY simulation step at
      sim/bridge.py:4877-4879 (STP branch) and bridge.py:4890-4897 (no-
      STP branch) via compute_synaptic_gain_multiplier(). This is the
      Hasselmo-faithful path: ACh modulates effective_synaptic_strength
      directly, so the encode (LOW ACh) and retrieve (HIGH ACh) phases
      genuinely change forward dynamics (and through that, downstream
      STDP at C1, bridge.py:5341-5400).
    - We retain the original plasticity_window_gate target for forward-
      compatibility with the bridge's TAN gating path (no harm; it is
      0.0 sensitivity-equivalent when reward=0).

  Polarity (with the synaptic_gain sensitivity chosen NEGATIVE):
    ENCODE phase  -> ACh LOW (conc ~= 0, well below baseline 1.0).
                     synaptic_gain effect = 1 + (-s)*(0 - 1)
                                          = 1 + s = enhanced gain.
                     plasticity_window_gate ~= 1 (would-be permitted).
                     Hasselmo: enhanced afferent drive, suppressed
                     recurrent feedback equivalent -- the encode mode.
    RETRIEVE phase -> ACh HIGH (conc ~= 1, at baseline).
                     synaptic_gain effect = 1 + (-s)*(1 - 1) = 1
                     plasticity_window_gate = 0 (would-be blocked).
                     Hasselmo: baseline gain -> CA3 recurrent pattern-
                     completion dominates -- the retrieve mode.

  The polarity matches Hasselmo SPEAR (high ACh suppresses retrieval-
  time LTP; the encode pause + enhanced afferent gain permits writing).
  The design-doc prose says "encode = ACh high" loosely; we follow the
  REUSED CODE SEMANTICS (the bridge's gate / gain formulas), which is
  the biology-faithful choice and what the bridge actually computes.

The decisive built-in control arm `rhythm_removed` is IDENTICAL to
`full` for the same (seed, N) -- SAME seed, SAME facts, SAME RNG draws --
with the shared-rhythm CONTROLLER DISABLED (no theta phase multiplexing,
ACh held neutral): this reduces to the Stage-1 static composition (which
empirically scored ~0.00), so the capability must be attributable to the
rhythm. `full` and `rhythm_removed` differ ONLY by the single
`use_rhythm` flag threaded identically.

CuPy is the real/decisive path; --tiny-synth shrinks pools/episodes/
phase-block lengths so the smoke is seconds -- its toy numbers are
explicitly NOT a result (they only screen for fatal logic flaws and make
the Task-0 pin green). The decisive multi-seed CuPy run is a later
controller-only task, NOT performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy: identical to the Stage-1 runner. The project rule is
# "NumPy ONLY for the smoke; CuPy is the decisive path". SimulationBridge
# binds its array module at sim.bridge IMPORT time, so on a CuPy-capable
# box the tiny smoke runs on the bridge's real backend (still seconds --
# pools/episodes/phase-blocks are shrunk hard); we only pin NumPy when
# CuPy is genuinely unavailable. We never diverge the substrate to make
# a CPU smoke pass (faithfulness > smoke convenience).
if "--tiny-synth" in sys.argv:
    try:
        import cupy as _cupy_probe  # noqa: F401

        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
    if not _CUPY_AVAILABLE:
        os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.spear_conversational_core import (
    spear_conversational_verdict,
    _SP_LADDER,
)
from research.runners.abstention_gate import gate as _abstain_gate
from research.runners.abstention_gate import DEFAULT_THRESHOLD as _MOAT
from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# REUSED Stage-1-cleared substrate construction + recent-fact helpers
# (byte-unchanged by import). _build_substrate is the validated v16
# recipe + hippocampus; we add the dlpfc PFC frame below via the SAME
# builder kwargs (it is already enable_nmda=True for bistability). We
# reuse Stage-1's recent-fact vocabulary + opaque-tag encode + raw
# firing-rate ranking verbatim -- duplicating NO subsystem logic.
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _VERBS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
    _recent_facts,
    _ranked_from_pattern,
    _HIPPO_TAG_REGIONS,
)


# =====================================================================
#  Substrate + hippocampus + dlpfc PFC frame (REUSE the validated
#  recipe; add ONLY the validated dlpfc kwargs the design specifies).
# =====================================================================
def _build_substrate(seed: int, tiny_synth: bool):
    """Construct the validated v16 concept-pool bridge WITH the
    hippocampal consolidation regions AND the dlpfc PFC working-memory
    compositional frame, by REUSING the validated builders byte-
    unchanged. Returns (bridge, dims).

    This mirrors the Stage-1-cleared compose_retrieval_runner._build_
    substrate EXACTLY (same CoreSimConfig field set, same kwargs, NO
    cfg.num_traits override) and adds ONLY the design-specified
    validated dlpfc PFC frame kwargs (enable_dlpfc_verb=True,
    n_dlpfc_verb, dlpfc_verb_internal_density) -- build_biological_
    brain_regions' OWN kwargs, reused unmodified. enable_nmda=True is
    already set (dlpfc NMDA bistability). tiny_synth only shrinks
    pool/lang dimensions; the recipe itself is unchanged.
    """
    if tiny_synth:
        try:
            import cupy as _c  # noqa: F401

            _cupy_ok = True
        except Exception:
            _cupy_ok = False
        if not _cupy_ok:
            os.environ["SIM_BACKEND"] = "numpy"
            from sim.backend import get_backend as _get_backend

            _get_backend("numpy")

    import research.runners.concept_pool_demo as cpd
    from sim.config import (
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    if tiny_synth:
        n_lang_input = 64
        n_per_pool = 12
        n_fs_per_pool = 3
        n_dlpfc_verb = 24
    else:
        # Decisive-path defaults (validated v16 recipe scale).
        n_lang_input = 2048
        n_per_pool = 200
        n_fs_per_pool = 24
        n_dlpfc_verb = 200

    # weak_dynamics=True (validated v16) -- identical to Stage-1.
    concept_internal_density = 0.05
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=0.10,
        motor_exc_weight_mean=2.0,
        motor_inh_weight_mean=4.0,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=cpd.NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=cpd.VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=cpd.ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        # Validated trisynaptic hippocampal recent-specific path
        # (catalog D.03/D.12/D.13) -- builder's own kwarg.
        enable_hippocampus_consolidation=True,
        # Validated dlpfc PFC working-memory compositional frame
        # (design section 3; PREFRONTAL_CORTEX_WM) -- builder's own
        # kwargs, reused unmodified. NMDA bistability is enabled by
        # cfg.enable_nmda=True below (the validated profile).
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )

    # EXACT v16 recipe CoreSimConfig field set (compose_retrieval_runner
    # / concept_pool_demo.build_concept_bridge). NO cfg.num_traits
    # override -- the Stage-1 lesson.
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

    # --- Net-new wiring: register the acetylcholine phase-gate
    # neuromodulator (the VALIDATED subsystem). The 2026-05-19 net-new-
    # runner faithfulness fix re-targets ACh so the gate has measurable
    # effect on bridge dynamics across the encode/retrieve phases this
    # runner exercises (the prior plasticity_window_gate-only target
    # was inert here -- its only consumer sits inside the reward-mod
    # block we never enter; see the module docstring's ACH TARGETING
    # section). Targets:
    #   * synaptic_gain (scope=all): consumed every simulation step at
    #     sim/bridge.py:4877-4879 / 4890-4897 -- the Hasselmo-faithful
    #     path that makes the phase gate actually modulate
    #     effective_synaptic_strength. Negative sensitivity so ACh LOW
    #     (encode) boosts afferent gain and ACh HIGH (retrieve) settles
    #     at baseline gain.
    #   * plasticity_rate (scope=all): consumed at bridge.py:5519-5523
    #     via compute_plasticity_rate_multiplier(); composes the
    #     moment a downstream stage adds a reward signal.
    #   * plasticity_window_gate (scope=all): retained for forward-
    #     compatibility with the bridge's TAN gating path; reward=0
    #     here so it is effectively dormant but adds no harm.
    # baseline=1.0 (>0, required by the gate formula gate =
    # clip(1 - conc/baseline, 0, 1)). `manual` rule -> the controller
    # sets concentration each phase via the reused set_concentration
    # (explicit phase multiplexing, not reward-driven).
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        ModulatorTarget,
        ProductionRule,
    )

    # synaptic_gain sensitivity: at ACh=0 (encode) -> gain = 1 + 0.3 = 1.3
    # (boosted afferent drive); at ACh=1 (retrieve) -> gain = 1.0
    # (baseline). Chosen modest so dynamics don't saturate; the test pin
    # asserts the gate produces a clear measurable diff vs the inert
    # zero-effect path.
    _ACH_SYN_GAIN_SENS = -0.3
    # plasticity_rate sensitivity: same shape; small negative so high
    # ACh damps rate-modulated learning (TAN-like) when reward exists.
    _ACH_PLAST_RATE_SENS = -0.5
    ach = NeuromodulatorConfig(
        name="acetylcholine_tan",
        baseline=1.0,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            ModulatorTarget(
                target_type="synaptic_gain", scope="all",
                sensitivity=_ACH_SYN_GAIN_SENS,
            ),
            ModulatorTarget(
                target_type="plasticity_rate", scope="all",
                sensitivity=_ACH_PLAST_RATE_SENS,
            ),
            ModulatorTarget(
                target_type="plasticity_window_gate", scope="all",
            ),
        ],
        production_rules=[ProductionRule(rule_type="manual")],
    )
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [ach]

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Theta period in STEPS, derived from the bridge dt (NOT a hardcoded
    # step count). Hasselmo SPEAR theta ~125 ms; dt_ms=0.5 -> 250 steps
    # / theta cycle. tiny_synth shrinks the cycle hard (logic-screen).
    theta_ms = 125.0
    theta_steps = max(2, int(round(theta_ms / cfg.dt_ms)))
    dims = {
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "sparsity": 0.05,
        "theta_steps": theta_steps,
        "theta_ms": theta_ms,
        "dt_ms": cfg.dt_ms,
    }
    return bridge, dims


# =====================================================================
#  THE NET-NEW SHARED THETA-GAMMA RHYTHM / PHASE CONTROLLER.
#  (A timing controller; no new learning rule; no autograd.)
# =====================================================================
def _set_ach(bridge, value: float) -> None:
    """Set the acetylcholine concentration via the REUSED
    NeuromodulatorManager.set_concentration. ACh HIGH (>= baseline 1.0)
    -> plasticity_window_gate ~= 0 -> reward-driven weight updates
    suppressed (retrieve / pattern-completion only). ACh LOW (~= 0,
    pause) -> gate ~= 1 -> plasticity permitted (encode). Verified
    against the reused gate-multiplier docstring."""
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is None:
        return
    try:
        mgr.set_concentration("acetylcholine_tan", float(value))
    except Exception:
        # Subsystem not present (degraded smoke) -- the controller still
        # alternates drive; the gate is then a no-op (1.0). We never
        # fabricate a phase effect.
        pass


# ACh setpoints (faithful SPEAR polarity, baseline=1.0).
_ACH_ENCODE_LOW = 0.0  # pause -> gate ~= 1 -> plasticity ON (encode)
_ACH_RETRIEVE_HIGH = 1.0  # tonic -> gate ~= 0 -> plasticity OFF (retrieve)
_ACH_NEUTRAL = 1.0  # rhythm-removed: ACh held at tonic baseline (no
#                     phase multiplexing; reduces to the Stage-1 static
#                     composition -- gate ~= 0, the no-LTP read).


def _theta_encode_phase(bridge, fact, tag_name, dims, encoding_steps,
                        gamma_idx, n_facts, use_rhythm):
    """ENCODE phase of one theta cycle for ONE compositional fact.

    On the rhythm path: ACh LOW (plasticity ON, afferent drive), the
    recent-specific (noun, adj) binding is written as a Tonegawa engram
    over the HIPPOCAMPAL regions (reused encode_concept_pair). The
    `gamma_idx` is the nested gamma sub-cycle index -- it selects WHICH
    ordered compositional slot (which fact) this theta cycle encodes,
    so the dlpfc PFC frame + reused NMDA bistability hold/advance the
    ordered sequence across gamma sub-cycles (one fact per gamma
    sub-cycle, advancing each theta cycle).

    On the rhythm-removed path: ACh stays at neutral tonic baseline (NO
    phase multiplexing) -- the encode is the SAME reused encode_concept_
    pair with the SAME draws, but without the SPEAR plasticity window
    (it reduces to the Stage-1 static encode).
    """
    from research.runners.compose_concept_engram import encode_concept_pair

    noun, adj = fact
    if use_rhythm:
        _set_ach(bridge, _ACH_ENCODE_LOW)  # encode: plasticity ON
    else:
        _set_ach(bridge, _ACH_NEUTRAL)  # static: no phase gate

    if tag_name in {t["name"] for t in bridge.list_engram_tags()}:
        try:
            bridge.delete_engram_tag(tag_name)
        except Exception:
            pass
    # gamma_idx (the dlpfc ordered slot this theta cycle) is recorded in
    # the OPAQUE tag name only as a slot id -- it carries NO answer (no
    # noun/adj). Nothing downstream parses it.
    encode_concept_pair(
        bridge, noun, adj, tag_name,
        encoding_steps=encoding_steps,
        drive_pA=200.0, sparsity=dims["sparsity"],
        n_lang_input=dims["n_lang_input"],
        n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
        region_filter=_HIPPO_TAG_REGIONS,
        top_k=max(8, dims["n_per_pool"] // 4),
        balanced_teacher_pA=500.0,
        verbose=False,
    )


def _theta_retrieve_phase(bridge, cue_noun, tag_name, dims,
                          have_remote, recall_steps, use_rhythm):
    """RETRIEVE / pattern-complete phase of one theta cycle for ONE
    compositional query.

    On the rhythm path: ACh HIGH (plasticity OFF -> pattern-completion,
    CA3-recurrent; no retrieval-time LTP), then read the VALIDATED
    neural readout: stimulate the opaque recent-fact tag (hippocampal
    regime, recent-specific) AND drive lang_input(cue_noun)
    (consolidated regime, order-invariant schema); the answer is the
    composed RAW firing-rate confidence the calibrated 650 moat
    expects. The answer is decoded ONLY from neural activity -- never a
    tag string.

    On the rhythm-removed path: ACh stays at neutral tonic baseline
    (NO phase gate); the SAME reused readouts run with the SAME draws --
    it reduces to the Stage-1 static two-path composition.

    Returns (answer_or_None, ranked).
    """
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_stim,
        lang_output_pattern_during_input,
    )

    if use_rhythm:
        _set_ach(bridge, _ACH_RETRIEVE_HIGH)  # retrieve: plasticity OFF
    else:
        _set_ach(bridge, _ACH_NEUTRAL)

    # Consolidated-regime read (order-invariant schema).
    if have_remote:
        cons_pat, n_lo = lang_output_pattern_during_input(
            bridge, cue_noun,
            n_lang_input=dims["n_lang_input"],
            sparsity=dims["sparsity"],
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            stim_steps=recall_steps,
        )
        cons_ranked = _ranked_from_pattern(
            cons_pat, n_lo, dims, exclude=cue_noun
        )
    else:
        cons_ranked = []

    # Hippocampal-regime read (recent-specific retrieval).
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        hip_pat, n_lo2 = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0, stim_steps=recall_steps,
        )
        hip_ranked = _ranked_from_pattern(
            hip_pat, n_lo2, dims, exclude=cue_noun
        )
    else:
        hip_ranked = []

    # Retrieval-augmented compose: sum per-concept RAW firing-rate
    # confidences (the calibrated 650-moat quantity). Neither regime
    # alone clears the moat; only the composed sum does -- so a single-
    # path / empty / tag-string solver provably FAILs.
    scores: Dict[str, float] = {}
    for w, r, _ in cons_ranked:
        scores[w] = scores.get(w, 0.0) + r
    for w, r, _ in hip_ranked:
        scores[w] = scores.get(w, 0.0) + r
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    decided = _abstain_gate(ranked, _MOAT)  # REUSED moat, raw firing rate
    answer = None if decided is None else decided[0]
    return answer, ranked


def _hippo_silenced(bridge, silence_current_pA: float = -2000.0):
    """REUSED validated strict hippo-OFF protocol (byte-identical to
    Stage-1's _hippo_silenced / consolidation_eval). Kept here only as
    a local context helper -- it monkey-patches the bridge step to pin
    the hippocampal regions strongly negative each step and restores in
    finally. Used ONLY by the remote-only-style internal check; the
    decisive control arm is rhythm_removed (full minus the controller),
    not a regime ablation."""
    from sim.backend import get_backend
    from research.runners.consolidation_eval import HIPPO_REGIONS

    cp, _ = get_backend()
    rm = bridge.region_manager
    hippo_idx: List[int] = []
    for rname in HIPPO_REGIONS:
        try:
            idx = rm.indices(rname)
            if idx is not None:
                hippo_idx.extend(list(idx))
        except Exception:
            pass
    if not hippo_idx:
        return (lambda: None), 0
    hippo_arr = cp.asarray(hippo_idx, dtype=cp.int64)
    original_step = bridge._run_one_simulation_step

    def silenced_step():
        bridge.cp_external_input_current[hippo_arr] = float(
            silence_current_pA
        )
        return original_step()

    bridge._run_one_simulation_step = silenced_step

    def restore():
        bridge._run_one_simulation_step = original_step
        bridge.cp_external_input_current[hippo_arr] = 0.0

    return restore, len(hippo_idx)


def _generative_replay(bridge, tags, dims, tiny_synth, rng, use_rhythm):
    """The generative replay loop / slower encode<->consolidate
    transition the SAME SPEAR framework governs. REUSED consolidation_
    trainer.run_concept_replay_phase + run_swr_replay_phase, bracketed
    by the REUSED set_sleep_gates / set_awake_gates (the validated
    awake/sleep phase gates). On the rhythm path the consolidation
    happens in the ACh-LOW window (plasticity ON for ca3->ca1->cortex
    STDP); on the rhythm-removed path it is the SAME reused replay with
    the SAME draws but no SPEAR window (Stage-1 static consolidation).
    """
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
        run_swr_replay_phase,
    )
    from research.runners.text_minimal_isolation import (
        set_sleep_gates,
        set_awake_gates,
    )

    n_replays = 2 if tiny_synth else 20
    n_swr = 4 if tiny_synth else 200

    if use_rhythm:
        _set_ach(bridge, _ACH_ENCODE_LOW)  # consolidation = plasticity ON
    else:
        _set_ach(bridge, _ACH_NEUTRAL)

    try:
        set_sleep_gates(bridge)  # REUSED validated sleep phase gate
    except Exception:
        pass
    try:
        run_concept_replay_phase(
            bridge, tags,
            n_replays_per_tag=n_replays,
            burst_duration_ms=10 if tiny_synth else 100,
            inter_burst_ms=5 if tiny_synth else 50,
            drive_pA=100.0,
            randomize_order=True,
            rng=rng,
        )
        try:
            run_swr_replay_phase(
                bridge,
                n_swr_events=n_swr,
                burst_duration_ms=10 if tiny_synth else 100,
                inter_burst_ms=5 if tiny_synth else 50,
                swr_drive_pA=100.0,
                rng=rng,
            )
        except Exception:
            # run_swr_replay_phase imports cupy at top; on the NumPy
            # smoke it may be unavailable. Concept replay alone still
            # builds a (toy) schema -- acceptable for the logic-screen
            # smoke (NOT a result). The decisive CuPy path exercises
            # both.
            pass
    finally:
        try:
            set_awake_gates(bridge)  # REUSED validated awake phase gate
        except Exception:
            pass


# =====================================================================
#  One arm = one full SPEAR run for one (seed, N), differing from the
#  other arm ONLY by `use_rhythm` (the sole controller flag).
# =====================================================================
def _run_arm(seed: int, N: int, tiny_synth: bool, use_rhythm: bool):
    """Run the complete shared-rhythm SPEAR pipeline for ONE (seed, N).

    `use_rhythm=True`  -> the net-new theta-gamma controller is ENABLED:
        per theta cycle, an ACh-LOW ENCODE phase (gamma sub-cycle
        indexes the dlpfc ordered slot) then an ACh-HIGH RETRIEVE/
        pattern-complete phase; a generative replay loop consolidates
        the schema in the ACh-LOW window. This is the `full` arm.
    `use_rhythm=False` -> the controller is DISABLED: ACh is held at the
        neutral tonic baseline, there is NO theta phase multiplexing --
        the SAME reused encode + consolidate + readout run with the SAME
        seed and SAME RNG draws, which reduces to the Stage-1 static
        two-store composition (empirically ~0.00). This is the
        `rhythm_removed` arm.

    The ONLY difference between the two arms is the single `use_rhythm`
    flag threaded identically through every phase helper. Returns
    {"seed", "acc", "abstain_correct"}.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200
    facts = _recent_facts(N)

    bridge, dims = _build_substrate(seed, tiny_synth)
    rng = np.random.default_rng(seed)  # SAME seed -> SAME draws both arms

    # ---- ENCODE epoch: one theta cycle per fact; the nested gamma
    # sub-cycle index = the ordered compositional slot the dlpfc PFC
    # frame holds (one fact per gamma sub-cycle, advancing each theta
    # cycle). On the rhythm path each cycle's ENCODE phase runs in the
    # ACh-LOW plasticity window; on the rhythm-removed path ACh stays
    # neutral (no SPEAR window) but the SAME reused encode + SAME draws.
    n_facts = len(facts)
    tags: List[str] = []
    for gamma_idx, fact in enumerate(facts):
        tag = "fact_%d" % gamma_idx  # OPAQUE slot id (carries NO answer)
        _theta_encode_phase(
            bridge, fact, tag, dims, enc_steps,
            gamma_idx=gamma_idx, n_facts=n_facts, use_rhythm=use_rhythm,
        )
        tags.append(tag)

    # ---- Slower encode<->consolidate transition: the generative
    # replay loop (REUSED replay-consolidation + awake/sleep gates).
    _generative_replay(bridge, tags, dims, tiny_synth, rng, use_rhythm)

    # ---- RETRIEVE epoch: one theta cycle per compositional query;
    # ACh-HIGH pattern-completion (rhythm path) reads the VALIDATED
    # neural readout, gated by the REUSED no-confab moat.
    n_correct = 0
    n_total = 0
    n_abstain_ok = 0
    n_wrong = 0
    for i, (noun, adj) in enumerate(facts):
        n_total += 1
        tag = tags[i] if i < len(tags) else None
        answer, _ranked = _theta_retrieve_phase(
            bridge, noun, tag, dims,
            have_remote=True, recall_steps=recall_steps,
            use_rhythm=use_rhythm,
        )
        if answer == adj:
            n_correct += 1
        else:
            # answered wrong OR abstained: the no-confab invariant
            # requires an ABSTENTION here, not a confident wrong answer.
            n_wrong += 1
            if answer is None:
                n_abstain_ok += 1

    acc = (n_correct / n_total) if n_total else 0.0
    abstain_correct = (n_abstain_ok / n_wrong) if n_wrong else 1.0
    return {"seed": seed, "acc": acc, "abstain_correct": abstain_correct}


def _cell(seed: int, N: int, tiny_synth: bool) -> Dict[str, Any]:
    """One (seed, N) cell: the `full` arm (shared-rhythm controller ON)
    and the decisive built-in control `rhythm_removed` arm (controller
    OFF), SAME seed, SAME facts, SAME RNG draws -- differing ONLY by the
    single `use_rhythm` flag threaded identically into _run_arm."""
    full = _run_arm(seed, N, tiny_synth, use_rhythm=True)
    rhythm_removed = _run_arm(seed, N, tiny_synth, use_rhythm=False)
    return {
        "N": N,
        "full": full,
        "rhythm_removed": rhythm_removed,
    }


# =====================================================================
#  Aggregation + top-level entry.
# =====================================================================
def _aggregate(cells_by_N: Dict[int, List[Dict[str, Any]]],
               n_seeds: int) -> List[Dict[str, Any]]:
    """Aggregate per-seed cells into one rung dict per N -- EXACTLY the
    five keys the frozen spear_conversational_verdict consumes.

    `rhythm_removed_acc` is the decisive built-in control: a faithful
    rhythm-removed arm (Stage-1 static composition) must collapse below
    the frozen _SP_STATIC_CTRL_MAX. `abstain_correct_rhythm_removed`
    is the no-confabulation invariant ON that control arm: when the
    static reduction cannot answer, it must ABSTAIN rather than
    confabulate.
    """
    rungs = []
    for N in sorted(cells_by_N):
        cells = cells_by_N[N]

        def _mean(arm: str, field: str) -> float:
            vals = [c[arm][field] for c in cells]
            return float(sum(vals) / len(vals)) if vals else 0.0

        rungs.append({
            "N": int(N),
            "n_seeds": int(n_seeds),
            "full_acc": _mean("full", "acc"),
            "rhythm_removed_acc": _mean("rhythm_removed", "acc"),
            "abstain_correct_rhythm_removed": _mean(
                "rhythm_removed", "abstain_correct"
            ),
        })
    return rungs


def run_spear_conversational(seeds, loads=_SP_LADDER,
                             tiny_synth: bool = False,
                             out_path: Optional[str] = None,
                             ckpt: Optional[str] = None) -> Dict[str, Any]:
    """Run the shared theta-gamma SPEAR conversational capability test.

    Per seed, per load N in the frozen ladder: build the validated
    substrate + hippocampus + dlpfc PFC frame, then run the `full` arm
    (the net-new shared-rhythm theta-gamma controller ENABLED: ACh-gated
    encode/retrieve phase multiplexing + gamma-indexed dlpfc slot +
    generative replay loop) and the decisive built-in control
    `rhythm_removed` arm (controller DISABLED -> reduces to the Stage-1
    static composition), SAME seed and SAME draws, differing ONLY by the
    `use_rhythm` flag. Aggregate to rungs and score with the FROZEN
    spear_conversational_verdict.

    Kill-safe/resumable via the REUSED sim.train_checkpoint: completed
    (seed, N) cells are flushed; re-running resumes past them.
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)

    cells: List[Dict[str, Any]] = []
    start = 0
    schedule = [(s, N) for s in seeds for N in loads]
    if ckpt:
        prev = load_checkpoint(ckpt)
        if prev is not None:
            start = resume_epoch(prev)
            blob = prev.get("weights", [None])[0]
            if blob is not None:
                try:
                    cells = json.loads(
                        bytes(np.asarray(blob)).decode("utf-8")
                    )
                except Exception:
                    cells = []

    try:
        for epoch in range(start, len(schedule)):
            s, N = schedule[epoch]
            cell = _cell(s, N, tiny_synth)
            cells.append({"seed": s, **cell})
            if ckpt:
                blob = np.frombuffer(
                    json.dumps(cells).encode("utf-8"), dtype=np.uint8
                )
                save_checkpoint(ckpt, epoch, {"cells": [blob]}, None, [])
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable",
              flush=True)
        if not cells:
            raise

    cells_by_N: Dict[int, List[Dict[str, Any]]] = {}
    seeds_seen_by_N: Dict[int, set] = {}
    for c in cells:
        cells_by_N.setdefault(c["N"], []).append(c)
        seeds_seen_by_N.setdefault(c["N"], set()).add(c["seed"])

    if seeds_seen_by_N:
        n_seeds = min(len(v) for v in seeds_seen_by_N.values())
    else:
        n_seeds = 0
    rungs = _aggregate(cells_by_N, n_seeds)
    verdict = spear_conversational_verdict(rungs)

    result = {
        "rungs": rungs,
        "verdict": verdict,
        "tiny_synth": bool(tiny_synth),
        "seeds": seeds,
        "loads": list(loads),
        "raw_cells": cells,
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only."
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Shared theta-gamma SPEAR conversational runner "
                    "(Architecture A; net-new rhythm controller; "
                    "reuse-only; no autograd)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--loads", type=int, nargs="+",
                    default=list(_SP_LADDER),
                    help="Load ladder (default the frozen ladder).")
    ap.add_argument("--tiny-synth", action="store_true",
                    help="Shrink pools/episodes/phase-blocks (+ NumPy "
                         "backend when CuPy is unavailable) for the "
                         "logic-screen smoke. Toy numbers are NOT a "
                         "result.")
    ap.add_argument("--ckpt", default=None,
                    help="Kill-safe checkpoint path (REUSED "
                         "sim.train_checkpoint; re-run resumes).")
    ap.add_argument("--out", default=None,
                    help="Write the full result JSON here.")
    a = ap.parse_args(argv)

    result = run_spear_conversational(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        out_path=a.out,
        ckpt=a.ckpt,
    )
    g = result["verdict"]["gate"]
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    print("GATE=%s%s" % (g, tag), flush=True)
    print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
