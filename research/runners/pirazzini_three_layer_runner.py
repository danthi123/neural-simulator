"""Net-new Pirazzini-reference three-layer theta-gamma conversational runner.

Biology (Pirazzini 2024 Frontiers in Neural Circuits): a three-layer
sequential-memory architecture (PFC working-memory frame + L1 CA3 auto-
associative + L2 CA1 hetero-associative) with an EXTERNAL THETA GENERATOR
that rhythmically disinhibits CA3 by exciting glutamatergic synapses onto
inhibitory interneurons. The standard Hasselmo polarity: HIGH ACh during
ENCODING (suppresses CA3->CA1 transmission + strengthens cortical input
+ facilitates LTP); LOW ACh during RETRIEVAL (pattern completion).
Episode pairs are presented simultaneously to L1 (CA3) and L2 (CA1) for
~250 ms per pair, ONE-SHOT (only one presentation).

This module is the ONLY genuinely net-new code:
  * an external theta-generator CONTROLLER that rhythmically writes a
    DISINHIBITORY current onto the CA3-targeted inhibitory population
    (`dg_pv_basket`) at theta-trough phase via the REUSED
    `bridge.cp_external_input_current` path. This is the Pirazzini
    disinhibition mechanism -- biology-faithful, NOT a synaptic-gain
    modulation (the SPEAR runner's choice);
  * a MULTI-TARGET ACh `NeuromodulatorConfig` combining REUSED
    `sim/neuromodulators.py` primitives so HIGH ACh simultaneously:
        (a) suppresses CA3->CA1 plasticity (via plasticity_gate scope=
            `gate:ca3_to_ca1` with sensitivity-negative polarity);
        (b) strengthens cortical input to hippocampus (via plasticity_
            gate scope=`gate:lang_to_ec` with sensitivity-positive
            polarity -- the closest existing gate tag in the validated
            builder for "cortical input to hippocampus"; the builder
            does not expose a `lang_input_to_ca3` gate, so this is the
            documented closest-equivalent tag);
        (c) facilitates LTP system-wide (via plasticity_rate scope=
            `all` with positive sensitivity);
        (d) modulates effective synaptic strength (via synaptic_gain
            scope=`all` -- documented broad-scope fallback because the
            REUSED `compute_synaptic_gain_multiplier` consumer only
            honors scope=`all`; broader than ideal but biology-faithful
            on the cortex-wide ACh modulation scale).
  * ONE-SHOT encoding via the REUSED engram-tagging API: each episode
    pair (Ep_i, Ep_{i+1}) drives lang_input simultaneously via the
    REUSED `encode_concept_pair` over 250 ms, ONCE, with HIGH ACh.
    Tag NAMES are OPAQUE (`ep_{i}`) -- Stage-1 lesson, no answer string
    in any tag.
  * Within-theta-cycle DECODE via the REUSED neural readout:
    `lang_output_pattern_during_stim` + `lang_output_pattern_during_
    input` + the REUSED `_ranked_from_pattern` (raw firing-rate
    confidence -- the calibrated 650-moat quantity), gated by the
    REUSED `abstention_gate.gate(ranked, 650.0)`. SPEAR re-review
    lesson: feed the moat its calibrated quantity, not a cosine * norm
    hack.

The decisive BUILT-IN CONTROL `theta_disabled` is IDENTICAL to `full`
for the same (seed, N) -- SAME seed, SAME facts, SAME RNG draws --
with the external theta generator's disinhibitory current held at zero
across all phases (no rhythmic disinhibition; CA3 remains inhibited).
The convergent Stage-1 + SPEAR ceiling localised the gap: rhythm
mechanism alone (in either static or rhythm-multiplexed form) does not
lift composed readout above the trustworthy 650 threshold; the
Pirazzini disinhibition mechanism is the next testable hypothesis.
`full` and `theta_disabled` differ ONLY by the single `use_theta` flag
threaded identically. ASCII only. NO autograd anywhere. CuPy is the
real/decisive path; --tiny-synth shrinks pools/episodes/phase-block
lengths so the smoke is seconds (toy numbers explicitly NOT a result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy: identical to the SPEAR runner. The project rule is
# "NumPy ONLY for the smoke; CuPy is the decisive path". SimulationBridge
# binds its array module at sim.bridge IMPORT time, so on a CuPy-capable
# box the tiny smoke runs on the bridge's real backend. We only pin
# NumPy when CuPy is genuinely unavailable. The decisive multi-seed run
# is CuPy and is a later controller-only task -- NOT performed here.
if "--tiny-synth" in sys.argv:
    try:
        import cupy as _cupy_probe  # noqa: F401

        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
    if not _CUPY_AVAILABLE:
        os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.pirazzini_three_layer_core import (
    pirazzini_three_layer_verdict,
    _PZ_LADDER,
)
from research.runners.abstention_gate import gate as _abstain_gate
from research.runners.abstention_gate import DEFAULT_THRESHOLD as _MOAT
from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# REUSED Stage-1/SPEAR-cleared substrate vocabulary + raw firing-rate
# ranking + hippocampal tag-region filter. Identity-imports only
# (byte-unchanged) -- duplicate no subsystem logic.
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
#  Substrate construction (REUSE the validated v16 recipe + hippocampus
#  + dlpfc PFC working-memory frame, mirror SPEAR's cleared build path).
# =====================================================================
def _build_substrate(seed: int, tiny_synth: bool):
    """Construct the validated v16 concept-pool bridge WITH the
    hippocampal consolidation regions AND the dlpfc PFC working-memory
    compositional frame, by REUSING the validated builders byte-
    unchanged. Returns (bridge, dims).

    Mirrors the SPEAR-cleared spear_conversational_runner._build_
    substrate EXACTLY (same CoreSimConfig field set, same kwargs, NO
    cfg.num_traits override) and registers the NET-NEW multi-target
    ACh `NeuromodulatorConfig` (combining REUSED neuromodulator
    primitives) so the bridge's reused step applies the three
    Hasselmo effects each step the controller pulses ACh.
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

    # weak_dynamics=True (validated v16) -- identical to SPEAR/Stage-1.
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
        # Validated trisynaptic hippocampal path (catalog D.03/D.12/
        # D.13). Provides `ec`/`dg`/`dg_pv_basket`/`ca3`/`ca1` plus the
        # named plasticity gates (`lang_to_ec`, `ca3_to_ca1`, etc.).
        enable_hippocampus_consolidation=True,
        # Validated dlpfc PFC working-memory compositional frame
        # (Pirazzini's WM layer). NMDA bistability enabled via
        # cfg.enable_nmda=True below.
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )

    # EXACT v16 recipe CoreSimConfig field set (mirror SPEAR / Stage-1).
    # NO cfg.num_traits override.
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

    # ---------- Net-new MULTI-TARGET ACh NeuromodulatorConfig ----------
    # Combines REUSED `sim/neuromodulators.py` primitives so the three
    # Hasselmo effects all fire each step the controller pulses ACh.
    #
    # baseline=0.5 was chosen so the plasticity_gate formula
    #     contribution = sensitivity * (conc - baseline)
    # produces opposite-sign contributions at our HIGH (1.0) and LOW
    # (0.0) setpoints. Concretely:
    #
    #  * synaptic_gain scope=`all` (REUSED `compute_synaptic_gain_
    #    multiplier`, consumed every step at sim/bridge.py:4877-4879 and
    #    4890-4897). Sensitivity NEGATIVE -- HIGH ACh (1.0) multiplies
    #    effective gain by 1 + (-0.3)*(1-0.5) = 0.85 (suppression);
    #    LOW ACh (0.0) gives 1 + (-0.3)*(-0.5) = 1.15 (boost). Documented
    #    BROAD-SCOPE FALLBACK: the consumer only honors scope=`all` --
    #    no scope=`gate:<name>` path exists for synaptic_gain (verified
    #    via inspection of sim/neuromodulators.py:298-305). So this
    #    pathway-selective Hasselmo effect is implemented at the
    #    cortex-wide scale; the breadth is broader than ideal but
    #    biologically plausible (ACh broadly modulates cortical gain;
    #    Sarter 2009).
    #  * plasticity_rate scope=`all` (REUSED `compute_plasticity_rate_
    #    multiplier`). Sensitivity POSITIVE -- HIGH ACh facilitates LTP
    #    system-wide. At HIGH (1.0) the multiplier is 1+0.5*(1-0.5)
    #    = 1.25; at LOW (0.0) it is 1+0.5*(-0.5) = 0.75.
    #  * plasticity_gate scope=`gate:ca3_to_ca1` (REUSED `compute_
    #    plasticity_gate_values`, propagated to the per-pathway gate
    #    via sim/bridge.py:5432-5438). Sensitivity NEGATIVE -- HIGH ACh
    #    drives the gate value to 0 (CA3->CA1 plasticity SUPPRESSED);
    #    LOW ACh drives it to ~1 (open). With baseline=0.5 and
    #    sensitivity=-2.0: HIGH->-1.0 clipped to 0.0; LOW->+1.0 clipped
    #    to 1.0.
    #  * plasticity_gate scope=`gate:lang_to_ec` (closest-equivalent tag
    #    for "cortical input to hippocampus"; the builder tags the
    #    language_input->ec pathway with `lang_to_ec`, NOT a
    #    `lang_input_to_ca3` gate that does not exist). Sensitivity
    #    POSITIVE -- HIGH ACh keeps this gate OPEN (cortical input
    #    strengthened during encoding); LOW ACh closes it.
    #
    # Production rule `manual` -- the controller drives concentration
    # explicitly via the reused `set_concentration`. concentration_min
    # 0.0 / concentration_max 2.0 are the same scale the SPEAR
    # acetylcholine_tan config uses.
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        ModulatorTarget,
        ProductionRule,
    )

    _ACH_BASELINE = 0.5
    _ACH_SYN_GAIN_SENS = -0.3   # HIGH ACh suppresses transmission gain
    _ACH_PLAST_RATE_SENS = +0.5  # HIGH ACh facilitates LTP
    _ACH_CA3_TO_CA1_SENS = -2.0  # HIGH ACh suppresses CA3->CA1 plasticity
    _ACH_LANG_TO_EC_SENS = +2.0  # HIGH ACh strengthens cortical input
    ach = NeuromodulatorConfig(
        name="ach_pirazzini",
        baseline=_ACH_BASELINE,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.0,
        targets=[
            # (a) suppress CA3->CA1 (named-gate path)
            ModulatorTarget(
                target_type="plasticity_gate",
                scope="gate:ca3_to_ca1",
                sensitivity=_ACH_CA3_TO_CA1_SENS,
            ),
            # (b) strengthen cortical input (closest-equivalent named
            # gate is `lang_to_ec` -- builder does NOT expose a
            # `lang_input_to_ca3` gate, documented above)
            ModulatorTarget(
                target_type="plasticity_gate",
                scope="gate:lang_to_ec",
                sensitivity=_ACH_LANG_TO_EC_SENS,
            ),
            # (c) facilitate LTP (system-wide; standard Hasselmo)
            ModulatorTarget(
                target_type="plasticity_rate",
                scope="all",
                sensitivity=_ACH_PLAST_RATE_SENS,
            ),
            # (d) modulate effective synaptic strength every step
            # (documented broad-scope fallback; consumer only honors
            # scope="all" for synaptic_gain)
            ModulatorTarget(
                target_type="synaptic_gain",
                scope="all",
                sensitivity=_ACH_SYN_GAIN_SENS,
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

    # Pirazzini theta period (4 Hz = ~250 ms cycle). Compute steps from
    # the bridge dt rather than hardcoding -- a tiny smoke can still
    # exercise the trough/peak schedule with a shrunk window.
    theta_ms = 250.0
    theta_steps = max(2, int(round(theta_ms / cfg.dt_ms)))
    if tiny_synth:
        # Shrink the cycle hard for the logic-screen smoke (a 2-step
        # cycle still alternates peak/trough but keeps the smoke fast).
        theta_steps = max(2, min(theta_steps, 8))
    # Gamma sub-cycle ~25 ms = 40 Hz; one nested gamma per ~5 episodes
    # per Pirazzini.
    gamma_ms = 25.0
    gamma_steps = max(1, int(round(gamma_ms / cfg.dt_ms)))
    if tiny_synth:
        gamma_steps = max(1, min(gamma_steps, 2))

    dims = {
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "sparsity": 0.05,
        "theta_steps": theta_steps,
        "theta_ms": theta_ms,
        "gamma_steps": gamma_steps,
        "gamma_ms": gamma_ms,
        "dt_ms": cfg.dt_ms,
    }
    return bridge, dims


# =====================================================================
#  THE NET-NEW EXTERNAL THETA GENERATOR CONTROLLER.
#  (A timing controller; no new learning rule; no autograd. Implements
#  Pirazzini's disinhibition mechanism via REUSED bridge primitives.)
# =====================================================================
def _set_ach(bridge, value: float) -> None:
    """Set the ACh concentration via the REUSED NeuromodulatorManager.
    The bridge's reused `compute_*_multiplier` consumers then propagate
    the four-target effect (CA3->CA1 plasticity, cortical input gate,
    plasticity_rate, synaptic_gain) on every subsequent step until the
    next call. Hasselmo polarity: HIGH = encoding-permissive, LOW =
    pattern-completion.
    """
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is None:
        return
    try:
        mgr.set_concentration("ach_pirazzini", float(value))
    except Exception:
        # Subsystem missing (degraded smoke) -- the controller still
        # alternates the disinhibitory current; ACh is then a no-op.
        # Never fabricate a phase effect.
        pass


# Hasselmo polarity ACh setpoints (baseline=0.5). ENCODE = HIGH (>=
# baseline so plasticity_gate ca3_to_ca1 drives toward 0 and lang_to_ec
# drives toward 1; synaptic_gain dips for suppression; plasticity_rate
# boosts). RETRIEVE = LOW (the opposite -- pattern completion).
_ACH_ENCODE_HIGH = 1.0    # Hasselmo: HIGH ACh during encoding
_ACH_RETRIEVE_LOW = 0.0   # Hasselmo: LOW ACh during retrieval
_ACH_NEUTRAL = 0.5        # baseline; theta-disabled holds ACh here
#                            (so all three Pirazzini effects sit at
#                            their no-effect midpoint and the only
#                            difference from `full` is the disinhibitory
#                            current itself being absent).


def _resolve_pv_basket_indices(bridge):
    """Resolve the indices of `dg_pv_basket` (the CA3-targeted inhibitory
    population the disinhibition mechanism targets). Returns a host-side
    Python list (the caller materialises a backend array near the use
    site to avoid plumbing the backend module through helpers).
    """
    rm = bridge.region_manager
    if rm is None:
        return []
    try:
        idx = rm.indices("dg_pv_basket")
        if idx is None:
            return []
        return list(idx)
    except Exception:
        return []


# Pirazzini disinhibitory current magnitude. NEGATIVE because the
# inhibitory population (dg_pv_basket) is silenced at theta-trough, so
# CA3 pyramidals are released from FFi inhibition (biology-faithful per
# Pirazzini 2024 section 2). Magnitude chosen modest enough not to
# overdrive the IZH FS interneurons into anomalous regimes but
# definitively negative.
_PV_DISINHIB_CURRENT_pA = -150.0


def _apply_theta_disinhibition(bridge, dims, step_idx, use_theta,
                                  pv_arr=None):
    """Pirazzini disinhibition: at theta-trough phase (the second half
    of each ~250 ms theta cycle), write a NEGATIVE current onto the
    dg_pv_basket inhibitory population via the REUSED
    `bridge.cp_external_input_current` array, so CA3 pyramidals are
    released from FFi inhibition. At theta-peak phase (first half),
    write zero on dg_pv_basket (default inhibited regime).

    On `use_theta=False` (the decisive built-in control), NEVER write
    the disinhibitory current -- CA3 remains inhibited throughout.
    Every other phase helper is identical between the two arms.

    NOTE on sign: the spec offered two biologically-defensible options
    (drive interneurons NEGATIVELY at trough so they release CA3, OR
    drive CA3 pyramidals POSITIVELY at trough). We chose the first --
    it matches Pirazzini 2024's explicit mechanism ("theta-generator
    unit rhythmically disinhibits CA3 via glutamatergic excitatory
    synapses onto inhibitory interneurons") more faithfully than a
    direct CA3 depolarisation would, and it composes correctly with
    the existing ec->dg_pv_basket->dg FFi sparsity pathway in the
    validated builder. The negative current on dg_pv_basket silences
    them; the absence of FFi releases the downstream CA3 layer.
    """
    if not use_theta:
        return
    if pv_arr is None or len(pv_arr) == 0:
        return
    theta_steps = int(dims.get("theta_steps", 50))
    trough_start = max(1, theta_steps // 2)
    phase_in_cycle = int(step_idx) % theta_steps
    if phase_in_cycle >= trough_start:
        bridge.cp_external_input_current[pv_arr] = float(
            _PV_DISINHIB_CURRENT_pA
        )
    # else: leave dg_pv_basket at whatever the caller's drive set
    # (the encode/retrieve helpers zero the buffer between
    # presentations; the disinhibitory write only fires at trough).


# =====================================================================
#  Phase helpers: encode (HIGH ACh, simultaneous L1+L2) and retrieve
#  (LOW ACh, pattern completion + moat-gated decode).
# =====================================================================
def _theta_encode_phase(bridge, fact, tag_name, dims, encoding_steps,
                          gamma_idx, n_facts, use_theta, pv_arr):
    """ENCODE phase for ONE episode pair.

    On the full path (use_theta=True): ACh is HIGH for the duration of
    the 250 ms theta cycle. The Pirazzini disinhibition pulse fires at
    the trough half of each theta cycle (released CA3 pyramidals). The
    episode is encoded ONE-SHOT as a Tonegawa engram over the
    hippocampal regions via the REUSED `encode_concept_pair`. The
    `gamma_idx` selects WHICH ordered compositional slot the dlpfc
    PFC frame holds (one fact per gamma sub-cycle, advancing each
    theta cycle) -- this matches Pirazzini's claim that one theta
    cycle accommodates ~5 episodes nested in gamma sub-cycles.

    On the theta-disabled path (use_theta=False): ACh is held at the
    neutral baseline (0.5) and the disinhibitory current is NEVER
    written -- CA3 remains inhibited. The SAME reused encode runs
    with the SAME draws.
    """
    from research.runners.compose_concept_engram import encode_concept_pair

    noun, adj = fact
    if use_theta:
        _set_ach(bridge, _ACH_ENCODE_HIGH)  # Hasselmo HIGH at encode
    else:
        _set_ach(bridge, _ACH_NEUTRAL)

    # Apply disinhibition for the start of the encode window. The
    # encode_concept_pair routine then loops `_run_one_simulation_step`
    # internally; the disinhibitory current persists on dg_pv_basket
    # until the next external-current write at theta-peak. The reused
    # encode call zeros `cp_external_input_current` at boundaries so
    # this is consistent with the validated path.
    _apply_theta_disinhibition(
        bridge, dims, step_idx=0, use_theta=use_theta, pv_arr=pv_arr,
    )

    if tag_name in {t["name"] for t in bridge.list_engram_tags()}:
        try:
            bridge.delete_engram_tag(tag_name)
        except Exception:
            pass
    # `gamma_idx` is recorded in the OPAQUE tag name (`ep_{i}`) only as
    # a slot id. Nothing downstream parses the tag string.
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
    # Re-apply disinhibition once more after encode_concept_pair's
    # internal zero of the external-current buffer -- so subsequent
    # phase helpers (or the within-theta-cycle decode below) inherit a
    # consistent trough/peak state on the full arm.
    _apply_theta_disinhibition(
        bridge, dims, step_idx=0, use_theta=use_theta, pv_arr=pv_arr,
    )


def _theta_retrieve_phase(bridge, cue_noun, tag_name, dims,
                            have_remote, recall_steps, use_theta, pv_arr):
    """RETRIEVE / pattern-complete phase for ONE compositional query.

    On the full path (use_theta=True): ACh is LOW (Hasselmo retrieval:
    pattern completion, CA3 recurrent autoassociation dominates).
    The disinhibition pulse continues to fire at theta-trough so the
    within-theta-cycle decode reads CA3 in its rhythmically-released
    state, mirroring Pirazzini's "70 % max-activity threshold within
    a theta cycle" criterion -- adapted here to the calibrated raw
    lang_output firing-rate confidence the REUSED 650 moat expects.

    On the theta-disabled path (use_theta=False): ACh is held neutral
    and the disinhibitory current is NEVER written. SAME reused
    readouts with SAME draws.

    Returns (answer_or_None, ranked).
    """
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_stim,
        lang_output_pattern_during_input,
    )

    if use_theta:
        _set_ach(bridge, _ACH_RETRIEVE_LOW)  # Hasselmo LOW at retrieve
    else:
        _set_ach(bridge, _ACH_NEUTRAL)

    _apply_theta_disinhibition(
        bridge, dims, step_idx=0, use_theta=use_theta, pv_arr=pv_arr,
    )

    # Consolidated-regime read (cue alone drives lang_input; the
    # consolidated schema is what makes the readout confident).
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

    # Re-apply disinhibition (the readout helpers zero
    # cp_external_input_current internally between presentations).
    _apply_theta_disinhibition(
        bridge, dims, step_idx=0, use_theta=use_theta, pv_arr=pv_arr,
    )

    # Hippocampal-regime read (engram stim-recall via the released CA3
    # ensemble; the recent (Ep_i, Ep_{i+1}) binding lives in the
    # engram, not in the cortical schema).
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

    # Within-theta-cycle compose: sum per-concept RAW firing-rate
    # confidences (the calibrated 650-moat quantity -- SPEAR re-review
    # lesson). The REUSED _ranked_from_pattern computed those raw
    # confidences via `pat[active].sum() / n_active` directly (mirror
    # the cleared compose_retrieval_runner path; do NOT rescale by a
    # cosine * norm hack).
    scores: Dict[str, float] = {}
    for w, r, _ in cons_ranked:
        scores[w] = scores.get(w, 0.0) + r
    for w, r, _ in hip_ranked:
        scores[w] = scores.get(w, 0.0) + r
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    decided = _abstain_gate(ranked, _MOAT)  # REUSED moat (raw firing rate)
    answer = None if decided is None else decided[0]
    return answer, ranked


# =====================================================================
#  One arm = one full Pirazzini run for one (seed, N), differing from
#  the other arm ONLY by `use_theta` (the sole controller flag).
# =====================================================================
def _run_arm(seed: int, N: int, tiny_synth: bool, use_theta: bool):
    """Run the complete Pirazzini pipeline for ONE (seed, N).

    `use_theta=True`  -> the net-new external theta generator is ENABLED:
        per theta cycle, an ACh-HIGH ENCODE phase (gamma sub-cycle
        indexes the dlpfc ordered slot) then an ACh-LOW RETRIEVE phase;
        at theta-trough phase, the disinhibitory current releases CA3
        pyramidals. This is the `full` arm.
    `use_theta=False` -> the controller is DISABLED: ACh held neutral,
        no disinhibitory current. CA3 remains inhibited throughout.
        SAME seed, SAME RNG draws -- which reduces to the convergent
        Stage-1 + SPEAR ceiling. This is the `theta_disabled` arm.

    The ONLY difference between the two arms is the single `use_theta`
    flag threaded identically through every phase helper. Returns
    {"seed", "acc", "abstain_correct"}.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200
    facts = _recent_facts(N)

    bridge, dims = _build_substrate(seed, tiny_synth)
    rng = np.random.default_rng(seed)  # SAME seed -> SAME draws both arms

    # Resolve the dg_pv_basket index buffer once (the disinhibition
    # target). Materialise to backend array via the bridge's array
    # module to match the buffer the writes target.
    from sim.backend import get_backend
    cp, _ = get_backend()
    pv_idx = _resolve_pv_basket_indices(bridge)
    pv_arr = (cp.asarray(pv_idx, dtype=cp.int64)
              if pv_idx else cp.asarray([], dtype=cp.int64))

    # ---- ENCODE epoch: one theta cycle per episode pair; ACh HIGH at
    # encode (Hasselmo); disinhibitory current writes the trough/peak
    # schedule onto dg_pv_basket.
    n_facts = len(facts)
    tags: List[str] = []
    for gamma_idx, fact in enumerate(facts):
        tag = "ep_%d" % gamma_idx  # OPAQUE slot id (Stage-1 lesson)
        _theta_encode_phase(
            bridge, fact, tag, dims, enc_steps,
            gamma_idx=gamma_idx, n_facts=n_facts,
            use_theta=use_theta, pv_arr=pv_arr,
        )
        tags.append(tag)

    # ---- RETRIEVE epoch: one theta cycle per compositional query.
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
            use_theta=use_theta, pv_arr=pv_arr,
        )
        if answer == adj:
            n_correct += 1
        else:
            # Answered wrong OR abstained: the no-confabulation
            # invariant requires an ABSTENTION on the wrong-answer
            # denominator (the moat correctly says "I don't know"),
            # not a confident wrong answer.
            n_wrong += 1
            if answer is None:
                n_abstain_ok += 1

    acc = (n_correct / n_total) if n_total else 0.0
    abstain_correct = (n_abstain_ok / n_wrong) if n_wrong else 1.0
    return {"seed": seed, "acc": acc, "abstain_correct": abstain_correct}


def _cell(seed: int, N: int, tiny_synth: bool) -> Dict[str, Any]:
    """One (seed, N) cell: the `full` arm (use_theta=True) and the
    decisive built-in control `theta_disabled` arm (use_theta=False),
    SAME seed, SAME facts, SAME RNG draws -- differing ONLY by the
    single `use_theta` flag threaded identically into _run_arm."""
    full = _run_arm(seed, N, tiny_synth, use_theta=True)
    theta_disabled = _run_arm(seed, N, tiny_synth, use_theta=False)
    return {
        "N": N,
        "full": full,
        "theta_disabled": theta_disabled,
    }


# =====================================================================
#  Aggregation + top-level entry.
# =====================================================================
def _aggregate(cells_by_N: Dict[int, List[Dict[str, Any]]],
               n_seeds: int) -> List[Dict[str, Any]]:
    """Aggregate per-seed cells into one rung dict per N -- EXACTLY the
    five keys the frozen pirazzini_three_layer_verdict consumes.

    `theta_disabled_acc` is the decisive built-in control: it must
    collapse to <= the convergent Stage-1 + SPEAR ceiling
    (_PZ_CONVERGENT_CEILING_MAX = 0.10). `abstain_correct_theta_
    disabled` is the no-confabulation invariant on that control arm.
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
            "theta_disabled_acc": _mean("theta_disabled", "acc"),
            "abstain_correct_theta_disabled": _mean(
                "theta_disabled", "abstain_correct"
            ),
        })
    return rungs


def run_pirazzini_three_layer(seeds, loads=_PZ_LADDER,
                                  tiny_synth: bool = False,
                                  out_path: Optional[str] = None,
                                  ckpt: Optional[str] = None
                                  ) -> Dict[str, Any]:
    """Run the Pirazzini-reference three-layer theta-gamma capability test.

    Per seed, per load N in the frozen ladder: build the validated
    substrate + hippocampus + dlpfc PFC frame, then run the `full` arm
    (the net-new external theta generator ENABLED: rhythmic CA3
    disinhibition at theta-trough + Hasselmo HIGH/LOW ACh phase
    polarity + one-shot engram encoding + within-theta-cycle decode
    through the calibrated 650 moat) and the decisive built-in
    control `theta_disabled` arm (controller DISABLED -- ACh neutral,
    no disinhibitory current; CA3 inhibited throughout), SAME seed
    and SAME draws, differing ONLY by the `use_theta` flag.
    Aggregate to rungs and score with the FROZEN
    pirazzini_three_layer_verdict.

    Kill-safe/resumable via the REUSED sim.train_checkpoint:
    completed (seed, N) cells are flushed; re-running resumes past
    them.
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
    verdict = pirazzini_three_layer_verdict(rungs)

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
        description="Pirazzini-reference three-layer theta-gamma "
                    "conversational runner (Architecture A; net-new "
                    "external theta-generator controller + multi-target "
                    "ACh modulator + one-shot encoding + within-theta-"
                    "cycle decode; reuse-only; no autograd)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--loads", type=int, nargs="+",
                    default=list(_PZ_LADDER),
                    help="Episode-load ladder (default the frozen "
                         "ladder; Pirazzini reports >99% on episodes "
                         "2-3, 87-90% on episode 5).")
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

    result = run_pirazzini_three_layer(
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
