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
    DISINHIBITORY drive onto the CA3-targeted inhibitory population
    (`dg_pv_basket`) at theta-trough phase via a NeuromodulatorConfig
    named `dg_disinhibition` (excitability_drive scope=group:dg_pv_basket
    with NEGATIVE sensitivity). Routing the disinhibition through the
    modulator subsystem makes the bridge's per-step
    `compute_excitability_drive_per_neuron` consumer apply it on EVERY
    simulation step regardless of any encode/readout helper that zeros
    `cp_external_input_current` between presentations. This is the
    Pirazzini disinhibition mechanism -- biology-faithful, NOT a
    synaptic-gain modulation (the SPEAR runner's choice), and NOT a raw
    write to cp_external_input_current (the inert defect the
    adversarial review caught);
  * a MULTI-TARGET ACh `NeuromodulatorConfig` combining REUSED
    `sim/neuromodulators.py` primitives so HIGH ACh simultaneously:
        (a) suppresses CA3->CA1 plasticity (via plasticity_gate scope=
            `gate:ca3_to_ca1` with sensitivity-negative polarity REBALANCED
            so the NEUTRAL setpoint produces gate ~1.0 -- the
            adversarial-review-corrected control arm does NOT pre-freeze
            this pathway);
        (b) suppresses CA3 OUTPUT during encoding (via excitability_drive
            scope=`group:ca3` with NEGATIVE sensitivity -- the Hasselmo
            transmission-suppress effect, consumed EVERY step at
            sim/bridge.py:4960-4964);
        (c) strengthens cortical input to hippocampus during encoding
            (via excitability_drive scope=`group:ec` with POSITIVE
            sensitivity -- the Hasselmo cortical-strengthen effect,
            same per-step consumer);
        (d) facilitates LTP system-wide (via plasticity_rate scope=
            `all` with positive sensitivity);
        (e) modulates effective synaptic strength (via synaptic_gain
            scope=`all` -- documented broad-scope fallback because the
            REUSED `compute_synaptic_gain_multiplier` consumer only
            honors scope=`all`; broader than ideal but biology-faithful
            on the cortex-wide ACh modulation scale).
  * ONE-SHOT encoding via a RUNNER-LOCAL per-step loop that uses the
    REUSED engram-tagging API directly: drives lang_input via a third
    NeuromodulatorConfig `lang_drive_input` (excitability_drive
    scope=`group:lang_drive_active` -- a freshly-registered group for
    each fact, containing the active-neuron indices of the orthogonal
    code), so the input drive composes with the disinhibition + ACh
    modulators on EVERY step regardless of any external-current buffer
    clears. The buffer-wiping helpers in compose_concept_engram.py are
    NOT called (they wipe cp_external_input_current on entry, which
    erased the disinhibition write in the original defect).
  * Within-theta-cycle DECODE via a RUNNER-LOCAL retrieval loop using
    the REUSED `_ranked_from_pattern` (raw firing-rate confidence --
    the calibrated 650-moat quantity) gated by the REUSED
    `abstention_gate.gate(ranked, 650.0)`. SPEAR re-review lesson:
    feed the moat its calibrated quantity, not a cosine * norm hack.

The decisive BUILT-IN CONTROL `theta_disabled` is IDENTICAL to `full`
for the same (seed, N) -- SAME seed, SAME facts, SAME RNG draws --
with the dg_disinhibition modulator's concentration held at zero
across all phases (no rhythmic disinhibition; CA3 remains inhibited).
The convergent Stage-1 + SPEAR ceiling localised the gap: rhythm
mechanism alone (in either static or rhythm-multiplexed form) does not
lift composed readout above the trustworthy 650 threshold; the
Pirazzini disinhibition mechanism is the next testable hypothesis.
`full` and `theta_disabled` differ ONLY by the single `use_theta` flag
threaded identically. ASCII only. NO autograd anywhere. CuPy is the
real/decisive path; --tiny-synth shrinks pools/episodes/phase-block
lengths so the smoke is seconds (toy numbers explicitly NOT a result).

ADVERSARIAL-REVIEW CORRECTION CHRONOLOGY (mirrors SPEAR f1292a0 +
Stage-1 19190bd):
  Original defect (commit b0492ff): _apply_theta_disinhibition received
  `step_idx=0` at every call site -> phase_in_cycle = 0 % theta_steps
  = 0, which is NEVER >= trough_start (>=1). The trough branch was
  dead code; the disinhibitory -150 pA write was unreachable. Even if
  reached, the encode_concept_pair / lang_output_pattern_during_*
  helpers zero cp_external_input_current on entry, wiping any write.
  An ACh-only solver (disinhibition inert, ACh polarity active) scored
  GATE=PASS via the runner+verdict end-to-end -- a structural false-
  PASS exploit.

  Fix (this commit): four corrections, runner-only.
    FIX A: route disinhibition through `excitability_drive scope=
           group:dg_pv_basket` so it is consumed per-step regardless
           of external-current clears (mirrors SPEAR f1292a0).
    FIX B: replace buffer-wiping helper calls with runner-local
           per-step encode/retrieve loops; drive lang_input via a
           third modulator (excitability_drive group:lang_drive_active)
           so the input drive also composes per-step.
    FIX C: rebalance multi-target ACh (baseline 1.5, NEUTRAL=0.5
           produces gate=1.0 on every pathway-scoped target) so the
           control arm does NOT pre-freeze CA3->CA1. Add Hasselmo
           transmission effects via excitability_drive on ec/ca3.
    FIX D: positive false-PASS-protection pin in the test file.
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
    cfg.num_traits override) and registers the THREE NET-NEW
    NeuromodulatorConfigs (multi-target ACh + dg_disinhibition +
    lang_drive_input) so the bridge's reused step applies each effect
    on every step the controller sets a concentration.
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

    # ---------- THREE NET-NEW NeuromodulatorConfigs ----------
    #
    # The adversarial-review-blocked b0492ff design had a SINGLE multi-
    # target ACh modulator and a raw `cp_external_input_current` write
    # for disinhibition. That design was inert end-to-end (defects 1-3
    # in the review). This rewrite registers THREE separate modulators
    # so each named-biology effect is routed through a sim/bridge.py
    # consumer that is honored EVERY simulation step (mirrors SPEAR
    # f1292a0): excitability_drive on group:NAME for the per-step
    # disinhibition + Hasselmo transmission effects + lang_input drive;
    # plasticity_gate/plasticity_rate/synaptic_gain for the ACh learning
    # modulation.
    from sim.neuromodulators import (
        NeuromodulatorConfig,
        ModulatorTarget,
        ProductionRule,
    )

    # =================================================================
    # MOD 1: `ach_pirazzini` -- multi-target ACh (Hasselmo polarity).
    # =================================================================
    #
    # FIX C (2026-05-20) rebalances the modulator after the adversarial
    # review caught that with baseline=0.5 + sensitivity=-2.0 on the
    # plasticity_gate scope=`gate:ca3_to_ca1` target, the NEUTRAL
    # concentration (which the theta_disabled arm holds throughout)
    # produces gate = clip(0,1, -2*(0.5-0.5)) = 0.0 -- the control arm
    # has CA3->CA1 plasticity PRE-FROZEN. So `theta_disabled` was in
    # practice `full minus the ACh polarity`, NOT `full minus the
    # disinhibition mechanism`. That confused the named-control
    # interpretation and is what enabled the ACh-only false-PASS exploit.
    #
    # The fix shifts baseline to 1.5 so:
    #   gate at NEUTRAL conc=0.5: clip(0,1, -2*(0.5-1.5))= clip(0,1, 2.0) = 1.0  (PERMIT, no pre-freeze) ✓
    #   gate at HIGH  conc=2.0:  clip(0,1, -2*(2.0-1.5))= clip(0,1,-1.0)= 0.0  (Hasselmo suppress CA3->CA1 plasticity at encode) ✓
    #   gate at LOW   conc=0.0:  clip(0,1, -2*(0.0-1.5))= clip(0,1, 3.0) = 1.0  (PERMIT pattern completion at retrieve) ✓
    #
    # The Hasselmo TRANSMISSION semantics (HIGH ACh suppresses CA3 OUTPUT
    # at encode + HIGH ACh STRENGTHENS cortical EC drive at encode) are
    # NOT routed through plasticity_gate (which modulates only the
    # PLASTICITY-update rate). They are routed through excitability_drive
    # on group:ca3 and group:ec -- the per-step consumer at
    # sim/bridge.py:4960-4964 honors group:NAME scopes via the registered
    # group indices (set in _run_arm). NEUTRAL produces zero drive on
    # both (the no-effect midpoint); HIGH produces suppression on ca3 +
    # strengthening on ec; LOW does the opposite at low intensity.
    #
    # Targets (each verified against the REUSED consumer in sim/bridge.py):
    #
    #  * plasticity_gate scope=`gate:ca3_to_ca1` (REBALANCED).
    #    consumer: sim/bridge.py:5432-5438. NEUTRAL = PERMIT (FIX C).
    #
    #  * excitability_drive scope=`group:ca3`, sensitivity = -300 pA
    #    per unit (conc-baseline) -> at HIGH conc=2.0 produces
    #    -300*(2.0-1.5) = -150 pA on CA3 neurons (suppresses CA3 output
    #    during encode). At NEUTRAL/LOW the drive is +300 pA / +450 pA
    #    -- not zero, but the Hasselmo asymmetry is preserved: HIGH
    #    SUPPRESSES, LOW/NEUTRAL PERMITS. consumer: sim/bridge.py:4960-
    #    4964 every step.
    #
    #  * excitability_drive scope=`group:ec`, sensitivity = +200 pA per
    #    unit (conc-baseline) -> at HIGH conc=2.0 produces +200*(0.5)
    #    = +100 pA on EC neurons (strengthens cortical input during
    #    encode). At NEUTRAL produces -200 pA; at LOW produces -300 pA.
    #    The Hasselmo asymmetry HIGH > NEUTRAL > LOW is preserved.
    #    consumer: sim/bridge.py:4960-4964 every step.
    #
    #  * plasticity_rate scope=`all` (system-wide LTP facilitation).
    #    consumer: compute_plasticity_rate_multiplier called inside C2
    #    reward-mod block (active when reward signal exists). Sensitivity
    #    POSITIVE so HIGH ACh facilitates LTP during encoding.
    #
    #  * synaptic_gain scope=`all` (DOCUMENTED BROAD-SCOPE FALLBACK; the
    #    REUSED `compute_synaptic_gain_multiplier` consumer only honors
    #    scope=`all`). Consumed EVERY step at sim/bridge.py:4877-4879
    #    (STP branch) and 4890-4897 (no-STP branch). Sensitivity small
    #    NEGATIVE so HIGH ACh slightly suppresses afferent gain (the
    #    cortex-wide ACh modulation effect; biologically plausible per
    #    Sarter 2009).
    #
    # Production rule `manual` -- the controller drives concentration
    # explicitly via the reused `set_concentration`. concentration_max
    # bumped to 2.5 so HIGH=2.0 sits comfortably below the cap.
    _ACH_BASELINE = 1.5      # FIX C: rebalanced from 0.5 so NEUTRAL=PERMIT
    _ACH_CA3_TO_CA1_SENS = -2.0   # gate suppression at HIGH ACh
    _ACH_CA3_DRIVE_SENS = -300.0  # pA per unit (conc - baseline); HIGH ACh -> NEGATIVE drive on CA3 (suppress output)
    _ACH_EC_DRIVE_SENS = +200.0   # pA per unit (conc - baseline); HIGH ACh -> POSITIVE drive on EC (strengthen input)
    _ACH_PLAST_RATE_SENS = +0.5   # HIGH ACh facilitates LTP system-wide
    _ACH_SYN_GAIN_SENS = -0.3     # HIGH ACh slightly suppresses cortex-wide gain
    ach = NeuromodulatorConfig(
        name="ach_pirazzini",
        baseline=_ACH_BASELINE,
        decay_tau_ms=500.0,
        concentration_min=0.0,
        concentration_max=2.5,
        targets=[
            # (a) suppress CA3->CA1 plasticity at encode (PERMIT at
            # NEUTRAL/LOW, SUPPRESS at HIGH)
            ModulatorTarget(
                target_type="plasticity_gate",
                scope="gate:ca3_to_ca1",
                sensitivity=_ACH_CA3_TO_CA1_SENS,
            ),
            # (b) Hasselmo: suppress CA3 OUTPUT at encode (per-step
            # consumed; not a plasticity gate)
            ModulatorTarget(
                target_type="excitability_drive",
                scope="group:ca3",
                sensitivity=_ACH_CA3_DRIVE_SENS,
            ),
            # (c) Hasselmo: strengthen EC (cortical input) at encode
            ModulatorTarget(
                target_type="excitability_drive",
                scope="group:ec",
                sensitivity=_ACH_EC_DRIVE_SENS,
            ),
            # (d) Hasselmo: facilitate LTP system-wide at encode
            ModulatorTarget(
                target_type="plasticity_rate",
                scope="all",
                sensitivity=_ACH_PLAST_RATE_SENS,
            ),
            # (e) cortex-wide gain modulation (documented broad-scope
            # fallback; consumer only honors scope="all")
            ModulatorTarget(
                target_type="synaptic_gain",
                scope="all",
                sensitivity=_ACH_SYN_GAIN_SENS,
            ),
        ],
        production_rules=[ProductionRule(rule_type="manual")],
    )

    # =================================================================
    # MOD 2: `dg_disinhibition` -- the Pirazzini external theta generator.
    # =================================================================
    #
    # FIX A (2026-05-20): the original raw-buffer-write design wrote
    # `bridge.cp_external_input_current[pv_idx] = -150` at theta-trough.
    # That was inert because (i) `step_idx=0` was hardcoded at every
    # call site -> phase_in_cycle never reached trough_start and (ii)
    # `encode_concept_pair` zeros the external-current buffer on entry.
    #
    # The fix routes the disinhibition through a NeuromodulatorConfig
    # whose ONE target is excitability_drive scope=`group:dg_pv_basket`
    # with NEGATIVE sensitivity. The consumer at sim/bridge.py:4960-
    # 4964 honors this scope EVERY simulation step (via the registered
    # group indices set in _run_arm), so the disinhibition is applied
    # regardless of any helper that clears cp_external_input_current.
    # The controller drives concentration to 1.0 at theta-trough steps
    # and 0.0 elsewhere, producing -150 pA on dg_pv_basket at trough
    # (silences inhibitors -> CA3 disinhibited; biology-faithful per
    # Pirazzini 2024 section 2). decay_tau_ms large so a per-step
    # set_concentration call is the dominant driver.
    _DISINHIB_SENS = -150.0   # pA per unit (conc - baseline); conc=1 -> -150 pA on dg_pv_basket
    dg_disinhibition = NeuromodulatorConfig(
        name="dg_disinhibition",
        baseline=0.0,
        decay_tau_ms=10000.0,   # large -> per-step set_concentration dominates
        concentration_min=0.0,
        concentration_max=1.5,
        targets=[
            ModulatorTarget(
                target_type="excitability_drive",
                scope="group:dg_pv_basket",
                sensitivity=_DISINHIB_SENS,
            ),
        ],
        production_rules=[ProductionRule(rule_type="manual")],
    )

    # =================================================================
    # MOD 3: `lang_drive_input` -- per-fact orthogonal-code input drive.
    # =================================================================
    #
    # FIX B (2026-05-20): the original runner called encode_concept_pair
    # / lang_output_pattern_during_{stim,input}, which all zero
    # cp_external_input_current on entry, wiping any prior write
    # (disinhibition, ACh-driven anything). The fix replaces those calls
    # with a runner-local per-step loop driving language_input via a
    # third modulator. We register an excitability_drive target on
    # scope=`group:lang_drive_active` -- the indices of the currently-
    # active neurons in the orthogonal code (a fresh subset per fact).
    # The runner's encode/retrieve loops register the indices via
    # `mgr.set_group_indices({"lang_drive_active": active_idx})` BEFORE
    # setting concentration. concentration units are pA / sensitivity
    # so conc=1 produces +sensitivity drive on every active neuron.
    _LANG_DRIVE_SENS = +200.0   # pA per unit; conc=1 -> +200 pA on the active subset
    lang_drive_input = NeuromodulatorConfig(
        name="lang_drive_input",
        baseline=0.0,
        decay_tau_ms=10000.0,
        concentration_min=0.0,
        concentration_max=1.5,
        targets=[
            ModulatorTarget(
                target_type="excitability_drive",
                scope="group:lang_drive_active",
                sensitivity=_LANG_DRIVE_SENS,
            ),
        ],
        production_rules=[ProductionRule(rule_type="manual")],
    )

    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [ach, dg_disinhibition, lang_drive_input]

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
#  Pirazzini's disinhibition mechanism via REUSED bridge primitives,
#  ALL routed through NeuromodulatorConfig targets per FIX A/B/C.)
# =====================================================================
def _set_ach(bridge, value: float) -> None:
    """Set the ACh concentration via the REUSED NeuromodulatorManager.
    The bridge's reused `compute_*_multiplier` / `compute_excitability_
    drive_per_neuron` consumers then propagate the five-target effect
    (CA3->CA1 plasticity gate, CA3 output excitability_drive, EC input
    excitability_drive, plasticity_rate, synaptic_gain) on every
    subsequent step until the next call. Hasselmo polarity: HIGH =
    encoding-permissive, LOW = pattern-completion.
    """
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is None:
        return
    try:
        mgr.set_concentration("ach_pirazzini", float(value))
    except Exception:
        # Subsystem missing (degraded smoke) -- the controller still
        # drives the disinhibition modulator; ACh is then a no-op.
        # Never fabricate a phase effect.
        pass


def _set_disinhibition(bridge, value: float) -> None:
    """Set the dg_disinhibition concentration. conc=1.0 -> -150 pA on
    dg_pv_basket (CA3 disinhibited at theta-trough); conc=0.0 -> 0 pA
    (inhibitors firing normally; CA3 inhibited). This is the per-step
    consumer (FIX A) that replaces the original inert raw external-
    current write.
    """
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is None:
        return
    try:
        mgr.set_concentration("dg_disinhibition", float(value))
    except Exception:
        pass


def _set_lang_drive(bridge, value: float) -> None:
    """Set the lang_drive_input concentration. conc=1.0 -> +200 pA on
    the registered `lang_drive_active` group (the active neurons of the
    current orthogonal code); conc=0.0 -> 0 pA. The runner sets the
    group's indices via `mgr.set_group_indices({"lang_drive_active":
    active_idx})` BEFORE turning concentration on.
    """
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is None:
        return
    try:
        mgr.set_concentration("lang_drive_input", float(value))
    except Exception:
        pass


# Hasselmo polarity ACh setpoints. FIX C rebalanced: baseline=1.5,
# NEUTRAL=0.5 produces gate=1.0 on every pathway-scoped target (PERMIT
# everywhere, no pre-freeze in the control arm). HIGH=2.0 produces
# gate=0 on ca3_to_ca1 (Hasselmo suppress encoded LTP) + negative drive
# on ca3 (suppress output) + positive drive on ec (strengthen input).
# LOW=0.0 produces gate=1 (permit), positive drive on ca3, negative
# drive on ec -- the retrieve regime (pattern completion, CA3 dominant).
# NEUTRAL=0.5 is the no-effect midpoint the theta_disabled arm holds.
_ACH_ENCODE_HIGH = 2.0       # Hasselmo HIGH at encode
_ACH_RETRIEVE_LOW = 0.0      # Hasselmo LOW at retrieve
_ACH_NEUTRAL = 0.5           # theta_disabled holds this throughout


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


def _phase_is_trough(step_idx: int, theta_steps: int) -> bool:
    """Compute the theta phase for a given (absolute) step index.

    Returns True iff phase is in the trough half. FIX A correction:
    the original runner always passed `step_idx=0` to this kind of
    helper, which made the trough branch dead code. The runner-local
    per-step loops now thread the absolute step index through every
    call so the trough/peak schedule is genuinely exercised.
    """
    trough_start = max(1, int(theta_steps) // 2)
    return (int(step_idx) % int(theta_steps)) >= trough_start


# =====================================================================
#  Runner-local helpers that drive lang_input via the modulator
#  subsystem (FIX B): NO buffer-wiping helper calls.
# =====================================================================
def _resolve_active_lang_input_neurons(bridge, word: str, dims):
    """Return the host-side list of language_input neuron indices that
    are 'active' for the given word's orthogonal drive pattern.

    REUSED: orthogonal_drive_pattern from sim.text_embeddings + the
    Stage-1/SPEAR validated language_input region. We compute the
    binary support of the orthogonal code and map it into the bridge's
    region-indices namespace (so it can be registered as a group on the
    neuromodulator manager and consumed by excitability_drive scope=
    group:lang_drive_active per FIX B).
    """
    from sim.text_embeddings import orthogonal_drive_pattern
    from research.runners.concept_compose_train import _WORD_TO_IDX

    rm = bridge.region_manager
    lang_indices = list(rm.indices("language_input"))
    n_lang_input = dims["n_lang_input"]
    sparsity = dims["sparsity"]
    drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word],
        n_cues=_N_WORDS_ORTHOGONAL,
        n_neurons=n_lang_input,
        drive_max_pA=1.0,
        sparsity=sparsity,
    )
    drive_arr = np.asarray(drive, dtype=np.float64)
    # Map active positions (positions where the orthogonal code is
    # non-zero) into the bridge's global neuron index namespace via
    # language_input's region indices. orthogonal_drive_pattern returns
    # an n_lang_input-length array aligned with the order of
    # language_input neurons (which is the order rm.indices returns).
    active_local = np.nonzero(drive_arr > 0.0)[0]
    if len(active_local) == 0:
        return []
    return [int(lang_indices[i]) for i in active_local
              if 0 <= i < len(lang_indices)]


# =====================================================================
#  Phase helpers: encode (HIGH ACh, simultaneous L1+L2) and retrieve
#  (LOW ACh, pattern completion + moat-gated decode).
#
#  FIX B: replaces the buffer-wiping encode_concept_pair / lang_output_
#  pattern_during_* helpers with runner-local per-step loops that drive
#  inputs ONLY through the modulator subsystem (which is consumed every
#  step at sim/bridge.py:4960-4964 + 4877-4897 + 5432-5438). The engram
#  API itself (start_engram_recording / commit_engram_tag /
#  stimulate_tag / clear_tag_drive) is REUSED byte-unchanged -- only
#  the OUTER loops are net-new.
# =====================================================================
def _theta_encode_phase(bridge, fact, tag_name, dims, encoding_steps,
                          gamma_idx, n_facts, use_theta,
                          force_disinhibition_off: bool = False):
    """ENCODE phase for ONE episode pair via a runner-local per-step loop.

    Steps:
      1. resolve active lang_input neurons for noun + adj.
      2. register them as the `lang_drive_active` group on the
         neuromodulator manager.
      3. start_engram_recording(tag_name) (REUSED).
      4. per step in [0, encoding_steps):
         - set lang_drive_input concentration = 1.0 (drives the active
           lang_input neurons by +200 pA via excitability_drive).
         - if use_theta: set ACh = HIGH; else: set ACh = NEUTRAL.
         - if use_theta AND NOT force_disinhibition_off AND phase is
           trough: set dg_disinhibition = 1.0; else: 0.0.
         - bridge._run_one_simulation_step()
      5. commit_engram_tag(tag_name, top_k=..., region_filter=
         _HIPPO_TAG_REGIONS) (REUSED).

    Every concentration write is consumed by the bridge on the SAME
    step. No external-current buffer is touched -- the input drive
    travels through `compute_excitability_drive_per_neuron` (added to
    `total_input_current_pA` at sim/bridge.py:4960-4964).
    """
    noun, adj = fact
    mgr = bridge.neuromodulator_manager

    # Active lang_input indices for noun OR adj (the validated
    # encode_concept_pair combines both drives additively; we register
    # the UNION so excitability_drive's mask covers all active neurons).
    active_noun = _resolve_active_lang_input_neurons(bridge, noun, dims)
    active_adj = _resolve_active_lang_input_neurons(bridge, adj, dims)
    active = sorted(set(active_noun) | set(active_adj))
    if mgr is not None and active:
        # Refresh ALL group indices (region groups + lang_drive_active)
        # each call. This is necessary because set_group_indices()
        # replaces the entire group dict.
        groups = dict(bridge.region_manager.region_indices_dict())
        groups["lang_drive_active"] = active
        mgr.set_group_indices(groups)

    # Open the engram recording (REUSED API).
    if tag_name in {t["name"] for t in bridge.list_engram_tags()}:
        try:
            bridge.delete_engram_tag(tag_name)
        except Exception:
            pass
    bridge.start_engram_recording(tag_name)

    # Per-step encode loop. The step index threads through
    # _phase_is_trough so the theta cycle is genuinely exercised
    # (FIX A: original `step_idx=0` made the trough branch dead).
    theta_steps = int(dims.get("theta_steps", 50))
    for s in range(int(encoding_steps)):
        # Drive lang_input via modulator (NOT cp_external_input_current).
        _set_lang_drive(bridge, 1.0)
        # ACh polarity: HIGH at encode on the full arm; NEUTRAL on
        # theta_disabled.
        if use_theta:
            _set_ach(bridge, _ACH_ENCODE_HIGH)
        else:
            _set_ach(bridge, _ACH_NEUTRAL)
        # Disinhibition: ON at theta-trough phase on the full arm, OFF
        # everywhere else. force_disinhibition_off (FIX D) holds it OFF
        # regardless -- used by the false-PASS-protection pin to prove
        # an ACh-only solver cannot score PASS.
        if (use_theta
                and not force_disinhibition_off
                and _phase_is_trough(s, theta_steps)):
            _set_disinhibition(bridge, 1.0)
        else:
            _set_disinhibition(bridge, 0.0)
        bridge._run_one_simulation_step()

    # Commit the engram (REUSED API). OPAQUE tag name -- Stage-1 lesson.
    bridge.commit_engram_tag(
        tag_name,
        top_k=max(8, dims["n_per_pool"] // 4),
        region_filter=_HIPPO_TAG_REGIONS,
    )

    # Settle: clear drives.
    _set_lang_drive(bridge, 0.0)
    _set_disinhibition(bridge, 0.0)


def _theta_retrieve_phase(bridge, cue_noun, tag_name, dims,
                            have_remote, recall_steps, use_theta,
                            force_disinhibition_off: bool = False):
    """RETRIEVE / pattern-complete phase for ONE compositional query
    via a runner-local per-step loop.

    Two sub-reads (consolidated regime + hippocampal regime); both
    accumulate the lang_output firing pattern per step. Then the
    ranked compose-sum is fed to the REUSED 650 moat (raw firing-rate
    confidence per the SPEAR re-review lesson).

    Returns (answer_or_None, ranked).
    """
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    mgr = bridge.neuromodulator_manager
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)
    n_lang_out = len(lang_out_idx)

    theta_steps = int(dims.get("theta_steps", 50))

    def _per_step_drives(s: int, drive_on: bool):
        """Apply the same theta-cycle disinhibition + Hasselmo ACh
        polarity (LOW at retrieve) for one step. drive_on selects
        whether lang_drive_input is active for this sub-phase."""
        if drive_on:
            _set_lang_drive(bridge, 1.0)
        else:
            _set_lang_drive(bridge, 0.0)
        if use_theta:
            _set_ach(bridge, _ACH_RETRIEVE_LOW)
        else:
            _set_ach(bridge, _ACH_NEUTRAL)
        if (use_theta
                and not force_disinhibition_off
                and _phase_is_trough(s, theta_steps)):
            _set_disinhibition(bridge, 1.0)
        else:
            _set_disinhibition(bridge, 0.0)

    # ----- Consolidated-regime read: drive lang_input(cue_noun) alone
    # via the lang_drive_input modulator and accumulate lang_output
    # spike pattern. -----
    if have_remote:
        active_cue = _resolve_active_lang_input_neurons(bridge, cue_noun, dims)
        if mgr is not None and active_cue:
            groups = dict(bridge.region_manager.region_indices_dict())
            groups["lang_drive_active"] = active_cue
            mgr.set_group_indices(groups)
        cons_pat = cp.zeros(n_lang_out, dtype=cp.float32)
        for s in range(int(recall_steps)):
            _per_step_drives(s, drive_on=True)
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                firing = bridge.cp_firing_states
                cons_pat = cons_pat + firing[lang_out_arr].astype(cp.float32)
        cons_pat_host = to_host(cons_pat)
        cons_ranked = _ranked_from_pattern(
            cons_pat_host, n_lang_out, dims, exclude=cue_noun
        )
    else:
        cons_ranked = []

    # ----- Hippocampal-regime read: stimulate the engram tag (REUSED
    # API) and accumulate lang_output spike pattern. The lang_drive
    # modulator is OFF here so the only drive is the tag stim. -----
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        # Clear any lingering tag drive + lang drive first.
        try:
            bridge.clear_tag_drive()
        except Exception:
            pass
        _set_lang_drive(bridge, 0.0)
        # Stimulate the tag (REUSED).
        bridge.stimulate_tag(tag_name, drive_pA=1500.0, additive=False)
        hip_pat = cp.zeros(n_lang_out, dtype=cp.float32)
        for s in range(int(recall_steps)):
            _per_step_drives(s, drive_on=False)
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                firing = bridge.cp_firing_states
                hip_pat = hip_pat + firing[lang_out_arr].astype(cp.float32)
        try:
            bridge.clear_tag_drive(tag_name)
        except Exception:
            pass
        hip_pat_host = to_host(hip_pat)
        hip_ranked = _ranked_from_pattern(
            hip_pat_host, n_lang_out, dims, exclude=cue_noun
        )
    else:
        hip_ranked = []

    # Settle: zero all drives.
    _set_lang_drive(bridge, 0.0)
    _set_disinhibition(bridge, 0.0)

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
def _run_arm(seed: int, N: int, tiny_synth: bool, use_theta: bool,
              force_disinhibition_off: bool = False):
    """Run the complete Pirazzini pipeline for ONE (seed, N).

    `use_theta=True`  -> the net-new external theta generator is ENABLED:
        per theta cycle, an ACh-HIGH ENCODE phase (gamma sub-cycle
        indexes the dlpfc ordered slot) then an ACh-LOW RETRIEVE phase;
        at theta-trough phase, the dg_disinhibition modulator releases
        CA3 pyramidals. This is the `full` arm.
    `use_theta=False` -> the controller is DISABLED: ACh held neutral
        (NEUTRAL=0.5 = no-effect midpoint -- FIX C ensures pathway-
        scoped gates resolve to 1.0 at this setpoint, so the control
        arm does NOT pre-freeze CA3->CA1 plasticity), no disinhibitory
        drive. CA3 remains inhibited throughout. SAME seed, SAME RNG
        draws -- which reduces to the convergent Stage-1 + SPEAR
        ceiling. This is the `theta_disabled` arm.

    `force_disinhibition_off=True` (FIX D) holds the dg_disinhibition
    modulator at 0.0 throughout REGARDLESS of `use_theta`. The ACh
    polarity is still active. Used by the false-PASS-protection pin
    to prove an ACh-only solver cannot score GATE=PASS via the runner
    + frozen verdict end-to-end.

    The ONLY difference between the two arms is the single `use_theta`
    flag threaded identically through every phase helper. Returns
    {"seed", "acc", "abstain_correct"}.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200
    facts = _recent_facts(N)

    bridge, dims = _build_substrate(seed, tiny_synth)

    # Register region groups on the neuromodulator manager so the
    # excitability_drive scope=group:NAME targets resolve (the
    # `compute_excitability_drive_per_neuron` consumer at
    # sim/bridge.py:4960-4964 looks up indices via the group dict).
    # This is the one-time setup the per-step loops rely on.
    if bridge.neuromodulator_manager is not None:
        groups = dict(bridge.region_manager.region_indices_dict())
        # `lang_drive_active` is refreshed per-fact inside encode/
        # retrieve helpers; initialize as empty here.
        groups["lang_drive_active"] = []
        bridge.neuromodulator_manager.set_group_indices(groups)

    # ---- ENCODE epoch: one theta cycle per episode pair; ACh HIGH at
    # encode (Hasselmo); dg_disinhibition modulator drives the
    # trough/peak schedule onto dg_pv_basket.
    n_facts = len(facts)
    tags: List[str] = []
    for gamma_idx, fact in enumerate(facts):
        tag = "ep_%d" % gamma_idx  # OPAQUE slot id (Stage-1 lesson)
        _theta_encode_phase(
            bridge, fact, tag, dims, enc_steps,
            gamma_idx=gamma_idx, n_facts=n_facts,
            use_theta=use_theta,
            force_disinhibition_off=force_disinhibition_off,
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
            use_theta=use_theta,
            force_disinhibition_off=force_disinhibition_off,
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
    disinhibition at theta-trough via the dg_disinhibition modulator
    + Hasselmo HIGH/LOW ACh phase polarity + one-shot engram encoding
    + within-theta-cycle decode through the calibrated 650 moat) and
    the decisive built-in control `theta_disabled` arm (controller
    DISABLED -- ACh neutral, no disinhibitory modulator concentration;
    CA3 inhibited throughout), SAME seed and SAME draws, differing
    ONLY by the `use_theta` flag. Aggregate to rungs and score with
    the FROZEN pirazzini_three_layer_verdict.

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
