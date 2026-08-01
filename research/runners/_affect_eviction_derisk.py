"""P0.3-E — AFFECT-MOOD **EVICTION**: can a slow negative feedback let a LATCHED affect pool turn OFF?

THE DEFECT (measured, not assumed). The P0.3 affect-state region holds a mood that PERSISTS, CAUSALLY
BIASES cognition, and is value-perpendicular (3 core faculties, 6-seed) -- but its 4th gate FAILS:
`history_MAGNITUDE_r_mean>=0.6` reads **0.326** (artifact `research/findings/raw/_affect_state_region_6seed.json`,
`GO:false`, `n_seeds_go:2`). The mechanism under that failure is a **RATCHET**: drive the mood HIGH once and
it never comes down. Held mood (the post-drive plateau, read 300-400 ms after drive-off) across the episode
sequence HIGH(1.0) -> LOW(0.15) -> LOW(0.15) -> SILENCE reads ~0.0942 / 0.0962 / 0.0904 / 0.0984 --
**100-102% of the HIGH value, 3/3 seeds**. The attractor latches UP with no way DOWN, so the sustained mood
cannot grade with appraised-valence magnitude no matter what arrives afterwards.

THE MECHANISM UNDER TEST (biology, engine-shipped, NO `sim/` edit). Slow **GABA_B -> GIRK feedback
inhibition**: each affect pool drives its OWN slow-feedback interneuron (`sfb_plus` / `sfb_minus` /
`sfb_arousal`), which inhibits **that same pool** through a `RegionPathway(receptor="gaba_b")`. GABA_B is
metabotropic: E_K = -90 mV (independent of the chloride gradient) and tau ~150 ms, i.e. ~15x slower than
GABA_A -- so it builds up only under SUSTAINED incumbent firing and decays slowly. That is delayed
self-inhibition = **incumbent fatigue at the synapse** (Tamas neurogliaform/SST slow IPSP; catalog B.15 /
J.11), the negative-feedback limb a pure WTA-hysteresis latch lacks. Engine support is pre-existing and
default-off: `cfg.enable_gabab`, `gabab_reversal_potential=-90`, `gabab_tau_decay=150`,
`gabab_propagation_strength=0.105`, `gabab_conductance_max` (GIRK saturation cap), and the per-synapse
routing mask built in `inject_explicit_wiring`.

RESEARCH GATE RUN BEFORE BUILDING (`tools/before_you_build.sh` + RAG, corpus=all). What the corpus says:
  * `2026-07-07-GNW-rung2-...-salience-eviction-PENDING.md` -- the SAME defect family, already named:
    "an established attractor is either un-evictable (locks in) or annihilates ... The missing ingredient is
    ADAPTATION / FATIGUE on the incumbent." This de-risk is that named-next-mechanism, on the affect region.
  * `2026-07-23-gap5-replay-candidate1-intrinsic-fatigue-alone-NEGATIVE-...md` -- intrinsic fatigue ALONE is
    NEGATIVE for **DIRECTING** replay ORDER, and its own diagnosis is "somatic adaptation mainly SILENCES;
    short-term depression DIRECTS." **Silencing is exactly what eviction needs**, so that negative does not
    transfer -- it is evidence FOR the fatigue family here. (Its companion note "u-adaptation cannot de-latch
    a self-regenerating DENDRITIC plateau" also does not transfer: this latch is a slow-NMDA recurrent.)
  * `2026-07-23-...-candidate2-STD-NEGATIVE` -- short-term depression DESTROYS a stored discrete chain. Not
    used here (nothing is stored in these weights; the state is dynamic), but it is why STD is not a lever.

============================== PRE-REGISTERED GATE (written BEFORE any run) ==============================
All quantities come from ONE ratchet trace per arm: episodes of [drive_ms @ level] -> [post_ms silence] ->
[read_ms silence, RECORDED]. `held[i]` = mood in episode i's read window (baseline-subtracted);
`during[i]` = mood while episode i's drive is on. Timings are IDENTICAL to the baseline runner's
`measure_persistence` (settle 40 / probe 100 / burst 120 / post 300 / probe 100), so `held[0]/during[0]` IS
the baseline's persistence-retention statistic and is directly comparable to its committed 0.62.

  LOW protocol   levels = [1.0, 0.15, 0.15, 0.0, 1.0]   (HIGH, two LOWs, silence, RE-IGNITE)
  HIGH protocol  levels = [1.0, 1.0,  1.0,  1.0, 1.0]   (the elapsed-TIME control)

  G1  EVICTION (the target)     max(held[1], held[2]) / held[0]  <  0.60   [today: 1.00-1.02]
  G2  DRIVE-DEPENDENCE          HIGH protocol: min(held[1], held[2]) / held[0] >= 0.60
                                (without G2, a mood that merely decays with TIME would pass G1 while
                                 restoring no graded drive-dependence at all)
  G3  RE-IGNITION               LOW protocol: held[4] / held[0] >= 0.60
                                (an evicted pool must be re-ignitable; eviction != destroying the attractor)
  G4  PERSISTENCE SURVIVES      held[0] / during[0] >= 0.50  with eviction ON
                                (the baseline's own persistence gate, unchanged; an eviction that flattens
                                 the latch is NOT a fix -- explicit anti-cheat from the task)
  G5  NMDA-OFF STAYS ~0         same protocol, nmda_on=False: held[0] / during[0] < 0.10
                                (what is being evicted must still be the NMDA attractor)
  G6  LESION HAS POWER          SAME substrate, `evict_out` transmission gate = 0:
                                max(held[1], held[2]) / held[0] >= 0.90  (the ratchet REAPPEARS)
                                (a control that agrees with its treatment has no power; this is the
                                 instrument check, and it is byte-comparable -- identical neurons, identical
                                 wiring, only the eviction synapses' CURRENT is gated off)
  A5  VALIDITY PRECONDITION     during[0](evict ON) >= 0.50 x during[0](the UNTOUCHED baseline brain).
                                If the eviction merely CRUSHES the pool, every ratio above is UNDEFINED,
                                not a pass. Reported as UNDEFINED, never as a score. A second, independent
                                UNDEFINED guard applies to the RATIOS themselves: any ratio whose
                                denominator held[0] falls below 25% of the baseline held mood is None, not
                                a number (the first smoke printed +21.750 from held[0]=0.0004).

  6-SEED VERDICT: GO iff G1 & G2 & G3 & G4 hold on >= 5/6 seeds, with G5 and G6 (instrument) also >= 5/6.

  KILL CRITERION (pre-registered). If NO point in the swept (gabab_weight x gabab_tau x GIRK-cap) grid
  satisfies G1 & G2 & G3 & G4 with A5 valid -- i.e. every point either KEEPS the ratchet (G1 fails) or
  DESTROYS persistence / re-ignition (G4 or G3 fails) -- then **slow GABA_B feedback is KILLED as the
  eviction method for this defect** and the finding banks the failing method. THE LAW: that is a verdict on
  the METHOD, not on the CAPABILITY. The pre-registered NEXT method is **intrinsic spike-frequency
  adaptation on the affect-pool slice** (`cp_izh_d_increment` / `cp_izh_a`, the precedent is
  `_gap5_intrinsic_fatigue_replay_derisk.py:70-75`; corpus says adaptation SILENCES, which is what eviction
  needs), wired here as the opt-in `--sfa` arm and run under the SAME gate.

  CHARACTERIZATION ARM (not gated): the smoke's step 2 re-runs the SELECTED point with identical wiring but
  `enable_gabab=False`, so the same synapses deliver only their FAST GABA_A component (GABA_B is ADDITIVE in
  this engine -- a gaba_b-tagged synapse also feeds g_i). If the fast arm evicts equally, the attribution
  "the SLOWNESS is what evicts" is REFUTED and the honest claim shrinks to "added feedback inhibition
  evicts". If NEITHER arm evicts, the attribution is UNDEFINED -- there is no effect to attribute.

--------------------- SMOKE LOG (post-hoc, seed 42 ONLY -- NOT a result, no claim) ---------------------
Recorded here so the next session does not re-derive it. Every number below is in `research/findings/raw/`:
`_affect_eviction_widesweep_smoke.json` (10 pts), `_affect_eviction_finesweep_smoke.json` (28 pts),
`_affect_eviction_capsweep_smoke.json` (12 pts), `_affect_eviction_sfa400_smoke.json` (the SFA point),
`_affect_eviction_pathcheck.json` (a 2-seed battery-path check, NOT a verdict).
  * INSTRUMENT VERIFIED. The untouched baseline brain reproduces the ratchet (1.051) AND matches the
    committed artifact exactly: held[0] = 0.0938 vs its seed-42 `mood_ret_on` 0.0938, persistence
    held[0]/during[0] = 0.621 vs its `persistence_retention_nmda_on` 0.621. Same protocol, same numbers.
  * 50 swept GABA_B configurations on seed 42 (w 0.25-4.0 x tau 60-300 x cap 0/2/4/6) produced NO point
    passing G1-G4. The landscape is a CLIFF with no window: below it the brake RESCALES the latch (held
    0.0938 -> 0.0586 at w=0.25, ratchet ratio still ~1.00, persistence 0.46-0.66); above it the held state
    is ANNIHILATED (held[0] ~ 0 while the pool still ignites during drive, so G4 fails / ratio UNDEFINED).
  * WHY, from the trace: `g_gabab` on the pool is FLAT across all five read windows (e.g. 7.37 / 7.49 /
    7.41 / 7.37 / 7.41). The brake EQUILIBRATES to the pool's held firing rate inside episode 1 and then
    acts as a CONSTANT OFFSET -- it lowers the latch's height without making that height depend on the
    drive. A negative feedback driven by the INSTANTANEOUS rate cannot evict a latch whose held rate is
    the same in every episode; eviction needs a brake that integrates CUMULATIVE activity over a timescale
    longer than one episode. The GIRK cap does not change this (it only lowers the plateau, i.e. it is
    degenerate with the weight: at cap=2.0, g pins to 1.99 in every window and the ratio stays 0.97-1.01).
  * The `--sfa` fallback has NOT been properly swept. Its first probe (a=0.008, d=40) was MIS-SPECIFIED --
    the affect pools are RS with default d_increment=100, so d=40 LOWERED adaptation while the arm was
    labelled "cranked". A `lever()` check now prints before/after and warns on that inversion. One
    corrected point (a=0.008, d=400) annihilated the held state (same cliff shape). ONE point either side
    is not a sweep and no verdict on SFA is warranted yet.

DISCIPLINE: SIM_BACKEND=numpy (CPU lane). Reuse-by-import from `_affect_state_region_derisk`
(`AffectStateBrain` subclassed -- only `__init__` is overridden to APPEND the eviction regions/pathways;
`step` / `mood_rate` / `_set_appraisal` / `set_affect_lesion` are inherited unchanged, and the eviction
regions are appended LAST so neuron indices 0..359 and every prior wiring draw are preserved). That file is
NOT edited. NO `sim/` edit: `enable_gabab` + `RegionPathway(receptor=...)` + `transmission_gate` are all
pre-existing additive attributes, default-off.

Run (smoke, 1 seed):  SIM_BACKEND=numpy python -u -m research.runners._affect_eviction_derisk --smoke
Run (6-seed):         SIM_BACKEND=numpy python -u -m research.runners._affect_eviction_derisk \
                          --seeds 42 43 44 100 101 102 --gabab-weight W --gabab-tau T
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  (passthrough on numpy)
from tools.lab import lever, void_if  # noqa: E402  (executable discipline, not remembered rules)

# --- reuse-by-import: the VALIDATED baseline brain + its operating-point constants (that file is NOT edited)
from research.runners._affect_state_region_derisk import (  # noqa: E402
    AffectStateBrain, DEFAULT_RECUR_WEIGHT, RECUR_DENSITY, N_AFF, N_RECALL, N_ACC, N_WTA,
    DRIVE_GAIN_PA, APPRAISAL_TAU_MS, BIAS_WEIGHT, XINH_N, XINH_EXC_W, XINH_INH_W,
)

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_eviction_6seed.json"
BASELINE_ARTIFACT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_state_region_6seed.json"

# ---- eviction-loop constants -----------------------------------------------------------------------------
SFB_N = 15                  # slow-feedback interneurons per affect pool (matches the opponent XINH_N)
SFB_EXC_W = XINH_EXC_W      # affect pool -> its OWN slow-feedback interneuron (same weight as the opponent
                            # limb, so recruitment is on the same footing as the already-validated circuit)
SFB_EXC_DENSITY = 0.6
SFB_INH_DENSITY = 0.7
EVICT_GATE = "evict_out"    # ONE runtime transmission gate over every sfb -> pool synapse (the clean lesion)

# GABA_B weight scale. NB this is NOT comparable to the GABA_A weights (8-16) in the baseline: the GABA_B
# increment is w * gabab_propagation_strength (0.105) per presynaptic spike into a conductance with
# tau=150 ms, so its steady state is ~150x the per-step increment. Order-of-magnitude target: an
# I_gabab = g*(E_K - V) ~= -300 pA brake against the 500 pA appraisal drive needs g ~= 10, i.e. a per-step
# increment ~0.07 -- with ~10 connected sfb inputs firing at ~0.05 spk/ms that is w ~ 1.5. Swept, not assumed.
DEFAULT_GABAB_W = 1.5
DEFAULT_GABAB_TAU_MS = 150.0     # engine default; GABA_B/GIRK IPSP 150-300 ms
DEFAULT_GABAB_MAX = 0.0          # 0.0 = no GIRK saturation cap (engine default)

# ---- the ratchet protocol (timings CLONED from the baseline's measure_persistence, so held[0]/during[0]
#      is the SAME statistic as its committed persistence retention) ----------------------------------------
SETTLE_MS = 40
BASE_PROBE_MS = 100
DRIVE_MS = 120
POST_MS = 300
READ_MS = 100
EPISODE_AROUSAL = 0.6            # arousal co-drive while a valence drive is on (matches the baseline burst)
LOW_LEVELS = (1.0, 0.15, 0.15, 0.0, 1.0)
HIGH_LEVELS = (1.0, 1.0, 1.0, 1.0, 1.0)

# ---- ACTIVE-CLEAR / QUENCH GATE (the P0.3-E #1 mechanism after the outward-brake class -- GABA_B, STP, SFA --
#      was killed: a bistable slow-NMDA loop has no graded middle, so an outward brake either fails to evict or
#      annihilates+blocks re-ignition; a TRANSIENT open-loop clear works WITH the bistability instead). --------
DEFAULT_QUENCH_MS = 250.0        # clear-pulse duration; measured OFF-basin threshold is ~180 ms (NOT the ~100 ms
                                 # NMDA-decay estimate -- residual recovery/adaptation must drain too), so default
                                 # sits safely above it and inside POST_MS=300. Below ~180 ms the loop re-ignites
                                 # from OU noise on release. (Was 120.0 -- below threshold, would read as no-evict.)
                                 # Must fit inside the post gap (POST_MS=300) so the quench is entirely OFF during
                                 # the read window (the anti-cheat -- see ratchet_trace).

# ---- BRAIN-BASED ACTIVE-CLEAR (the deliverable: the host `step_quench` negative-current injection converted to a
#      SPIKING mechanism). A dedicated FS inhibitory pool `quench_fs` (IZH2007_FS_CORTICAL_INTERNEURON) is wired to
#      all three affect pools via STRONG GABA_A, gated at runtime behind the transmission gate `quench_out` (mirrors
#      evict_out), and RECRUITED during the clear window by a phasic manual neuromodulator `quench_drive`
#      (excitability_drive on group:quench_fs, mirrors the appraisal bus). The CLEAR is now caused by a spiking
#      inhibitory pool firing GABA_A onto the affect pools -- NOT host current on the affect pools. -----------------
QUENCH_GATE = "quench_out"       # ONE runtime transmission gate over every quench_fs -> affect synapse (the lesion)
QUENCH_DRIVE_MOD = "quench_drive"  # the phasic neuromodulator that recruits quench_fs (excitability_drive)
QUENCH_FS_N = 30                 # quench_fs FS interneurons (PV-basket-like; non-adapting high-freq)
QUENCH_GABA_DENSITY = 0.8        # quench_fs -> each affect pool connection density (dense -> strong pooled inhibition)
QUENCH_GABA_W = 15.0             # quench_fs -> affect pool GABA_A weight -- the CALIBRATED clear strength. This is the
                                 # load-bearing knob (seed 43 sweep 2026-08-01): the clear must drain the opponent
                                 # latch to the NEUTRAL OFF basin, NOT overshoot it. At w~15 (drive 150-400, ms
                                 # 180-300) the held mood falls to ~0.000 and RE-IGNITES positive (G1~0, G3~0.98). At
                                 # w>=25 the synaptic quench OVERSHOOTS the neutral basin and tips the opponent into
                                 # the V- attractor (held goes NEGATIVE, re-ignition flips negative, G3 fails) -- a
                                 # real brain-based finding the host -800 pA drain did not show (a shunting GABA_A
                                 # quench of an OPPONENT latch has an overshoot regime a raw current clamp lacks).
                                 # Swept on the pool. GABA_A reversal E_i=-75 mV (cfg default).
QUENCH_DRIVE_PA = 250.0          # excitability_drive sensitivity (pA into each quench_fs neuron at concentration 1.0).
                                 # FS rheobase ~50 pA (C=20, k=1.0) -> 250 pA drives sustained high-freq FS firing;
                                 # 150-400 all land neutral at w~15 (drive is secondary; the WEIGHT sets overshoot).
QUENCH_DRIVE_TAU_MS = 20.0       # quench_drive decay tau (matches appraisal); re-set to 1.0 each step while ON, and
                                 # EXPLICITLY released to 0.0 when the window ends (so conc==0 at the read -> the
                                 # spiking anti-cheat: no standing drive on quench_fs at read).


# =============================================================================================================
# The eviction brain = the VALIDATED P0.3 affect brain + a slow GABA_B self-feedback limb on each affect pool.
# Only __init__ is overridden. The eviction regions/pathways are APPENDED LAST so that
#   - neuron indices 0..359 (the baseline's 11 regions) are unchanged, hence the per-neuron heterogeneity /
#     firing-threshold draws for those neurons are unchanged, and
#   - every baseline pathway is sampled from the shared wiring RNG BEFORE any eviction pathway,
# which makes the eviction-OFF arm as close to the baseline substrate as an additive change can be. The
# LOAD-BEARING control is still the same-substrate `evict_out` gate=0 arm (G6), not this.
# =============================================================================================================
class EvictionAffectBrain(AffectStateBrain):
    def __init__(self, seed, nmda_on=True, recur_weight=DEFAULT_RECUR_WEIGHT, ou_pA=8.0,
                 gabab_w=DEFAULT_GABAB_W, gabab_tau_ms=DEFAULT_GABAB_TAU_MS, gabab_max=DEFAULT_GABAB_MAX,
                 enable_gabab=True, evict=True, sfa_a=None, sfa_d=None, with_eviction_wiring=True,
                 stp=False, stp_tau_d=None, stp_U=None,
                 brain_quench=False, quench_fs_n=QUENCH_FS_N, quench_gaba_w=QUENCH_GABA_W,
                 quench_drive_pA=QUENCH_DRIVE_PA):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway
        from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

        self.seed = int(seed)
        self.nmda_on = bool(nmda_on)
        self.opponent_style = "cross"
        self._evict = bool(evict)
        self._sfa = (None if (sfa_a is None or sfa_d is None) else (float(sfa_a), float(sfa_d)))
        self._with_eviction_wiring = bool(with_eviction_wiring)
        self._brain_quench = bool(brain_quench)
        self._quench_drive_pA = float(quench_drive_pA)

        cfg = CoreSimConfig()
        # ---- CLONED VERBATIM from AffectStateBrain.__init__ (the validated operating point) ----------------
        cfg.enable_brain_region_framework = True
        cfg.enable_neuromodulator_subsystem = True
        cfg.enable_nmda = bool(nmda_on)
        cfg.nmda_ratio = 0.5
        cfg.nmda_tau_decay = 100.0
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)                 # SEEDS THE SUBSTRATE (NOT actual_seed_used)
        cfg.stdp_w_max = 400.0
        cfg.hebbian_max_weight = 400.0
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = False
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = bool(stp)
        cfg.enable_per_type_stp = False   # use the GLOBAL stp_tau_d so --stp-tau-d actually governs (per-type defaults True and would override it)
        if stp and stp_tau_d is not None:
            cfg.stp_tau_d = float(stp_tau_d)
        if stp and stp_U is not None:
            cfg.stp_U = float(stp_U)   # LOWER U = gentler per-spike depression; tau_d {50-200} all annihilate at U=0.15
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(ou_pA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        # ---- END clone; everything below is the EVICTION addition -----------------------------------------
        cfg.enable_gabab = bool(enable_gabab and with_eviction_wiring)
        cfg.gabab_tau_decay = float(gabab_tau_ms)
        cfg.gabab_conductance_max = float(gabab_max)
        self._gabab_w = float(gabab_w)
        self._gabab_tau_ms = float(gabab_tau_ms)
        self._enable_gabab = bool(cfg.enable_gabab)

        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
        FS = "IZH2007_FS_CORTICAL_INTERNEURON"

        def aff(name):
            return BrainRegion(name=name, n_neurons=N_AFF, exc_fraction=1.0, internal_density=RECUR_DENSITY,
                               exc_weight_mean=float(recur_weight), inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=bool(nmda_on))

        def exc_pool(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        def fs_pool(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=FS)

        regions = [
            aff("affect_vplus"), aff("affect_vminus"), aff("affect_arousal"),
            fs_pool("inh_plus", XINH_N), fs_pool("inh_minus", XINH_N),
            exc_pool("recall_pos", N_RECALL), exc_pool("recall_neg", N_RECALL),
            BrainRegion(name="speak_acc", n_neurons=N_ACC, exc_fraction=1.0, internal_density=0.4,
                        exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                        izh_neuron_type=RS, enable_nmda=bool(nmda_on)),
            BrainRegion(name="silence_acc", n_neurons=N_ACC, exc_fraction=1.0, internal_density=0.4,
                        exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                        izh_neuron_type=RS, enable_nmda=bool(nmda_on)),
            fs_pool("wta_fs", N_WTA),
        ]
        G = "affect_out"
        pathways = [
            RegionPathway(from_region="affect_vplus", to_region="inh_plus", density=0.6, weight_mean=XINH_EXC_W,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="inh_plus", to_region="affect_vminus", density=0.7, weight_mean=XINH_INH_W,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="affect_vminus", to_region="inh_minus", density=0.6, weight_mean=XINH_EXC_W,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="inh_minus", to_region="affect_vplus", density=0.7, weight_mean=XINH_INH_W,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="affect_vplus", to_region="recall_pos", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            RegionPathway(from_region="affect_vminus", to_region="recall_neg", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            RegionPathway(from_region="affect_arousal", to_region="speak_acc", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            RegionPathway(from_region="speak_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="silence_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="wta_fs", to_region="speak_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="wta_fs", to_region="silence_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        ]

        # ---- the EVICTION limb, APPENDED LAST (see the class comment for why the order matters) ------------
        # pool -> its OWN slow-feedback interneuron -> SLOW GABA_B back onto the SAME pool. Delayed
        # self-inhibition: the interneuron tracks the pool's rate immediately, but the GIRK conductance it
        # opens integrates with tau=gabab_tau_ms, so the brake grows only under SUSTAINED incumbent firing.
        self._sfb_pairs = [("affect_vplus", "sfb_plus"), ("affect_vminus", "sfb_minus"),
                           ("affect_arousal", "sfb_arousal")]
        if with_eviction_wiring:
            for pool, sfb in self._sfb_pairs:
                regions.append(fs_pool(sfb, SFB_N))
            for pool, sfb in self._sfb_pairs:
                pathways.append(RegionPathway(from_region=pool, to_region=sfb, density=SFB_EXC_DENSITY,
                                              weight_mean=SFB_EXC_W, weight_jitter=0.1, plastic=False))
                pathways.append(RegionPathway(from_region=sfb, to_region=pool, density=SFB_INH_DENSITY,
                                              weight_mean=float(gabab_w), weight_jitter=0.1, plastic=False,
                                              receptor=("gaba_b" if enable_gabab else "gaba_a"),
                                              transmission_gate=EVICT_GATE))

        # ---- the BRAIN-BASED ACTIVE-CLEAR limb, APPENDED LAST (after the sfb limb, so ALL prior neuron indices
        #      and wiring draws are preserved). A dedicated spiking quench_fs FS pool projects STRONG GABA_A onto
        #      all three affect pools, behind the quench_out transmission gate (the lesion). It is RECRUITED by
        #      the phasic quench_drive neuromodulator (below). This is the spiking replacement for step_quench:
        #      the clear is a spiking inhibitory pool firing onto the affect pools, not host current on them.
        if brain_quench:
            regions.append(fs_pool("quench_fs", int(quench_fs_n)))
            for pool in ("affect_vplus", "affect_vminus", "affect_arousal"):
                pathways.append(RegionPathway(from_region="quench_fs", to_region=pool,
                                              density=QUENCH_GABA_DENSITY, weight_mean=float(quench_gaba_w),
                                              weight_jitter=0.1, plastic=False, receptor="gaba_a",
                                              transmission_gate=QUENCH_GATE))

        def appraisal_mod(name, group):
            return NeuromodulatorConfig(
                name=name, baseline=0.0, decay_tau_ms=APPRAISAL_TAU_MS,
                concentration_min=0.0, concentration_max=2.0,
                targets=[ModulatorTarget(target_type="excitability_drive", scope=f"group:{group}",
                                         sensitivity=DRIVE_GAIN_PA)],
                production_rules=[ProductionRule(rule_type="manual")])
        cfg.neuromodulators = [
            appraisal_mod("appraisal_vplus", "affect_vplus"),
            appraisal_mod("appraisal_vminus", "affect_vminus"),
            appraisal_mod("appraisal_arousal", "affect_arousal"),
        ]
        if brain_quench:
            # the phasic recruit for quench_fs: a manual neuromodulator delivering excitability_drive (pA) to the
            # quench_fs group. Pulsed to 1.0 while clearing (step_brain_quench) and released to 0.0 before the read.
            cfg.neuromodulators.append(NeuromodulatorConfig(
                name=QUENCH_DRIVE_MOD, baseline=0.0, decay_tau_ms=QUENCH_DRIVE_TAU_MS,
                concentration_min=0.0, concentration_max=1.0,
                targets=[ModulatorTarget(target_type="excitability_drive", scope="group:quench_fs",
                                         sensitivity=float(quench_drive_pA))],
                production_rules=[ProductionRule(rule_type="manual")]))
        cfg.brain_regions = regions
        cfg.region_pathways = pathways

        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._idx = {n: np.asarray(v, dtype=np.int64)
                     for n, v in self._bridge.region_manager.region_indices_dict().items()}
        self._apply_post_init()

    # ------------------------------------------------------------------ runtime state that reset() clears
    def _apply_post_init(self):
        """Re-apply everything `_initialize_simulation_data` wipes: the transmission gates (restored to 1.0)
        and, for the `--sfa` arm, the per-neuron Izhikevich adaptation params on the affect-pool slice."""
        if self._with_eviction_wiring:
            self._bridge.set_transmission_gate(EVICT_GATE, 1.0 if self._evict else 0.0)
        if self._brain_quench:
            # OPEN the quench limb (the gate is the structural lesion control, like evict_out; the ON/OFF of the
            # clear is the quench_drive neuromodulator, NOT this gate). Held OPEN through the read window so the
            # anti-cheat genuinely shows NO standing inhibition -- quench_fs is silent because it is UNDRIVEN,
            # not because its output was gated off.
            self._bridge.set_transmission_gate(QUENCH_GATE, 1.0)
        if self._sfa is not None:
            a_abs, d_abs = self._sfa
            b = self._bridge
            if getattr(b, "cp_izh_d_increment", None) is not None:
                idx = np.concatenate([self._idx[p] for p, _ in self._sfb_pairs])
                # RECORD what the manipulation actually changed. The RS default here is a=0.03,
                # d_increment=100 (sim/enums.py:593) -- the first SFA probe was run at d=40, which
                # *lowered* adaptation while the arm was labelled "cranked". Values borrowed from another
                # runner's pool are not a crank; only the before/after pair says which direction it moved.
                self.sfa_before = (float(np.asarray(to_host(b.cp_izh_a))[idx].mean()),
                                   float(np.asarray(to_host(b.cp_izh_d_increment))[idx].mean()))
                b.cp_izh_a[idx] = np.float32(a_abs)
                b.cp_izh_d_increment[idx] = np.float32(d_abs)
                self.sfa_after = (float(a_abs), float(d_abs))

    def reset(self):
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._apply_post_init()

    def set_eviction_lesion(self, lesion: bool):
        """Clamp the eviction limb's OUTPUT (zero the `evict_out` transmission gate). The sfb interneurons
        keep firing identically; only their synaptic current onto the affect pools is removed -- and because
        the GABA_B increment is taken from the SAME gated connection matrix, g_gabab stays at 0 too.

        ⛔ THIS LESION DOES NOT REACH THE `--sfa` ARM. It gates a SYNAPTIC pathway; intrinsic spike-frequency
        adaptation lives in per-neuron Izhikevich parameters, which no transmission gate can touch. Measured
        2026-07-31: G6 ("the lesion restores the ratchet") failed on 2 of 3 sAHP settings for that structural
        reason, and the third setting's PASS was the synaptic limb still being gateable -- a different claim
        from "the sAHP arm was controlled". Use `set_sfa_lesion` for the sAHP arm; see FAILURE_LOG."""
        self._evict = not bool(lesion)
        if self._with_eviction_wiring:
            self._bridge.set_transmission_gate(EVICT_GATE, 0.0 if lesion else 1.0)

    def set_sfa_lesion(self, lesion: bool):
        """THE POWER CONTROL FOR THE INTRINSIC ARM: restore the per-neuron adaptation parameters to the values
        they held BEFORE `--sfa` modified them, on the SAME substrate, same neurons, same wiring.

        This is the sAHP counterpart of the `evict_out` gate, and it is the arm that was missing: without it
        an sAHP result is UNCONTROLLED rather than negative, because nothing demonstrates the instrument can
        see the ratchet return when the mechanism is removed. `sfa_before` is already captured by
        `_apply_post_init`, so the control costs one write and needs no new measurement.

        Returns True if it acted; False if this arm carries no sfa (nothing to lesion)."""
        b = self._bridge
        if self._sfa is None or getattr(self, "sfa_before", None) is None:
            return False
        if getattr(b, "cp_izh_d_increment", None) is None:
            return False
        idx = np.concatenate([self._idx[p] for p, _ in self._sfb_pairs])
        a_val, d_val = self.sfa_before if lesion else self._sfa
        b.cp_izh_a[idx] = np.float32(a_val)
        b.cp_izh_d_increment[idx] = np.float32(d_val)
        self._sfa_lesioned = bool(lesion)
        # ASSERT the write landed. A silent no-op here would produce a "control" identical to the treatment,
        # which is the failure class this whole arm exists to close.
        got = (float(np.asarray(to_host(b.cp_izh_a))[idx].mean()),
               float(np.asarray(to_host(b.cp_izh_d_increment))[idx].mean()))
        assert abs(got[0] - a_val) < 1e-6 and abs(got[1] - d_val) < 1e-3, (
            "set_sfa_lesion(%s) did not take: wanted (%s, %s), read %s" % (lesion, a_val, d_val, got))
        return True

    # ------------------------------------------------------------------ diagnostics (mechanism assertion)
    def mean_gabab(self, region="affect_vplus"):
        """Mean slow-GIRK conductance on a pool. This is the LEVER read: it must be 0 with the eviction
        lesioned and > 0 with it intact, or the A/B is void."""
        g = getattr(self._bridge, "cp_conductance_g_gabab", None)
        if g is None:
            return 0.0
        return float(np.asarray(to_host(g))[self._idx[region]].mean())

    def mean_v(self, region="affect_vplus"):
        return float(np.asarray(to_host(self._bridge.cp_membrane_potential_v))[self._idx[region]].mean())

    # ------------------------------------------------------------------ the ACTIVE-CLEAR quench
    def _affect_idx(self):
        """Concatenated neuron indices of the three affect pools (the clear targets the whole state)."""
        return np.concatenate([self._idx["affect_vplus"], self._idx["affect_vminus"],
                               self._idx["affect_arousal"]])

    def step_quench(self, n_steps, quench_pA):
        """ACTIVE-CLEAR quench: run `n_steps` of otherwise-SILENT simulation while injecting a constant
        `quench_pA` (a STRONG NEGATIVE current) into the three affect-pool neuron slices -- nothing else is
        driven (this only ever fires in the post-drive silence). Returns per-pool spike counts for diagnostics.

        The base `step()` zeroes `cp_external_input_current[:]` at the top of EVERY internal iteration, so a
        one-shot write before `step()` would be wiped; the injection must live inside the loop, which is why
        this override exists rather than a pre-`step` current write.

        The physics under test (the 2026-08-01 reframe): the affect ratchet is a SATURATED BISTABLE slow-NMDA
        attractor. Pushing the pool's firing to ~0 for longer than the recurrent-NMDA decay (~100 ms) collapses
        the reverberatory drive; the OFF (down-state) fixed point then holds the state with ZERO standing force.
        Persistence in normal operation is untouched (the clear is OFF then) and re-ignition survives (unlike
        STP, the synapses are left fully recovered). This is Compte-Wang (2000) persistent-activity termination.

        DOCUMENTED HOST SHORTCUT (to be biologized). The decision to fire AND the current itself are issued by
        HOST CODE here -- the 'clear command'. The brain-based replacement is a dedicated `quench_fs`
        interneuron pool that a control input recruits to silence the affect pools SYNAPTICALLY. This lever
        answers the PHYSICS -- does an open-loop transient silence collapse THIS loop and leave it OFF? --
        before that pool is wired. It is NOT a graded brake (those were killed); it is all-or-none by design.
        """
        b = self._bridge
        aff_idx = self._affect_idx()
        q = np.float32(quench_pA)
        counts = {"affect_vplus": 0.0, "affect_vminus": 0.0}
        for _ in range(int(n_steps)):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[aff_idx] = q
            b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            counts["affect_vplus"] += float(fs[self._idx["affect_vplus"]].sum())
            counts["affect_vminus"] += float(fs[self._idx["affect_vminus"]].sum())
        return counts

    # ------------------------------------------------------------------ the BRAIN-BASED active-clear
    def set_quench_drive(self, level):
        """Set the phasic neuromodulator that recruits the quench_fs FS pool (excitability_drive on
        group:quench_fs). level=1.0 -> full drive (clear ON); 0.0 -> released (clear OFF). No-op if this
        brain has no brain-quench limb."""
        if not self._brain_quench:
            return
        self._bridge.neuromodulator_manager.set_concentration(QUENCH_DRIVE_MOD, float(level))

    def quench_drive_conc(self):
        """The quench_drive neuromodulator concentration RIGHT NOW -- the brain-quench anti-cheat read. Must be
        0 at every read window: a held-low read under a STANDING quench drive would be driven inhibition (the
        GABA_B current-subtraction failure mode), not the basin switch this de-risk tests. The spiking analog
        of the host version's 'external current == 0 at read'."""
        nm = getattr(self._bridge, "neuromodulator_manager", None)
        if nm is None or QUENCH_DRIVE_MOD not in nm.modulator_names():
            return 0.0
        return float(nm.get_concentration(QUENCH_DRIVE_MOD))

    def quench_fs_rate(self, counts, n_steps):
        """quench_fs firing rate (spikes/neuron/ms) from a step()'s recorded counts. ~0 at read = the pool is
        SILENT there (it is undriven), which is why the affect pools hold low with no standing inhibition."""
        n = len(self._idx["quench_fs"]) if "quench_fs" in self._idx else 0
        return (counts.get("quench_fs", 0.0) / (n * max(1, n_steps))) if n else 0.0

    def step_brain_quench(self, n_steps):
        """BRAIN-BASED ACTIVE-CLEAR (the spiking replacement for step_quench). Recruit the quench_fs FS pool via
        the phasic quench_drive neuromodulator for `n_steps`; quench_fs fires GABA_A onto the three affect pools
        (through the open quench_out gate), collapsing the reverberatory NMDA loop. NOTHING is injected into the
        affect pools -- the clear is caused ENTIRELY by a spiking inhibitory pool firing onto them. The drive is
        re-set each step (it decays) and EXPLICITLY released to 0 at the end, so the read window sees a silent,
        undriven quench_fs. Returns per-pool spike counts incl. quench_fs (the mechanism-assertion read: quench_fs
        must be firing HARD while ON and ~0 at read).

        Host->brain conversion: step_quench wrote a NEGATIVE cp_external_input_current onto the affect pools (the
        'clear command' issued by host code). Here the only host action is pulsing a diffuse neuromodulator that
        recruits an interneuron pool -- the affect pools receive their silencing as SYNAPTIC GABA_A current from
        spiking neurons, exactly as a real cortical circuit would deliver it (Compte-Wang persistent-activity
        termination via feedback inhibition)."""
        b = self._bridge
        counts = {"affect_vplus": 0.0, "affect_vminus": 0.0, "quench_fs": 0.0}
        qidx = self._idx["quench_fs"]
        for _ in range(int(n_steps)):
            self.set_quench_drive(1.0)             # hold the phasic recruit ON (re-set each step; it decays)
            b.cp_external_input_current[:] = 0.0    # NOTHING host-injected -- affect pools get only synapses
            b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            counts["affect_vplus"] += float(fs[self._idx["affect_vplus"]].sum())
            counts["affect_vminus"] += float(fs[self._idx["affect_vminus"]].sum())
            counts["quench_fs"] += float(fs[qidx].sum())
        self.set_quench_drive(0.0)                  # RELEASE the clear (drive -> 0) before the read window
        return counts


# =============================================================================================================
# The RATCHET trace. One pass = the whole measurement (held + during + the persistence statistic + g_gabab).
# =============================================================================================================
def _affect_ext_current(brain):
    """Mean external-input current on the three affect pools RIGHT NOW -- the quench anti-cheat read. Works on
    both AffectStateBrain (baseline) and EvictionAffectBrain: both expose `_idx` and `_bridge`, and neither
    normally drives the affect pools via external current (appraisal enters through the neuromodulator bus),
    so a nonzero read here means only the quench."""
    idx = np.concatenate([brain._idx["affect_vplus"], brain._idx["affect_vminus"],
                          brain._idx["affect_arousal"]])
    cur = np.asarray(to_host(brain._bridge.cp_external_input_current))
    return float(cur[idx].mean())


def ratchet_trace(brain, levels, drive_ms=DRIVE_MS, post_ms=POST_MS, read_ms=READ_MS,
                  settle_ms=SETTLE_MS, base_probe_ms=BASE_PROBE_MS, arousal=EPISODE_AROUSAL,
                  quench_pA=0.0, quench_ms=DEFAULT_QUENCH_MS, brain_quench=False):
    """Run the episode sequence and return per-episode (held, during) mood plus g_gabab diagnostics.

    Each episode: `drive_ms` of appraisal at `level` -> `post_ms` of SILENCE -> `read_ms` of silence, which
    is the RECORDED window. Baseline-subtracted by a pre-episode quiescent probe. With levels[0]=1.0 the
    first episode is byte-for-byte the baseline runner's persistence protocol, so `held[0]/during[0]` is its
    persistence-retention statistic (committed value 0.62, artifact _affect_state_region_6seed.json).

    ACTIVE-CLEAR quench (`quench_pA` != 0 arms the clear; it fires during the POST-drive silence of every LOW
    episode -- a drive level STRICTLY BELOW the protocol's max, i.e. a LOWER appraisal has arrived -- for
    `quench_ms`, then is OFF for the remainder of the post gap and the ENTIRE read window). Two implementations:

      * HOST SHORTCUT (`brain_quench=False`): step_quench injects a strong NEGATIVE `quench_pA` external current
        directly onto the three affect pools. `quench_pA` is BOTH the trigger and the magnitude.
      * BRAIN-BASED (`brain_quench=True`): step_brain_quench recruits the spiking quench_fs FS pool via the
        quench_drive neuromodulator; quench_fs fires GABA_A onto the affect pools. `quench_pA` is ONLY the
        ON/OFF trigger (its MAGNITUDE is unused -- the real drive is the brain's quench_drive_pA); the affect
        pools receive NO external current at all.

    The clear is drive-dependent by construction, so the constant-HIGH control (all levels == max) NEVER fires
    it, and the episode-0 HIGH and the re-ignite HIGH are untouched. ANTI-CHEAT, measured (not assumed) at every
    read window: (1) external current on the affect pools == 0 (host + brain: a held-low read under a standing
    quench current would be CURRENT SUBTRACTION, the GABA_B failure mode, not a basin switch); and for the brain
    version additionally (2) the quench_drive neuromodulator concentration == 0 -- no standing recruit on
    quench_fs -- and (3) quench_fs firing rate ~0 (the pool is SILENT, being undriven). (2)+(3) prove the affect
    pools hold low with NO standing inhibition, the spiking analog of the host 'current == 0 at read'."""
    brain.reset()
    brain.step(settle_ms)
    base = brain.mood_rate(brain.step(base_probe_ms), base_probe_ms)
    held, during, g_end, g_peak = [], [], [], []
    quench_at_read, quench_fired = [], []
    quench_drive_at_read, quench_fs_rate_at_read, quench_fs_rate_during = [], [], []
    read_rec = (("affect_vplus", "affect_vminus", "quench_fs")
                if (brain_quench and "quench_fs" in brain._idx) else ("affect_vplus", "affect_vminus"))
    hi = max(float(x) for x in levels)
    for lv in levels:
        lv = float(lv)
        ar = float(arousal) if lv > 0.0 else 0.0
        c = brain.step(drive_ms, vp=lv, vm=0.0, ar=ar)
        during.append(brain.mood_rate(c, drive_ms) - base)
        g_peak.append(brain.mean_gabab("affect_vplus"))
        # POST-DRIVE SILENCE. For a LOW episode (a lower appraisal than the protocol's HIGH) the active clear
        # fires INSIDE the post gap and switches OFF before the read; a HIGH episode gets an untouched silence.
        do_quench = bool(quench_pA) and (lv < hi)
        if do_quench:
            qms = min(float(quench_ms), float(post_ms))     # keep the clear strictly inside the post gap
            if brain_quench:
                qc = brain.step_brain_quench(qms)           # spiking quench_fs -> GABA_A onto the affect pools
                quench_fs_rate_during.append(brain.quench_fs_rate(qc, qms))   # the FS pool fires HARD here
            else:
                brain.step_quench(qms, quench_pA)           # host shortcut: negative current on the pools
            rest = float(post_ms) - qms
            if rest > 0:
                brain.step(rest)                            # quench OFF: the OFF fixed point must hold here
        else:
            brain.step(post_ms)
        c_read = brain.step(read_ms, record=read_rec)       # READ WINDOW (quench OFF)
        held.append(brain.mood_rate(c_read, read_ms) - base)
        g_end.append(brain.mean_gabab("affect_vplus"))
        quench_at_read.append(_affect_ext_current(brain))   # anti-cheat: MEASURED (not assumed) at the read
        quench_fired.append(do_quench)
        quench_drive_at_read.append(brain.quench_drive_conc() if brain_quench else 0.0)
        quench_fs_rate_at_read.append(brain.quench_fs_rate(c_read, read_ms) if brain_quench else 0.0)
    # THE LOAD-BEARING ANTI-CHEAT: zero standing quench current at EVERY read window => a held-low read is a
    # genuine basin switch, not current subtraction. This is precisely what separates the active clear from
    # the killed outward brakes (GABA_B held its rate down with a *standing* g*(E_K-V) offset).
    max_q_at_read = max((abs(x) for x in quench_at_read), default=0.0)
    assert max_q_at_read < 1e-6, (
        "ANTI-CHEAT FAILED: |quench current| = %.6g pA at a read window (must be 0); a held-low read here "
        "would be CURRENT SUBTRACTION, not a basin switch" % max_q_at_read)
    max_qdrive = max((abs(x) for x in quench_drive_at_read), default=0.0)
    max_qfs_rate = max((abs(x) for x in quench_fs_rate_at_read), default=0.0)
    if brain_quench:
        # THE BRAIN-QUENCH ANTI-CHEAT: no standing recruit on quench_fs at any read window. If the drive were
        # still ON, a held-low read would be DRIVEN inhibition (the current-subtraction failure mode), not a
        # basin switch. quench_fs rate is reported (soft) and the drive conc is hard-asserted to 0.
        assert max_qdrive < 1e-6, (
            "BRAIN-QUENCH ANTI-CHEAT FAILED: quench_drive conc = %.6g at a read window (must be 0); a held-low "
            "read under a standing quench drive would be DRIVEN inhibition, not a basin switch" % max_qdrive)
    return {"held": [float(x) for x in held], "during": [float(x) for x in during],
            "g_gabab_end_of_drive": [float(x) for x in g_peak],
            "g_gabab_end_of_read": [float(x) for x in g_end],
            "quench_at_read_pA": [float(x) for x in quench_at_read],
            "quench_fired": [bool(x) for x in quench_fired],
            "max_quench_at_read_pA": float(max_q_at_read),
            "quench_drive_at_read": [float(x) for x in quench_drive_at_read],
            "max_quench_drive_at_read": float(max_qdrive),
            "quench_fs_rate_at_read": [float(x) for x in quench_fs_rate_at_read],
            "max_quench_fs_rate_at_read": float(max_qfs_rate),
            "quench_fs_rate_during_quench": [float(x) for x in quench_fs_rate_during],
            "min_quench_fs_rate_during_quench": float(min(quench_fs_rate_during, default=0.0)),
            "brain_quench": bool(brain_quench),
            "quench_pA": float(quench_pA), "quench_ms": float(quench_ms),
            "mood_base": float(base), "levels": [float(x) for x in levels]}


# A "held mood" whose magnitude is a rounding error is not a held mood, and dividing by it manufactures
# nonsense: the first smoke printed an eviction ratio of +21.750 from held[0]=0.0004. Any ratio whose
# DENOMINATOR is below this floor is UNDEFINED, not a number. Floor = 25% of the committed baseline held
# mood (~0.09 spikes/neuron/ms), i.e. ~0.023.
HELD_FLOOR_FRAC = 0.25
HELD_FLOOR_ABS = 0.02


def _ratio(num, den, floor=1e-6):
    """held-ratio, or None when the denominator is below `floor` (UNDEFINED, never a score)."""
    if abs(den) < floor:
        return None
    return float(num) / float(den)


def evaluate_arm(low_trace, high_trace, nmda_off_trace, lesion_trace, during0_reference=None,
                 held0_reference=None):
    """Apply the pre-registered gate to one (gabab_w, tau, cap) operating point. Returns the six gate
    booleans plus the raw ratios; ratios that cannot be formed are None (UNDEFINED), never 0."""
    lh, ld = low_trace["held"], low_trace["during"]
    hh = high_trace["held"] if high_trace else None
    floor = (HELD_FLOOR_FRAC * abs(held0_reference)) if held0_reference else HELD_FLOOR_ABS

    evict_ratio = _ratio(max(lh[1], lh[2]), lh[0], floor)
    time_ratio = None if hh is None else _ratio(min(hh[1], hh[2]), hh[0], floor)
    reignite_ratio = _ratio(lh[4], lh[0], floor) if len(lh) >= 5 else None
    persist_ratio = _ratio(lh[0], ld[0])          # G4's denominator is the DRIVEN peak, not a held state
    nmda_off_ratio = (_ratio(nmda_off_trace["held"][0], nmda_off_trace["during"][0])
                      if nmda_off_trace else None)
    lesion_ratio = None
    if lesion_trace is not None:
        lesion_ratio = _ratio(max(lesion_trace["held"][1], lesion_trace["held"][2]),
                              lesion_trace["held"][0], floor)

    # A5 validity: an eviction that CRUSHES the pool makes every ratio above meaningless.
    not_crushed = None
    if during0_reference is not None and abs(during0_reference) > 1e-6:
        not_crushed = bool(ld[0] >= 0.5 * during0_reference)

    gates = {
        "G1_eviction(low<0.60)": (evict_ratio is not None and evict_ratio < 0.60),
        "G2_drive_dependence(high>=0.60)": (time_ratio is not None and time_ratio >= 0.60),
        "G3_reignition(>=0.60)": (reignite_ratio is not None and reignite_ratio >= 0.60),
        "G4_persistence_survives(>=0.50)": (persist_ratio is not None and persist_ratio >= 0.50),
        "G5_nmda_off(<0.10)": (nmda_off_ratio is not None and nmda_off_ratio < 0.10),
        "G6_lesion_restores_ratchet(>=0.90)": (lesion_ratio is not None and lesion_ratio >= 0.90),
    }
    valid = (not_crushed is not False)
    return {
        "gates": gates,
        "core_go(G1-G4)": all(gates[k] for k in list(gates)[:4]) and valid,
        "instrument_ok(G5,G6)": gates["G5_nmda_off(<0.10)"] and gates["G6_lesion_restores_ratchet(>=0.90)"],
        "A5_not_crushed": not_crushed, "arm_valid": bool(valid),
        "held0_floor_used": float(floor),
        "evict_ratio_low": evict_ratio, "time_ratio_high": time_ratio, "reignite_ratio": reignite_ratio,
        "persistence_retention": persist_ratio, "nmda_off_retention": nmda_off_ratio,
        "lesion_evict_ratio": lesion_ratio,
        "held_low": lh, "during_low": ld,
        "held_high": hh, "during_high": (high_trace["during"] if high_trace else None),
        "g_gabab_end_of_read_low": low_trace["g_gabab_end_of_read"],
    }


def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:+.{nd}f}"


# =============================================================================================================
# One operating point, all arms
# =============================================================================================================
def run_point(seed, gabab_w, gabab_tau, ou_pA=8.0, recur_weight=DEFAULT_RECUR_WEIGHT, gabab_max=0.0,
              enable_gabab=True, sfa=None, verbose=True, baseline_during0=None, baseline_held0=None,
              stp=False, stp_tau_d=None, stp_U=None, quench_pA=0.0, quench_ms=DEFAULT_QUENCH_MS,
              brain_quench=False, quench_fs_n=QUENCH_FS_N, quench_gaba_w=QUENCH_GABA_W,
              quench_drive_pA=QUENCH_DRIVE_PA):
    """Run the full pre-registered arm set at one (gabab_w, tau) point and return the evaluated gate."""
    def mk(nmda_on=True, evict=True, wiring=True, egb=enable_gabab):
        return EvictionAffectBrain(seed, nmda_on=nmda_on, recur_weight=recur_weight, ou_pA=ou_pA,
                                   gabab_w=gabab_w, gabab_tau_ms=gabab_tau, gabab_max=gabab_max,
                                   enable_gabab=egb, evict=evict,
                                   sfa_a=(None if sfa is None else sfa[0]),
                                   sfa_d=(None if sfa is None else sfa[1]),
                                   with_eviction_wiring=wiring,
                                   stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                                   brain_quench=brain_quench, quench_fs_n=quench_fs_n,
                                   quench_gaba_w=quench_gaba_w, quench_drive_pA=quench_drive_pA)

    low = ratchet_trace(mk(), LOW_LEVELS, quench_pA=quench_pA, quench_ms=quench_ms, brain_quench=brain_quench)
    high = ratchet_trace(mk(), HIGH_LEVELS, quench_pA=quench_pA, quench_ms=quench_ms, brain_quench=brain_quench)
    nmda_off = ratchet_trace(mk(nmda_on=False), LOW_LEVELS, quench_pA=quench_pA, quench_ms=quench_ms,
                             brain_quench=brain_quench)
    les_brain = mk(evict=False)
    # G6 same-substrate control. When the QUENCH is the active evictor (quench_pA != 0), the correct
    # "mechanism removed" arm is the CLEAR turned OFF (quench_pA=0) -- NOT the evict_out gate, which lesions the
    # GABA_B limb (inert at gabab_w=0). So the lesion arm always runs quench-OFF: removing the clear must let
    # the ratchet RETURN (lesion_evict_ratio -> ~1.0, i.e. the >=0.90 G6 asks for), same neurons/wiring/seed.
    # (When quench_pA==0 this reduces to the original GABA_B evict_out lesion, unchanged.) For brain_quench the
    # same-substrate quench_fs wiring is present but simply undriven, so the ratchet returns identically.
    lesion = ratchet_trace(les_brain, LOW_LEVELS, quench_pA=0.0, quench_ms=quench_ms, brain_quench=brain_quench)

    ev = evaluate_arm(low, high, nmda_off, lesion, during0_reference=baseline_during0,
                      held0_reference=baseline_held0)
    ev.update({"seed": int(seed), "gabab_weight": float(gabab_w), "gabab_tau_ms": float(gabab_tau),
               "gabab_max": float(gabab_max), "enable_gabab": bool(enable_gabab),
               "quench_pA": float(quench_pA), "quench_ms": float(quench_ms),
               "brain_quench": bool(brain_quench),
               "quench_fired_low": low["quench_fired"], "quench_at_read_low_pA": low["quench_at_read_pA"],
               "max_quench_at_read_low_pA": low["max_quench_at_read_pA"],
               "max_quench_drive_at_read_low": low["max_quench_drive_at_read"],
               "max_quench_fs_rate_at_read_low": low["max_quench_fs_rate_at_read"],
               "quench_fs_rate_at_read_low": low["quench_fs_rate_at_read"],
               "sfa": (None if sfa is None else [float(sfa[0]), float(sfa[1])]),
               "g_gabab_lesion_end_of_read": lesion["g_gabab_end_of_read"],
               "trace_low": low, "trace_high": high, "trace_nmda_off": nmda_off, "trace_lesion": lesion})
    if verbose:
        print(f"    held(LOW)  {[round(x, 4) for x in low['held']]}   during {[round(x, 3) for x in low['during']]}")
        print(f"    held(HIGH) {[round(x, 4) for x in high['held']]}")
        print(f"    g_gabab(pool, end of each read) intact {[round(x, 2) for x in low['g_gabab_end_of_read']]} "
              f"| lesion {[round(x, 2) for x in lesion['g_gabab_end_of_read']]}")
    return ev


# =============================================================================================================
# SMOKE — reproduce the ratchet on the untouched baseline, then sweep the eviction operating point
# =============================================================================================================
def run_smoke(seed, weights, taus, maxes=(0.0,), ou_pA=8.0, recur_weight=DEFAULT_RECUR_WEIGHT, sfa=None,
              stp=False, stp_tau_d=None, stp_U=None, quench_pA=0.0, quench_ms=DEFAULT_QUENCH_MS,
              brain_quench=False, quench_fs_n=QUENCH_FS_N, quench_gaba_w=QUENCH_GABA_W,
              quench_drive_pA=QUENCH_DRIVE_PA):
    t0 = time.time()
    out = {"seed": int(seed), "sweep": [], "sfa_arm": None, "fast_arm": None,
           "quench_pA": float(quench_pA), "quench_ms": float(quench_ms), "brain_quench": bool(brain_quench),
           "quench_fs_n": int(quench_fs_n), "quench_gaba_w": float(quench_gaba_w),
           "quench_drive_pA": float(quench_drive_pA)}
    gabab_max = 0.0   # replaced by the SELECTED sweep point's cap in step 2 (see below)

    # ---- 0. INSTRUMENT VERIFICATION: the untouched BASELINE brain must reproduce the RATCHET -------------
    # (and its held[0] / persistence retention must match the committed 6-seed artifact, or the protocol is
    #  not measuring the thing the artifact measured and everything downstream is void.)
    print("[EVICT SMOKE] 0. instrument check — the UNTOUCHED baseline brain (no eviction wiring at all)",
          flush=True)
    base_brain = AffectStateBrain(seed, nmda_on=True, recur_weight=recur_weight, ou_pA=ou_pA)
    base_brain.mean_gabab = lambda region="affect_vplus": 0.0   # baseline has no GABA_B at all
    base_low = ratchet_trace(base_brain, LOW_LEVELS)
    base_high = ratchet_trace(base_brain, HIGH_LEVELS)
    base_evict_ratio = _ratio(max(base_low["held"][1], base_low["held"][2]), base_low["held"][0],
                              HELD_FLOOR_ABS)
    base_persist = _ratio(base_low["held"][0], base_low["during"][0])
    ref = None
    if BASELINE_ARTIFACT.exists():
        try:
            _a = json.loads(BASELINE_ARTIFACT.read_text())
            ref = {"persistence_retention_nmda_on": _a["means"]["persistence_retention_nmda_on"],
                   "per_seed_mood_ret_on": {int(r["seed"]): r["mood_ret_on"] for r in _a["per_seed"]}}
        except Exception as e:                                     # narrow enough to see, never silent
            print(f"    (could not read the baseline artifact: {type(e).__name__}: {e})", flush=True)
    print(f"    baseline held(LOW)  {[round(x, 4) for x in base_low['held']]}", flush=True)
    print(f"    baseline held(HIGH) {[round(x, 4) for x in base_high['held']]}", flush=True)
    print(f"    baseline RATCHET ratio max(held[1],held[2])/held[0] = {_fmt(base_evict_ratio)}   "
          f"(the defect: ~1.00; the gate wants < 0.60)", flush=True)
    if ref is not None:
        art_ret = ref["persistence_retention_nmda_on"]
        art_held = ref["per_seed_mood_ret_on"].get(int(seed))
        print(f"    baseline persistence held[0]/during[0] = {_fmt(base_persist)}  vs committed artifact mean "
              f"{art_ret:+.3f}; held[0]={base_low['held'][0]:+.4f} vs artifact seed-{seed} mood_ret_on "
              f"{'n/a' if art_held is None else format(art_held, '+.4f')}", flush=True)
    instrument_ok = bool(base_evict_ratio is not None and base_evict_ratio >= 0.90)
    print(f"    => instrument {'VERIFIED (the ratchet reproduces)' if instrument_ok else 'NOT VERIFIED'}",
          flush=True)
    out["baseline"] = {"trace_low": base_low, "trace_high": base_high,
                       "ratchet_ratio": base_evict_ratio, "persistence_retention": base_persist,
                       "instrument_reproduces_ratchet": instrument_ok,
                       "committed_artifact_reference": ref}
    if void_if(not instrument_ok,
               "the untouched baseline did NOT reproduce the ratchet (max(held[1],held[2])/held[0] < 0.90), "
               "so this protocol is not measuring the defect the 6-seed artifact measured"):
        out["VOID"] = True
        return out
    baseline_during0 = base_low["during"][0]
    baseline_held0 = base_low["held"][0]

    # ---- 1b. THE QUENCH PHYSICS (dedicated read; only when the active clear is armed) -------------------
    # The whole point of this de-risk, answered directly: does a TRANSIENT open-loop clear collapse the
    # bistable slow-NMDA loop (G1 evict), while (G3) the loop RE-IGNITES on the next HIGH drive -- proving the
    # attractor was NOT destroyed, the STP failure mode -- and (G4) episode-0 persistence (read with NO clear)
    # survives? Same substrate, quench-ON vs quench-OFF, plus the zero-current-at-read anti-cheat.
    if quench_pA:
        w0, tau0, cap0 = float(weights[0]), float(taus[0]), float(maxes[0])
        def _qbrain():
            return EvictionAffectBrain(seed, nmda_on=True, recur_weight=recur_weight, ou_pA=ou_pA,
                                       gabab_w=w0, gabab_tau_ms=tau0, gabab_max=cap0,
                                       enable_gabab=(w0 > 0.0), stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                                       brain_quench=brain_quench, quench_fs_n=quench_fs_n,
                                       quench_gaba_w=quench_gaba_w, quench_drive_pA=quench_drive_pA)
        on = ratchet_trace(_qbrain(), LOW_LEVELS, quench_pA=quench_pA, quench_ms=quench_ms,
                           brain_quench=brain_quench)
        off = ratchet_trace(_qbrain(), LOW_LEVELS, quench_pA=0.0, quench_ms=quench_ms,
                            brain_quench=brain_quench)
        ho, hf, du = on["held"], off["held"], on["during"]
        g1 = _ratio(max(ho[1], ho[2]), ho[0], HELD_FLOOR_ABS)
        g3 = _ratio(ho[4], ho[0], HELD_FLOOR_ABS) if len(ho) >= 5 else None
        g4 = _ratio(ho[0], du[0])
        g1_ok = g1 is not None and g1 < 0.60
        g3_ok = g3 is not None and g3 >= 0.60
        g4_ok = g4 is not None and g4 >= 0.50
        anti = on["max_quench_at_read_pA"] < 1e-6
        # BRAIN-QUENCH anti-cheat: additionally no standing recruit (drive conc == 0) and quench_fs ~silent at
        # read. QFS_RATE_FLOOR is a soft bound -- OU noise makes a tiny undriven FS rate acceptable; what must
        # be zero is the DRIVE (hard-asserted in ratchet_trace) and the affect-pool external current.
        QFS_RATE_FLOOR = 0.02   # spikes/neuron/ms (~20 Hz); undriven FS on OU noise sits well below this
        anti_drive = on["max_quench_drive_at_read"] < 1e-6
        anti_fs = on["max_quench_fs_rate_at_read"] < QFS_RATE_FLOOR
        if brain_quench:
            anti = anti and anti_drive and anti_fs
        evict_with_reignite = bool(g1_ok and g3_ok and g4_ok and anti)
        _mode = ("BRAIN-BASED (spiking quench_fs -> GABA_A; drive=quench_drive nm)" if brain_quench
                 else "HOST SHORTCUT (negative current on affect pools)")
        print(f"\n[EVICT SMOKE] 1b. ACTIVE-CLEAR quench physics [{_mode}] @ quench_ms={quench_ms} "
              f"(w={w0} tau={tau0}"
              + (f" | quench_fs_n={quench_fs_n} gaba_w={quench_gaba_w} drive_pA={quench_drive_pA}"
                 if brain_quench else f" | quench_pA={quench_pA}") + ")", flush=True)
        print(f"    held(LOW) quench-ON   {[round(x, 4) for x in ho]}   during {[round(x, 3) for x in du]}",
              flush=True)
        print(f"    held(LOW) quench-OFF  {[round(x, 4) for x in hf]}   (the ratchet control -- clear removed)",
              flush=True)
        print(f"    quench_fired/episode {on['quench_fired']} | affect-pool ext-current@read(pA) "
              f"{[round(x, 4) for x in on['quench_at_read_pA']]} | max|ext@read| "
              f"{on['max_quench_at_read_pA']:.2e}", flush=True)
        if brain_quench:
            print(f"    [brain] quench_fs rate DURING clear {[round(x, 3) for x in on['quench_fs_rate_during_quench']]} "
                  f"spk/nrn/ms (min {on['min_quench_fs_rate_during_quench']:.3f} -- the FS pool IS the evictor) "
                  f"vs @read {[round(x, 4) for x in on['quench_fs_rate_at_read']]} "
                  f"(max {on['max_quench_fs_rate_at_read']:.4f} < {QFS_RATE_FLOOR} -> {anti_fs})", flush=True)
            print(f"    [brain] quench_drive@read {[round(x, 6) for x in on['quench_drive_at_read']]} "
                  f"(max {on['max_quench_drive_at_read']:.2e}, ==0 -> {anti_drive})", flush=True)
        print(f"    G1 evict {_fmt(g1)} (<0.60 -> {g1_ok}) | G3 re-ignite {_fmt(g3)} (>=0.60 -> {g3_ok}) | "
              f"G4 persist {_fmt(g4)} (>=0.50 -> {g4_ok}) | ANTI-CHEAT quencher-silent@read -> {anti}",
              flush=True)
        print(f"    => EVICTION-WITH-RE-IGNITION (physics) {'YES' if evict_with_reignite else 'NO'} "
              f"(ONE seed; a smoke, not a verdict)", flush=True)
        out["quench_physics"] = {
            "mode": ("brain_based" if brain_quench else "host_shortcut"),
            "quench_pA": float(quench_pA), "quench_ms": float(quench_ms), "w": w0, "tau": tau0, "cap": cap0,
            "brain_quench": bool(brain_quench), "quench_fs_n": int(quench_fs_n),
            "quench_gaba_w": float(quench_gaba_w), "quench_drive_pA": float(quench_drive_pA),
            "held_low_quench_on": ho, "held_low_quench_off": hf, "during_low_quench_on": du,
            "quench_fired": on["quench_fired"], "quench_at_read_pA": on["quench_at_read_pA"],
            "max_quench_at_read_pA": on["max_quench_at_read_pA"],
            "quench_drive_at_read": on["quench_drive_at_read"],
            "max_quench_drive_at_read": on["max_quench_drive_at_read"],
            "quench_fs_rate_at_read": on["quench_fs_rate_at_read"],
            "max_quench_fs_rate_at_read": on["max_quench_fs_rate_at_read"],
            "quench_fs_rate_during_quench": on["quench_fs_rate_during_quench"],
            "min_quench_fs_rate_during_quench": on["min_quench_fs_rate_during_quench"],
            "anti_cheat_drive_zero_at_read": bool(anti_drive), "anti_cheat_fs_silent_at_read": bool(anti_fs),
            "G1_evict_ratio": g1, "G1_ok": g1_ok, "G3_reignite_ratio": g3, "G3_ok": g3_ok,
            "G4_persist_ratio": g4, "G4_ok": g4_ok, "anti_cheat_quencher_silent_at_read": bool(anti),
            "evict_with_reignition": evict_with_reignite}

    # ---- 1. SWEEP the eviction operating point ---------------------------------------------------------
    n_pts = len(weights) * len(taus) * len(maxes)
    print(f"\n[EVICT SMOKE] 1. sweeping GABA_B (weight x tau x GIRK-cap) — {n_pts} points", flush=True)
    print(f"    {'w':>6} {'tau':>6} {'cap':>6} | {'G1 evict':>9} {'G2 time':>8} {'G3 re-ig':>9} "
          f"{'G4 persist':>11} {'G5 nmdaoff':>11} {'G6 lesion':>10} | verdict", flush=True)
    best = None
    for w in weights:
        for tau in taus:
            for cap in maxes:
                ev = run_point(seed, w, tau, ou_pA=ou_pA, recur_weight=recur_weight, gabab_max=cap,
                               verbose=False, baseline_during0=baseline_during0,
                               baseline_held0=baseline_held0, stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                               quench_pA=quench_pA, quench_ms=quench_ms, brain_quench=brain_quench,
                               quench_fs_n=quench_fs_n, quench_gaba_w=quench_gaba_w,
                               quench_drive_pA=quench_drive_pA)
                g = ev["gates"]
                passed = sum(1 for k in list(g)[:4] if g[k])
                if ev["evict_ratio_low"] is None:
                    # held[0] ~ 0: the pool did not hold ANY mood, so no ratio exists. UNDEFINED, not a
                    # "the ratchet holds" read and not a pass -- reporting either would fabricate a verdict
                    # out of a dead arm.
                    verdict = "UNDEFINED (held[0]~0: no state to evict)"
                elif ev["A5_not_crushed"] is False:
                    verdict = "UNDEFINED (pool crushed: during[0] < 0.5x baseline)"
                elif ev["core_go(G1-G4)"]:
                    verdict = "PASS G1-G4"
                elif not g["G1_eviction(low<0.60)"]:
                    verdict = "ratchet holds"
                elif not g["G4_persistence_survives(>=0.50)"]:
                    verdict = "persistence lost"
                elif not g["G3_reignition(>=0.60)"]:
                    verdict = "no re-ignition"
                else:
                    verdict = "time-decay only (G2 fails)"
                print(f"    {w:>6.2f} {tau:>6.0f} {cap:>6.1f} | {_fmt(ev['evict_ratio_low']):>9} "
                      f"{_fmt(ev['time_ratio_high']):>8} {_fmt(ev['reignite_ratio']):>9} "
                      f"{_fmt(ev['persistence_retention']):>11} {_fmt(ev['nmda_off_retention']):>11} "
                      f"{_fmt(ev['lesion_evict_ratio']):>10} | {verdict} ({passed}/4)", flush=True)
                row = {k: v for k, v in ev.items() if not k.startswith("trace_")}
                row["trace_low_held"] = ev["trace_low"]["held"]
                row["sweep_verdict"] = verdict
                out["sweep"].append(row)
                if ev["core_go(G1-G4)"] and (best is None):
                    best = ev

    if best is None:
        # rank by how far the eviction ratio fell while persistence survived — the honest "closest point"
        cand = [r for r in out["sweep"] if r["gates"]["G4_persistence_survives(>=0.50)"]
                and r["evict_ratio_low"] is not None and r["arm_valid"]]
        near = min(cand, key=lambda r: r["evict_ratio_low"]) if cand else None
        out["kill_criterion_met"] = True
        out["closest_point"] = near
        if near is None:
            near_txt = "none (no point even kept persistence)"
        else:
            near_txt = (f"w={near['gabab_weight']} tau={near['gabab_tau_ms']} "
                        f"evict_ratio={near['evict_ratio_low']:+.3f} (needs < 0.60)")
        print(f"\n[EVICT SMOKE] KILL CRITERION MET on seed {seed}: no swept (w, tau) point satisfied "
              f"G1-G4 together. Closest point that KEPT persistence: {near_txt}. "
              f"Per THE LAW this kills the METHOD (slow GABA_B feedback), not the CAPABILITY; the "
              f"pre-registered next method is intrinsic spike-frequency adaptation (--sfa). "
              f"ONE SEED — a smoke, not a 6-seed verdict.", flush=True)
    else:
        out["kill_criterion_met"] = False
        out["chosen_point"] = {"gabab_weight": best["gabab_weight"], "gabab_tau_ms": best["gabab_tau_ms"],
                               "gabab_max": best["gabab_max"]}
        print(f"\n[EVICT SMOKE] chosen operating point: gabab_w={best['gabab_weight']} "
              f"tau={best['gabab_tau_ms']} ms cap={best['gabab_max']} — G1-G4 pass on ONE seed "
              f"(NOT a result; 6 seeds decide)", flush=True)

    # ---- 2. MECHANISM ASSERTION: the lever must actually MOVE, and the SLOW component must be load-bearing.
    # SKIPPED for brain_quench: the GABA_B limb is inert (gabab_w=0), and the mechanism under test is the spiking
    # quench, already fully characterized in 1b (G1/G3/G4 + the drive/quench_fs-silent-at-read anti-cheat). A
    # g_gabab lever on a zero-weight limb would be a meaningless null pair.
    pt = (best if best is not None
          else (out.get("closest_point") or (out["sweep"][0] if out["sweep"] else None)))
    if pt is not None and brain_quench:
        print("\n[EVICT SMOKE] 2. mechanism assertion SKIPPED (brain_quench) — the evictor is the spiking "
              "quench_fs pool, characterized in 1b; the GABA_B limb is inert here.", flush=True)
    if pt is not None and not brain_quench:
        w_sel, tau_sel = float(pt["gabab_weight"]), float(pt["gabab_tau_ms"])
        cap_sel = float(pt.get("gabab_max", 0.0))
        gabab_max = cap_sel   # the mechanism-assertion + attribution arms follow the SELECTED point
        print(f"\n[EVICT SMOKE] 2. mechanism assertion @ w={w_sel} tau={tau_sel} cap={cap_sel}", flush=True)
        intact = run_point(seed, w_sel, tau_sel, ou_pA=ou_pA, recur_weight=recur_weight, gabab_max=gabab_max,
                           verbose=True, baseline_during0=baseline_during0,
                           baseline_held0=baseline_held0, stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                           quench_pA=quench_pA, quench_ms=quench_ms)
        g_intact = max(intact["trace_low"]["g_gabab_end_of_read"])
        g_lesion = max(intact["g_gabab_lesion_end_of_read"])
        try:
            lever("g_gabab on affect_vplus", round(g_lesion, 4), round(g_intact, 4),
                  continuous=f"evict_ratio intact={_fmt(intact['evict_ratio_low'])} "
                             f"lesion={_fmt(intact['lesion_evict_ratio'])}")
            out["lever_moved"] = True
        except Exception as e:
            print(f"  ⛔ {type(e).__name__}: {e}", flush=True)
            out["lever_moved"] = False
        # the SLOWNESS-attribution arm: identical wiring, enable_gabab=False -> the same synapses deliver
        # ONLY their fast GABA_A component (GABA_B is ADDITIVE in this engine).
        fast = run_point(seed, w_sel, tau_sel, ou_pA=ou_pA, recur_weight=recur_weight, gabab_max=gabab_max,
                         enable_gabab=False, verbose=False, baseline_during0=baseline_during0,
                               baseline_held0=baseline_held0, stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                               quench_pA=quench_pA, quench_ms=quench_ms)
        # THREE-WAY, not two: if the SLOW arm did not evict either, there is NOTHING to attribute, and
        # printing "the SLOW component is what evicts" would be a claim manufactured from two null arms
        # (the first smoke printed exactly that from evict_ratio 0.997 vs 0.992).
        fr, ir = fast["evict_ratio_low"], intact["evict_ratio_low"]
        if ir is None or ir >= 0.60:
            attrib = ("ATTRIBUTION UNDEFINED — the slow-GABA_B arm did not evict either, so there is no "
                      "effect to attribute to slowness")
        elif fr is not None and fr < 0.60:
            attrib = "slowness NOT load-bearing — the FAST GABA_A-only arm evicts too"
        else:
            attrib = "the SLOW component is what evicts (fast-only does not)"
        print(f"    SLOWNESS ATTRIBUTION — same wiring, GABA_A only (enable_gabab=False): "
              f"evict_ratio {_fmt(fr)} vs slow-GABA_B {_fmt(ir)}  => {attrib}", flush=True)
        out["slowness_attribution"] = attrib
        out["fast_arm"] = {k: v for k, v in fast.items() if not k.startswith("trace_")}
        out["selected_point_full"] = {k: v for k, v in intact.items() if not k.startswith("trace_")}

    # ---- 3. optional: the pre-registered FALLBACK method (intrinsic SFA) under the SAME gate -------------
    if sfa is not None:
        print(f"\n[EVICT SMOKE] 3. FALLBACK method — intrinsic spike-frequency adaptation "
              f"(izh a={sfa[0]}, d_increment={sfa[1]}) on the affect pools, SAME gate", flush=True)
        # Verify the manipulation MOVED in the intended direction before reading its result. The affect
        # pools are RS (a=0.03, d_increment=100): "more adaptation" means a LARGER d and a SMALLER a.
        _probe = EvictionAffectBrain(seed, recur_weight=recur_weight, ou_pA=ou_pA, enable_gabab=False,
                                     sfa_a=sfa[0], sfa_d=sfa[1], with_eviction_wiring=False,
                                     stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U)
        _b, _a = getattr(_probe, "sfa_before", None), getattr(_probe, "sfa_after", None)
        out["sfa_lever"] = {"before_(a,d)": _b, "after_(a,d)": _a}
        if _b is not None:
            try:
                lever("izh (a, d_increment) on affect pools", tuple(round(x, 4) for x in _b),
                      tuple(round(x, 4) for x in _a),
                      continuous=("d %+.1f (LARGER d = MORE per-spike adaptation), a %+.4f "
                                  "(SMALLER a = SLOWER recovery)" % (_a[1] - _b[1], _a[0] - _b[0])))
            except Exception as e:
                print(f"  ⛔ {type(e).__name__}: {e}", flush=True)
            if _a[1] < _b[1]:
                print(f"  ⚠️  d_increment was LOWERED ({_b[1]} -> {_a[1]}): this arm has LESS spike-frequency "
                      f"adaptation than the default, not more. It is NOT a test of cranked SFA.", flush=True)
                out["sfa_arm_mis_specified"] = True
        s = run_point(seed, 0.0, DEFAULT_GABAB_TAU_MS, ou_pA=ou_pA, recur_weight=recur_weight,
                      gabab_max=0.0, enable_gabab=False, sfa=sfa, verbose=True,
                      baseline_during0=baseline_during0, baseline_held0=baseline_held0,
                      stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U, quench_pA=quench_pA, quench_ms=quench_ms,
                      brain_quench=brain_quench, quench_fs_n=quench_fs_n, quench_gaba_w=quench_gaba_w,
                      quench_drive_pA=quench_drive_pA)
        print(f"    SFA: G1 evict {_fmt(s['evict_ratio_low'])} | G2 time {_fmt(s['time_ratio_high'])} | "
              f"G3 re-ig {_fmt(s['reignite_ratio'])} | G4 persist {_fmt(s['persistence_retention'])} "
              f"=> {'PASS G1-G4' if s['core_go(G1-G4)'] else 'no'}", flush=True)
        out["sfa_arm"] = {k: v for k, v in s.items() if not k.startswith("trace_")}

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    print("\n[EVICT SMOKE] ONE SEED ONLY — this is a smoke, not a result. A verdict needs the 6-seed gate.",
          flush=True)
    return out


# =============================================================================================================
# 6-SEED gate
# =============================================================================================================
def run_battery(seeds, gabab_w, gabab_tau, ou_pA=8.0, recur_weight=DEFAULT_RECUR_WEIGHT, gabab_max=0.0,
                sfa=None, stp=False, stp_tau_d=None, stp_U=None, quench_pA=0.0, quench_ms=DEFAULT_QUENCH_MS,
                brain_quench=False, quench_fs_n=QUENCH_FS_N, quench_gaba_w=QUENCH_GABA_W,
                quench_drive_pA=QUENCH_DRIVE_PA):
    rows = []
    for s in seeds:
        t0 = time.time()
        base_brain = AffectStateBrain(s, nmda_on=True, recur_weight=recur_weight, ou_pA=ou_pA)
        base_brain.mean_gabab = lambda region="affect_vplus": 0.0
        base_low = ratchet_trace(base_brain, LOW_LEVELS)   # baseline instrument arm: NO quench (the control)
        base_ratio = _ratio(max(base_low["held"][1], base_low["held"][2]), base_low["held"][0],
                            HELD_FLOOR_ABS)
        ev = run_point(s, gabab_w, gabab_tau, ou_pA=ou_pA, recur_weight=recur_weight, gabab_max=gabab_max,
                       sfa=sfa, verbose=False, baseline_during0=base_low["during"][0],
                       baseline_held0=base_low["held"][0], stp=stp, stp_tau_d=stp_tau_d, stp_U=stp_U,
                       quench_pA=quench_pA, quench_ms=quench_ms, brain_quench=brain_quench,
                       quench_fs_n=quench_fs_n, quench_gaba_w=quench_gaba_w, quench_drive_pA=quench_drive_pA)
        ev["baseline_ratchet_ratio"] = base_ratio
        ev["baseline_held_low"] = base_low["held"]
        ev["elapsed_seconds"] = round(time.time() - t0, 1)
        print(f"  [seed {s}] baseline-ratchet {_fmt(base_ratio)} | G1 {_fmt(ev['evict_ratio_low'])} "
              f"G2 {_fmt(ev['time_ratio_high'])} G3 {_fmt(ev['reignite_ratio'])} "
              f"G4 {_fmt(ev['persistence_retention'])} G5 {_fmt(ev['nmda_off_retention'])} "
              f"G6 {_fmt(ev['lesion_evict_ratio'])} | core {ev['core_go(G1-G4)']} "
              f"instr {ev['instrument_ok(G5,G6)']} ({ev['elapsed_seconds']}s)", flush=True)
        rows.append(ev)
    return rows


def summarize(rows, seeds, cfgd):
    n = len(rows)
    n_core = sum(1 for r in rows if r["core_go(G1-G4)"])
    n_instr = sum(1 for r in rows if r["instrument_ok(G5,G6)"])
    n_baseline_ratchet = sum(1 for r in rows
                             if r.get("baseline_ratchet_ratio") is not None
                             and r["baseline_ratchet_ratio"] >= 0.90)
    per_gate = {k: sum(1 for r in rows if r["gates"][k]) for k in rows[0]["gates"]} if rows else {}

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    means = {k: mean(k) for k in ["evict_ratio_low", "time_ratio_high", "reignite_ratio",
                                  "persistence_retention", "nmda_off_retention", "lesion_evict_ratio",
                                  "baseline_ratchet_ratio"]}
    instrument_verified = n_baseline_ratchet >= max(1, int(np.ceil(5 * n / 6)))
    go = bool(instrument_verified and n_core >= max(1, int(np.ceil(5 * n / 6)))
              and n_instr >= max(1, int(np.ceil(5 * n / 6))))

    # ---- EARN the verdict (2026-07-31). THE DEFECT THIS CLOSES, in this exact file: A5 is pre-registered
    # ("during(evict ON) >= 0.5 x during(baseline); else UNDEFINED, not a pass"), was computed correctly as
    # `arm_valid=False` on 3/3 seeds, and the verdict string still read "NO-GO / BOUNDARY". Every downstream
    # ratio in that artifact is null because no signal survived. The runner HAD the number and did not
    # consult it. Registering it here makes that impossible rather than remembered.
    n_valid = sum(1 for r in rows if r.get("arm_valid"))
    from tools.verdict import Verdict                                          # noqa: E402
    _v = Verdict("affect mood eviction")
    _v.require("A5: the treatment arm was not CRUSHED (>=5/6 seeds)",
               n_valid >= max(1, int(np.ceil(5 * n / 6))), expect=True,
               note=f"{n_valid}/{n} seeds had a valid arm; a crushed arm yields UNDEFINED, never a negative")
    _v.require("the instrument is verified (baseline reproduces the ratchet)", instrument_verified, expect=True)
    _v.require("every seed produced an eviction ratio", 
               all(r.get("evict_ratio_low") is not None for r in rows if r.get("arm_valid")), expect=True)
    for _proc in ("short-term plasticity", "STDP", "Hebbian learning", "homeostasis",
                  "reward modulation", "structural plasticity"):
        _v.disabled(_proc, why="isolation block in this runner — the attractor is the only live dynamic, so "
                               "any latch observed is a property of the mechanism UNDER THIS ISOLATION")
    _verdict_block = _v.decide(go=go)
    if _verdict_block["status"] == "UNDEFINED":
        go = False

    def _m(k):
        return "n/a" if means[k] is None else f"{means[k]:.3f}"

    # MECHANISM-AWARE label (2026-08-01): name the ACTIVE evictor, DERIVED from the config that actually ran --
    # never a hardcoded "GABA_B". A --brain-quench or --sfa battery previously printed "slow GABA_B/GIRK feedback
    # EVICTS" while GABA_B was OFF (weight 0), a mislabel the value-gates cannot catch (the gate NUMBERS were
    # correct while the prose lied). The label must follow the flags, not a template.
    if cfgd.get("brain_quench"):
        mech = "a spiking quench_fs GABA_A active-clear gate"
    elif cfgd.get("quench_pA"):
        mech = "a transient active-clear current pulse (host shortcut)"
    elif cfgd.get("sfa"):
        mech = "intrinsic spike-frequency adaptation (sAHP)"
    elif cfgd.get("stp"):
        mech = "short-term synaptic depression"
    elif cfgd.get("gabab_weight"):
        mech = "slow GABA_B/GIRK feedback"
    else:
        mech = "the active evictor"

    if not instrument_verified:
        verdict = (f"UNDEFINED ({n}-seed) — the INSTRUMENT is not verified: the untouched baseline "
                   f"reproduced the ratchet (>=0.90) on only {n_baseline_ratchet}/{n} seeds "
                   f"(mean {_m('baseline_ratchet_ratio')}). Nothing downstream can be read as a result.")
    elif go:
        verdict = (f"GO ({n}-seed, {n_core}/{n}) — {mech} EVICTS the latched affect mood. "
                   f"After a HIGH episode, two LOW episodes drop the held mood to "
                   f"{_m('evict_ratio_low')} of its high value (baseline ratchet "
                   f"{_m('baseline_ratchet_ratio')}; gate < 0.60), while the elapsed-TIME control "
                   f"(constant HIGH) holds at {_m('time_ratio_high')} (so the fall is caused by the "
                   f"LOWERED DRIVE, not by time), the pool RE-IGNITES to {_m('reignite_ratio')}, and "
                   f"PERSISTENCE SURVIVES at {_m('persistence_retention')} (baseline 0.62, gate "
                   f">= 0.50). Instrument: NMDA-off held {_m('nmda_off_retention')}; the same-substrate "
                   f"evict_out lesion restores the ratchet at {_m('lesion_evict_ratio')}. numpy-CPU; "
                   f"NO sim/ edit.")
    else:
        miss = [k for k, v in per_gate.items() if v < max(1, int(np.ceil(5 * n / 6)))]
        verdict = (f"NO-GO / BOUNDARY ({n}-seed, core {n_core}/{n}, instrument {n_instr}/{n}) — gates short of "
                   f"5/6: {miss}. evict {_m('evict_ratio_low')} (gate <0.60, baseline ratchet "
                   f"{_m('baseline_ratchet_ratio')}) | time-control {_m('time_ratio_high')} | "
                   f"re-ignition {_m('reignite_ratio')} | persistence {_m('persistence_retention')} | "
                   f"NMDA-off {_m('nmda_off_retention')} | lesion {_m('lesion_evict_ratio')}. "
                   f"Per THE LAW this banks the METHOD, not the capability: the pre-registered next method is "
                   f"intrinsic spike-frequency adaptation on the affect pools (--sfa).")
    return {
        "probe": f"affect mood EVICTION (P0.3-E): {mech} vs the latched-mood RATCHET",
        "verdict": (verdict if _verdict_block["status"] != "UNDEFINED" else
                    f"UNDEFINED ({n}-seed) — " + "; ".join(_verdict_block["undefined_reasons"])),
        "GO": go,
        **{k: _verdict_block[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "instrument_verified(baseline reproduces the ratchet)": bool(instrument_verified),
        "n_seeds_core_go": n_core, "n_seeds_instrument_ok": n_instr,
        "n_seeds_baseline_ratchet_reproduced": n_baseline_ratchet,
        "per_gate_seed_counts": per_gate, "means": means,
        "defect_reference": {
            "artifact": str(BASELINE_ARTIFACT),
            "failing_gate": "history_MAGNITUDE_r_mean>=0.6", "measured": 0.326,
            "ratchet": "held mood HIGH->LOW->LOW->silence stays at 100-102% of the high value"},
        "pre_registered_gate": {
            "G1_eviction": "max(held[1],held[2])/held[0] < 0.60 (LOW protocol)",
            "G2_drive_dependence": "min(held[1],held[2])/held[0] >= 0.60 (constant-HIGH protocol; the fall "
                                   "must be caused by the lowered drive, not by elapsed time)",
            "G3_reignition": "held[4]/held[0] >= 0.60 (eviction must not destroy the attractor)",
            "G4_persistence_survives": "held[0]/during[0] >= 0.50 (the baseline's own persistence gate)",
            "G5_nmda_off": "held[0]/during[0] < 0.10 with nmda_on=False",
            "G6_lesion_power": "evict_out gate=0 on the SAME substrate restores the ratchet (>= 0.90)",
            "A5_validity": "during[0](evict ON) >= 0.5 x during[0](baseline); else UNDEFINED, not a pass",
            "verdict_rule": "GO iff G1-G4 on >=5/6 seeds AND G5,G6 on >=5/6 AND the baseline reproduces "
                            "the ratchet on >=5/6",
            "kill_criterion": "no (weight x tau) point satisfies G1-G4 together => slow GABA_B feedback is "
                              "KILLED as the METHOD; next method = intrinsic spike-frequency adaptation "
                              "(cp_izh_a / cp_izh_d_increment on the affect pools, the --sfa arm)"},
        "mechanism": "each affect pool drives its OWN slow-feedback interneuron (sfb_plus/sfb_minus/"
                     "sfb_arousal, FS) which inhibits THAT SAME pool through RegionPathway(receptor='gaba_b') "
                     "-> the metabotropic GIRK K+ conductance (E_K=-90 mV, tau=gabab_tau_ms), i.e. delayed "
                     "self-inhibition that grows only under sustained incumbent firing. All eviction "
                     "synapses sit behind ONE runtime transmission gate 'evict_out' (the same-substrate "
                     "lesion). Appended LAST so neuron indices 0..359 and every baseline wiring draw are "
                     "preserved.",
        "HONEST_NOTE": "numpy-CPU (the backend, not a host shortcut). GABA_B is ADDITIVE in this engine: a "
                       "gaba_b-tagged synapse ALSO feeds the fast g_i, so the eviction limb is 'fast GABA_A "
                       "+ slow GABA_B', and the slowness attribution rests on the --arm fast control "
                       "(enable_gabab=False, identical wiring). The eviction limb adds 45 neurons, so the "
                       "no-wiring baseline arm is NOT substrate-identical; the load-bearing control is the "
                       "same-substrate evict_out gate=0 arm (G6). NO sim/ edit.",
        "per_seed": rows, "config": cfgd,
    }


# =============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true",
                    help="1-seed cheap-first: verify the ratchet reproduces, then sweep the GABA_B point")
    ap.add_argument("--gabab-weight", type=float, default=DEFAULT_GABAB_W)
    ap.add_argument("--gabab-tau", type=float, default=DEFAULT_GABAB_TAU_MS)
    ap.add_argument("--gabab-max", type=float, default=DEFAULT_GABAB_MAX,
                    help="GIRK saturation cap on g_gabab (0 = engine default, no cap)")
    # The FIRST smoke (seed 42, w in {0.25 .. 4.0} x tau in {150, 300}) located a sharp CLIFF between
    # w=0.25 (the brake rescales the latch: held 0.0938 -> 0.0586, ratchet ratio still 0.97) and w=0.50
    # (the held state is GONE: held[0]=0 while the pool still ignites during drive). These defaults resolve
    # that cliff, add a FAST tau (80 ms, so the brake can decay between episodes instead of equilibrating)
    # and the GIRK saturation cap (cfg.gabab_conductance_max — a bounded g_gabab is exactly the engine's
    # "graded subtraction at any presynaptic rate" lever).
    ap.add_argument("--sweep-weights", type=float, nargs="+", default=[0.28, 0.32, 0.36, 0.40, 0.45])
    ap.add_argument("--sweep-taus", type=float, nargs="+", default=[80.0, 150.0])
    ap.add_argument("--sweep-maxes", type=float, nargs="+", default=[0.0],
                    help="GIRK saturation caps on g_gabab to sweep (0 = no cap)")
    ap.add_argument("--sfa", type=float, nargs=2, metavar=("IZH_A", "IZH_D"), default=None,
                    help="ALSO run the pre-registered FALLBACK method: intrinsic spike-frequency adaptation "
                         "on the affect pools (cp_izh_a, cp_izh_d_increment), e.g. --sfa 0.005 60")
    ap.add_argument("--stp", action="store_true",
                    help="enable short-term plasticity (Tsodyks-Markram depression) as a candidate ratchet evictor")
    ap.add_argument("--stp-tau-d", type=float, default=None,
                    help="global STP depression recovery tau (ms); engine default 200 when --stp and this is unset")
    ap.add_argument("--stp-U", type=float, default=None,
                    help="global STP utilization/release prob = per-spike depression strength; engine default 0.15. "
                         "LOWER = gentler (the graded-eviction lever: tau_d 50-200 all ANNIHILATE the held state at 0.15)")
    ap.add_argument("--quench-pA", type=float, default=0.0,
                    help="ACTIVE-CLEAR quench: strong (NEGATIVE) external current pulsed into the three affect "
                         "pools during the post-drive silence of LOW episodes. 0.0 = OFF (the quench-off control "
                         "still shows the ratchet). e.g. --quench-pA -800. A DOCUMENTED host shortcut (the clear "
                         "command is host-issued), to be biologized to a spiking quench_fs pool.")
    ap.add_argument("--quench-ms", type=float, default=DEFAULT_QUENCH_MS,
                    help="quench-pulse duration (ms); must exceed the recurrent-NMDA decay (~100 ms) and fit inside "
                         "the post gap (POST_MS=300) so the clear is OFF during the read window (the anti-cheat)")
    # ---- BRAIN-BASED active-clear (the deliverable): the clear is done by a SPIKING quench_fs FS pool firing
    #      GABA_A onto the affect pools, recruited by the quench_drive neuromodulator -- NOT host current on the
    #      pools. --brain-quench turns it on; when set, --quench-pA acts ONLY as the ON trigger (magnitude unused,
    #      auto-armed if left 0). The real drive strength is --quench-drive-pA.
    ap.add_argument("--brain-quench", action="store_true",
                    help="BRAIN-BASED active-clear: a spiking quench_fs FS pool (GABA_A onto the affect pools, "
                         "recruited by the quench_drive neuromodulator) does the clear, not host current. The "
                         "deliverable conversion of the host --quench-pA shortcut.")
    ap.add_argument("--quench-fs-n", type=int, default=QUENCH_FS_N,
                    help="number of quench_fs FS interneurons (brain-quench)")
    ap.add_argument("--quench-gaba-w", type=float, default=QUENCH_GABA_W,
                    help="quench_fs -> affect pool GABA_A weight (brain-quench); the clear strength via synapses")
    ap.add_argument("--quench-drive-pA", type=float, default=QUENCH_DRIVE_PA,
                    help="excitability_drive sensitivity (pA into each quench_fs neuron @ conc 1.0) recruiting "
                         "quench_fs during the clear window (brain-quench). Swept on the pool.")
    ap.add_argument("--recur-weight", type=float, default=DEFAULT_RECUR_WEIGHT)
    ap.add_argument("--ou-pA", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    sfa = tuple(a.sfa) if a.sfa else None
    # BRAIN-QUENCH trigger: --quench-pA is the universal ON/OFF trigger (0 => the clear never fires). In brain
    # mode its MAGNITUDE is unused (the drive is --quench-drive-pA), so auto-arm the trigger if the user left it
    # at 0 -- the OFF control inside 1b/run_point explicitly passes quench_pA=0.0, which stays OFF regardless.
    quench_pA = a.quench_pA
    if a.brain_quench and quench_pA == 0.0:
        quench_pA = -1.0   # trigger sentinel only (never injected -- brain mode routes through step_brain_quench)
    if a.smoke:
        res = run_smoke(a.seeds[0], a.sweep_weights, a.sweep_taus, maxes=a.sweep_maxes, ou_pA=a.ou_pA,
                        recur_weight=a.recur_weight, sfa=sfa, stp=a.stp, stp_tau_d=a.stp_tau_d, stp_U=a.stp_U,
                        quench_pA=quench_pA, quench_ms=a.quench_ms, brain_quench=a.brain_quench,
                        quench_fs_n=a.quench_fs_n, quench_gaba_w=a.quench_gaba_w,
                        quench_drive_pA=a.quench_drive_pA)
        res["config"] = {"seed": a.seeds[0], "sweep_weights": a.sweep_weights, "sweep_taus": a.sweep_taus,
                         "sweep_maxes": a.sweep_maxes, "recur_weight": a.recur_weight, "ou_pA": a.ou_pA,
                         "sfa": sfa, "stp": bool(a.stp), "stp_tau_d": a.stp_tau_d, "stp_U": a.stp_U,
                         "quench_pA": quench_pA, "quench_ms": a.quench_ms, "brain_quench": bool(a.brain_quench),
                         "quench_fs_n": a.quench_fs_n, "quench_gaba_w": a.quench_gaba_w,
                         "quench_drive_pA": a.quench_drive_pA,
                         "LOW_LEVELS": list(LOW_LEVELS), "HIGH_LEVELS": list(HIGH_LEVELS),
                         "timings_ms": {"settle": SETTLE_MS, "base_probe": BASE_PROBE_MS, "drive": DRIVE_MS,
                                        "post": POST_MS, "read": READ_MS},
                         "SFB_N": SFB_N, "SFB_EXC_W": SFB_EXC_W,
                         "QUENCH_FS_N": QUENCH_FS_N, "QUENCH_GABA_W": QUENCH_GABA_W,
                         "QUENCH_GABA_DENSITY": QUENCH_GABA_DENSITY, "QUENCH_DRIVE_PA": QUENCH_DRIVE_PA}
        p = Path(str(a.out).replace(".json", "_smoke.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2, default=str))
        print(f"[EVICT SMOKE] wrote {p}  ({round(time.time() - t0, 1)}s)", flush=True)
        return 0

    print(f"[EVICT] {len(a.seeds)}-seed pre-registered gate @ gabab_w={a.gabab_weight} tau={a.gabab_tau} ms",
          flush=True)
    rows = run_battery(a.seeds, a.gabab_weight, a.gabab_tau, ou_pA=a.ou_pA, recur_weight=a.recur_weight,
                       gabab_max=a.gabab_max, sfa=sfa, stp=a.stp, stp_tau_d=a.stp_tau_d, stp_U=a.stp_U,
                       quench_pA=quench_pA, quench_ms=a.quench_ms, brain_quench=a.brain_quench,
                       quench_fs_n=a.quench_fs_n, quench_gaba_w=a.quench_gaba_w,
                       quench_drive_pA=a.quench_drive_pA)
    cfgd = {"seeds": a.seeds, "gabab_weight": a.gabab_weight, "gabab_tau_ms": a.gabab_tau,
            "gabab_max": a.gabab_max, "recur_weight": a.recur_weight, "ou_pA": a.ou_pA, "sfa": sfa,
            "stp": bool(a.stp), "stp_tau_d": a.stp_tau_d, "stp_U": a.stp_U,
            "quench_pA": quench_pA, "quench_ms": a.quench_ms, "brain_quench": bool(a.brain_quench),
            "quench_fs_n": a.quench_fs_n, "quench_gaba_w": a.quench_gaba_w, "quench_drive_pA": a.quench_drive_pA,
            "LOW_LEVELS": list(LOW_LEVELS), "HIGH_LEVELS": list(HIGH_LEVELS),
            "timings_ms": {"settle": SETTLE_MS, "base_probe": BASE_PROBE_MS, "drive": DRIVE_MS,
                           "post": POST_MS, "read": READ_MS},
            "SFB_N": SFB_N, "SFB_EXC_W": SFB_EXC_W, "SFB_EXC_DENSITY": SFB_EXC_DENSITY,
            "SFB_INH_DENSITY": SFB_INH_DENSITY, "N_AFF": N_AFF, "DRIVE_GAIN_PA": DRIVE_GAIN_PA}
    summary = summarize(rows, a.seeds, cfgd)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[EVICT] VERDICT: {summary['verdict']}", flush=True)
    print(f"[EVICT] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
