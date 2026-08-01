"""DR-1 CURIOSITY INVERSION — the ON-BRIDGE SPIKING realization (the numpy GO promoted to a real
SimulationBridge). Roadmap: docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md (§2.8, DR-1 Phase-0->on-bridge).

The numpy cheap-first (`_curiosity_seek_learn_cheap_first_probe.py`, 6-seed GO) proxied the curiosity
modulator + the value at RATE level. This runner realizes the loop with REAL SPIKES on ONE `SimulationBridge`:

  (1) CURIOSITY DRIVE = SPIKING (fills the reserved `from_novelty` hook).
      A `curiosity` neuromodulator whose production rule is `from_novelty` reads the brain's epistemic-gap
      scalar `core_config.current_novelty_signal` (the Bogacz-Brown familiarity-gate novelty of the concept
      under consideration) and drives an `excitability_drive` on an ASK pool (scope=group:ask). HIGH novelty
      -> higher curiosity concentration -> more ASK-pool SPIKING. The WANTING is read from real
      `cp_firing_states[ask]` (Hz), so gate (a) corr(gap, wanting) is measured ON SPIKES.
      (This is the ONE sim/ edit: `from_novelty` filled in sim/neuromodulators.py + the additive
      `current_novelty_signal`/`novelty_baseline` fields in sim/config.py; additive, default-off,
      byte-identical when unused; unit-pinned by tests/test_from_novelty_curiosity.py.)

  (2) LEARNING-PROGRESS REWARD via the SPIKING-SNc RPE MACHINERY (the `_limbic_core_rpe_battery` organ).
      reward_us (US) --exc--> snc <-- striosome_value (GABA_B/GIRK -V critic). Each ASK delivers the intrinsic
      reward r = LEARNING PROGRESS (g_before - g_after) as a reward_us drive; the SNc computes the RPE on SPIKES
      (bursts when r>V, at neutral when r~0), and the DA teaching signal is READ from the SNc firing (DA release
      ∝ SNc rate; Schultz), routed as the three-factor plasticity gate so the striosome critic LEARNS a value V.
      The VETO value is a per-concept expected-learning-progress ELP (the numpy probe's TD tracker, optimistic
      init) fed by the SPIKING reward READ = (SNc-with-reward - SNc-without-reward for the same concept, so the
      learned V cancels and the read isolates r). A LEARNABLE concept evokes a positive SNc reward burst ->
      reward_read>0 -> ELP stays high; a NOISY/un-learnable concept realizes r~0 each ask -> reward_read~0 ->
      ELP DECAYS below the veto threshold -> the policy STOPS asking it, WHILE its epistemic gap stays HIGH
      (never spuriously learned). Curious AND honest (Oudeyer/Schmidhuber learning-progress; the noisy-TV cure).
      (The SNc uses a rebound-free RS integrator, NOT the DOPAMINE preset which post-inhibitory-rebound-bursts
      into a runaway once the critic has learned any V; the strio->snc GABA_B value-subtraction is kept gentle.)

  (3) EPISTEMIC GAP = the REAL on-bridge Bogacz-Brown familiarity gate (`RealAntiHebbianFamiliarity`,
      catalog D.04 perirhinal repetition suppression) — reused-by-import, the SAME gate that drives the
      no-confab moat. novelty(x) ~0 familiar / ~1 novel. INGEST = imprint the teacher's render (raises
      familiarity -> lowers future novelty). moat-by-construction: only asks when the gate reads NOVEL; the
      CONFIDENT set is a subset of the INGESTED set.

GO GATES (mirror the numpy probe; the runner prints its OWN verdict):
  (a) corr(epistemic-gap g, SPIKING wanting) >= 0.9      -- the ASK-pool firing tracks the gap.
  (b) ask-rate on UNKNOWN (novel) >= 2x on KNOWN         -- the spiking drive biases seeking toward the gap.
  (c) post-answer confidence on a learnable concept RISES above the abstain floor.
  NOISY-STOPS (decisive honesty guard): late noisy ask-rate << early WHILE noisy g stays HIGH AND the learned
      value (ELP) vetoed it -- curious AND honest, not noise-chasing / confabulating.
  MOAT: confident set (gate-familiar) subset of ingested (asked) set.
ANTI-CHEATS: lesion the curiosity modulator (no drive -> no asking); yoked-random reward (value gets an
  uninformative signal -> wrongly vetoes learnable + wastes budget on noisy -> masters fewer); permuted-gap
  (corr collapses); (reported) critic-lesion.

Reuse-by-import (the familiarity gate + the limbic-core build pattern). SPIKING on a real bridge:
CPU-smoke first (SIM_BACKEND=numpy --smoke), then GPU 6-seed (SIM_BACKEND=cupy).
Run: SIM_BACKEND=cupy python -u -m research.runners._curiosity_seek_learn_onbridge_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The brain's EXISTING uncertainty signal — reused verbatim (the no-confab moat's gate).
from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity  # noqa: E402


# ------------------------------- the ENVIRONMENT (host-legit) -------------------------------
class World:
    """The concepts/teacher (ENVIRONMENT, host-legit per the brain-based-only rule). LEARNABLE = fixed unit
    code + tiny dimension-independent jitter (masterable over a few asks). NOISY = a FRESH random code every
    render (the noisy TV: un-learnable). Identical structure to the numpy cheap-first World, parameterized."""

    def __init__(self, seed, D, n_learn, n_noisy, obs_noise):
        self.rng = np.random.default_rng(seed * 7 + 1)
        self.D, self.n_learn, self.n_noisy, self.obs_noise = D, n_learn, n_noisy, obs_noise
        self.concepts = list(range(n_learn + n_noisy))
        self.is_noisy = {c: (c >= n_learn) for c in self.concepts}
        self._code = {}
        for c in self.concepts:
            if not self.is_noisy[c]:
                v = self.rng.standard_normal(D)
                self._code[c] = v / (np.linalg.norm(v) + 1e-12)

    def render(self, c):
        if self.is_noisy[c]:
            v = self.rng.standard_normal(self.D)                     # fresh random -> nothing to learn
        else:
            n = self.rng.standard_normal(self.D)
            n = n / (np.linalg.norm(n) + 1e-12) * self.obs_noise     # dimension-independent jitter
            v = self._code[c] + n
        return v / (np.linalg.norm(v) + 1e-12)


# ------------------------------- the ON-BRIDGE limbic + ASK organ -------------------------------
def build_curiosity_bridge(seed, n_concepts, *, n_per_cue=40, n_strio=60, n_reward_us=40,
                           n_snc=30, n_ask=80, cue_to_strio_weight=11.0,
                           reward_us_to_snc_weight=10.0, strio_to_snc_weight=2.0,
                           gabab_prop=0.22, gabab_tau_decay=150.0, reward_learning_rate=0.30,
                           curiosity_prod_sensitivity=0.10,
                           curiosity_excit_sensitivity=320.0, curiosity_decay_tau=50.0,
                           enable_heterogeneity=True):
    """One SimulationBridge holding BOTH the spiking-SNc RPE value critic (reward_us->snc<-striosome(GABA_B),
    cue->striosome PLASTIC) AND the ASK/curiosity pool driven by the `curiosity` neuromodulator (from_novelty
    -> excitability_drive on group:ask). Per-concept `cue` slices give disjoint credit assignment.

    The DA teaching signal is the SNc SPIKING RPE read directly from cp_firing_states (DA release is
    proportional to SNc firing rate; Schultz): the reward_us->snc<-striosome(GABA_B) organ computes the RPE
    delta = r - V ON SPIKES, and the read snc burst-minus-neutral gates the three-factor plasticity of the
    critic (routed via current_reward_signal, exactly as a spiking read-out feeds a downstream gate). This
    replaces the autonomous da-concentration integrator (from_region_firing_signed), which has a
    baseline-drift artifact under the pulsed ask protocol (a silent inter-ask gap pushes the da concentration
    below baseline, mis-signing the teaching signal). reward_aversive_scale=1.0 keeps LTD symmetric so an
    OMITTED predicted reward (noisy concept: r~0, V>0 -> SNc dips) depresses the value below the veto floor.

    Set `curiosity_excit_sensitivity=0.0` for the CURIOSITY-LESION control (no drive -> ASK pool silent)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.current_novelty_signal = 0.0            # the DR-1 curiosity drive input (written each step)
    cfg.novelty_baseline = 0.0
    cfg.reward_aversive_scale = 1.0             # symmetric LTD so an omitted predicted reward depresses V
    cfg.stdp_w_max = 40.0

    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    n_cue = n_concepts * n_per_cue
    cfg.brain_regions = [
        # per-concept cue slices (the state/concept identity for the value critic)
        BrainRegion(name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="reward_us", n_neurons=n_reward_us, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        # SNc as a rebound-FREE integrator (IZH2007_RS), NOT the IZH2007_DOPAMINE preset. The dopamine preset
        # POST-INHIBITORY-REBOUND-BURSTS: the strio's hyperpolarizing GABA_B value-subtraction deinactivates its
        # T-current -> a ~400 Hz rebound runaway once the critic has learned any V (strio just 20 Hz drives snc
        # to 421 Hz). The RS integrator computes snc = tonic + r - V cleanly (no rebound), which is exactly the
        # Rescorla-Wagner reward-prediction-error the machinery needs; the DA teaching signal is still read from
        # this SNc firing (DA release ∝ SNc rate). syn_reversal_potential_i_override=-90 keeps the GABA_B
        # value-subtraction firmly hyperpolarizing (inhibitory) so a large learned V subtracts, not excites.
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS,
                    syn_reversal_potential_i_override=-90.0),
        # the ASK / curiosity pool (driven by the curiosity neuromodulator's excitability_drive)
        BrainRegion(name="ask", n_neurons=n_ask, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=float(cue_to_strio_weight),
                      weight_jitter=0.5, plastic=True),
        RegionPathway(from_region="reward_us", to_region="snc",
                      density=0.6, weight_mean=float(reward_us_to_snc_weight),
                      weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False, receptor="gaba_b"),
    ]
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        # CURIOSITY: from_novelty -> excitability_drive on the ASK pool (the DR-1 sim/ fill). This is the
        # ONLY registered neuromodulator: the critic's DA teaching signal is the SNc firing read directly
        # (see the docstring), so `effective_signal` in the reward path is `current_reward_signal` = the
        # SNc-derived RPE, not an autonomous da concentration.
        NeuromodulatorConfig(
            name="curiosity", baseline=0.0, decay_tau_ms=float(curiosity_decay_tau),
            concentration_min=0.0, concentration_max=5.0,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="group:ask",
                                     sensitivity=float(curiosity_excit_sensitivity))],
            production_rules=[ProductionRule(rule_type="from_novelty",
                                             sensitivity=float(curiosity_prod_sensitivity))]),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _idx(bridge, name):
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _advance(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (
        bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _settle(bridge, n_steps=40):
    """Clean-reset (limbic-core protocol): zero external current + the slow GABA_B/GIRK, run a silent gap so
    fast conductances/membranes decay to rest before the next frozen read (the order-artifact fix)."""
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_novelty_signal = 0.0
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    for _ in range(n_steps):
        _advance(bridge)


# The dynamic per-neuron state that `_run_one_simulation_step` mutates and that DRIFTS across asks. The
# load-bearing one is `cp_recovery_variable_u` — the Izhikevich slow spike-frequency adaptation current, which
# ACCUMULATES with every spike (EMERGE-61 root cause) so the SAME concept reads a different value/burst late in
# the run vs early. A byte-for-byte wash-out to the clean post-init state before each want/value/deliver op makes
# every op a function of the LEARNED weights alone (drift-free), NOT of prior-ask history. It restores DYNAMIC
# state only (v/u/conductances/firing/STP), never the learned cue->striosome weights (cp_connections).
_STATE_ARRAYS = (
    "cp_membrane_potential_v", "cp_recovery_variable_u",
    "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
    "cp_conductance_g_gabab", "cp_conductance_g_graded", "cp_conductance_g_coincidence",
    "cp_firing_states", "cp_stp_x", "cp_stp_u", "cp_last_spike_time", "cp_ou_current",
)


def _snapshot_state(bridge):
    snap = {}
    for name in _STATE_ARRAYS:
        arr = getattr(bridge, name, None)
        if arr is not None:
            snap[name] = np.asarray(_host(arr)).copy()
    return snap


def _restore_state(bridge, snap):
    from sim.backend import from_host
    for name, val in snap.items():
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = from_host(val)
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_novelty_signal = 0.0
    bridge.core_config.current_reward_signal = 0.0
    # Reset the neuromodulator CONCENTRATIONS to baseline too: the wash-out restores neuron state but the
    # curiosity concentration accumulates across ops (decay tau 50ms >> the ~10-step inter-op gap), so without
    # this the SAME novelty reads a rising want on successive ops (drift). Baseline reset makes every want read
    # a drift-free function of the novelty signal alone.
    mgr = getattr(bridge, "neuromodulator_manager", None)
    if mgr is not None:
        for nm_cfg in mgr._configs:
            mgr._concentrations[nm_cfg.name] = float(nm_cfg.baseline)


# the region names allocated on the bridge (used to build the index map)
drives_regions = ("cue", "striosome_value", "reward_us", "snc", "ask")


def _measure_snc_neutral(bridge, idx_map, xp, snap0):
    """The SNc firing rate over a delivery-length window (W_REWARD steps) at tonic drive ALONE — no cue (V
    absent) and no reward_us (r absent) — starting from the SAME restored clean state as a delivery build.
    Measuring over the identical window cancels the tonic startup transient, so the per-ask RPE
    `snc_burst - snc_neutral` = (tonic + r - V) - tonic = r - V (the Rescorla-Wagner error), drift-free."""
    _restore_state(bridge, snap0)
    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
    spk = 0
    for i in range(W_WARMUP + W_MEASURE):
        _advance(bridge)
        if i >= W_WARMUP:
            spk += int(bridge.cp_firing_states[snc_idx].sum())
    _restore_state(bridge, snap0)
    return spk / max(n_snc, 1) / (W_MEASURE * 1e-3)


def _lesion_gabab_mask(bridge):
    """Critic-lesion: zero the GABA_B routing mask so the value subtraction (-V) vanishes -> the SNc can no
    longer dip on omission -> the noisy value cannot decay -> noisy is NOT vetoed (the reported anti-cheat)."""
    from sim.backend import get_backend
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    xp, _ = get_backend()
    n_was = int(_host(m).sum())
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


# ------------------------------- config (mirrors the numpy cheap-first) -------------------------------
D = 1024
N_LEARN = 8
N_NOISY = 4
N_TURNS = 220
ASK_BUDGET = 30
NOVEL_THRESH = 0.35
EPS = 0.10
OBS_NOISE = 0.70
WANT_FLOOR_HZ = 18.0         # a concept is drive-active (candidate) iff its spiking wanting exceeds this
SNC_TONIC_PA = 220.0
CUE_DRIVE_PA = 600.0
US_GAIN_PA = 2400.0          # reward_us drive per unit learning-progress (LP in [0, ~1])
SNC_SCALE = 60.0             # Hz normalizer for the SNc-read RPE (snc_burst - snc_neutral) -> ~[-1.5, 1.5]
RPE_GAIN = 6.0               # gain so lr*rpe dominates the reward-independent direct-STDP drift
# The VETO value: a per-concept expected-learning-progress ELP (the numpy probe's exact TD tracker), fed by the
# SPIKING SNc reward read (the burst the LP reward evokes, paired against the same concept's no-reward burst so
# the learned striosome V cancels -> the read isolates r). Optimistic init so every novel concept is TRIED a
# couple times; a noisy concept's reward-read ~0 < init -> ELP decays below VALUE_THRESH -> vetoed (WHILE its gap
# stays HIGH). This robustly realizes the veto on spikes without reading the (drift-prone) striosome rate.
ELP_INIT = 0.35
BETA_ELP = 0.55              # TD rate for the per-concept expected-learning-progress value
VALUE_THRESH = 0.12          # veto floor on ELP
SNC_REWARD_SCALE = 180.0     # Hz normalizer mapping the SNc reward burst (snc_with - snc_without) -> ~[0, 1]
W_WANT = 18
W_WARMUP = 18                # SNc/strio settle to steady state after a wash-out (skip the tonic transient)
W_MEASURE = 24               # then measure the SNc rate over this window (eligibility builds across both)
W_REWARD = W_WARMUP + W_MEASURE  # total eligibility-BUILD + SNc-read window
W_APPLY = 12                 # the reward-conversion window (SNc-read RPE * eligibility)
W_VALUE = 25
W_SETTLE = 40
# --- SPIKING-VETO conversion (2026-08-01, additive) ---------------------------------------------------------
# The 2026-07-31 critic-lesion finding named a shortcut: the noisy-veto reads a HOST-side ELP tracker (a Python
# TD low-pass fed by the SNc paired-subtraction read snc_B-snc_A), NOT the spiking substrate — proven because the
# veto survived the GABA_B critic lesion 6/6. Its conversion target: compute the veto decision FROM THE SPIKING
# STRIOSOME VALUE. Under --spiking-veto the veto value tracks the LEARNED striosome rate `read_value(c)` (a direct
# spiking read, drift-free via the same wash-out): a LEARNABLE concept's reward keeps V up; a NOISY concept's
# omission (r~0, RPE=r-V<0 via the GABA_B critic -> LTD) depresses V below a floor. The load-bearing NEW
# dissociation: with the striosome critic LESIONED the spiking veto must COLLAPSE (V no longer dips -> noisy not
# vetoed) — i.e. the striosome becomes load-bearing exactly because the veto now reads it (the opposite of the
# host-ELP veto, which survived the lesion). Floor = fraction of the fresh (pre-learning) value v0.
STRIO_VETO_FRAC = 0.60       # veto floor as a fraction of the fresh striosome value v0 (un-tried concept ~ v0)
BETA_STRIO = 0.55            # TD rate low-passing the spiking striosome value read (same as BETA_ELP)


def run(seed, mode, *, n_learn=N_LEARN, n_noisy=N_NOISY, n_turns=N_TURNS, ask_budget=ASK_BUDGET,
        d=D, verbose=False, spiking_veto=False, **build_kw):
    from sim.backend import get_backend
    xp, _ = get_backend()
    rng = np.random.default_rng(seed * 101 + 5)
    n_concepts = n_learn + n_noisy
    world = World(seed, d, n_learn, n_noisy, OBS_NOISE)
    gate = RealAntiHebbianFamiliarity()
    concepts = world.concepts
    perm = {c: concepts[(i + 3) % len(concepts)] for i, c in enumerate(concepts)}

    lesion_curiosity = (mode == "lesion")
    bk = dict(build_kw)
    if lesion_curiosity:
        bk["curiosity_excit_sensitivity"] = 0.0
    bridge, cfg = build_curiosity_bridge(seed, n_concepts, **bk)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in drives_regions}
    cue_all = _host(idx_map["cue"]).astype(np.int64)
    n_per_cue = len(cue_all) // n_concepts
    cue_slice = {c: xp.asarray(cue_all[c * n_per_cue:(c + 1) * n_per_cue]) for c in concepts}
    if mode == "critic_lesion":
        _lesion_gabab_mask(bridge)

    # Wash-out reference: the CLEAN post-init dynamic state. Every want/value/deliver op restores to it first,
    # so each is a function of the LEARNED weights alone (no cross-ask adaptation drift; EMERGE-61 fix).
    _settle(bridge, W_SETTLE)
    snap0 = _snapshot_state(bridge)

    snc_neutral = _measure_snc_neutral(bridge, idx_map, xp, snap0)

    def read_value(c):
        _restore_state(bridge, snap0)
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        bridge.core_config.current_novelty_signal = 0.0
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
        strio_idx = idx_map["striosome_value"]; n_strio = len(_host(strio_idx)); spk = 0
        for _ in range(W_VALUE):
            _advance(bridge)
            spk += int(bridge.cp_firing_states[strio_idx].sum())
        cfg.reward_learning_rate = saved
        return spk / max(n_strio, 1) / (W_VALUE * 1e-3)

    # The VETO value: expected-learning-progress per concept (optimistic init), TD-tracked from the SPIKING SNc
    # reward read. v0_per is the striosome's initial firing (reported at the end vs the learned value as evidence
    # the critic moved). ELP is the numpy probe's value logic, now fed by a spiking reward.
    v0_per = {c: read_value(c) for c in concepts}
    v0 = float(np.mean(list(v0_per.values())))
    ELP = {c: ELP_INIT for c in concepts}
    # SPIKING-VETO value: the LEARNED striosome rate itself (Hz), TD-low-passed. Init at each concept's fresh
    # read v0_per[c] (un-tried ~ un-depressed, so nothing is vetoed before it has been asked). Floor is a
    # fraction of the mean fresh value; a noisy concept whose learned value dips below it is vetoed.
    Vstrio = {c: v0_per[c] for c in concepts}
    strio_veto_floor = STRIO_VETO_FRAC * v0

    def read_want(c, novelty):
        _restore_state(bridge, snap0)
        bridge.core_config.current_novelty_signal = float(novelty)
        ask_idx = idx_map["ask"]; n_ask = len(_host(ask_idx)); spk = 0
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
        for _ in range(W_WANT):
            _advance(bridge)
            spk += int(bridge.cp_firing_states[ask_idx].sum())
        cfg.reward_learning_rate = saved
        return spk / max(n_ask, 1) / (W_WANT * 1e-3)

    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))

    def _snc_window(c, LP):
        """One SNc measurement window from the restored clean state: drive cue slice + reward_us(∝LP) + snc
        tonic for W_REWARD steps (building cue->striosome eligibility on the way), return the mean SNc Hz."""
        _restore_state(bridge, snap0)
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        bridge.cp_external_input_current[idx_map["reward_us"]] = xp.float32(US_GAIN_PA * max(LP, 0.0))
        bridge.cp_external_input_current[idx_map["snc"]] = xp.float32(SNC_TONIC_PA)
        bridge.core_config.current_reward_signal = 0.0            # build eligibility only (no conversion)
        spk = 0
        for i in range(W_WARMUP + W_MEASURE):
            _advance(bridge)
            if i >= W_WARMUP:                                     # skip the tonic startup transient
                spk += int(bridge.cp_firing_states[snc_idx].sum())
        return spk / max(n_snc, 1) / (W_MEASURE * 1e-3)

    def deliver_reward(c, LP):
        """Deliver r = learning-progress LP through the spiking-SNc RPE machinery, three-factor.
        Window B (WITH reward): cue + reward_us(∝LP) + snc tonic builds cue->striosome eligibility while the SNc
        computes the RPE on spikes; the SNc burst minus the neutral tonic (= r - V; DA release ∝ SNc firing) gates
        the three-factor APPLY that converts eligibility -> striosome weight (the value critic LEARNS V).
        Window A (WITHOUT reward): the SAME concept's SNc with cue but reward_us=0. The paired read
        `snc_B - snc_A` isolates the reward's SNc contribution (the learned V cancels) -> a spiking reward
        magnitude in [0,1] that TD-updates the per-concept expected-learning-progress ELP (the veto value). A
        noisy concept evokes ~no reward burst -> reward_read~0 -> ELP decays -> vetoed, WHILE its gap stays HIGH.
        Returns (snc_burst_B_Hz, reward_read)."""
        bridge.cp_eligibility_trace[:] = 0.0                       # disjoint per-ask credit
        snc_B = _snc_window(c, LP)                                 # WITH reward (also builds eligibility)
        rpe = float(np.clip((snc_B - snc_neutral) / SNC_SCALE, -1.5, 1.5)) * RPE_GAIN
        # APPLY: the SNc-derived RPE gates the three-factor conversion of the built eligibility (critic learns V).
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx_map["snc"]] = xp.float32(SNC_TONIC_PA)
        bridge.core_config.current_reward_signal = rpe
        for _ in range(W_APPLY):
            _advance(bridge)
        bridge.core_config.current_reward_signal = 0.0
        snc_A = _snc_window(c, 0.0)                                # WITHOUT reward -> isolates r (V cancels)
        reward_read = float(np.clip((snc_B - snc_A) / SNC_REWARD_SCALE, 0.0, 1.0))
        return snc_B, reward_read

    # bookkeeping (mirror the numpy probe)
    corr_gap, corr_want = [], []
    asked = set(); ask_events = []; conf_first_ask = {}; n_asks = 0
    elig_unknown = elig_known = ask_unknown = ask_known = 0
    third = max(1, n_turns // 3)
    noisy_elig = [0, 0, 0]; noisy_ask = [0, 0, 0]
    yoke_pool = rng.permutation(np.linspace(0.0, 1.0, 200)); yi = 0
    snc_learn_burst, snc_noisy_burst = [], []

    for turn in range(n_turns):
        if n_asks >= ask_budget:
            break
        true_gaps = {c: gate.novelty(world.render(c)) for c in concepts}
        # the gap the novel-gate uses (permuted mode mis-maps it)
        gate_gap = ({c: true_gaps[perm[c]] for c in concepts} if mode == "permuted" else true_gaps)
        # the novelty signal fed to the curiosity drive per concept (the modulator input)
        if mode == "yoked":
            drive_nov = {c: float(rng.choice(list(true_gaps.values()))) for c in concepts}
        elif mode == "permuted":
            drive_nov = {c: float(true_gaps[perm[c]]) for c in concepts}
        else:
            drive_nov = {c: float(true_gaps[c]) for c in concepts}

        # the SPIKING wanting is read for EVERY concept (as the numpy probe reads its modulator for all), so the
        # (gap, wanting) corr spans the FULL g-range [~0 mastered .. ~1 novel] -> a real correlation, not a
        # narrow-range noise estimate. A concept is a CANDIDATE to ask iff it reads NOVEL (the moat) AND is
        # drive-active (spiking wanting above the baseline floor) AND is not value-vetoed (expected-LP ELP > thr).
        want = {c: read_want(c, drive_nov[c]) for c in concepts}
        for c in concepts:
            corr_gap.append(true_gaps[c]); corr_want.append(want[c])
            unknown = true_gaps[c] > NOVEL_THRESH
            if unknown:
                elig_unknown += 1
            else:
                elig_known += 1
            if world.is_noisy[c] and unknown:
                noisy_elig[min(turn // third, 2)] += 1

        # the veto: host-ELP tracker (default) OR the SPIKING striosome value read (--spiking-veto).
        not_vetoed = ((lambda c: Vstrio[c] > strio_veto_floor) if spiking_veto
                      else (lambda c: ELP[c] > VALUE_THRESH))
        cands = [c for c in concepts
                 if gate_gap[c] > NOVEL_THRESH and want[c] > WANT_FLOOR_HZ and not_vetoed(c)]
        if not cands:
            continue

        if rng.random() < EPS:
            c_ask = int(rng.choice(cands))
        else:
            mx = max(want[c] for c in cands)
            c_ask = int(rng.choice([c for c in cands if want[c] >= mx - 1e-9]))

        if true_gaps[c_ask] > NOVEL_THRESH:
            ask_unknown += 1
        else:
            ask_known += 1
        if world.is_noisy[c_ask]:
            noisy_ask[min(turn // third, 2)] += 1

        g_before = true_gaps[c_ask]
        if (not world.is_noisy[c_ask]) and c_ask not in conf_first_ask:
            conf_first_ask[c_ask] = 1.0 - g_before
        gate.imprint(world.render(c_ask))                      # INGEST
        g_after = gate.novelty(world.render(c_ask))            # fresh test render
        if mode == "yoked":
            yb = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            ya = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            LP = yb - ya
        else:
            LP = g_before - g_after

        snc_hz, reward_read = deliver_reward(c_ask, LP)        # spiking-SNc RPE; reward_read = spiking r
        # the value's learning signal: REAL/permuted = the spiking SNc reward read; YOKED = the uninformative
        # yb-ya draw (mean 0, numpy-faithful) so the value CANNOT tell learnable from noise -> it wrongly
        # vetoes learnable asked once (stuck un-mastered) AND wastes budget on un-vetoed noisy -> masters fewer.
        elp_reward = LP if mode == "yoked" else reward_read
        ELP[c_ask] += BETA_ELP * (elp_reward - ELP[c_ask])     # per-concept expected-learning-progress value
        if spiking_veto:
            # read the LEARNED striosome value ON SPIKES (drift-free via wash-out) and TD-low-pass it: the veto
            # decision now comes FROM the substrate (striosome rate), not the host SNc-paired-subtraction read.
            Vstrio[c_ask] += BETA_STRIO * (read_value(c_ask) - Vstrio[c_ask])
        (snc_noisy_burst if world.is_noisy[c_ask] else snc_learn_burst).append(snc_hz)

        asked.add(c_ask); n_asks += 1
        ask_events.append((turn, c_ask, float(g_before), float(g_before - g_after), bool(world.is_noisy[c_ask])))
        if verbose and n_asks <= 10:
            print(f"    [ask {n_asks:02d}] c={c_ask} noisy={world.is_noisy[c_ask]} g {g_before:.2f}->{g_after:.2f}"
                  f" LP {g_before-g_after:+.2f} sncHz {snc_hz:5.1f} reward_read {reward_read:.2f} "
                  f"ELP {ELP[c_ask]:.2f} (thr {VALUE_THRESH})", flush=True)

    # ---- metrics (mirror the numpy probe) ----
    corr_gap = np.array(corr_gap); corr_want = np.array(corr_want)
    corr = (float(np.corrcoef(corr_gap, corr_want)[0, 1])
            if corr_want.std() > 1e-9 and corr_gap.std() > 1e-9 else 0.0)
    rate_unknown = ask_unknown / max(elig_unknown, 1)
    rate_known = ask_known / max(elig_known, 1)
    ratio_b = rate_unknown / (rate_known + 1e-9)

    conf_after = {c: 1.0 - gate.novelty(world.render(c)) for c in concepts}
    learn_after = [conf_after[c] for c in range(n_learn)]
    learn_before = [conf_first_ask.get(c, 0.0) for c in range(n_learn) if c in conf_first_ask]
    abstain_floor = float(np.mean([conf_after[c] for c in range(n_learn, n_learn + n_noisy)]))
    conf_rise = float(np.mean(learn_after)) - (float(np.mean(learn_before)) if learn_before else 0.0)
    conf_after_mean = float(np.mean(learn_after))

    late_asks = [e for e in ask_events if e[0] >= 2 * third]
    late_learnable_frac = (sum(1 for e in late_asks if not e[4]) / len(late_asks)) if late_asks else 1.0
    noisy_early_rate = noisy_ask[0] / max(noisy_elig[0], 1)
    noisy_late_rate = noisy_ask[2] / max(noisy_elig[2], 1)
    noisy_g_final = float(np.mean([gate.novelty(world.render(c)) for c in range(n_learn, n_learn + n_noisy)]))
    # the VETO value is the per-concept ELP (spiking-reward-fed TD tracker). A noisy concept whose ELP fell
    # below VALUE_THRESH was vetoed (stopped being asked) BECAUSE its reward-read said no learning-progress.
    asked_noisy = [c for c in range(n_learn, n_learn + n_noisy) if c in asked]
    if spiking_veto:
        # the veto quantity IS the spiking striosome value Vstrio, thresholded at strio_veto_floor
        veto_val, veto_thr = Vstrio, strio_veto_floor
    else:
        veto_val, veto_thr = ELP, VALUE_THRESH
    noisy_V_final = float(np.mean([veto_val[c] for c in (asked_noisy or range(n_learn, n_learn + n_noisy))]))
    noisy_vetoed_frac = (float(np.mean([veto_val[c] <= veto_thr for c in asked_noisy])) if asked_noisy else 0.0)
    noisy_vetoed = bool(noisy_vetoed_frac >= 0.75)   # most asked-noisy concepts' veto value fell below threshold
    asked_learn_V = [veto_val[c] for c in range(n_learn) if c in asked]
    learn_V_final = float(np.mean(asked_learn_V)) if asked_learn_V else 0.0
    value_sep = learn_V_final - noisy_V_final
    # striosome learned-value evidence (the actor-critic organ): learned V read vs the never-asked (fresh) init
    strio_learn = float(np.mean([read_value(c) for c in range(n_learn) if c in asked]) if asked_learn_V else 0.0)
    strio_v0 = v0
    noisy_asks_total = sum(1 for e in ask_events if e[4])
    mean_LP_learn = float(np.mean([e[3] for e in ask_events if not e[4]])) if any(not e[4] for e in ask_events) else 0.0
    mean_LP_noisy = float(np.mean([e[3] for e in ask_events if e[4]])) if noisy_asks_total else 0.0
    snc_learn_hz = float(np.mean(snc_learn_burst)) if snc_learn_burst else 0.0
    snc_noisy_hz = float(np.mean(snc_noisy_burst)) if snc_noisy_burst else 0.0

    confident_set = {c for c in concepts if conf_after[c] > 0.5}
    moat_ok = confident_set.issubset(asked)
    learnable_mastered = int(sum(1 for c in range(n_learn) if conf_after[c] > 0.5))

    return {
        "mode": mode, "seed": seed, "v0": v0, "spiking_veto": bool(spiking_veto),
        "value_thresh": (strio_veto_floor if spiking_veto else VALUE_THRESH),
        "corr_gap_want": corr, "rate_unknown": rate_unknown, "rate_known": rate_known, "ratio_b": ratio_b,
        "conf_rise": conf_rise, "conf_after_mean": conf_after_mean, "abstain_floor": abstain_floor,
        "total_asks": len(ask_events), "noisy_asks_total": noisy_asks_total,
        "noisy_early_rate": noisy_early_rate, "noisy_late_rate": noisy_late_rate,
        "noisy_g_final": noisy_g_final, "noisy_ELP_final": noisy_V_final, "noisy_vetoed": noisy_vetoed,
        "noisy_vetoed_frac": noisy_vetoed_frac,
        "learn_ELP_final": learn_V_final, "value_sep": value_sep,
        "strio_learn": strio_learn, "strio_v0": strio_v0,
        "late_learnable_frac": late_learnable_frac, "learnable_mastered": learnable_mastered,
        "mean_LP_learn": mean_LP_learn, "mean_LP_noisy": mean_LP_noisy,
        "snc_learn_hz": snc_learn_hz, "snc_noisy_hz": snc_noisy_hz, "moat_ok": bool(moat_ok),
    }


def evaluate(seed, *, spiking_veto=False, **kw):
    real = run(seed, "real", spiking_veto=spiking_veto, **kw)
    lesion = run(seed, "lesion", spiking_veto=spiking_veto, **kw)
    yoked = run(seed, "yoked", spiking_veto=spiking_veto, **kw)
    permuted = run(seed, "permuted", spiking_veto=spiking_veto, **kw)

    gate_a = real["corr_gap_want"] >= 0.9
    gate_b = real["ratio_b"] >= 2.0
    gate_c = (real["conf_rise"] > 0.3) and (real["conf_after_mean"] > real["abstain_floor"] + 0.3)
    noisy_stops = ((real["noisy_late_rate"] <= 0.5 * real["noisy_early_rate"] + 1e-9)
                   and real["noisy_g_final"] > 0.7 and real["noisy_vetoed"])
    lesion_collapses = lesion["total_asks"] <= 1 and lesion["conf_rise"] < 0.15
    yoked_collapses = yoked["learnable_mastered"] < real["learnable_mastered"]
    permuted_collapses = (permuted["corr_gap_want"] < 0.5
                          or permuted["learnable_mastered"] < real["learnable_mastered"])

    go = bool(gate_a and gate_b and gate_c and noisy_stops and real["moat_ok"]
              and lesion_collapses and yoked_collapses and permuted_collapses)
    out = {
        "seed": seed, "real": real, "lesion": lesion, "yoked": yoked, "permuted": permuted,
        "gate_a_corr": bool(gate_a), "gate_b_askratio": bool(gate_b), "gate_c_conf_rise": bool(gate_c),
        "noisy_stops_honest": bool(noisy_stops), "moat_ok": bool(real["moat_ok"]),
        "lesion_collapses": bool(lesion_collapses), "yoked_collapses": bool(yoked_collapses),
        "permuted_collapses": bool(permuted_collapses), "spiking_veto": bool(spiking_veto),
    }
    if spiking_veto:
        # THE NEW DISSOCIATION for the spiking-veto conversion: with the striosome critic LESIONED the spiking
        # veto must COLLAPSE (noisy no longer vetoed -> the striosome is now load-bearing for the decision). This
        # is the exact opposite of the host-ELP veto, which SURVIVED the same lesion (2026-07-31 finding).
        critic = run(seed, "critic_lesion", spiking_veto=spiking_veto, **kw)
        critic_lesion_collapses_veto = bool(not critic["noisy_vetoed"])
        go = bool(go and critic_lesion_collapses_veto)
        out["critic_lesion"] = critic
        out["critic_lesion_collapses_veto"] = critic_lesion_collapses_veto
    out["GO"] = go
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke (5 concepts, short budget)")
    ap.add_argument("--critic-lesion", action="store_true",
                    help="report-only: run the real config with the GABA_B critic lesioned (noisy not vetoed)")
    ap.add_argument("--spiking-veto", action="store_true",
                    help="convert the noisy-veto from the host ELP tracker to the SPIKING striosome value read; "
                         "adds the critic-lesion-COLLAPSES-veto dissociation to the GO (2026-08-01, additive)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = ("research/findings/raw/_curiosity_seek_learn_onbridge_spikingveto.json" if a.spiking_veto
                 else "research/findings/raw/_curiosity_seek_learn_onbridge.json")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    kw = {}
    if a.smoke:
        kw = dict(n_learn=3, n_noisy=2, n_turns=90, ask_budget=14, d=512)

    from sim.backend import get_backend
    _, backend = get_backend()
    print(f"[DR-1 ON-BRIDGE curiosity] backend={backend} smoke={a.smoke} spiking_veto={a.spiking_veto}  "
          f"fill from_novelty -> spiking ASK pool, spiking-SNc RPE value critic on the LEARNING-PROGRESS reward,"
          f" real Bogacz-Brown gate.\n"
          f"  GO: (a) corr(gap,SPIKING-want)>=0.9  (b) ask unknown>=2x known  (c) conf rises;"
          f" noisy STOPS (veto) while g stays HIGH; lesion/yoked/permuted collapse; moat holds"
          f"{'; + critic-lesion COLLAPSES the veto (striosome now load-bearing)' if a.spiking_veto else ''}.\n",
          flush=True)

    if a.critic_lesion:
        r = run(a.seeds[0], "critic_lesion", verbose=True, spiking_veto=a.spiking_veto, **kw)
        print(f"  [critic-lesion seed {a.seeds[0]}] noisy late-rate {r['noisy_late_rate']:.2f} (vs early "
              f"{r['noisy_early_rate']:.2f}) | noisy ELP {r['noisy_ELP_final']:.2f} vetoed={r['noisy_vetoed']} "
              f"(thr {r['value_thresh']}) -> the veto should FAIL to fire without the critic", flush=True)
        return

    results = []
    for seed in a.seeds:
        r = evaluate(seed, spiking_veto=a.spiking_veto, **kw)
        results.append(r)
        re = r["real"]
        print(f"  [seed {seed}] corr(gap,want) {re['corr_gap_want']:+.3f} | ask-ratio unk/known {re['ratio_b']:.2f} | "
              f"conf-rise {re['conf_rise']:+.2f} (after {re['conf_after_mean']:.2f} vs floor {re['abstain_floor']:.2f})",
              flush=True)
        print(f"            SNc RPE: learn-burst {re['snc_learn_hz']:.1f}Hz vs noisy {re['snc_noisy_hz']:.1f}Hz | "
              f"LP learn {re['mean_LP_learn']:+.3f} vs noisy {re['mean_LP_noisy']:+.3f}", flush=True)
        print(f"            NOISY asks early {re['noisy_early_rate']:.2f} -> late {re['noisy_late_rate']:.2f} "
              f"(g stays {re['noisy_g_final']:.2f}; ELP {re['noisy_ELP_final']:.2f}<=thr {re['value_thresh']}: "
              f"{re['noisy_vetoed']}); value-sep {re['value_sep']:+.2f} | strio V {re['strio_v0']:.0f}->{re['strio_learn']:.0f}",
              flush=True)
        print(f"            controls: lesion asks={r['lesion']['total_asks']} | yoked mastered "
              f"{r['yoked']['learnable_mastered']} vs real {re['learnable_mastered']} | permuted corr "
              f"{r['permuted']['corr_gap_want']:+.2f} (mastered {r['permuted']['learnable_mastered']}) | "
              f"moat {r['moat_ok']}", flush=True)
        flags = (f"a={r['gate_a_corr']} b={r['gate_b_askratio']} c={r['gate_c_conf_rise']} "
                 f"noisy-stops={r['noisy_stops_honest']} lesion={r['lesion_collapses']} "
                 f"yoked={r['yoked_collapses']} permuted={r['permuted_collapses']}")
        if a.spiking_veto:
            cl = r["critic_lesion"]
            flags += f" critic-lesion-collapses={r['critic_lesion_collapses_veto']}"
            print(f"            SPIKING-VETO: real noisy V {re['noisy_ELP_final']:.1f}<=floor {re['value_thresh']:.1f}"
                  f" vetoed={re['noisy_vetoed']} (learn V {re['learn_ELP_final']:.1f}) | critic-lesion noisy V "
                  f"{cl['noisy_ELP_final']:.1f} vetoed={cl['noisy_vetoed']} -> collapse="
                  f"{r['critic_lesion_collapses_veto']}", flush=True)
        print(f"            [{flags}]  ==>  {'GO' if r['GO'] else 'NO'}\n", flush=True)

    n_go = sum(r["GO"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "backend": backend, "smoke": a.smoke, "kw": kw}, fh, indent=2, default=str)
    print(f"{'='*104}", flush=True)
    print(f"  ON-BRIDGE DR-1: {n_go}/{len(results)} seeds GO "
          f"({'ALL GO' if n_go == len(results) else 'partial/negative — pins the exact spiking wall'})", flush=True)
    print(f"  [saved] {a.out}\n{'='*104}", flush=True)


if __name__ == "__main__":
    main()
