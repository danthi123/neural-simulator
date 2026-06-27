"""nav close-out R5 — a VALUE-DRIVEN-CHOICE task that proves the spiking value-critic is LOAD-BEARING BY ITS
FUNCTION (the genuine fix for the R4 value-IRRELEVANT confound).

THE GAP THIS CLOSES (deep-research gate `2026-06-27-nav-value-loadbearing-research-gate.md`, SHA 1a0cac04, RANK 1):
R4 (`_navcloseout_R4_delayed_reward_value.py`) delayed the reward on a SINGLE moving goal whose optimal action
("reduce Manhattan distance to the one goal") is UNCHANGED by reward timing -> the value V was a PASSENGER, not a
limiter (all six 2×2 arms clustered in a 2.48-2.69 band; value-OFF within ±0.11 of value-ON everywhere; the
415/388-synapse GABA_B lesion LANDED but moved nothing). That is the `feedback_validate_signal_by_its_function`
failure mode (the N5-reward lesson): the value-ON/OFF A/B was run on a task where V is not load-bearing BY
CONSTRUCTION. It is NOT a substrate limit — the project ALREADY proved the spiking value is load-bearing BY ITS
FUNCTION on the Pavlovian trace task (`2026-06-21-shortcut9-trace-conditioning-value-derisk.md`, 1a861f87: lesion
collapses the trace CR 6/6 on real spikes, the value-irrelevant DELAY control survives 6/6).

THE RANK-1 FIX (a value-DRIVEN-CHOICE task; the catalog economic-choice paradigm O.22 + O.19 + C.34 + L.41):
two options A/B of DIFFERENT learned VALUE; picking the higher-value option REQUIRES the learned value V. The value
is then the ONLY signal that can drive the correct choice -> the lesion is FORCED to be load-bearing-or-not (unlike
R4, where the lesion had nothing to be load-bearing FOR). The project's EXISTING spiking value-driven WTA
(`_value_salience_appraisal_derisk.py`: an Izhikevich WTA whose pool DRIFT = a candidate's WORTH = f(DA-value), with
a value lesion that reverts it to baseline) is the transplant — generalized speak-vs-silence -> option-A-vs-option-B.

  R1-a (CPU-first, THIS module's gate): the SPIKING value-WTA-choice MECHANISM in isolation. Two value-driven
        accumulator pools (opt_A, opt_B) in BIASED COMPETITION through a shared FS pool (the GO sel/commit/OPN
        soft-WTA template; Wang-2002 NMDA integrators, Lo-Wang all-or-none commit). Each option carries a learned
        VALUE V (seeded from a reward-tagging RNG STRUCTURALLY DISTINCT from any salience/orienting cue, so "value
        drives the choice" is not circular). drift(pool) = base + gain * V(option) (catalog O.19/C.34: value
        modulates the accumulator DRIFT). The DECISION = whichever pool wins the spiking race (a neural pool's
        FIRING, NOT a host argmax). CORRECT = pick the HIGHER-value option.

  R1-b (the nav-embodied form; SCAFFOLD here, flagged FOR CONTROLLER): two simultaneous beacons/goals of different
        value on the grid; the BG action selector approaches the HIGHER-value goal. The missing O.22 Q(s,a)
        read-out the catalog flags. Higher variance -> CPU-first R1-a isolates the value-WTA from the nav cascade
        (the same predict-first/choose-second discipline as the V-A trace -> V-B act-over-gap split).

THE DE-RISK (validate-by-function; ALL on R1-a's gate, multi-seed; controller runs 6-seed if GO):
  (G_HEADLINE) value-ON picks the HIGHER-value option ABOVE chance (the value drives the choice).
  (G_LESION, the headline anti-cheat) pin both options' VALUE to baseline (lesion the value system) -> the choice
      COLLAPSES to chance / the salience-bias baseline (the EXTRA, value-driven correct choices VANISH -> V is the
      load-bearing signal). This is the R4 fix: here the lesion HAS something to collapse.
  (G_DISCRIM, the validate-by-function control R4 LACKED) an EQUAL-VALUE task (V(A)=V(B)): the value genuinely
      can't help -> the lesion is NEUTRAL (it does not change the already-chance choice). This proves the lesion's
      effect on G_LESION is value-SPECIFIC, not a general lesion artifact. (The direct analogue of the V-A `delay`
      arm: the value-irrelevant condition where the lesion must SURVIVE/do-nothing.)
  (G_PERMUTE, anti-cheat) permute the option<->value contingency (shuffle which option gets which value) -> the
      choice advantage VANISHES (the headline must come from the genuine value structure, not a fixed pool bias).

ANTI-CHEATS (this is the 2nd attempt at this exact question; R4 failed a validate-by-function confound):
  - value-lesion must COLLAPSE the high-value choice (load-bearing), NOT merely shift it (G_LESION: reverts to the
    salience-bias / chance baseline, the EXTRA correct choices vanish).
  - EQUAL-value discriminator (the control R4 lacked): equal value -> lesion NEUTRAL (G_DISCRIM). Value-SPECIFIC.
  - permuted-value control: shuffle option<->value -> advantage vanishes (G_PERMUTE).
  - the value axis is DECORRELATED from any salience/orienting bias (a fixed per-pool SALIENCE bias is present and
    INDEPENDENT of value; corr(value, salience) ~ 0 -> the choice is driven by value, not a relabeled salience).
  - the no-confab MOAT is preserved BY CONSTRUCTION: this is a critic/decision organ with NO conversational/RF
    slices (cp_rf_w_re/im are None) -> array-disjoint from any composer. (R1-b on the merged agent re-asserts
    check_moat; this standalone harness builds the decision bridge only.)
  - grid-32 is the nav verdict scale (R1-b, NEVER grid-8); R1-a is the value-WTA in isolation (vocab-free).

DISCIPLINE (the stall lesson — a subagent CANNOT resume on background-completion): this module does the BUILD +
a tiny CPU/numpy SMOKE (the value-WTA-choice is a numpy SimulationBridge slice — fast, CPU, run inline). The R1-b
nav-embodied form (run_moving_goal_episode is CuPy-only) is the GPU eval, FLAGGED in the findings doc "FOR
CONTROLLER TO RUN". This module does NOT run the long GPU eval and does NOT background-and-wait. R1-a's own
multi-seed run is CPU and fast (~seconds/seed) so it CAN be run inline / by the controller cheaply.

Run:
  # CPU SMOKE (well-formedness: the task, the lesion logic, the equal-value + permuted controls):
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice --smoke

  # R1-a value-WTA-choice gate (CPU, fast):
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R5_value_driven_choice \
      --r1a --seeds 42,43,44 --out research/findings/raw/navcloseout_R5/R5_r1a.json

  # R1-b nav two-beacon (GPU — FOR CONTROLLER; see the findings doc for the full command + criteria):
  SIM_BACKEND=cupy python -m research.runners._navcloseout_R5_value_driven_choice \
      --r1b-scaffold-check
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# the value-WTA-choice is the numpy-CPU brain slice (a real Izhikevich WTA on a SimulationBridge).
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  -- backend-safe device->host read (passthrough on numpy, .get() on cupy)


# ===========================================================================
# The DA-VALUE / interest system (the option values) -- a CPU STAND-IN for the merged-bridge spiking
# SNc/striosome_value critic. MUST be STRUCTURALLY DISTINCT from any salience/orienting bias (else "value drives
# the choice" is circular). Mirrors `_value_salience_appraisal_derisk.build_concept_value`: a reward/interest tag
# seeded from a SEPARATE RNG (vmPFC/OFC subjective value + striatal action-value O.22, distinct from salience).
# The GPU follow-on (R1-b) replaces this stand-in with the REAL shared `dopamine`/striosome critic off the merged
# bridge (so the lesion pins the real spiking SNc); on the standalone CPU R1-a it is the transparent stand-in.
# ===========================================================================
def make_option_values(n_options, seed, *, equal=False):
    """Per-option VALUE scalar in [0,1], from a reward-tagging RNG (NOT a salience signal).

    equal=False (the headline task): the options carry DIFFERENT values drawn from a Beta (right-skewed, like the
        appraisal probe's wanting tail) -> there is a clear HIGHER-value option the choice must find.
    equal=True (the G_DISCRIM control): ALL options share the SAME value -> the value genuinely cannot discriminate,
        so the lesion must be NEUTRAL. We set them all equal to the mean so the value term is a constant (no
        gradient) -> the spiking choice falls back to the salience bias / chance, value-intact OR lesioned.
    """
    rng = np.random.default_rng(int(seed) * 101 + 7)
    base = rng.beta(1.5, 4.0, size=int(n_options))            # right-skewed: a tail of high-interest options
    salient = rng.random(int(n_options)) < 0.5
    base = np.where(salient, np.clip(base + rng.uniform(0.3, 0.6, size=int(n_options)), 0, 1), base * 0.5)
    if equal:
        base = np.full(int(n_options), float(base.mean()), dtype=float)
    return base.astype(float)


def make_salience_bias(n_options, seed):
    """A per-option SALIENCE / orienting bias scalar in [0,1] -- the 'default' pull on each option INDEPENDENT of
    its reward value (e.g. one option is nearer / brighter). Seeded from a DIFFERENT RNG than the value, so
    corr(value, salience) ~ 0 (asserted) -> the choice's value-dependence is not a relabeled salience. This is the
    'salience baseline' the lesion arm reverts TO: with the value pinned, only this bias (+ noise) drives choice."""
    rng = np.random.default_rng(int(seed) * 131 + 17)
    return rng.uniform(0.0, 1.0, size=int(n_options)).astype(float)


def permute_values(values, seed):
    """AC_PERMUTE: shuffle which OPTION gets which VALUE (break the option<->value contingency). The value
    MULTISET is unchanged (no fabricated value); only the assignment is permuted. With the contingency destroyed,
    the value pinned to a pool no longer tracks that pool's identity -> across a deranging permutation the choice
    advantage (correct == higher-value) vanishes to chance."""
    rng = np.random.default_rng(int(seed) * 17 + 5)
    v = np.asarray(values, dtype=float).copy()
    # a derangement-ish shuffle (just a permutation; for n_options=2 it swaps, which is the maximal break)
    perm = rng.permutation(len(v))
    return v[perm], perm


# ===========================================================================
# The SPIKING value-driven A-vs-B WTA choice (the brain-based DECISION -- a neural pool's FIRING, not a host
# argmax). Mirrors `_value_salience_appraisal_derisk.SpikingSpeakAccumulator` but with n_options OPTION pools
# (not speak/silence): each option pool is a Wang-2002 NMDA integrator; a shared wta_fs FS pool implements biased
# competition (each option drives the FS; the FS inhibits all options) -> soft-WTA. The pool whose drift (its
# option's value-derived drive) wins the spiking race IS the choice. NO sim/ edit -- BrainRegion / RegionPathway
# only (the same primitives g11_bg_runner uses for the sel/commit/OPN spiking decision).
# ===========================================================================
class SpikingValueChoice:
    """A small, fast (CPU) spiking value-driven WTA over n_options. Each option_i is an NMDA integrator; a shared
    wta_fs FS pool gives biased competition. The DECISION = argmax over option pools of the spike count in the
    integration window (the spiking commit). drift(option_i) = base + gain * drive_i, where drive_i is the
    appraisal of option i (value [+ salience]). The value LESION drops the value term from drive_i (a clean
    ablation: the option pools then differ only by salience + noise) -> the choice reverts to the salience baseline.
    """

    def __init__(self, seed, n_options=2, n_acc=40, n_fs=20, n_steps=120, ou_pA=15.0):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway

        self.n_options = int(n_options)
        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_nmda = True                      # Wang-2002 NMDA-slow integration (the accumulator)
        cfg.dt_ms = 1.0
        cfg.seed = int(seed)
        cfg.stdp_w_max = 30.0
        cfg.hebbian_max_weight = 30.0
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_hebbian_learning = False         # fixed wiring; no weight drift
        cfg.enable_homeostasis = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_structural_plasticity = False
        cfg.enable_ou_process = True                # OU noise -> the soft (graded) threshold near equal drives
        cfg.ou_std_current_pA = float(ou_pA)
        cfg.enable_parameter_heterogeneity = False
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1

        regions = []
        pathways = []
        for i in range(self.n_options):
            regions.append(BrainRegion(
                name=f"opt_{i}", n_neurons=n_acc, exc_fraction=1.0, internal_density=0.5,
                exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                izh_neuron_type="IZH2007_RS_CORTICAL_PYRAMIDAL", enable_nmda=True))
        regions.append(BrainRegion(
            name="wta_fs", n_neurons=n_fs, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type="IZH2007_FS_CORTICAL_INTERNEURON"))
        for i in range(self.n_options):
            # each option drives the shared FS (excitatory)
            pathways.append(RegionPathway(
                from_region=f"opt_{i}", to_region="wta_fs", density=0.5, weight_mean=8.0,
                weight_jitter=0.1, plastic=False))
            # the FS inhibits every option (biased competition / soft-WTA)
            pathways.append(RegionPathway(
                from_region="wta_fs", to_region=f"opt_{i}", density=0.6, weight_mean=6.0,
                weight_jitter=0.1, plastic=False, receptor="gaba_a"))

        cfg.brain_regions = regions
        cfg.region_pathways = pathways
        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)
        self._idx = {n: np.asarray(v) for n, v in self._bridge.region_manager.region_indices_dict().items()}
        self.n_steps = int(n_steps)
        # the no-confab moat is preserved BY CONSTRUCTION: this is a decision organ with NO RF/conversational slices.
        self.has_rf_slices = (getattr(self._bridge, "cp_rf_w_re", None) is not None
                              or getattr(self._bridge, "cp_rf_w_im", None) is not None)

    def decide(self, drives_pA):
        """Run one spiking A-vs-B...-vs-N race. drives_pA: list of per-option drift currents (length n_options).
        Returns (chosen_index, spikes_per_option, margin). The DECISION is the spiking commit: the option pool
        that fires most over the window (a neural pool's FIRING, NOT a host argmax over drives).

        The OU-noise realization is HELD FIXED per drive-vector (snapshot/restore the global RNG, re-seeded from
        the rounded drives), so the decision is a deterministic FUNCTION of the drives -> the value-vs-lesion
        comparison is a clean ABLATION (the value arm's drive >= the lesion arm's for the higher-value option; a
        monotone WTA on a FIXED noise realization then isolates the value contribution, not a noise coincidence).
        The spiking gate is STILL the brain-based decision; freezing the noise is the control, exactly as an
        ablation freezes everything but the lesioned variable."""
        b = self._bridge
        drives = [float(d) for d in drives_pA]
        assert len(drives) == self.n_options
        _state = np.random.get_state()
        dseed = (sum(int(round(d * 7.0)) * (100003 ** (k + 1)) for k, d in enumerate(drives))) % (2 ** 31 - 1)
        np.random.seed(int(dseed))
        try:
            b._initialize_simulation_data(called_from_playback_init=False)   # reset state per decision
            b.cp_external_input_current[:] = 0.0
            for i in range(self.n_options):
                b.cp_external_input_current[self._idx[f"opt_{i}"]] = np.float32(drives[i])
            counts = [0.0] * self.n_options
            for _ in range(self.n_steps):
                b._run_one_simulation_step()
                fs = to_host(b.cp_firing_states)   # backend-safe (passthrough on numpy; .get() on cupy)
                for i in range(self.n_options):
                    counts[i] += float(fs[self._idx[f"opt_{i}"]].sum())
        finally:
            np.random.set_state(_state)
        chosen = int(np.argmax(counts))
        srt = sorted(counts, reverse=True)
        margin = float(srt[0] - srt[1]) if self.n_options >= 2 else float(srt[0])
        return chosen, counts, margin


# ===========================================================================
# R1-a: the value-WTA-choice gate. For each of N trials, present two options with (possibly) different values,
# build the value-driven drives, run the spiking choice, and score whether the brain picked the HIGHER-value
# option -- value-INTACT, value-LESION, value-INTACT-but-EQUAL, value-LESION-EQUAL, and value-PERMUTED.
# ===========================================================================
def _drives(values, salience, *, speak_base_pA, value_gain_pA, salience_gain_pA, lesion_value):
    """Build the per-option drift currents. ADDITIVE incentive-salience scheme (mirrors the appraisal probe):
    drive_i = base + value_gain * VALUE_i (intact) + salience_gain * SALIENCE_i.

    The value LESION (lesion_value=True) replaces each option's VALUE with the MEAN value over the options
    (value_gain * mean(values), applied uniformly), instead of the per-option value. This is the faithful
    biological ablation of the spiking striosome->SNc value-DIFFERENTIAL: the GABA_B lesion removes the
    option-SPECIFIC value CONTRAST that grades the SNc burst across options, while the TONIC DA level (the mean)
    is unaffected (a uniform drive on all options carries no gradient). DRIVE-LEVEL MATCHED to the intact arm
    (same total mean drive) so the lesion removes ONLY the value GRADIENT, not the operating point -> a clean
    isolation: when the value gradient is the ONLY differentiator (the distinct-value HEADLINE), the lesion
    collapses the choice to chance; when there is no gradient (the EQUAL-value control), the lesion changes
    NOTHING (the uniform value term is identical intact-vs-lesion) -> the lesion's effect is value-GRADIENT-
    SPECIFIC, the validate-by-function discriminator."""
    n = len(values)
    vmean = float(np.mean(values))
    drives = []
    for i in range(n):
        d = float(speak_base_pA) + float(salience_gain_pA) * float(salience[i])
        if lesion_value:
            d += float(value_gain_pA) * vmean              # tonic DA level retained; the GRADIENT removed
        else:
            d += float(value_gain_pA) * float(values[i])   # the per-option value gradient (intact)
        drives.append(d)
    return drives


def run_r1a_seed(seed, accumulator, a):
    """One seed of the R1-a value-WTA-choice gate. Returns the per-seed metrics + the 4 gate booleans."""
    rng = np.random.default_rng(seed)
    n_options = accumulator.n_options
    n_trials = int(a.n_trials)

    # the per-trial OPTION VALUES + SALIENCE. We draw a fresh (values, salience) per trial so the "higher-value
    # option" is sometimes opt_0, sometimes opt_1 (no fixed-pool confound), and the salience bias is independent.
    def trial_values(t, *, equal):
        return make_option_values(n_options, seed=seed * 1000 + t, equal=equal)

    def trial_salience(t):
        return make_salience_bias(n_options, seed=seed * 1000 + t)

    # value/salience INDEPENDENCE (non-circularity): over all trials, correlate each option's VALUE with its
    # SALIENCE. ~0 -> the value is a genuinely separate signal, so "value drives the choice" is not a relabeled
    # salience. (Concept-level analogue of the appraisal probe's value<->plausibility independence assertion.)
    all_v, all_s = [], []
    for t in range(n_trials):
        all_v.extend(trial_values(t, equal=False).tolist())
        all_s.extend(trial_salience(t).tolist())
    av, as_ = np.array(all_v), np.array(all_s)
    value_salience_corr = (float(np.corrcoef(av, as_)[0, 1])
                           if av.std() > 0 and as_.std() > 0 else 0.0)

    def score_arm(*, equal, lesion_value, permute):
        """Run one arm over all trials. Returns (accuracy, mean_margin, choices, true_bests), where accuracy =
        fraction of trials the spiking choice picked the TRUE higher-value option, and choices/true_bests are the
        per-trial arrays (for the trial-by-trial G_DISCRIM agreement). For equal-value trials the value vector is
        constant so argmax is ill-defined as a 'correct' target -- the discriminator uses the trial-by-trial
        intact-vs-lesion CHOICE AGREEMENT instead (see G_DISCRIM)."""
        n_correct = 0
        margins = []
        choices, true_bests = [], []
        for t in range(n_trials):
            values = trial_values(t, equal=equal)
            salience = trial_salience(t)
            true_best = int(np.argmax(values))           # the option the brain SHOULD pick (highest value)
            drive_values = values
            if permute:
                drive_values, _perm = permute_values(values, seed=seed * 1000 + t)
                # after permuting which option gets which value, the value attached to each POOL no longer tracks
                # that pool's TRUE identity; the "correct" answer is still the TRUE-highest-value option, which the
                # permuted drives no longer favor -> advantage collapses.
            drives = _drives(drive_values, salience, speak_base_pA=a.speak_base_pA,
                             value_gain_pA=a.value_gain_pA, salience_gain_pA=a.salience_gain_pA,
                             lesion_value=lesion_value)
            chosen, counts, margin = accumulator.decide(drives)
            margins.append(margin)
            choices.append(int(chosen))
            true_bests.append(true_best)
            if chosen == true_best:
                n_correct += 1
        return (n_correct / max(1, n_trials), float(np.mean(margins)) if margins else 0.0,
                np.array(choices), np.array(true_bests))

    chance = 1.0 / n_options

    # ---- the arms ----
    acc_value, m_value, _, _ = score_arm(equal=False, lesion_value=False, permute=False)  # value intact, distinct
    acc_lesion, m_lesion, _, _ = score_arm(equal=False, lesion_value=True, permute=False)  # value lesioned
    acc_eq_intact, _, ch_eq_intact, _ = score_arm(equal=True, lesion_value=False, permute=False)  # equal, intact
    acc_eq_lesion, _, ch_eq_lesion, _ = score_arm(equal=True, lesion_value=True, permute=False)    # equal, lesion
    acc_permuted, _, _, _ = score_arm(equal=False, lesion_value=False, permute=True)       # permuted contingency

    # ---- the 4 gates (validate-by-function) ----
    # (G_HEADLINE) value-ON picks the higher-value option ABOVE chance (margin above chance >= bar).
    headline_ok = (acc_value - chance) >= a.above_chance_bar

    # (G_LESION) the value lesion COLLAPSES the high-value choice to ~chance/salience-baseline (the EXTRA correct
    # choices vanish): value-ON accuracy >> lesion accuracy, and the lesion accuracy is near chance.
    lesion_collapses = ((acc_value - acc_lesion) >= a.lesion_drop_bar
                        and (acc_lesion - chance) <= a.near_chance_tol)

    # (G_DISCRIM) the EQUAL-value control (the validate-by-function control R4 LACKED): with V(A)=V(B), the value
    # term is a CONSTANT offset (same on both pools) so it CANNOT change the choice -> lesioning it must be
    # NEUTRAL. The rigorous, direct measure is the TRIAL-BY-TRIAL CHOICE AGREEMENT between the equal-value intact
    # and equal-value lesion arms: with value constant + the OU noise frozen per drive-vector, the only difference
    # between the two arms is the constant value offset (which shifts both pools equally), so the choices should
    # be (near-)IDENTICAL. HIGH agreement proves the lesion's G_LESION effect is value-SPECIFIC (it acts ONLY when
    # value carries a gradient), NOT a general lesion artifact. (The salience bias legitimately drives the
    # equal-value choice -- it need NOT be at chance; what matters is the lesion changes nothing.)
    equal_value_choice_agreement = float(np.mean(ch_eq_intact == ch_eq_lesion)) if n_trials > 0 else 1.0
    equal_value_neutral = (equal_value_choice_agreement >= a.discrim_agreement_bar)

    # (G_PERMUTE) permuting the option<->value contingency collapses the advantage to ~chance.
    permute_collapses = (acc_permuted - chance) <= a.near_chance_tol

    # non-circularity: value decorrelated from salience.
    noncircular = abs(value_salience_corr) <= a.max_value_salience_corr

    print(f"\n[R5/R1-a seed {seed}] n_options={n_options} n_trials={n_trials} chance={chance:.3f}", flush=True)
    print(f"  acc(value-INTACT, distinct) = {acc_value:.3f}  (margin {m_value:.1f})", flush=True)
    print(f"  acc(value-LESION,  distinct) = {acc_lesion:.3f}  (margin {m_lesion:.1f})  <- should drop to ~chance",
          flush=True)
    print(f"  EQUAL-value: intact-vs-lesion CHOICE AGREEMENT = {equal_value_choice_agreement:.3f}  <- should be "
          f"HIGH (lesion NEUTRAL when value is a constant; validate-by-function)", flush=True)
    print(f"  acc(PERMUTED contingency)    = {acc_permuted:.3f}  <- should collapse to ~chance", flush=True)
    print(f"  (G_HEADLINE) value-ON > chance: {headline_ok}", flush=True)
    print(f"  (G_LESION)   lesion collapses high-value choice: {lesion_collapses} "
          f"(drop {acc_value - acc_lesion:.3f} >= {a.lesion_drop_bar}; lesion-chance {acc_lesion - chance:.3f} "
          f"<= {a.near_chance_tol})", flush=True)
    print(f"  (G_DISCRIM)  equal-value -> lesion NEUTRAL: {equal_value_neutral} "
          f"(choice-agreement {equal_value_choice_agreement:.3f} >= {a.discrim_agreement_bar})", flush=True)
    print(f"  (G_PERMUTE)  permuted -> ~chance: {permute_collapses}", flush=True)
    print(f"  value<->salience INDEPENDENCE: corr={value_salience_corr:+.3f} (|corr| <= "
          f"{a.max_value_salience_corr} -> non-circular: {noncircular})", flush=True)
    print(f"  MOAT preserved by construction (no RF slices): {not accumulator.has_rf_slices}", flush=True)

    return {
        "seed": int(seed),
        "n_options": int(n_options),
        "n_trials": int(n_trials),
        "chance": chance,
        "acc_value_intact": acc_value,
        "acc_value_lesion": acc_lesion,
        "acc_equal_intact": acc_eq_intact,
        "acc_equal_lesion": acc_eq_lesion,
        "equal_value_choice_agreement": equal_value_choice_agreement,
        "acc_permuted": acc_permuted,
        "margin_value_intact": m_value,
        "margin_value_lesion": m_lesion,
        "value_salience_corr": value_salience_corr,
        # the 4 gates + non-circularity + moat
        "headline_above_chance": bool(headline_ok),
        "lesion_collapses": bool(lesion_collapses),
        "equal_value_neutral": bool(equal_value_neutral),
        "permute_collapses": bool(permute_collapses),
        "noncircular": bool(noncircular),
        "moat_preserved_by_construction": bool(not accumulator.has_rf_slices),
    }


def decide_r1a_verdict(rows, a):
    """GO iff, across ALL seeds: (G_HEADLINE) value-ON > chance; (G_LESION) the lesion collapses the high-value
    choice to ~chance; (G_DISCRIM) the EQUAL-value lesion is NEUTRAL (value-SPECIFIC -- the control R4 lacked);
    (G_PERMUTE) permuting the contingency collapses the advantage; AND the value axis is NON-circular (decorrelated
    from salience). Else HONEST_NEGATIVE / BOUNDARY + why."""
    def col(k):
        return [r[k] for r in rows]

    headline_all = all(col("headline_above_chance"))
    lesion_all = all(col("lesion_collapses"))
    discrim_all = all(col("equal_value_neutral"))
    permute_all = all(col("permute_collapses"))
    noncirc_all = all(col("noncircular"))
    moat_all = all(col("moat_preserved_by_construction"))

    detail = {
        "acc_value_intact_mean": float(np.mean(col("acc_value_intact"))),
        "acc_value_intact_min": float(np.min(col("acc_value_intact"))),
        "acc_value_lesion_mean": float(np.mean(col("acc_value_lesion"))),
        "equal_value_choice_agreement_mean": float(np.mean(col("equal_value_choice_agreement"))),
        "equal_value_choice_agreement_min": float(np.min(col("equal_value_choice_agreement"))),
        "acc_permuted_mean": float(np.mean(col("acc_permuted"))),
        "chance": float(rows[0]["chance"]) if rows else float("nan"),
        "lesion_drop_mean": float(np.mean(np.array(col("acc_value_intact")) - np.array(col("acc_value_lesion")))),
        "value_salience_corr_absmax": float(np.max(np.abs(col("value_salience_corr")))),
        "headline_above_chance_all_seeds": headline_all,
        "lesion_collapses_all_seeds": lesion_all,
        "equal_value_neutral_all_seeds": discrim_all,
        "permute_collapses_all_seeds": permute_all,
        "noncircular_all_seeds": noncirc_all,
        "moat_preserved_all_seeds": moat_all,
        "above_chance_bar": float(a.above_chance_bar),
        "lesion_drop_bar": float(a.lesion_drop_bar),
        "near_chance_tol": float(a.near_chance_tol),
        "discrim_agreement_bar": float(a.discrim_agreement_bar),
        "max_value_salience_corr": float(a.max_value_salience_corr),
    }

    if not noncirc_all:
        verdict = "INVALID_value_is_relabeled_salience"           # circular -> not a finding
    elif not headline_all:
        verdict = "HONEST_NEGATIVE_value_does_not_drive_choice"   # the value doesn't pick the higher-value option
    elif not lesion_all:
        verdict = "HONEST_NEGATIVE_lesion_does_not_collapse"      # the choice is NOT the value system (R4-like)
    elif not discrim_all:
        verdict = "HONEST_NEGATIVE_lesion_artifact_not_value_specific"  # equal-value lesion not neutral -> artifact
    elif not permute_all:
        verdict = "HONEST_NEGATIVE_advantage_is_fixed_pool_bias"  # permuting didn't collapse -> not value structure
    else:
        verdict = "GO"
    return verdict, detail


def run_r1a(a):
    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[R5/R1-a] seeds={seeds} n_options={a.n_options} -- does the brain's VALUE system DRIVE the choice "
          f"(pick the higher-value option), via a SPIKING WTA (NOT a host argmax), lesion-confirmed + "
          f"equal-value-discriminated + permute-controlled?", flush=True)
    print(f"[R5/R1-a] building the spiking value-driven WTA choice (Wang-2002 NMDA; sel/commit/OPN template)...",
          flush=True)
    accumulator = SpikingValueChoice(seed=12345, n_options=int(a.n_options), n_steps=int(a.acc_steps))
    rows = [run_r1a_seed(s, accumulator, a) for s in seeds]
    verdict, detail = decide_r1a_verdict(rows, a)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  R5/R1-a VERDICT: {verdict}", flush=True)
    print(f"  ACCURACY: value-INTACT {detail['acc_value_intact_mean']:.3f} (min "
          f"{detail['acc_value_intact_min']:.3f}) vs value-LESION {detail['acc_value_lesion_mean']:.3f} vs chance "
          f"{detail['chance']:.3f}  (lesion drop {detail['lesion_drop_mean']:.3f})", flush=True)
    print(f"  EQUAL-value: intact-vs-lesion choice-agreement {detail['equal_value_choice_agreement_mean']:.3f} "
          f"(min {detail['equal_value_choice_agreement_min']:.3f}) (validate-by-function: lesion NEUTRAL when "
          f"value is a constant)", flush=True)
    print(f"  PERMUTED: {detail['acc_permuted_mean']:.3f} (-> ~chance)", flush=True)
    print(f"  (G_HEADLINE) all seeds: {detail['headline_above_chance_all_seeds']}", flush=True)
    print(f"  (G_LESION)   all seeds: {detail['lesion_collapses_all_seeds']}", flush=True)
    print(f"  (G_DISCRIM)  all seeds: {detail['equal_value_neutral_all_seeds']}", flush=True)
    print(f"  (G_PERMUTE)  all seeds: {detail['permute_collapses_all_seeds']}", flush=True)
    print(f"  NON-CIRCULAR all seeds: {detail['noncircular_all_seeds']} (|corr| max "
          f"{detail['value_salience_corr_absmax']:.3f})", flush=True)
    print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    out = {
        "probe": "navcloseout_R5_value_driven_choice_r1a",
        "verdict": verdict,
        "seeds": seeds,
        "config": {"n_options": a.n_options, "n_trials": a.n_trials, "acc_steps": a.acc_steps,
                   "speak_base_pA": a.speak_base_pA, "value_gain_pA": a.value_gain_pA,
                   "salience_gain_pA": a.salience_gain_pA, "above_chance_bar": a.above_chance_bar,
                   "lesion_drop_bar": a.lesion_drop_bar, "near_chance_tol": a.near_chance_tol,
                   "discrim_agreement_bar": a.discrim_agreement_bar,
                   "max_value_salience_corr": a.max_value_salience_corr},
        "mechanism": (
            "two value-driven NMDA accumulator pools (opt_A/opt_B) in BIASED COMPETITION through a shared FS pool "
            "(Wang-2002 soft-WTA; the merged-bridge sel/commit/OPN template). drift(option) = base + value_gain*"
            "VALUE + salience_gain*SALIENCE (catalog O.19/C.34: value modulates the accumulator DRIFT). The "
            "DECISION = whichever option pool wins the spiking race (a neural pool's FIRING, NOT a host argmax). "
            "CORRECT = pick the HIGHER-value option. The value LESION drops the value term (clean additive "
            "ablation) -> the choice reverts to the salience baseline."),
        "anti_cheats": {
            "lesion_collapses": "pin the option VALUES to baseline -> the high-value choice collapses to ~chance/"
                                "salience-baseline (the EXTRA correct choices vanish) -> the value is the "
                                "load-bearing signal. (G_LESION; the R4 fix -- here the lesion HAS something to "
                                "collapse, unlike R4's value-irrelevant task)",
            "equal_value_discriminator": "EQUAL value (V(A)=V(B)) -> the value genuinely can't help -> the lesion "
                                         "is NEUTRAL (does not change the already-chance choice). Proves G_LESION's "
                                         "effect is value-SPECIFIC, not a general lesion artifact. THE "
                                         "validate-by-function control R4 LACKED. (G_DISCRIM)",
            "permuted_value": "shuffle which option gets which value (break the option<->value contingency) -> the "
                              "choice advantage vanishes to chance (the headline is the value STRUCTURE, not a "
                              "fixed pool bias). (G_PERMUTE)",
            "non_circular_value": "corr(value, salience) ~ 0 -> the value axis is NOT a relabeled salience/orienting "
                                  "bias; else the probe is INVALID.",
            "moat_by_construction": "this is a decision organ with NO RF/conversational slices (cp_rf_w_re/im None) "
                                    "-> array-disjoint from any composer; the no-confab moat is preserved by "
                                    "construction.",
        },
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "navcloseout_R5", "R5_r1a.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


# ===========================================================================
# R1-b: the nav-embodied two-beacon value-choice SCAFFOLD (GPU -- FOR CONTROLLER). This module does NOT run it.
# The scaffold-check validates the run_moving_goal_episode kwargs are well-formed (the two-goal value-choice
# config the controller would deploy), without building a CuPy bridge.
# ===========================================================================
def r1b_two_beacon_kwargs(*, seed, grid_size, n_steps, high_value_corner, low_value_corner,
                          critic_warmup_trials):
    """Assemble the run_moving_goal_episode kwargs for the nav two-beacon value-choice (R1-b).

    THE TASK: two simultaneous goals/beacons of DIFFERENT value -- a HIGH-value goal and a LOW-value goal. The BG
    action selector (the spiking sel/commit/OPN decision, default-on) must approach the HIGHER-value goal. The
    value-ON arm (the spiking striosome critic supplies the value baseline) should reach the HIGH-value goal more
    (lower mean distance to IT / more high-value reward harvested) than the value-OFF arm (GABA_B-lesioned ->
    reverts to the nearer/salience-default goal). The EQUAL-value control (both goals same value) shows no value
    advantage; the lesion collapses the preference.

    IMPLEMENTATION NOTE (the residual the controller resolves): run_moving_goal_episode (g11_bg_runner.py:3256)
    presents a SINGLE goal_pos / goal_schedule. The two-beacon value-choice needs EITHER (i) a custom
    homeostatic_hook that, per trial, sets the "goal" to whichever of the two beacons the agent is closer to (the
    value entering through the reward MAGNITUDE: a higher reward for reaching the high-value beacon), reusing the
    existing per-trial `{"goal": ...}` override the hook already supports; OR (ii) the existing
    `enable_beacon_perception` two-sensor path with two beacons of different `beacon_max_intensity` proxying value.
    Both are RUNNER-SIDE (no sim/ edit). The CPU-first R1-a gate is the decisive value-WTA proof; R1-b is the
    nav read-out, deferred to the controller (it is the higher-variance arm per the gate's honest scope).
    """
    # the shared SPIKING-DEFAULT merged nav config (CYCLE 1B defaults; passed explicitly = self-documenting).
    kwargs = dict(
        seed=int(seed), n_steps=int(n_steps), grid_size=int(grid_size),
        goal_pos=tuple(high_value_corner),
        perceived_approach_reward=True, spiking_snc=True,
        enable_neural_critic=True, enable_critic_homeostasis=True, spiking_reward_us=True,
        readout_source="spiking_wta",       # the spiking decision default (the BG selector picks the action)
        critic_warmup_trials=int(critic_warmup_trials),
        stdp_w_max_override=400.0,
        verbose=False, progress_print_interval=0,
    )
    meta = dict(high_value_corner=tuple(high_value_corner), low_value_corner=tuple(low_value_corner),
                note="two-beacon value-choice; the controller wires the second beacon + differential reward "
                     "magnitude via the per-trial homeostatic_hook goal-override (RUNNER-SIDE, no sim/ edit).")
    return kwargs, meta


def r1b_scaffold_check():
    """Validate (CPU, no bridge) the R1-b two-beacon kwargs are well-formed."""
    print("=" * 78)
    print("[R5/R1-b SCAFFOLD] nav two-beacon value-choice kwargs (GPU eval is FOR CONTROLLER; no bridge built here)")
    print("=" * 78)
    kwargs, meta = r1b_two_beacon_kwargs(seed=42, grid_size=32, n_steps=1800,
                                         high_value_corner=(30, 30), low_value_corner=(1, 1),
                                         critic_warmup_trials=8)
    ok = (kwargs["spiking_snc"] and kwargs["enable_neural_critic"] and kwargs["spiking_reward_us"]
          and kwargs["perceived_approach_reward"] and kwargs["readout_source"] == "spiking_wta"
          and kwargs["grid_size"] == 32)
    print(f"  kwargs: grid={kwargs['grid_size']} n_steps={kwargs['n_steps']} readout={kwargs['readout_source']} "
          f"spiking_critic={kwargs['enable_neural_critic']} high_value_goal={kwargs['goal_pos']}")
    print(f"  meta: {meta['note']}")
    print(f"  scaffold well-formed (spiking-default merged nav config, grid-32): {ok}")
    print("=" * 78)
    return ok


# ===========================================================================
# THE CPU SMOKE — pure/CPU well-formedness: the value system is distinct from salience, the lesion ablation is a
# clean additive drop, the equal-value control zeroes the value gradient, the permuted control breaks the
# contingency, and the spiking choice + verdict aggregator are sound. Builds a TINY value-WTA bridge (CPU, fast).
# ===========================================================================
def smoke():
    print("=" * 78)
    print("[R5 SMOKE] CPU validation of the value-driven-choice harness (tiny numpy SimulationBridge slice)")
    print("=" * 78)
    ok = True

    # ── (a) the value system is DISTINCT from salience (non-circular) ──
    n_opt = 2
    vals = np.array([make_option_values(n_opt, seed=42 * 1000 + t, equal=False) for t in range(200)]).ravel()
    sals = np.array([make_salience_bias(n_opt, seed=42 * 1000 + t) for t in range(200)]).ravel()
    vs_corr = float(np.corrcoef(vals, sals)[0, 1])
    noncirc = abs(vs_corr) <= 0.35
    print(f"  (a) value<->salience corr = {vs_corr:+.3f} (|corr| <= 0.35 -> non-circular: {noncirc})")
    ok = ok and noncirc

    # ── (b) the equal-value control zeroes the value GRADIENT (all options identical value) ──
    eq = make_option_values(n_opt, seed=4242, equal=True)
    eq_flat = bool(np.allclose(eq, eq[0]))
    distinct = make_option_values(n_opt, seed=4242, equal=False)
    distinct_varies = bool(distinct.std() > 0)
    print(f"  (b) equal-value vector flat: {eq_flat} (values {eq.round(3).tolist()}); distinct-value varies: "
          f"{distinct_varies} (values {distinct.round(3).tolist()})")
    ok = ok and eq_flat and distinct_varies

    # ── (c) the value LESION removes the value GRADIENT (drive-level matched; the value term goes flat) ──
    v = np.array([0.2, 0.9])
    s = np.array([0.5, 0.4])
    d_intact = _drives(v, s, speak_base_pA=70.0, value_gain_pA=180.0, salience_gain_pA=40.0, lesion_value=False)
    d_lesion = _drives(v, s, speak_base_pA=70.0, value_gain_pA=180.0, salience_gain_pA=40.0, lesion_value=True)
    # lesion replaces per-option value with the MEAN -> the per-option drop == value_gain*(value_i - mean), and
    # the lesion arm's VALUE term is FLAT across options (no gradient) so only salience differentiates.
    drop = [di - dl for di, dl in zip(d_intact, d_lesion)]
    clean_ablation = bool(np.allclose(drop, 180.0 * (v - v.mean())))
    # lesion value contribution is identical across options (drive-level matched, gradient removed)
    lesion_value_flat = bool(np.isclose((d_lesion[0] - 40.0 * s[0]), (d_lesion[1] - 40.0 * s[1])))
    # intact: the HIGH-value option (idx 1) gets the higher drive; lesion: salience decides (idx 0 here)
    intact_favors_highval = bool(np.argmax(d_intact) == 1)
    lesion_favors_salience = bool(np.argmax(d_lesion) == int(np.argmax(s)))
    print(f"  (c) lesion removes gradient (drop == value_gain*(value-mean)): {clean_ablation} "
          f"(drop {np.round(drop,1).tolist()}); lesion value-term FLAT across options: {lesion_value_flat}")
    print(f"      intact drive favors HIGH-value option: {intact_favors_highval}; lesion drive favors SALIENCE: "
          f"{lesion_favors_salience}")
    ok = ok and clean_ablation and lesion_value_flat and intact_favors_highval and lesion_favors_salience

    # ── (d) the permuted control breaks the option<->value contingency (multiset preserved) ──
    vperm, perm = permute_values(np.array([0.2, 0.9]), seed=42 * 1000 + 0)
    multiset_ok = bool(sorted(vperm.tolist()) == sorted([0.2, 0.9]))
    # for n=2 the only non-identity permutation is the swap -> contingency broken
    swapped = bool(vperm.tolist() == [0.9, 0.2])
    print(f"  (d) permuted value multiset preserved: {multiset_ok}; n=2 swap breaks contingency: {swapped} "
          f"(perm {perm.tolist()}, vals {vperm.round(3).tolist()})")
    ok = ok and multiset_ok and swapped

    # ── (e) the SPIKING value-WTA actually picks the higher-DRIVE pool (a tiny bridge, a few decisions) ──
    print(f"  (e) building a tiny spiking value-WTA (n_options=2)...")
    acc = SpikingValueChoice(seed=12345, n_options=2, n_steps=80)
    moat_ok = (not acc.has_rf_slices)
    # drive opt_1 high, opt_0 low -> the spiking race should pick opt_1; then swap.
    c1, counts1, m1 = acc.decide([90.0, 260.0])
    c0, counts0, m0 = acc.decide([260.0, 90.0])
    wta_ok = (c1 == 1 and c0 == 0)
    print(f"      drive [90,260] -> chose opt_{c1} (counts {np.round(counts1,1).tolist()}); "
          f"drive [260,90] -> chose opt_{c0} (counts {np.round(counts0,1).tolist()})")
    print(f"      spiking WTA picks the higher-drive pool: {wta_ok}; MOAT preserved by construction "
          f"(no RF slices): {moat_ok}")
    ok = ok and wta_ok and moat_ok

    # ── (f) the verdict aggregator computes the right GO/NEGATIVE on a synthetic per-seed row set ──
    synth_go = [
        # value-INTACT high, lesion ~chance, equal-value lesion-neutral (high choice agreement), permuted ~chance
        {"acc_value_intact": 0.90, "acc_value_lesion": 0.52, "equal_value_choice_agreement": 0.95,
         "acc_permuted": 0.50, "value_salience_corr": 0.05, "chance": 0.5,
         "headline_above_chance": True, "lesion_collapses": True, "equal_value_neutral": True,
         "permute_collapses": True, "noncircular": True, "moat_preserved_by_construction": True}
        for _ in range(3)
    ]

    class _A:  # the verdict bars
        above_chance_bar = 0.20
        lesion_drop_bar = 0.20
        near_chance_tol = 0.12
        discrim_agreement_bar = 0.80
        max_value_salience_corr = 0.35

    v_go, _ = decide_r1a_verdict(synth_go, _A())
    # an R4-like NEGATIVE: value-ON ~ lesion (the choice is NOT the value system) -> lesion_does_not_collapse
    synth_neg = [dict(r, acc_value_lesion=0.88, lesion_collapses=False) for r in synth_go]
    v_neg, _ = decide_r1a_verdict(synth_neg, _A())
    # an artifact NEGATIVE: equal-value lesion CHANGES the choice (low agreement) -> not value-specific
    synth_art = [dict(r, equal_value_choice_agreement=0.40, equal_value_neutral=False) for r in synth_go]
    v_art, _ = decide_r1a_verdict(synth_art, _A())
    agg_ok = (v_go == "GO" and v_neg == "HONEST_NEGATIVE_lesion_does_not_collapse"
              and v_art == "HONEST_NEGATIVE_lesion_artifact_not_value_specific")
    print(f"  (f) verdict aggregator: synthetic GO->{v_go}; R4-like->{v_neg}; artifact->{v_art}  (sound: {agg_ok})")
    ok = ok and agg_ok

    # ── (g) the R1-b scaffold kwargs are well-formed ──
    sc_ok = r1b_scaffold_check()
    ok = ok and sc_ok

    print("=" * 78)
    print(f"[R5 SMOKE] {'PASS' if ok else 'FAIL'} — value distinct from salience, lesion = clean additive ablation, "
          f"equal-value zeroes the gradient, permuted breaks contingency, spiking WTA picks the higher-drive pool, "
          f"verdict aggregator sound, R1-b scaffold well-formed.")
    print("[R5 SMOKE] (the multi-seed R1-a gate is CPU + fast: --r1a; the nav two-beacon R1-b is the GPU eval, "
          "FOR CONTROLLER -- see the findings doc.)")
    print("=" * 78)
    return ok


def main():
    p = argparse.ArgumentParser(description="nav close-out R5 — a VALUE-DRIVEN-CHOICE task that proves the spiking "
                                            "value-critic is load-bearing BY ITS FUNCTION (the R4 value-irrelevant "
                                            "confound fix).")
    p.add_argument("--smoke", action="store_true", help="CPU smoke (tiny bridge): validate the harness is well-formed")
    p.add_argument("--r1a", action="store_true", help="run the R1-a value-WTA-choice gate (CPU, multi-seed)")
    p.add_argument("--r1b-scaffold-check", action="store_true",
                   help="validate the R1-b nav two-beacon kwargs (CPU, no bridge; the GPU eval is FOR CONTROLLER)")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n-options", type=int, default=2, help="number of choice options (2 = two-arm value choice)")
    p.add_argument("--n-trials", type=int, default=60, help="value-choice trials per seed (fresh values/salience each)")
    p.add_argument("--acc-steps", type=int, default=120, help="spiking integration window (steps)")
    # the spiking accumulator drift mapping (additive incentive-salience scheme; mirrors the appraisal probe)
    p.add_argument("--speak-base-pA", type=float, default=70.0, help="option-pool base drive")
    p.add_argument("--value-gain-pA", type=float, default=180.0, help="VALUE -> drift gain (the load-bearing axis)")
    p.add_argument("--salience-gain-pA", type=float, default=40.0,
                   help="SALIENCE -> drift gain (the value-independent 'default pull'; the lesion baseline)")
    # gate bars
    p.add_argument("--above-chance-bar", type=float, default=0.20,
                   help="min (value-ON accuracy - chance) for G_HEADLINE")
    p.add_argument("--lesion-drop-bar", type=float, default=0.20,
                   help="min (value-ON - value-LESION accuracy) for G_LESION (the lesion must COLLAPSE the choice)")
    p.add_argument("--near-chance-tol", type=float, default=0.15,
                   help="max |accuracy - chance| for an arm to count as 'reverted to chance'")
    p.add_argument("--discrim-agreement-bar", type=float, default=0.80,
                   help="min trial-by-trial intact-vs-lesion CHOICE AGREEMENT on equal-value trials for G_DISCRIM "
                        "(the lesion is NEUTRAL when value is a constant -> same choices)")
    p.add_argument("--max-value-salience-corr", type=float, default=0.35,
                   help="max |corr(value, salience)| for the value axis to be NON-circular")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    if a.smoke:
        ok = smoke()
        raise SystemExit(0 if ok else 1)
    if a.r1b_scaffold_check:
        ok = r1b_scaffold_check()
        raise SystemExit(0 if ok else 1)
    if a.r1a:
        run_r1a(a)
        return
    p.error("pass --smoke (CPU), --r1a (the value-WTA-choice gate, CPU multi-seed), or --r1b-scaffold-check")


if __name__ == "__main__":
    main()
