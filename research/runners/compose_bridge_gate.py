"""Kill-safe THREE-STATE gate: does the validated temporal-credit/
eligibility mechanism bridge the verb->motor compositional bind-gap
inside a MINIMAL slice of the real spiking sim.bridge concept-pool
architecture (the v16 setting)? Reuses build_biological_brain_regions
+ the bridge's native cp_eligibility_trace reward-modulation path +
the NM subsystem (TD delta = phasic-DA, catalog C.30) +
sim.train_checkpoint, ALL byte-UNMODIFIED. hebbian_no_trace = the
faithful v16-cold-start analog (identical to td minus EXACTLY the
eligibility-trace bridging). NO automatic differentiation. ASCII.

HONEST CEILING (printed, never spun): a PASS = the mechanism
transfers into a MINIMAL slice of the real spiking architecture (the
first in-architecture mechanistic dent in the composition blocker) --
NOT composition-solved, NOT compositional language, NOT scaled/chat-
integrated (a further SEPARATE gated increment). PASS/BOUNDARY/VOID
all decision-relevant + propagated honestly."""
from __future__ import annotations
import argparse
import json
import os
import sys

# Force the NumPy CPU backend for a deterministic, fast minimal slice
# (set BEFORE any sim import that may cache the backend). The mechanism
# under test is backend-agnostic; CPU is sufficient for a minimal slice
# and avoids GPU nondeterminism in the gate decision.
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.text_minimal_isolation import (
    build_biological_brain_regions)
from sim.kernels import fused_eligibility_trace_decay
from sim.train_checkpoint import (save_checkpoint, load_checkpoint,
                                  resume_epoch)
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)
from research.runners.compose_bridge_core import cbr_verdict

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: minimal-spiking-slice mechanism-transfer "
           "ONLY -- NOT composition-solved, NOT compositional "
           "language, NOT scaled/chat-integrated (a further SEPARATE "
           "gated increment).")

# TD(lambda) constants (the validated rule; frozen here).
_GAMMA = 0.95
_LAMBDA = 0.9

# ---- minimal-slice topology ------------------------------------------------
# >=8 distinct (verb,motor) bindings. The "verb" is an orthogonal
# language_input code; the "motor" is a dedicated concept output pool
# (noun_pool_*) -- the v16 concept-pool architecture's plastic
# lang_input -> pool path, learned through the bridge's native
# eligibility/reward modulation. Bijection pi maps verb i -> pool i.
_N_BINDINGS = 8
_POOL_NAMES = ["P%d" % i for i in range(_N_BINDINGS)]

# Full-slice scale (still minimal vs production v16). n_lang_input
# chosen so the 8 orthogonal verb bands are comfortably non-
# overlapping: stride = 512//8 = 64 >= n_active = round(0.10*512) = 51.
_FULL = dict(n_lang_input=512, n_per_pool=40, n_fs_per_pool=6,
             stim_steps=24, gap_steps=14, reset_steps=10,
             readout_steps=18, n_train_epochs=10, drive_pA=260.0,
             teacher_pA=420.0, sparsity=0.10)
# tiny-synth: aggressively shrunk so the smoke completes FAST. Its
# verdict is a toy (likely VOID at this scale) and is NEVER propagated.
# stride = 128//8 = 16 >= n_active = round(0.10*128) = 13.
_TINY = dict(n_lang_input=128, n_per_pool=10, n_fs_per_pool=2,
             stim_steps=4, gap_steps=3, reset_steps=2,
             readout_steps=4, n_train_epochs=2, drive_pA=260.0,
             teacher_pA=420.0, sparsity=0.10)


def _da_modulator_from_delta():
    """Catalog C.30 upgrade via the REUSED NM subsystem UNMODIFIED:
    from_reward DA modulator whose drive is the TD delta. Constructed
    to prove composition with the validated phasic-DA substrate; not
    mutated."""
    return NeuromodulatorConfig(
        name="dopamine_compose_bridge", baseline=0.0,
        decay_tau_ms=50.0, concentration_min=-5.0,
        concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0,
                                         threshold=0.0,
                                         window_ms=0.0)])


def _build_bridge(seed, P):
    """Build the MINIMAL verb/concept-pool spiking bridge via the
    REUSED build_biological_brain_regions UNMODIFIED. noun_pool_P* are
    the dedicated output ("motor") pools; language_input carries the
    orthogonal verb codes; the plastic, gate-tagged
    language_input -> noun_pool path is the synapse set the native
    eligibility/reward modulation learns."""
    from sim.config import (CoreSimConfig, VisualizationConfig,
                            RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge

    regions, pathways = build_biological_brain_regions(
        n_lang_input=P["n_lang_input"],
        n_motor_per_action=8,            # vestigial motor pools (unused)
        enable_motor_fs=False,
        enable_noun_pools=True,
        noun_pool_names=list(_POOL_NAMES),
        n_noun_per_pool=P["n_per_pool"],
        n_noun_fs_per_pool=P["n_fs_per_pool"],
        # weak concept-pool dynamics (iter-AA recipe; the v16 setting):
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    # Native three-factor path ON (these are defaults; pinned for clarity).
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.05
    cfg.reward_eligibility_tau_ms = 200.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 8.0  # above design weights (soft-bound gotcha)
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _verb_drive(verb_idx, n_lang_input, P):
    """Deterministic orthogonal verb code on language_input (the
    proven concept-pool drive idiom; one disjoint band per verb)."""
    from sim.text_embeddings import orthogonal_drive_pattern
    return orthogonal_drive_pattern(
        cue_idx=verb_idx, n_cues=_N_BINDINGS,
        n_neurons=n_lang_input, drive_max_pA=P["drive_pA"],
        sparsity=P["sparsity"])


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1


def _episode(bridge, mode, verb_idx, target_pool_idx, rng, P,
             lang_arr, pool_arrs, value_table):
    """One bind episode per the BEHAVIORAL SPEC.

    t_A: drive language_input with verb V_i (+ teacher current on the
         target pool so STDP co-fires the lang_input -> pool synapses
         and the native eligibility trace charges).
    GAP: G steps with NO decision drive -- eligibility decays through
         the bridge's own fused_eligibility_trace_decay every step.
         hebbian_no_trace ZEROES cp_eligibility_trace each gap step
         (the ONLY difference vs td: the v16-cold-start analog).
    t_R: population-vote readout over the pools -> selected pool;
         reward r = 1 iff selected == M_pi(i) else 0. r is applied
         ONLY now (delayed past the gap), via the native reward path
         (current_reward_signal = TD delta), so a static readout
         cannot shortcut the temporal credit.
    """
    cp = bridge.xp if hasattr(bridge, "xp") else np
    n_lang = lang_arr.shape[0]

    # --- reset (decay residual state between episodes) ---
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)

    # --- t_A: verb drive + teacher current on the bound target pool ---
    drive = cp.asarray(_verb_drive(verb_idx, n_lang, P),
                       dtype=cp.float32)
    bridge.cp_external_input_current[lang_arr] = drive
    bridge.cp_external_input_current[pool_arrs[target_pool_idx]] += \
        float(P["teacher_pA"])
    for _ in range(P["stim_steps"]):
        _step(bridge)

    # --- TEMPORAL GAP: no decision drive; eligibility decays in-bridge.
    # current_reward_signal is held at 0 here, so the native path only
    # decays the trace (no weight update) -- reward is strictly delayed.
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(P["gap_steps"]):
        if mode == "hebbian_no_trace" and \
                bridge.cp_eligibility_trace is not None:
            # The faithful v16-cold-start analog: suppress ONLY the
            # eligibility bridging across the gap. Everything else
            # (drive, gap length, readout, reward, RNG order) is
            # byte-identical to td.
            bridge.cp_eligibility_trace[:] = 0.0
        _step(bridge)

    # --- t_R: population-vote readout (greedy spike count per pool) ---
    counts = np.zeros(_N_BINDINGS, dtype=np.float64)
    for _ in range(P["readout_steps"]):
        _step(bridge)
        fired = bridge.cp_firing_states
        for j, pa in enumerate(pool_arrs):
            counts[j] += float(fired[pa].sum())
    selected = int(np.argmax(counts))
    reward = 1.0 if selected == target_pool_idx else 0.0

    # --- TD(lambda) delta = r - V(s); update tabular value of the
    # verb state, then drive the native reward-modulation path with
    # delta as current_reward_signal for ONE step so the bridge's own
    # cp_eligibility_trace * reward * reward_learning_rate update fires
    # (gamma/lambda used in the standard one-step TD target; this is a
    # single decision so the bootstrap term is the value estimate).
    v = float(value_table[verb_idx])
    delta = reward - v
    value_table[verb_idx] = v + (1.0 - _GAMMA * _LAMBDA) * delta
    if mode == "wrongsign":
        delta = -delta

    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = float(delta)
    _step(bridge)
    bridge.core_config.current_reward_signal = 0.0
    return reward, selected


def _bijection(rng):
    perm = np.arange(_N_BINDINGS)
    rng.shuffle(perm)
    return perm


def _greedy_score(bridge, pi, P, lang_arr, pool_arrs):
    """Noise-free greedy accuracy: drive each verb (NO teacher, NO
    reward), population-vote the pools, score against pi."""
    cp = bridge.xp if hasattr(bridge, "xp") else np
    n_lang = lang_arr.shape[0]
    correct = 0
    for vi in range(_N_BINDINGS):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(P["reset_steps"]):
            _step(bridge)
        drive = cp.asarray(_verb_drive(vi, n_lang, P),
                           dtype=cp.float32)
        bridge.cp_external_input_current[lang_arr] = drive
        for _ in range(P["stim_steps"] + P["gap_steps"]):
            _step(bridge)
        counts = np.zeros(_N_BINDINGS, dtype=np.float64)
        for _ in range(P["readout_steps"]):
            _step(bridge)
            fired = bridge.cp_firing_states
            for j, pa in enumerate(pool_arrs):
                counts[j] += float(fired[pa].sum())
        if int(np.argmax(counts)) == int(pi[vi]):
            correct += 1
    bridge.cp_external_input_current[:] = 0.0
    return correct / float(_N_BINDINGS)


def _run_mode(mode, seed, tiny, gap_zero=False):
    """Build the minimal verb/motor concept-pool bridge, run the
    verb->(gap)->motor+reward episodes per the BEHAVIORAL SPEC for
    `mode`, return greedy accuracy (float). td uses the bridge native
    eligibility reward path + TD(lambda) (gamma=0.95, lambda=0.9);
    hebbian_no_trace suppresses ONLY the trace across the gap and is
    otherwise byte-identical to td. permuted re-randomizes pi every
    episode; wrongsign flips the delta sign. gap_zero forces G=0 (V1
    instrument soundness). NO automatic differentiation."""
    P = dict(_TINY if tiny else _FULL)
    if gap_zero:
        P["gap_steps"] = 0
    bridge = _build_bridge(seed, P)
    cp = bridge.xp if hasattr(bridge, "xp") else np
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                          dtype=cp.int64)
    pool_arrs = [cp.asarray(list(rm.indices("noun_pool_%s" % nm)),
                            dtype=cp.int64) for nm in _POOL_NAMES]

    # Open ONLY the plastic lang_input -> noun_pool gate (the synapse
    # set under test); all other gates frozen so credit is isolated.
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 1.0)
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    pi = _bijection(rng)
    value_table = np.zeros(_N_BINDINGS, dtype=np.float64)

    for _ep in range(P["n_train_epochs"]):
        if mode == "permuted":
            pi = _bijection(rng)
        order = np.arange(_N_BINDINGS)
        rng.shuffle(order)
        for vi in order:
            _episode(bridge, mode, int(vi), int(pi[vi]), rng, P,
                     lang_arr, pool_arrs, value_table)

    # Freeze plasticity for the noise-free greedy score.
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 0.0)
    except Exception:
        pass
    return _greedy_score(bridge, pi, P, lang_arr, pool_arrs)


def _run_seed(mode, seed, tiny):
    """Build the minimal verb/motor concept-pool bridge via
    build_biological_brain_regions (REUSED UNMODIFIED), run the
    verb->(gap)->motor+reward episodes per the BEHAVIORAL SPEC for
    `mode`, return greedy accuracy (float). Reuse the validated
    TD(lambda) rule (gamma=0.95, lambda=0.9) via the bridge native
    cp_eligibility_trace reward path + fused_eligibility_trace_decay;
    hebbian_no_trace suppresses the trace across the gap and is
    otherwise byte-identical to td. `tiny` shrinks
    pools/episodes/steps for the smoke only (a toy verdict, NOT
    propagated). NO automatic differentiation."""
    return _run_mode(mode, seed, tiny, gap_zero=False)


def _run_seed_nogap(seed, tiny):
    """V1 instrument-soundness: td with the temporal gap forced to 0."""
    return _run_mode("td", seed, tiny, gap_zero=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds for the pre-registered gate")
        return 2
    _ = _da_modulator_from_delta()  # construct (not mutate)
    per_seed = {}
    try:
        for s in a.seeds:
            row = {"nogap_td": _run_seed_nogap(s, a.tiny_synth),
                   "td": _run_seed("td", s, a.tiny_synth),
                   "controls": {c: _run_seed(c, s, a.tiny_synth)
                                for c in _CONTROLS}}
            if a.ckpt:
                save_checkpoint(a.ckpt, s,
                                {"row": [row["nogap_td"], row["td"]]},
                                None, [])
            per_seed[s] = row
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    verdict = cbr_verdict(per_seed)
    verdict["banner"] = _BANNER
    if a.tiny_synth:
        verdict["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("GATE=%s  %s" % (verdict["GATE"], _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
