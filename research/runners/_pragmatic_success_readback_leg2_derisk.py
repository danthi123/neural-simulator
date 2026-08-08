"""D (Stage-4 CONVERSANT) · PRAGMATICS -- LEG 2: read the neural communicative-SUCCESS coincidence rate BACK as
group-scoped DA to TRAIN the speaker, so the intent->utterance CHOICE becomes a WTA over a LEARNED assembly (NOT a
host argmax over an imported RSA table).

Builds ON Leg 1 (2026-08-08-pragmatics-communicative-success-neural-coincidence-detector-leg1-6seed-GO) which
established, 6/6 seeds, that communicative success = neural <belief,intent> by the dendritic-coincidence plateau
(decisive vs a linear-sham). Leg 2 closes the loop: that same coincidence rate becomes the reinforcement that
shapes speaking.

ONE spiking bridge, all four populations co-resident:
  intent[K]      -- the one-hot communicative goal (drive).
  utterance[K]   -- the SPEAKER: K assemblies sharing an FS pool = a WTA. Driven by PLASTIC intent->utterance
                    synapses (three-factor, reward-modulated) + per-trial exploration bias. The winner = the
                    spoken utterance (a NEURAL WTA read = the body acting on motor output, the legitimate boundary).
  belief[K]      -- the LISTENER'S response to the spoken utterance, sourced from the RSA social environment
                    (build_rsa_bridge; L1 posterior where non-degenerate, literal-truth lexicon for the utterance
                    RSA leaves degenerate). This is world/social input.
  success[K]     -- the FIXED Leg-1 coincidence detector (belief[k] AND intent[k]); success = Sum_k rate.

TRAINING (three-factor, coincidence-contingent DA -- better-posed than the 2026-08-03 vocal-credit v1 NO-GO where
a naive reward-only DA->three-factor loop over-reinforced the early-active utterance): each trial (1) drives intent
+ exploration, the WTA picks an utterance u and the plastic intent->u synapses build eligibility (pre x post);
(2) the RSA listener responds to u -> belief -> the coincidence evaluator -> success; (3) RPE = success - running
baseline is delivered as current_reward_signal, converting eligibility to dw. A MISMATCH -> no coincidence -> low
success -> RPE below baseline -> the choice is WEAKENED (the negative arm the Gate-B WTA history says a reward-only
signal lacks). Over trials the WTA converges to the communicatively-optimal utterance per intent.

TEETH (each must behave, or it is an honest negative):
  - CONVERGENCE: after training, the neural WTA picks the aligned utterance per intent, acc >= 0.85, and ABOVE
    both the UNTRAINED control (pre-training WTA, ~chance) and the YOKED control (identical RPE magnitudes shuffled
    across trials so DA is decoupled from the actual choice -- the v1 failure mode).
  - UNTRAINED-ENGRAM: the pre-training WTA acc is at/near chance (the mapping is LEARNED, not wired).
  - YOKED: yoked acc <= 0.55 and << trained acc (attribution: the gain is the contingent DA, not the DA magnitude).
  - The CHOICE is a neural WTA (utterance population winner via shared inhibition), NOT a host argmax over RSA.

DECLARED FALLBACK (spec): if Leg 2 fails the yoked/convergence teeth, a spiking value-critic baseline is the
declared next method (still NO sim/ edit).

GO GATE (6-seed 42 43 44 100 101 102, CPU numpy):
  - trained_choice_acc     >= 0.85
  - untrained_choice_acc   <= 0.55   (chance for K=3 is 0.33)
  - yoked_choice_acc       <= 0.55
  - trained - yoked        >= 0.30

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_derisk --smoke --seed 42 \
      --json research/findings/raw/_pragmatic_success/leg2_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_success_readback_leg2_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_summary_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402
from research.runners._self_schema_region_derisk import WS_LOOP_GATE, WS_TO_FS_WEIGHT  # noqa: E402
from research.runners._recursive_tom_rsa_derisk import (  # noqa: E402
    build_rsa_bridge, _rsa_recursion, TRUTH, STATES, UTTS,
)
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    ITEM, DET, K, BELIEF_TOTAL, INTENT_PA, W_SYN, K_THR, GAIN, PLATEAU,
)

# ── speaker geometry ─────────────────────────────────────────────────────────────────────────────────────────
UTT_ITEM = 60            # utterance assembly size
UTT_FS_N = 40            # shared FS pool (WTA inhibition over utterances)
UTT_FS_W = 6.0           # utterance -> FS
FS_UTT_W = 16.0          # FS -> utterance (mutual inhibition; single winner)
W_I2U_INIT = 4.0         # initial intent->utterance plastic weight mean (small, symmetric -> choice is LEARNED)
W_I2U_JIT = 0.4          # small jitter so the untrained WTA is not perfectly tied (still ~chance)
SPEAK_GATE = "speak"     # plasticity gate for the plastic intent->utterance synapses
UTT_DRIVE_PA = 900.0     # tonic drive to all utterance assemblies (lets the WTA + learned weights pick a winner)
EXPLORE_PA = 1400.0      # per-trial random exploration bias added to one/each utterance (epsilon-like sampling)
SETTLE_MS = 40           # WTA settle window (eligibility built here)
READ_UTT = 25            # read the utterance winner over the last READ_UTT steps
BELIEF_MS = 45           # listener-response + coincidence read window
REWARD_MS = 8            # RPE conversion window (deliver current_reward_signal)
N_TRAIN = 360            # training trials
LR = 0.25                # reward_learning_rate
ELIG_TAU = 60.0          # eligibility tau (ms)
BASELINE_TAU = 0.02      # EMA rate for the running success baseline
REWARD_GAIN = 30.0       # success-rate -> RPE scale


def _belief_sources(seed):
    """The listener's response per utterance = RSA L1 posterior where non-degenerate; literal-truth normalized
    where RSA leaves it degenerate (documented). Returns dict u -> belief_vec[K] (sum 1) and the aligned state."""
    b, xp, item_dev, snap = build_rsa_bridge(seed, normalize=True)
    L0, S1, L1 = _rsa_recursion(b, xp, item_dev, snap, TRUTH, 25)
    out = {}
    for j, u in enumerate(UTTS):
        v = np.asarray(L1[j], dtype=np.float64).copy()
        if v.sum() <= 1e-9:                       # degenerate RSA posterior -> literal-truth lexicon fallback
            v = np.array([TRUTH[u][s] for s in STATES], dtype=np.float64)
        v = v / v.sum()
        out[u] = v
    return out


def build_speaker_bridge(seed):
    """ONE bridge: intent[K] -> (PLASTIC) utterance[K]-WTA ; belief[K]+intent[K] -> (coincidence) success[K]."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="intent", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="utter", n_neurons=UTT_ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="utter_fs", n_neurons=UTT_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="belief", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="success", n_neurons=DET * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="utter", to_region="utter_fs", density=0.6, weight_mean=UTT_FS_W,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="utter_fs", to_region="utter", density=0.6, weight_mean=FS_UTT_W,
                      weight_jitter=0.0, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process", "enable_nmda"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    # three-factor reward modulation on the intent->utterance synapses
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = float(LR)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.reward_eligibility_tau_ms = float(ELIG_TAU)
    # eligibility must build from pre x post COACTIVITY (not from STDP/Hebbian weight-change, which is OFF here) --
    # otherwise the eligibility trace stays 0 and the three-factor rule has nothing to convert (diagnosed 2026-08-08).
    cfg.reward_eligibility_from_coactivity = True
    cfg.reward_coactivity_trace_tau_ms = float(ELIG_TAU)
    cfg.reward_coactivity_scale = 0.2
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    # coincidence evaluator (Leg 1), fixed
    cfg.enable_coincidence_detection = True
    cfg.coincidence_k_threshold = float(K_THR)
    cfg.coincidence_gain = float(GAIN)
    cfg.coincidence_plateau_strength = float(PLATEAU)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    intent = np.asarray(rm.indices("intent"), dtype=np.int64)
    utter = np.asarray(rm.indices("utter"), dtype=np.int64)
    belief = np.asarray(rm.indices("belief"), dtype=np.int64)
    suc = np.asarray(rm.indices("success"), dtype=np.int64)
    intent_k = {k: intent[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    utter_k = {k: utter[k * UTT_ITEM:(k + 1) * UTT_ITEM] for k in range(K)}
    belief_k = {k: belief[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    suc_k = {k: suc[k * DET:(k + 1) * DET] for k in range(K)}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    # PLASTIC intent[t] -> utter[u], all t x u (the learned speaker mapping), reward-gated (SPEAK_GATE).
    rng = np.random.default_rng(seed * 17 + 3)
    for t in range(K):
        for u in range(K):
            pre = np.repeat(intent_k[t], UTT_ITEM)
            post = np.tile(utter_k[u], ITEM)
            w = (W_I2U_INIT + rng.normal(0.0, W_I2U_JIT, pre.shape[0])).astype(np.float32)
            union[f"i2u_{t}_{u}"] = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
                                    "initial_weights": np.clip(w, 0.1, None), "plastic": True,
                                    "plasticity_gate": SPEAK_GATE, "conn_type": "E_TO_E", "count": int(pre.size)}
    # FIXED coincidence: belief[k]->success[k], intent[k]->success[k] (Leg-1 AND)
    for k in range(K):
        d1 = _dense_projection(belief_k[k], suc_k[k], W_SYN, WS_LOOP_GATE); d1["coincidence_detector"] = True
        d2 = _dense_projection(intent_k[k], suc_k[k], W_SYN, WS_LOOP_GATE); d2["coincidence_detector"] = True
        union[f"bel2suc_{k}"] = d1
        union[f"itn2suc_{k}"] = d2

    inh = list(rm.inhibitory_indices("utter_fs"))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)           # freeze the coincidence pathways
    bridge.set_plasticity_gate(SPEAK_GATE, 1.0)             # enable learning on intent->utterance

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {"intent": {k: xp.asarray(intent_k[k]) for k in range(K)},
           "utter": {k: xp.asarray(utter_k[k]) for k in range(K)},
           "belief": {k: xp.asarray(belief_k[k]) for k in range(K)},
           "suc_all": xp.asarray(suc)}
    return bridge, xp, idx, snap


def _choose_utterance(bridge, xp, idx, snap, intent_t, explore_rng=None):
    """Drive intent[t] + tonic utterance drive (+ per-trial exploration bias); let the WTA settle; read the
    winning utterance (the utterance assembly with the highest late-window rate = the neural speaker CHOICE).
    Builds eligibility on the plastic intent->utterance synapses (pre x post). Returns (winner, utt_rates)."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0          # eligibility-only phase (no conversion yet)
    bias = np.zeros(K)
    if explore_rng is not None:
        bias[explore_rng.integers(K)] = EXPLORE_PA          # epsilon-like: boost a random utterance this trial
    acc = np.zeros(K)
    for s in range(SETTLE_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][intent_t]] = xp.float32(INTENT_PA)
        for u in range(K):
            bridge.cp_external_input_current[idx["utter"][u]] = xp.float32(UTT_DRIVE_PA + bias[u])
        bridge._run_one_simulation_step()
        if s >= SETTLE_MS - READ_UTT:
            for u in range(K):
                acc[u] += float(to_host(bridge.cp_firing_states[idx["utter"][u]].astype(xp.float64).sum()))
    rates = acc / (READ_UTT * UTT_ITEM)
    return int(np.argmax(rates)), rates


def _evaluate_success(bridge, xp, idx, intent_t, belief_vec):
    """Continue the SAME trial: the listener responds to the chosen utterance (inject belief_vec) with intent[t]
    held; read the coincidence success rate. Eligibility keeps decaying (tau) so the intent->utterance synapses
    active in the choice phase remain eligible for the reward that follows."""
    acc = 0.0
    for s in range(BELIEF_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][intent_t]] = xp.float32(INTENT_PA)
        for k in range(K):
            if belief_vec[k] > 0.0:
                bridge.cp_external_input_current[idx["belief"][k]] = xp.float32(BELIEF_TOTAL * float(belief_vec[k]))
        bridge._run_one_simulation_step()
        acc += float(to_host(bridge.cp_firing_states[idx["suc_all"]].astype(xp.float64).sum()))
    return acc / (BELIEF_MS * DET * K)


def _deliver_reward(bridge, xp, idx, rpe):
    """Conversion phase: set current_reward_signal = RPE for REWARD_MS steps -> the engine converts the standing
    eligibility to dw on the plastic (SPEAK_GATE) synapses. Intent/utterance held lightly so state is stable."""
    bridge.core_config.current_reward_signal = float(rpe)
    for _ in range(REWARD_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    bridge.core_config.current_reward_signal = 0.0


def _readout_policy(bridge, xp, idx, snap):
    """GREEDY read-out (no exploration, no learning): for each intent, the WTA winner = the learned choice."""
    saved = bridge.core_config.reward_learning_rate
    bridge.core_config.reward_learning_rate = 0.0
    choice = {}
    for t in range(K):
        w, _ = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None)
        choice[t] = w
    bridge.core_config.reward_learning_rate = saved
    return choice


def evaluate_seed(seed, n_train=N_TRAIN, verbose=True, yoked=False):
    t0 = time.time()
    belief_src = _belief_sources(seed)
    # aligned utterance per intent = the utterance whose listener-belief argmax == the intent state.
    aligned_utt = {}
    for t in range(K):
        best_u, best_mass = None, -1.0
        for ui, u in enumerate(UTTS):
            if belief_src[u][t] > best_mass:
                best_mass, best_u = belief_src[u][t], ui
        aligned_utt[t] = best_u
    belief_by_uidx = {ui: belief_src[u] for ui, u in enumerate(UTTS)}

    bridge, xp, idx, snap = build_speaker_bridge(seed)
    untrained = _readout_policy(bridge, xp, idx, snap)
    untrained_acc = float(np.mean([untrained[t] == aligned_utt[t] for t in range(K)]))

    rng = np.random.default_rng(seed * 71 + 13)
    baseline = 0.0
    # For the YOKED control: collect the RPE stream from a real run, then replay it SHUFFLED, decoupled from choice.
    yoked_rpes = None
    if yoked:
        # first do a real pass to harvest an RPE distribution, then rebuild a fresh bridge and replay shuffled.
        _tmp = evaluate_seed(seed, n_train=n_train, verbose=False, yoked=False)
        yoked_rpes = np.array(_tmp["_rpe_stream"])
        rng.shuffle(yoked_rpes)
        bridge, xp, idx, snap = build_speaker_bridge(seed)   # fresh substrate
        baseline = 0.0

    rpe_stream = []
    for i in range(n_train):
        t = int(rng.integers(K))
        winner, _ = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=rng)
        success = _evaluate_success(bridge, xp, idx, t, belief_by_uidx[winner])
        if yoked:
            rpe = float(yoked_rpes[i % len(yoked_rpes)])      # DA decoupled from THIS choice's success
        else:
            rpe = REWARD_GAIN * (success - baseline)
            baseline += BASELINE_TAU * (success - baseline)
        rpe_stream.append(REWARD_GAIN * (success - baseline) if not yoked else success)
        _deliver_reward(bridge, xp, idx, rpe)

    trained = _readout_policy(bridge, xp, idx, snap)
    trained_acc = float(np.mean([trained[t] == aligned_utt[t] for t in range(K)]))

    m = {
        "seed": int(seed), "yoked": bool(yoked),
        "aligned_utt": {str(t): int(aligned_utt[t]) for t in range(K)},
        "untrained_choice": {str(t): int(untrained[t]) for t in range(K)},
        "trained_choice": {str(t): int(trained[t]) for t in range(K)},
        "untrained_choice_acc": untrained_acc,
        "trained_choice_acc": trained_acc,
        "n_train": n_train, "chance": 1.0 / K,
        "_rpe_stream": rpe_stream,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  [seed {seed}]{' YOKED' if yoked else ''} ({m['elapsed_seconds']}s) untrained_acc={untrained_acc:.3f} "
              f"trained_acc={trained_acc:.3f} aligned={m['aligned_utt']} trained={m['trained_choice']}", flush=True)
    return m


def evaluate_seed_full(seed, n_train=N_TRAIN, verbose=True):
    real = evaluate_seed(seed, n_train=n_train, verbose=verbose, yoked=False)
    yok = evaluate_seed(seed, n_train=n_train, verbose=verbose, yoked=True)
    m = {k: v for k, v in real.items() if k != "_rpe_stream"}
    m["yoked_choice_acc"] = yok["trained_choice_acc"]
    m["go"] = _seed_go(m)
    if verbose:
        print(f"    >>> seed {seed} GO={m['go']}  trained={m['trained_choice_acc']:.3f} "
              f"untrained={m['untrained_choice_acc']:.3f} yoked={m['yoked_choice_acc']:.3f}", flush=True)
    return m


THR = {"trained": 0.85, "untrained_max": 0.55, "yoked_max": 0.55, "trained_minus_yoked": 0.30}


def _seed_go(m):
    return bool(m["trained_choice_acc"] >= THR["trained"]
                and m["untrained_choice_acc"] <= THR["untrained_max"]
                and m["yoked_choice_acc"] <= THR["yoked_max"]
                and (m["trained_choice_acc"] - m["yoked_choice_acc"]) >= THR["trained_minus_yoked"])


def _mean(ps, k):
    vals = [r[k] for r in ps if r.get(k) is not None]
    return float(np.mean(vals)) if vals else None


def build_summary(per_seed, seeds, backend):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed) and len(per_seed) > 0)
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")
    agg = {"mean_trained_acc": _mean(per_seed, "trained_choice_acc"),
           "mean_untrained_acc": _mean(per_seed, "untrained_choice_acc"),
           "mean_yoked_acc": _mean(per_seed, "yoked_choice_acc")}

    v = Verdict("D pragmatics (Leg 2): coincidence-success read back to TRAIN a neural WTA speaker", chance=1.0 / K)
    v.require("6 seeds (project bar)", len(seeds) >= 6, expect=True)
    v.floor("trained choice acc vs chance 1/K", agg["mean_trained_acc"], 1.0 / K)
    v.require("trained choice acc >= 0.85", agg["mean_trained_acc"], expect=lambda x: x >= THR["trained"])
    v.require("UNTRAINED choice ~chance (the mapping is LEARNED, not wired)",
              agg["mean_untrained_acc"], expect=lambda x: x <= THR["untrained_max"])
    v.require("YOKED (DA decoupled from choice) does NOT learn the mapping",
              agg["mean_yoked_acc"], expect=lambda x: x <= THR["yoked_max"])
    v.control("trained vs YOKED (the gain is contingent DA, not DA magnitude)",
              treatment=agg["mean_trained_acc"], control=agg["mean_yoked_acc"])
    v.control("trained vs UNTRAINED (the gain is learning)",
              treatment=agg["mean_trained_acc"], control=agg["mean_untrained_acc"])
    v.require("all seeds GO", all_go, expect=True)
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU/NMDA",
               "only the reward-modulated three-factor rule learns (on intent->utterance); the coincidence "
               "evaluator + WTA inhibition are frozen.")
    vb = v.decide(go=all_go)
    # The Verdict is authoritative: when any require/precondition is unmet (e.g. an underpowered <6-seed smoke, or
    # a result-gate below bar) it returns UNDEFINED, which must NOT be reported as NEGATIVE (verdict-preconditions
    # gate). The teeth-backed NEGATIVE for Leg 2 lives in the oracle-weight probe artifact, not in this Verdict.
    verdict = vb["status"] if vb["status"] != "GO" else ("GO" if all_go else "PARTIAL")
    attributable_to("trained-vs-chance gain attributable to CONTINGENT DA (vs yoked)",
                    agg["mean_trained_acc"] - 1.0 / K, agg["mean_yoked_acc"] - 1.0 / K)

    summary = {
        "runner": "_pragmatic_success_readback_leg2_derisk",
        "leg": "LEG 2 -- coincidence-success DA trains a neural WTA speaker",
        "faculty": "D pragmatics: the neural communicative-success coincidence rate read BACK as group-scoped DA "
                   "to learn the intent->utterance mapping (WTA over a LEARNED assembly).",
        "builds_on": "2026-08-08-pragmatics-communicative-success-neural-coincidence-detector-leg1-6seed-GO",
        "seeds": list(seeds), "backend": backend, "chance": 1.0 / K,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "thresholds": THR, "plasticity_off": False,
        "fallback_if_negative": "spiking value-critic baseline (declared; still no sim/ edit)",
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "aggregate": agg, "per_seed": per_seed,
        "honest_scope": ("Leg 2 trains the speaker with the Leg-1 neural coincidence success as reinforcement. The "
                         "CHOICE is a neural WTA over the utterance population (read as the spoken utterance = the "
                         "body acting on motor output); the RSA listener posterior is the environment's response. "
                         "The DA is coincidence-CONTINGENT (RPE = success - baseline, with a negative arm). GO "
                         "requires the trained WTA to beat BOTH the untrained control and the yoked control. If "
                         "the yoked/convergence teeth fail, the declared fallback is a spiking value-critic. NOT a "
                         "claim of understanding another mind; numpy-CPU; NO sim/ edit."),
    }
    return summary, verdict, all_go


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    a = summary["aggregate"]
    print("\n" + "=" * 100, flush=True)
    print(f"[leg2] === VERDICT: {verdict} ({summary['n_go']}/{summary['n_seeds']} seeds GO) ===", flush=True)
    print(f"[leg2]  trained={a['mean_trained_acc']} untrained={a['mean_untrained_acc']} yoked={a['mean_yoked_acc']}",
          flush=True)
    print(f"[leg2]  wrote {out_path}\n" + "=" * 100, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--aggregate", type=str, default=None)
    ap.add_argument("--json", type=str, default="research/findings/raw/_pragmatic_success/leg2_summary.json")
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)

    if args.aggregate:
        files = sorted(glob.glob(args.aggregate))
        if not files:
            print(f"[leg2] no files match {args.aggregate}", flush=True)
            return 2
        per_seed = []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            per_seed.extend(d["per_seed"] if "per_seed" in d and "seed" not in d else [d])
        per_seed = [p for p in per_seed if "seed" in p and "go" in p]
        per_seed.sort(key=lambda p: p["seed"])
        seeds = [p["seed"] for p in per_seed]
        summary, verdict, _ = build_summary(per_seed, seeds, args.backend)
        _emit(summary, verdict, args.json)
        return 0 if verdict == "GO" else 1

    n_train = min(args.n_train, 120) if args.smoke else args.n_train
    seeds = args.seeds if args.seeds is not None else [args.seed]
    print(f"[leg2] D pragmatics LEG 2 -- coincidence success -> DA -> neural WTA speaker | seeds={seeds} "
          f"n_train={n_train} backend={args.backend}", flush=True)
    per_seed = [evaluate_seed_full(s, n_train=n_train, verbose=True) for s in seeds]

    if len(seeds) == 1 and args.seeds is None and not args.smoke:
        Path(os.path.dirname(os.path.abspath(args.json))).mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(per_seed[0], f, indent=2, default=str)
        print(f"[leg2] wrote per-seed record {args.json} (go={per_seed[0]['go']})", flush=True)
        return 0 if per_seed[0]["go"] else 1

    summary, verdict, all_go = build_summary(per_seed, seeds, args.backend)
    _emit(summary, verdict, args.json)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
