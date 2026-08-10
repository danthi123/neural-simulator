"""D (Stage-4 CONVERSANT) · PRAGMATICS -- LEG 2 v3: fix the LEARNING-stage value-critic wall with a per-CONTEXT
STATE-VALUE baseline (songbird Area-X / VTA actor-critic proper), replacing the per-ACTION Q-critic that collapses
the actor's teaching signal at convergence.

THE ISOLATED BUG (v2, established 2026-08-10): v2's critic V is a K-vector of per-UTTERANCE critic rates
(crit[u] = V(intent, u)); the actor update is rpe = REWARD_GAIN*(success - V[winner]) = a PER-ACTION advantage. At
convergence each V[u] -> success(intent,u), so the advantage collapses to ~0 for EVERY utterance -> the actor loses
its differential teacher and decays to heterogeneity chance (critic-argmax 0.556, actor 0.500; seed 100 inverted).

THE FIX (distinct in kind): replace the per-ACTION Q-critic with a per-CONTEXT STATE-VALUE baseline Vctx[intent] =
ONE scalar per intent predicting the EXPECTED success over the current policy's utterance distribution. Then the
actor advantage A = success(chosen) - Vctx[intent] is SIGNED: positive for above-context-average (aligned)
utterances -> potentiate; negative for below-average (misaligned) -> actively DEPRESS. The signed increment COMPOUNDS
across trials (aligned weight climbs, misaligned falls, soft-bounded), so the final separation is set by TRIAL COUNT,
not the tiny single-trial gap -- exactly why it can succeed where readout amplification failed.
Grounding: Kasdin et al. 2025 Nature (Area-X DA = contrast of current rendition vs recent-rendition HISTORY = a
STATE baseline, not per-action); Gadagkar et al. 2016 Science 354:1278 (bidirectional prediction-relative
performance error); Chen et al. 2018 (ventral state-value critic -> VTA relays signed error -> Area-X actor).

TWO CHANGES ONLY vs v2 (everything else reused by import; NO sim/ edit):
  1. CRITIC representation -> per-CONTEXT state value Vctx[intent].
       - Default (host-EMA, FLAGGED SHORTCUT): Vctx[intent] = host EMA of success-per-intent (Vctx[t] += beta*
         (success - Vctx[t])). The ACTOR remains fully neural (spiking soft-WTA choice, eligibility trace,
         DA-gated plasticity); only the scalar baseline is host-computed. This cleanly ISOLATES the learning fix
         (the advantage SOURCE) from any critic-representation confound.
       - --neural-critic: Vctx[intent] = rate(crit[intent]) from a spiking per-INTENT critic population (crit
         indexed by INTENT, not utterance), trained by a TD error delta_crit = success - rate(crit[intent])
         (Chen-2018 ventral state-value critic). Reported when run; calibration-sensitive.
  2. ACTOR advantage A = success - Vctx[intent] (replacing rpe = success - V[winner]), delivered as the
     decision-locked DA pulse that converts the (action-localized) eligibility.

TARGET (honest, reframed 2026-08-10): the neural success signal s(t,u) is DETERMINISTIC but does NOT rank the
belief-ALIGNED utterance highest for ~56% of targets (success_opt == aligned only 8/18 across these 6 seeds). So a
success-maximizing speaker CANNOT reach the aligned mapping -- the v2 GO bar (>=0.85 vs aligned) was unreachable by
ANY rule. To test the LEARNING FIX in isolation we measure convergence toward SUCCESS-OPTIMAL (argmax_u s(t,u) = what
the reward actually teaches); the aligned metric is reported separately as a distinct reward-misspecification wall.

THE DECISIVE CONTROL -- YOKED / shuffled-reward (the exact gate gateB-stage2 FAILED): the same distribution of DA
magnitudes is delivered DECOUPLED from the action taken (reward permuted across trials). The yoked arm MUST NOT
converge (actor stays ~chance, weight separation ~0). A fix that converges AND yoked ALSO converges is NOT contingent
learning -> NEGATIVE. This contingency gate is THE test.

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3_statevalue_derisk --smoke --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3_statevalue_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_v3_statevalue_6seed.json
  # neural spiking state-value critic (secondary):
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3_statevalue_derisk --neural-critic \
      --seeds 42 43 44 100 101 102 --json .../leg2_v3_neuralcritic_6seed.json
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

from sim.backend import get_backend, to_host  # noqa: E402

# Reuse the ENTIRE v2 machinery by import (actor, spiking utterance assemblies, neural coincidence success detector,
# phase-separated commit->reward, eligibility/DA hooks, executed epsilon-greedy). NO sim/ edit; additive.
from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    build_speaker_bridge, _choose_utterance, _evaluate_success, _commit_action, _deliver_reward,
    _readout_policy, _belief_sources, _aligned_utts, oracle_probe_seed,
    K, ITEM, UTT_ITEM, CRIT_ITEM, INTENT_PA, REWARD_GAIN, EPSILON, N_TRAIN,
    SPEAK_GATE, CRIT_GATE, COMMIT_MS, COMMIT_PA,
)
from research.runners._recursive_tom_rsa_derisk import UTTS  # noqa: E402

# ── v3 state-value knobs ───────────────────────────────────────────────────────────────────────────────────────
EMA_BETA = 0.10          # host-EMA baseline tracking rate for Vctx[intent] (slow policy-mean success tracker)
CRIT_READ_GAIN_V3 = 1.0  # neural critic: rate(crit[intent]) -> Vctx scale (calibration-sensitive)


def _region_blocks(bridge):
    rm = bridge.region_manager
    intent = np.asarray(rm.indices("intent"), dtype=np.int64)
    utter = np.asarray(rm.indices("utter"), dtype=np.int64)
    intent_blk = {t: intent[t * ITEM:(t + 1) * ITEM] for t in range(K)}
    utter_blk = {u: utter[u * UTT_ITEM:(u + 1) * UTT_ITEM] for u in range(K)}
    return intent_blk, utter_blk


def _mean_w(bridge, pre_idx, post_idx):
    """Mean synaptic weight over the (all-to-all, dense) pre->post block. cp_connections[i,j]=weight i->j (CSR)."""
    M = bridge.cp_connections
    sub = M[np.asarray(pre_idx)][:, np.asarray(post_idx)]
    arr = sub.toarray() if hasattr(sub, "toarray") else to_host(sub)
    return float(np.asarray(arr).mean())


def _weight_table(bridge):
    """w[t][u] = mean intent[t]->utter[u] actor weight after training."""
    intent_blk, utter_blk = _region_blocks(bridge)
    return np.array([[_mean_w(bridge, intent_blk[t], utter_blk[u]) for u in range(K)] for t in range(K)])


def _sep(wtab, target):
    """mean_t ( w[t, target[t]] - mean_{u!=target[t]} w[t,u] )."""
    vals = []
    for t in range(K):
        tt = target[t]
        others = [wtab[t, u] for u in range(K) if u != tt]
        vals.append(float(wtab[t, tt] - np.mean(others)))
    return float(np.mean(vals))


def measure_success_table(seed):
    """s(t,u) is INVARIANT to actor training (success = belief AND intent coincidence; actor weights feed the
    CHOICE, not the success). Measure it once, deterministically. Returns S, aligned, succ_opt, succ_worst,
    belief_by_u."""
    belief_src = _belief_sources(seed)
    aligned = _aligned_utts(belief_src)
    belief_by_u = {ui: belief_src[u] for ui, u in enumerate(UTTS)}
    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    S = np.zeros((K, K))
    for t in range(K):
        for u in range(K):
            _commit_action(bridge, xp, idx, snap, t, u)
            S[t, u] = _evaluate_success(bridge, xp, idx, t, belief_by_u[u])
    succ_opt = {t: int(np.argmax(S[t])) for t in range(K)}
    succ_worst = {t: int(np.argmin(S[t])) for t in range(K)}
    return S, aligned, succ_opt, succ_worst, belief_by_u


def _commit_action_neural_critic(bridge, xp, idx, snap, intent_t, winner, crit_intent):
    """Neural-critic commit: drive intent[t] + executed utter[winner] + crit[INTENT] (state-value column), so the
    actor eligibility localizes to intent[t]->utter[winner] AND the critic eligibility localizes to the per-INTENT
    diagonal intent[t]->crit[t]. (v2's _commit drives crit[winner]; here we drive crit[intent] to make a STATE
    value.)"""
    bridge.cp_eligibility_trace[:] = 0.0
    if getattr(bridge, "cp_reward_coactivity_trace", None) is not None:
        bridge.cp_reward_coactivity_trace[:] = 0.0
    from research.runners._pragmatic_success_readback_leg2_v2_derisk import _restore_state  # local import
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(COMMIT_MS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[idx["intent"][intent_t]] = xp.float32(INTENT_PA)
        bridge.cp_external_input_current[idx["utter"][winner]] = xp.float32(COMMIT_PA)
        bridge.cp_external_input_current[idx["crit"][crit_intent]] = xp.float32(COMMIT_PA)
        bridge._run_one_simulation_step()


def train_seed(seed, n_train, arm, yoked_stream=None, neural_critic=False, verbose=True):
    """Train one arm. arm in {'fix','yoked'}. Returns metrics dict (+ '_a_stream' for building the yoked stream)."""
    t0 = time.time()
    S, aligned, succ_opt, succ_worst, belief_by_u = measure_success_table(seed)
    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    if not neural_critic:
        bridge.set_plasticity_gate(CRIT_GATE, 0.0)   # host-EMA path: the per-action critic is OFF (isolates the fix)

    untrained = _readout_policy(bridge, xp, idx, snap)
    untrained_wtab = _weight_table(bridge)

    rng = np.random.default_rng(seed * 71 + 13)
    Vctx = np.zeros(K)
    a_stream = []
    sign_hits, sign_n = 0, 0
    for i in range(n_train):
        t = int(rng.integers(K))
        greedy, _, Vread = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=neural_critic)
        winner = int(rng.integers(K)) if (rng.random() < EPSILON) else greedy
        if neural_critic:
            Vctx[t] = float(Vread[t]) * CRIT_READ_GAIN_V3          # NEURAL state value = rate(crit[intent])
            _commit_action_neural_critic(bridge, xp, idx, snap, t, winner, t)
        else:
            _commit_action(bridge, xp, idx, snap, t, winner)
        success = _evaluate_success(bridge, xp, idx, t, belief_by_u[winner])
        A = success - Vctx[t]                                     # SIGNED per-CONTEXT advantage (the fix)
        # leading indicator: sign(A) vs whether winner is the success-optimal (unambiguous best/worst only)
        if winner == succ_opt[t] or winner == succ_worst[t]:
            sign_n += 1
            if (A > 0) == (winner == succ_opt[t]):
                sign_hits += 1
        a_stream.append(REWARD_GAIN * A)
        rpe = float(yoked_stream[i % len(yoked_stream)]) if arm == "yoked" else REWARD_GAIN * A
        _deliver_reward(bridge, xp, rpe)
        if not neural_critic:
            Vctx[t] += EMA_BETA * (success - Vctx[t])             # host-EMA TD update of the state baseline

    trained = _readout_policy(bridge, xp, idx, snap)
    wtab = _weight_table(bridge)
    warg = {t: int(np.argmax(wtab[t])) for t in range(K)}

    def acc(choice, target):
        return float(np.mean([choice[t] == target[t] for t in range(K)]))

    m = {
        "seed": int(seed), "arm": arm, "neural_critic": bool(neural_critic),
        "aligned": {str(t): int(aligned[t]) for t in range(K)},
        "succ_opt": {str(t): int(succ_opt[t]) for t in range(K)},
        "trained_choice": {str(t): int(trained[t]) for t in range(K)},
        "weight_argmax": {str(t): int(warg[t]) for t in range(K)},
        # LEADING indicator
        "advantage_sign_acc": round(sign_hits / sign_n, 4) if sign_n else None,
        "advantage_sign_n": int(sign_n),
        # PRIMARY: actor-WTA acc + weight-argmax acc, vs the reward's OWN target (success-optimal)
        "actor_wta_acc_vs_succopt": round(acc(trained, succ_opt), 4),
        "weight_argmax_acc_vs_succopt": round(acc(warg, succ_opt), 4),
        # vs the pragmatic (belief-aligned) target -- reward-misspecification-bounded
        "actor_wta_acc_vs_aligned": round(acc(trained, aligned), 4),
        "untrained_wta_acc_vs_succopt": round(acc(untrained, succ_opt), 4),
        # PRIMARY: weight separation (isolates learning from readout), vs success-optimal, trained & untrained
        "weight_sep_vs_succopt": round(_sep(wtab, succ_opt), 5),
        "untrained_weight_sep_vs_succopt": round(_sep(untrained_wtab, succ_opt), 5),
        "weight_sep_vs_aligned": round(_sep(wtab, aligned), 5),
        "chance": round(1.0 / K, 4),
        "elapsed_seconds": round(time.time() - t0, 1),
        "_a_stream": a_stream,
    }
    if verbose:
        print(f"  [seed {seed} {arm}{'/neuralV' if neural_critic else ''}] ({m['elapsed_seconds']}s) "
              f"sign_acc={m['advantage_sign_acc']} "
              f"actorWTA(succopt)={m['actor_wta_acc_vs_succopt']} "
              f"wargmax(succopt)={m['weight_argmax_acc_vs_succopt']} "
              f"wsep(succopt)={m['weight_sep_vs_succopt']} (untr {m['untrained_weight_sep_vs_succopt']}) "
              f"actorWTA(aligned)={m['actor_wta_acc_vs_aligned']}", flush=True)
    return m


def evaluate_seed_v3(seed, n_train, neural_critic=False, verbose=True):
    fix = train_seed(seed, n_train, arm="fix", neural_critic=neural_critic, verbose=verbose)
    yoked_stream = np.array(fix["_a_stream"], dtype=float)
    np.random.default_rng(seed * 999 + 7).shuffle(yoked_stream)
    yok = train_seed(seed, n_train, arm="yoked", yoked_stream=yoked_stream,
                     neural_critic=neural_critic, verbose=verbose)
    for d in (fix, yok):
        d.pop("_a_stream", None)
    return {"seed": int(seed), "fix": fix, "yoked": yok}


def _agg(per_seed, arm, key):
    vals = [r[arm][key] for r in per_seed if r[arm].get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--neural-critic", action="store_true",
                    help="use the SPIKING per-intent state-value critic (secondary); default = host-EMA shortcut")
    ap.add_argument("--ema-beta", type=float, default=EMA_BETA,
                    help="host-EMA baseline tracking rate (faster -> better-centered advantage -> less yoked leak)")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/leg2_v3_statevalue.json")
    args = ap.parse_args()
    globals()["EMA_BETA"] = float(args.ema_beta)

    seeds = args.seeds if args.seeds is not None else [args.seed]
    n_train = min(args.n_train, 120) if args.smoke else args.n_train
    critic_kind = "NEURAL spiking per-intent state-value critic" if args.neural_critic \
        else "host-EMA success-per-intent baseline (FLAGGED SHORTCUT; actor is fully neural)"
    print(f"[leg2-v3] STATE-VALUE baseline fix | critic={critic_kind} | seeds={seeds} n_train={n_train}", flush=True)

    per_seed = [evaluate_seed_v3(s, n_train, neural_critic=args.neural_critic, verbose=True) for s in seeds]

    agg = {
        "critic": critic_kind,
        "fix_mean_advantage_sign_acc": _agg(per_seed, "fix", "advantage_sign_acc"),
        "fix_mean_actor_wta_vs_succopt": _agg(per_seed, "fix", "actor_wta_acc_vs_succopt"),
        "fix_mean_weight_argmax_vs_succopt": _agg(per_seed, "fix", "weight_argmax_acc_vs_succopt"),
        "fix_mean_weight_sep_vs_succopt": _agg(per_seed, "fix", "weight_sep_vs_succopt"),
        "fix_mean_untrained_weight_sep": _agg(per_seed, "fix", "untrained_weight_sep_vs_succopt"),
        "fix_mean_actor_wta_vs_aligned": _agg(per_seed, "fix", "actor_wta_acc_vs_aligned"),
        "yoked_mean_actor_wta_vs_succopt": _agg(per_seed, "yoked", "actor_wta_acc_vs_succopt"),
        "yoked_mean_weight_argmax_vs_succopt": _agg(per_seed, "yoked", "weight_argmax_acc_vs_succopt"),
        "yoked_mean_weight_sep_vs_succopt": _agg(per_seed, "yoked", "weight_sep_vs_succopt"),
        "chance": round(1.0 / K, 4),
    }
    # contingency verdict: fix learns (WTA & weight above chance/untrained) AND yoked does NOT.
    fix_wta = agg["fix_mean_actor_wta_vs_succopt"] or 0.0
    fix_warg = agg["fix_mean_weight_argmax_vs_succopt"] or 0.0
    yok_wta = agg["yoked_mean_actor_wta_vs_succopt"] or 0.0
    yok_warg = agg["yoked_mean_weight_argmax_vs_succopt"] or 0.0
    fix_sep = agg["fix_mean_weight_sep_vs_succopt"] or 0.0
    yok_sep = agg["yoked_mean_weight_sep_vs_succopt"] or 0.0
    from tools.lab import attributable_to
    # attribute the learned weight separation: FIX (reward contingent on action) vs YOKED (reward decoupled).
    # A non-contingent source (heterogeneity/DC/lagging-baseline) would move YOKED equally; contingency = the gap.
    attributable_to("state-value critic weight separation: FIX (contingent) vs YOKED (reward-decoupled)",
                    fix_sep, yok_sep)
    contingency = bool(fix_warg >= 0.60 and fix_warg - yok_warg >= 0.20 and fix_sep - yok_sep > 0
                       and abs(yok_sep) < max(0.25 * fix_sep, 1e-3))
    agg["contingency_pass"] = contingency

    print("\n" + "=" * 100, flush=True)
    print(f"[leg2-v3] critic = {critic_kind}", flush=True)
    print(f"[leg2-v3] FIX   : sign_acc={agg['fix_mean_advantage_sign_acc']} "
          f"actorWTA(succopt)={agg['fix_mean_actor_wta_vs_succopt']} "
          f"wargmax(succopt)={agg['fix_mean_weight_argmax_vs_succopt']} "
          f"wsep(succopt)={agg['fix_mean_weight_sep_vs_succopt']} (untr {agg['fix_mean_untrained_weight_sep']}) "
          f"actorWTA(aligned)={agg['fix_mean_actor_wta_vs_aligned']}", flush=True)
    print(f"[leg2-v3] YOKED : actorWTA(succopt)={agg['yoked_mean_actor_wta_vs_succopt']} "
          f"wargmax(succopt)={agg['yoked_mean_weight_argmax_vs_succopt']} "
          f"wsep(succopt)={agg['yoked_mean_weight_sep_vs_succopt']}", flush=True)
    print(f"[leg2-v3] chance={agg['chance']}  CONTINGENCY_PASS={contingency}", flush=True)
    print("=" * 100, flush=True)

    out = {"runner": "_pragmatic_readback_leg2_v3_statevalue_derisk",
           "seeds": list(seeds), "n_train": n_train, "aggregate": agg, "per_seed": per_seed}
    Path(os.path.dirname(os.path.abspath(args.json))).mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[leg2-v3] wrote {args.json}", flush=True)
    return 0 if contingency else 1


if __name__ == "__main__":
    raise SystemExit(main())
