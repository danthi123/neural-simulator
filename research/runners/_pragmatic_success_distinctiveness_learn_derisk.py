"""D · PRAGMATICS -- STEP 2: plug the PRAGMATIC (distinctiveness) success reward into the v3 state-value learner.

KEY QUESTION 2: with the distinctiveness reward s_prag(t,u)=S[t,u]-mean_{t'!=t}S[t',u], does the learned spiking
speaker say the belief-ALIGNED utterance more often (actor-WTA / weight-argmax vs aligned rises above the ~0.44 plain
cap and above chance 0.333) WHILE the yoked/shuffled-reward contingency control still FAILS?

CONTROLS (mandatory):
  (a) PLAIN reward (S[t,u]) reproduces the ~0.44 aligned cap in THIS harness (baseline sound).
  (b) YOKED (shuffled reward stream) must NOT converge under the pragmatic reward (contingency preserved).
  (c) the pragmatic succ_opt is not a constant utterance (degeneracy) -- checked in STEP 1.

The success table S[t,u] is DETERMINISTIC and INVARIANT to actor training (frozen coincidence detector; no OU) -- so
the reward is a table lookup on the precomputed S. Plain and pragmatic arms use the SAME S table, isolating the
REWARD-SHAPE effect from per-trial measurement noise. Actor = fully neural spiking soft-WTA (v3 machinery, DA-gated
eligibility). Vctx = per-context host-EMA state-value baseline (the v3 fix; actor stays neural). NO sim/ edit.
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

from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    build_speaker_bridge, _choose_utterance, _commit_action, _deliver_reward, _readout_policy,
    K, EPSILON, REWARD_GAIN, CRIT_GATE,
)
from research.runners._pragmatic_readback_leg2_v3_statevalue_derisk import _weight_table, _sep  # noqa: E402
from research.runners._pragmatic_success_distinctiveness_derisk import (  # noqa: E402
    measure_full_tables, distinctiveness_table,
)
from tools.lab import attributable_to  # noqa: E402

EMA_BETA = 0.40   # faster-centered baseline (v3 finding: kills the lagging-baseline yoked leak)


def _argmax_map(tab_t_u):
    return {t: int(np.argmax(tab_t_u[t])) for t in range(K)}


def train_arm(seed, R, target, n_train, arm, yoked_stream=None, verbose=True):
    """Train the spiking actor with a table reward R[t][u] (t=intent, u=utterance). arm in {'fix','yoked'}.
    Returns metrics + '_a_stream'."""
    t0 = time.time()
    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    bridge.set_plasticity_gate(CRIT_GATE, 0.0)          # host-EMA state-value path (per v3): per-action critic OFF
    untrained = _readout_policy(bridge, xp, idx, snap)
    untr_wtab = _weight_table(bridge)

    rng = np.random.default_rng(seed * 71 + 13)
    Vctx = np.zeros(K)
    a_stream = []
    for i in range(n_train):
        t = int(rng.integers(K))
        greedy, _, _ = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=False)
        winner = int(rng.integers(K)) if (rng.random() < EPSILON) else greedy
        _commit_action(bridge, xp, idx, snap, t, winner)  # action-localized eligibility
        reward = float(R[t, winner])
        A = reward - Vctx[t]
        a_stream.append(REWARD_GAIN * A)
        rpe = float(yoked_stream[i % len(yoked_stream)]) if arm == "yoked" else REWARD_GAIN * A
        _deliver_reward(bridge, xp, rpe)
        Vctx[t] += EMA_BETA * (reward - Vctx[t])

    trained = _readout_policy(bridge, xp, idx, snap)
    wtab = _weight_table(bridge)
    warg = {t: int(np.argmax(wtab[t])) for t in range(K)}

    def acc(choice, tgt):
        return float(np.mean([choice[t] == tgt[t] for t in range(K)]))

    m = {
        "seed": int(seed), "arm": arm,
        "trained_choice": {str(t): int(trained[t]) for t in range(K)},
        "weight_argmax": {str(t): int(warg[t]) for t in range(K)},
        "actor_wta_acc_vs_target": round(acc(trained, target), 4),
        "weight_argmax_acc_vs_target": round(acc(warg, target), 4),
        "weight_sep_vs_target": round(_sep(wtab, target), 5),
        "untrained_weight_sep_vs_target": round(_sep(untr_wtab, target), 5),
        "chance": round(1.0 / K, 4),
        "elapsed_seconds": round(time.time() - t0, 1),
        "_a_stream": a_stream,
    }
    if verbose:
        print(f"  [seed {seed} {arm}] ({m['elapsed_seconds']}s) "
              f"wargmax(tgt)={m['weight_argmax_acc_vs_target']} wsep(tgt)={m['weight_sep_vs_target']} "
              f"(untr {m['untrained_weight_sep_vs_target']}) warg={m['weight_argmax']}", flush=True)
    return m


def evaluate_seed(seed, n_train, verbose=True):
    belief, S, aligned, _ = measure_full_tables(seed)
    Sp = distinctiveness_table(S)
    succ_opt_plain = _argmax_map(S)
    succ_opt_prag = _argmax_map(Sp)

    out = {"seed": int(seed),
           "aligned": {str(t): int(aligned[t]) for t in range(K)},
           "succ_opt_plain": {str(t): int(succ_opt_plain[t]) for t in range(K)},
           "succ_opt_prag": {str(t): int(succ_opt_prag[t]) for t in range(K)}}

    def run(R, name, own_target):
        fix = train_arm(seed, R, own_target, n_train, "fix", verbose=verbose)
        yoked_stream = np.array(fix["_a_stream"], dtype=float)
        np.random.default_rng(seed * 999 + 7).shuffle(yoked_stream)
        yok = train_arm(seed, R, own_target, n_train, "yoked", yoked_stream=yoked_stream, verbose=verbose)
        # score BOTH arms against BOTH targets (own reward target + the aligned/pragmatic target)
        def acc_vs(choice_map, tgt):
            return round(float(np.mean([choice_map[str(t)] == tgt[t] for t in range(K)])), 4)
        res = {
            "reward": name,
            "fix_weight_argmax_vs_own": fix["weight_argmax_acc_vs_target"],
            "fix_weight_argmax_vs_aligned": acc_vs(fix["weight_argmax"], aligned),
            "fix_actor_wta_vs_aligned": acc_vs(fix["trained_choice"], aligned),
            "fix_weight_sep_vs_own": fix["weight_sep_vs_target"],
            "yoked_weight_argmax_vs_own": yok["weight_argmax_acc_vs_target"],
            "yoked_weight_sep_vs_own": yok["weight_sep_vs_target"],
            "untrained_weight_sep_vs_own": fix["untrained_weight_sep_vs_target"],
            # contingency: the CONTINGENT arm learns its target AND the reward-DECOUPLED (yoked) arm does NOT
            # (yoked wargmax at/below chance, and its separation toward the target is far below the fix arm's).
            "contingency_pass": bool(fix["weight_argmax_acc_vs_target"] >= 0.60
                                     and yok["weight_argmax_acc_vs_target"] <= 0.40
                                     and fix["weight_sep_vs_target"] - yok["weight_sep_vs_target"] > 0.05),
        }
        # ATTRIBUTION: what fraction of the FIX arm's weight-separation-to-target is NOT reproduced by the
        # reward-DECOUPLED yoked arm — i.e. is genuinely reward-contingent (near 1.0 = clean contingency, the
        # separation is driven by the reward stream, not by an arm-independent proxy). This is the load-bearing
        # contingency read, made a subtraction rather than two numbers sitting one key apart.
        res["contingency_attributable_fraction"] = attributable_to(
            "[%s] weight-separation-to-target: FIX (reward-contingent) vs YOKED (reward-decoupled)" % name,
            fix["weight_sep_vs_target"], yok["weight_sep_vs_target"])
        return res

    out["PLAIN"] = run(S, "plain(S)", succ_opt_plain)
    out["PRAG"] = run(Sp, "pragmatic(distinctiveness)", succ_opt_prag)
    if verbose:
        p, q = out["PLAIN"], out["PRAG"]
        print(f"  >>> seed {seed}: PLAIN aligned={p['fix_weight_argmax_vs_aligned']} (contingency {p['contingency_pass']}) "
              f"| PRAG aligned={q['fix_weight_argmax_vs_aligned']} (contingency {q['contingency_pass']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 44])
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--ema-beta", type=float, default=EMA_BETA)
    ap.add_argument("--json", type=str, default="research/findings/raw/_pragmatic_success/distinctiveness_learn.json")
    args = ap.parse_args()
    globals()["EMA_BETA"] = float(args.ema_beta)

    t0 = time.time()
    per_seed = [evaluate_seed(s, args.n_train) for s in args.seeds]

    def mean(key_path):
        vals = []
        for r in per_seed:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        return round(float(np.mean(vals)), 4)

    agg = {
        "n_seeds": len(args.seeds), "n_train": args.n_train, "ema_beta": args.ema_beta,
        "PLAIN_fix_aligned": mean(["PLAIN", "fix_weight_argmax_vs_aligned"]),
        "PRAG_fix_aligned": mean(["PRAG", "fix_weight_argmax_vs_aligned"]),
        "PLAIN_fix_own": mean(["PLAIN", "fix_weight_argmax_vs_own"]),
        "PRAG_fix_own": mean(["PRAG", "fix_weight_argmax_vs_own"]),
        "PLAIN_all_contingent": all(r["PLAIN"]["contingency_pass"] for r in per_seed),
        "PRAG_all_contingent": all(r["PRAG"]["contingency_pass"] for r in per_seed),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print("\n" + "=" * 90, flush=True)
    print(f"[step2] PLAIN reward -> weight_argmax vs ALIGNED = {agg['PLAIN_fix_aligned']} "
          f"(learns its own target {agg['PLAIN_fix_own']}; contingent={agg['PLAIN_all_contingent']})", flush=True)
    print(f"[step2] PRAG  reward -> weight_argmax vs ALIGNED = {agg['PRAG_fix_aligned']} "
          f"(learns its own target {agg['PRAG_fix_own']}; contingent={agg['PRAG_all_contingent']})", flush=True)
    print(f"[step2] chance=0.333 ; prior plain aligned cap ~0.44", flush=True)
    out = {"runner": Path(__file__).stem, "seeds": args.seeds, "aggregate": agg, "per_seed": per_seed}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1, default=str))
    print(f"[step2] wrote {args.json} ({agg['elapsed_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
