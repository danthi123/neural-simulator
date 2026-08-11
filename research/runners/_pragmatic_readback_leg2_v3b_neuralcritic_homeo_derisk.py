"""D (Stage-4 CONVERSANT) · PRAGMATICS -- LEG 2 v3b: CALIBRATE the fully-NEURAL per-intent state-value critic with a
HOMEOSTATIC readout gain, so the fully-brain-based critic recovers the host-EMA's decision quality on the runner's OWN
strict contingency gate (not just the directional wsep read).

WHY (the un-tried lever, per THE LAW -- a negative is an undiscovered mechanism):
  v3's --neural-critic path (landed 2026-08-10, commit 2fdfc3e54) is a GENUINE spiking critic: crit[intent] is a
  spiking population, Vctx[t]=rate(crit[t]) read from spikes, trained by DA-gated plasticity on the intent[t]->crit[t]
  diagonal with the advantage A=success-Vctx (simultaneously the critic's TD error -- a clean advantage actor-critic).
  It passes the LOOSE directional read (fix wsep>0 & yoked~0) on 6/6, but on the runner's STRICT contingency_pass gate
  (fix weight-argmax>=0.60 AND fix-yoked>=0.20 AND |yoked_sep| < 0.25*fix_sep) it passes only 3/6 (s42,s100,s101;
  s43/s44/s102 fail on a residual YOKED LEAK). Root cause: the critic READOUT SCALE is fixed by the constant
  CRIT_READ_GAIN_V3=1.0 ("calibration-sensitive" -- v3 docstring). If rate(crit) cannot reach the success scale under
  the read drive, E[A] stays >0 (the baseline under-predicts), and a net-positive DA non-contingently potentiates the
  heterogeneity-favored assembly -> the yoked arm leaks -> strict gate fails. This is EXACTLY the lagging-baseline leak
  that beta-centering killed for the HOST EMA (2026-08-10), but the neural critic has no beta -- its scale is a CONSTANT.

THE COMPANION HOMEOSTATIC PROCESS WE REPLACED WITH A CONSTANT (the wall-reframe):
  The real VTA/critic readout is gain-controlled to the reward set-point (synaptic scaling / intrinsic-excitability
  homeostasis toward a target output). We proxied that slow gain-control with the fixed CRIT_READ_GAIN_V3=1.0. This
  runner restores it: a scalar readout gain g = succ_bar / crit_bar, where succ_bar (EMA of the NEURAL success rate =
  the reward set-point) and crit_bar (EMA of the raw NEURAL critic rate) are running means. Then Vctx[t] = g*rate(crit[t]).
  This forces E[Vctx] -> E[success] (E[A]->0, leak killed) WHILE PRESERVING the per-context differentiation (Vctx[t]
  still varies with rate(crit[t]) = the LEARNED per-intent value). The VALUE (which intent is worth more) stays 100%
  neural in the intent->crit weights; only the global readout SCALE is homeostatically calibrated -- a companion
  homeostatic process, not a value shortcut.

DISTINCT from prior homeostat levers: _pragmatic_readback_leg2_v2_homeostat_derisk (threshold homeostat on the
  utterance-ASSEMBLY readout CV -- REFUTED, a readout-SNR problem on the v2/host line) and _wta_afferent_winner_homeostat
  (common-mode remover for the WTA). Neither calibrates the CRITIC readout gain to the reward scale. This is the
  named next conversion for the neural critic.

TWO CHANGES ONLY vs v3 (everything else reused by import; NO sim/ edit; neural critic ALWAYS on):
  1. Vctx[t] = g_t * rate(crit[t])   with g_t = succ_bar / max(crit_bar, EPS)  (homeostatic readout gain)
  2. after each trial: crit_bar += hb*(raw - crit_bar);  succ_bar += hb*(success - succ_bar)   (running set-points)

The DECISIVE CONTROL is unchanged: YOKED (same DA-magnitude distribution, DECOUPLED from the action). The homeostat is
present IDENTICALLY in both arms (it is part of the critic readout, not the reward), so any fix-vs-yoked separation is
attributable to reward CONTINGENCY, not to the calibration.

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3b_neuralcritic_homeo_derisk --smoke --seed 44
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3b_neuralcritic_homeo_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_6seed.json
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

# Reuse the ENTIRE v3 + v2 machinery by import. NO sim/ edit; additive.
from research.runners._pragmatic_readback_leg2_v3_statevalue_derisk import (  # noqa: E402
    measure_success_table, _region_blocks, _weight_table, _sep, _agg,
    _commit_action_neural_critic,
)
from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    build_speaker_bridge, _choose_utterance, _evaluate_success, _readout_policy, _deliver_reward,
    K, REWARD_GAIN, EPSILON, N_TRAIN,
)

# ── v3b homeostat knobs ──────────────────────────────────────────────────────────────────────────────────────────
HOMEO_BETA = 0.10   # running-mean rate for crit_bar / succ_bar (the readout-gain calibration timescale)
EPS_HOMEO = 1e-4    # floor on crit_bar to avoid div-by-zero before the critic warms up


def train_seed_homeo(seed, n_train, arm, yoked_stream=None, homeo_beta=HOMEO_BETA, verbose=True):
    """Train one arm with the NEURAL critic + HOMEOSTATIC readout-gain calibration. arm in {'fix','yoked'}.
    Returns metrics dict (+ '_a_stream' for building the yoked stream)."""
    t0 = time.time()
    S, aligned, succ_opt, succ_worst, belief_by_u = measure_success_table(seed)
    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    # neural critic path: CRIT_GATE stays 1.0 (set in build_speaker_bridge) -> the critic LEARNS. Do NOT disable it.

    untrained = _readout_policy(bridge, xp, idx, snap)
    untrained_wtab = _weight_table(bridge)

    rng = np.random.default_rng(seed * 71 + 13)
    Vctx = np.zeros(K)
    crit_bar = 0.0          # running mean of the RAW neural critic rate (homeostat state)
    succ_bar = 0.0          # running mean of the NEURAL success rate (reward set-point)
    warm = False
    g_first = None
    g_last = 1.0
    a_stream = []
    sign_hits, sign_n = 0, 0
    for i in range(n_train):
        t = int(rng.integers(K))
        greedy, _, Vread = _choose_utterance(bridge, xp, idx, snap, t, explore_rng=None, read_crit=True)
        winner = int(rng.integers(K)) if (rng.random() < EPSILON) else greedy
        raw = float(Vread[t])                                    # NEURAL critic rate for intent t (pre-gain)
        # HOMEOSTATIC readout gain: calibrate E[Vctx] -> E[success]. Fallback g=1.0 until the running means warm.
        g = (succ_bar / crit_bar) if (warm and crit_bar > EPS_HOMEO) else 1.0
        if g_first is None:
            g_first = g
        g_last = g
        Vctx[t] = g * raw                                       # calibrated NEURAL state value
        _commit_action_neural_critic(bridge, xp, idx, snap, t, winner, t)
        success = _evaluate_success(bridge, xp, idx, t, belief_by_u[winner])
        A = success - Vctx[t]                                    # SIGNED per-CONTEXT advantage (the fix)
        # leading indicator: sign(A) vs whether winner is the success-optimal (unambiguous best/worst only)
        if winner == succ_opt[t] or winner == succ_worst[t]:
            sign_n += 1
            if (A > 0) == (winner == succ_opt[t]):
                sign_hits += 1
        a_stream.append(REWARD_GAIN * A)
        rpe = float(yoked_stream[i % len(yoked_stream)]) if arm == "yoked" else REWARD_GAIN * A
        _deliver_reward(bridge, xp, rpe)
        # update the homeostat set-points AFTER the trial (part of the critic readout, present in BOTH arms)
        crit_bar += homeo_beta * (raw - crit_bar)
        succ_bar += homeo_beta * (success - succ_bar)
        warm = True

    trained = _readout_policy(bridge, xp, idx, snap)
    wtab = _weight_table(bridge)
    warg = {t: int(np.argmax(wtab[t])) for t in range(K)}

    def acc(choice, target):
        return float(np.mean([choice[t] == target[t] for t in range(K)]))

    m = {
        "seed": int(seed), "arm": arm, "neural_critic": True, "homeo": True,
        "homeo_gain_first": round(float(g_first), 4) if g_first is not None else None,
        "homeo_gain_last": round(float(g_last), 4),
        "crit_bar_final": round(float(crit_bar), 5), "succ_bar_final": round(float(succ_bar), 5),
        "aligned": {str(t): int(aligned[t]) for t in range(K)},
        "succ_opt": {str(t): int(succ_opt[t]) for t in range(K)},
        "trained_choice": {str(t): int(trained[t]) for t in range(K)},
        "weight_argmax": {str(t): int(warg[t]) for t in range(K)},
        "advantage_sign_acc": round(sign_hits / sign_n, 4) if sign_n else None,
        "advantage_sign_n": int(sign_n),
        "actor_wta_acc_vs_succopt": round(acc(trained, succ_opt), 4),
        "weight_argmax_acc_vs_succopt": round(acc(warg, succ_opt), 4),
        "actor_wta_acc_vs_aligned": round(acc(trained, aligned), 4),
        "untrained_wta_acc_vs_succopt": round(acc(untrained, succ_opt), 4),
        "weight_sep_vs_succopt": round(_sep(wtab, succ_opt), 5),
        "untrained_weight_sep_vs_succopt": round(_sep(untrained_wtab, succ_opt), 5),
        "weight_sep_vs_aligned": round(_sep(wtab, aligned), 5),
        "chance": round(1.0 / K, 4),
        "elapsed_seconds": round(time.time() - t0, 1),
        "_a_stream": a_stream,
    }
    if verbose:
        print(f"  [seed {seed} {arm}/homeoV] ({m['elapsed_seconds']}s) g:{m['homeo_gain_first']}->{m['homeo_gain_last']} "
              f"sign_acc={m['advantage_sign_acc']} "
              f"actorWTA(succopt)={m['actor_wta_acc_vs_succopt']} "
              f"wargmax(succopt)={m['weight_argmax_acc_vs_succopt']} "
              f"wsep(succopt)={m['weight_sep_vs_succopt']} (untr {m['untrained_weight_sep_vs_succopt']})", flush=True)
    return m


def evaluate_seed_v3b(seed, n_train, homeo_beta=HOMEO_BETA, verbose=True):
    fix = train_seed_homeo(seed, n_train, arm="fix", homeo_beta=homeo_beta, verbose=verbose)
    yoked_stream = np.array(fix["_a_stream"], dtype=float)
    np.random.default_rng(seed * 999 + 7).shuffle(yoked_stream)
    yok = train_seed_homeo(seed, n_train, arm="yoked", yoked_stream=yoked_stream,
                           homeo_beta=homeo_beta, verbose=verbose)
    for d in (fix, yok):
        d.pop("_a_stream", None)
    return {"seed": int(seed), "fix": fix, "yoked": yok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--homeo-beta", type=float, default=HOMEO_BETA,
                    help="running-mean rate for the critic readout-gain calibration (succ_bar/crit_bar)")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo.json")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    n_train = min(args.n_train, 180) if args.smoke else args.n_train
    print(f"[leg2-v3b] NEURAL critic + HOMEOSTATIC readout gain (g=succ_bar/crit_bar) | "
          f"homeo_beta={args.homeo_beta} | seeds={seeds} n_train={n_train}", flush=True)

    per_seed = [evaluate_seed_v3b(s, n_train, homeo_beta=args.homeo_beta, verbose=True) for s in seeds]

    agg = {
        "critic": "NEURAL spiking per-intent state-value critic + HOMEOSTATIC readout-gain calibration",
        "homeo_beta": float(args.homeo_beta),
        "fix_mean_advantage_sign_acc": _agg(per_seed, "fix", "advantage_sign_acc"),
        "fix_mean_actor_wta_vs_succopt": _agg(per_seed, "fix", "actor_wta_acc_vs_succopt"),
        "fix_mean_weight_argmax_vs_succopt": _agg(per_seed, "fix", "weight_argmax_acc_vs_succopt"),
        "fix_mean_weight_sep_vs_succopt": _agg(per_seed, "fix", "weight_sep_vs_succopt"),
        "fix_mean_untrained_weight_sep": _agg(per_seed, "fix", "untrained_weight_sep_vs_succopt"),
        "fix_mean_actor_wta_vs_aligned": _agg(per_seed, "fix", "actor_wta_acc_vs_aligned"),
        "fix_mean_homeo_gain_last": _agg(per_seed, "fix", "homeo_gain_last"),
        "yoked_mean_actor_wta_vs_succopt": _agg(per_seed, "yoked", "actor_wta_acc_vs_succopt"),
        "yoked_mean_weight_argmax_vs_succopt": _agg(per_seed, "yoked", "weight_argmax_acc_vs_succopt"),
        "yoked_mean_weight_sep_vs_succopt": _agg(per_seed, "yoked", "weight_sep_vs_succopt"),
        "chance": round(1.0 / K, 4),
    }
    # per-seed strict contingency verdict (SAME gate as v3), so we can report x/6 directly.
    def _seed_pass(r):
        f, y = r["fix"], r["yoked"]
        fw = f["weight_argmax_acc_vs_succopt"]; yw = y["weight_argmax_acc_vs_succopt"]
        fs = f["weight_sep_vs_succopt"]; ys = y["weight_sep_vs_succopt"]
        return bool(fw >= 0.60 and fw - yw >= 0.20 and fs - ys > 0 and abs(ys) < max(0.25 * fs, 1e-3))
    per_seed_pass = {int(r["seed"]): _seed_pass(r) for r in per_seed}
    agg["per_seed_contingency_pass"] = per_seed_pass
    agg["n_contingent"] = int(sum(per_seed_pass.values()))

    fix_warg = agg["fix_mean_weight_argmax_vs_succopt"] or 0.0
    yok_warg = agg["yoked_mean_weight_argmax_vs_succopt"] or 0.0
    fix_sep = agg["fix_mean_weight_sep_vs_succopt"] or 0.0
    yok_sep = agg["yoked_mean_weight_sep_vs_succopt"] or 0.0

    from tools.lab import attributable_to, lever
    # LEVER: the homeostatic readout gain must actually move off the fixed 1.0 (else v3b == v3 and the A/B is void).
    lever("critic_readout_gain(1.0->homeo)", 1.0, agg["fix_mean_homeo_gain_last"], required=False)
    # attribute the learned weight separation to reward CONTINGENCY: FIX vs YOKED.
    attributable_to("neural-critic(homeo) weight separation: FIX (contingent) vs YOKED (reward-decoupled)",
                    fix_sep, yok_sep)
    contingency = bool(fix_warg >= 0.60 and fix_warg - yok_warg >= 0.20 and fix_sep - yok_sep > 0
                       and abs(yok_sep) < max(0.25 * fix_sep, 1e-3))
    agg["contingency_pass_aggregate"] = contingency

    print("\n" + "=" * 100, flush=True)
    print(f"[leg2-v3b] critic = {agg['critic']}", flush=True)
    print(f"[leg2-v3b] FIX   : g_last={agg['fix_mean_homeo_gain_last']} sign_acc={agg['fix_mean_advantage_sign_acc']} "
          f"actorWTA(succopt)={agg['fix_mean_actor_wta_vs_succopt']} "
          f"wargmax(succopt)={agg['fix_mean_weight_argmax_vs_succopt']} "
          f"wsep(succopt)={agg['fix_mean_weight_sep_vs_succopt']} (untr {agg['fix_mean_untrained_weight_sep']})", flush=True)
    print(f"[leg2-v3b] YOKED : actorWTA(succopt)={agg['yoked_mean_actor_wta_vs_succopt']} "
          f"wargmax(succopt)={agg['yoked_mean_weight_argmax_vs_succopt']} "
          f"wsep(succopt)={agg['yoked_mean_weight_sep_vs_succopt']}", flush=True)
    print(f"[leg2-v3b] per-seed strict contingency: {per_seed_pass}  => {agg['n_contingent']}/{len(per_seed)}", flush=True)
    print(f"[leg2-v3b] chance={agg['chance']}  CONTINGENCY_PASS(agg)={contingency}", flush=True)
    print("=" * 100, flush=True)

    out = {"runner": "_pragmatic_readback_leg2_v3b_neuralcritic_homeo_derisk",
           "seeds": list(seeds), "n_train": n_train, "aggregate": agg, "per_seed": per_seed}
    Path(os.path.dirname(os.path.abspath(args.json))).mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[leg2-v3b] wrote {args.json}", flush=True)
    return 0 if contingency else 1


if __name__ == "__main__":
    raise SystemExit(main())
