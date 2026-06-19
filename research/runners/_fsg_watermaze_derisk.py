"""Fremaux-Sprekeler-Gerstner spiking actor-critic, TRIAL-STRUCTURED hidden-goal de-risk
(2026-06-19).

STEP 3 -- the FINAL point-neuron rigor step before the dendrite is proposed for the
actor-critic credit-assignment wall.

THE DE-RISK QUESTION (the ONLY thing this tests): does the F-S-G (2013) spiking
actor-critic form the hidden-goal place->action map on the POINT-NEURON substrate when
given its PROPER TRIAL-STRUCTURED protocol -- MANY reset trials at the SAME hidden goal,
learned weights persisting across trials -- AND with the structural cascade SYMMETRIZED
(so the no-reward baseline sits at the random-walk floor, not a fixed-drift corner)?

This is the LITERATURE-VALIDATED setting (the water-maze learns the place->action map
over many reset trials). The earlier probe (`_advantage_actor_critic_probe.py` +
`2026-06-19-spiking-actor-critic-advantage-routing-derisk.md`) ran ONE long static phase
and was PRELIMINARY NEGATIVE; it explicitly flagged the missing trial structure +
symmetrization as the remaining point-neuron rigor. If the actor STILL fails here, the
point-neuron path is exhausted and the dendrite (apical-basal credit assignment) is the
clearly-proposed obvious unlocker.

WHAT IS ALREADY ROUTED (verified by code-read in the Step-2 finding): with
`enable_neural_critic + spiking_snc + spiking_reward_us` the actor's three-factor signal
IS the ADVANTAGE delta = r - V(place) (the SNc fires r minus the striosome_value critic's
V(place); the `dopamine` modulator deviation = the signed plasticity third factor). This
de-risk USES that path -- it adds only the TRIAL STRUCTURE (the new additive default-OFF
`trial_reset_steps` runner param: reset to a random start every K steps, weights persist).

THE LOAD-BEARING PROOF (owner standard validate_signal_by_its_function): across >=2
away-from-drift goals (distinct learned policies),
  - reward ON  : the per-trial-final-distance LEARNING CURVE DECREASES (late trials end
                 closer to the goal) AND converges near the goal;
  - reward OFF (lesioned) : the curve stays FLAT at the random-walk floor, goal-INDEPENDENT
                 (this is also the SYMMETRIZATION guard -- a goal-dependent lesion curve
                 means a residual structural drift confounds the reward-ON result).
A fixed-corner drift that happens to equal one goal is NOT learning -> the >=2-goal
distinct-policy requirement + the symmetric lesion baseline guard against it.

Run (GPU/cupy only -- the moving-goal path imports cupy directly):
  python -X utf8 -m research.runners._fsg_watermaze_derisk \
      --seed 42 --goals "1,6;6,1" --n-trials 40 --steps-per-trial 200 --grid-size 8

NO sim/ edit. Uses the additive default-OFF `trial_reset_steps` / `trial_reset_seed`
runner params (this commit) + the pre-existing hidden_goal / lesion_reward / advantage
limbic core (enable_neural_critic / spiking_snc / spiking_reward_us).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from research.runners.g11_bg_runner import run_moving_goal_episode


def _trial_curve_stats(trial_dists, n_early, n_late):
    """Summarize a per-trial final-distance learning curve."""
    td = [float(d) for d in (trial_dists or [])]
    n = len(td)
    if n == 0:
        return {"n_trials": 0}
    n_early = min(n_early, n)
    n_late = min(n_late, n)
    early = td[:n_early]
    late = td[-n_late:]
    return {
        "n_trials": n,
        "trial_final_distances": [round(d, 2) for d in td],
        "early_mean": round(float(np.mean(early)), 3),
        "late_mean": round(float(np.mean(late)), 3),
        "delta_early_minus_late": round(float(np.mean(early) - np.mean(late)), 3),
        "best_trial": round(float(np.min(td)), 2),
        "overall_mean": round(float(np.mean(td)), 3),
    }


def run_condition(tag, seed, n_trials, steps_per_trial, grid_size, goal, lesion,
                  critic_warmup_trials, trial_reset_seed):
    """One trial-structured advantage-actor-critic run at a fixed hidden goal.

    The agent is reset to a fresh RANDOM start every `steps_per_trial` steps (the
    additive `trial_reset_steps` runner param) while the learned weights PERSIST across
    all `n_trials` trials -- the F-S-G water-maze training. Single static goal (1 phase);
    hidden_goal hides the goal coords from the brain (only the scalar reward conveys it).
    The full deployed advantage limbic core gives the actor delta = r - V(place) as the
    signed third factor (verified by code-read; the Step-2 finding).

    critic_warmup_trials > 0 seeds V(place) before nav via reward-paired drives at the
    goal -- the legitimate F-S-G value warm-up (all-neural LTP; only agent placement +
    reward delivery is scaffolding). NOT a goal cheat to the actor (only the critic's
    value, not the actor's goal-cell drive, is informed; the actor's goal drive stays
    hidden).
    """
    n_steps = int(n_trials * steps_per_trial)
    kw = dict(
        seed=seed,
        n_steps=n_steps,
        grid_size=grid_size,
        start_pos=(1, 1),
        goal_pos=tuple(goal),           # static single goal (1 phase)
        heuristic_strength=0.0,         # no goal-direction teacher (the agent MUST learn)
        hidden_goal=True,               # goal coords never enter the brain
        enable_hippocampus=True,        # the sparse selective place code -> cortex (the ACTOR substrate)
        enable_bg_lateral_inhibition=True,  # MSN cross-pool WTA (flagship default)
        # --- the ADVANTAGE-routed spiking actor-critic core (delta = r - V(place)) ---
        enable_neural_critic=True,      # striosome_value V(place); GABA_B subtracts V at the SNc
        spiking_snc=True,               # the dopamine modulator = SNc firing = the RPE (signed 3rd factor)
        spiking_reward_us=True,         # r delivered SYNAPTICALLY (reward_us -> SNc); whole delta neural
        critic_warmup_trials=int(critic_warmup_trials),  # seed V(place) before nav (F-S-G value warm-up)
        # --- the TRIAL STRUCTURE (this de-risk's addition) ---
        trial_reset_steps=int(steps_per_trial),  # reset to a random start every K steps; weights persist
        trial_reset_seed=int(trial_reset_seed),  # deterministic random-start sequence
        verbose=False,
    )
    if lesion:
        kw["lesion_reward"] = True      # the load-bearing anti-cheat + symmetrization guard
    out = os.path.join("research/findings/raw", f"_fsgwm_{tag}_seed{seed}.json")
    kw["out_path"] = out
    t0 = time.time()
    res = run_moving_goal_episode(**kw)
    n_early = max(1, n_trials // 4)
    n_late = max(1, n_trials // 4)
    sc = _trial_curve_stats(res.get("trial_final_distances", []), n_early, n_late)
    sc["tag"] = tag
    sc["seed"] = seed
    sc["goal"] = list(goal)
    sc["lesion"] = bool(lesion)
    sc["wall_s"] = round(time.time() - t0, 1)
    traj = res.get("trajectory", [])
    sc["end_pos"] = list(traj[-1]) if traj else None
    snc = res.get("snc_rate_log", []) or []
    striov = res.get("striov_rate_log", []) or []
    sc["snc_rate_mean"] = round(float(np.mean(snc)), 2) if snc else None
    sc["striov_rate_mean"] = round(float(np.mean(striov)), 2) if striov else None
    sc["critic_weight_initial"] = res.get("critic_weight_initial")
    sc["critic_weight_final"] = res.get("critic_weight_final")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goals", type=str, default="1,6;6,1",
                    help="';'-separated 'x,y' AWAY-FROM-DRIFT goals (>=2 distinct policies).")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--steps-per-trial", type=int, default=200)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--critic-warmup-trials", type=int, default=20,
                    help="Reward-paired drives at the goal to seed V(place) before nav (F-S-G "
                         "value warm-up). NOT a goal cheat to the actor.")
    ap.add_argument("--trial-reset-seed", type=int, default=12345)
    ap.add_argument("--random-floor", type=float, default=5.52,
                    help="The grid-8 random-walk reference floor (Step-1 Monte-Carlo).")
    ap.add_argument("--converge-thresh", type=float, default=2.5,
                    help="late_mean below this = 'converged near the goal'.")
    ap.add_argument("--learn-delta-thresh", type=float, default=1.0,
                    help="(early_mean - late_mean) above this = 'the curve decreased'.")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_fsg_watermaze_summary.json")
    args = ap.parse_args()

    goals = []
    for g in args.goals.split(";"):
        g = g.strip()
        if not g:
            continue
        xy = [int(v) for v in g.split(",")]
        goals.append((xy[0], xy[1]))

    rows = []
    for goal in goals:
        gtag = f"g{goal[0]}{goal[1]}_ON"
        sc_on = run_condition(gtag, args.seed, args.n_trials, args.steps_per_trial,
                              args.grid_size, goal, lesion=False,
                              critic_warmup_trials=args.critic_warmup_trials,
                              trial_reset_seed=args.trial_reset_seed)
        rows.append(sc_on)
        sys.stderr.write(
            f"FSGWM seed={args.seed} goal={goal} reward=ON  "
            f"early={sc_on.get('early_mean')} late={sc_on.get('late_mean')} "
            f"delta={sc_on.get('delta_early_minus_late')} best={sc_on.get('best_trial')} "
            f"end_pos={sc_on.get('end_pos')} snc={sc_on.get('snc_rate_mean')} "
            f"striov={sc_on.get('striov_rate_mean')} "
            f"cw={sc_on.get('critic_weight_initial')}->{sc_on.get('critic_weight_final')} "
            f"wall_s={sc_on.get('wall_s')}\n")
        sys.stderr.flush()

        ltag = f"g{goal[0]}{goal[1]}_OFF"
        sc_off = run_condition(ltag, args.seed, args.n_trials, args.steps_per_trial,
                               args.grid_size, goal, lesion=True,
                               critic_warmup_trials=args.critic_warmup_trials,
                               trial_reset_seed=args.trial_reset_seed)
        rows.append(sc_off)
        sys.stderr.write(
            f"FSGWM seed={args.seed} goal={goal} reward=OFF "
            f"early={sc_off.get('early_mean')} late={sc_off.get('late_mean')} "
            f"delta={sc_off.get('delta_early_minus_late')} best={sc_off.get('best_trial')} "
            f"end_pos={sc_off.get('end_pos')} wall_s={sc_off.get('wall_s')}\n")
        sys.stderr.flush()

    # ----- Verdict logic -----
    on_rows = [r for r in rows if not r["lesion"]]
    off_rows = [r for r in rows if r["lesion"]]
    # Reward-ON: per goal, did the curve DECREASE and CONVERGE near the goal?
    n_learned = sum(
        1 for r in on_rows
        if r.get("delta_early_minus_late", 0.0) >= args.learn_delta_thresh
        and r.get("late_mean", 99.0) <= args.converge_thresh
    )
    # Reward-ON beats its own lesion at the same goal (the load-bearing contrast)?
    off_by_goal = {tuple(r["goal"]): r for r in off_rows}
    n_on_beats_lesion = 0
    for r in on_rows:
        lr = off_by_goal.get(tuple(r["goal"]))
        if lr is not None and r.get("late_mean", 99.0) <= lr.get("late_mean", 99.0) - 1.0:
            n_on_beats_lesion += 1
    # SYMMETRIZATION guard: the lesion curve is goal-INDEPENDENT (within ~1.0 across goals)
    # AND near the random floor (not a fixed-drift corner). A goal-dependent lesion =
    # residual structural drift = confounded result.
    off_late = [r.get("late_mean") for r in off_rows if r.get("late_mean") is not None]
    lesion_goal_independent = (max(off_late) - min(off_late) <= 1.5) if len(off_late) >= 2 else None
    lesion_at_floor = (float(np.mean(off_late)) >= args.random_floor - 1.5) if off_late else None

    point_neuron_go = (
        n_learned >= max(2, len(goals))
        and n_on_beats_lesion >= max(2, len(goals))
    )
    verdict = {
        "n_goals": len(goals),
        "n_goals_learned_and_converged": n_learned,
        "n_goals_on_beats_lesion": n_on_beats_lesion,
        "lesion_goal_independent": lesion_goal_independent,
        "lesion_at_random_floor": lesion_at_floor,
        "lesion_late_means": [round(float(x), 2) for x in off_late] if off_late else [],
        "random_floor": args.random_floor,
        "converge_thresh": args.converge_thresh,
        "learn_delta_thresh": args.learn_delta_thresh,
        "POINT_NEURON_GO": bool(point_neuron_go),
        "verdict": (
            "POINT_NEURON_GO (F-S-G actor-critic learns hidden-goal place->action with trial "
            "structure -> limbic core LOAD-BEARING, no dendrite needed for nav)"
            if point_neuron_go else
            "NEGATIVE (point-neuron path EXHAUSTED -> the dendrite / apical-basal credit "
            "assignment is the clearly-proposed obvious unlocker)"
        ),
    }
    summary = {
        "task": "hidden_goal_fsg_watermaze_trial_structured",
        "grid_size": args.grid_size,
        "n_trials": args.n_trials,
        "steps_per_trial": args.steps_per_trial,
        "goals": [list(g) for g in goals],
        "seed": args.seed,
        "critic_warmup_trials": int(args.critic_warmup_trials),
        "core": "neural_critic + spiking_snc + spiking_reward_us (advantage delta=r-V) + trial_reset",
        "rows": rows,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write("FSGWM_VERDICT " + json.dumps(verdict) + "\n")
    sys.stderr.write("FSGWM_WROTE " + args.out + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
