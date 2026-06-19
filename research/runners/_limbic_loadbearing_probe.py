"""Limbic-core load-bearing diagnostic on a HIDDEN-GOAL (Morris-water-maze
analogue) task (2026-06-19).

THE QUESTION: is the reward/value/dopamine limbic core BEHAVIORALLY LOAD-BEARING
when the goal is NOT directly perceivable? On the standard visible/orient-solvable
gridworld the limbic core is GREEN_INERT (validated but inert) because the
heuristic/SC orienting can navigate WITHOUT reward. Here the goal's coordinates
never enter the brain (--hidden-goal zeroes the ppc_goal_input goal drive; the
own-position place drive stays), and the heuristic teacher is OFF, so the ONLY
goal-related signal is the SCALAR reward. The agent must learn the goal location
via reward -> value -> dopamine -> corticostriatal STDP.

3 conditions, single static goal, grid-8:
  (iii) control_visible    : heuristic ON, goal visible  -> harness/agent sanity (low score)
  (i)   hidden_reward_ON    : goal hidden, heuristic OFF, reward ON  -> limbic core must learn
  (ii)  hidden_reward_OFF   : (i) + reward LESIONED       -> the load-bearing test

Metric: sum over phases of final_quarter_mean_distance (Manhattan; LOWER = better).
A single static goal => 1 phase. We also report mean_distance overall and
n_steps_at_goal for context.

VERDICT LOGIC (owner standard validate_signal_by_its_function — the lesion must
collapse the BEHAVIOR, not merely the firing):
  - if (i) << (iii)+slack solves AND (ii) is MUCH WORSE than (i) => reward LOAD-BEARING.
  - if (ii) ~= (i) (agent solves hidden goal even with reward off) OR (i) ~= random
    (fails even with reward on) => NOT reward-load-bearing (honest negative).

Run:
  SIM_BACKEND=numpy python -X utf8 -m research.runners._limbic_loadbearing_probe \
      --seeds 42 --n-steps 600 --grid-size 8 [--with-critic] [--out <json>]
NO sim/ edit; uses the additive default-OFF run_moving_goal_episode params.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from research.runners.g11_bg_runner import run_moving_goal_episode


def _score(result):
    ps = result["phase_stats"]
    sum_finalq = float(sum(p["final_quarter_mean_distance"] for p in ps))
    mean_dist = float(np.mean(result["distance_log"][1:])) if len(result["distance_log"]) > 1 else float("nan")
    n_at_goal = int(sum(p["n_steps_at_goal"] for p in ps))
    n_steps = int(sum(p["n_steps"] for p in ps))
    return {
        "sum_finalQ": round(sum_finalq, 3),
        "mean_distance": round(mean_dist, 3),
        "n_steps_at_goal": n_at_goal,
        "frac_at_goal": round(n_at_goal / max(1, n_steps), 3),
        "n_phases": len(ps),
    }


def run_condition(name, seed, n_steps, grid_size, with_critic, **overrides):
    kw = dict(
        seed=seed,
        n_steps=n_steps,
        grid_size=grid_size,
        start_pos=(1, 1),
        goal_pos=(6, 6),            # static single goal (1 phase)
        enable_hippocampus=True,    # place code (own (x,y)) -> cortex, plastic
        enable_bg_lateral_inhibition=True,  # MSN cross-pool WTA (--enable-msn-lateral-inhibition; flagship default)
        verbose=False,
    )
    if with_critic:
        # The fuller spiking limbic core (value baseline + spiking SNc + spiking
        # reward delivery). Heavier; opt-in. The reward-STDP corticostriatal
        # learner is present regardless; the critic adds the value baseline.
        kw.update(
            enable_neural_critic=True,
            spiking_snc=True,
            spiking_reward_us=True,
        )
    kw.update(overrides)
    out = os.path.join("research/findings/raw", f"_limbic_{name}_seed{seed}.json")
    kw["out_path"] = out
    t0 = time.time()
    res = run_moving_goal_episode(**kw)
    sc = _score(res)
    sc["wall_s"] = round(time.time() - t0, 1)
    sc["condition"] = name
    sc["seed"] = seed
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--with-critic", action="store_true",
                    help="Add the fuller spiking limbic core (neural critic + spiking SNc + "
                         "spiking reward delivery). Heavier; for the GPU confirm.")
    ap.add_argument("--out", type=str, default="research/findings/raw/_limbic_loadbearing_summary.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    conditions = [
        # name,                  overrides
        ("control_visible",      dict(heuristic_strength=1.0)),
        ("hidden_reward_ON",     dict(heuristic_strength=0.0, hidden_goal=True)),
        ("hidden_reward_OFF",    dict(heuristic_strength=0.0, hidden_goal=True, lesion_reward=True)),
    ]

    rows = []
    for seed in seeds:
        for name, ov in conditions:
            sc = run_condition(name, seed, args.n_steps, args.grid_size, args.with_critic, **ov)
            rows.append(sc)
            # Single clean RESULT line per condition (sentinel for grepping past bridge logs).
            sys.stderr.write(
                f"PROBE_RESULT seed={seed} cond={name:18s} sum_finalQ={sc['sum_finalQ']:.3f} "
                f"mean_dist={sc['mean_distance']:.3f} frac_at_goal={sc['frac_at_goal']:.3f} "
                f"wall_s={sc['wall_s']}\n")
            sys.stderr.flush()

    # Aggregate per condition across seeds.
    agg = {}
    for name, _ in conditions:
        vals = [r["sum_finalQ"] for r in rows if r["condition"] == name]
        fa = [r["frac_at_goal"] for r in rows if r["condition"] == name]
        agg[name] = {
            "sum_finalQ_mean": round(float(np.mean(vals)), 3),
            "sum_finalQ_std": round(float(np.std(vals)), 3),
            "frac_at_goal_mean": round(float(np.mean(fa)), 3),
            "n_seeds": len(vals),
        }

    summary = {
        "task": "hidden_goal_morris_water_maze",
        "grid_size": args.grid_size,
        "n_steps": args.n_steps,
        "with_critic": bool(args.with_critic),
        "seeds": seeds,
        "rows": rows,
        "aggregate": agg,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write("PROBE_AGGREGATE " + json.dumps(agg) + "\n")
    sys.stderr.write("PROBE_WROTE " + args.out + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
