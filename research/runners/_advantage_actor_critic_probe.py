"""Spiking actor-critic ADVANTAGE-routing de-risk on the HIDDEN-GOAL task (2026-06-19).

STEP 2 of the limbic-core arc. Step 1 (`_limbic_loadbearing_probe.py` +
`2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md`) found the spiking
reward/value/dopamine limbic core is NOT behaviorally load-bearing on a hidden goal:
the lesion does not collapse navigation, and the agent drifts to a FIXED corner
regardless of goal location (a structural cascade bias, NOT reward-driven learning).
Diagnosed mechanism: the place->action map never forms because raw global reward-STDP
does not overcome the cascade's random-init directional bias (the 2026-05-05
"global scalar feedback fails at biological scale" family).

THE DE-RISK QUESTION (the ONLY thing this tests): does the Fremaux-Sprekeler-Gerstner
(2013) spiking actor-critic recipe -- routing the ADVANTAGE delta = r - V(place) (the
already-deployed spiking-SNc RPE with the neural value critic ON) as the actor's third
factor instead of raw reward -- let the actor LEARN the hidden goal's location on the
point-neuron substrate? Advantage r-V is a far better credit signal than raw reward (it
is ~0 once V predicts r), and is the canonical F-S-G water-maze fix.

KEY VERIFICATION (done by reading the code, see the findings doc): with
`--spiking-snc --enable-neural-critic` the actor IS ALREADY advantage-gated -- the
`dopamine` modulator concentration (from_region_firing_signed over the SNc) becomes
the SIGNED `effective_signal` in the bridge's three-factor weight update
(`Delta_w = lr * effective_signal * eligibility`, bridge.py:6904/6952), and the SNc
fires r (reward_us excitation) minus V(place) (the striosome_value critic's GABA_B
subtraction at the membrane). So `effective_signal ~= delta = r - V(place)` = the
advantage. The actor's eligibility is therefore advantage-gated, not raw-reward-gated,
whenever the neural critic is on. This probe USES that path -- no new mechanism; the
existing flags compose. The Step-1 confounds addressed here:
  (a) sparse selective place code -- `sensor_place_readout` (sigma=0.5 => 1-3 cells/
      position, per-position preferred grid) is already selective. [present]
  (b) a SINGLE long goal-stable phase -- one static goal, long n_steps (no multi-goal).
  (c) the structural-bias confound -- reported via the goal-location anti-cheat (the
      agent must TRACK the goal across >=3 locations; a fixed-corner drift that happens
      to equal one goal is NOT tracking).

LOAD-BEARING TEST (owner standard validate_signal_by_its_function -- the lesion must
collapse the BEHAVIOR across goal locations, not merely the SNc firing):
  - advantage-routed actor + reward ON  must TRACK the goal across >=3 locations
    (end-position near each goal; sum_finalQ below the random floor at each).
  - reward OFF (lesioned)  must COLLAPSE to the random-walk floor (~5.52),
    goal-independent.
That contrast IS the load-bearing proof.

Reference floor (uniform random-cardinal walk, grid-8, start (1,1)):
  final_quarter_mean_distance ~= 5.52 (Step-1 Monte-Carlo). Perfect nav ~= 0.

Run (GPU/cupy only -- the moving-goal runner imports cupy directly; numpy incompatible):
  python -X utf8 -m research.runners._advantage_actor_critic_probe \
      --seeds 42 --n-steps 1500 --grid-size 8 \
      --goals "6,6;1,6;6,1" [--also-lesion] [--out <json>]

NO sim/ edit; uses the additive default-OFF run_moving_goal_episode params
(hidden_goal, lesion_reward) from Step 1 + the deployed neural-critic / spiking-SNc /
spiking-reward-us limbic core. The advantage routing is pre-existing -- this is a
configuration de-risk, not a code change.
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
    dl = result.get("distance_log", [])
    mean_dist = float(np.mean(dl[1:])) if len(dl) > 1 else float("nan")
    n_at_goal = int(sum(p["n_steps_at_goal"] for p in ps))
    n_steps = int(sum(p["n_steps"] for p in ps))
    traj = result.get("trajectory", [])
    end_pos = list(traj[-1]) if traj else None
    snc = result.get("snc_rate_log", []) or []
    striov = result.get("striov_rate_log", []) or []
    return {
        "sum_finalQ": round(sum_finalq, 3),
        "mean_distance": round(mean_dist, 3),
        "n_steps_at_goal": n_at_goal,
        "frac_at_goal": round(n_at_goal / max(1, n_steps), 3),
        "end_pos": end_pos,
        "snc_rate_mean": round(float(np.mean(snc)), 2) if snc else None,
        "striov_rate_mean": round(float(np.mean(striov)), 2) if striov else None,
        "n_phases": len(ps),
    }


def _end_dist_from_goal(end_pos, goal):
    if end_pos is None:
        return None
    return int(abs(end_pos[0] - goal[0]) + abs(end_pos[1] - goal[1]))


def run_condition(tag, seed, n_steps, grid_size, goal, lesion):
    """One advantage-routed actor-critic episode at a fixed hidden goal.

    The full deployed spiking limbic core (neural value critic + spiking SNc RPE +
    spiking reward delivery) gives the actor the ADVANTAGE delta=r-V(place) as its
    signed third factor (verified by code-read; see module docstring). Single static
    goal => 1 long phase (lever b). hidden_goal=True hides the goal coords (lever a's
    setting); the own-position place drive stays.
    """
    kw = dict(
        seed=seed,
        n_steps=n_steps,
        grid_size=grid_size,
        start_pos=(1, 1),
        goal_pos=tuple(goal),       # static single goal (1 phase)
        heuristic_strength=0.0,     # no goal-direction teacher (the agent MUST learn)
        hidden_goal=True,           # goal coords never enter the brain
        enable_hippocampus=True,    # the sparse selective place code -> cortex (the ACTOR substrate)
        enable_bg_lateral_inhibition=True,  # MSN cross-pool WTA (flagship default)
        # --- the ADVANTAGE-routed spiking actor-critic core ---
        enable_neural_critic=True,  # striosome_value V(place); its GABA_B subtracts V at the SNc membrane
        spiking_snc=True,           # the dopamine modulator = SNc firing = the RPE (signed 3rd factor)
        spiking_reward_us=True,     # r delivered SYNAPTICALLY (reward_us -> SNc); whole delta=r-V neural
        verbose=False,
    )
    if lesion:
        kw["lesion_reward"] = True  # the load-bearing anti-cheat: clamp reward=0
    out = os.path.join("research/findings/raw", f"_advac_{tag}_seed{seed}.json")
    kw["out_path"] = out
    t0 = time.time()
    res = run_moving_goal_episode(**kw)
    sc = _score(res)
    sc["wall_s"] = round(time.time() - t0, 1)
    sc["tag"] = tag
    sc["seed"] = seed
    sc["goal"] = list(goal)
    sc["lesion"] = bool(lesion)
    sc["end_dist_from_goal"] = _end_dist_from_goal(sc["end_pos"], goal)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-steps", type=int, default=1500)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--goals", type=str, default="6,6;1,6;6,1",
                    help="';'-separated 'x,y' goal locations for the anti-cheat (>=3).")
    ap.add_argument("--also-lesion", action="store_true",
                    help="Also run the reward-LESIONED condition at each goal (the load-bearing "
                         "contrast). Doubles wall-clock; recommended for the decisive smoke.")
    ap.add_argument("--random-floor", type=float, default=5.52,
                    help="The random-walk reference floor (grid-8 default 5.52 from Step-1 MC).")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_advantage_actor_critic_summary.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    goals = []
    for g in args.goals.split(";"):
        g = g.strip()
        if not g:
            continue
        xy = [int(v) for v in g.split(",")]
        goals.append((xy[0], xy[1]))

    rows = []
    for seed in seeds:
        for goal in goals:
            gtag = f"g{goal[0]}{goal[1]}_rewardON"
            sc = run_condition(gtag, seed, args.n_steps, args.grid_size, goal, lesion=False)
            rows.append(sc)
            sys.stderr.write(
                f"PROBE_RESULT seed={seed} goal={goal} reward=ON  "
                f"sum_finalQ={sc['sum_finalQ']:.3f} end_pos={sc['end_pos']} "
                f"end_dist_from_goal={sc['end_dist_from_goal']} "
                f"frac_at_goal={sc['frac_at_goal']:.3f} snc={sc['snc_rate_mean']} "
                f"striov={sc['striov_rate_mean']} wall_s={sc['wall_s']}\n")
            sys.stderr.flush()
            if args.also_lesion:
                ltag = f"g{goal[0]}{goal[1]}_rewardOFF"
                scl = run_condition(ltag, seed, args.n_steps, args.grid_size, goal, lesion=True)
                rows.append(scl)
                sys.stderr.write(
                    f"PROBE_RESULT seed={seed} goal={goal} reward=OFF "
                    f"sum_finalQ={scl['sum_finalQ']:.3f} end_pos={scl['end_pos']} "
                    f"end_dist_from_goal={scl['end_dist_from_goal']} "
                    f"frac_at_goal={scl['frac_at_goal']:.3f} wall_s={scl['wall_s']}\n")
                sys.stderr.flush()

    # Verdict logic: tracking = reward-ON end-pos is NEAR the goal (<=2 Manhattan) AND
    # below floor, at MOST/ALL goal locations; lesion (if run) at/above floor + goal-
    # independent. The anti-cheat: a FIXED end position across goals is NOT tracking.
    on_rows = [r for r in rows if not r["lesion"]]
    end_positions = [tuple(r["end_pos"]) for r in on_rows if r["end_pos"] is not None]
    distinct_end_positions = len(set(end_positions))
    n_tracking = sum(
        1 for r in on_rows
        if r["end_dist_from_goal"] is not None and r["end_dist_from_goal"] <= 2
        and r["sum_finalQ"] < args.random_floor
    )
    verdict = {
        "n_goals": len(goals),
        "n_reward_on_runs": len(on_rows),
        "n_tracking_goals": n_tracking,
        "distinct_end_positions": distinct_end_positions,
        "random_floor": args.random_floor,
        "tracks_across_locations": n_tracking >= max(2, len(goals)),
        "anti_cheat_distinct_ends": distinct_end_positions >= max(2, len(goals)),
    }
    summary = {
        "task": "hidden_goal_advantage_actor_critic",
        "grid_size": args.grid_size,
        "n_steps": args.n_steps,
        "goals": [list(g) for g in goals],
        "seeds": seeds,
        "core": "neural_critic + spiking_snc + spiking_reward_us (advantage delta=r-V(place) routed)",
        "rows": rows,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write("PROBE_VERDICT " + json.dumps(verdict) + "\n")
    sys.stderr.write("PROBE_WROTE " + args.out + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
