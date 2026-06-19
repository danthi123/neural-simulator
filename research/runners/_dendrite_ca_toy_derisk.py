"""STAGE-1 dendrite credit-assignment de-risk (CPU/numpy, NO bridge, NO GPU,
NO sim/ edit). Reuse-by-import of sim/dendritic_neuron.py +
sim/dendritic_plasticity.py.

THE QUESTION (owner-approved cheap-first gate, 2026-06-19): the point-neuron
spiking actor-critic does NOT learn a hidden-goal place->action map (multiple
rigorous NEGATIVES). Does an APICAL-BASAL two-compartment ("dendrite") actor
crack it on a rate-level numpy gridworld actor-critic TOY, where the
point-neuron form fails on the identical setup?

THE MECHANISM (per research/findings/2026-06-19-dendrite-credit-assignment-
derisk-scoping.md):
  - BASAL compartment integrates the bottom-up PLACE code (one-hot per cell;
    held FIXED + SELECTIVE -> the #5 place-selectivity confound is excluded by
    construction).
  - APICAL compartment integrates the top-down ADVANTAGE delta = r - V(place)
    (a learned tabular value baseline V), projected through a FIXED-RANDOM
    apical weight B_apical (feedback alignment; NO weight transport).
  - The apical-driven burst GATES the plasticity so the place->action weight on
    the TAKEN action changes proportional to (apical-burst x signed-advantage):
    Delta_w ~ pre(place) . burst(apical delta) -- place-AND-advantage-specific
    credit (Payeur-Naud-Richards 2021; Guerguiev-Lillicrap-Richards 2017).

THE POINT-NEURON CONTROL = the SAME toy with the standard global
delta x eligibility three-factor rule (no apical gating). With a structural
directional bias it reproduces the documented fixed-corner-drift / no-learning
failure.

CRITICAL PRECEDENT (research/findings/2026-05-17-dendritic-credit-assignment-
NEGATIVE.md): a prior dendritic-credit-assignment de-risk was an honest NEGATIVE
(the local rule did not do hidden credit assignment in a W2-frozen supervised
isolation test). This STAGE-1 tests a DIFFERENT, RL-specific question (the
genuine advantage teaching signal). Be skeptical of a positive: the
point-neuron control MUST genuinely fail (fair baseline), and BOTH the
apical-lesion AND the wrong-sign controls MUST fail, or a GO is void/confounded.

ASCII only. Pure numpy. Reuse-by-import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Reuse-by-import (NO sim/ edit). The apical/basal split + the fixed-random
# apical feedback (NO weight transport) + the apical-gated local rule.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from sim.dendritic_neuron import DendriticLayer  # noqa: E402
# The apical-gated local rule is also reused as a cross-check arm
# ("dendrite_us"): the project's EXISTING Urbanczik-Senn rule applied verbatim
# to the RL advantage (this is the rule the 2026-05-17 NEGATIVE tested; we run
# it alongside the steelman burst-dependent form so the result is faithful to
# the existing machinery, not just a hand-rolled variant).
from sim.dendritic_plasticity import urbanczik_senn_update  # noqa: E402


# ----------------------------------------------------------------------------
# Environment (HOST -- legitimate: the world's state + the body acting on the
# motor output; NO cognition here).
# ----------------------------------------------------------------------------
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E (dy, dx)
N_ACTIONS = 4


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_env(pos, action, grid):
    dy, dx = ACTIONS[action]
    ny = _clip(pos[0] + dy, 0, grid - 1)
    nx = _clip(pos[1] + dx, 0, grid - 1)
    return (ny, nx)


def random_start(rng, grid, goal, min_dist=3):
    """Random start cell, Manhattan >= min_dist from the goal (so the curve has
    room to decrease)."""
    for _ in range(1000):
        p = (int(rng.integers(0, grid)), int(rng.integers(0, grid)))
        if manhattan(p, goal) >= min_dist:
            return p
    return (0, 0)


# ----------------------------------------------------------------------------
# Place code (held FIXED + SELECTIVE -- one-hot per cell. This is the #5
# place-selectivity confound EXCLUDED BY CONSTRUCTION: the place code is
# perfectly selective and never learned, so any result is attributable to the
# credit-assignment RULE, not the place input).
# ----------------------------------------------------------------------------
def place_code(pos, grid):
    n = grid * grid
    v = np.zeros(n, dtype=float)
    v[pos[0] * grid + pos[1]] = 1.0
    return v


# ----------------------------------------------------------------------------
# Softmax action selection (shared by both arms; the policy read-out is a host
# argmax/sample over the actor pre-activation -- analogous to the spiking-WTA
# read-out, which the project treats as brain-based-compliant; the COGNITION
# being tested is the LEARNING RULE that shapes the place->action weights).
# ----------------------------------------------------------------------------
def softmax(z, temp=1.0):
    z = np.asarray(z, float) / max(temp, 1e-6)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ----------------------------------------------------------------------------
# The structural directional bias = a PRIOR IN THE WEIGHTS (a "cascade bias"),
# NOT a persistent additive term at selection. Every place row's actor weights
# are initialised with a directional prior toward the NW corner (N + W up; S + E
# down). NW is NOT either test goal ((1,6) NE-ish, (6,1) SW-ish), so the bias is
# goal-NEUTRAL (drives AWAY from BOTH goals). Because it lives in the WEIGHTS,
# learning CAN overwrite it -- so an ideal learner (the oracle tabular-Q arm)
# converges, while the global three-factor rule cannot reshape the biased
# weights fast enough (the documented credit-assignment failure). Calibrated
# (bias_mag ~2.0) so the oracle SUCCEEDS but the point control FAILS (the
# mandatory two-sided anti-cheat: a fair baseline that genuinely fails AND a
# difficulty bound that proves the task is learnable; scoping section 3.3 + 5.1).
# ----------------------------------------------------------------------------
def init_actor_weights(rng, n_place, bias_mag):
    W = rng.normal(0.0, 0.01, (n_place, N_ACTIONS))
    # additive directional PRIOR toward NW in every place row (N=0, S=1, W=2,
    # E=3): N and W up, S and E down.
    W[:, 0] += bias_mag
    W[:, 2] += bias_mag
    W[:, 1] -= bias_mag
    W[:, 3] -= bias_mag
    return W


# ----------------------------------------------------------------------------
# ONE actor-critic arm. mode in {"point", "dendrite"}.
#   point    : global three-factor   Delta_w ~ (delta) x eligibility(pre x taken)
#   dendrite : apical-gated          Delta_w ~ burst(apical delta) x pre, sign
#              from the signed advantage on the taken action (no weight
#              transport; the advantage enters via the fixed-random B_apical).
# apical_lesion : zero the advantage into the apical compartment (delta -> 0).
# wrong_sign    : flip the advantage sign into the apical gate.
# ----------------------------------------------------------------------------
def run_arm(mode, seed, goal, grid, n_trials, steps_per_trial,
            bias_mag, actor_lr, value_lr, temp, gamma,
            apical_lesion=False, wrong_sign=False, rule="burst"):
    rng = np.random.default_rng(seed)
    n_place = grid * grid

    # W_actor carries the structural NW prior; learning must reshape it.
    W_actor = init_actor_weights(rng, n_place, bias_mag)
    V = np.zeros(n_place, dtype=float)  # tabular value baseline V(place)
    # mode == "oracle": tabular-Q (W_actor IS Q, max-bootstrap). The
    # DIFFICULTY-BOUND positive control -- an ideal value-iteration learner that
    # MUST solve the task (else the bias is uninformatively too strong).

    # The dendritic layer (reuse of sim/dendritic_neuron.py): basal = place
    # code; apical teacher = the advantage broadcast (dim 1) -> N_ACTIONS via the
    # FIXED-RANDOM B_apical (feedback alignment, no weight transport). Used for
    # its BAC apical-depol burst-gate machinery; the actor pre-activation for
    # SELECTION is W_actor (so all arms select identically) -- the dendrite
    # differs ONLY in how the apical advantage GATES the plasticity.
    dend = DendriticLayer(n_pre=n_place, n_post=N_ACTIONS, n_teacher=1,
                          seed=seed + 777, theta_high=1.0, apical_gain=0.5,
                          leak=0.0)

    def potential_reward(pos, new_pos, at_goal):
        # Potential-based shaping (Ng 1999): reward = reduction in Manhattan
        # distance + a goal bonus. HOST environment reward (legitimate -- the
        # world scores the body's move; the brain computes V + the advantage).
        return (manhattan(pos, goal) - manhattan(new_pos, goal)) \
            + (5.0 if at_goal else 0.0)

    trial_final_distances = []

    for _t in range(n_trials):
        pos = random_start(rng, grid, goal)
        # one episode (trial); weights PERSIST across trials.
        for _s in range(steps_per_trial):
            pc = place_code(pos, grid)
            idx = pos[0] * grid + pos[1]

            pre_act = pc @ W_actor                  # actor pre-activation
            probs = softmax(pre_act, temp=temp)
            a = int(rng.choice(N_ACTIONS, p=probs))

            new_pos = step_env(pos, a, grid)
            new_idx = new_pos[0] * grid + new_pos[1]

            at_goal = (new_pos == goal)
            r = potential_reward(pos, new_pos, at_goal)

            # Critic: TD value learning of V(place). The advantage the actor
            # uses is the TD error delta = r + gamma V(s') - V(s).
            v_s = V[idx]
            v_sp = 0.0 if at_goal else V[new_idx]
            td = r + gamma * v_sp - v_s
            V[idx] = v_s + value_lr * td
            delta = td  # the advantage teaching signal

            # ----- the LEARNING RULE (the cognition under test) -----
            if mode == "oracle":
                # Tabular-Q value iteration (off-policy max bootstrap). NOT a
                # biological rule -- the difficulty-bound positive control.
                q_target = r + (0.0 if at_goal
                                else gamma * float(np.max(W_actor[new_idx])))
                W_actor[idx, a] += value_lr * (q_target - W_actor[idx, a])

            elif mode == "point":
                # Global three-factor: the SAME scalar delta multiplies the
                # eligibility (pre x taken-action one-hot). No place/action
                # specificity beyond the eligibility trace -- the documented
                # global-scalar credit-assignment limit.
                elig = np.zeros((n_place, N_ACTIONS))
                elig[:, a] = pc
                W_actor += actor_lr * delta * elig

            elif mode == "dendrite":
                # Apical-gated BURST-DEPENDENT plasticity (Payeur-Naud-Richards
                # 2021; Larkum BAC). The advantage delta is the apical teaching
                # signal, projected through the FIXED-RANDOM B_apical (feedback
                # alignment; NO weight transport). The apical-driven Ca2+
                # plateau MAGNITUDE (|B_apical-projected drive|, reusing the
                # DendriticLayer BAC machinery) is the per-action BURST that
                # GATES the plasticity; the SIGN of the advantage sets LTP vs
                # LTD (dopamine's third-factor sign -- NOT scrambled by the
                # random feedback sign; only the GATE magnitude rides B_apical).
                #
                # Honest note: for a SINGLE trainable layer (the actor) there
                # are no hidden units to assign credit to, so feedback
                # alignment has nothing to align -- the apical compartment's
                # role here is the BURST GATE (place-AND-advantage coincidence),
                # not hidden-layer credit routing. That gate is a per-action
                # |delta|-scaled gain on the update; it does NOT add input/
                # action specificity beyond what the place pre-code + the taken
                # -action eligibility already give. This is exactly the form the
                # scoping flagged to test explicitly; whether the burst gate
                # lets the dendrite overcome the structural bias where the
                # global rule cannot is the empirical question.
                adv = 0.0 if apical_lesion else delta
                if wrong_sign:
                    adv = -adv
                teacher = np.array([adv], dtype=float)  # (n_teacher=1,)
                # The BURST gate per post-unit: the BAC Ca2+ plateau magnitude
                # (>= 0). Reuses DendriticLayer._apical_depol = |B_apical @
                # teacher|. Zero teacher (lesion) => zero burst => zero
                # plasticity (the gate is load-bearing).
                burst = dend._apical_depol(teacher)          # (N_ACTIONS,)
                # The eligibility (place pre-code on the TAKEN action only).
                elig = np.zeros((n_place, N_ACTIONS))
                elig[:, a] = pc

                if rule == "burst":
                    # Steelman burst-dependent three-factor (Payeur 2021):
                    # Delta_w ~ sign(delta) x burst x eligibility. The
                    # advantage's SIGN sets potentiation vs depression; the
                    # apical burst GATES the magnitude.
                    sgn = np.sign(adv)
                    W_actor += actor_lr * sgn * (burst[None, :] * elig)
                elif rule == "us":
                    # The project's EXISTING apical-gated Urbanczik-Senn rule
                    # applied verbatim to the RL advantage (the rule the
                    # 2026-05-17 NEGATIVE tested). The advantage is the
                    # apical_signal, projected through the FIXED-RANDOM B_apical
                    # by DendriticLayer (so its per-action sign rides feedback
                    # alignment, NOT delta directly). apical_gate = burst.
                    apical_signed = dend._apical_drive(teacher)  # (N_ACTIONS,)
                    dw = urbanczik_senn_update(
                        pre_rate=pc, soma_rate=probs, v_basal=pre_act,
                        apical_gate=burst, apical_signal=apical_signed,
                        lr=actor_lr)                              # (n_place, N)
                    # dw is already place-specific (outer with the one-hot pc
                    # row); restrict to the TAKEN-action column (the body's move).
                    col_mask = np.zeros(N_ACTIONS)
                    col_mask[a] = 1.0
                    W_actor += dw * col_mask[None, :]
                else:
                    raise ValueError(rule)
            else:
                raise ValueError(mode)

            pos = new_pos
            if at_goal:
                break

        trial_final_distances.append(manhattan(pos, goal))

    # ---- GREEDY EVAL of the LEARNED policy (separates learning from softmax
    # exploration noise: deterministic argmax over the actor pre-activation,
    # n_eval fresh trials). This is the honest "did the rule learn a
    # goal-directed policy" read-out. ----
    n_eval = 12
    eval_dists = []
    eval_final = (0, 0)
    for _e in range(n_eval):
        pos = random_start(rng, grid, goal)
        for _s in range(steps_per_trial):
            idx = pos[0] * grid + pos[1]
            a = int(np.argmax(W_actor[idx]))
            pos = step_env(pos, a, grid)
            if pos == goal:
                break
        eval_dists.append(manhattan(pos, goal))
        eval_final = pos

    return {
        "trial_final_distances": [int(d) for d in trial_final_distances],
        "final_pos": [int(eval_final[0]), int(eval_final[1])],
        "eval_dists": [int(d) for d in eval_dists],
        "eval_mean": float(np.mean(eval_dists)),
    }


def curve_stats(td, eval_mean=None, n_early=12, n_late=12, random_floor=5.52,
                converge=2.5, learn_delta=1.0):
    td = np.asarray(td, float)
    if td.size == 0:
        return {}
    n_early = min(n_early, td.size)
    n_late = min(n_late, td.size)
    early = float(np.mean(td[:n_early]))
    late = float(np.mean(td[-n_late:]))
    # The LEARNED-POLICY convergence is judged on the GREEDY eval (eval_mean) --
    # the clean read-out free of softmax-exploration noise. The training curve's
    # early->late drop is the supporting "the curve decreased" narrative.
    em = float(eval_mean) if eval_mean is not None else late
    return {
        "early_mean": round(early, 3),
        "late_mean": round(late, 3),
        "eval_mean": round(em, 3),
        "delta_early_minus_late": round(early - late, 3),
        "decreased": (early - late) >= learn_delta,
        "converged": em <= converge,
        # learned_and_converged := the GREEDY learned policy reaches near the
        # goal (eval_mean <= converge). (The training-curve decrease is reported
        # but the convergence gate is the clean greedy read-out.)
        "learned_and_converged": em <= converge,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--goals", type=str, default="1,6;6,1")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--steps-per-trial", type=int, default=25)
    ap.add_argument("--bias-mag", type=float, default=2.0,
                    help="Structural weight-init bias toward NW (a learnable "
                         "prior in the weights). Calibrated (~2.0) so the ORACLE "
                         "tabular-Q SUCCEEDS but the point control FAILS.")
    ap.add_argument("--actor-lr", type=float, default=0.2)
    ap.add_argument("--value-lr", type=float, default=0.4)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--random-floor", type=float, default=5.5)
    ap.add_argument("--converge-thresh", type=float, default=2.5)
    ap.add_argument("--learn-delta-thresh", type=float, default=1.0)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_dendrite_ca_toy_summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    goals = []
    for g in args.goals.split(";"):
        if g.strip():
            y, x = g.split(",")
            goals.append((int(y), int(x)))

    def stats(td, eval_mean):
        return curve_stats(td, eval_mean=eval_mean,
                           random_floor=args.random_floor,
                           converge=args.converge_thresh,
                           learn_delta=args.learn_delta_thresh)

    results = {"config": vars(args), "goals": [list(g) for g in goals],
               "seeds": seeds, "per_seed": []}

    # arms:
    #   point             -- the point-neuron control (global three-factor)
    #   dendrite          -- the STEELMAN burst-dependent apical-gated test arm
    #   dendrite_lesion   -- apical gate lesioned (advantage -> 0): must collapse
    #   dendrite_wrongsign-- advantage sign flipped: must fail (it's the advantage)
    #   dendrite_us       -- the project's EXISTING Urbanczik-Senn rule (the
    #                        2026-05-17 rule) applied to the RL advantage, verbatim
    #   oracle            -- tabular-Q value iteration (difficulty bound: MUST
    #                        succeed, else the bias is uninformatively too strong)
    arm_specs = [
        ("oracle", dict(mode="oracle")),
        ("point", dict(mode="point")),
        ("dendrite", dict(mode="dendrite", rule="burst")),
        ("dendrite_lesion", dict(mode="dendrite", rule="burst",
                                 apical_lesion=True)),
        ("dendrite_wrongsign", dict(mode="dendrite", rule="burst",
                                    wrong_sign=True)),
        ("dendrite_us", dict(mode="dendrite", rule="us")),
    ]

    for seed in seeds:
        seed_block = {"seed": seed, "goals": {}}
        for goal in goals:
            gkey = f"{goal[0]},{goal[1]}"
            gblock = {}
            for arm_name, kw in arm_specs:
                kw = dict(kw)
                mode = kw.pop("mode")
                res = run_arm(
                    mode, seed=seed, goal=goal, grid=args.grid_size,
                    n_trials=args.n_trials, steps_per_trial=args.steps_per_trial,
                    bias_mag=args.bias_mag, actor_lr=args.actor_lr,
                    value_lr=args.value_lr, temp=args.temp, gamma=args.gamma,
                    **kw)
                sc = stats(res["trial_final_distances"], res["eval_mean"])
                gblock[arm_name] = {
                    "final_pos": res["final_pos"],
                    **sc,
                    # keep a downsampled curve for the findings doc
                    "curve_every10": res["trial_final_distances"][::10],
                }
            seed_block["goals"][gkey] = gblock
        results["per_seed"].append(seed_block)

    # ---- Pre-registered verdict aggregation ----
    # GO requires (all):
    #  (1) dendrite: learned_and_converged at BOTH goals, ALL seeds
    #  (2) point control: NOT learned_and_converged (fails) -- fair baseline
    #  (3) dendrite_lesion -> collapses (not learned) -- apical gate load-bearing
    #  (4) dendrite_wrongsign -> fails (not learned) -- it's the advantage
    #  (5) distinct goal-appropriate end positions across the >=2 goals
    # VALIDITY (the toy is a fair test) requires:
    #  - ORACLE tabular-Q SUCCEEDS at both goals, all seeds (the task IS
    #    learnable -- difficulty bound; else the bias is uninformatively strong)
    #  - point control FAILS (else the toy is too easy)
    n_goals = len(goals)
    n_seeds = len(seeds)
    NG = n_goals * n_seeds

    oracle_lc = 0     # oracle learned_and_converged (want NG -- task learnable)
    dend_lc = 0       # dendrite learned_and_converged (goal,seed) count
    point_lc = 0      # point control learned_and_converged count (want 0)
    lesion_lc = 0     # dendrite_lesion learned count (want 0)
    wrong_lc = 0      # wrong-sign learned count (want 0)
    us_lc = 0         # the existing U-S rule learned count (reported)
    distinct_ok = 0   # seeds where dendrite ends at distinct goal-appropriate cells

    for sb in results["per_seed"]:
        # distinctness per seed: the dendrite eval_final for the >=2 goals must
        # differ AND each greedy policy converge near its own goal.
        dend_finals = []
        near_own_goal = True
        for goal in goals:
            gkey = f"{goal[0]},{goal[1]}"
            d = sb["goals"][gkey]["dendrite"]
            dend_finals.append(tuple(d["final_pos"]))
            if d.get("eval_mean", 99) > args.converge_thresh:
                near_own_goal = False
            if sb["goals"][gkey]["oracle"].get("learned_and_converged"):
                oracle_lc += 1
            if d.get("learned_and_converged"):
                dend_lc += 1
            if sb["goals"][gkey]["point"].get("learned_and_converged"):
                point_lc += 1
            if sb["goals"][gkey]["dendrite_lesion"].get("learned_and_converged"):
                lesion_lc += 1
            if sb["goals"][gkey]["dendrite_wrongsign"].get("learned_and_converged"):
                wrong_lc += 1
            if sb["goals"][gkey]["dendrite_us"].get("learned_and_converged"):
                us_lc += 1
        if len(set(dend_finals)) == n_goals and near_own_goal:
            distinct_ok += 1

    verdict = {
        "oracle_learned_and_converged": f"{oracle_lc}/{NG}",
        "dendrite_learned_and_converged": f"{dend_lc}/{NG}",
        "point_control_learned_and_converged": f"{point_lc}/{NG}",
        "dendrite_lesion_learned": f"{lesion_lc}/{NG}",
        "dendrite_wrongsign_learned": f"{wrong_lc}/{NG}",
        "dendrite_us_learned": f"{us_lc}/{NG}",
        "distinct_goal_appropriate_ends_seeds": f"{distinct_ok}/{n_seeds}",
        # validity flags
        "VALID_oracle_succeeds": oracle_lc == NG,
        "VALID_point_control_fails": point_lc == 0,
        # the pre-registered GO gate flags
        "test_dendrite_learns_all": dend_lc == NG,
        "lesion_collapses": lesion_lc == 0,
        "wrongsign_fails": wrong_lc == 0,
        "distinct_policies_all_seeds": distinct_ok == n_seeds,
    }
    valid = verdict["VALID_oracle_succeeds"] and verdict["VALID_point_control_fails"]
    verdict["DERISK_VALID"] = bool(valid)
    verdict["GO"] = bool(
        valid
        and verdict["test_dendrite_learns_all"]
        and verdict["lesion_collapses"]
        and verdict["wrongsign_fails"]
        and verdict["distinct_policies_all_seeds"])
    # NEGATIVE := valid setup (oracle succeeds, point fails) AND the dendrite
    # arm does NOT learn both goals all seeds.
    verdict["NEGATIVE"] = bool(valid and not verdict["test_dendrite_learns_all"])
    results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console report ----
    print("=" * 70)
    print("STAGE-1 dendrite credit-assignment toy -- VERDICT")
    print("=" * 70)
    for sb in results["per_seed"]:
        print(f"\nseed {sb['seed']}:")
        for goal in goals:
            gkey = f"{goal[0]},{goal[1]}"
            gb = sb["goals"][gkey]
            print(f"  goal {gkey}:")
            for arm in ["oracle", "point", "dendrite", "dendrite_lesion",
                        "dendrite_wrongsign", "dendrite_us"]:
                a = gb[arm]
                flag = "LEARN" if a.get("learned_and_converged") else "  -  "
                print(f"    {arm:22s} train_late={a.get('late_mean'):5} "
                      f"GREEDY_eval={a.get('eval_mean'):5} "
                      f"end={a.get('final_pos')} [{flag}]")
    print("\n" + "-" * 70)
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("-" * 70)
    final = "GO" if verdict["GO"] else (
        "NEGATIVE" if verdict["NEGATIVE"] else "VOID/INVALID")
    print(f"\nVERDICT: {final}   (DERISK_VALID={verdict['DERISK_VALID']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
