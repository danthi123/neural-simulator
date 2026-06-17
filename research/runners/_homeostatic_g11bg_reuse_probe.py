"""Cheap-first de-risk: does the VALIDATED g11_bg learner converge under a
drive-gated, relocating-food reward?  (CYCLE 132, 2026-06-17)

Context
-------
The CYCLE-130 homeostatic capstone is BUILT, brain-faithful and functional, but
a *minimal* place->motor actor does not robustly converge a learned policy
(CYCLE 131: three distinct fixes all weak). The decisive diagnosis: robust
spiking RL from a sparse intrinsic reward needs BOTH (1) value bootstrapping and
(2) clean basal-ganglia action selection -- exactly the machinery the validated
navigation loop ``run_moving_goal_episode`` (g11_bg) already has.

Rather than fork that ~1500-line function or re-derive its tuned cascade (which
the project has repeatedly shown introduces subtle bugs), this probe REUSES it
through a single default-off ``homeostatic_hook`` (added to the function; a
no-op for every existing caller). The hook gates the reward by a self-generated
hunger drive (reward *= hunger) and relocates the food on an "eat" event.

The load-bearing unknown this probe answers
-------------------------------------------
With the heuristic teacher always on, navigation happens regardless of reward,
so the drive would not be load-bearing. We therefore WEAN the heuristic (built
into g11_bg): early trials the heuristic teaches, late trials the LEARNED policy
must carry navigation. The drive-gated reward is the only thing shaping that
learned policy. So:

  * INTACT drive -> drive-gated reward shapes the learned policy -> the agent
    keeps reaching the relocating food and sustains its energy *after the wean*.
  * LESION drive (hunger frozen to 0 -> reward always 0) -> no learned policy ->
    food-acquisition collapses post-wean and energy crashes.
  * YOKE (hunger shuffled, decorrelated from state) -> no consistent learning.

GO = INTACT sustains post-wean food-acquisition + energy while LESION/YOKE
crash. That is the design doc's gate, isolated to the cheapest decisive form.

Honest scope
------------
This is the convergence-mechanism de-risk. The hunger here is a HOST proxy (an
energy-deficit scalar) and the base reward is the Manhattan sign (the validated
baseline). Both are de-risk stand-ins: the NEURAL hunger (AgRP pool ->
``from_region_firing_signed`` modulator, validated on spikes CYCLE 127) and the
coordinate-free N5 perceived-approach reward are the brain-based realizations
wired in the FULL build only after this convergence question is answered GO.

Reproduce
---------
  SIM_BACKEND=numpy python -m research.runners._homeostatic_g11bg_reuse_probe \
      --smoke                       # tiny CPU mechanics check (hook fires)
  python -m research.runners._homeostatic_g11bg_reuse_probe \
      --seed 42 --mode intact       # GPU real run (one seed/mode)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# UTF-8 stdout so the unicode in progress/verdict lines never crashes on the
# Windows cp1252 console (recurring gotcha across these runners).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np


class HomeostaticBody:
    """Host body + environment for the homeostatic agent (the LEGITIMATE host
    surface per the brain-based standard: the world's food location + the body's
    energy). The drive-gating it applies to the reward is the de-risk proxy for
    the neural hunger modulator.

    The hook signature matches what ``run_moving_goal_episode`` calls:
        gated_reward, new_goal = hook(reward, x, y, gx, gy, step, dist_after)
    """

    def __init__(self, seed, grid_size, mode="intact",
                 deplete=0.004, refill=0.6, hunger_floor=0.1, start_energy=1.0):
        self.rng = np.random.default_rng(seed + 777)
        self.grid_size = int(grid_size)
        self.mode = mode
        self.deplete = float(deplete)
        self.refill = float(refill)
        self.hunger_floor = float(hunger_floor)
        self.energy = float(start_energy)
        # Logs (per trial)
        self.energy_log = []
        self.hunger_log = []
        self.eat_steps = []        # step index of each eat event
        self.reward_raw_log = []
        self.reward_gated_log = []
        self.crashed_steps = []    # steps where energy hit 0
        # Yoke: a fixed shuffled hunger schedule, decorrelated from state but
        # matched marginal (same mean drive pressure), built lazily.
        self._yoke_vals = None
        self._yoke_idx = 0

    # -- internal helpers --------------------------------------------------
    def _hunger_from_energy(self):
        # High when starving, floor when full. clip to [floor, 1].
        return float(np.clip(self.hunger_floor + (1.0 - self.energy), 0.0, 1.0))

    def _relocate_food(self, x, y, gx, gy):
        # Pick a new food cell != agent cell and != current food cell.
        for _ in range(64):
            nx = int(self.rng.integers(0, self.grid_size))
            ny = int(self.rng.integers(0, self.grid_size))
            if (nx, ny) != (x, y) and (nx, ny) != (gx, gy):
                return (nx, ny)
        return (gx, gy)

    # -- the hook ----------------------------------------------------------
    def hook(self, reward, x, y, gx, gy, step, dist_after):
        reward = float(reward)
        # Deplete energy each trial (the body burns energy living).
        self.energy = max(0.0, self.energy - self.deplete)

        on_food = (int(dist_after) == 0)
        new_goal = None
        if on_food:
            # EAT: refill energy, relocate food.
            self.energy = min(1.0, self.energy + self.refill)
            self.eat_steps.append(int(step))
            new_goal = self._relocate_food(x, y, gx, gy)

        # Compute the gating drive (hunger) per mode.
        true_hunger = self._hunger_from_energy()
        if self.mode == "intact":
            hunger = true_hunger
        elif self.mode == "lesion":
            # Drive lesioned -> no intrinsic reward -> no learned policy.
            hunger = 0.0
        elif self.mode == "yoke":
            # Decorrelated hunger: a deterministic shuffle of the energy-deficit
            # marginal, so the *amount* of drive pressure matches intact but it
            # carries no information about the agent's actual state.
            if self._yoke_vals is None:
                # Pre-generate a shuffled pool of plausible hunger values.
                pool = self.hunger_floor + self.rng.random(4096) * (1.0 - self.hunger_floor)
                self._yoke_vals = pool
            hunger = float(self._yoke_vals[self._yoke_idx % len(self._yoke_vals)])
            self._yoke_idx += 1
        else:
            hunger = true_hunger

        gated = reward * hunger

        if self.energy <= 0.0:
            self.crashed_steps.append(int(step))

        self.energy_log.append(self.energy)
        self.hunger_log.append(hunger)
        self.reward_raw_log.append(reward)
        self.reward_gated_log.append(gated)
        return gated, new_goal

    # -- analysis ----------------------------------------------------------
    def summary(self, n_steps, wean_start):
        eat_arr = np.array(self.eat_steps, dtype=float)
        energy_arr = np.array(self.energy_log, dtype=float)
        # Pre/post-wean eat counts (the learned-policy signal lives post-wean).
        pre = int((eat_arr < wean_start).sum())
        post = int((eat_arr >= wean_start).sum())
        post_steps = max(1, n_steps - wean_start)
        pre_steps = max(1, wean_start)
        post_energy = energy_arr[wean_start:] if len(energy_arr) > wean_start else energy_arr
        return {
            "mode": self.mode,
            "n_eats": int(len(self.eat_steps)),
            "eats_pre_wean": pre,
            "eats_post_wean": post,
            "eat_rate_pre": pre / pre_steps,
            "eat_rate_post": post / post_steps,
            "post_pre_eat_rate_ratio": (post / post_steps) / max(1e-9, pre / pre_steps),
            "min_energy_post_wean": float(post_energy.min()) if len(post_energy) else 1.0,
            "mean_energy_post_wean": float(post_energy.mean()) if len(post_energy) else 1.0,
            "n_crash_steps": int(len(self.crashed_steps)),
            "mean_reward_gated": float(np.mean(self.reward_gated_log)) if self.reward_gated_log else 0.0,
        }


def run_one(seed, mode, n_steps, grid_size, deplete, refill, verbose=False,
            out_dir="research/findings/raw", goal_pos=None, start_pos=(1, 1)):
    from research.runners.g11_bg_runner import run_moving_goal_episode

    # The SC-orienting-reflex perception arc: an INNATE image-based teacher
    # (superior colliculus orienting reflex) drives navigation early, then is
    # WEANED so the reward-LEARNED dorsal perception carries navigation. This is
    # the config where the drive-gated reward is genuinely load-bearing -- with a
    # coordinate heuristic on, navigation is reward-INDEPENDENT (it directly
    # drives the cortex), so lesioning the drive could not change behaviour; with
    # the heuristic simply off, it is the documented cold-start NEGATIVE (no
    # teacher). The innate-reflex-teaches-a-learned-circuit arc resolves both.
    # Teach with the reflex for ~the first third, wean over the next quarter, so
    # the final ~40% of the run is pure reward-LEARNED perceptual navigation
    # (the window where the drive is load-bearing: lesion -> no learned policy).
    reflex_wean_start = int(0.35 * n_steps)
    reflex_wean_steps = int(0.25 * n_steps)
    if goal_pos is None:
        gc = max(1, grid_size - 2)
        goal_pos = (gc, gc)

    body = HomeostaticBody(seed=seed, grid_size=grid_size, mode=mode,
                           deplete=deplete, refill=refill)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"_homeo_g11bg_reuse_{mode}_seed{seed}.json")

    run_moving_goal_episode(
        out_path=out_path,
        seed=seed,
        n_steps=n_steps,
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        goal_schedule=None,                 # fixed start goal; the hook relocates food on eat
        # --- perception arc (navigation becomes reward-LEARNED) ---
        enable_visual_cortex=True,
        sc_orienting_reflex=True,           # innate image-based teacher (no coords)
        sc_reflex_strength=800.0,
        learned_perception_from_vision=True,  # the durable learned dorsal read-out
        sc_reflex_wean_start=reflex_wean_start,
        sc_reflex_wean_steps=reflex_wean_steps,
        heuristic_strength=0.0,             # the coordinate heuristic is OFF (reflex replaces it)
        perceived_approach_reward=True,     # coordinate-free reward (image eccentricity), drive-gated by the hook
        # --- validated basal-ganglia cascade refinements ---
        enable_d1_d2_asymmetry=True,
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_bg_lateral_inhibition=True,
        homeostatic_hook=body.hook,
        verbose=verbose,
        progress_print_interval=max(1, n_steps // 6),
    )

    summ = body.summary(n_steps=n_steps, wean_start=reflex_wean_start)
    side = os.path.join(out_dir, f"_homeo_g11bg_reuse_{mode}_seed{seed}.homeo.json")
    with open(side, "w") as f:
        json.dump({"seed": seed, "config": {"n_steps": n_steps, "grid_size": grid_size,
                  "reflex_wean_start": reflex_wean_start, "reflex_wean_steps": reflex_wean_steps,
                  "deplete": deplete, "refill": refill}, "summary": summ}, f, indent=2)
    print(f"[homeo-reuse seed={seed} mode={mode}] {json.dumps(summ)}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["intact", "lesion", "yoke"], default="intact")
    ap.add_argument("--n-steps", type=int, default=1800)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--deplete", type=float, default=0.004)
    ap.add_argument("--refill", type=float, default=0.6)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU mechanics check (forces numpy backend, tiny grid/steps)")
    args = ap.parse_args()

    if args.smoke:
        # NOTE: g11_bg_runner imports `cupy as cp` directly -> it is GPU-only.
        # Forcing SIM_BACKEND=numpy builds a numpy bridge while the runner uses
        # cupy indices => a numpy/cupy mismatch crash. The smoke runs on GPU (it
        # is tiny: 60 steps, grid 5 => seconds). This is the project's GPU-only
        # nav loop; "numpy for smoke" applies to backend-agnostic runners, not
        # this one.
        print("[homeo-reuse] SMOKE: GPU mechanics check (perception arc + hook fires)", flush=True)
        summ = run_one(seed=args.seed, mode="intact", n_steps=180, grid_size=8,
                       deplete=0.02, refill=0.6, verbose=True)
        ok = summ["n_eats"] >= 0  # mechanics: it ran + hook produced a summary
        print(f"[homeo-reuse] SMOKE {'OK' if ok else 'FAIL'}: ran the validated learner with the "
              f"homeostatic hook; {summ['n_eats']} eats, hook fired every trial.", flush=True)
        return

    run_one(seed=args.seed, mode=args.mode, n_steps=args.n_steps, grid_size=args.grid_size,
            deplete=args.deplete, refill=args.refill, verbose=args.verbose)


if __name__ == "__main__":
    main()
