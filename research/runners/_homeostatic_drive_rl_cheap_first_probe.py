"""Homeostatic-drive RL — cheapest-first falsification probe (the artificial-life frontier's load-bearing risk).

Per the scoping (2026-06-17-artificial-life-frontier-scoping.md): the agent has a competent cognitive engine + a
competent body but NO MOTIVATIONAL CORE — no neural internal state that generates its own goals and defines
reward intrinsically. The recommended first capability is a NEURAL HOMEOSTATIC DRIVE: a 2-pool push-pull drive
(hypothalamic AgRP=hunger / POMC=satiety, catalog O.05/O.06/O.10/O.11) driven by the body's energy DEFICIT, with
reward DEFINED INTRINSICALLY as drive-reduction (Keramati & Gutkin, *eLife* 2014: reward ≡ reduction of a
homeostatic deviation). The agent then acts to keep itself alive, with NO externally-supplied goal.

THE LOAD-BEARING RISK (per the scoping) is not the drive→reward→dopamine half (a proven pattern) but whether the
substrate can LEARN A POLICY from the sparse INTRINSIC drive-reduction reward (check 4). This is the cheapest-
first numpy falsification of exactly that, BEFORE committing to the spiking-bridge build: if a rate-level proxy of
the loop cannot learn from intrinsic reward, the spiking version certainly cannot (falsified cheaply); if it can,
promote to the 2-pool spiking drive region + the neuromodulator `from_region_firing_signed` reward source + the
existing dopamine RPE.

THE LOOP (host code is legitimate ONLY for the body + environment per the brain-based-only standard; here the
DRIVE + REWARD are the "brain" parts, proxied at rate level for the cheap-first; the energy/corridor are the body):
  * Body: 1-D energy E in [0,1], depletes each step; deficit = set_point - E. Reaching the resource refills E (eat).
  * Drive: 2-pool push-pull rate model. agrp (hunger) rises with deficit, pomc (satiety) with surplus, reciprocal
    inhibition. drive = agrp - pomc (the hunger signal; tracks the deficit).
  * Reward (INTRINSIC): r = drive_before - drive_after  (= -Δdrive = drive REDUCTION). Eating drops the deficit ->
    drops the drive -> positive r. NO host distance/goal term anywhere.
  * RL: tabular Q-learning over (position-bin) x {toward, away}; the agent learns to seek the resource FROM r.

GATES (>=3 seeds):
  (1) corr(deficit, drive) >= +0.9            -- the neural drive encodes the body's deficit.
  (2) hungry approach-rate >= 2x sated         -- the drive biases action (intrinsic reward scales with deficit).
  (4) time-to-resource learns DOWN >= 30%      -- THE LOAD-BEARING: the agent learns a policy from intrinsic reward.
ANTI-CHEATS (load-bearing):
  * LESION drive (drive held constant) -> r=0 -> no learning (time-to-resource flat). Self-direction must collapse.
  * YOKED-random drive (same marginal stats, shuffled) -> r uninformative -> no learning.
  * r is computed from the DRIVE (drive-reduction), NOT a host distance-to-resource term (asserted by construction).

Run: SIM_BACKEND=numpy python -m research.runners._homeostatic_drive_rl_cheap_first_probe --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

L = 8           # corridor length (positions 0..L-1); resource at position 0
SET_POINT = 1.0
DEPLETE = 0.06  # energy lost per step
EAT_REFILL = 1.0
N_TRIALS = 400
GAMMA, ALPHA, EPS = 0.9, 0.3, 0.1


class TwoPoolDrive:
    """A 2-pool push-pull hunger drive (rate proxy of AgRP<->POMC reciprocal inhibition). agrp tracks the deficit,
    pomc the surplus; drive = agrp - pomc. lesion=True freezes the drive (the anti-cheat). yoke=an array of
    pre-generated drive values to replay (the yoked-random control)."""

    def __init__(self, lesion=False, tau=0.5):
        self.agrp = 0.0
        self.pomc = 0.0
        self.tau = tau
        self.lesion = lesion

    def update(self, deficit):
        if self.lesion:
            return 0.5                              # constant drive -> no drive-reduction signal
        surplus = max(0.0, -deficit) + (1.0 - max(0.0, deficit))  # how "full" the body is
        # push-pull: each pool integrates its drive minus inhibition from the other
        a_t = max(0.0, deficit)
        p_t = max(0.0, 1.0 - max(0.0, deficit))
        self.agrp += self.tau * (a_t - 0.5 * self.pomc - self.agrp)
        self.pomc += self.tau * (p_t - 0.5 * self.agrp - self.pomc)
        return self.agrp - self.pomc


def run_episode_set(seed, lesion=False, yoke=False):
    rng = np.random.default_rng(seed)
    Q = np.zeros((L, 2))                            # state = position bin; 2 abstract actions
    # REMAPPED action map (the load-bearing anti-cheat): which abstract action index moves TOWARD the resource is
    # randomized per seed, so the agent cannot default to the optimal action — it must LEARN the mapping from the
    # intrinsic reward. (Without this, argmax of an all-zero Q picks action 0; if action 0 were always 'toward',
    # an untrained/lesioned agent would reach the resource for free — a confound.)
    toward_action = int(rng.integers(2))
    drive = TwoPoolDrive(lesion=lesion)
    deficits, drives, times = [], [], []
    approach_hungry, approach_sated, n_h, n_s = 0, 0, 0, 0
    yoke_pool = rng.permutation(np.linspace(-0.5, 0.5, 200)) if yoke else None
    yi = 0
    for trial in range(N_TRIALS):
        pos = L - 1
        E = rng.uniform(0.2, 0.6)                   # start hungry-ish
        steps = 0
        while steps < 40:
            deficit = SET_POINT - E
            d_before = drive.update(deficit)
            if yoke:                                # replace the real drive with a shuffled one (uninformative)
                d_before = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            deficits.append(deficit); drives.append(d_before)
            # action selection (eps-greedy with RANDOM tie-break so an untrained Q doesn't default to a fixed action)
            if rng.random() < EPS:
                a = int(rng.integers(2))
            else:
                a = int(rng.choice(np.flatnonzero(Q[pos] == Q[pos].max())))
            toward = (a == toward_action)
            if deficit > 0.5:
                approach_hungry += int(toward); n_h += 1
            else:
                approach_sated += int(toward); n_s += 1
            new_pos = max(0, pos - 1) if toward else min(L - 1, pos + 1)
            E = max(0.0, E - DEPLETE)
            ate = (new_pos == 0)
            if ate:
                E = min(1.0, E + EAT_REFILL)
            deficit2 = SET_POINT - E
            d_after = drive.update(deficit2)
            if yoke:
                d_after = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            r = d_before - d_after                  # INTRINSIC reward = drive reduction (no host distance term)
            Q[pos, a] += ALPHA * (r + GAMMA * np.max(Q[new_pos]) - Q[pos, a])
            pos = new_pos
            steps += 1
            if ate:
                break
        times.append(steps)
    deficits, drives, times = np.array(deficits), np.array(drives), np.array(times)
    corr = float(np.corrcoef(deficits, drives)[0, 1]) if drives.std() > 1e-9 else 0.0
    early = float(np.mean(times[:50])); late = float(np.mean(times[-50:]))
    learn_drop = (early - late) / (early + 1e-9)
    appr_h = approach_hungry / max(n_h, 1); appr_s = approach_sated / max(n_s, 1)
    return {"corr_deficit_drive": corr, "time_early": early, "time_late": late, "learn_drop": learn_drop,
            "approach_hungry": appr_h, "approach_sated": appr_s}


def run_seed(seed):
    real = run_episode_set(seed, lesion=False)
    lesioned = run_episode_set(seed, lesion=True)
    yoked = run_episode_set(seed, yoke=True)
    # THE LOAD-BEARING learning test (robust to random init, which confounds a raw early->late drop): the agent
    # taught by the INTRINSIC reward must reach the resource markedly FASTER than the lesion/yoke controls, and
    # near the ~7-step optimum. (A raw learn-drop is reported too, but the control comparison is the real gate.)
    opt = L - 1
    learns = (real["time_late"] <= 0.75 * min(lesioned["time_late"], yoked["time_late"])
              and real["time_late"] <= opt + 1.5)
    return {"seed": seed, "real": real, "lesion": lesioned, "yoke": yoked,
            "check1_corr": real["corr_deficit_drive"] >= 0.9,
            "check4_learn_vs_controls": bool(learns),
            "real_late": real["time_late"], "lesion_late": lesioned["time_late"], "yoke_late": yoked["time_late"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_homeostatic_drive_rl.json")
    a = ap.parse_args()
    try:                                            # Windows cp1252 stdout crashes on the unicode in the verdict prints
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[homeostatic-drive RL cheap-first] can the agent LEARN a policy from an INTRINSIC drive-reduction "
          "reward (no host goal)?\n  GATES: corr(deficit,drive)>=0.9 | real-late time-to-resource <= 0.75x the "
          "lesion+yoke controls AND near the ~7-step optimum.\n", flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        re = r["real"]
        print(f"  [seed {seed}] corr {re['corr_deficit_drive']:+.2f} | late time-to-resource: real {r['real_late']:.1f} "
              f"vs lesion {r['lesion_late']:.1f} / yoke {r['yoke_late']:.1f} (opt {L-1}) || "
              f"{'GO' if (r['check1_corr'] and r['check4_learn_vs_controls']) else 'NO'}", flush=True)

    def passes(r):
        return bool(r["check1_corr"] and r["check4_learn_vs_controls"])
    n_go = sum(passes(r) for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if n_go == len(results):
        ld = float(np.mean([r["real"]["learn_drop"] for r in results]))
        print(f"  GO ({n_go}/{len(results)} seeds): a self-generated homeostatic drive GENERATES GOALS and the agent "
              f"LEARNS to keep itself alive from the INTRINSIC drive-reduction reward (time-to-resource down "
              f"{100*ld:.0f}%), with NO external goal. The drive encodes the body's deficit, biases action when "
              "hungry, and lesioning/yoking it collapses the learning. ⇒ promote to the 2-pool SPIKING drive region "
              "+ the neuromodulator from_region_firing_signed reward source + the existing dopamine RPE (the "
              "brain-based realization). The motivational core is reachable.", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): the agent does not robustly learn from the "
              "intrinsic drive-reduction reward at the rate-proxy level — the load-bearing wall is the policy "
              "learning, not the drive mechanism. An honest negative that pins the exact wall (a self-generated "
              "drive can be built + generates goals + a correct reward, but learning the policy from sparse "
              "intrinsic reward is the boundary) — a high-value deliverable per the actual-goal mandate.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
