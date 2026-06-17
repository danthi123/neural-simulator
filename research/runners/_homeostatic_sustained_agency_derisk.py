"""Sustained homeostatic agency — does the self-generated drive keep the agent ALIVE over time?

The cheapest-first GO (2026-06-17-homeostatic-drive-rl-cheap-first-GO.md) showed the agent LEARNS a policy from
the intrinsic drive-reduction reward. This probe tests the complementary "alive over time" property at the
algorithm level: over a long survival episode (energy continuously depleting), does the drive-equipped agent
MAINTAIN its energy in a healthy band by repeated self-directed food-seeking — i.e. self-regulate, the essence of
artificial life — while a control with no drive (no intrinsic reward -> no learned policy) CRASHES?

This is a rate-proxy scaffold (the brain-based spiking realization reuses the validated navigation learning loop
with the neural drive-reduction reward -- 2026-06-17-homeostatic-reward-plasticity-link-BY-COMPOSITION.md). It
demonstrates the homeostatic-regulation BEHAVIOUR the de-risked motivational core produces.

THE EPISODE: a corridor (food at position 0). Energy E depletes each step; deficit = 1−E; the 2-pool drive tracks
it. The agent online-Q-learns to navigate from r = drive-reduction (eating a real deficit -> reward). The
action->direction map is REMAPPED per seed (the agent must LEARN which way is food; no free default).

GATE (>=3 seeds): the DRIVE agent keeps energy in the healthy band (E >= 0.3) most of the time and never crashes
(min E > 0.1), markedly better than the LESION control (drive frozen -> r=0 -> no learned policy -> random
wandering rarely reaches food -> energy crashes). ⇒ the self-generated drive produces SUSTAINED self-regulation:
the agent keeps itself alive.

Run: SIM_BACKEND=numpy python -m research.runners._homeostatic_sustained_agency_derisk --seeds 42 43 44 100 101 102
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

from research.runners._homeostatic_drive_rl_cheap_first_probe import TwoPoolDrive

# Dynamics chosen so the LEARNED policy reliably survives while RANDOM wandering reliably crashes: a refill
# (0.3) larger than the learned-policy round-trip cost (~6 steps x 0.015 = 0.09 -> net +0.21) but smaller than
# the random-walk cost (~L^2 ≈ 36 steps x 0.015 = 0.54 -> net -0.24). Full initial reserves give time to reach
# food once (and start learning) before starving. The difference is the LEARNED policy, which only the drive
# (intrinsic reward) produces.
L = 6
SET_POINT = 1.0
DEPLETE = 0.015    # energy lost per step
EAT_REFILL = 0.3   # energy gained on reaching food
START_E = 1.0
N_STEPS = 3000
HEALTHY = 0.3      # energy band floor
CRASH = 0.1
GAMMA, ALPHA, EPS = 0.9, 0.25, 0.1


def run_episode(seed, lesion=False):
    rng = np.random.default_rng(seed)
    Q = np.zeros((L, 2))
    toward_action = int(rng.integers(2))             # remapped: the agent must LEARN which action is 'toward food'
    drive = TwoPoolDrive(lesion=lesion)
    pos = L - 1
    E = START_E
    energies = []
    for _ in range(N_STEPS):
        deficit = SET_POINT - E
        d_before = drive.update(deficit)
        # eps-greedy with random tie-break (no free default action)
        if rng.random() < EPS:
            a = int(rng.integers(2))
        else:
            a = int(rng.choice(np.flatnonzero(Q[pos] == Q[pos].max())))
        toward = (a == toward_action)
        new_pos = max(0, pos - 1) if toward else min(L - 1, pos + 1)
        E = max(0.0, E - DEPLETE)
        if new_pos == 0:                              # reached food -> eat
            E = min(1.0, E + EAT_REFILL)
        deficit2 = SET_POINT - E
        d_after = drive.update(deficit2)
        r = d_before - d_after                        # intrinsic reward = drive reduction (0 under lesion)
        Q[pos, a] += ALPHA * (r + GAMMA * np.max(Q[new_pos]) - Q[pos, a])
        pos = new_pos
        energies.append(E)
    energies = np.array(energies)
    # measure over the second half (after the agent has had time to learn)
    half = energies[N_STEPS // 2:]
    return {"band_occupancy": float(np.mean(half >= HEALTHY)), "min_energy": float(half.min()),
            "mean_energy": float(half.mean()), "crash_frac": float(np.mean(half < CRASH))}


def run_seed(seed):
    drive = run_episode(seed, lesion=False)
    lesion = run_episode(seed, lesion=True)
    out = {"seed": seed, "drive": drive, "lesion": lesion}
    # The discriminator is CRASH-AVOIDANCE (genuine regulation), not band-occupancy: the drive agent keeps energy
    # well above the crash floor (never starves), while the lesion crashes (survives only by chance recovery).
    out["go"] = bool(drive["min_energy"] > HEALTHY and drive["crash_frac"] < 0.01
                     and lesion["min_energy"] < CRASH and drive["min_energy"] >= lesion["min_energy"] + 0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default="research/findings/raw/_homeostatic_sustained_agency.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("[sustained homeostatic agency] does the self-generated drive keep the agent ALIVE over a long episode?\n"
          "  GATE: DRIVE agent NEVER crashes (min E > 0.3, crash-frac ~0) while the LESION control crashes "
          "(min E < 0.1) -- genuine self-regulation, not luck.\n", flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        d, l = r["drive"], r["lesion"]
        print(f"  [seed {seed}] DRIVE minE {d['min_energy']:.2f} crash% {100*d['crash_frac']:.1f} meanE "
              f"{d['mean_energy']:.2f} | LESION minE {l['min_energy']:.2f} crash% {100*l['crash_frac']:.1f} || "
              f"{'GO' if r['go'] else 'NO'}", flush=True)

    n_go = sum(r["go"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    if n_go == len(results):
        dmin = float(np.mean([r["drive"]["min_energy"] for r in results]))
        lmin = float(np.mean([r["lesion"]["min_energy"] for r in results]))
        print(f"  GO ({n_go}/{len(results)} seeds): the self-generated homeostatic drive produces SUSTAINED "
              f"self-regulation — the agent NEVER crashes (mean min-energy {dmin:.2f}, well above the floor), keeping "
              "itself alive over the whole episode by repeated self-directed food-seeking, with NO external goal. "
              f"Without the drive (no intrinsic reward -> no learned policy) the agent CRASHES (min-energy {lmin:.2f}, "
              "starving repeatedly, surviving only by chance recovery). ⇒ the de-risked motivational core yields a "
              "genuinely self-maintaining agent — the 'alive over time' property. (Rate-proxy scaffold; the brain-"
              "based realization reuses the validated nav learning loop with the neural reward.)", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): the drive does not robustly sustain homeostasis vs "
              "the control — localize (deplete/refill balance, corridor length, learning rate). Honest boundary.",
              flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
