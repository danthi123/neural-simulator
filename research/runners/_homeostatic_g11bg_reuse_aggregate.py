"""Aggregate + verdict for the homeostatic g11_bg-reuse de-risk (CYCLE 132).

Reads the per-(mode, seed) sidecars written by
``_homeostatic_g11bg_reuse_probe.py`` and answers the load-bearing question:

  Does the validated g11_bg learner, with its reward DRIVE-GATED, converge a
  learned policy that keeps the agent alive AFTER the heuristic teacher is
  weaned -- and does that collapse when the drive is lesioned or yoked?

GO  = INTACT clearly beats LESION and YOKE on post-wean survival (min-energy +
      eat-rate) across seeds -> the drive-gated reward produces load-bearing
      learning; the reuse path is viable -> proceed to the neural-drive build.
NEGATIVE / BOUNDARY = no clean intact>lesion,yoke separation -> honest wall
      (e.g. the bare cascade needs the perception arc for learned nav). Either
      way it is a real, reportable finding.

Usage:
  python -m research.runners._homeostatic_g11bg_reuse_aggregate --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RAW = "research/findings/raw"
MODES = ["intact", "lesion", "yoke"]


def _load(mode, seed):
    path = os.path.join(RAW, f"_homeo_g11bg_reuse_{mode}_seed{seed}.homeo.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    print(f"=== Homeostatic g11_bg-reuse de-risk — seeds {args.seeds} ===\n")
    per_mode = {m: [] for m in MODES}
    for seed in args.seeds:
        for mode in MODES:
            d = _load(mode, seed)
            if d is None:
                print(f"  [missing] seed={seed} mode={mode}")
                continue
            s = d["summary"]
            per_mode[mode].append(s)
            print(f"  seed={seed:>3} {mode:>6}: "
                  f"eats={s['n_eats']:>3} (post={s['eats_post_wean']:>3})  "
                  f"eat_rate_post={s['eat_rate_post']:.4f}  "
                  f"min_E_post={s['min_energy_post_wean']:.3f}  "
                  f"mean_E_post={s['mean_energy_post_wean']:.3f}  "
                  f"crashes={s['n_crash_steps']:>3}")

    print("\n--- mode means (across seeds) ---")
    agg = {}
    for mode in MODES:
        rows = per_mode[mode]
        if not rows:
            continue
        agg[mode] = {
            "eat_rate_post": _mean([r["eat_rate_post"] for r in rows]),
            "min_energy_post": _mean([r["min_energy_post_wean"] for r in rows]),
            "mean_energy_post": _mean([r["mean_energy_post_wean"] for r in rows]),
            "crashes": _mean([r["n_crash_steps"] for r in rows]),
            "n": len(rows),
        }
        a = agg[mode]
        print(f"  {mode:>6} (n={a['n']}): eat_rate_post={a['eat_rate_post']:.4f}  "
              f"min_E_post={a['min_energy_post']:.3f}  mean_E_post={a['mean_energy_post']:.3f}  "
              f"crashes={a['crashes']:.1f}")

    # Verdict (needs all three modes present).
    if all(m in agg for m in MODES):
        I, L, Y = agg["intact"], agg["lesion"], agg["yoke"]
        ctrl_eat = max(L["eat_rate_post"], Y["eat_rate_post"])
        ctrl_minE = max(L["min_energy_post"], Y["min_energy_post"])
        # GO: intact survives (min-E above a crash floor) AND clearly beats the
        # better control on both post-wean eat-rate and min-energy.
        survives = I["min_energy_post"] > 0.30
        beats_eat = I["eat_rate_post"] > 1.5 * max(1e-6, ctrl_eat)
        beats_energy = I["min_energy_post"] > ctrl_minE + 0.20
        print("\n--- VERDICT ---")
        print(f"  intact survives post-wean (min_E>0.30): {survives} ({I['min_energy_post']:.3f})")
        print(f"  intact eat-rate > 1.5x best control:     {beats_eat} "
              f"({I['eat_rate_post']:.4f} vs {ctrl_eat:.4f})")
        print(f"  intact min-E > best control + 0.20:      {beats_energy} "
              f"({I['min_energy_post']:.3f} vs {ctrl_minE:.3f})")
        if survives and (beats_eat or beats_energy):
            verdict = "GO — drive-gated reward produces load-bearing learning on the validated learner"
        elif survives and not (beats_eat or beats_energy):
            verdict = "BOUNDARY — intact survives but controls also survive (drive not yet load-bearing; sharpen depletion or check learned-nav)"
        else:
            verdict = "NEGATIVE — intact does not sustain post-wean (bare cascade may need the perception arc for learned nav)"
        print(f"\n  ==> {verdict}")
    else:
        print("\n  (need all 3 modes present for a verdict)")


if __name__ == "__main__":
    main()
