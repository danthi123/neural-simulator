#!/usr/bin/env python
"""Paired analysis: does the supervised motor-teacher tighten the learned read-out?

Post-wean = last-quarter mean distance (single-goal, reflex weaned @2000). LOWER better.
  TEACHER = R2 + --sensory-cortex-teacher-pA 1500 (supervised feedback-error-learning)
  PLAIN   = R2 reward-STDP only (the ~3.9 plateau)
Reference: reflex (Rank 1) ~2.0 = precision ceiling; IT-only floor ~6.1.

GO (learning-rule lever confirmed multi-seed): TEACHER < PLAIN at every seed (teacher ~3.3,
plain ~3.9). The residual (teacher ~3.3 vs reflex ~2.0) is the categorical-WTA readout
-> population-vector readout is the next lever.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44]


def postwean(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    if ps and ps[-1].get("final_quarter_mean_distance") is not None:
        return float(ps[-1]["final_quarter_mean_distance"])
    dl = d.get("distance_log") or []
    return float(sum(dl[len(dl) * 3 // 4:]) / max(1, len(dl) - len(dl) * 3 // 4)) if dl else None


def main():
    print("\nRank 2 supervised-teacher paired analysis (post-wean last-quarter, LOWER better)\n")
    print(f"  {'seed':>4}  {'TEACHER':>8}  {'PLAIN R2':>9}  {'delta':>7}  tighter?")
    print("  " + "-" * 46)
    n_tighter, n_pairs = 0, 0
    t_vals = []
    for s in SEEDS:
        t = postwean(f"_rank2_teacher_s{s}.json")
        p = postwean(f"_rank2_R2_s{s}.json")
        if t is None or p is None:
            print(f"  {s:>4}  {('%.2f'%t) if t else '—':>8}  {('%.2f'%p) if p else '—':>9}  {'—':>7}  PENDING")
            continue
        n_pairs += 1
        t_vals.append(t)
        tighter = t < p
        n_tighter += tighter
        print(f"  {s:>4}  {t:>8.2f}  {p:>9.2f}  {t-p:>+7.2f}  {'YES' if tighter else 'no'}")
    print("  " + "-" * 46)
    if n_pairs == len(SEEDS):
        mean_t = sum(t_vals) / len(t_vals)
        print(f"\n  mean TEACHER = {mean_t:.2f}  (reflex ceiling ~2.0; reward-STDP plateau ~3.9)  "
              f"{n_tighter}/{len(SEEDS)} seeds tighter")
        if n_tighter == len(SEEDS):
            print("\nVERDICT: GO multi-seed — the supervised motor-teacher (feedback-error-learning) "
                  "tightens the learned read-out on every seed (learning-rule diagnosis confirmed). "
                  "Residual vs reflex ~2.0 = categorical readout -> population-vector lever next.")
        else:
            print(f"\nVERDICT: PARTIAL — {n_tighter}/{len(SEEDS)} tighter; inspect the non-tightening seed.")
    else:
        print("\n(awaiting seeds; re-run when the batch completes.)")


if __name__ == "__main__":
    main()
