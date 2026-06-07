#!/usr/bin/env python
"""Finalize the N1 adaptive-wean re-run (b1f1y2oid, probe-window 500).

Reads research/findings/raw/_n1_adaptive_pw500_s{42,43,44}.json and reports,
per seed: the adaptive commit step (when the online readiness probe decided the
IT->cortex mapping was self-sufficient and weaned the heuristic permanently OFF),
the committing probe's mean distance, and the POST-WEAN HOLD = the run's last-
quarter mean distance (for a --goal-schedule single run the last quarter is
~steps 8250-11000, far after any commit, so it is the durable heuristic-OFF
performance). HOLD ~1-2 = the agent navigates from learned perception with NO
heuristic (N1 biologized); COLLAPSE ~5-6 = the mapping did not consolidate.

Verdict: 3/3 HOLD -> N1 robustly biologized via adaptive activity-gated weaning.
Not 3/3 -> characterize + bank "biologizable-in-principle, robust-auto-wean hard"
per the reasonable-budget gate (one targeted iteration, then stop).

Usage:  python research/findings/raw/_n1_adaptive_finalize.py
No deps beyond the stdlib; reads whatever seed files exist (skips missing ones).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 43, 44]
HOLD_BAR = 2.5      # post-wean last-quarter mean distance <= this = HOLD
COLLAPSE_BAR = 4.0  # >= this = COLLAPSE (the ~5 cold-start floor neighborhood)


def at_goal_fraction(distance_log, last_frac=0.25):
    if not distance_log:
        return float("nan")
    n = len(distance_log)
    tail = distance_log[int(n * (1.0 - last_frac)):]
    if not tail:
        return float("nan")
    return sum(1 for d in tail if d <= 0.5) / len(tail)


def post_wean_hold(data):
    """Last-quarter mean distance = durable heuristic-OFF performance."""
    ps = data.get("phase_stats") or []
    if ps and isinstance(ps[-1], dict) and "final_quarter_mean_distance" in ps[-1]:
        return float(ps[-1]["final_quarter_mean_distance"])
    # Fallback: compute from distance_log directly.
    dl = data.get("distance_log") or []
    if not dl:
        return float("nan")
    tail = dl[len(dl) * 3 // 4:]
    return float(sum(tail) / len(tail)) if tail else float("nan")


def main():
    rows, n_found, n_hold = [], 0, 0
    for s in SEEDS:
        path = os.path.join(HERE, f"_n1_adaptive_pw500_s{s}.json")
        if not os.path.exists(path):
            rows.append((s, None, None, None, None, "NOT-DONE"))
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            rows.append((s, None, None, None, None, f"UNREADABLE ({e})"))
            continue
        n_found += 1
        commit = data.get("adaptive_wean_commit_step", -1)
        hist = data.get("adaptive_wean_probe_history") or []
        commit_probe = next((h for h in hist if h.get("committed")), None)
        commit_dist = commit_probe.get("mean_dist") if commit_probe else None
        hold = post_wean_hold(data)
        atg = at_goal_fraction(data.get("distance_log") or [])
        if hold == hold and hold <= HOLD_BAR:        # not NaN and <= bar
            verdict = "HOLD"
            n_hold += 1
        elif hold == hold and hold >= COLLAPSE_BAR:
            verdict = "COLLAPSE"
        else:
            verdict = "MARGINAL"
        rows.append((s, commit, commit_dist, hold, atg, verdict))

    print(f"\nN1 adaptive-wean (probe-window 500) finalizer  bar: HOLD<={HOLD_BAR} COLLAPSE>={COLLAPSE_BAR}\n")
    print(f"{'seed':>4} {'commit_step':>11} {'commit_dist':>11} {'postwean':>9} {'atgoal%':>8}  verdict")
    print("-" * 62)
    for s, commit, cdist, hold, atg, v in rows:
        cs = "-" if commit in (None, -1) else str(commit)
        cd = "-" if cdist is None else f"{cdist:.2f}"
        hd = "-" if (hold is None or hold != hold) else f"{hold:.2f}"
        ag = "-" if (atg is None or atg != atg) else f"{100*atg:.1f}"
        print(f"{s:>4} {cs:>11} {cd:>11} {hd:>9} {ag:>8}  {v}")

    print("-" * 62)
    if n_found < len(SEEDS):
        print(f"\n{n_found}/{len(SEEDS)} seed files present; re-run when all complete.")
    else:
        if n_hold == len(SEEDS):
            print(f"\nVERDICT: {n_hold}/{len(SEEDS)} HOLD -> N1 ROBUSTLY BIOLOGIZED via "
                  "adaptive activity-gated weaning. Write the GO finding, push both "
                  "remotes; nav arc = N8 + N6 + N1 all biologized.")
        else:
            print(f"\nVERDICT: {n_hold}/{len(SEEDS)} HOLD -> NOT robust. Characterize + bank "
                  "N1 'biologizable-in-principle, robust-auto-wean genuinely hard' per the "
                  "reasonable-budget gate (no further grinding of the wean knob).")


if __name__ == "__main__":
    main()
