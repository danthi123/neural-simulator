#!/usr/bin/env python
"""Bug-ON escalation for the broader audit (2026-06-08) — CONDITIONAL.

Only run this if _audit_ae_summary.json (the bug-OFF main run) reports
|delta_vs_cluster_eval_baseline| > 1 sigma, i.e. the fixed tree's A+E ceiling
differs materially from the documented (bug-ON) number AND we want to rule out
methodology/non-determinism drift between "documented" and "now".

It runs the SAME A+E multi-goal deterministic config that _audit_ae_batch.py ran
on the fixed main tree, but from a worktree pinned at commit 103ded0b -- the last
BUG-PRESENT state (immediately before the first fix acdb65b4). Same seeds, same
deterministic flag => the ONLY navigation-relevant difference is the bridge
gate-array fix. That is the clean, same-methodology bug-ON vs bug-OFF A/B.

Pairing:
  bug-OFF: research/findings/raw/_audit_ae_main_s{seed}.json   (fixed main tree)
  bug-ON : research/findings/raw/_audit_ae_bugon_s{seed}.json  (worktree @103ded0b)

The expectation (from the de-risk: the cheat config shifted only 3.83->4.24,
+0.41, well within A+E's 1.58 sigma) is that the bug's impact on A+E is small.
A large clean delta here would mean the documented cluster-stacking conclusions
(all measured against the bug-ON A+E ceiling) need re-examination.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable

BUG_ON_SHA = "103ded0b"        # last bug-present commit (just before acdb65b4)
WORKTREE = os.path.abspath(os.path.join(REPO, "..", "sim-audit-bugon"))
SEEDS = [42, 43, 44]

A_E_FLAGS = [
    "--moving-goal", "--goal-schedule", "multi", "--deterministic",
    "--enable-msn-lateral-inhibition", "--enable-d1-d2-asymmetry",
    "--enable-striatal-pv-fsi", "--enable-cluster-a-closed-loop",
    "--enable-cluster-e-topography",
    "--n-steps", "1800",
]


def log(msg):
    print(f"[audit_bugon {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sum_finalq(path):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    ps = d.get("phase_stats") or []
    vals = [p.get("final_quarter_mean_distance") for p in ps
            if p.get("final_quarter_mean_distance") is not None]
    return sum(float(v) for v in vals) if vals else None


def ensure_worktree():
    if os.path.isdir(WORKTREE):
        log(f"worktree already exists: {WORKTREE}")
        return True
    log(f"creating worktree @ {BUG_ON_SHA} -> {WORKTREE}")
    rc = subprocess.run(["git", "worktree", "add", "--detach", WORKTREE, BUG_ON_SHA],
                        cwd=REPO, capture_output=True, text=True)
    if rc.returncode != 0:
        log(f"worktree add FAILED: {rc.stderr.strip()}")
        return False
    return True


def run_seed(seed):
    # Write results back to the MAIN tree's raw dir (absolute) so the analyzer
    # finds bug-ON and bug-OFF side by side.
    out = os.path.join(HERE, f"_audit_ae_bugon_s{seed}.json")
    if sum_finalq(out) is not None:
        log(f"seed {seed}: bug-ON result already present -- skipping")
        return
    cmd = [PY, "-m", "research.runners.g11_bg_runner", *A_E_FLAGS,
           "--seed", str(seed), "--out", out]
    logf = os.path.join(HERE, f"_audit_ae_bugon_s{seed}.log")
    log(f"seed {seed}: launching from bug-ON worktree (serial)")
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=WORKTREE, stdout=lf, stderr=subprocess.STDOUT).returncode
    sq = sum_finalq(out)
    log(f"seed {seed}: rc={rc}  sum_finalQ={('--' if sq is None else f'{sq:.2f}')}  ({(time.time()-t0)/60:.1f} min)")


def summarize():
    rows, on_vals, off_vals, deltas = [], [], [], []
    for s in SEEDS:
        on = sum_finalq(os.path.join(HERE, f"_audit_ae_bugon_s{s}.json"))
        off = sum_finalq(os.path.join(HERE, f"_audit_ae_main_s{s}.json"))
        d = (on - off) if (on is not None and off is not None) else None
        rows.append({"seed": s, "bug_on": on, "bug_off": off, "delta_on_minus_off": d})
        if d is not None:
            on_vals.append(on); off_vals.append(off); deltas.append(d)
    summary = {
        "config": "A+E multi-goal deterministic (5 cluster flags) — clean bug-ON vs bug-OFF A/B",
        "bug_on_commit": BUG_ON_SHA, "bug_off_commit": "main (fixed, 512026ee)",
        "metric": "sum_finalQ = sum of per-phase final_quarter_mean_distance (LOWER better)",
        "seeds": rows,
    }
    if deltas:
        md = sum(deltas) / len(deltas)
        summary["mean_bug_on"] = round(sum(on_vals) / len(on_vals), 3)
        summary["mean_bug_off"] = round(sum(off_vals) / len(off_vals), 3)
        summary["mean_delta_on_minus_off"] = round(md, 3)
        summary["interpretation"] = (
            f"bug-ON {summary['mean_bug_on']} vs bug-OFF {summary['mean_bug_off']} "
            f"(delta {md:+.2f}). " + (
                "Small clean delta -> the bug did not materially change the A+E ceiling; "
                "the documented cluster-stacking conclusions stand."
                if abs(md) <= 1.0 else
                "Large clean delta -> the bug DID change A+E; re-examine every cluster result "
                "that was measured against the bug-ON A+E ceiling."))
    out = os.path.join(HERE, "_audit_ae_bugon_summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    for r in rows:
        log(f"  seed {r['seed']}: bug-ON {r['bug_on']}  bug-OFF {r['bug_off']}  delta {r['delta_on_minus_off']}")
    if deltas:
        log(f"  {summary['interpretation']}")
    log(f"  wrote {out}")
    log("NOTE: `git worktree remove ../sim-audit-bugon` once results are confirmed.")


def main():
    log(f"bug-ON escalation A/B (conditional): A+E det seeds {SEEDS} @ {BUG_ON_SHA}")
    if not ensure_worktree():
        log("ABORT: could not create worktree")
        return
    for s in SEEDS:
        run_seed(s)
    summarize()
    log("DONE")


if __name__ == "__main__":
    main()
