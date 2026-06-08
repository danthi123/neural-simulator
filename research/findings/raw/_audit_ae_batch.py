#!/usr/bin/env python
"""Broader bug-audit orchestrator (2026-06-08).

The silent bridge plasticity bug (per-synapse gate arrays cp_d1_d2_sign /
cp_transmission_gain / cp_plasticity_rate_gain under-sized vs cp_connections.nnz
-> reward-modulated weight update raised+caught EVERY step -> reward-driven
plasticity silently dropped) was fixed in `512026ee` (_ensure_gate_capacity).

EVERY documented reward-modulated navigation result that used
--enable-d1-d2-asymmetry was produced with this bug ACTIVE. The de-risk already
showed the cheat baseline shifted 3.83 -> 4.24 post-fix. This orchestrator
quantifies the bug's impact on the load-bearing A+E multi-goal config -- the
"robust operational ceiling" that every cluster-stacking conclusion was measured
against.

DESIGN (cheap-first A/B): the DOCUMENTED A+E numbers ARE the bug-ON condition
(they were produced pre-fix). So running the SAME config on the fixed (bug-OFF)
main tree and comparing to documented is already an A/B. We escalate to a fresh
bug-ON worktree run (commit 103ded0b, just before the fix) only if the delta is
ambiguous -- handled by a sibling script, not here.

What this fixes vs the watchdog's failed attempt:
  1. ADDS --deterministic  (CLAUDE.md: required to detect cluster effects below
     the +/-3-5 non-deterministic noise floor; tightens to +/-0.7)
  2. DROPS --emit-activity  (that is a live-viz flag; on a science run it dumps
     ~166 KB of per-step JSON and massively slows the run)
  3. SERIAL, one run at a time  (the watchdog launched 2 concurrently + a 3rd
     demo joined -> OOM killed them mid-flight, no results)

Config = the documented A+E multi-goal deterministic baseline (cluster-eval
"A+E baseline 7.18 +/- 1.58", n=6; also quoted elsewhere as 6.97 +/- 0.83).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable

SEEDS = [42, 43, 44]
# The seed-99 --emit-activity demo holds the GPU; its --out marks completion.
DEMO_OUT = os.path.join(HERE, "_activitydemo.json")
DEMO_PID = 33296  # the running demo process (informational; file-signal is primary)
MAX_WAIT_S = 75 * 60   # hard cap so a dead demo never blocks the audit forever
POLL_S = 20

# Documented A+E multi-goal baselines (BUG-ON, produced pre-fix).
DOC_BASELINES = {
    "A+E_cluster_eval_det_n6": 7.18,   # std 1.58 -- the cluster-stacking comparison baseline
    "A+E_robust_ceiling": 6.97,        # std 0.83 -- quoted as the operational ceiling
}

A_E_FLAGS = [
    "--moving-goal", "--goal-schedule", "multi", "--deterministic",
    "--enable-msn-lateral-inhibition", "--enable-d1-d2-asymmetry",
    "--enable-striatal-pv-fsi", "--enable-cluster-a-closed-loop",
    "--enable-cluster-e-topography",
    "--n-steps", "1800",
]


def log(msg):
    print(f"[audit_ae {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid):
    """Windows-safe liveness probe via tasklist."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return str(pid) in out
    except Exception:
        return False


def gpu_is_free():
    # Free when the demo wrote its output (completed) OR its pid is gone.
    if os.path.exists(DEMO_OUT):
        return True
    if not pid_alive(DEMO_PID):
        return True
    return False


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


def wait_for_gpu():
    waited = 0
    while not gpu_is_free() and waited < MAX_WAIT_S:
        if waited % 120 == 0:
            log(f"waiting for GPU (demo pid {DEMO_PID} / {os.path.basename(DEMO_OUT)})... {waited}s")
        time.sleep(POLL_S)
        waited += POLL_S
    if gpu_is_free():
        log("GPU free -- starting audit runs")
    else:
        log(f"max wait {MAX_WAIT_S}s elapsed -- proceeding anyway (demo presumed dead)")


def run_seed(seed):
    out = os.path.join(HERE, f"_audit_ae_main_s{seed}.json")
    if sum_finalq(out) is not None:
        log(f"seed {seed}: result already present -- skipping")
        return out
    cmd = [PY, "-m", "research.runners.g11_bg_runner", *A_E_FLAGS,
           "--seed", str(seed), "--out", out]
    logf = os.path.join(HERE, f"_audit_ae_main_s{seed}.log")
    log(f"seed {seed}: launching (serial, deterministic, no emit-activity)")
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    sq = sum_finalq(out)
    log(f"seed {seed}: rc={rc}  sum_finalQ={('--' if sq is None else f'{sq:.2f}')}  ({dt/60:.1f} min)")
    return out


def summarize():
    rows, vals = [], []
    for s in SEEDS:
        sq = sum_finalq(os.path.join(HERE, f"_audit_ae_main_s{s}.json"))
        rows.append({"seed": s, "sum_finalQ": sq})
        if sq is not None:
            vals.append(sq)
    summary = {
        "config": "A+E multi-goal deterministic (5 cluster flags), bug-OFF (fixed main tree)",
        "fix_commit": "512026ee",
        "metric": "sum_finalQ = sum of per-phase final_quarter_mean_distance (LOWER better)",
        "seeds": rows,
        "documented_bug_on_baselines": DOC_BASELINES,
    }
    if vals:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        summary["mean"] = round(mean, 3)
        summary["std"] = round(std, 3)
        summary["n"] = len(vals)
        # Compare to the cluster-eval baseline the stacking conclusions used.
        doc = DOC_BASELINES["A+E_cluster_eval_det_n6"]
        doc_std = 1.58
        delta = mean - doc
        summary["delta_vs_cluster_eval_baseline"] = round(delta, 3)
        if abs(delta) <= doc_std:
            verdict = (f"bug-OFF A+E mean {mean:.2f} is within 1 std ({doc_std}) of the documented "
                       f"bug-ON {doc:.2f} (delta {delta:+.2f}). The bug did NOT materially shift the "
                       f"A+E ceiling -> the cluster-stacking conclusions remain robust. (Confirm with a "
                       f"bug-ON worktree A/B only if exactness is needed.)")
        else:
            verdict = (f"bug-OFF A+E mean {mean:.2f} differs from documented bug-ON {doc:.2f} by "
                       f"{delta:+.2f} (> 1 std {doc_std}). The bug MATERIALLY shifted the A+E ceiling -> "
                       f"escalate to the bug-ON worktree A/B (commit 103ded0b) for a clean same-methodology "
                       f"delta, and re-examine the cluster-stacking conclusions measured against the old A+E.")
        summary["verdict"] = verdict
    else:
        summary["verdict"] = "no completed seeds yet"

    out = os.path.join(HERE, "_audit_ae_summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    log("==== AUDIT SUMMARY ====")
    for r in rows:
        sq = r["sum_finalQ"]
        log(f"  seed {r['seed']}: sum_finalQ {('--' if sq is None else f'{sq:.2f}')}")
    if vals:
        log(f"  mean {summary['mean']}  std {summary['std']}  (n={summary['n']})")
        log(f"  documented bug-ON A+E: 7.18 +/- 1.58 (cluster-eval) / 6.97 +/- 0.83 (ceiling)")
        log(f"  VERDICT: {summary['verdict']}")
    log(f"  wrote {out}")


def main():
    log(f"broader bug-audit: A+E multi-goal det, bug-OFF (fixed main), seeds {SEEDS}")
    wait_for_gpu()
    for s in SEEDS:
        run_seed(s)
    summarize()
    log("DONE")


if __name__ == "__main__":
    main()
