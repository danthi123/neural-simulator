"""Session D.C.3: Pavlovian/R-STDP learning curve analysis.

Takes a directory of experiment_headless output JSONs and extracts:
  - Pre-test vs post-test firing-rate delta (learning magnitude)
  - Intergroup weight trajectory across phases (weight-level learning)
  - Rescorla-Wagner fit quality on the weight growth (biology check)

Rescorla-Wagner model: ΔV = α·β·(λ - V), closed form V(t) = λ(1 - exp(-t/τ))
where τ = 1 / (α·β). The intergroup weight trajectory, if the learning
mechanism is a leaky integrator of reinforcement, should approach an
asymptote exponentially.

Usage:
    python research/analyze_pavlovian.py research/findings/raw/pavlovian/*.json
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def extract_weight_trajectory(log: list) -> list[dict]:
    """Pull intergroup_weights events from an experiment log in order."""
    events = []
    for e in log:
        if e.get("event") == "intergroup_weights":
            events.append({
                "label": e.get("label", "?"),
                "from": e.get("from_group"),
                "to": e.get("to_group"),
                "mean_weight": e.get("mean_weight", 0.0),
                "std_weight": e.get("std_weight", 0.0),
                "n_connections": e.get("n_connections", 0),
            })
    return events


def extract_cs_response_trajectory(log: list, input_group: str, output_group: str) -> dict:
    """Pull output-group firing rates during input-ON windows, grouped by phase.

    Defines 'input-ON' as readout entries where the input group's rate > 20 Hz
    (stimulus active and driving input). Works for both associative (CS->US)
    and reinforcement (stimulus->response) experiments.
    """
    by_phase = {"pre_test": [], "training": [], "post_test": [],
                "baseline": [], "rl_training": []}
    for e in log:
        if e.get("event") != "readout":
            continue
        rates = e.get("rates", {})
        cs = rates.get(input_group, 0.0)
        us = rates.get(output_group, 0.0)
        phase = e.get("phase", "")
        if cs > 20 and phase in by_phase:
            by_phase[phase].append({
                "time_ms": e.get("time_ms"),
                "cs_rate": cs,
                "us_rate": us,
            })
    return by_phase


def rescorla_wagner_fit(t: np.ndarray, V: np.ndarray) -> dict:
    """Fit V(t) = V0 + (lambda - V0) * (1 - exp(-t/tau)).

    Uses simple least-squares over a grid of tau. V0 taken as V[0],
    lambda initialized as V[-1].

    Returns:
        fit dict with 'lambda', 'tau', 'V0', 'r_squared', 'predicted'.
    """
    t = np.asarray(t, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if len(t) < 3:
        return {"lambda": float(V[-1]) if len(V) else 0.0, "tau": 0.0,
                "V0": float(V[0]) if len(V) else 0.0, "r_squared": 0.0,
                "predicted": V.tolist(), "n_points": len(t)}

    V0 = float(V[0])
    lam = float(V[-1])
    # Grid over tau in log space
    best = {"tau": 1.0, "sse": float("inf")}
    for tau in np.logspace(math.log10(1), math.log10(max(t[-1], 10)), 100):
        pred = V0 + (lam - V0) * (1 - np.exp(-t / tau))
        sse = float(np.sum((V - pred) ** 2))
        if sse < best["sse"]:
            best = {"tau": float(tau), "sse": sse}

    tau = best["tau"]
    predicted = V0 + (lam - V0) * (1 - np.exp(-t / tau))
    ss_res = float(np.sum((V - predicted) ** 2))
    ss_tot = float(np.sum((V - V.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "lambda": lam, "tau": tau, "V0": V0,
        "r_squared": r_squared,
        "predicted": predicted.tolist(),
        "n_points": len(t),
    }


def extract_per_trial_curve(trial_data: list, input_group: str, output_group: str
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From trial_data list, extract (trial_index, input_rate, output_rate, success) arrays."""
    if not trial_data:
        return (np.array([]),) * 4
    trials = np.array([t.get("trial", i) for i, t in enumerate(trial_data)])
    cs_rates = np.array([t.get("rates", {}).get(input_group, 0.0) for t in trial_data])
    us_rates = np.array([t.get("rates", {}).get(output_group, t.get("output_rate", 0.0))
                          for t in trial_data])
    success = np.array([1.0 if t.get("success") else 0.0 for t in trial_data])
    return trials, cs_rates, us_rates, success


def analyze_one(path: str) -> dict:
    data = json.load(open(path))
    # run_experiment_headless outputs use "log_entries"; legacy / alt layouts may use "log"
    log = (
        data.get("log_entries")
        or data.get("log")
        or (data if isinstance(data, list) else [])
    )
    trial_data = data.get("trial_data", [])

    # Determine input/output group names from the groups dict
    groups = data.get("groups") or {}
    group_names = list(groups.keys()) if isinstance(groups, dict) else []
    # Heuristic: first group = input, second = output
    input_group = group_names[0] if len(group_names) >= 1 else "cs_input"
    output_group = group_names[1] if len(group_names) >= 2 else "us_output"

    exp_name = data.get("experiment_name", "")
    training_summary = data.get("training_summary", {}) or {}

    weights = extract_weight_trajectory(log)
    cs_resp = extract_cs_response_trajectory(log, input_group, output_group)

    pre_us = np.array([e["us_rate"] for e in cs_resp["pre_test"]])
    post_us = np.array([e["us_rate"] for e in cs_resp["post_test"]])
    # Combine both associative "training" and reinforcement "rl_training" phases
    training_entries = cs_resp.get("training", []) + cs_resp.get("rl_training", [])
    training_us = np.array([e["us_rate"] for e in training_entries])
    training_t = np.array([e["time_ms"] for e in training_entries])

    # Per-trial learning curve (richer than log-extracted CS-ON windows)
    trials_arr, trial_cs_rates, trial_us_rates, trial_success = extract_per_trial_curve(
        trial_data, input_group, output_group
    )

    # Pavlovian delta
    pavlov_delta = float(post_us.mean() - pre_us.mean()) if pre_us.size and post_us.size else 0.0
    t_stat = 0.0
    if pre_us.size and post_us.size:
        se = np.sqrt(pre_us.var() / max(len(pre_us), 1) + post_us.var() / max(len(post_us), 1))
        if se > 1e-9:
            t_stat = float(pavlov_delta / se)

    # Weight growth (across phases): need a numeric t for R-W fit
    # Log entries are ordered; use the sequence index as abstract time
    w_traj = [w["mean_weight"] for w in weights]
    w_t = np.arange(len(w_traj), dtype=np.float64)
    rw_weights = (
        rescorla_wagner_fit(w_t, np.array(w_traj)) if len(w_traj) >= 3 else None
    )

    # R-W fit on training-phase US rate (if we have enough samples, useful
    # for ASSOCIATIVE_PAIRING where US input drives response)
    rw_training_us = None
    if training_us.size >= 10:
        rw_training_us = rescorla_wagner_fit(training_t - training_t[0], training_us)

    # R-W fit on per-trial data — the richer signal
    # Raw per-trial firing rates are Poisson-noisy; smooth with 20-trial window
    # to expose the underlying associative-strength curve that R-W models.
    rw_trial_us = None
    rw_trial_cs = None
    rw_trial_us_smooth = None
    rw_trial_cs_smooth = None
    smoothed_us = None
    smoothed_cs = None
    if trials_arr.size >= 10:
        rw_trial_us = rescorla_wagner_fit(trials_arr.astype(float), trial_us_rates)
        rw_trial_cs = rescorla_wagner_fit(trials_arr.astype(float), trial_cs_rates)
        # Smoothed variant: rolling mean over W=20 trials
        W = min(20, trials_arr.size // 4)
        if W >= 3:
            kernel = np.ones(W) / W
            smoothed_us = np.convolve(trial_us_rates, kernel, mode="valid")
            smoothed_cs = np.convolve(trial_cs_rates, kernel, mode="valid")
            smoothed_t = np.arange(len(smoothed_us), dtype=float)
            rw_trial_us_smooth = rescorla_wagner_fit(smoothed_t, smoothed_us)
            rw_trial_cs_smooth = rescorla_wagner_fit(smoothed_t, smoothed_cs)

    # Reinforcement-specific metrics: success rate over time, tail success
    n_success_total = int(trial_success.sum()) if trial_success.size else 0
    success_rate_total = float(trial_success.mean()) if trial_success.size else 0.0
    tail_success_rate = float(trial_success[-30:].mean()) if trial_success.size >= 30 else 0.0

    return {
        "file": path,
        "experiment_name": exp_name,
        "input_group": input_group,
        "output_group": output_group,
        "n_log_entries": len(log),
        "n_trials": int(trials_arr.size),
        "pavlov_delta_hz": round(pavlov_delta, 3),
        "pavlov_t_stat": round(t_stat, 2),
        "pre_us_mean": round(float(pre_us.mean()), 3) if pre_us.size else None,
        "pre_us_n": int(pre_us.size),
        "post_us_mean": round(float(post_us.mean()), 3) if post_us.size else None,
        "post_us_n": int(post_us.size),
        "weight_trajectory": weights,
        "rw_fit_weights": rw_weights,
        "rw_fit_training_us": rw_training_us,
        "rw_fit_per_trial_us": rw_trial_us,
        "rw_fit_per_trial_cs": rw_trial_cs,
        "rw_fit_per_trial_us_smooth20": rw_trial_us_smooth,
        "rw_fit_per_trial_cs_smooth20": rw_trial_cs_smooth,
        "trial_us_rates": trial_us_rates.tolist() if trials_arr.size else [],
        "trial_cs_rates": trial_cs_rates.tolist() if trials_arr.size else [],
        "n_success_trials": n_success_total,
        "success_rate_total": round(success_rate_total, 3),
        "tail_success_rate_last30": round(tail_success_rate, 3),
        "training_summary": training_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Glob(s) or file(s)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("No files matched.")
        raise SystemExit(1)

    all_summaries = []
    for f in files:
        try:
            s = analyze_one(f)
        except Exception as e:
            print(f"  FAILED: {f}  ({e})")
            continue
        print(f"\n  {Path(f).name}  [{s.get('experiment_name', '?')}]")
        print(f"    groups: {s['input_group']} -> {s['output_group']}  "
              f"log entries: {s['n_log_entries']}")
        print(
            f"    PRE  in->out: {s['pre_us_mean']} Hz (n={s['pre_us_n']})  |  "
            f"POST in->out: {s['post_us_mean']} Hz (n={s['post_us_n']})"
        )
        print(f"    Pavlov delta: {s['pavlov_delta_hz']:+.3f} Hz  (t = {s['pavlov_t_stat']:+.2f})")
        if s.get("n_success_trials", 0) > 0 or s.get("success_rate_total", 0) > 0:
            print(
                f"    R-STDP success: {s['n_success_trials']}/{s['n_trials']} "
                f"({s['success_rate_total']*100:.1f}%)  tail-30 = "
                f"{s['tail_success_rate_last30']*100:.1f}%"
            )
        if s["weight_trajectory"]:
            w_start = s["weight_trajectory"][0]["mean_weight"]
            w_end = s["weight_trajectory"][-1]["mean_weight"]
            print(f"    W trajectory: {len(s['weight_trajectory'])} snapshots, mean {w_start:.4f} -> {w_end:.4f}")
            if s.get("rw_fit_weights"):
                rw = s["rw_fit_weights"]
                print(
                    f"      RW fit on weights: lambda={rw['lambda']:.4f} tau={rw['tau']:.1f} "
                    f"R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
                )
        if s.get("rw_fit_training_us"):
            rw = s["rw_fit_training_us"]
            print(
                f"    RW fit on training US rate: lambda={rw['lambda']:.2f} Hz  tau={rw['tau']:.0f} ms  "
                f"R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
            )
        if s.get("n_trials", 0) > 0:
            print(f"    n_trials recorded: {s['n_trials']}")
        if s.get("rw_fit_per_trial_us"):
            rw = s["rw_fit_per_trial_us"]
            print(
                f"    RW fit per-trial US rate: lambda={rw['lambda']:.2f} Hz  "
                f"tau={rw['tau']:.0f} trials  R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
            )
        if s.get("rw_fit_per_trial_cs"):
            rw = s["rw_fit_per_trial_cs"]
            print(
                f"    RW fit per-trial CS rate: lambda={rw['lambda']:.2f} Hz  "
                f"tau={rw['tau']:.0f} trials  R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
            )
        if s.get("rw_fit_per_trial_us_smooth20"):
            rw = s["rw_fit_per_trial_us_smooth20"]
            print(
                f"    RW fit smoothed US (W=20): lambda={rw['lambda']:.2f} Hz  "
                f"tau={rw['tau']:.0f} trials  R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
            )
        if s.get("rw_fit_per_trial_cs_smooth20"):
            rw = s["rw_fit_per_trial_cs_smooth20"]
            print(
                f"    RW fit smoothed CS (W=20): lambda={rw['lambda']:.2f} Hz  "
                f"tau={rw['tau']:.0f} trials  R^2={rw['r_squared']:.3f} (n={rw['n_points']})"
            )
        all_summaries.append(s)

    # Aggregate
    if all_summaries:
        deltas = [s["pavlov_delta_hz"] for s in all_summaries if s["pavlov_delta_hz"] is not None]
        t_stats = [s["pavlov_t_stat"] for s in all_summaries if s["pavlov_t_stat"] is not None]
        print(f"\n  === Aggregate across {len(all_summaries)} runs ===")
        if deltas:
            print(f"    Pavlov delta: mean = {np.mean(deltas):+.3f} Hz  std = {np.std(deltas):.3f}")
            print(f"    t-stats: mean |t| = {np.mean(np.abs(t_stats)):.2f}  max |t| = {np.max(np.abs(t_stats)):.2f}")
            n_sig = sum(1 for t in t_stats if abs(t) > 2.0)
            print(f"    Significant at |t|>2: {n_sig}/{len(t_stats)}")


if __name__ == "__main__":
    main()
