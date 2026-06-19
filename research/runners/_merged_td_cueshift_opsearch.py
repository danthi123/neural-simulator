"""ROADMAP #3 — bounded coordinate-descent OP-POINT SEARCH driver for the merged-bridge A-CSC TD cue-shift.

Goal: land the validated standalone A-CSC TD cue-shift (migration r = -0.80/-0.77/-0.89, finding
2026-06-10-N9-TD-cue-shift-A-CSC-GO.md) onto the MERGED nav+conv "one brain"
(build_merged_nav_conv_bridge, co_resident_td_cueshift slice). The science is settled; this is
ENGINEERING ONLY — find the merged operating point that reaches the GO bar migration r < -0.7.

This driver wraps research.runners._merged_td_cueshift_consolidation_derisk.run_td_csc_merged and
runs a small set of (op-point -> r) evaluations, appending each to a results JSON so the search is
budget-defended (re-committable as it runs; never starts an unfinishable run then idle-waits).

DIAGNOSIS (anchor run, het-mask shipped, seed 42, --td-stdp-w-max OFF): the per-tap critic weights
RUN AWAY to ~330 (V(strio) ~360 Hz, far above the standalone's sparse ~70 Hz band) because the
merged config pins the GLOBAL stdp_w_max=400 (the 5a conversational-weight clip) which REMOVES the
per-tap cap (40) the standalone relied on. The het-mask alone does NOT cap weights. => the primary
lever is the per-tap weight clip --td-stdp-w-max (the runner re-clips ONLY the td_value synapses).

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_opsearch \
        --out research/findings/raw/_merged_td_cueshift_opsearch.json \
        --pass NAME --op '{"td_stdp_w_max": 40}' --op '{"td_stdp_w_max": 40, "td_gabab_conductance_max": 0.5}'

Each --op is a JSON dict of OP overrides merged onto the runner's OP baseline. The driver records
(label, op, r_migration, support, gates, key rates) per run. CPU-only (SIM_BACKEND=numpy); each
build is ~100s so keep --n-train small (30) and the op list short per invocation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners._merged_td_cueshift_consolidation_derisk import run_td_csc_merged, OP


def _load(path):
    if path and os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return {"runs": []}
    return {"runs": []}


def _save(path, obj):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)


def _summ(r):
    """Compact one-line summary record from a full run_td_csc_merged result."""
    g = r["gates"]
    return {
        "r_migration": r["r_migration"],
        "peak_early": r["peak_bin_early"], "peak_late": r["peak_bin_late"],
        "cue_v_early": r["cue_v_early_hz"], "cue_v_late": r["cue_v_late_hz"],
        "cue_rate_early": r["cue_rate_early"], "cue_rate_late": r["cue_rate_late"],
        "us_rate_early": r["us_rate_early"], "us_rate_late": r["us_rate_late"],
        "tonic_rate": r["tonic_rate"],
        "w_sub_late_max": max(r["w_sub_late"]) if r["w_sub_late"] else 0.0,
        "support": sum([g["early_burst_at_us"], g["late_burst_at_cue"],
                        g["omission_dip_at_reward"], g["cue_value_grows"]]),
        "gates": g,
        "migration_r_pass": bool(g["migration_r_pass"]),
        "migration_dir_pass": bool(g["migration_dir_pass"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--td-csc-n", type=int, default=8)
    ap.add_argument("--pass", dest="pass_name", type=str, default="search",
                    help="label prefix for this coordinate-descent pass")
    ap.add_argument("--op", action="append", default=[],
                    help="a JSON dict of OP overrides; repeatable (one run per --op)")
    ap.add_argument("--label", action="append", default=[],
                    help="optional per-op label (parallel to --op); auto-generated if omitted")
    ap.add_argument("--out", type=str, default="research/findings/raw/_merged_td_cueshift_opsearch.json")
    args = ap.parse_args()

    store = _load(args.out)
    ops = [json.loads(s) for s in args.op]
    if not ops:
        print("no --op given; nothing to do")
        return
    labels = args.label + [None] * (len(ops) - len(args.label))

    for op_over, lab in zip(ops, labels):
        op = dict(op_over)  # only the overrides; run_td_csc_merged merges onto OP internally
        label = lab or f"{args.pass_name}:" + ",".join(f"{k}={v}" for k, v in sorted(op_over.items()))
        t0 = time.time()
        print(f"\n[opsearch] === {label} ===  (op overrides: {op_over})")
        r = run_td_csc_merged(args.seed, td_csc_n=args.td_csc_n, op=op,
                              n_train_override=args.n_train, verbose=True)
        s = _summ(r)
        s["label"] = label
        s["op"] = op_over
        s["seed"] = args.seed
        s["n_train"] = args.n_train
        s["wall_s"] = round(time.time() - t0, 1)
        store.setdefault("runs", []).append(s)
        _save(args.out, store)
        print(f"[opsearch] r={s['r_migration']:+.3f}  dir={s['migration_dir_pass']}  support={s['support']}/4  "
              f"V(cue) {s['cue_v_early']:.0f}->{s['cue_v_late']:.0f}  w_max_late={s['w_sub_late_max']:.0f}  "
              f"({s['wall_s']:.0f}s)  [saved {args.out}]")

    # Roll-up table of everything in the store (sorted by r, most-negative first).
    runs = sorted(store["runs"], key=lambda x: x["r_migration"])
    print("\n=== OP-SEARCH ROLL-UP (sorted by migration r, most-negative best) ===")
    print(f"{'r':>8}  {'dir':>4}  {'sup':>3}  {'Vcue_e->l':>14}  {'wmaxL':>6}  label")
    for x in runs:
        print(f"{x['r_migration']:+8.3f}  {str(x['migration_dir_pass'])[0]:>4}  {x['support']:>3}  "
              f"{x['cue_v_early']:6.0f}->{x['cue_v_late']:<6.0f}  {x['w_sub_late_max']:6.0f}  {x['label']}")
    best = runs[0] if runs else None
    if best:
        print(f"\n=== BEST: r={best['r_migration']:+.3f} ({best['label']}) | GO bar r<-0.7: "
              f"{'REACHED' if best['r_migration'] < -0.7 else 'not yet'} ===")


if __name__ == "__main__":
    main()
