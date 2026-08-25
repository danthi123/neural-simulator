"""Diagnostic: run the FROZEN v6 order-STDP config on arbitrary seeds.

Bypasses the v6/v5/v5s bounded-seed guard (read-only diagnostic; no mechanism
change) so we can measure the order effect + full battery on the DECISIVE seeds
42 43 44 100 101 102 and quantify the operating-point / backend drift.

Usage:
    SIM_BACKEND=numpy python -m research.runners._order_recalib_diag --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate_v5 as v5
from research.runners import _replay_cortical_consolidation_gate_v5_sfa as v5s
from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6


def _passthrough(seeds):
    return tuple(int(s) for s in seeds)


# Lift the bounded-seed guard for this diagnostic only.
v5.validate_calibration_seeds = _passthrough
v5s.validate_calibration_seeds = _passthrough
v6.validate_calibration_seeds = _passthrough


def diag_seed(seed: int, config: v6.GateConfig) -> dict:
    conditions = {c: v6.run_condition(seed, c, config) for c in v6.CONDITIONS}
    verdict = v6._calibration_verdict(conditions)
    ctrl = verdict["control_mean_recovery"]
    row = {
        "seed": int(seed),
        "calibration_status": verdict["calibration_status"],
        "intact_recovery": verdict["intact_mean_recovery"],
        "intact_margin": verdict["intact_mean_margin"],
        "intact_false_recall": verdict["intact_mean_false_recall"],
        "shuffled_recovery": ctrl["shuffled_replay_order"],
        "order_recovery_margin": verdict["intact_vs_shuffled_recovery_margin"],
        "intact_beats_shuffled_order": verdict["checks"]["intact_beats_shuffled_order"],
        "intact_stdp_delta": verdict["intact_stdp_cortical_delta"],
        "shuffled_stdp_delta": verdict["shuffled_stdp_cortical_delta"],
        "stdp_delta_ratio": (
            verdict["intact_stdp_cortical_delta"] / verdict["shuffled_stdp_cortical_delta"]
            if verdict.get("shuffled_stdp_cortical_delta")
            else None
        ),
        "no_sleep_recovery": ctrl["no_sleep"],
        "stdpoff_note": "run separately",
        "checks_failed": [k for k, v in verdict["checks"].items() if not v],
        "control_recovery": ctrl,
        "control_false_recall": verdict["control_mean_false_recall"],
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    # optional overrides for the recalibration sweep
    ap.add_argument("--sfa-d", type=float, default=None)
    ap.add_argument("--stdp-a-plus", type=float, default=None)
    ap.add_argument("--stdp-a-minus", type=float, default=None)
    ap.add_argument("--stdp-w-max-scale", type=float, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.sfa_d is not None:
        overrides["target_sfa_d_increment"] = args.sfa_d
    if args.stdp_a_plus is not None:
        overrides["stdp_a_plus"] = args.stdp_a_plus
    if args.stdp_a_minus is not None:
        overrides["stdp_a_minus"] = args.stdp_a_minus
    if args.stdp_w_max_scale is not None:
        overrides["stdp_w_max_scale"] = args.stdp_w_max_scale
    config = v6.GateConfig(**overrides) if overrides else v6.GateConfig()

    started = time.time()
    rows = [diag_seed(s, config) for s in args.seeds]
    n_order = sum(1 for r in rows if r["intact_beats_shuffled_order"])
    payload = {
        "diag": "v6_order_stdp_frozen_on_decisive_seeds",
        "overrides": overrides,
        "n_seeds": len(rows),
        "n_order_pass": n_order,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
