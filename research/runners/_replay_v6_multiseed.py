"""Hands-off multiseed validation of the FROZEN v6 order-STDP consolidation.

The 2-seed calibration verdict is LANDED: v6 (order-sensitive STDP consolidation,
``_replay_cortical_consolidation_gate_v6_order_stdp.py``) is GO on calibration
seeds 412 and 413 -- both pass every frozen v5+SFA check including
``intact_beats_shuffled_order`` and ``false_recall_bounded<=0.15``, and a
``stdp_sleep=False`` power control causally attributes the ordered-vs-shuffled
margin to the STDP. Seed 413's order margin (+0.0139) clears the +0.01 bar but
is slim, so multiseed validation on the DISJOINT seed partition is the
load-bearing confirmation.

This aggregator runs the IDENTICAL frozen v6 mechanism, config
(``v6.GateConfig()`` defaults -- nothing tuned) and evaluator (``v6.run_seed``,
whose criteria are inherited unchanged from v5+SFA) on a seed GROUP, then
collapses the per-seed GO/NO-GO into ONE earned ``tools.verdict.Verdict``. A
seed PASSES iff its v6 calibration status is CALIBRATION_PROMISING (every frozen
check true, i.e. per-seed verdict GO). The group is GO iff ALL its seeds pass.

Sealed-seed discipline: DEVELOPMENT (414/415/410) runs first; HELD-OUT
(417/418/419) stays sealed and is only run once development is GO. ``--group``
selects which; ``--group heldout`` refuses to run unless a development GO
artifact already exists next to the output.

    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_v6_multiseed \
        --group dev --out research/findings/raw/replay_v5_sfa_order/replay_v6_multiseed_dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


GROUPS = {
    "dev": tuple(v6.DEVELOPMENT_SEEDS),      # 414, 415, 410
    "heldout": tuple(v6.HELD_OUT_SEEDS),     # 417, 418, 419
}

# The two seed-level criteria that were the residual wall; both are ALREADY part
# of v6.run_seed's frozen check set (a PASS requires ALL checks). They are pulled
# out here only for the summary table, NOT re-thresholded.
ORDER_MARGIN_BAR = 0.01
FALSE_RECALL_CEIL = 0.15


def _seed_summary(seed_row: dict) -> dict:
    c = seed_row["calibration"]
    checks = c["checks"]
    return {
        "seed": int(seed_row["seed"]),
        "verdict": seed_row["verdict"],
        "calibration_status": seed_row["calibration_status"],
        "passes_all_checks": bool(all(checks.values())),
        "intact_mean_recovery": c["intact_mean_recovery"],
        "intact_mean_false_recall": c["intact_mean_false_recall"],
        "shuffled_order_recovery": c["control_mean_recovery"]["shuffled_replay_order"],
        "order_margin": c["intact_vs_shuffled_recovery_margin"],
        # Whose is the ordered-vs-shuffled recovery difference on THIS seed?
        # treatment = intact (ordered), control = shuffled_replay_order. Recorded
        # per seed so the multiseed non-generalization can be read as "the order
        # effect the calibration attributed to STDP does not reproduce here".
        "order_recovery_attribution": attributable_to(
            f"order-sensitive STDP consolidation on retest recovery (seed {seed_row['seed']})",
            c["intact_mean_recovery"],
            c["control_mean_recovery"]["shuffled_replay_order"],
        ),
        "intact_beats_shuffled_order": bool(checks["intact_beats_shuffled_order"]),
        "false_recall_bounded": bool(checks["false_recall_bounded"]),
        "stdp_delta_intact": c.get("intact_stdp_cortical_delta"),
        "stdp_delta_shuffled": c.get("shuffled_stdp_cortical_delta"),
        "failing_checks": [k for k, v in checks.items() if not v],
    }


def run_group(group: str, out_path: Path | None) -> dict:
    if group not in GROUPS:
        raise ValueError(f"Unknown group {group!r}; expected one of {tuple(GROUPS)}.")
    seeds = GROUPS[group]

    # Sealed-seed discipline: held-out cannot run until development is GO.
    if group == "heldout":
        dev_art = (out_path.parent if out_path else Path(".")) / "replay_v6_multiseed_dev.json"
        dev_go = False
        if dev_art.exists():
            try:
                dev_go = json.loads(dev_art.read_text())["group_go"] is True
            except Exception:
                dev_go = False
        if not dev_go:
            raise RuntimeError(
                "held-out seeds are SEALED until development is GO; no development GO "
                f"artifact found at {dev_art}. Run --group dev first."
            )

    started = time.time()
    # FROZEN config: v6 defaults, nothing tuned.
    config = v6.GateConfig()
    seed_rows = [v6.run_seed(seed, config) for seed in seeds]
    summaries = [_seed_summary(row) for row in seed_rows]

    earned = Verdict(f"v6 order-STDP consolidation -- {group} multiseed validation")
    # PRECONDITIONS (a valid comparison, not the outcome): every seed produced a
    # decidable per-seed verdict (not UNDEFINED / instrument failure).
    earned.require(
        "every seed produced a decidable (non-UNDEFINED) per-seed verdict",
        all(s["calibration_status"] != "UNDEFINED" for s in summaries),
        expect=True,
    )
    earned.disabled(
        "any mechanism/criterion tuning for multiseed (config is v6.GateConfig() defaults, evaluator inherited from v5+SFA)",
        why="multiseed is a frozen-mechanism generalization test, not a calibration",
    )
    # The OUTCOME: group GO iff every seed passes every frozen v6 check.
    group_go = all(s["passes_all_checks"] for s in summaries)
    decided = earned.decide(go=group_go, verbose=False)

    payload = {
        "gate": "replay_v6_order_stdp_multiseed",
        "group": group,
        "seeds": list(seeds),
        "frozen_config": asdict(config),
        "mechanism": "v6 order-sensitive STDP consolidation (frozen); v5+SFA evaluator inherited unchanged",
        "order_margin_bar": ORDER_MARGIN_BAR,
        "false_recall_ceiling": FALSE_RECALL_CEIL,
        "group_go": bool(group_go),
        "group_status": (
            "UNDEFINED" if any(s["calibration_status"] == "UNDEFINED" for s in summaries)
            else "MULTISEED_GO" if group_go else "MULTISEED_NO_GO"
        ),
        "verdict": decided["status"],
        "preconditions": decided["preconditions"],
        "seed_summaries": summaries,
        "seed_rows": seed_rows,
        "elapsed_seconds": time.time() - started,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = run_group(args.group, args.out)
    # Compact human-readable summary to stdout.
    print(f"[v6 multiseed :: {args.group}] group_status={payload['group_status']} go={payload['group_go']}")
    for s in payload["seed_summaries"]:
        print(
            f"  seed {s['seed']}: {s['verdict']} rec={s['intact_mean_recovery']:.4f} "
            f"false={s['intact_mean_false_recall']:.4f} "
            f"order_margin={s['order_margin']:+.4f} "
            f"(intact {s['intact_mean_recovery']:.4f} vs shuffled {s['shuffled_order_recovery']:.4f}) "
            f"beats_order={s['intact_beats_shuffled_order']} false_ok={s['false_recall_bounded']} "
            f"fails={s['failing_checks'] or 'NONE'}"
        )


if __name__ == "__main__":
    main()
