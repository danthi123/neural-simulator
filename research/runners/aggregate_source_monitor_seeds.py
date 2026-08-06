"""Hands-off self-sweep + aggregator for source-monitor v6 generalization.

Launch one self-contained process, get one earned verdict. This runs EVERY seed
of a phase against the FROZEN v6 mechanism (imported unchanged from
``_laneC_source_monitor_coresidency_gate_v6``), collapses per-seed PASS/FAIL into
one GO/NO-GO, and writes a single aggregate verdict artifact under
``research/findings/raw/source_monitor_v6_generalization/``.

GO requires EVERY seed in the phase to satisfy the frozen acceptance rule; a
single FAIL (or UNDEFINED) is a phase NO-GO. The aggregator only sequences seeds
and collapses verdicts -- it never changes the mechanism, thresholds, or
acceptance rule (that would be gaming the gate). ``held_out`` stays sealed until
``development`` records a GO here, enforced by the runner's ``validate_phase_seed``.

Run (hands-off; provenance is auto-stamped via ``research.runners.__init__``):

    SIM_BACKEND=numpy python -m research.runners.aggregate_source_monitor_seeds \
        --phase development
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    GENERALIZATION_DIR,
    PHASE_SEEDS,
    _development_is_go,
    evaluate_calibration_seed,
)
from tools.lab import attributable_to
from tools.verdict import Verdict

RUNNER = "research/runners/_laneC_source_monitor_coresidency_gate_v6.py"
MECHANISM = "source-monitor-coresidency-v6"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _seed_summary(row: dict) -> dict:
    """Compact per-seed record: status, key margins, and any failing criteria."""

    metrics = row.get("metrics", {})
    components = row.get("components", {})
    failing = sorted(name for name, ok in components.items() if not ok)
    intact_min = metrics.get("minimum_source_margin")
    lesion_min = metrics.get("minimum_lesion_margin")
    # Whose is the weakest-source margin -- competition, or the fixed circuit the
    # competition-lesion arm still runs? This is the redistributive-win question
    # the frozen criterion tests: a fraction of 0 means competition did not lift
    # the weakest source at all (seed 654's failure), not merely that both arms
    # were measured. Attributing it out loud is the class-AT discipline.
    weakest_lift_fraction = (
        attributable_to(
            f"competition weakest-source lift (seed {row['seed']})",
            float(intact_min),
            float(lesion_min),
        )
        if intact_min is not None and lesion_min is not None
        else None
    )
    return {
        "seed": row["seed"],
        "status": row["status"],
        "pass": bool(row["status"].endswith("_PASS")),
        "minimum_source_margin": intact_min,
        "minimum_lesion_margin": lesion_min,
        "weakest_lift_attributable_to_competition": weakest_lift_fraction,
        "learning_off_source_spikes": metrics.get("learning_off_source_spikes"),
        "bounded_loss": metrics.get("bounded_loss"),
        "spendable_surplus": metrics.get("spendable_surplus"),
        "failing_components": failing,
        "elapsed_seconds": row.get("elapsed_seconds"),
    }


def run_phase(phase: str, out_dir: Path = GENERALIZATION_DIR) -> dict:
    """Self-sweep every seed of ``phase`` in one process and collapse the verdict."""

    if phase not in PHASE_SEEDS:
        raise ValueError(f"phase {phase!r} unknown; choose from {tuple(PHASE_SEEDS)}")
    if phase == "held_out" and not _development_is_go():
        raise SystemExit(
            "held_out is sealed: development has not recorded a GO verdict "
            f"({GENERALIZATION_DIR / 'development_verdict.json'}). Run development first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = PHASE_SEEDS[phase]
    t0 = time.time()
    per_seed: list[dict] = []
    fixed_criteria: dict | None = None
    component_counts: list[int] = []
    for seed in seeds:
        row = evaluate_calibration_seed(int(seed), phase=phase)
        (out_dir / f"{phase}_{seed}.json").write_text(json.dumps(row, indent=2))
        per_seed.append(_seed_summary(row))
        component_counts.append(len(row.get("components", {})))
        if fixed_criteria is None:
            fixed_criteria = row.get("fixed_criteria")

    all_pass = all(s["pass"] for s in per_seed)
    undefined = [s["seed"] for s in per_seed if s["status"] == "UNDEFINED"]
    failed = [s["seed"] for s in per_seed if not s["pass"] and s["status"] != "UNDEFINED"]

    # The verdict is EARNED: its integrity preconditions (every seed measured, the
    # frozen criteria applied unchanged, the full control set run) travel with it,
    # and the GO/NO-GO flag is the hypothesis result -- all seeds pass the frozen
    # criteria. A NO-GO is a clean verdict only if the instrument was sound.
    earned = Verdict(f"source-monitor v6 {phase} generalization")
    earned.require(
        "every phase seed produced a defined status (no instrument failure)",
        all(s["status"] != "UNDEFINED" for s in per_seed),
        expect=True,
    )
    earned.require(
        "the frozen v6 acceptance criteria were applied unchanged on every seed",
        fixed_criteria is not None,
        expect=True,
    )
    earned.require(
        "every seed ran the full preregistered control set (20 components)",
        component_counts and min(component_counts) == 20,
        expect=True,
    )
    decided = earned.decide(go=bool(all_pass), verbose=False)
    verdict = decided["status"]

    aggregate = {
        "phase": phase,
        "mechanism": MECHANISM,
        "runner": RUNNER,
        "aggregator": "research/runners/aggregate_source_monitor_seeds.py",
        "frozen_mechanism": True,
        "seeds": [int(s) for s in seeds],
        "verdict": verdict,
        "all_seeds_pass": bool(all_pass),
        "failed_seeds": failed,
        "undefined_seeds": undefined,
        "acceptance_rule": "GO iff every seed satisfies the frozen v6 criteria",
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "fixed_criteria": fixed_criteria,
        "per_seed": per_seed,
        "git_sha": _git_sha(),
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    verdict_path = out_dir / f"{phase}_verdict.json"
    verdict_path.write_text(json.dumps(aggregate, indent=2))
    aggregate["verdict_path"] = str(verdict_path)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-sweep a source-monitor v6 phase and write one aggregate verdict."
    )
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), default="development")
    parser.add_argument("--out-dir", default=str(GENERALIZATION_DIR))
    args = parser.parse_args()

    aggregate = run_phase(args.phase, Path(args.out_dir))
    print(
        f"[aggregate-source-monitor] phase={aggregate['phase']} "
        f"verdict={aggregate['verdict']} seeds={aggregate['seeds']} "
        f"failed={aggregate['failed_seeds']} undefined={aggregate['undefined_seeds']}",
        flush=True,
    )
    for s in aggregate["per_seed"]:
        print(
            f"  seed={s['seed']} status={s['status']} "
            f"min_M={s['minimum_source_margin']} min_L={s['minimum_lesion_margin']} "
            f"learning_off={s['learning_off_source_spikes']} "
            f"failing={s['failing_components']}",
            flush=True,
        )
    print(f"[aggregate-source-monitor] wrote {aggregate['verdict_path']}", flush=True)
    return 0 if aggregate["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
