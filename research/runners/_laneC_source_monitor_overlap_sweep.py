"""Episode-pattern OVERLAP sweep for the source-monitor competition lane.

The instrument fix (2026-08-06-source-monitor-stepping-history-instrument-FIXED)
showed the core wall: with DISJOINT episode patterns + silent-by-construction
recall the recall-time rival burden is 0, so a source's margin equals its OWN
firing rate and the biased-competition GABA-A circuit has nothing to suppress.
`weakest_source_margin_strictly_improved` (min(M) > min(L)) is therefore
UNSATISFIABLE by any competition mechanism under that protocol.

The named fix is to attack the PROTOCOL, not the (inert) inhibition: let the WORLD
build episode patterns that genuinely OVERLAP, so recalling one source's episode
pattern also drives its rivals through the shared cells (rival_burden > 0). Only
then can the fixed v6 GABA-A competition causally move the weakest source's margin.

Overlap lives in `make_overlapping_episode_patterns` (the world constructing the
episode patterns -- the legitimate host boundary, exactly like `make_episode_patterns`).
The shared core is placed ONLY among the three pure-source patterns (0=seen,
1=heard, 2=self_generated); the mixed (3) and unseen (4) patterns stay disjoint so
the mixed-reinstatement and unseen-silence controls are untouched. overlap=0
reproduces the fully disjoint baseline.

This runner does NOT loosen a frozen criterion: it reuses the v6 fixed-instrument
`SourceMonitorCoresidencyGateV6` (per-recall `reset_dynamical_state`) and the same
`_source_margin`; it measures margins M (competition ON) vs L (competition OFF),
the recall-time rival burden, and the `weakest_source_strictly_improved` predicate
verbatim -- across overlap levels. At every level it first VERIFIES the instrument
with a zero-learned-weight control that must yield strict=False (no stepping-history
artifact). NumPy backend, deterministic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    SOURCES,
    _dominant_source,
    _source_margin,
)
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    SourceMonitorCoresidencyGateV6,
)
from tools.lab import attributable_to

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
DEFAULT_OVERLAPS = (0.0, 0.2, 0.4)


def make_overlapping_episode_patterns(
    seed: int,
    config: SourceMonitorConfigV2,
    overlap_fraction: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """World-supplied episode patterns with a controlled shared core.

    A ``core`` of ``round(overlap_fraction * episode_pattern_size)`` cells is
    shared by the three pure-source patterns (seen/heard/self); each pure pattern
    fills the remainder with its own unique cells. The mixed pattern (3) and the
    unseen pattern (4) are fully disjoint from all others. overlap=0 -> a fully
    disjoint set (the original protocol). Returns (patterns, core_indices).
    """

    c = config
    psize = int(c.episode_pattern_size)
    k = int(round(float(overlap_fraction) * psize))
    k = max(0, min(k, psize))
    needed = k + 3 * (psize - k) + 2 * psize
    if needed > int(c.n_episode):
        raise ValueError("overlap layout exceeds the episode population")
    order = np.random.default_rng(int(seed)).permutation(int(c.n_episode))
    cur = 0
    core = np.sort(order[cur : cur + k]).astype(np.int64)
    cur += k
    patterns: list[np.ndarray] = []
    for _ in range(3):  # seen, heard, self_generated
        uniq = order[cur : cur + (psize - k)]
        cur += psize - k
        patterns.append(np.sort(np.concatenate([core, uniq])).astype(np.int64))
    for _ in range(2):  # mixed, unseen -- disjoint by construction
        patterns.append(np.sort(order[cur : cur + psize]).astype(np.int64))
        cur += psize
    return patterns, core


def _rival_burden(record: dict, expected: str) -> float:
    """Sum of the non-expected source rates during ``expected``'s own recall."""

    rates = record["source_rates"]
    return float(sum(rates[s] for s in SOURCES if s != expected))


def _recall_three(
    gate: SourceMonitorCoresidencyGateV6, patterns: Sequence[np.ndarray]
) -> dict[str, dict]:
    return {
        "seen": gate.recall(patterns[0]),
        "heard": gate.recall(patterns[1]),
        "self_generated": gate.recall(patterns[2]),
    }


def evaluate_overlap(
    seed: int,
    overlap_fraction: float,
    config: SourceMonitorConfigV2 | None = None,
) -> dict:
    """Run the v6 fixed-instrument competition at one overlap level, one seed."""

    c = config or SourceMonitorConfigV2()
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- Instrument verification: zero-learned-weight control (no experience) ---
    # Weights are zeroed at construction; without experience nothing can fire, so
    # the fixed instrument must yield strict=False at this overlap level (proves
    # the margin is not a stepping-history artifact).
    ctrl = SourceMonitorCoresidencyGateV6(seed=seed + 30000, config=c)
    ctrl_on = _recall_three(ctrl, patterns)
    ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        ctrl_off = _recall_three(ctrl, patterns)
    finally:
        ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- The real arm: learn, then competition ON (M) vs OFF (L) ----------------
    intact = SourceMonitorCoresidencyGateV6(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()

    on = _recall_three(intact, patterns)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        off = _recall_three(intact, patterns)
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    rival_burden_on = {s: _rival_burden(on[s], s) for s in SOURCES}
    rival_burden_off = {s: _rival_burden(off[s], s) for s in SOURCES}
    dominant_correct = {
        s: bool(_dominant_source(on[s]) == s) for s in SOURCES
    }
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))

    # Attribute the WEAKEST source's margin (the one that sets min(L), the binding
    # constraint) to the competition gate: treatment = competition ON, control =
    # competition OFF. A negative fraction means the manipulation REDUCED the
    # margin (competition hurt the weak source) -- the whole point of this gate is
    # to force us to subtract the two arms rather than eyeball min(M) vs min(L).
    weak_src = min(SOURCES, key=lambda s: margins_L[s])
    weak_margin_attribution = attributable_to(
        f"weakest-source ({weak_src}) margin vs SOURCE_COMPETITION_GATE",
        treatment_value=margins_M[weak_src],
        control_value=margins_L[weak_src],
    )

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "core_size": int(core.size),
        "episode_pattern_size": int(c.episode_pattern_size),
        "weights_initial_l1": float(initial["l1"]),
        "weights_learned_l1": float(learned["l1"]),
        "control_zero_weight_strict": control_strict,
        "control_min_M": float(min(ctrl_M.values())),
        "control_min_L": float(min(ctrl_L.values())),
        "rival_burden_off": rival_burden_off,
        "rival_burden_on": rival_burden_on,
        "min_rival_burden_off": float(min(rival_burden_off.values())),
        "margins_M": margins_M,
        "margins_L": margins_L,
        "min_margin_M": float(min(margins_M.values())),
        "min_margin_L": float(min(margins_L.values())),
        "weakest_source_strictly_improved": weakest_strict,
        "weakest_source_by_L": weak_src,
        "weak_margin_attributable_to_competition": weak_margin_attribution,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": bool(all(dominant_correct.values())),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep episode-pattern overlap for the source-monitor competition lane."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(CALIBRATION_SEEDS),
    )
    parser.add_argument(
        "--overlaps",
        type=float,
        nargs="+",
        default=list(DEFAULT_OVERLAPS),
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rows = []
    for overlap in args.overlaps:
        for seed in args.seeds:
            row = evaluate_overlap(int(seed), float(overlap))
            rows.append(row)
            print(
                "[overlap-sweep] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"core={row['core_size']} "
                f"ctrl_strict={row['control_zero_weight_strict']} "
                f"min_rival_off={row['min_rival_burden_off']:.4f} "
                f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                f"strict={row['weakest_source_strictly_improved']} "
                f"dom_ok={row['all_dominant_correct']}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_overlap_sweep.py",
        "seeds": list(args.seeds),
        "overlaps": list(args.overlaps),
        "instrument": "v6 fixed (per-recall reset_dynamical_state)",
        "rows": rows,
    }
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[overlap-sweep] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
