"""Competitive (heterosynaptic) ENCODING for the source-monitor lane.

The fair-inhibition + own-gain NO-GO (2026-08-06-source-monitor-fair-inhibition-
and-own-gain-both-NO-GO-...) isolated the binding constraint at ENCODING, not
recall: with symmetric Hebbian learning each SHARED overlap (core) episode cell
potentiates EQUALLY to every source it co-activated with, so at recall the shared
cells drive the rival source pools at the same ceiling as the target and no
recall-time mechanism (fair inhibition rebounds; own-gain saturates) can separate
them. The named next mechanism is COMPETITIVE learning at encoding: heterosynaptic
LTD / outgoing-weight conservation on the episode->source synapses so a shared cell
that potentiates to one source is DEPRESSED to the others (its total OUTGOING
weight is conserved). A shared cell then stops driving all sources equally.

This runner implements OUTGOING-WEIGHT CONSERVATION as a LOCAL synaptic rule on
the substrate's real ``episode->source`` synapses (von der Malsburg / Chistiakova-
Volgushev presynaptic heterosynaptic normalization): for each PREsynaptic episode
neuron i, if the sum S_i of its OWN outgoing plastic weights exceeds a budget B,
its outgoing weights are divisively rescaled to sum B. The rule uses ONLY each
cell's own outgoing weights -- NO source label, NO correct answer, NO host oracle
(unlike the own-gain ceiling probe, which scaled a NAMED source's synapses). A
UNIQUE episode cell projects to one source (fan-out 1, S_i = B) and is untouched;
a SHARED core cell projects to k sources (S_i ~ k*B) and each of its outgoing
weights is cut to ~B/k. At recall the target keeps its full-strength unique-cell
drive while the rivals lose ~(k-1)/k of the shared-cell drive that buried the
weakest source.

The instrument is UNCHANGED: the v6 fixed recall protocol (per-recall
``reset_dynamical_state`` + settle-to-quiescence), the overlapping episode
patterns from ``make_overlapping_episode_patterns``, ``_source_margin``, the
``weakest_source_strictly_improved`` predicate and the 0.15 floor are reused
verbatim. At every overlap level the zero-learned-weight control must still yield
strict=False. Only the encoding gains a conservation step. NumPy backend,
deterministic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
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
from research.runners._laneC_source_monitor_overlap_sweep import (
    make_overlapping_episode_patterns,
    _rival_burden,
)
from sim.backend import get_backend, to_host
from tools.lab import attributable_to

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
DEFAULT_OVERLAPS = (0.0, 0.2, 0.4)


class SourceMonitorHeteroGate(SourceMonitorCoresidencyGateV6):
    """v6 fixed-instrument gate with outgoing-weight conservation on encoding.

    Everything except the added conservation step is inherited from v6/v5/base:
    the per-recall ``reset_dynamical_state`` + settle-to-quiescence read, the
    Hebbian episode->source potentiation, the biased-competition circuit (left in
    place, gated by ``SOURCE_COMPETITION_GATE``), the aPFC/ACC read-out. The
    conservation rule is a LOCAL heterosynaptic normalization of each presynaptic
    episode cell's own outgoing plastic weights -- it is called explicitly during
    encoding (see ``evaluate_hetero_overlap``), never at recall.
    """

    def _plastic_presyn_rows(self) -> np.ndarray:
        """Presynaptic (row) neuron index for each learnable episode->source synapse.

        ``cp_connections`` is CSR with rows = presynaptic and ``.indices`` =
        postsynaptic (verified against the own-gain probe, which selects post ==
        source-memory). The row for each nnz entry is recovered from ``indptr``.
        """

        indptr = np.asarray(to_host(self.bridge.cp_connections.indptr), dtype=np.int64)
        n_rows = int(indptr.size - 1)
        rows_all = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
        learned = np.asarray(self._learned_synapse_indices(), dtype=np.int64)
        return rows_all[learned]

    def conserve_outgoing_weights(self, budget: float) -> dict:
        """LOCAL outgoing-weight conservation on the episode->source synapses.

        For each presynaptic episode neuron i, let S_i be the sum of the absolute
        values of its OWN outgoing plastic weights. If S_i > budget, every one of
        those weights is multiplied by budget / S_i (divisive presynaptic
        normalization -> the cell's total outgoing weight is conserved at budget).
        No source label or correct answer is used: the rule reads only each cell's
        own fan-out. Returns a small diagnostic of the shared-vs-unique split.
        """

        xp, _ = get_backend()
        learned = np.asarray(self._learned_synapse_indices(), dtype=np.int64)
        pre = self._plastic_presyn_rows()
        data = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
        w = data[learned].copy()
        n_rows = int(np.asarray(to_host(self.bridge.cp_connections.indptr)).size - 1)

        sums = np.zeros(n_rows, dtype=np.float64)
        np.add.at(sums, pre, np.abs(w))
        scale = np.ones(n_rows, dtype=np.float64)
        over = sums > float(budget)
        scale[over] = float(budget) / np.maximum(sums[over], 1e-12)

        # Fan-out (number of source pools a cell projects to with a meaningful
        # weight) -- diagnostic of shared (fan-out>1) vs unique (fan-out==1) cells.
        active = np.abs(w) > 1e-6
        fanout = np.zeros(n_rows, dtype=np.int64)
        np.add.at(fanout, pre[active], 1)

        w_new = w * scale[pre]
        data[learned] = w_new
        self.bridge.cp_connections.data[:] = xp.asarray(data, dtype=self.bridge.cp_connections.data.dtype)

        shared = fanout[pre] > 1
        return {
            "budget": float(budget),
            "n_presyn_conserved": int(over.sum()),
            "n_shared_synapses": int(shared.sum()),
            "n_unique_synapses": int((~shared & active).sum()),
            "shared_weight_mean_before": float(np.abs(w[shared]).mean()) if shared.any() else 0.0,
            "shared_weight_mean_after": float(np.abs(w_new[shared]).mean()) if shared.any() else 0.0,
            "unique_weight_mean_before": float(np.abs(w[~shared & active]).mean()) if (~shared & active).any() else 0.0,
            "unique_weight_mean_after": float(np.abs(w_new[~shared & active]).mean()) if (~shared & active).any() else 0.0,
        }


def _recall_three(
    gate: SourceMonitorHeteroGate, patterns: Sequence[np.ndarray]
) -> dict[str, dict]:
    return {
        "seen": gate.recall(patterns[0]),
        "heard": gate.recall(patterns[1]),
        "self_generated": gate.recall(patterns[2]),
    }


def _encode(gate: SourceMonitorHeteroGate, patterns: Sequence[np.ndarray]) -> None:
    gate.experience(patterns[0], visual_activity=True)
    gate.experience(patterns[1], auditory_activity=True)
    gate.experience(patterns[2], corollary_discharge=True)
    gate.experience(patterns[3], visual_activity=True, auditory_activity=True)


def evaluate_hetero_overlap(
    seed: int,
    overlap_fraction: float,
    config: SourceMonitorConfigV2 | None = None,
    budget: float | None = None,
    conserve: bool = True,
) -> dict:
    """Competitive-encoding evaluation at one overlap level, one seed.

    ``conserve=False`` reproduces the v6 baseline (no encoding competition) under
    THIS runner's harness, so the conservation delta is measured like-for-like.
    """

    c = config or SourceMonitorConfigV2()
    B = float(c.hebbian_max_weight) if budget is None else float(budget)
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- Instrument verification: zero-learned-weight control (no experience) ---
    # No experience => weights stay zero; conservation of a zero fan-out is a
    # no-op, so the fixed instrument must still yield strict=False at this overlap.
    ctrl = SourceMonitorHeteroGate(seed=seed + 30000, config=c)
    if conserve:
        ctrl.conserve_outgoing_weights(B)  # no-op on zero weights; proves harmlessness
    ctrl_on = _recall_three(ctrl, patterns)
    ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        ctrl_off = _recall_three(ctrl, patterns)
    finally:
        ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- The real arm: encode, conserve outgoing weights, then measure ----------
    intact = SourceMonitorHeteroGate(seed=seed, config=c)
    initial = intact.weight_summary()
    _encode(intact, patterns)
    learned = intact.weight_summary()
    conserve_diag = intact.conserve_outgoing_weights(B) if conserve else None
    conserved = intact.weight_summary()

    # M = competition ON, L = competition OFF -- both AFTER conservation, so the
    # frozen predicate is evaluated on the competitively-encoded substrate.
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
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))
    floor_cleared = bool(min(margins_M.values()) >= 0.15)

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
        "mechanism": "outgoing-weight-conservation encoding" if conserve else "v6 baseline (no encoding competition)",
        "conserve": bool(conserve),
        "budget": B,
        "weights_initial_l1": float(initial["l1"]),
        "weights_learned_l1": float(learned["l1"]),
        "weights_conserved_l1": float(conserved["l1"]),
        "conservation": conserve_diag,
        "control_zero_weight_strict": control_strict,
        "control_min_M": float(min(ctrl_M.values())),
        "control_min_L": float(min(ctrl_L.values())),
        "rival_burden_off": rival_burden_off,
        "rival_burden_on": rival_burden_on,
        "min_rival_burden_off": float(min(rival_burden_off.values())),
        "max_rival_burden_on": float(max(rival_burden_on.values())),
        "margins_M": margins_M,
        "margins_L": margins_L,
        "min_margin_M": float(min(margins_M.values())),
        "min_margin_L": float(min(margins_L.values())),
        "weakest_source_strictly_improved": weakest_strict,
        "min_margin_M_meets_0p15_floor": floor_cleared,
        "weakest_source_by_L": weak_src,
        "weak_margin_attributable_to_competition": weak_margin_attribution,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": bool(all(dominant_correct.values())),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Competitive (heterosynaptic outgoing-weight conservation) encoding sweep."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="outgoing-weight budget per presynaptic cell (default hebbian_max_weight)",
    )
    parser.add_argument(
        "--no-conserve",
        action="store_true",
        help="baseline: skip conservation (v6 encoding under this harness)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    conserve = not args.no_conserve
    rows = []
    for overlap in args.overlaps:
        for seed in args.seeds:
            row = evaluate_hetero_overlap(
                int(seed), float(overlap), budget=args.budget, conserve=conserve
            )
            rows.append(row)
            cd = row["conservation"]
            split = (
                f"shared {cd['shared_weight_mean_before']:.0f}->{cd['shared_weight_mean_after']:.0f} "
                f"uniq {cd['unique_weight_mean_before']:.0f}->{cd['unique_weight_mean_after']:.0f}"
                if cd
                else "(no-conserve)"
            )
            print(
                "[hetero-encoding] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"core={row['core_size']} B={row['budget']:.0f} "
                f"ctrl_strict={row['control_zero_weight_strict']} "
                f"maxRival_on={row['max_rival_burden_on']:.4f} "
                f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                f"strict={row['weakest_source_strictly_improved']} "
                f"floor={row['min_margin_M_meets_0p15_floor']} "
                f"dom_ok={row['all_dominant_correct']} | {split}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_hetero_encoding_sweep.py",
        "seeds": list(args.seeds),
        "overlaps": list(args.overlaps),
        "budget": args.budget,
        "conserve": conserve,
        "instrument": "v6 fixed (per-recall reset_dynamical_state + settle)",
        "mechanism": "presynaptic outgoing-weight conservation (heterosynaptic LTD) at encoding",
        "rows": rows,
    }
    if args.json:
        from pathlib import Path

        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[hetero-encoding] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
