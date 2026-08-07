"""Thresholded heterosynaptic competition AT ENCODING for the source-monitor lane.

The 4 prior levers against the source-monitor encoding wall all acted at RECALL or
on ACTIVITY and never touched the encoding fan-out; the overlap sweep therefore
stays a NO-GO (strict 1/5, best min M ~= +0.005, ~30x below the F=0.15 floor). The
root cause is at ENCODING: a shared-core episode cell (fires in the seen/heard/self
encodings) potentiates its ``episode -> source_memory[s]`` synapses EQUALLY to all
three sources, so at recall a rival "pedestal" sits under the correct-source "peak"
and the weakest source's margin collapses.

This runner adds ONE knob, ``--lam-hetero`` (heterosynaptic depression coeff), that
applies "protect the peak, depress the pedestal" AFTER encoding, keyed to each
presynaptic episode cell's CUMULATIVE per-source co-activation eligibility (NOT
per-event -- per-event keying collapses to recency, per the CA3 GO and the scoping
doc ``_source_monitor_heterosynaptic_encoding_scoping.md``). The committed GO kernel
``sim.kernels.fused_htm_winner_inactive_depression`` (the CA3 competitive-Hebbian GO
kernel, imported BY REFERENCE -- NO ``sim/`` edit) is applied runner-side to the
``episode -> source`` CSR weights: it is SUBTRACTIVE (``w - dep``) and THRESHOLDED
(only sources whose cumulative eligibility is below ``theta_frac * peak`` are
depressed; the peak and any near-peak source are protected). ``--lam-hetero 0`` is a
strict no-op and is asserted byte-identical to the symmetric-Hebbian overlap NO-GO
(the load-bearing null control).

The instrument is the honest v6 fixed evaluator (per-recall ``reset_dynamical_state``
+ ``_source_margin`` + the zero-learned-weight ``control_strict=False`` guard) reused
verbatim from ``_laneC_source_monitor_overlap_sweep.py`` (the original is left
intact). Only the encoding fan-out is altered; the recall competition circuit and the
frozen v6 thresholds are untouched. NumPy backend, deterministic.

Anti-cheats reported EVERY arm (all must hold for a promising smoke):
  (a) lam_hetero=0 reproduces the overlap NO-GO byte-identically (asserted);
  (b) zero-learned-weight control stays strict=False (no stepping-history artifact);
  (c) commitment-distribution entropy spans all THREE sources (guards recency
      collapse masquerading as a win -- the scoping doc's one real risk);
  (d) reliability guard: all_dominant_correct stays True AND no source's own recall
      rate drops (the peak is protected).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from sim.backend import get_backend, to_host
from sim.kernels import fused_htm_winner_inactive_depression
from research.runners._laneC_source_monitor_coresidency_gate import (
    SOURCES,
    SOURCE_LEARNING_GATE,
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
    CALIBRATION_SEEDS,
    _recall_three,
    evaluate_overlap as _orig_evaluate_overlap,
    make_overlapping_episode_patterns,
)
from tools.lab import attributable_to

HELD_OUT_SEEDS = (655, 656, 657)
DEFAULT_OVERLAPS = (0.2, 0.4)
DEFAULT_THETA_FRAC = 0.9
FLOOR = 0.15  # frozen v6 min-source-margin floor F


class HeteroEncodingGate(SourceMonitorCoresidencyGateV6):
    """v6 gate that accumulates per-(episode-cell, source) co-activation eligibility
    during ``experience`` and can apply thresholded heterosynaptic depression to the
    learned ``episode -> source`` weights after encoding."""

    def __init__(self, *, seed: int, config=None):
        super().__init__(seed=seed, config=config)
        # elig[e, s] = cumulative co-activation of episode cell e with source_memory[s]
        # across ALL encoding events (raw pre*post coincidence integral -- UNSATURATED,
        # unlike the clamped learned weight, so a genuine per-cell source lead survives).
        self._elig = np.zeros((int(self.config.n_episode), len(SOURCES)), dtype=np.float64)

    def experience(
        self,
        episode_pattern: Sequence[int],
        *,
        visual_activity: bool = False,
        auditory_activity: bool = False,
        corollary_discharge: bool = False,
        learning_enabled: bool = True,
        source_afferent_lesion: bool = False,
    ) -> dict:
        """Base ``experience`` VERBATIM plus a side-effect-free per-step co-firing read
        that accumulates the cumulative per-source eligibility. Reading
        ``cp_firing_states`` advances no RNG and mutates no substrate state, so with
        ``lam_hetero=0`` (no depression applied) the weights and recall are byte-identical
        to the base gate -- the null-control guarantee.
        """

        episode_global = self._episode_global_indices(episode_pattern)
        active_sources = self._active_sources(
            visual_activity=visual_activity,
            auditory_activity=auditory_activity,
            corollary_discharge=corollary_discharge,
        )
        pattern_local = np.asarray(episode_pattern, dtype=np.int64)
        active_idx = [SOURCES.index(s) for s in active_sources]
        before = self.weight_summary()
        self.bridge.set_plasticity_gate(
            SOURCE_LEARNING_GATE, 1.0 if learning_enabled else 0.0
        )
        self.bridge.set_transmission_gate(
            "source_afferent_transmission", 0.0 if source_afferent_lesion else 1.0
        )
        try:
            for _ in range(int(self.config.training_cycles)):
                self._drive(episode_global, active_sources)
                for _ in range(int(self.config.training_steps)):
                    self.bridge._run_one_simulation_step()
                    firing = np.asarray(
                        to_host(self.bridge.cp_firing_states), dtype=np.float64
                    )
                    pre = firing[episode_global]  # per driven episode cell
                    for s, si in zip(active_sources, active_idx):
                        post_rate = float(
                            firing[self._source_memory_indices[s]].mean()
                        )
                        if post_rate > 0.0:
                            self._elig[pattern_local, si] += pre * post_rate
                self._rest()
        finally:
            self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
            self.bridge.set_transmission_gate("source_afferent_transmission", 1.0)
            self.bridge.cp_external_input_current[:] = 0.0
        after = self.weight_summary()
        return {
            "active_afferents": list(active_sources),
            "learning_enabled": bool(learning_enabled),
            "source_afferent_lesion": bool(source_afferent_lesion),
            "weight_l1_before": float(before["l1"]),
            "weight_l1_after": float(after["l1"]),
            "weight_l1_delta": float(after["l1"] - before["l1"]),
        }

    def _learned_synapse_pre_post(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(flat_pos, pre_global, post_global) for the learned episode->source synapses.

        Flat CSR positions come from the learning plasticity gate (identical indexing
        to ``cp_connections.data``); pre = CSR row, post = CSR column, exactly like the
        CA3 GO derisk ``_extract_ca3ca3_coincidence``.
        """

        conn = self.bridge.cp_connections
        nnz = int(conn.nnz)
        indptr = np.asarray(to_host(conn.indptr))
        indices = np.asarray(to_host(conn.indices))
        flat = np.asarray(self._learned_synapse_indices(), dtype=np.int64)
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        pre = pre_of[flat].astype(np.int64)
        post = indices[flat].astype(np.int64)
        return flat, pre, post

    def commitment(self, core_local: np.ndarray) -> dict:
        """Per-core-cell committed source (argmax cumulative eligibility) + the
        commitment-distribution entropy across the three sources (anti-cheat c)."""

        core_local = np.asarray(core_local, dtype=np.int64)
        counts = {s: 0 for s in SOURCES}
        committed = []
        for e in core_local:
            row = self._elig[e]
            if float(row.sum()) <= 0.0:
                committed.append(None)
                continue
            si = int(np.argmax(row))
            committed.append(SOURCES[si])
            counts[SOURCES[si]] += 1
        total = sum(counts.values())
        entropy_norm = 0.0
        if total > 0:
            probs = [counts[s] / total for s in SOURCES if counts[s] > 0]
            ent = -sum(p * math.log(p) for p in probs)
            entropy_norm = ent / math.log(len(SOURCES))  # 1.0 == uniform over 3 sources
        return {
            "core_cells": int(core_local.size),
            "committed_counts": counts,
            "commitment_entropy_norm": float(entropy_norm),
            "n_sources_committed": int(sum(1 for s in SOURCES if counts[s] > 0)),
        }

    def apply_hetero_depression(self, lam_hetero: float, theta_frac: float) -> dict:
        """Thresholded, subtractive heterosynaptic depression on the encoding fan-out.

        For each episode cell e that co-activated at all: peak_s = argmax_s elig[e][s].
        A source s is PROTECTED iff elig[e][s] >= theta_frac * elig[e][peak_s] (the peak
        and any near-peak source); every other (pedestal) source is depressed. The kernel
        ``fused_htm_winner_inactive_depression`` subtracts ``dep = (1-protect)*active*
        lam_dep_wi`` per synapse and clips to [0, w_max], with the per-cell subtraction
        amount ``lam_dep_wi = lam_hetero * peak_weight[e]`` (Miller-MacKay subtractive
        normalization scaled by the cell's peak drive: lowers the pedestal WITHOUT
        lowering the peak). lam_hetero=0 -> dep=0 -> a strict no-op.
        """

        if lam_hetero <= 0.0:
            return {"applied": False, "depressed_synapses": 0}

        cp, _ = get_backend()
        conn = self.bridge.cp_connections
        flat, pre, post = self._learned_synapse_pre_post()
        n_syn = int(flat.size)
        n_ep = int(self.config.n_episode)

        # --- vectorized synapse -> (episode-local cell, source column) maps --------
        g2local = np.full(int(self.bridge.cp_connections.shape[0]), -1, dtype=np.int64)
        g2local[np.asarray(self._episode_indices, dtype=np.int64)] = np.arange(
            self._episode_indices.size, dtype=np.int64
        )
        g2src = np.full(g2local.size, -1, dtype=np.int64)
        for si, s in enumerate(SOURCES):
            g2src[np.asarray(self._source_memory_indices[s], dtype=np.int64)] = si
        syn_e = g2local[pre]      # episode-local cell per synapse (-1 if not episode)
        syn_si = g2src[post]      # source column per synapse (-1 if not source_memory)
        valid = (syn_e >= 0) & (syn_si >= 0)

        # --- per-cell peak source + peak eligibility (cumulative) ------------------
        elig = self._elig  # (n_ep, 3)
        row_sum = elig.sum(axis=1)
        peak_val = elig.max(axis=1)
        peak_si = np.argmax(elig, axis=1)
        cell_active = row_sum > 0.0

        # --- per-cell peak-source mean learned weight (the protected magnitude) ----
        w_host = np.asarray(to_host(conn.data), dtype=np.float64)
        syn_w = w_host[flat]
        on_peak = valid & (syn_si == peak_si[np.where(valid, syn_e, 0)])
        peak_w_sum = np.zeros(n_ep, dtype=np.float64)
        peak_w_cnt = np.zeros(n_ep, dtype=np.float64)
        np.add.at(peak_w_sum, syn_e[on_peak], syn_w[on_peak])
        np.add.at(peak_w_cnt, syn_e[on_peak], 1.0)
        peak_w_cell = np.divide(
            peak_w_sum, peak_w_cnt, out=np.zeros_like(peak_w_sum), where=peak_w_cnt > 0
        )

        # --- per-synapse protect mask / active mask / subtraction amount ----------
        protect = np.zeros(n_syn, dtype=np.float64)
        active = np.zeros(n_syn, dtype=np.float64)
        dep_amt = np.zeros(n_syn, dtype=np.float64)
        ve = syn_e[valid]
        vsi = syn_si[valid]
        vactive = cell_active[ve]
        active[valid] = vactive.astype(np.float64)
        protected = elig[ve, vsi] >= theta_frac * peak_val[ve]
        protect[valid] = np.where(vactive, protected.astype(np.float64), 0.0)
        dep_amt[valid] = float(lam_hetero) * peak_w_cell[ve] * vactive.astype(np.float64)

        w = conn.data[cp.asarray(flat, dtype=cp.int64)]
        w = fused_htm_winner_inactive_depression(
            w,
            cp.asarray(protect, dtype=w.dtype),
            cp.asarray(active, dtype=w.dtype),
            cp.asarray(dep_amt, dtype=w.dtype),
            0.0,
            float(self.config.hebbian_max_weight),
        )
        conn.data[cp.asarray(flat, dtype=cp.int64)] = w
        self.bridge._invalidate_coo_cache()
        depressed = int(((1.0 - protect) * active).sum())
        return {
            "applied": True,
            "learned_synapses": int(n_syn),
            "depressed_synapses": depressed,
            "protected_synapses": int((protect * active).sum()),
            "theta_frac": float(theta_frac),
        }


def run_arm(
    seed: int,
    overlap_fraction: float,
    lam_hetero: float,
    theta_frac: float,
    config: SourceMonitorConfigV2 | None = None,
) -> dict:
    """One encoding-lever arm at one (seed, overlap, lam_hetero)."""

    c = config or SourceMonitorConfigV2()
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- Instrument verification: zero-learned-weight control (no experience) -----
    ctrl = HeteroEncodingGate(seed=seed + 30000, config=c)
    ctrl_on = _recall_three(ctrl, patterns)
    ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        ctrl_off = _recall_three(ctrl, patterns)
    finally:
        ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- Real arm: learn, apply hetero depression at ENCODING, then recall --------
    intact = HeteroEncodingGate(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
    commit = intact.commitment(core)
    hetero = intact.apply_hetero_depression(lam_hetero, theta_frac)
    after_hetero = intact.weight_summary()

    on = _recall_three(intact, patterns)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        off = _recall_three(intact, patterns)
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    own_rate = {s: float(on[s]["source_rates"][s]) for s in SOURCES}
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))
    weak_src = min(SOURCES, key=lambda s: margins_L[s])
    weak_attr = attributable_to(
        f"weakest-source ({weak_src}) margin vs SOURCE_COMPETITION_GATE",
        treatment_value=margins_M[weak_src],
        control_value=margins_L[weak_src],
    )

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "lam_hetero": float(lam_hetero),
        "theta_frac": float(theta_frac),
        "core_size": int(core.size),
        "weights_learned_l1": float(learned["l1"]),
        "weights_after_hetero_l1": float(after_hetero["l1"]),
        "control_zero_weight_strict": control_strict,
        "hetero": hetero,
        "commitment": commit,
        "margins_M": margins_M,
        "margins_L": margins_L,
        "own_rate": own_rate,
        "min_margin_M": float(min(margins_M.values())),
        "min_margin_L": float(min(margins_L.values())),
        "min_own_rate": float(min(own_rate.values())),
        "weakest_source_strictly_improved": weakest_strict,
        "weakest_source_by_L": weak_src,
        "weak_margin_attributable_to_competition": weak_attr,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": bool(all(dominant_correct.values())),
        "clears_floor": bool(min(margins_M.values()) >= FLOOR),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def evaluate(
    seed: int,
    overlap_fraction: float,
    lam_hetero: float,
    theta_frac: float,
) -> dict:
    """Treatment arm (lam_hetero) + the lam=0 null baseline, with all anti-cheats."""

    base = run_arm(seed, overlap_fraction, 0.0, theta_frac)
    treat = run_arm(seed, overlap_fraction, lam_hetero, theta_frac)

    # (a) lam_hetero=0 byte-identical to the original overlap NO-GO instrument.
    orig = _orig_evaluate_overlap(int(seed), float(overlap_fraction))
    byte_identical = bool(
        base["min_margin_M"] == orig["min_margin_M"]
        and base["min_margin_L"] == orig["min_margin_L"]
        and base["weights_learned_l1"] == orig["weights_learned_l1"]
    )

    # (d) reliability guard: dominant stays correct AND no source's own recall rate drops.
    no_rate_drop = all(
        treat["own_rate"][s] >= base["own_rate"][s] - 1e-9 for s in SOURCES
    )
    reliability_ok = bool(treat["all_dominant_correct"] and no_rate_drop)

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "lam_hetero": float(lam_hetero),
        "theta_frac": float(theta_frac),
        "baseline_arm": base,
        "treatment_arm": treat,
        "anti_cheats": {
            "a_lam0_byte_identical_to_nogo": byte_identical,
            "b_zero_weight_control_strict_false": bool(
                not base["control_zero_weight_strict"]
                and not treat["control_zero_weight_strict"]
            ),
            "c_commitment_spans_three_sources": bool(
                treat["commitment"]["n_sources_committed"] == 3
            ),
            "c_commitment_entropy_norm": treat["commitment"]["commitment_entropy_norm"],
            "d_reliability_preserved": reliability_ok,
            "d_no_own_rate_drop": bool(no_rate_drop),
        },
        "headline": {
            "min_margin_M_treatment": treat["min_margin_M"],
            "min_margin_L_treatment": treat["min_margin_L"],
            "clears_floor_0.15": treat["clears_floor"],
            "beats_lesion_arm": bool(treat["min_margin_M"] > treat["min_margin_L"]),
            "min_margin_M_baseline": base["min_margin_M"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thresholded heterosynaptic competition at encoding for the "
        "source-monitor lane (numpy, deterministic)."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument("--lam-hetero", type=float, default=1.0)
    parser.add_argument("--theta-frac", type=float, default=DEFAULT_THETA_FRAC)
    parser.add_argument(
        "--mode",
        choices=("calibration", "development", "held_out"),
        default="calibration",
        help="seed-partition label recorded in the artifact (does not gate seeds).",
    )
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=None,
                        help="explicit seed override (parent-supplied validation seeds).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = args.dev_seeds if args.dev_seeds is not None else args.seeds
    rows = []
    for overlap in args.overlaps:
        for seed in seeds:
            row = evaluate(int(seed), float(overlap), float(args.lam_hetero), float(args.theta_frac))
            rows.append(row)
            ac = row["anti_cheats"]
            hl = row["headline"]
            print(
                "[hetero-encoding] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"lam={row['lam_hetero']:.2f} theta={row['theta_frac']:.2f} | "
                f"minM={hl['min_margin_M_treatment']:.4f} "
                f"minL={hl['min_margin_L_treatment']:.4f} "
                f"minM_base={hl['min_margin_M_baseline']:.4f} | "
                f"floor={hl['clears_floor_0.15']} beatsL={hl['beats_lesion_arm']} | "
                f"a_byteid={ac['a_lam0_byte_identical_to_nogo']} "
                f"b_ctrl={ac['b_zero_weight_control_strict_false']} "
                f"c_3src={ac['c_commitment_spans_three_sources']}({ac['c_commitment_entropy_norm']:.2f}) "
                f"d_reliab={ac['d_reliability_preserved']}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_hetero_encoding.py",
        "mode": args.mode,
        "seeds": list(seeds),
        "overlaps": list(args.overlaps),
        "lam_hetero": float(args.lam_hetero),
        "theta_frac": float(args.theta_frac),
        "floor": FLOOR,
        "instrument": "v6 fixed (per-recall reset_dynamical_state) + hetero encoding",
        "kernel": "sim.kernels.fused_htm_winner_inactive_depression (CA3 GO, by reference)",
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[hetero-encoding] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
