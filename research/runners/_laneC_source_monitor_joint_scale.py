"""JOINT storage+recall+SCALE de-risk for co-resident source monitoring.

The decisive cross-gap test for the identity-discrimination boundary shared by #3
(source monitoring), #4 (replay consolidation) and the gap#5 DG arc.  The synthesis
(``research/findings/raw/_crossgap_identity_discrimination_synthesis.md``) isolates
the ONE variable never properly tested: SCALE, in the attractor-competition +
storage-separation config.

Prior rungs on this exact wall:
  * ``_laneC_source_monitor_attractor_competition`` -- single ``g_comp`` recall-only
    attractor: NO-GO (the lateral-inhibition WTA reads INTRINSIC assembly strength,
    dominated by the shared core + the mixed-episode boost, NOT the cue-specific
    uniq advantage, so ``all_dominant_correct`` is False at every ``g_comp>0``).
  * ``_laneC_source_monitor_attractor_joint`` -- + a storage-side selectivity gain
    ``uniq_emphasis`` (``b(p)**(-uniq_emphasis)``, label-free, fixed overlap): still
    NO-GO at n=12 (nothing clears ``min_margin >= 0.15``; the weakest source
    ``self_generated`` sits at a negative or near-zero margin).
  * The 2026-08-07 feedforward scale probe scaled the WRONG thing (the readout pop,
    no attractor recurrence) and found no gain.

The un-run decisive config (this runner): attractor competition (``g_comp``, fixed
recurrent-E : lateral-I ratio) x storage-side separation (``uniq_emphasis``) x
``n_source`` swept 12 -> 48 -> 96.  ``n_source`` (= ``n_source_memory``) is the
CA3-GO's load-bearing variable (``2026-07-14-ca3-competitive-hebbian-formation``
root-caused the residual completion failure as SCALE-bounded; the joint-uniq NO-GO
literally reported "weakest pinned at n12").  At CA3 scale the redundant sparse
assemblies give reliable winner selection.

SCALE construction (all in the WORLD/host boundary -- pattern layout + population
sizes -- NEVER a source label):
  * Population sizes (episode, source-afferent, source-memory, aPFC, ACC,
    interneuron) are ALL multiplied by F = n_source / 12.  Both the episode
    ASSEMBLY and the source-memory ASSEMBLY grow proportionally.
  * The overlap FRACTION is held EXACTLY constant across n: the shared core is
    ``base_k * F`` cells out of ``base_psize * F`` = ``base_k / base_psize`` -- the
    identical co-residency fraction the n=12 rung ran at.  ``do NOT`` reduce the
    pattern overlap to make scale look better -- that is the goalpost-moving cheat;
    it is asserted equal across n per run.
  * Fan-in structural weights (recurrent-E, lateral-I, source->interneuron,
    afferent->source, source->aPFC/ACC, and the Hebbian cap) are scaled by 1 / F to
    PRESERVE THE PER-NEURON OPERATING POINT.  Each postsynaptic neuron now has F x
    more presynaptic partners (density 1.0); without the 1/F the total synaptic
    current would scale with F and every neuron would saturate -- a raw-current
    confound, NOT a redundancy test.  This is the honest scale test: same
    per-neuron drive, F x more neurons sharing the code.

Everything else -- the mechanism, the decisive anti-cheat, the honesty guards -- is
reused BY REFERENCE from ``_laneC_source_monitor_attractor_joint.evaluate_joint``
and ``_laneC_source_monitor_attractor_competition`` (no ``sim/`` edit; prior runners
left intact).

  ==> THE DECISIVE ANTI-CHEAT (unchanged): ``all_dominant_correct`` must stay True on
      EVERY source INCLUDING the structurally-weakest ``self_generated``.  A high
      ``g_comp`` that wins margin by ALWAYS silencing two pools regardless of
      correctness is the rich-get-richer cheat and is a NO-GO -- reported per row.

GO (smoke, needs full validation): ``min_margin_M >= 0.15`` AND
``min_margin_M > min_margin_L`` AND ``all_dominant_correct`` True on EVERY source,
on both calibration seeds 650/651, with the realized overlap fraction PROVEN
constant across n.  numpy, deterministic, minutes/seed.

A GO at n>=48 that did NOT hold at n=12 is the load-bearing result (scale closes the
cluster).  A NO-GO with ``all_dominant_correct`` False even at n=96 is the DECISIVE
BOUNDARY VERDICT: identity of same-core co-resident sources is not recoverable on a
point-neuron rate substrate -> the honest next substrate is dendritic compartments
(per the synthesis; NOT built here).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_attractor_competition import (
    MIN_SOURCE_MARGIN,
)
from research.runners._laneC_source_monitor_attractor_joint import (
    SourceMonitorJointConfig,
    evaluate_joint,
)

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
HELD_OUT_SEEDS = (655, 656, 657)

# n_source (= n_source_memory) sweep.  12 is the prior NO-GO rung; 48/96 raise the
# source-memory + episode + assembly sizes proportionally (CA3 scale redundancy).
DEFAULT_N_SOURCE = (12, 48, 96)
DEFAULT_UNIQ_EMPHASIS = (0.0, 1.0, 2.0)
DEFAULT_G_COMP = (0.0, 1.0)
DEFAULT_OVERLAP = 0.2

# The n=12 reference sizes (SourceMonitorConfig defaults).  F = n_source / 12 scales
# every population; fan-in weights scale by 1 / F.
BASE_N_SOURCE = 12
BASE_PSIZE = 12  # SourceMonitorConfig.episode_pattern_size default


def _scale_factor(n_source: int) -> int:
    if int(n_source) % BASE_N_SOURCE != 0 or int(n_source) < BASE_N_SOURCE:
        raise ValueError(
            f"n_source must be a positive multiple of {BASE_N_SOURCE}; got {n_source}"
        )
    return int(n_source) // BASE_N_SOURCE


def _effective_overlap(overlap: float) -> tuple[float, int]:
    """Return (effective_overlap, base_core_size) so the REALIZED overlap fraction
    (core_size / episode_pattern_size) is EXACTLY the same at every scale.

    ``make_overlapping_episode_patterns`` computes ``k = round(overlap * psize)``.
    Passing ``base_k / base_psize`` at scaled ``psize = base_psize * F`` yields
    ``k = round(base_k * F) = base_k * F`` (base_k, F integers) -> realized fraction
    ``base_k * F / (base_psize * F) = base_k / base_psize`` for all F.
    """

    base_k = int(round(float(overlap) * BASE_PSIZE))
    base_k = max(0, min(base_k, BASE_PSIZE))
    return base_k / BASE_PSIZE, base_k


def scaled_config(
    n_source: int,
    uniq_emphasis: float,
    g_comp: float,
    base: SourceMonitorJointConfig | None = None,
) -> SourceMonitorJointConfig:
    """A joint config with every population scaled by F = n_source/12 and every
    fan-in weight scaled by 1/F (operating-point-preserving redundancy scale)."""

    base = base or SourceMonitorJointConfig()
    f = _scale_factor(n_source)
    fields = {k: getattr(base, k) for k in base.__dataclass_fields__}
    # -- populations x F (episode + both assemblies + interneuron pool) ----------
    fields["n_episode"] = base.n_episode * f
    fields["episode_pattern_size"] = base.episode_pattern_size * f
    fields["n_source_afferent"] = base.n_source_afferent * f
    fields["n_source_memory"] = int(n_source)
    fields["n_apfc"] = base.n_apfc * f
    fields["n_acc"] = base.n_acc * f
    fields["n_source_interneuron"] = base.n_source_interneuron * f
    # -- fan-in weights / F (preserve per-neuron total drive; density stays 1.0) --
    fields["recurrent_e_weight_base"] = base.recurrent_e_weight_base / f
    fields["lateral_i_weight_base"] = base.lateral_i_weight_base / f
    fields["source_to_interneuron_weight"] = base.source_to_interneuron_weight / f
    fields["source_afferent_weight"] = base.source_afferent_weight / f
    fields["source_to_apfc_weight"] = base.source_to_apfc_weight / f
    fields["source_to_acc_weight"] = base.source_to_acc_weight / f
    fields["hebbian_max_weight"] = base.hebbian_max_weight / f
    # -- swept knobs -------------------------------------------------------------
    fields["g_comp"] = float(g_comp)
    fields["uniq_emphasis"] = float(uniq_emphasis)
    return SourceMonitorJointConfig(**fields)


def evaluate_scale(
    seed: int,
    n_source: int,
    uniq_emphasis: float,
    g_comp: float,
    overlap: float,
) -> dict:
    """One (seed, n_source, uniq_emphasis, g_comp) cell.  Reuses evaluate_joint's
    full anti-cheat + honesty machinery; augments with the scale bookkeeping and
    the strict three-part GO (min M >= 0.15 AND min M > min L AND all_dominant)."""

    f = _scale_factor(n_source)
    eff_overlap, base_k = _effective_overlap(overlap)
    cfg = scaled_config(n_source, uniq_emphasis, g_comp)
    row = evaluate_joint(int(seed), float(uniq_emphasis), float(g_comp), eff_overlap, config=cfg)

    psize = int(cfg.episode_pattern_size)
    realized = float(row["core_size"]) / float(psize)
    clears = bool(row["clears_floor"])
    strict = bool(row["weakest_source_strictly_improved"])
    dom_ok = bool(row["all_dominant_correct"])
    # The frozen three-part GO gate.  g_comp==0 has M==L so strict is False by
    # construction (the mechanism under test IS the competition).
    smoke_go_strict = bool(clears and strict and dom_ok)
    # Decision-relevant secondary flag: does pure feedforward + storage-separation
    # ALONE clear the floor with all sources dominant (competition not even needed)?
    feedforward_clears = bool(float(g_comp) == 0.0 and clears and dom_ok)

    row["n_source"] = int(n_source)
    row["scale_factor"] = int(f)
    row["episode_pattern_size"] = psize
    row["requested_overlap"] = float(overlap)
    row["effective_overlap"] = float(eff_overlap)
    row["realized_overlap_fraction"] = realized
    row["weight_scaling"] = {
        "policy": "fan-in weights / F (operating-point-preserving)",
        "recurrent_e_weight_base": float(cfg.recurrent_e_weight_base),
        "lateral_i_weight_base": float(cfg.lateral_i_weight_base),
        "source_afferent_weight": float(cfg.source_afferent_weight),
        "hebbian_max_weight": float(cfg.hebbian_max_weight),
    }
    row["smoke_go_strict"] = smoke_go_strict
    row["feedforward_clears"] = feedforward_clears
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JOINT storage(uniq_emphasis) x recall(g_comp) x SCALE(n_source) "
        "de-risk for co-resident source monitoring."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--n-source", type=int, nargs="+", default=list(DEFAULT_N_SOURCE))
    parser.add_argument(
        "--uniq-emphasis", type=float, nargs="+", default=list(DEFAULT_UNIQ_EMPHASIS)
    )
    parser.add_argument("--g-comp", type=float, nargs="+", default=list(DEFAULT_G_COMP))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for n_source in args.n_source:
        for ue in args.uniq_emphasis:
            for g_comp in args.g_comp:
                for seed in args.seeds:
                    row = evaluate_scale(
                        int(seed), int(n_source), float(ue), float(g_comp), float(args.overlap)
                    )
                    rows.append(row)
                    ac = row["anti_cheats"]
                    print(
                        "[joint-scale] "
                        f"seed={row['seed']} n={row['n_source']} F={row['scale_factor']} "
                        f"ue={row['uniq_emphasis']:.2f} g_comp={row['g_comp']:.2f} "
                        f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                        f"clears={row['clears_floor']} strict={row['weakest_source_strictly_improved']} "
                        f"dom_ok={row['all_dominant_correct']} "
                        f"GO={row['smoke_go_strict']} ff_clears={row['feedforward_clears']} "
                        f"| ovl={row['realized_overlap_fraction']:.4f} core={row['core_size']}/{row['episode_pattern_size']} "
                        f"overlap_intact={ac['overlap_intact_core_fires_every_cue']} "
                        f"aff0={ac['afferent_firing_zero_at_recall']}/{ac['afferent_current_zero_at_recall']} "
                        f"ctrl_strict={ac['control_zero_weight_strict']} "
                        f"forced_moves={ac['forced_afferent_moves_winner']} "
                        f"no_collapse={ac['no_own_rate_collapse']} "
                        f"byte_null={ac['byte_identical_null_at_0_0']}",
                        flush=True,
                    )

    # -- Prove the realized overlap fraction is IDENTICAL across every n_source ----
    # (the co-residency anti-cheat: scale must NOT reduce overlap).  One realized
    # fraction per n_source; all must coincide.
    realized_by_n = {}
    for r in rows:
        realized_by_n.setdefault(r["n_source"], set()).add(round(r["realized_overlap_fraction"], 12))
    distinct = sorted({v for vs in realized_by_n.values() for v in vs})
    overlap_constant_across_n = len(distinct) == 1
    print(
        f"[joint-scale] overlap_fraction_constant_across_n={overlap_constant_across_n} "
        f"(realized per n: { {k: sorted(v) for k, v in realized_by_n.items()} })",
        flush=True,
    )

    # -- The decisive question, per the synthesis --------------------------------
    go_cells = [
        (r["n_source"], r["uniq_emphasis"], r["g_comp"])
        for r in rows
        if r["smoke_go_strict"]
    ]
    # A cell is a GO only if it holds on BOTH calibration seeds.
    from collections import Counter

    cell_go_counts = Counter(
        (r["n_source"], r["uniq_emphasis"], r["g_comp"])
        for r in rows
        if r["smoke_go_strict"]
    )
    n_seeds = len({r["seed"] for r in rows})
    both_seed_go = sorted(k for k, v in cell_go_counts.items() if v == n_seeds)
    any_scale_helped = any(
        r["smoke_go_strict"] and r["scale_factor"] > 1 for r in rows
    )
    print(
        f"[joint-scale] both_seed_GO_cells={both_seed_go} "
        f"any_GO_rows={len(go_cells)} scale_helped={any_scale_helped}",
        flush=True,
    )

    out = {
        "runner": "research/runners/_laneC_source_monitor_joint_scale.py",
        "seeds": list(args.seeds),
        "n_source": list(args.n_source),
        "uniq_emphasis": list(args.uniq_emphasis),
        "g_comp": list(args.g_comp),
        "requested_overlap": float(args.overlap),
        "min_source_margin_floor": MIN_SOURCE_MARGIN,
        "overlap_fraction_constant_across_n": overlap_constant_across_n,
        "realized_overlap_by_n": {str(k): sorted(v) for k, v in realized_by_n.items()},
        "both_seed_go_cells": [list(c) for c in both_seed_go],
        "any_scale_helped": any_scale_helped,
        "go_gate": "min_margin_M >= 0.15 AND min_margin_M > min_margin_L AND all_dominant_correct, on BOTH seeds",
        "scale_policy": (
            "populations x F=n_source/12 (episode + both assemblies + interneuron); "
            "fan-in weights / F (operating-point-preserving); overlap fraction held "
            "EXACTLY constant via effective_overlap = base_k / base_psize"
        ),
        "mechanism": (
            "per-presynaptic-cell selectivity gain b(p)**(-uniq_emphasis) on learned "
            "episode->source synapses (storage) x within-population slow-NMDA recurrent "
            "excitation + GABA-A lateral inhibition (recall, one knob g_comp) x "
            "n_source scale (CA3 redundancy)"
        ),
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[joint-scale] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
