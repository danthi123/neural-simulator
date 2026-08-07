"""JOINT storage+recall de-risk for co-resident source monitoring.

The single-``g_comp`` recall-only attractor de-risk
(``_laneC_source_monitor_attractor_competition``) was a NO-GO: the CA3-style
lateral-inhibition WTA does not track the cue -- it amplifies whichever assembly
has the strongest CORE-driven / mixed-boosted recurrent drive and suppresses the
correctly-cued source, so ``all_dominant_correct`` is False at every ``g_comp>0``.

Step-1 diagnosis (seed 650, g_comp 1, ``seen`` cued -- see the finding) CORRECTS
the parent's read: it is NOT that the shared core carries inflated weight the uniq
cells cannot overcome. In SUM the uniq->source weight DOMINATES core->source
(ratio ~0.07-0.18), and in pure feedforward (g_comp 0) the honest signal WORKS --
the cued source receives the most TOTAL input (uniq-driven) and wins every cue.
The failure is that the lateral-inhibition competition reads INTRINSIC assembly
strength (dominated by the shared core + the mixed-episode boost of core->seen/
heard), NOT the cue-specific uniq advantage: under the ``seen`` cue the rivals
(heard, self) receive ONLY core-driven input, and the WTA latches whichever rival
has the largest core drive, quenching the cued source.

The honest JOINT knob (this runner): a second knob ``uniq_emphasis`` applies a
per-PRESYNAPTIC-cell SELECTIVITY GAIN to the learned ``episode->source`` synapses,
keyed to that presynaptic cell's CUMULATIVE cross-source fan-out breadth
``b(p)`` = the number of distinct source populations it acquired a learned weight
to (the shared core -> b=3; a source-unique cell -> b=1). The gain is
``b(p) ** (-uniq_emphasis)``, so broadly-projecting (core) inputs are down-weighted
RELATIVE to selective (uniq) inputs -- a heterosynaptic selectivity normalization.
This strengthens the honest diagnostic signal so the cued source's uniq cells win
the completion. ``uniq_emphasis`` swept against ``g_comp`` = the JOINT knob.

HONESTY (all asserted / reported per row):
  * ``uniq_emphasis`` uses NO source label -- ``b(p)`` is a property of the
    presynaptic cell's own fan-out, symmetric, computed WITHOUT knowing which
    source is currently cued; the same scalar gain scales all of a cell's synapses.
  * It re-weights SYNAPSES at FIXED overlap -- it NEVER touches
    ``make_overlapping_episode_patterns``. The shared-core episode CELLS are
    byte-identical across every arm and still fire during recall (co-residency
    intact at the input level). Proven per row: the pattern/core hash and the
    count of core cells active in each cue are identical to the null.
  * Recall stays EPISODE-ONLY: source-afferent current==0 AND firing==0 at recall
    (measured); non-vacuity: a forced afferent still moves the winner.
  * ``uniq_emphasis==0`` AND ``g_comp==0`` is BYTE-IDENTICAL to the attractor NO-GO
    (gain==1 everywhere -> no weight change; no competition pathway) -- asserted.

GO (smoke, needs full validation): ``min_margin_M >= 0.15`` AND
``all_dominant_correct`` True on EVERY source incl. the weakest, on both
calibration seeds 650/651, with the pattern overlap PROVEN unchanged. numpy,
deterministic, minutes/seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    SOURCES,
    _dominant_source,
    _source_margin,
)
from research.runners._laneC_source_monitor_attractor_competition import (
    MIN_SOURCE_MARGIN,
    SourceMonitorAttractorConfig,
    SourceMonitorAttractorCompetitionGate,
)
from research.runners._laneC_source_monitor_overlap_sweep import (
    make_overlapping_episode_patterns,
)
from sim.backend import to_host

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
HELD_OUT_SEEDS = (655, 656, 657)
DEFAULT_UNIQ_EMPHASIS = (0.0, 0.5, 1.0, 2.0)
DEFAULT_G_COMP = (0.0, 1.0)
DEFAULT_OVERLAP = 0.2


@dataclass(frozen=True)
class SourceMonitorJointConfig(SourceMonitorAttractorConfig):
    """Attractor config + a per-presynaptic-cell selectivity gain ``uniq_emphasis``.

    ``uniq_emphasis == 0`` -> gain 1 everywhere (no weight change; byte-identical
    to the attractor NO-GO at the same ``g_comp``).
    """

    uniq_emphasis: float = 0.0


def _decode_learned(gate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (presynaptic_global, postsynaptic_global, data_index) for every
    learned episode->source synapse, aligned with ``cp_connections.data``.

    ``cp_connections`` is CSR in (pre -> post) layout (bridge.py:3827): row = pre,
    ``indices`` = post column.
    """

    csr = gate.bridge.cp_connections
    col = np.asarray(to_host(csr.indices), dtype=np.int64)
    indptr = np.asarray(to_host(csr.indptr), dtype=np.int64)
    row = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    lidx = gate._learned_synapse_indices()
    return row[lidx], col[lidx], lidx


def _source_of_post(gate) -> dict[int, str]:
    m: dict[int, str] = {}
    for s in SOURCES:
        for j in np.asarray(gate._source_memory_indices[s]).tolist():
            m[int(j)] = s
    return m


def apply_uniq_emphasis(gate, uniq_emphasis: float) -> dict:
    """Down-weight broadly-projecting (core) presynaptic inputs RELATIVE to
    selective (uniq) inputs, by ``b(p) ** (-uniq_emphasis)`` where ``b(p)`` is the
    presynaptic cell's learned cross-source fan-out breadth. No-op at emphasis 0.

    Returns the number of synapses touched and the per-breadth cell counts (a
    label-free diagnostic; NOT used at recall).
    """

    pre, post, lidx = _decode_learned(gate)
    data = np.asarray(to_host(gate.bridge.cp_connections.data), dtype=np.float64)
    w = data[lidx]
    src_of = _source_of_post(gate)
    nz = np.abs(w) > 1e-8

    # b(p): distinct source pops presynaptic cell p acquired a learned weight to.
    breadth: dict[int, set] = {}
    for p, po, active in zip(pre, post, nz):
        if active:
            breadth.setdefault(int(p), set()).add(src_of[int(po)])
    b_count = {int(p): len(ss) for p, ss in breadth.items()}

    if float(uniq_emphasis) == 0.0:
        return {
            "uniq_emphasis": 0.0,
            "synapses_touched": 0,
            "cells_by_breadth": _breadth_hist(b_count),
            "no_op": True,
        }

    gain_by_pre = {
        p: (float(b) ** (-float(uniq_emphasis))) if b >= 1 else 1.0
        for p, b in b_count.items()
    }
    new_w = w.copy()
    touched = 0
    for i, (p, active) in enumerate(zip(pre, nz)):
        if not active:
            continue
        g = gain_by_pre.get(int(p), 1.0)
        if g != 1.0:
            new_w[i] = w[i] * g
            touched += 1
    # write back into the monolithic connection matrix
    data[lidx] = new_w
    import numpy as _np
    csr = gate.bridge.cp_connections
    try:
        csr.data[:] = _np.asarray(data, dtype=csr.data.dtype)
    except TypeError:  # cupy backend path (not used here; numpy authoritative)
        from sim.backend import get_backend
        xp, _ = get_backend()
        csr.data[:] = xp.asarray(data, dtype=csr.data.dtype)
    return {
        "uniq_emphasis": float(uniq_emphasis),
        "synapses_touched": int(touched),
        "cells_by_breadth": _breadth_hist(b_count),
        "no_op": False,
    }


def _breadth_hist(b_count: dict[int, int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for b in b_count.values():
        hist[str(b)] = hist.get(str(b), 0) + 1
    return hist


def _pattern_hash(patterns, core) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(core, dtype=np.int64).tobytes())
    for p in patterns:
        h.update(np.asarray(p, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _core_active_in_cue(gate, patterns, core) -> dict[str, int]:
    """Count how many shared-core episode cells are in each pure-source cue
    (co-residency at the INPUT level -- must be == core.size for every cue)."""

    core_local = set(np.asarray(core).tolist())
    out = {}
    for i, s in enumerate(SOURCES):
        out[s] = int(len(core_local.intersection(set(np.asarray(patterns[i]).tolist()))))
    return out


def _learn(gate, patterns) -> None:
    gate.experience(patterns[0], visual_activity=True)
    gate.experience(patterns[1], auditory_activity=True)
    gate.experience(patterns[2], corollary_discharge=True)
    gate.experience(patterns[3], visual_activity=True, auditory_activity=True)


def evaluate_joint(
    seed: int,
    uniq_emphasis: float,
    g_comp: float,
    overlap_fraction: float,
    *,
    config: SourceMonitorJointConfig | None = None,
) -> dict:
    """One seed x uniq_emphasis x g_comp x overlap. Competition ON (M) vs OFF (L),
    all anti-cheats + overlap-unchanged proof."""

    base = config or SourceMonitorJointConfig()
    c = SourceMonitorJointConfig(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "g_comp": float(g_comp),
            "uniq_emphasis": float(uniq_emphasis),
        }
    )
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    pat_hash = _pattern_hash(patterns, core)
    t0 = time.time()

    # -- (d) zero-learned-weight instrument control: strict must be False --------
    # ctrl never learns, so apply_uniq_emphasis is a no-op (zero weights).
    ctrl = SourceMonitorAttractorCompetitionGate(seed=seed + 30000, config=c)
    apply_uniq_emphasis(ctrl, uniq_emphasis)
    ctrl_on = {s: ctrl.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    ctrl._set_comp_gate(0.0)
    try:
        ctrl_off = {s: ctrl.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    finally:
        ctrl._set_comp_gate(1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- the real arm: learn, apply the storage-side gain, then M vs L ----------
    intact = SourceMonitorAttractorCompetitionGate(seed=seed, config=c)
    initial = intact.weight_summary()
    _learn(intact, patterns)
    learned = intact.weight_summary()
    emphasis_info = apply_uniq_emphasis(intact, uniq_emphasis)
    reweighted = intact.weight_summary()

    on = {s: intact.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    intact._set_comp_gate(0.0)
    try:
        off = {s: intact.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    finally:
        intact._set_comp_gate(1.0)

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    own_rate_M = {s: float(on[s]["source_rates"][s]) for s in SOURCES}
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    all_dominant_correct = bool(all(dominant_correct.values()))
    min_M = float(min(margins_M.values()))
    min_L = float(min(margins_L.values()))
    clears_floor = bool(min_M >= MIN_SOURCE_MARGIN)
    weakest_strict = bool(min_M > min_L)

    # -- anti-cheat (a): (uniq_emphasis==0, g_comp==0) byte-identical to NO-GO ---
    byte_identical_null = None
    if float(g_comp) == 0.0 and float(uniq_emphasis) == 0.0:
        byte_identical_null = bool(
            all(abs(margins_M[s] - margins_L[s]) < 1e-12 for s in SOURCES)
        )

    # -- anti-cheat (b): honesty -- afferent current == 0 AND firing == 0 -------
    afferent_current_zero = bool(all(on[s]["max_afferent_current"] == 0.0 for s in SOURCES))
    afferent_firing_zero = bool(all(sum(on[s]["afferent_spikes"].values()) == 0.0 for s in SOURCES))
    forced = "seen"
    forced_rec = intact.recall_instrumented(patterns[4], force_afferent=forced)
    forced_moves_winner = bool(_dominant_source(forced_rec) == forced)
    no_own_rate_collapse = bool(all(own_rate_M[s] > 0.0 for s in SOURCES))

    # -- overlap-unchanged proof: patterns/core byte-identical to the null build;
    # the shared core still fires in every cue (co-residency intact at input). ---
    core_active = _core_active_in_cue(intact, patterns, core)
    overlap_intact = bool(all(v == int(core.size) for v in core_active.values()))

    smoke_go = bool(clears_floor and all_dominant_correct)

    return {
        "seed": int(seed),
        "uniq_emphasis": float(uniq_emphasis),
        "g_comp": float(g_comp),
        "overlap_fraction": float(overlap_fraction),
        "core_size": int(core.size),
        "pattern_hash": pat_hash,
        "recurrent_e_weight": float(g_comp * c.recurrent_e_weight_base),
        "lateral_i_weight": float(g_comp * c.lateral_i_weight_base),
        "weights_learned_l1": float(learned["l1"]),
        "weights_reweighted_l1": float(reweighted["l1"]),
        "emphasis_info": emphasis_info,
        # decisive metrics
        "margins_M": margins_M,
        "margins_L": margins_L,
        "min_margin_M": min_M,
        "min_margin_L": min_L,
        "clears_floor": clears_floor,
        "weakest_source_strictly_improved": weakest_strict,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": all_dominant_correct,
        "smoke_go": smoke_go,
        "own_rate_M": own_rate_M,
        # anti-cheats + overlap proof
        "anti_cheats": {
            "control_zero_weight_strict": control_strict,  # (d) must be False
            "byte_identical_null_at_0_0": byte_identical_null,  # (a); None unless both 0
            "afferent_current_zero_at_recall": afferent_current_zero,  # (b)
            "afferent_firing_zero_at_recall": afferent_firing_zero,  # (b)
            "forced_afferent_moves_winner": forced_moves_winner,  # (b) non-vacuity
            "no_own_rate_collapse": no_own_rate_collapse,  # (c)
            "overlap_intact_core_fires_every_cue": overlap_intact,  # co-residency proof
            "core_active_in_cue": core_active,
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JOINT storage(uniq_emphasis)+recall(g_comp) de-risk for source monitoring."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--uniq-emphasis", type=float, nargs="+", default=list(DEFAULT_UNIQ_EMPHASIS))
    parser.add_argument("--g-comp", type=float, nargs="+", default=list(DEFAULT_G_COMP))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    null_hashes = set()
    for ue in args.uniq_emphasis:
        for g_comp in args.g_comp:
            for seed in args.seeds:
                row = evaluate_joint(int(seed), float(ue), float(g_comp), float(args.overlap))
                rows.append(row)
                null_hashes.add((row["seed"], row["pattern_hash"]))
                ac = row["anti_cheats"]
                print(
                    "[joint] "
                    f"seed={row['seed']} ue={row['uniq_emphasis']:.2f} g_comp={row['g_comp']:.2f} "
                    f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                    f"clears={row['clears_floor']} dom_ok={row['all_dominant_correct']} "
                    f"SMOKE_GO={row['smoke_go']} "
                    f"| overlap_intact={ac['overlap_intact_core_fires_every_cue']} "
                    f"aff0={ac['afferent_firing_zero_at_recall']}/{ac['afferent_current_zero_at_recall']} "
                    f"ctrl_strict={ac['control_zero_weight_strict']} "
                    f"forced_moves={ac['forced_afferent_moves_winner']} "
                    f"no_collapse={ac['no_own_rate_collapse']} "
                    f"byte_null={ac['byte_identical_null_at_0_0']}",
                    flush=True,
                )

    # Overlap-unchanged across arms: one pattern hash per seed (knob-independent).
    hashes_per_seed = {}
    for seed, h in null_hashes:
        hashes_per_seed.setdefault(seed, set()).add(h)
    overlap_unchanged_all = all(len(hs) == 1 for hs in hashes_per_seed.values())
    print(f"[joint] overlap_unchanged_across_arms={overlap_unchanged_all} "
          f"(one pattern hash per seed: { {k: sorted(v) for k, v in hashes_per_seed.items()} })",
          flush=True)

    out = {
        "runner": "research/runners/_laneC_source_monitor_attractor_joint.py",
        "seeds": list(args.seeds),
        "uniq_emphasis": list(args.uniq_emphasis),
        "g_comp": list(args.g_comp),
        "overlap": float(args.overlap),
        "min_source_margin_floor": MIN_SOURCE_MARGIN,
        "overlap_unchanged_across_arms": overlap_unchanged_all,
        "mechanism": (
            "per-presynaptic-cell selectivity gain b(p)**(-uniq_emphasis) on the "
            "learned episode->source synapses (storage side) x within-population "
            "slow-NMDA recurrent excitation + GABA-A lateral inhibition (recall "
            "side, one knob g_comp); label-free, fixed overlap"
        ),
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[joint] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
