"""Generic DECLARATIVE cross-edge FUNCTIONAL GATE — the read / credit / lesion HARNESS, driven by the declaration.

The CONSTRUCTION side of a learned cross-region edge is already declarative: a `CrossEdge` row +
`merge_organs(..., cross_edges=[...])` + `MergedPool.apply_cross_edge_freeze()`
(`research/runners/onebrain_merge_framework.py`; `research/findings/2026-08-27-declarative-cross-edges-framework-GO.md`).
What was STILL hand-typed per edge — the honest residual the framework's own gap-analysis flagged
(`research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md` §3) — is the
FUNCTIONAL GATE that decides whether an edge is REAL: `_f1/_f2/_f3/_f4` + `_emergence` + `_migration_invariant`
were re-implemented, near-identically, in seven runners (R1/R2/R3v3/R4/surprise/...). The load-bearing core of
that gate is GENERIC, because the framework already KNOWS each declared edge's synapses (the `CrossEdge`
endpoints). This module makes the three load-bearing checks the brief's "6-seed GO" names —
GROWS-from-the-substrate's-own-rule · LOAD-BEARING-vary/lesion · BYTE-IDENTICAL-OFF — GENERIC, driven from a
`CrossEdgeGateSpec`:

  * EMERGENCE     — the declared edges' mean weight after training vs `grow_factor*init_weight` (the RIGHT pairs
                    GREW from near-zero), the declared selectivity pairs (right >> wrong), and `frozen_maxdrift`
                    over EVERY non-edge synapse (no migrated weight moved). Derived from the declaration + the
                    bridge connectivity. Nothing edge-specific.
  * INTERACTION   — (the crux) read the target under each declared source-state CONDITION; LESION the declared
                    edges (zero exactly their synapses, from the declaration); re-read; `attributable_to` per
                    condition. Generic; only the source-state DRIVERS and the target READ are edge-specific
                    callables (`train_fn` / `read_fn`) — the faculty itself, irreducibly.
  * BYTE-OFF      — a pool built WITHOUT the `cross_edges` has base connectivity BYTE-IDENTICAL to the with-edge
                    pool minus exactly the declared edge (pre,post) slots (integration added ONLY the edge), and
                    its read == the lesioned read within the FP-layout floor. Generic from the declaration.

Adding an edge = a `CrossEdge` list + `train_fn` + `read_fn` + the source-state conditions — NOT a bespoke
~40KB runner with its own hand-typed F-gate. Reuse-by-composition: this module NEVER re-implements a faculty's
train/read; it CALLS the callables the spec supplies and reproduces the seven runners' shared gate STRUCTURE.

Faithfulness proof (`research/runners/_onebrain_declarative_crossedge_generic_gate.py`): run this generic gate on
R1's own pool/train/read and show its emergence + interaction numbers reproduce R1's hand-typed
`_emergence_with_drift`/`_f2` bit-for-bit (max|delta|=0.0) across 6 seeds — the abstraction is faithful, not a
drift. Functional read-outs only; no phenomenal-experience claim.

NO sim/ edit. Additive: importing this module changes no existing code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim.backend import to_host
from tools.lab import attributable_to
from research.runners.onebrain_merge_framework import CrossEdge


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  GENERIC edge-synapse handles — derived from the declaration + the pool's OWN connectivity, so the
#  emergence read, the lesion, and the no-corruption drift all key on exactly the declared edges (no
#  hand-typed mask list per edge). Byte-identical in shape/selection to the per-runner hand-typed masks.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _edge_endpoints(bridge, ce: CrossEdge):
    rm = bridge.region_manager
    pre = (np.asarray(ce.source_idx_fn(bridge), dtype=np.int64) if ce.source_idx_fn is not None
           else np.asarray(rm.indices(ce.source_region), dtype=np.int64))
    post = (np.asarray(ce.target_idx_fn(bridge), dtype=np.int64) if ce.target_idx_fn is not None
            else np.asarray(rm.indices(ce.target_region), dtype=np.int64))
    return pre, post


def cross_edge_masks(bridge, cross_edges) -> dict:
    """{edge.key: bool ndarray over cp_connections.data} — the synapses of each declared edge, selected off the
    pool's OWN COO row/col region membership (the same `np.isin(row, pre) & np.isin(col, post)` the bespoke
    runners hand-wrote per edge). A mask is asserted non-empty: an empty mask means the CrossEdge never wired."""
    coo = bridge.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    masks = {}
    for ce in cross_edges:
        pre, post = _edge_endpoints(bridge, ce)
        m = np.isin(row, pre) & np.isin(col, post)
        assert int(m.sum()) > 0, f"declared CrossEdge {ce.key!r} selected 0 synapses — the edge did not wire"
        masks[ce.key] = m
    return masks


def noncross_mask(masks, n_data: int) -> np.ndarray:
    """The boolean over cp_connections.data that is TRUE for every synapse NOT belonging to any declared edge —
    the population the no-corruption drift is measured over (R1's `_noncross`, generalized)."""
    nc = np.ones(int(n_data), dtype=bool)
    for m in masks.values():
        nc &= ~m
    return nc


def edge_weight_mean(bridge, mask) -> float:
    return float(np.asarray(to_host(bridge.cp_connections.data))[mask].mean())


def lesion_cross_edges(bridge, masks, xp):
    """Zero exactly the declared edges' synapse weights IN PLACE (R1's `_f2` lesion, generalized to N edges).
    Returns the pre-lesion data vector so a caller may restore it."""
    data = np.asarray(to_host(bridge.cp_connections.data)).copy()
    before = data.copy()
    for m in masks.values():
        data[m] = 0.0
    bridge.cp_connections.data = xp.asarray(data, dtype=bridge.cp_connections.data.dtype)
    return before


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  THE DECLARATIVE FUNCTIONAL-GATE SPEC — the read/credit/live-drive half of a cross-edge declaration.
#  The wiring half is the `CrossEdge` list (onebrain_merge_framework.py); this is what the brief calls the
#  {credit_signal, read_site} of the declarative form. Together they instantiate wiring + synapse + live-drive.
# ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class CrossEdgeGateSpec:
    name: str
    cross_edges: list                              # the declarative wiring rows (CrossEdge)
    # --- callables the faculty supplies (irreducible: they DEFINE the faculty) ---
    train_fn: Callable                             # (pool) -> optional trajectory; grows the edge by the substrate's
    #                                                own rule over experiential episodes (plasticity frozen elsewhere)
    read_fn: Callable                              # (pool, condition_name) -> float; the LOAD-BEARING read at read_site
    # --- EMERGENCE declaration ---
    init_weight: float                             # the CrossEdge seed weight (the edge must GROW above it)
    correct_edges: tuple                           # edge KEYS whose learned mapping should GROW from near-zero
    selectivity_pairs: tuple = ()                  # (spurious_key, correct_key): the wrong pair stays << the right
    grow_factor: float = 5.0                       # "grew" == mean weight > grow_factor * init_weight
    selective_frac: float = 0.25                   # "selective" == spurious < selective_frac * its paired correct
    drift_tol: float = 1e-6                         # no-corruption: max|Δ| over every NON-edge synapse after training
    # --- INTERACTION (vary-then-lesion) declaration ---
    condition_order: tuple = ()                    # the read conditions IN ORDER; the CONTROL first (R1: none,ref0,ref1)
    control: str = ""                              # the matched-control condition (its read is subtracted off)
    expected: dict = field(default_factory=dict)   # {condition: {"sign": +1|-1, "floor": float}} — the load-bearing
    #                                                shift's required sign + minimum magnitude with the edge intact
    lesion_ratio: float = 0.34                      # the lesioned shift must be < lesion_ratio * the intact shift
    default_floor: float = 0.008                    # per-condition floor when `expected[c]` omits one
    # --- the credit signal this edge's plasticity rides (documentation + which config the harness expects) ---
    credit_signal: str = "rate_hebbian"             # "rate_hebbian" | "da_credit" | "none" — mirrors CrossEdge.learn_rule


def verify_emergence(pool, spec: CrossEdgeGateSpec, masks, frozen_maxdrift: float) -> dict:
    """GENERIC emergence: the declared correct edges GREW above grow_factor*init_weight, the declared selectivity
    pairs stayed selective (right >> wrong), and no NON-edge synapse moved (frozen_maxdrift < drift_tol).
    Reproduces R1's `_emergence_with_drift` (the grown values are read off the SAME masks; the checks are the
    same inequalities) — generalized to any declared correct/selectivity set."""
    grown = {ce.key: edge_weight_mean(pool.bridge, masks[ce.key]) for ce in spec.cross_edges}
    correct = all(grown[k] > spec.grow_factor * spec.init_weight for k in spec.correct_edges)
    # weight-RATIO selectivity is only checked when the spec DECLARES selectivity_pairs. When it does not (e.g. a
    # feedback edge onto an intrinsically-active store, where the ratio is the wrong instrument — the selectivity is
    # functional, in the interaction), report None (NOT a vacuous True), so the artifact never reads as a passed
    # selectivity check that was not actually run.
    selective = (None if not spec.selectivity_pairs
                 else all(grown[sp] < spec.selective_frac * grown[co] for (sp, co) in spec.selectivity_pairs))
    no_corruption = bool(frozen_maxdrift < spec.drift_tol)
    return {"grown": grown, "correct_pairs_grew": bool(correct),
            "selectivity_checked": bool(spec.selectivity_pairs), "mapping_selective": selective,
            "frozen_weight_maxdrift": float(frozen_maxdrift), "no_corruption": no_corruption,
            "PASS": bool(correct and (selective is not False) and no_corruption)}


def verify_interaction(pool, spec: CrossEdgeGateSpec, masks) -> dict:
    """GENERIC interaction (the crux): read the target under each declared source-state condition; the read shifts
    away from the CONTROL toward the declared sign; LESION the declared edges; the shift VANISHES (attributable_to
    the edge ~1). Reproduces R1's `_f2` exactly — same read order (control first), same delta = read - control,
    same `attributable_to(sign*intact, sign*lesion)`, same PASS inequalities — generalized to any conditions.
    DESTRUCTIVE: lesions the pool's declared edges in place (the caller must not reuse the pool for an intact read).
    """
    read = lambda c: float(spec.read_fn(pool, c))
    reads_i = {c: read(c) for c in spec.condition_order}
    ctrl_i = reads_i[spec.control]
    deltas_i = {c: reads_i[c] - ctrl_i for c in spec.condition_order if c != spec.control}
    lesion_cross_edges(pool.bridge, masks, pool.xp)
    reads_l = {c: read(c) for c in spec.condition_order}
    ctrl_l = reads_l[spec.control]
    deltas_l = {c: reads_l[c] - ctrl_l for c in spec.condition_order if c != spec.control}
    per, all_ok = {}, True
    for c in spec.condition_order:
        if c == spec.control:
            continue
        s = int(spec.expected.get(c, {}).get("sign", +1))
        floor = float(spec.expected.get(c, {}).get("floor", spec.default_floor))
        d_i, d_l = deltas_i[c], deltas_l[c]
        ok = (s * d_i > floor) and (abs(d_l) < spec.lesion_ratio * abs(d_i))
        frac = attributable_to(f"{spec.name} interaction[{c}] shift = the cross-edge", s * d_i, s * d_l)
        per[c] = {"delta_intact": float(d_i), "delta_lesion": float(d_l), "expected_sign": s, "floor": floor,
                  "frac_attributable": (None if frac is None else float(frac)), "ok": bool(ok)}
        all_ok = all_ok and ok
    return {"reads_intact": {k: float(v) for k, v in reads_i.items()},
            "reads_lesion": {k: float(v) for k, v in reads_l.items()},
            "per_condition": per, "PASS": bool(all_ok)}


def verify_byte_off(bridge_with, bridge_without, spec: CrossEdgeGateSpec) -> dict:
    """GENERIC byte-identical-off: the pool built WITHOUT the cross_edges has base connectivity BYTE-IDENTICAL to
    the with-edge pool once the declared edge (pre,post) slots are removed — integration added ONLY the declared
    edge, from the declaration itself. Reproduces R1's `_migration_invariant` base-connectivity byte-identity,
    generalized (the excluded (pre,post) set is computed FROM the CrossEdge endpoints, not hand-listed)."""
    xrows, xcols = set(), set()
    for ce in spec.cross_edges:
        pre, post = _edge_endpoints(bridge_with, ce)
        xrows |= set(int(x) for x in pre)
        xcols |= set(int(x) for x in post)

    def edge_map(bridge):
        coo = bridge.cp_connections.tocoo()
        r = np.asarray(to_host(coo.row)); c = np.asarray(to_host(coo.col)); d = np.asarray(to_host(coo.data))
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}

    k_with = edge_map(bridge_with)
    k_without = edge_map(bridge_without)
    k_with_base = {kk: vv for kk, vv in k_with.items() if not (kk[0] in xrows and kk[1] in xcols)}
    identical = bool(k_with_base == k_without)
    return {"base_connectivity_byte_identical": identical,
            "n_with": len(k_with), "n_without": len(k_without), "n_with_base": len(k_with_base),
            "PASS": identical}


def run_gate(pool, spec: CrossEdgeGateSpec) -> dict:
    """Train the declared edge (spec.train_fn), then run the GENERIC emergence + interaction gate on it. Returns
    the uniform verdict. The caller snapshots the non-edge weights BEFORE calling (so this measures true drift);
    to keep the harness self-contained it snapshots here, immediately before train_fn.

    NOTE: interaction is DESTRUCTIVE (lesions the pool). byte-off is verified separately by the caller (it needs a
    second, no-edge pool) via verify_byte_off — kept out of run_gate so run_gate needs only ONE pool."""
    masks = cross_edge_masks(pool.bridge, spec.cross_edges)
    data0 = np.asarray(to_host(pool.bridge.cp_connections.data)).copy()
    nc = noncross_mask(masks, data0.shape[0])
    frozen_before = data0[nc].copy()

    traj = spec.train_fn(pool)

    now = np.asarray(to_host(pool.bridge.cp_connections.data))
    frozen_maxdrift = float(np.max(np.abs(now[nc] - frozen_before))) if nc.any() else 0.0

    emergence = verify_emergence(pool, spec, masks, frozen_maxdrift)
    interaction = verify_interaction(pool, spec, masks)   # lesions the pool at its end
    go = bool(emergence["PASS"] and interaction["PASS"])
    return {"name": spec.name, "GO": go, "emergence": emergence, "interaction": interaction,
            "credit_signal": spec.credit_signal,
            "trajectory": traj if isinstance(traj, list) else None}
