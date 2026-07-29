#!/usr/bin/env python3
"""Does weight-history METAPLASTICITY allocate distinct slots ON THE SUBSTRATE, with NO host teaching?

THE QUESTION. The consolidation store cannot ALLOCATE: distinct facts collapse onto one slot
("exactly one slot takes ~3.1-3.3 while the others sit at ~1.0, and WHICH slot varies by seed",
nmda_compositional_consolidation.py:281). Four FAIRNESS mechanisms are refuted at 6 seeds -- they
equalise WHO WINS, which cannot stop two facts claiming the SAME slot. Metaplasticity asks a different
question, IS THIS CELL ALREADY CLAIMED, which is exactly free-vs-taken. Toy: control 1/6 (reproducing
the substrate collapse: maps like [2,2,2]), beta=0.4/0.8 -> 6/6 valid+stable, monotonic dose-response.

WHY THE HOST TEACHING SIGNAL IS OFF IN BOTH ARMS. The shipped replay DRIVES the target slot
(`cp_external_input_current[slot_idx[i]] += slot_drive_pA`) -- the host telling fact i to go to slot i.
That is the supervision that makes the current result "a host-supervised associative write". An
ALLOCATION test must remove it, so `teach_slot=False` in BOTH arms; the ONLY difference between arms is
beta. Comparing a taught arm against an untaught one would measure supervision, not metaplasticity.

ANTI-CHEATS (each earned by a real failure in this arc):
  * INSTRUMENT CHECK -- read the claimed-ness vector and show it is non-degenerate BEFORE trusting any
    arm. A metric that cannot vary cannot refute anything.
  * LEVER CHECK -- assert the penalty actually changed the injected current. A no-op flag produced a
    whole 6-run A/B over two identical configurations on 2026-07-28.
  * INERTNESS ASSERTION -- beta=0 must be byte-identical to the shipped path. "This is inert" is a
    HYPOTHESIS; it belongs in an assertion, not a comment.
  * The control must still COLLAPSE. If beta=0 already allocates, the regime is uninformative.

STAGED SHORTCUT, TRACKED HONESTLY: the claimed-ness column sum is read on the HOST. This run asks only
whether the mechanism survives the substrate's saturation. If it does, the burn-down is a sim/ step-loop
term next to the existing per-postsynaptic-neuron synaptic scaling (bridge.py:8671-8690), which already
computes exactly this shape of quantity in-loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from types import SimpleNamespace

from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay, CONSOLIDATED_FACTS)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host


def _claimed_vector(b, n_slots, gate="concept_to_comp_attr"):
    """Mean STORE-PATHWAY weight onto each slot -- NOT the total afferent sum.

    WHY THIS IS RESTRICTED TO ONE GATE, and it is load-bearing. The first version summed ALL afferents
    per slot cell (recurrent self-loop + inhibition + ca1 + concept). That total is ~62/cell, while the
    selective store change lives in `concept_to_comp_attr` alone and is ~0.04 -- so the signal was
    swamped ~1500:1 and BOTH arms read a near-identical net depression (delta spread 0.039 with teaching
    ON and 0.039 with it OFF). That was a METRIC-TOO-COARSE null, not a refutation, and reading it as one
    would have retired a mechanism on an instrument artifact.

    For CSR, the column (post-neuron) of data[k] is indices[k], so intersecting the gate's synapse
    indices with each slot's neuron indices gives exactly the store weight landing on that slot.
    """
    cp, _ = get_backend()
    rm = b.region_manager
    gidx = getattr(b, "_plasticity_gate_indices_gpu", {}).get(gate, None)
    out = []
    if gidx is None or int(getattr(gidx, "size", 0)) == 0:
        return [float("nan")] * n_slots
    nnz = int(b.cp_connections.nnz)
    gidx = gidx[gidx < nnz]
    cols = np.asarray(to_host(b.cp_connections.indices[gidx])).ravel()
    data = np.asarray(to_host(b.cp_connections.data[gidx])).ravel()
    for s in range(n_slots):
        try:
            sidx = set(int(v) for v in rm.indices(f"comp_attr_{s}"))
            m = np.fromiter((int(c) in sidx for c in cols), dtype=bool, count=len(cols))
            out.append(float(data[m].mean()) if m.any() else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out


def _winner_map(b, facts, n_slots, pool_drive_pA=1400.0, settle=30):
    """Cue each fact's pools ONLY (no slot drive) and read which slot wins on spikes."""
    cp, _ = get_backend()
    rm = b.region_manager
    all_names = {r.name for r in b.core_config.brain_regions}
    mapping = []
    for (noun, adj) in facts:
        b.cp_external_input_current[:] = 0.0
        for nm in (f"noun_pool_{noun.upper()}", f"adjective_pool_{adj.upper()}"):
            if nm in all_names:
                b.cp_external_input_current[cp.asarray(list(rm.indices(nm)), dtype=cp.int64)] += float(pool_drive_pA)
        counts = np.zeros(n_slots)
        for _ in range(settle):
            b._run_one_simulation_step()
            fs = np.asarray(to_host(b.cp_firing_states)).ravel()
            for s in range(n_slots):
                try:
                    counts[s] += float(fs[np.asarray(list(rm.indices(f"comp_attr_{s}")), dtype=np.int64)].sum())
                except Exception:
                    pass
        mapping.append(int(np.argmax(counts)) if counts.sum() > 0 else -1)
    b.cp_external_input_current[:] = 0.0
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--n-facts", type=int, default=3)
    # DIAGNOSTIC ARM: with the host slot-teaching ON, does claimed-ness differentiate at all? If the
    # signal appears only under supervision, then allocation and supervision are the SAME problem --
    # nothing becomes "claimed" unless the host already told it which slot to be.
    ap.add_argument("--teach", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    facts = CONSOLIDATED_FACTS[: int(args.n_facts)]
    n_slots = len(facts)
    a = dict(BASE)
    a["comp_attractor_slots"] = n_slots
    # BASE sets comp_no_pool_slot=True, which DROPS the pool->slot pathway ENTIRELY -- so the
    # `concept_to_comp_attr` gate does not exist and every store read returns nan. _consol_cortical_store_probe.py:60
    # overrides it for exactly this reason; match it, or this probe measures a pathway that isn't wired.
    # NOTE (silent-failure hazard, recorded): _try_pgate swallows the KeyError and returns False, and
    # _mean_gate_weight returns 0.0, for a MISSING gate -- so freezing a nonexistent gate looks like a
    # perfect freeze (drift exactly +0.000000) rather than a no-op. Nothing checks those return values.
    a["comp_no_pool_slot"] = False
    a["comp_pool_slot_weight"] = 1.5
    b = build_substrate(args.seed, SimpleNamespace(**a))
    tags, _ = encode_facts_with_reinstatement(b, facts)

    print("=" * 76)
    print("METAPLASTIC ALLOCATION on the substrate  seed=%d beta=%.3f cycles=%d facts=%d"
          % (args.seed, args.beta, args.cycles, n_slots))
    print("=" * 76)

    # --- INSTRUMENT CHECK: is claimed-ness even readable and non-degenerate? ---
    c0 = _claimed_vector(b, n_slots)
    print("  claimed BEFORE replay: %s" % ["%.5f" % v for v in c0])
    if not np.isfinite(c0).all():
        print("  => UNDEFINED: claimed-ness unreadable. NOT a result; the instrument failed.")
        return 1

    # --- LEVER CHECK: does the penalty actually move the injected current? ---
    cp, _ = get_backend()
    rm = b.region_manager
    s0 = cp.asarray(list(rm.indices("comp_attr_0")), dtype=cp.int64)
    b.cp_external_input_current[:] = 0.0
    b.cp_external_input_current[s0] -= float(args.beta) * float(c0[0])
    moved = float(abs(float(to_host(b.cp_external_input_current[s0]).mean())))
    b.cp_external_input_current[:] = 0.0
    print("  LEVER penalty current on slot0: %.4f pA  [%s]"
          % (moved, "MOVED" if moved > 0 else ("inert (beta=0 control)" if args.beta == 0 else "NOT MOVED")))
    if args.beta > 0 and moved <= 0:
        print("  => VOID ARM: beta>0 but the penalty is zero. Do not interpret this arm.")
        return 1

    # --- the run: NO host slot teaching in either arm; beta is the only difference ---
    coactivation_replay(b, facts, tags, int(args.cycles), args.seed,
                        teach_slot=bool(args.teach), metaplastic_beta=float(args.beta))

    c1 = _claimed_vector(b, n_slots)
    mapping = _winner_map(b, facts, n_slots)
    valid = sorted(mapping) == list(range(n_slots))
    stable = mapping == _winner_map(b, facts, n_slots)
    d = [v - u for v, u in zip(c1, c0)]
    print("  claimed AFTER replay : %s" % ["%.5f" % v for v in c1])
    print("  DELTA (learned)      : %s   spread=%.6f  teach_slot=%s"
          % (["%+.5f" % v for v in d], float(max(d) - min(d)), bool(args.teach)))
    print("  fact->slot map       : %s" % mapping)
    print("  permutation_valid=%s  stable=%s" % (valid, stable))
    spread = float(np.nanmax(c1) - np.nanmin(c1))
    print("  claimed spread=%.6f %s" % (spread, "(degenerate - all slots equally claimed)" if spread < 1e-9 else ""))
    print("  VERDICT: %s" % ("ALLOCATES (valid+stable)" if valid and stable else
                             "does NOT allocate -- %s" % ("collapsed onto %d" % mapping[0]
                                                          if len(set(mapping)) == 1 else "map %s" % mapping)))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(dict(seed=args.seed, beta=args.beta, cycles=args.cycles, n_facts=n_slots,
                       teach_slot=False, mapping=mapping, valid=bool(valid), stable=bool(stable),
                       claimed_before=c0, claimed_after=c1, claimed_spread=spread,
                       argv=sys.argv), open(args.out, "w"), indent=1)
        print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
