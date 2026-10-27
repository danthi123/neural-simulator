"""Fingerprints of the two workspaces SIM_STABLE_RECAL re-calibrates (the GNW ignition bus and the thought-swap
workspace), for tests/test_stable_recal_switch.py. Each builds the organ the way production does, runs one short
decision and hashes the full decision record plus the final membrane potentials. The reference values in the test
were captured with these functions on the code before the switch existed (base e38904197 + the probes), so a
match with SIM_STABLE_RECAL unset proves the unset path is byte-identical, on the old integrator and on the stable
one alike.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _v_bytes(bridge):
    from sim.backend import to_host
    return np.asarray(to_host(bridge.cp_membrane_potential_v), dtype=np.float32).tobytes()


def _digest(record, bridges):
    h = hashlib.sha256()
    h.update(json.dumps(record, sort_keys=True, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
             .encode())
    for b in bridges:
        h.update(_v_bytes(b))
    return h.hexdigest()


def swap_fingerprint(seed=3):
    """The production thought-swap path: `build` with its defaults (as `ThoughtSwapWorkspace._ensure` calls it) and
    one isolated salient-mismatch `run_intention_swap` (as `ThoughtSwapWorkspace.observe` calls it)."""
    from research.runners import _gnw_neural_swap_intention_derisk as SM
    np.random.seed(int(seed))
    S = SM.build(seed=int(seed))
    std = SM.MultiLoopSTD(S["bridge"], S["xp"], S["ws_used"], S["patterns_host"])
    r = SM.run_intention_swap(S, std, incumbent=0, proposed=1, proposal_pa=SM.SALIENT_PA, isolate=True)
    return _digest(r, [S["bridge"]]), r


def bus_fingerprint(seed=3):
    """The production bus paths on one workspace built by `build_workspace_bridge` (production defaults): the
    three-organ unanimity hop, the two-organ coincidence hop, the P1.2 deliberation hop and the reentrant
    metacog-gated hop, each with its production drive."""
    from research.runners import _p1_2_workspace_deliberation_loop_derisk as P
    from research.runners import _gnw_norgan_bus_derisk as NB
    from research.runners import _gnw_two_distinct_organs_derisk as TD
    from research.runners import _gnw_coincidence_integrator_derisk as CI
    from research.runners import _gnw_reentrant_metacog_gated_deliberation_derisk as RE
    np.random.seed(int(seed))
    b, xp, slots, snap = P.build_workspace_bridge(int(seed))
    rec = {}
    c, rates, w, n = NB.norgan_hop(b, xp, slots, snap, ["fox", "fox", "fox"], "owl", NB.D_SUB_UNANIMITY[3],
                                   rng=np.random.default_rng(1))
    rec["norgan"] = [c, rates, w, n]
    c, rates, w, n, b_slot = TD.coincidence_hop(b, xp, slots, snap, "fox", "fox", "owl", CI.D_SUB_DEFAULT)
    rec["coincidence"] = [c, rates, w, n, b_slot]
    c, assign, rates, w, n = P._deliberate_hop(b, xp, slots, snap, "fox", ["owl", "hare"])
    rec["p12_hop"] = [c, sorted(assign.items()), rates, w, n]
    c, rates, w, n, conf, assign = RE._deliberate_hop_conf(b, xp, slots, snap, "fox", ["owl"])
    rec["reentrant_hop"] = [c, rates, w, n, conf, sorted(assign.items())]
    return _digest(rec, [b]), rec
