"""The shared spiking commit stage (research/runners/spiking_commit.py) and its BG live-seam application.

Plan: docs/drafts/spiking-commit/PLAN.md section 3; worklog docs/drafts/bg-commit/WORKLOG.md. Pins (numpy, CPU):
  * A1 moved onto the shared module is byte-identical: its commit battery on dev seed 1 at row 0 hashes to the value
    captured on S main cc9b4bf3e before the move;
  * the module: `commit_plan` and `region_parts` emit the same edges; `sequence_read` reads A1's semantics.
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

# captured on cc9b4bf3e (before any edit) with the same `_h` over `commit_battery(1, 0)`
A1_COMMIT_BATTERY_SHA256 = "6e0bd58ef1d460e64f6951c6f8819ab8db781b2b188138caf029fb2470f4a5bc"


def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                     else str(o)).encode()).hexdigest()


# ── A1 on the shared module ────────────────────────────────────────────────────────────────────────────────────────
def test_a1_commit_battery_byte_identical():
    import research.runners.word_role_commit_organ as C
    assert _h(C.commit_battery(1, 0)) == A1_COMMIT_BATTERY_SHA256


# ── the module ─────────────────────────────────────────────────────────────────────────────────────────────────────
P = dict(NG=3, NGI=2, NS=4, W_XG=50.0, W_GG=20.0, W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0)


def _edges_from_plan(plan):
    out = set()
    for e in plan.values():
        for a, b, w in zip(e["pre_indices"], e["post_indices"], e["initial_weights"]):
            out.add((int(a), int(b), round(float(w), 4), e.get("transmission_gate") or None))
    return out


def test_region_parts_match_commit_plan():
    from research.runners import spiking_commit as SC
    from research.runners._vocal_action_selector_gate import _region
    from sim.enums import NeuronType
    K, src_n = 3, 5
    src_regions = ["src_%d" % k for k in range(K)]
    regions, pathways = SC.region_parts(src_regions, P, prefix="c_", gate_in="i", gate_excl="x", latch_loop="l",
                                        make_region=_region, exc_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL,
                                        inh_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON)
    # place every region at the index block `layout` gives it (sources first), then compare edge sets
    lay = SC.layout(K, K * src_n, P["NG"], P["NGI"], P["NS"])
    sizes = {r: src_n for r in src_regions}
    sizes.update({r.name: r.n_neurons for r in regions})
    start = {nm: k * src_n for k, nm in enumerate(src_regions)}
    for k in range(K):
        for pop, nm in SC.region_names("c_", k).items():
            start[nm] = int(lay[pop][k][0])
            assert sizes[nm] == len(lay[pop][k])
    reg_edges = set()
    for pw in pathways:
        assert pw.density == 1.0 and pw.weight_jitter == 0.0 and not pw.plastic
        for a in range(sizes[pw.from_region]):
            for b in range(sizes[pw.to_region]):
                reg_edges.add((start[pw.from_region] + a, start[pw.to_region] + b, round(pw.weight_mean, 4),
                               pw.transmission_gate or None))
    plan = SC.commit_plan([np.arange(k * src_n, (k + 1) * src_n) for k in range(K)], lay, P, prefix="",
                          gate_in="i", gate_excl="x", latch_loop="l")
    assert reg_edges == _edges_from_plan(plan)
    inh = [pw for pw in pathways if pw.from_region.startswith("c_gate_inh_")]
    assert inh and all(pw.receptor == "gaba_a" and pw.transmission_gate == "x" for pw in inh)
    assert [r.exc_fraction for r in regions[:3]] == [1.0, 0.0, 1.0]
    assert all(r.internal_density == 0.0 for r in regions)


def test_sequence_read_semantics():
    from research.runners import spiking_commit as SC
    # thr = 0.5 * 4 * 10 = 20. Item 0 latches 0; item 1 latches 1 (0 still on); item 2 latches nothing new.
    rd = SC.sequence_read([[30, 0, 0], [30, 25, 0], [30, 25, 0]], 4, 10, 0.5)
    assert rd["threshold"] == 20.0 and [p["role"] for p in rd["per_item"]] == [0, 1, None]
    assert rd["filled"] == {0: 0, 1: 1} and rd["co_commit"] == 0 and rd["latched_at_end"] == [0, 1]
    co = SC.sequence_read([[30, 30, 0]], 4, 10, 0.5)
    assert co["per_item"][0]["role"] is None and co["co_commit"] == 1
    re = SC.sequence_read([[30, 0, 0], [0, 0, 0], [30, 0, 0]], 4, 10, 0.5)
    assert re["relatch"] == 1 and 0 not in re["filled"]
    one = SC.read_with_variants([77, 0], 20, 30, 0.03)                     # A1 dev margin: 77 vs 0, thr 18
    assert one["winner"] == 0 and one["read_invariant"]
