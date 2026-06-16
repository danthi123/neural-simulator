"""Stage-2 gen-moat: VERIFY the mechanism of the PERSISTENT gen-firing compression (weights vs a variable).

Attempt 1 (a clean-baseline membrane reset before the gen read) did NOT restore the gen firing (still 0.80 vs a
fresh-bridge 1.623), so the compression is PERSISTENT, not transient membrane state. Two candidate mechanisms:
  (A) the gen convergence WEIGHTS (gen_perception->gen_concept) eroded during the nav episode (the freeze thawed);
  (B) a persistent non-membrane VARIABLE (NMDA gate / conductance / eligibility) the reset doesn't clear.

This diagnostic decides between them: build the unified bridge, capture the gen_perception->gen_concept weight SUM
and the held-out gen firing RIGHT AFTER BUILD, run the live compose episode, then capture BOTH AGAIN. If the weight
sum drops -> (A) weights (fix: snapshot-after-build + restore-before-gen-check, the parser-silence pattern; or
re-freeze the gen edges by index for the episode). If the weight sum is ~unchanged but the firing dropped -> (B) a
variable. Reuse-by-import; no sim/ edit. (The Iron Law: verify before the next fix.)

Run: SIM_BACKEND=cupy python -m research.runners._stage2_gen_erosion_diag --seed 42
"""
import argparse
import json
import time

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.navigate_to_compose_then_answer import (
    build_compose_bridge, run_compose_episode, default_object_layout,
)
from research.runners._unified_stage1_merged import _read_gen_spikes, _category_of_concept_spikes


def _gen_weight_sum(bridge):
    """Sum |weight| of the gen_perception->gen_concept edges from the FINAL CSR (data-aligned via indptr/indices)."""
    rm = bridge.region_manager
    gp = np.asarray(list(rm.indices("gen_perception")), dtype=np.int64)
    gc = np.asarray(list(rm.indices("gen_concept")), dtype=np.int64)
    csr = bridge.cp_connections
    indptr = to_host(csr.indptr)
    post = to_host(csr.indices).astype(np.int64)
    data = np.abs(np.asarray(to_host(csr.data)).astype(np.float64))
    nnz = int(post.shape[0])
    pre = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre[int(indptr[r]):int(indptr[r + 1])] = r
    mask = np.isin(pre, gp) & np.isin(post, gc)
    return float(data[mask].sum()), int(mask.sum())


def _heldout_winfires(bridge, h, xp):
    gen = h["gen"]
    n_cat = int(gen["N_CAT"]); cat_ids = gen["gen_cat_ids"]
    vals = []
    for j in list(gen["gen_held_out"]):
        cpb, _f, _ct, _ft = _read_gen_spikes(bridge, gen, gen["vis_sets"][j], xp)
        _keyed, catmean = _category_of_concept_spikes(cpb, cat_ids, n_cat)
        vals.append(float(np.max(catmean)))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_stage2_gen_erosion_diag.json")
    args = ap.parse_args()
    xp, backend = get_backend()
    print(f"[erosion] backend={backend} seed={args.seed}", flush=True)

    t0 = time.time()
    bridge, composer, h, proj = build_compose_bridge(args.seed, with_body=True, co_resident_generalization=True)
    print(f"[erosion] built in {time.time()-t0:.0f}s", flush=True)

    w_build, n_edges = _gen_weight_sum(bridge)
    fire_build = _heldout_winfires(bridge, h, xp)
    print(f"[erosion] AFTER BUILD : gen W-sum {w_build:.2f} ({n_edges} edges) | held-out win-fire "
          f"{[round(v,3) for v in fire_build]} (mean {np.mean(fire_build):.3f})", flush=True)

    # run the live compose episode (the same setup as navigate_unified_episode.run_seed).
    layout = default_object_layout(args.seed)
    start_pos = (0, 2)
    sorted_cells = sorted(layout.keys(), key=lambda c: c[0])
    route_waypoints = [sorted_cells[1], sorted_cells[2]]
    ep = run_compose_episode(bridge, composer, h, proj, layout, start_pos, route_waypoints, perceive=True)
    print(f"[erosion] episode: grounded {len(h['grounded_objects'])} | moves {len(ep['moves'])}", flush=True)

    w_ep, _ = _gen_weight_sum(bridge)
    fire_ep = _heldout_winfires(bridge, h, xp)
    print(f"[erosion] AFTER EPISODE: gen W-sum {w_ep:.2f} | held-out win-fire "
          f"{[round(v,3) for v in fire_ep]} (mean {np.mean(fire_ep):.3f})", flush=True)

    w_ratio = w_ep / (w_build + 1e-9)
    f_ratio = float(np.mean(fire_ep)) / (float(np.mean(fire_build)) + 1e-9)
    weights_eroded = bool(w_ratio < 0.9)
    firing_dropped = bool(f_ratio < 0.9)
    diag = {
        "gen_wsum_build": round(w_build, 3), "gen_wsum_episode": round(w_ep, 3), "w_ratio": round(w_ratio, 3),
        "firing_build_mean": round(float(np.mean(fire_build)), 3), "firing_episode_mean": round(float(np.mean(fire_ep)), 3),
        "f_ratio": round(f_ratio, 3), "weights_eroded": weights_eroded, "firing_dropped": firing_dropped,
        "mechanism": ("(A) WEIGHTS eroded during the episode -> snapshot-after-build + restore-before-gen-check "
                      "(parser-silence pattern), or re-freeze gen edges by index" if weights_eroded else
                      "(B) a PERSISTENT VARIABLE (weights ~unchanged but firing dropped) -> fuller reset / isolate"
                      if firing_dropped else
                      "NEITHER reproduced (firing did not drop in this run) -> re-examine the Stage-2 read context"),
    }
    print(f"\n[erosion] VERDICT {json.dumps(diag, indent=2)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"[erosion] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
