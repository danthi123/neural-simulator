"""THROWAWAY cheap-first probe (GPU; reuses the cached n=98 hippo+dlpfc substrate
byte-unchanged): does the STRONG-dynamics dlpfc_wm region SUSTAIN role-distinct
activity where a WEAK concept pool collapses? This is the load-bearing substrate
property behind the genuinely-untried dlpfc-role conjunctive-engram idea (finding
2026-05-31-composition-arc-...-dlpfc-role-conjunction).

Root cause (DIRECTION-E): the v16 WEAK concept-pool dynamics make engram capture
fire ~equally across roles -> no role-distinctness -> sets-not-sequences. The fix
hypothesis: a STRONG-dynamics region (dlpfc_wm, NMDA-bistable) carries role-distinct
STABLE activity that the weak pools cannot. Test it directly + cheaply, BEFORE the
full conjunctive-engram build.

Test (per region, per seed):
  - pick 2 disjoint 'role' sub-populations of the region's neurons.
  - drive role-0 sub-pop (external current) for DRIVE_STEPS; capture region firing
    DURING; then REMOVE drive and capture region firing POST (persistence window).
  - repeat for role-1 sub-pop.
  - PERSISTENCE = mean(post firing) / mean(during firing)  (NMDA bistable -> high;
    weak pool -> low / collapses).
  - SUSTAINED ROLE-DISTINCTNESS = 1 - cosine(post_role0, post_role1)  (do the two
    roles stay DISTINCT after the drive stops? high -> role info preserved).
Regions: dlpfc_wm (strong, the hypothesis) vs a weak concept pool noun_pool_APPLE
(the reproduce-the-failure control).

FROZEN three-state (set before run): RESOLVES if dlpfc PERSISTENCE >= 0.50 AND
sustained-distinctness >= 0.50, AND materially beats the weak pool on BOTH (pool
persistence < 0.50 OR pool distinctness < 0.50) -> strong region carries role-
distinct stable activity the weak pool can't -> proceed to conjunctive-engram build.
PARTIAL if dlpfc beats the pool on one axis only. BOUNDARY if dlpfc does NOT sustain
role-distinct activity (persistence or distinctness < 0.50) -> the strong region
also collapses role info -> the convergent boundary deepens. Instrument check first.
"""
from __future__ import annotations
import os
import sys
import numpy as np

from research.findings.raw.mode_unification_with_hippo_dlpfc_probe import (
    _build_bridge_with_hippo_and_dlpfc, _bridge_save_path,
)
from sim.backend import get_backend, to_host

SEEDS = [42]
DRIVE_PA = 2000.0   # suprathreshold (400pA gave only ~2% firing; substrate concept drive ~1500pA)
RESET_STEPS = 40
DRIVE_STEPS = 60
POST_STEPS = 60
PERSIST_BAR = 0.50
DISTINCT_BAR = 0.50


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def load_substrate(seed):
    bridge = _build_bridge_with_hippo_and_dlpfc(seed, enable_adjective=True, verbose=False)
    bridge.load_checkpoint(_bridge_save_path(seed, smoke=False))
    return bridge


def drive_capture(bridge, drive_idx, cap_idx, xp):
    """Drive drive_idx for DRIVE_STEPS, capture cap_idx firing DURING + POST (drive removed)."""
    drive_arr = xp.asarray(list(drive_idx), dtype=xp.int64)
    cap_arr = xp.asarray(list(cap_idx), dtype=xp.int64)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[drive_arr] = DRIVE_PA
    during = xp.zeros(cap_arr.shape[0], dtype=xp.float64)
    for _ in range(DRIVE_STEPS):
        bridge._run_one_simulation_step()
        during += bridge.cp_firing_states[cap_arr].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    post = xp.zeros(cap_arr.shape[0], dtype=xp.float64)
    for _ in range(POST_STEPS):
        bridge._run_one_simulation_step()
        post += bridge.cp_firing_states[cap_arr].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(during) / DRIVE_STEPS, to_host(post) / POST_STEPS


def test_region(bridge, region_name, xp):
    idx = list(bridge.region_manager.indices(region_name))
    n = len(idx)
    half = n // 2
    role0_drive = idx[:half // 1][:max(1, half // 2)]  # drive ~ first quarter as role0 seed
    role1_drive = idx[half:half + max(1, half // 2)]   # drive ~ third quarter as role1 seed
    cap = idx  # capture the whole region
    d0, p0 = drive_capture(bridge, role0_drive, cap, xp)
    d1, p1 = drive_capture(bridge, role1_drive, cap, xp)
    during_mean = (d0.mean() + d1.mean()) / 2.0 + 1e-12
    post_mean = (p0.mean() + p1.mean()) / 2.0
    persistence = float(post_mean / during_mean)
    sustained_distinct = 1.0 - _cos(p0, p1)
    during_distinct = 1.0 - _cos(d0, d1)
    return {
        "region": region_name, "n": n,
        "during_mean": float(during_mean), "post_mean": float(post_mean),
        "persistence": persistence,
        "during_distinct": during_distinct,
        "sustained_distinct": sustained_distinct,
    }


def main():
    xp, backend = get_backend()
    print(f"=== dlpfc-role substrate property probe (backend={backend}) ===", flush=True)
    print(f"drive={DRIVE_PA}pA reset={RESET_STEPS} drive_steps={DRIVE_STEPS} post={POST_STEPS}", flush=True)
    rows = []
    for seed in SEEDS:
        p = _bridge_save_path(seed, smoke=False)
        if not os.path.exists(p):
            print(f"[seed {seed}: no cache {p}]", flush=True); continue
        bridge = load_substrate(seed)
        regions = [r.name for r in bridge.region_manager.regions()]
        strong = "dlpfc_wm"
        weak = next((r for r in regions
                     if r.startswith("noun_pool") and not r.endswith("_fs")), None)
        print(f"  [seed {seed}] {len(regions)} regions; strong={strong} weak={weak}", flush=True)
        for rn in [strong, weak]:
            if rn is None or rn not in regions:
                print(f"    [region {rn} absent; skip]", flush=True); continue
            res = test_region(bridge, rn, xp)
            res["seed"] = seed
            rows.append(res)
            print(f"    {rn:18} n={res['n']:3d} | during_mean={res['during_mean']:.3f} "
                  f"post_mean={res['post_mean']:.3f} | PERSIST={res['persistence']:.3f} | "
                  f"during_distinct={res['during_distinct']:.3f} SUSTAINED_distinct={res['sustained_distinct']:.3f}",
                  flush=True)
        del bridge

    # verdict (seed-42 smoke)
    d = next((r for r in rows if r["region"] == "dlpfc_wm"), None)
    w = next((r for r in rows if r["region"].startswith("noun_pool")), None)
    print("\n=== VERDICT ===", flush=True)
    if d is None:
        print("CANNOT-CONCLUDE (no dlpfc result)", flush=True); return
    print(f"dlpfc: persistence={d['persistence']:.3f} sustained_distinct={d['sustained_distinct']:.3f}", flush=True)
    if w is not None:
        print(f"weak pool {w['region']}: persistence={w['persistence']:.3f} "
              f"sustained_distinct={w['sustained_distinct']:.3f}", flush=True)
    dlpfc_ok = d["persistence"] >= PERSIST_BAR and d["sustained_distinct"] >= DISTINCT_BAR
    beats = w is not None and (w["persistence"] < PERSIST_BAR or w["sustained_distinct"] < DISTINCT_BAR)
    if dlpfc_ok and beats:
        print(f"VERDICT: RESOLVES (smoke) -- dlpfc SUSTAINS role-distinct activity "
              f"(persist {d['persistence']:.2f}>={PERSIST_BAR}, distinct {d['sustained_distinct']:.2f}>={DISTINCT_BAR}) "
              f"where the weak pool does not -> proceed to conjunctive-engram build (multi-seed first).", flush=True)
    elif dlpfc_ok:
        print("VERDICT: PARTIAL -- dlpfc sustains role-distinct activity but the weak pool isn't clearly worse "
              "(control weak). Re-examine the control before building.", flush=True)
    else:
        print(f"VERDICT: BOUNDARY (smoke) -- dlpfc does NOT sustain role-distinct activity "
              f"(persist {d['persistence']:.2f} / distinct {d['sustained_distinct']:.2f} below {PERSIST_BAR}). "
              f"The strong region also collapses role info -> the convergent compositional boundary deepens; "
              f"pivot to the next genuinely-new mechanism (stable-attractor role basis / dendritic binding).", flush=True)


if __name__ == "__main__":
    main()
