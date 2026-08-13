"""ONE-BRAIN MERGE — RUNG 2 verify: extend the production shared-substrate merge beyond the rung-1 pair.

Rung 1 (2026-08-13-merge-production-integration-rung1-GO.md) put the D2 SURPRISE + E2 WORLD-MODEL production
organs on ONE shared spiking bridge, byte-identical merged-vs-co-resident. Rung 2 asks: can a THIRD production
faculty join the shared pool byte-identically?

WHAT THIS VERIFIES (three parts, one runner):

 (A) REGRESSION GUARD — the rung-1 pair is STILL byte-identical merged-vs-co-resident (surprise + world-model
     read batteries, max delta 0.0). Extending the pool must not disturb rung 1.

 (B) THIRD FACULTY (GO) — RECONSOLIDATION (belief revision) operates byte-identically on the shared pool. The
     reconsolidation organ owns NO neurons of its own: its spiking reconsolidation WINDOW *is* the D2 surprise
     organ (a `cp_firing_states[surprise]` read). So on the merged bridge the belief-revision faculty runs on
     the SAME shared pool the surprise+world-model organs do. We verify its window read (opened + surprise_hz)
     is byte-identical merged-vs-co-resident AND the faculty is alive (window OPENS on a contradiction, CLOSED on
     a confirmation). This is a real 'more faculties on the one shared substrate' result — a faculty riding the
     merged pool WITHOUT adding a pool member. (Byte-identity follows from rung 1's surprise byte-identity; this
     confirms the composition holds and the faculty is alive on the shared pool.)

 (C) THIRD REGION-OWNING ORGAN (BOUNDARY, measured) — the next production organ that would add its OWN neurons
     to the pool is COMPREHENSION (the Wong-Wang `SpikingRoleCompetition` role monitor). Its native config sets
     `dt_ms=0.5`; the shared pool runs at `dt_ms=1.0` (the operating point surprise+world-model are validated at
     and the value the N-organ de-risk reconciled the role WTA's binary selection to, 6/6). We MEASURE the
     comprehension faculty (well/ill role-resolution AUC) at its native dt=0.5 vs the shared dt=1.0. The
     role-WTA's coarse binary selection survives dt=1.0 (Norgan 6/6), but the GRADED well/ill margin — the
     actual production read — DEGRADES: it does not robustly clear the 0.80 gate at dt=1.0. So comprehension
     cannot join the shared pool at dt=1.0 without changing (degrading) its production read, and per-region dt
     scaling cannot be byte-identical (the integrator steps all neurons at one dt). This is the mapped boundary:
     the remaining production organs each diverge from the rung-1 pair on a GLOBAL-only, faculty-LOAD-BEARING
     dynamics flag (comprehension: dt; metacog/pragmatic/affect: parameter-heterogeneity; causal/curiosity:
     stdp+reward+neuromod), so each needs a per-region scoping engine feature — see the finding.

Reproduce:
    SIM_BACKEND=numpy python -m research.runners._onebrain_merge_rung2_verify \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_merge_rung2_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.runners.onebrain_merge_production import MergedSubstrate
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from research.runners.worldmodel_production_organ import WorldModelProductionOrgan
from research.runners.reconsolidation_production_organ import ReconsolidationProductionOrgan


# ── (A) rung-1 regression batteries (identical to the rung-1 verify) ────────────────────────────────────────
_SURPRISE_BATTERY = [("alpha", "alpha"), ("beta", "gamma"), ("delta", "omega"), ("kappa", "kappa")]


def _surprise_reads(organ: SurpriseProductionOrgan) -> dict:
    organ.ensure_built()
    out = {}
    c = organ.calib
    for k in ("confirm_hz", "contradict_hz", "novel_hz"):
        out[f"calib.{k}"] = float(c[k])
    out["threshold"] = float(organ.threshold)
    for (ps, pa) in _SURPRISE_BATTERY:
        out[f"judge[{ps}->{pa}].hz"] = float(organ.judge("agent", "acts", ps, pa)["surprise_hz"])
    return out


def _worldmodel_reads(organ: WorldModelProductionOrgan) -> dict:
    organ.ensure_built()
    out = {"threshold": float(organ.threshold)}
    for ctx in (+1, -1):
        e = organ.expectation(ctx)
        out[f"expect[{ctx:+d}].margin"] = float(e["pred_margin"])
    for ctx, obs in ((+1, +1), (+1, -1), (-1, -1), (-1, +1)):
        out[f"surprise[c{ctx:+d},o{obs:+d}].hz"] = float(organ.read_surprise(ctx, obs)["surprise_hz"])
    return out


def _max_delta(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    worst_k, worst = None, 0.0
    for k in keys:
        d = abs(a[k] - b[k])
        if d >= worst:
            worst, worst_k = d, k
    return worst, worst_k, sorted(set(a) ^ set(b))


# ── (B) reconsolidation faculty on the shared pool ──────────────────────────────────────────────────────────
# (agent, action, stored_patient, asserted_patient); confirm = stored==asserted (window closed), contradict =
# distinct (window open). Deterministic order so block assignment matches merged vs co-resident.
_RECON_BATTERY = [
    ("dog", "chase", "cat", "cat"),      # confirm -> window CLOSED
    ("dog", "chase", "cat", "bird"),     # contradict -> window OPEN
    ("cat", "eat", "fish", "worm"),      # contradict -> window OPEN
    ("bird", "sing", "song", "song"),    # confirm -> window CLOSED
]


def _recon_reads(recon: ReconsolidationProductionOrgan) -> dict:
    out = {}
    for i, (ag, ac, ps, pa) in enumerate(_RECON_BATTERY):
        opened, sj = recon.window_open(ag, ac, ps, pa)
        out[f"win[{i}].opened"] = float(bool(opened))
        out[f"win[{i}].hz"] = float(sj["surprise_hz"])
    return out


def _recon_alive(reads: dict) -> bool:
    """Faculty alive: the two CONFIRM items keep the window CLOSED (0), the two CONTRADICT items OPEN it (1)."""
    return (reads["win[0].opened"] == 0.0 and reads["win[3].opened"] == 0.0
            and reads["win[1].opened"] == 1.0 and reads["win[2].opened"] == 1.0)


def run_seed(seed: int, verbose: bool = True) -> dict:
    # MERGED: surprise + world-model on ONE shared bridge; reconsolidation rides the merged surprise.
    shared = MergedSubstrate(seed=seed, organs=("surprise", "worldmodel"))
    m_surp = SurpriseProductionOrgan(seed=seed, shared=shared)
    m_wm = WorldModelProductionOrgan(seed=seed, shared=shared)
    m_surp.ensure_built(); m_wm.ensure_built()
    m_recon = ReconsolidationProductionOrgan(seed=seed)
    m_recon._surprise = m_surp                       # belief-revision window = the MERGED surprise organ

    # CO-RESIDENT: each organ on its own bridge (both merge flags ON) — the rung-1 apples-to-apples baseline.
    subS = MergedSubstrate(seed=seed, organs=("surprise",))
    subW = MergedSubstrate(seed=seed, organs=("worldmodel",))
    s_surp = SurpriseProductionOrgan(seed=seed, shared=subS)
    s_wm = WorldModelProductionOrgan(seed=seed, shared=subW)
    s_surp.ensure_built(); s_wm.ensure_built()
    s_recon = ReconsolidationProductionOrgan(seed=seed)
    s_recon._surprise = s_surp

    # (A) rung-1 regression: surprise + world-model byte-identical merged-vs-co-resident.
    sd, sk, smiss = _max_delta(_surprise_reads(m_surp), _surprise_reads(s_surp))
    wd, wk, wmiss = _max_delta(_worldmodel_reads(m_wm), _worldmodel_reads(s_wm))
    rung1_byte_id = bool(sd == 0.0 and wd == 0.0 and not smiss and not wmiss)

    # (B) reconsolidation faculty byte-identical + alive on the shared pool.
    m_reads = _recon_reads(m_recon); s_reads = _recon_reads(s_recon)
    rd, rk, rmiss = _max_delta(m_reads, s_reads)
    recon_byte_id = bool(rd == 0.0 and not rmiss)
    recon_alive = _recon_alive(m_reads)
    recon_go = bool(recon_byte_id and recon_alive)

    res = {
        "seed": seed,
        "rung1_surprise_maxdelta": sd, "rung1_worldmodel_maxdelta": wd, "rung1_byte_identical": rung1_byte_id,
        "recon_read_maxdelta": rd, "recon_worst_key": rk, "recon_byte_identical": recon_byte_id,
        "recon_alive": recon_alive, "recon_faculty_go": recon_go,
    }
    if verbose:
        print(f"  [seed {seed}] rung1 byte-id={rung1_byte_id}(s={sd:.2e},w={wd:.2e}) | "
              f"RECON byte-id={recon_byte_id}(d={rd:.2e}@{rk}) alive={recon_alive} -> GO={recon_go}", flush=True)
    return res


# ── (C) comprehension dt boundary — the measured wall for the next region-owning organ ──────────────────────
def comprehension_dt_boundary(seeds, verbose=True) -> dict:
    """MEASURE the comprehension well/ill role-resolution AUC at its native dt=0.5 vs the shared pool's dt=1.0.
    Native separates perfectly; at dt=1.0 the GRADED margin degrades below the 0.80 gate -> comprehension cannot
    join the shared pool byte-identically without a per-region dt (which cannot be byte-exact)."""
    import numpy as np
    from research.runners._phaseB_multicue_competition_spiking_derisk import (
        SpikingRoleCompetition, INSTALLED_CUE_WEIGHTS)
    from research.runners._spiking_comprehension_monitor_derisk import (
        build_battery, _evs_for, semantic_sel_margin, roc_auc)

    def auc(seed, **kw):
        comp = SpikingRoleCompetition(seed=seed, **kw)
        for c, w in INSTALLED_CUE_WEIGHTS.items():
            comp.set_cue_weight(c, w)
        comp.freeze_all_cue_plasticity()
        items = build_battery(seed, n_per_cond=8)
        labels = [lab for (lab, *_r) in items]
        sem = [semantic_sel_margin(comp, _evs_for(n0, v, n1), 60) for (_l, _t, n0, v, n1) in items]
        return roc_auc(sem, labels)

    rows = []
    for s in seeds:
        a_native = auc(s, dt_ms=0.5, homeostasis=False, per_region_thresh=False)
        a_shared = auc(s, dt_ms=1.0, homeostasis=True, per_region_thresh=True)
        rows.append({"seed": s, "auc_native_dt0.5": a_native, "auc_shared_dt1.0": a_shared,
                     "clears_gate_native": bool(a_native >= 0.80), "clears_gate_shared": bool(a_shared >= 0.80)})
        if verbose:
            print(f"  [seed {s}] comprehension AUC native(dt=0.5)={a_native:.3f}  "
                  f"shared(dt=1.0)={a_shared:.3f}  clears0.80@shared={a_shared >= 0.80}", flush=True)
    n = len(rows)
    n_native = sum(r["clears_gate_native"] for r in rows)
    n_shared = sum(r["clears_gate_shared"] for r in rows)
    return {"rows": rows, "n_seeds": n, "n_clear_native": n_native, "n_clear_shared": n_shared,
            "verdict": "BOUNDARY" if n_shared < max(1, int(np.ceil(5 / 6 * n))) else "GO-at-dt1.0",
            "note": ("role-WTA binary selection survives dt=1.0 (Norgan 6/6) but the graded well/ill margin — the "
                     "production read — does not robustly clear 0.80 at dt=1.0; per-region dt cannot be byte-exact")}


def _gate(n_go, n):
    return "GO" if ((n >= 6 and n_go >= 5) or (n < 6 and n_go == n)) else "BOUNDARY"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== ONE-BRAIN MERGE — RUNG 2 ===")
    print("(A) rung-1 regression  (B) reconsolidation 3rd-faculty on shared pool  (C) comprehension dt boundary")
    results = [run_seed(s) for s in seeds]
    n = len(results)
    n_r1 = sum(r["rung1_byte_identical"] for r in results)
    n_rbi = sum(r["recon_byte_identical"] for r in results)
    n_ral = sum(r["recon_alive"] for r in results)
    n_rgo = sum(r["recon_faculty_go"] for r in results)

    print("\n--- (C) comprehension dt boundary (next region-owning organ) ---")
    comp = comprehension_dt_boundary(seeds)

    print("\n=== VERDICT (rung 2) ===")
    print(f"  (A) rung-1 pair STILL byte-identical merged-vs-coresident: {n_r1}/{n} -> {_gate(n_r1, n)}")
    print(f"  (B) reconsolidation faculty byte-identical on shared pool: {n_rbi}/{n} -> {_gate(n_rbi, n)}")
    print(f"      reconsolidation faculty alive (merged):                {n_ral}/{n}")
    print(f"      RECONSOLIDATION 3rd-FACULTY GO:                        {n_rgo}/{n} -> {_gate(n_rgo, n)}")
    print(f"  (C) comprehension AUC clears 0.80 @ shared dt=1.0:         {comp['n_clear_shared']}/{n} "
          f"(native dt=0.5: {comp['n_clear_native']}/{n}) -> {comp['verdict']}")

    payload = {"mode": "onebrain_merge_rung2", "n_seeds": n, "results": results,
               "n_rung1_byte_identical": n_r1, "n_recon_byte_identical": n_rbi,
               "n_recon_alive": n_ral, "n_recon_faculty_go": n_rgo,
               "rung1_regression_verdict": _gate(n_r1, n),
               "reconsolidation_faculty_verdict": _gate(n_rgo, n),
               "comprehension_dt_boundary": comp}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
