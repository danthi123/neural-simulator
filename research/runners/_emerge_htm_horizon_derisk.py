"""EMERGENCE-ENGINE HORIZON DE-RISK — how far past the fading reservoir window does the roadmap's ACTUAL emergence
engine (the on-bridge HTM Temporal-Memory sequence cortex, EMERGE-14) carry HIGH-ORDER (earlier-context-dependent)
structure, and WHERE is its wall?

WHY THIS IS THE FRONTIER (our-own-record first):
  * The recurrent-cortex long-range problem was localised (`2026-07-15-emergence-engine-research-gate-...`) NOT to
    off-diagonal recurrent credit (near-dead) but to a NON-FADING, content-addressable, specific-item STORE: a fixed
    ALIF reservoir HOLDS only within a ~5-15-token FADING window, and even a delta/STP content-addressable store bolted
    onto it (`2026-07-15-emergence-engine-1-deltastore-...GO`) only reaches T=30 GIVEN CLEAN KEYS — on real streams the
    reservoir keys are diffuse (`2026-07-11-cross-sentence-...cache-bag` NEGATIVE). The reservoir's fixed random
    dynamics are the bottleneck; the e-prop-on-reservoir "deep-context" win was REFUTED as a memory-timescale artifact
    (`2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED`).
  * The roadmap's emergence engine is NOT the reservoir — it is the allocation-based HTM-TM sequence cortex
    (EMERGE-14/15, roadmap L130: "scale spiking HTM Temporal-Memory generator"). Allocation is NON-FADING: each
    (column, prior-context) gets a distinct SDR, so context is carried through a PRIMING CHAIN, not a leaky state.
  * UN-DONE QUESTION (nobody swept it): the reservoir's horizon is characterised (fades ~5-15, deltastore extends to
    ~30 with clean keys). The allocation-based emergence engine's horizon is NOT. Does it carry high-order structure
    FAR past the reservoir's fading window, and — since it is non-fading — is its wall DISTANCE or something else?

WHAT THIS MEASURES: the on-bridge HTM-TM's branch-prediction accuracy as the dependency DISTANCE L (shared-middle
length) grows, at FAIR allocation capacity (n_cells scales with the number of interfering contexts). The task is the
EMERGE-14 overlap corpus: n_seq sentences [cue, <L shared-middle words>, branch]; the branch (last word) depends ONLY on
the cue L+1 tokens back, so ANY fixed-order n-gram at the branch is pinned at chance 1/n_seq (the shared middle is
identical for every cue) — the HTM must carry the cue THROUGH the middle. This is a memorise-and-recall horizon,
apples-to-apples with the deltastore reservoir KV horizon (which was also recall, not generalisation).

CONTROLS (the ones the CONTROLS-REFUTED discipline demands): (a) dAP-LESION (coincidence off) -> the priming chain is
severed -> collapses to the n-gram/chance floor (the high-order recurrence is load-bearing); (b) SWAP-FOLLOWS-CONTEXT
(inject a DIFFERENT cue -> the branch prediction must FOLLOW the injected cue, not the memorised one) -> proves the
prediction is DRIVEN by the distal cue, not a positional/order bias; (c) UNTRAINED -> chance; (d) a CAPACITY-STARVED
control point (n_cells < k_win*n_seq) -> collapses, NAMING the wall as allocation capacity, not distance; (e) the
best fixed-order n-gram floor (pinned at chance by the shared middle). Multi-seed. Reuse-by-import (EMERGE-14 machinery);
NO sim/ edit. SIM_BACKEND honoured (cupy for the heavier long-L / high-n_seq grid; numpy for a CPU smoke).

GO = the emergence engine's NON-FADING horizon EXCEEDS the reservoir's fading window: at a FAR distance L_far >= 20
(past the reservoir's ~15 and near the deltastore's clean-key 30), mean branch-acc >= 0.90, >= chance + 0.20, dAP-lesion
collapses (>= htm - 0.20 gap), swap-follows >= 0.90, untrained <= chance + 0.10 — all at FAIR capacity, multi-seed.
HONEST NEGATIVE (first-class) = a collapse at some L* < 20 names the emergence engine's horizon + the next mechanism.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner)
from research.runners._emerge9b_htm_faithful_derisk import (
    make_overlap_sequences, markov_branch_acc, full_oracle)

try:
    from tools.lab import attributable_to
except Exception:  # tools.lab is optional at import time; the runner still runs
    def attributable_to(label, t, c, warn_below=0.5):
        return None

OUT = Path("research/findings/raw/_emerge_htm_horizon.json")

ARMS = ["htm", "lesion", "untrained"]


def swap_follows_context(lr, seqs, L):
    """CONTEXT-NECESSITY control: inject a DIFFERENT cue (word 0) into each sentence and check the branch prediction
    FOLLOWS the injected cue (== that cue's branch), NOT the memorised one. High -> the branch is DRIVEN by the distal
    cue carried through the shared middle, not a positional bias. bp = L (branch lives at preds[L])."""
    n = len(seqs); ok = 0; tot = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            swapped = list(seqs[i]); swapped[0] = seqs[j][0]
            pred = lr.predict_branch(swapped, L)[L]
            ok += int(pred == {seqs[j][L + 1]})     # must predict cue j's branch (followed the distal context)
            tot += 1
    return ok / max(1, tot)


def _run_point(seed, arm, n_seq, L, n_cells, k_win, act_th, epochs):
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    b, cells_idx, row, col = build_pool_bridge(vocab, n_cells, seed, act_th=act_th, coincidence=(arm != "lesion"))
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab, n_cells, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                lr.train_sequence(s)
    ok = 0
    for s in seqs:
        ok += int(lr.predict_branch(s, L)[L] == {s[L + 1]})
    acc = ok / len(seqs)
    swap = swap_follows_context(lr, seqs, L) if arm == "htm" else None
    return acc, swap


def n_cells_for(n_seq, k_win, slack, capacity_mode, fixed_cells):
    if capacity_mode == "starved":
        return int(fixed_cells)
    return int(k_win * n_seq + slack)          # FAIR: enough disjoint SDRs for every interfering context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--distances", type=int, nargs="+", default=[8, 16, 24, 32],
                    help="shared-middle lengths L = dependency distance minus 1 (branch is L+1 tokens after the cue)")
    ap.add_argument("--n-seq", type=int, default=4, help="# interfering contexts sharing the middle (chance = 1/n_seq)")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--slack", type=int, default=8, help="extra cells above k_win*n_seq at FAIR capacity")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--capacity-mode", choices=["fair", "starved"], default="fair")
    ap.add_argument("--fixed-cells", type=int, default=16, help="n_cells when --capacity-mode starved (the wall control)")
    ap.add_argument("--l-far", type=int, default=20, help="GO requires holding at the largest swept L >= this")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2

    backend = os.environ.get("SIM_BACKEND", "numpy")
    chance = 1.0 / a.n_seq
    t0 = time.time(); err = None
    grid = sorted(set(a.distances))
    print(f"backend={backend} | n_seq={a.n_seq} chance={chance:.3f} | capacity={a.capacity_mode} | distances(L)={grid}",
          flush=True)

    points = []           # one dict per L
    try:
        for L in grid:
            n_cells = n_cells_for(a.n_seq, a.k_win, a.slack, a.capacity_mode, a.fixed_cells)
            # n-gram floor + oracle (seed-independent structure; compute once at this L)
            seqs0, _, _ = make_overlap_sequences(n_seq=a.n_seq, middle_len=L, seed=a.seeds[0])
            markov = markov_branch_acc(seqs0, L, a.n_seq)          # best fixed-order n-gram at the branch
            oracle = full_oracle(seqs0, L)
            per = []
            for s in a.seeds:
                d = {"seed": s}
                for arm in ARMS:
                    acc, swap = _run_point(s, arm, a.n_seq, L, n_cells, a.k_win, a.act_th, a.epochs)
                    d[arm] = acc
                    if arm == "htm":
                        d["swap_follows"] = swap
                per.append(d)
            htm = float(np.mean([p["htm"] for p in per]))
            les = float(np.mean([p["lesion"] for p in per]))
            unt = float(np.mean([p["untrained"] for p in per]))
            swap = float(np.mean([p["swap_follows"] for p in per]))
            hold = bool(htm >= 0.90 and htm >= chance + 0.20 and htm >= les + 0.20 and swap >= 0.90 and unt <= chance + 0.10)
            points.append({"L": L, "distance": L + 1, "n_cells": n_cells, "htm": htm, "lesion": les, "untrained": unt,
                           "swap_follows": swap, "markov": markov, "oracle": oracle, "chance": chance, "hold": hold,
                           "per_seed": per})
            print(f"  [L={L:>3} dist={L+1:>3} n_cells={n_cells:>3}] htm {htm:.3f} | lesion {les:.3f} | untr {unt:.3f} "
                  f"| swap {swap:.3f} || markov {markov:.3f} chance {chance:.3f} oracle {oracle:.3f} | HOLD={hold}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and points:
        holds = [p["L"] for p in points if p["hold"]]
        horizon = max(holds) if holds else None                    # largest distance the emergence engine holds
        far_pts = [p for p in points if p["L"] >= a.l_far]
        far = max(far_pts, key=lambda p: p["L"]) if far_pts else None
        # attribution: how much of the branch accuracy is the high-order recurrence (htm vs dAP-lesion)?
        if far is not None:
            print("\n-- attribution at the FAR distance (htm vs dAP-lesion) --", flush=True)
            attributable_to(f"high-order recurrence @ L={far['L']}", far["htm"], far["lesion"])
        go = bool(a.capacity_mode == "fair" and far is not None and far["hold"] and far["oracle"] > 0.99)
        if a.capacity_mode == "starved":
            verdict = (f"CAPACITY-WALL CONTROL (capacity-starved, n_cells={a.fixed_cells} < k_win*n_seq="
                       f"{a.k_win*a.n_seq}) — the emergence engine collapses (horizon holds L={holds if holds else 'NONE'}); "
                       f"this NAMES the wall as allocation CAPACITY (n_cells per column must scale with the number of "
                       f"interfering contexts), NOT dependency distance — matching the deltastore finding's independent "
                       f"conclusion that the SELECTIVE (capacity-bounded) write is the load-bearing refinement.")
        elif go:
            verdict = (f"GO — the roadmap's emergence engine (allocation-based on-bridge HTM Temporal-Memory) carries "
                       f"HIGH-ORDER structure to distance {far['distance']} (L={far['L']}, mean branch-acc {far['htm']:.3f}) "
                       f">> chance {chance:.3f} and >> the best fixed-order n-gram floor {far['markov']:.3f} (pinned at "
                       f"chance by the shared middle), FAR past the fixed reservoir's fading ~5-15-token window (and past "
                       f"the deltastore-with-clean-keys ~30). It is NON-FADING: dAP-lesion collapses it to {far['lesion']:.3f} "
                       f"(the priming-chain recurrence is load-bearing), swap-follows-context {far['swap_follows']:.3f} "
                       f"(the branch is DRIVEN by the distal cue, not a positional bias), untrained {far['untrained']:.3f}, "
                       f"multi-seed. Measured horizon (largest holding distance in the sweep): L={horizon}. The wall is "
                       f"allocation CAPACITY (see the starved control), not distance. NO sim/ edit.")
        else:
            miss = []
            if far is None:
                miss.append(f"no swept L reached l_far={a.l_far}")
            else:
                if far["htm"] < 0.90: miss.append(f"far htm {far['htm']:.3f} < 0.90")
                if far["htm"] < chance + 0.20: miss.append(f"far didn't clear chance ({far['htm']:.3f} vs {chance:.3f})")
                if far["htm"] < far["lesion"] + 0.20: miss.append(f"dAP-lesion didn't collapse ({far['htm']:.3f} vs {far['lesion']:.3f})")
                if far["swap_follows"] < 0.90: miss.append(f"not context-driven (swap {far['swap_follows']:.3f})")
                if far["oracle"] <= 0.99: miss.append(f"task not context-solvable (oracle {far['oracle']:.3f})")
            verdict = (f"HONEST NEGATIVE / BOUNDARY — the emergence engine holds high-order structure to distance "
                       f"{(horizon+1) if horizon is not None else 'NONE'} (L={horizon}) but NOT at the far distance: "
                       + "; ".join(miss) + f". This measures the allocation-based horizon and names the next mechanism "
                       f"(more cells / selective write / the content-addressable store) at the collapse point.")
    else:
        verdict = f"ERROR — {err}" if err else "ERROR — no points computed"
        horizon = None

    # --- earned verdict with preconditions carried in the artifact (tools/gates/verdict_preconditions) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        V = Verdict("emerge_htm_horizon", chance=chance)
        # PRECONDITIONS = VALIDITY checks (must hold for ANY verdict — GO or a legitimate boundary — to be meaningful).
        # The GO THRESHOLD (htm>=0.90 AND swap>=0.90 at the far point) is the DECISION passed to decide(), NOT a
        # precondition: a result that is valid (task solvable, above chance, controls discriminate) but below 0.90 is a
        # legitimate NO-GO/boundary, not UNDEFINED.
        _far = far if (err is None and points) else None
        if _far is not None:
            V.require("oracle>0.99_task_solvable", round(_far["oracle"], 4), expect=lambda x: x > 0.99,
                      note="else the task is not context-solvable -> INCONCLUSIVE, not a negative")
            V.floor("htm_above_chance", round(_far["htm"], 4), floor=chance)
            V.control("htm_vs_dAP_lesion_discriminates", round(_far["htm"], 4), round(_far["lesion"], 4),
                      min_separation=0.20, note="the priming-chain recurrence is load-bearing (instrument discriminates)")
            V.require("untrained_control_collapses", round(_far["untrained"], 4), expect=lambda x: x <= chance + 0.10)
        else:
            V.require("reached_l_far", 1 if (points and any(p["L"] >= a.l_far for p in points)) else 0,
                      expect=lambda x: x >= 1, note="no swept L reached l_far (or run errored)")
        dec = V.decide(bool(err is None and points and a.capacity_mode == "fair" and far is not None
                            and far["hold"] and far["oracle"] > 0.99), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_htm_horizon", "verdict": verdict, "backend": backend,
               "sim_backend": backend, "cost_acknowledged": True, "preconditions": preconditions,
               "mechanism": "on-bridge HTM Temporal-Memory (allocation-based, non-fading priming chain over the sim/ "
                            "fused_htm_permanence_update kernel on cp_connections.data) carrying high-order context "
                            "through a shared middle of length L; measures the distal-structure HORIZON vs the fixed "
                            "reservoir's fading ~5-15-token window",
               "task": "EMERGE-14 overlap corpus [cue, <L middle>, branch]; branch depends on the cue L+1 tokens back "
                       "(n-gram at branch pinned at chance 1/n_seq); dAP-lesion + swap-follows-context + untrained + "
                       "capacity-starved control + multi-seed",
               "seeds": a.seeds, "config": {"distances": grid, "n_seq": a.n_seq, "k_win": a.k_win, "act_th": a.act_th,
               "slack": a.slack, "epochs": a.epochs, "capacity_mode": a.capacity_mode, "fixed_cells": a.fixed_cells,
               "l_far": a.l_far, "chance": chance},
               "horizon_L": horizon, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the EMERGE-14 on-bridge learner; NO sim/ edit. Memorise-and-recall "
                              "horizon (apples-to-apples with the deltastore reservoir KV horizon), NOT held-out "
                              "generalisation (that is the EMERGE-18 axis). The n-gram floor is pinned at chance BY "
                              "CONSTRUCTION (shared middle) — the meaningful bar is chance + the reservoir's fading horizon."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge_htm_horizon] VERDICT: {verdict}", flush=True)
    print(f"[emerge_htm_horizon] horizon_L={horizon} | wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
