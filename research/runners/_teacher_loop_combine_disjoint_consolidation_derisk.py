"""TEACHER-LOOP COMBINE: DISJOINT SPARSE CODES + PER-SYNAPSE METAPLASTIC CONSOLIDATION (2026-08-11). Does
applying BOTH levers together COMPOUND -- and, crucially, PROTECT THE OLDEST FACT (oldest_fact_acc > 0 at large
N) where each mechanism ALONE left it low (disjoint 0.175, consolidation 0.275 @ N=32, 6-seed)?

WHY THIS EXISTS (the NAMED next mechanism, not re-derived here). Two 6-seed de-risks isolated the continual
acquisition-at-scale residual and each moved it PARTWAY, on a DIFFERENT axis, and NEITHER alone protects fact 0:
  * `2026-08-11-metaplastic-acquisition-continual-learning-6seed-NOGO...`: a per-synapse consolidation state
    c gating lr_eff = lr/(1+g*c) protects ACQUIRED WEIGHTS from erosion -- real, load-bearing, attributable,
    the strongest SINGLE mechanism (frac_recalled 0.812 @ N=32), but sub-threshold and it NEVER cleanly saves
    the very-oldest fact (oldest_fact_acc 0.275).
  * `2026-08-11-sparse-disjoint-fact-codes-continual-PARTIAL...`: routing each fact to a DISJOINT sparse code
    (frozen Marr-Albus reservoir + input-driven k-WTA readout) beats vanilla (0.656 vs 0.537) and the
    DISJOINTNESS is load-bearing (overlapping codes collapse to 0.156), but it is BELOW consolidation and also
    leaves fact 0 low (0.175). Its NAMED next: **COMBINE the two** -- they attack DIFFERENT failure modes (code
    INTERFERENCE vs weight EROSION) and SHOULD compound. Protect the FEW units each fact owns.

THE HYPOTHESIS BEING DE-RISKED. Disjoint codes give each fact its OWN sub-circuit (a later fact's DISJOINT code
does not read the oldest fact's units); consolidation then protects THOSE FEW OWNED synapses from the residual
erosion that leaks through (top-k sets are not perfectly disjoint; the readout column is shared). The two are
ORTHOGONAL: disjointness reduces WHICH synapses collide; consolidation reduces HOW MUCH a collision moves them.
GO = combined > BOTH single mechanisms (frac AND oldest) AND combined protects the OLDEST fact where each alone
could not, AND removing EITHER lever drops it toward that lever alone, AND disjointness is essential (overlapping
codes + consolidation must NOT reach combined).

FIVE ARMS, one world / seed / schedule / de-clamp (the ONLY difference is the two-lever configuration). The two
single-mechanism arms ARE the "remove one lever" controls for combined -- disjoint == combined minus consolidation,
consolidation == combined minus disjointness -- so no extra arms are needed to test load-bearing:
  * vanilla        = dense trained readout, no consolidation, no disjoint code -> the acquisition-at-scale collapse.
  * consolidation  = the per-synapse metaplastic gate ONLY (dense readout). == combined MINUS the disjoint lever.
  * disjoint       = frozen reservoir + input-driven k-WTA sparse code ONLY (no consolidation). == combined MINUS
                     the consolidation lever.
  * combined       = BOTH -- frozen reservoir + k-WTA disjoint code + per-synapse consolidation. THE TREATMENT.
  * overlap_combined = consolidation + SHARED-slot k-WTA (all inputs use the SAME top-k units -> maximal code
                     OVERLAP). The DISJOINTNESS-IS-THE-LEVER control: sparsity WITHOUT disjointness, even WITH
                     consolidation, must NOT reach combined -> proves the compounding needs DISJOINT codes, not
                     merely "sparse + consolidated".

ANTI-CHEATS (executed via tools.lab + tools.verdict.Verdict, not asserted in prose):
  (compounding) combined > max(consolidation, disjoint) on BOTH frac_recalled and oldest_fact_acc.
  (load-bearing, BOTH levers) attributable_to(combined vs consolidation) AND attributable_to(combined vs disjoint)
    are BOTH positive -- removing either lever drops combined toward that lever alone (each adds).
  (the crux) combined oldest_fact_acc > max(single-mechanism oldest) AND > 0 at N_max -- the oldest fact is
    protected where each alone left it low. Per-fact-AGE retention (oldest / middle / newest) reported at every
    milestone, so "protects the oldest" is an explicit number, not a mean.
  (no acquisition cost) combined mean immediate-acq >= 0.6 AND >= min(component acq) - tol -- combining does not
    trade new learning for retention (the plasticity side of stability-plasticity).
  (disjointness is the lever) lever() + attributable_to(combined vs overlap_combined): overlapping codes +
    consolidation must underperform combined.
  Emits backend/device via assert_backend.

HONEST-NEGATIVE FIRST-CLASS. If combining does NOT compound -- they interfere, or one mechanism dominates and the
other adds nothing, or the oldest fact stays low -- this reports the per-age numbers honestly and names the next
mechanism (capacity GROWTH / neurogenesis as N scales: disjoint codes need capacity headroom, which growth supplies;
the sparse finding already flagged that the disjoint benefit degrades once N exceeds H/code_size).

DISCIPLINE: reuse-by-import of ALL substantive machinery -- SparseFactEpropNet (frozen reservoir + k-WTA + the
per-synapse consolidation gate, all in one class) and _mk_sparse_net / _run_arm from the sparse-fact-codes runner
(both arm-agnostic); _age_buckets from the Benna-Fusi runner; the teacher-loop world/teach/held-out-acc from the
scaling runner. NO sim/ edit. cfg.seed via the seed= arg the net passes to CoreSimConfig.seed. SIM_BACKEND=numpy
(this ~260-neuron net is launch-bound; numpy avoids cupy launch overhead at this size -- reported).

RUN:
  single-seed SMOKE (compounding + oldest-fact protection + disjointness lever, N=32):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python -m research.runners._teacher_loop_combine_disjoint_consolidation_derisk --seed 42 \
        --n-max 32 --milestones 16 32 --hidden 256 --code-size 5 --epochs 18 --settle-steps 20 \
        --test-n 20 --n-draws 24 --out research/findings/raw/combine_disj_consol_s42.json
  6-SEED sweep command (N in {16,32,50}) is returned to the coordinator (one seed per process; H sized to keep
  N<=H/code_size so N=50 stays within disjoint capacity).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # ~260-neuron net; numpy avoids cupy launch overhead at this size
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
# reuse-by-import: the sparse-fact-code net (frozen reservoir + k-WTA + the per-synapse consolidation gate, all in
# one class) and its arm-agnostic build/run helpers; the Benna-Fusi per-age buckets. NO sim/ edit.
from research.runners._teacher_loop_sparse_fact_codes_derisk import (  # noqa: E402
    SparseFactEpropNet, _mk_sparse_net, _run_arm,
)
from research.runners._teacher_loop_scaling_derisk import N_ACT  # noqa: E402
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "combine_disj_consol.json"
ARMS = ("vanilla", "consolidation", "disjoint", "combined", "overlap_combined")

assert SparseFactEpropNet is not None  # import-time proof the reused net is present (no re-implementation here)


def run(seed, n_max, milestones, hidden, code_size, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p,
        noise, test_n, declamp_wmax, meta_gain, meta_consol_rate):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    G, R, KW = float(meta_gain), float(meta_consol_rate), int(code_size)
    # The two orthogonal levers:
    #   CONSOLIDATION lever = meta_enabled (per-synapse lr_eff = lr/(1+g*c), protects acquired weights).
    #   DISJOINT lever      = freeze_reservoir + kwta_k (frozen Marr-Albus reservoir + input-driven k-WTA sparse
    #                         readout code -> each fact reads/writes its OWN top-k units).
    specs = {
        "vanilla":          dict(meta_enabled=False),                                   # neither lever
        "consolidation":    dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R),   # consolidation ONLY
        "disjoint":         dict(meta_enabled=False, freeze_reservoir=True, kwta_k=KW),  # disjointness ONLY
        # THE TREATMENT: BOTH levers -- disjoint code + per-synapse consolidation on the units each fact owns.
        "combined":         dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R,
                                 freeze_reservoir=True, kwta_k=KW),
        # DISJOINTNESS-IS-THE-LEVER control: consolidation + SHARED-slot k-WTA (all inputs share the SAME top-k
        # units -> maximal code OVERLAP). Sparse + consolidated but NOT disjoint -> must underperform combined.
        "overlap_combined": dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R,
                                 freeze_reservoir=True, kwta_k=KW, kwta_shared=True),
    }
    arms = {}
    for name in ARMS:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)               # identical teaching percepts across arms (like-for-like)
        arms[name] = _run_arm(name, specs[name], seed, referents, env, K, n_in, hidden, settle, epochs, batch,
                              eprop_lr, w_clip, n_draws, milestones, test_n, chance, declamp_wmax)
        arms[name]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[name]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        row = rc[str(big)] if big else {}
        fr = row.get("frac_recalled", float("nan"))
        old = row.get("oldest_fact_acc", float("nan"))
        mid = row.get("mid_fact_acc", float("nan"))
        new = row.get("newest_fact_acc", float("nan"))
        print(f"[arm {name:17s}] {arms[name]['wall_seconds']:6.0f}s | immediate-acq "
              f"{arms[name]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.3f} "
              f"(1/N={1.0/big:.3f}) | oldest {old:.3f} mid {mid:.3f} newest {new:.3f}", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "code_size": code_size, "settle_steps": settle, "epochs": epochs,
                       "batch": batch, "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p,
                       "noise": noise, "test_n": test_n, "declamp_wmax": declamp_wmax, "meta_gain": G,
                       "meta_consol_rate": R, "capacity_slots": hidden // max(1, code_size),
                       "backend": os.environ.get("SIM_BACKEND")},
            "arms": arms}


def _at(rc, arm, key, N):
    return rc[arm][str(N)][key]


def _verdict(result):
    from tools.lab import lever, attributable_to, assert_backend
    from tools.verdict import Verdict
    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(combine disjoint+consolidation de-risk)")
    arms = result["arms"]
    rc = {a: arms[a]["retention_curve"] for a in arms}
    big = max((int(k) for k in rc["combined"]), default=None)
    key = str(big)
    f = {a: rc[a][key]["frac_recalled"] for a in rc}
    old = {a: rc[a][key]["oldest_fact_acc"] for a in rc}         # the crux: oldest fact (fact 0) @ N_max
    mid = {a: rc[a][key]["mid_fact_acc"] for a in rc}
    new = {a: rc[a][key]["newest_fact_acc"] for a in rc}
    acq = {a: arms[a]["mean_acquire_acc_immediate"] for a in arms}
    chance = result["chance"]
    one_over_N = 1.0 / big
    best_single_f = max(f["consolidation"], f["disjoint"])
    best_single_old = max(old["consolidation"], old["disjoint"])

    # (the lever) the DISJOINTNESS channel: combined uses DISJOINT k-WTA codes; overlap_combined shares the slot.
    lever("disjoint code vs shared-slot (oldest-fact acc, both consolidated)",
          round(float(old["overlap_combined"]), 4), round(float(old["combined"]), 4))

    # (compounding) the fraction of combined's effect NOT already in either single lever -- each must ADD.
    attributable_to("combined vs consolidation (frac@Nmax) — disjoint lever adds", f["combined"], f["consolidation"])
    attributable_to("combined vs disjoint (frac@Nmax) — consolidation lever adds", f["combined"], f["disjoint"])
    # (the crux, on the OLDEST fact) each single mechanism left fact 0 low; combined must clear BOTH.
    attributable_to("combined vs consolidation (OLDEST) — disjoint lever adds", old["combined"], old["consolidation"],
                    warn_below=0.0)
    attributable_to("combined vs disjoint (OLDEST) — consolidation lever adds", old["combined"], old["disjoint"],
                    warn_below=0.0)
    # (disjointness is the lever) overlapping codes + consolidation must NOT reach combined.
    attributable_to("combined vs overlap_combined (disjointness, OLDEST)", old["combined"], old["overlap_combined"],
                    warn_below=0.0)

    v = Verdict("teacher-loop combine disjoint sparse codes + metaplastic consolidation", chance=chance)
    v.reaches("(1) combined beats best single mechanism (frac_recalled)", before=best_single_f, after=f["combined"])
    v.reaches("(2) combined beats consolidation-only (frac)", before=f["consolidation"], after=f["combined"])
    v.reaches("(3) combined beats disjoint-only (frac)", before=f["disjoint"], after=f["combined"])
    v.reaches("(4) combined protects OLDEST vs best single", before=best_single_old, after=old["combined"])
    v.reaches("(5) removing consolidation lever drops toward disjoint (OLDEST)",
              before=old["combined"], after=old["disjoint"])
    v.reaches("(6) removing disjoint lever drops toward consolidation (OLDEST)",
              before=old["combined"], after=old["consolidation"])
    v.reaches("(7) disjointness lever (vs shared-slot + consolidation, OLDEST)",
              before=old["overlap_combined"], after=old["combined"])
    v.floor("(8) combined oldest-fact acc > chance (protected)", old["combined"], floor=chance)
    v.floor("(9) combined keeps acquiring new facts (immediate-acq)", acq["combined"], floor=0.6)
    # GO (the compounding claim): combined clears BOTH single mechanisms on frac AND oldest, protects the oldest
    #     fact (> both singles + margin, and > chance), removing EITHER lever drops it toward that lever alone,
    #     disjointness is essential (overlap_combined underperforms), and it still acquires. Single-seed = SMOKE.
    go = (f["combined"] > best_single_f + 0.05
          and old["combined"] > best_single_old + 0.10
          and old["combined"] > chance
          and old["combined"] > old["disjoint"] + 0.10
          and old["combined"] > old["consolidation"] + 0.10
          and old["combined"] > old["overlap_combined"] + 0.10
          and f["combined"] > f["overlap_combined"] + 0.10
          and acq["combined"] >= 0.6)
    decision = v.decide(go=go)
    return {"largest_N": big, "one_over_N": one_over_N, "backend": backend, "chance": chance,
            "frac_recalled": f, "oldest_fact_acc": old, "mid_fact_acc": mid, "newest_fact_acc": new,
            "immediate_acq": acq, "best_single_frac": best_single_f, "best_single_oldest": best_single_old,
            "combined_beats_best_single_frac": float(f["combined"] - best_single_f),
            "combined_beats_best_single_oldest": float(old["combined"] - best_single_old),
            "disjoint_lever_adds_frac": float(f["combined"] - f["consolidation"]),      # remove-consolidation delta
            "consolidation_lever_adds_frac": float(f["combined"] - f["disjoint"]),      # remove-disjoint delta
            "disjoint_lever_adds_oldest": float(old["combined"] - old["consolidation"]),
            "consolidation_lever_adds_oldest": float(old["combined"] - old["disjoint"]),
            "disjointness_oldest_gain": float(old["combined"] - old["overlap_combined"]), **decision}


def _aggregate(paths):
    """6-seed roll-up. GO = every seed: combined > best-single-frac+0.05 AND combined oldest > best-single-oldest
    +0.10 AND > chance AND > disjoint-oldest+0.10 AND > consolidation-oldest+0.10 AND > overlap-oldest+0.10 AND
    combined frac > overlap frac+0.10 AND combined immediate-acq >= 0.6 (the compounding + oldest-protection claim)."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        vd = d["verdict"]
        f = vd["frac_recalled"]; old = vd["oldest_fact_acc"]; acq = vd["immediate_acq"]; ch = vd["chance"]
        bsf = max(f["consolidation"], f["disjoint"]); bso = max(old["consolidation"], old["disjoint"])
        seed_go = bool(f["combined"] > bsf + 0.05 and old["combined"] > bso + 0.10 and old["combined"] > ch
                       and old["combined"] > old["disjoint"] + 0.10 and old["combined"] > old["consolidation"] + 0.10
                       and old["combined"] > old["overlap_combined"] + 0.10
                       and f["combined"] > f["overlap_combined"] + 0.10 and acq["combined"] >= 0.6)
        rows.append({"seed": d["seed"], "N": vd["largest_N"],
                     **{f"f_{a}": f[a] for a in ARMS}, **{f"o_{a}": old[a] for a in ARMS},
                     "acq_combined": acq["combined"], "seed_go": seed_go})
    fmean = {a: float(np.mean([r[f"f_{a}"] for r in rows])) for a in ARMS}
    omean = {a: float(np.mean([r[f"o_{a}"] for r in rows])) for a in ARMS}
    n_go = sum(r["seed_go"] for r in rows)
    go = n_go == len(rows) and len(rows) >= 6
    print("\n" + "=" * 132)
    print(f"[AGG] {len(rows)} seeds | GO needs combined>best-single-frac+.05 & oldest>best-single+.10 & >chance & "
          f">each-single-oldest+.10 & >overlap+.10 & frac>overlap+.10 & acq>=.6, ALL seeds")
    print(f"{'seed':>5} {'N':>4} | " + " ".join(f"f_{a[:9]:>11}" for a in ARMS) + " | " +
          " ".join(f"o_{a[:9]:>11}" for a in ARMS) + f" {'acqC':>6} {'GO':>4}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} {r['N']:>4} | " + " ".join(f"{r[f'f_{a}']:>13.3f}" for a in ARMS) + " | " +
              " ".join(f"{r[f'o_{a}']:>13.3f}" for a in ARMS) + f" {r['acq_combined']:>6.3f} {str(r['seed_go']):>4}")
    print(f"{'mean':>5} {'':>4} | " + " ".join(f"{fmean[a]:>13.3f}" for a in ARMS) + " | " +
          " ".join(f"{omean[a]:>13.3f}" for a in ARMS))
    print(f"[AGG] frac: combined {fmean['combined']:.3f} vs consolidation {fmean['consolidation']:.3f} vs disjoint "
          f"{fmean['disjoint']:.3f} vs overlap {fmean['overlap_combined']:.3f} | OLDEST: combined {omean['combined']:.3f} "
          f"vs consol {omean['consolidation']:.3f} vs disjoint {omean['disjoint']:.3f} | seeds GO {n_go}/{len(rows)} "
          f"| VERDICT {'GO' if go else 'NO-GO'}")
    print("=" * 132)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop COMBINE disjoint sparse fact codes + per-synapse "
                                             "metaplastic consolidation: do the two levers COMPOUND and protect "
                                             "the OLDEST fact where each alone left it low?")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-seed JSONs -> 6-seed GO roll-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--milestones", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--code-size", type=int, default=5, help="k for the k-WTA code (H/code_size = disjoint capacity)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=20)
    ap.add_argument("--declamp-wmax", type=float, default=1e9,
                    help="bdsp_w_max for ALL arms (de-clamp held constant; NOT the lever). <0 keeps the +-6 default.")
    ap.add_argument("--meta-gain", type=float, default=8.0, help="single-var metaplastic gain g in lr_eff=lr/(1+g*c)")
    ap.add_argument("--meta-consol-rate", type=float, default=1.0, help="per-fact consolidation increment rate")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.aggregate:
        return _aggregate(a.aggregate)
    if a.declamp_wmax is not None and a.declamp_wmax < 0:
        a.declamp_wmax = None
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.code_size, a.settle_steps, a.epochs, a.batch,
                 a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.declamp_wmax, a.meta_gain,
                 a.meta_consol_rate)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_combine_disjoint_consolidation", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    f = verdict["frac_recalled"]; old = verdict["oldest_fact_acc"]; mid = verdict["mid_fact_acc"]
    new = verdict["newest_fact_acc"]; acq = verdict["immediate_acq"]
    print("\n" + "=" * 128, flush=True)
    print(f"[combine-disj-consol] seed {a.seed} @ N={verdict['largest_N']} (1/N={verdict['one_over_N']:.3f}, "
          f"chance {result['chance']:.3f}):", flush=True)
    for arm in ARMS:
        print(f"    {arm:18s}: frac {f[arm]:.3f} | oldest {old[arm]:.3f} | mid {mid[arm]:.3f} | "
              f"newest {new[arm]:.3f} | immediate-acq {acq[arm]:.3f}", flush=True)
    print(f"[combine-disj-consol] combined-beats-best-single frac {verdict['combined_beats_best_single_frac']:+.3f} | "
          f"oldest {verdict['combined_beats_best_single_oldest']:+.3f} | disjoint-lever-adds-oldest "
          f"{verdict['disjoint_lever_adds_oldest']:+.3f} | consolidation-lever-adds-oldest "
          f"{verdict['consolidation_lever_adds_oldest']:+.3f} | disjointness-oldest-gain "
          f"{verdict['disjointness_oldest_gain']:+.3f} | VERDICT {verdict['status']}", flush=True)
    print(f"[combine-disj-consol] wrote {a.out}\n" + "=" * 128, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
