"""SLEEP-REPLAY BUDGET SWEEP DE-RISK (WS-2, 2026-08-09): isolate replay-QUANTITY from store-FIDELITY.

THE QUESTION. The host-store sleep-replay consolidation runner
(_teacher_loop_sleep_replay_consolidation_derisk.py, main f8b3e7cf) is self-generated + robust (replay > no-replay
every seed; SCRAMBLE lesion -> ~0) but UNDER-CONSOLIDATES: 6-seed REPLAY frac_recalled mean 0.55 (range
0.20-0.90), no-replay 0.13, ceiling = interleaved 8/10 = 0.8. The declared likely cause is a LOSSY host
mean-vector engram (WS-1 store-fidelity). THIS runner tests the OTHER hypothesis (WS-2): is the 0.55 partly just
TOO LITTLE REPLAY? -- i.e. does MORE/BETTER replay on the SAME host store reach the 0.8 ceiling on its own?

WHAT IT DOES. Reuse-by-import of the sleep-replay runner's OWN arm (`_run_arm`) + Hippocampus + world helpers --
NO sim/ edit, NO re-derivation. For ONE seed it holds the net / world / per-fact WAKE teaching budget FIXED
(identical to the baseline) and sweeps ONLY the offline SLEEP replay budget across a grid of
(replay_epochs, replay_per_fact) ordered by total replay work = replay_epochs * replay_per_fact. It measures the
REPLAY-arm sequential retention frac_recalled@N=10 at each budget -> the retention-vs-replay-budget CURVE.
Anchored by the NOREPLAY floor (budget-independent) and guarded by SCRAMBLE (content-lesioned, same compute) at
the MIN and MAX budget -- the anti-cheat that must hold even when the replay compute is largest.

BRAIN-BASED / SELF-GENERATED (unchanged from the imported runner). The engram store is the brain's own captured
trace; replay is self-generated from that store with a brain-owned RNG; the consolidation phase has NO `env`
parameter and never calls env.draw (teacher/world ABSENT). This runner adds NO new store and NO host fact->slot
table -- it only turns the existing replay-budget knobs. (The store is still a LOSSY host mean-vector -- that is
exactly the WS-1 residual this sweep is designed to EXPOSE if budget plateaus below 0.8.)

TEETH (WS-2):
  (a) MONOTONE: REPLAY frac_recalled RISES with replay budget (non-decreasing across the work-ordered grid, noise
      tol; and max-budget > min-budget by a real margin) -- a real effect of quantity, not noise.
  (b) CEILING: at the BEST (max) budget, is frac_recalled at/near 0.8? Reported as the gap to 0.8. If the curve
      PLATEAUS below 0.8, that is the first-class HONEST NEGATIVE: store-fidelity (WS-1) is the real lever, not
      budget.
  (c) IMMEDIATE ACQUISITION stays perfect (>=0.9) at every budget -- more replay must not break learning new facts.
  (d) SELF-GENERATED holds under max compute: SCRAMBLE@max-budget (content lesioned, identical compute) forgets
      like NOREPLAY (frac <= noreplay + 0.10) -- the rise is the STORED CONTENT, not the extra gradient steps.

HONEST NEGATIVE IS A DELIVERABLE. If (a) holds but (b) plateaus below 0.8, the verdict is a well-formed NEGATIVE
that MAPS the residual to store-fidelity (WS-1). GO here = budget alone reaches the ceiling; PLATEAU = it does not.

SEED NOTE. The 0.55 is a 6-seed MEAN; seed 42 already reaches 0.90 at the baseline budget (top of the range) and
has NO headroom -- a budget sweep on 42 is flat and uninformative. The informative seeds are the LOW ones
(~0.20-0.55). This runner takes ONE seed per process; run it across 42..47 (the 6-seed command below) and read the
per-seed curves + the aggregate. The single-seed SMOKE should target a LOW seed (scout first with --grid baseline).

RUN (numpy is FASTER here -- the net is tiny, cupy is launch-bound; verified 2026-08-09: numpy N=10 arm ~110s,
cupy >180s and slower). Budget points run in a process Pool over cores.
  SCOUT (find the low seeds -- baseline budget only, replay arm, all 6 seeds in parallel, ~2min):
    for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed $s --grid baseline \
      --out research/findings/raw/sleep_replay_budget_scout_s$s.json & done; wait
  SMOKE (single low seed, full budget grid):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed <LOW_SEED> --grid full \
      --pool 10 --out research/findings/raw/sleep_replay_budget_s<LOW_SEED>.json
  6-SEED (the deliverable curve; one seed per process, full grid, in parallel):
    for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --seed $s --grid full --pool 6 \
      --out research/findings/raw/sleep_replay_budget_s$s.json & done; wait
    then aggregate: python -m research.runners._teacher_loop_sleep_replay_budget_sweep_derisk --aggregate \
      research/findings/raw/sleep_replay_budget_s{42,43,44,45,46,47}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny net -> numpy beats launch-bound cupy here
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
# reuse-by-import: the sleep-replay runner's OWN sequential arm (net build + wake-teach + hippocampal capture +
# self-replay consolidation) and the scaling world. NO sim/ edit, NO new store, NO host fact->slot table.
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import _run_arm  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import N_ACT  # noqa: E402
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "sleep_replay_budget.json"

# Budget grid: (replay_epochs, replay_per_fact) ordered by total replay work = epochs*per_fact. The baseline
# anchor (24, 16) sits in the middle; the grid spans 64x work (96 .. 6144) so a monotone rise (or a plateau) is
# unambiguous. Two 1-D sweeps (epochs @ per_fact=16; per_fact @ epochs=24) cross at the anchor.
FULL_GRID = [(6, 16), (12, 16), (24, 8), (24, 16), (48, 16), (24, 32), (96, 16), (24, 64), (96, 64)]
BASELINE_POINT = (24, 16)


def _work(pt):
    return pt[0] * pt[1]


def _build_world(seed, n_max, d_p, noise):
    """Replicate the sleep-replay runner's world setup EXACTLY so every arm/point is like-for-like: protos are
    created in ref order from the seeded env RNG; each arm resets env.rng to seed+101 before drawing (done inside
    the caller before every _run_arm)."""
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)
    return env, referents


def _one_point(args):
    """One independent arm run for (arm, replay_epochs, replay_per_fact). Fresh net + fresh world (deterministic
    from seed) so points are parallel-safe. Returns the retention curve + immediate-acq + wall for this budget."""
    (arm, re_epochs, re_per_fact, cfg) = args
    env, referents = _build_world(cfg["seed"], cfg["n_max"], cfg["d_p"], cfg["noise"])
    env.rng = np.random.default_rng(cfg["seed"] + 101)   # SAME reset the baseline run does per arm
    K = int(cfg["n_max"]); chance = 1.0 / K; n_in = cfg["d_p"] + N_ACT
    t0 = time.time()
    out = _run_arm(arm, cfg["seed"], referents, env, K, n_in, cfg["hidden"], cfg["settle"], cfg["epochs"],
                   cfg["batch"], cfg["eprop_lr"], cfg["w_clip"], cfg["n_draws"], cfg["milestones"], cfg["test_n"],
                   re_epochs, re_per_fact, cfg["replay_noise"], chance)
    big = max((int(k) for k in out["retention_curve"]), default=None)
    frac = out["retention_curve"][str(big)]["frac_recalled"] if big else float("nan")
    n_rec = out["retention_curve"][str(big)]["n_recalled"] if big else None
    return {"arm": arm, "replay_epochs": re_epochs, "replay_per_fact": re_per_fact,
            "work": re_epochs * re_per_fact, "largest_N": big, "frac_recalled": float(frac),
            "n_recalled": n_rec, "mean_immediate_acq": out["mean_acquire_acc_immediate"],
            "retention_curve": out["retention_curve"], "wall_seconds": round(time.time() - t0, 1)}


def run(seed, grid_name, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_noise, pool):
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    cfg = dict(seed=seed, n_max=n_max, milestones=milestones, hidden=hidden, settle=settle, epochs=epochs,
               batch=batch, eprop_lr=eprop_lr, w_clip=w_clip, n_draws=n_draws, d_p=d_p, noise=noise,
               test_n=test_n, replay_noise=replay_noise)
    if grid_name == "baseline":
        grid = [BASELINE_POINT]
    elif grid_name == "full":
        grid = list(FULL_GRID)
    else:
        raise ValueError("grid must be baseline|full")
    grid = sorted(set(grid), key=_work)
    minpt, maxpt = grid[0], grid[-1]

    # tasks: NOREPLAY once (budget-independent floor); REPLAY at every budget; SCRAMBLE at min+max budget.
    tasks = [("noreplay", *BASELINE_POINT, cfg)]
    for (re_e, re_pf) in grid:
        tasks.append(("replay", re_e, re_pf, cfg))
    for pt in sorted(set([minpt, maxpt]), key=_work):
        tasks.append(("scramble", pt[0], pt[1], cfg))

    t0 = time.time()
    if pool and pool > 1:
        import multiprocessing as mp
        with mp.Pool(processes=min(pool, len(tasks))) as p:
            results = p.map(_one_point, tasks)
    else:
        results = [_one_point(t) for t in tasks]
    wall = round(time.time() - t0, 1)

    noreplay = next(r for r in results if r["arm"] == "noreplay")
    replays = sorted([r for r in results if r["arm"] == "replay"], key=lambda r: r["work"])
    scrambles = sorted([r for r in results if r["arm"] == "scramble"], key=lambda r: r["work"])
    return {"seed": seed, "grid": grid_name, "n_max": n_max, "chance": 1.0 / n_max, "milestones": milestones,
            "ceiling_interleaved": 0.8, "baseline_point": list(BASELINE_POINT), "wall_seconds": wall,
            "config": {k: cfg[k] for k in ("hidden", "settle", "epochs", "batch", "eprop_lr", "w_clip",
                                           "n_draws", "d_p", "noise", "test_n", "replay_noise")},
            "noreplay": noreplay, "replay_curve": replays, "scramble": scrambles}


def _verdict(result):
    """WS-2 teeth. (a) monotone rise with budget; (b) best-budget vs 0.8 ceiling; (c) immediate acq perfect at
    every budget; (d) SCRAMBLE@max forgets like noreplay (self-generated under max compute)."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    ceiling = result["ceiling_interleaved"]
    chance = result["chance"]
    noreplay_frac = result["noreplay"]["frac_recalled"]
    replays = result["replay_curve"]
    fracs = [r["frac_recalled"] for r in replays]
    works = [r["work"] for r in replays]
    min_frac, max_frac = fracs[0], fracs[-1]
    best_frac = max(fracs)
    best_idx = int(np.argmax(fracs))
    best_pt = (replays[best_idx]["replay_epochs"], replays[best_idx]["replay_per_fact"])
    acqs = [r["mean_immediate_acq"] for r in replays]
    min_acq = min(acqs)
    scr_max = next((r for r in result["scramble"] if r["work"] == works[-1]), result["scramble"][-1])
    scramble_max_frac = scr_max["frac_recalled"]

    # monotone: non-decreasing across work-ordered grid within a tolerance (retention is a noisy 0.1-quantized
    # measure at N=10), AND max-budget beats min-budget by a real margin.
    tol = 0.15
    violations = sum(1 for i in range(1, len(fracs)) if fracs[i] < fracs[i - 1] - tol)
    rise = max_frac - min_frac
    # rank correlation (Spearman) budget-work vs frac, reported (not a hard gate; the margin is the gate).
    def _spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        rx = rx - rx.mean(); ry = ry - ry.mean()
        d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / d) if d > 0 else 0.0
    rho = _spearman(works, fracs)

    attributable_to("replay-budget quantity (max vs min budget)", max_frac, min_frac)
    attributable_to("stored content @max budget (replay vs scramble)", max_frac, scramble_max_frac)

    reaches_ceiling = best_frac >= ceiling - 1e-9
    monotone_ok = (violations == 0 and max_frac > min_frac + 0.15)

    # INSTRUMENT PRECONDITIONS ONLY are registered as Verdict checks -- the things that must hold for the result to
    # be INTERPRETABLE at all: (c) learning actually works (immediate acquisition perfect), and (d) the replay is
    # self-generated/content-dependent (SCRAMBLE, content-lesioned at max compute, does not consolidate). If EITHER
    # fails the run is UNDEFINED (an instrument failure). The WS-2 HYPOTHESIS itself -- "more replay lifts retention
    # to the 0.8 ceiling" -- is NOT a precondition: its FALSITY is a first-class NEGATIVE (budget is not the lever
    # -> store-fidelity/WS-1 is), NOT an instrument failure. So it enters `decide` via `go`, not as a check whose
    # failure would void the run. (Earned here: a flat-below-ceiling curve is the deliverable, and must read NO-GO.)
    v = Verdict("sleep-replay budget sweep (WS-2)", chance=chance)
    v.floor("(c) immediate acquisition stays perfect at every budget [instrument]", min_acq, floor=0.9)
    v.require("(d) SCRAMBLE@max forgets like no-replay (self-generated under max compute) [instrument]",
              scramble_max_frac <= noreplay_frac + 0.10, expect=True,
              note=f"scramble@max {scramble_max_frac:.2f} vs noreplay {noreplay_frac:.2f}")
    v.reads("chance", result, chance)

    # the WS-2 HYPOTHESIS outcome (reported, drives GO vs NEGATIVE once the instrument preconditions hold):
    budget_suffices = monotone_ok and reaches_ceiling
    go = budget_suffices  # decide() still returns UNDEFINED if an instrument precondition failed
    decision = v.decide(go=go)
    # classify honestly. NEGATIVE (not UNDEFINED) whenever the instrument preconditions held but budget did not
    # reach the ceiling -- that is the mapped, first-class result: store-fidelity (WS-1) is the lever.
    instrument_ok = (min_acq >= 0.9 and scramble_max_frac <= noreplay_frac + 0.10)
    if not instrument_ok:
        outcome = "UNDEFINED_INSTRUMENT (acquisition or self-generation control failed)"
    elif budget_suffices:
        outcome = "BUDGET_REACHES_CEILING (more replay alone reaches 0.8)"
    elif monotone_ok and not reaches_ceiling:
        outcome = "NEGATIVE_PLATEAU_BELOW_CEILING (budget rises retention monotonically but not to 0.8 -> store-fidelity/WS-1 is the lever)"
    else:
        outcome = "NEGATIVE_FLAT (more replay does NOT lift retention -> store-fidelity/WS-1 is the lever, not budget)"
    return {"noreplay_frac_recalled": noreplay_frac, "min_budget_frac": min_frac, "max_budget_frac": max_frac,
            "best_frac": best_frac, "best_point": list(best_pt), "gap_to_ceiling": float(ceiling - best_frac),
            "rise_max_minus_min": float(rise), "spearman_work_frac": rho, "monotone_violations": violations,
            "min_immediate_acq": min_acq, "scramble_max_frac": scramble_max_frac,
            "budget_reaches_ceiling": bool(reaches_ceiling), "ws2_outcome": outcome, **decision}


def _aggregate(paths):
    """6-seed aggregate: mean retention-vs-budget curve + per-seed best/gap. Emits a <!--derived--> marker line so
    the finding's derived table is machine-flagged."""
    seeds = []
    by_work = {}
    for pth in paths:
        d = json.loads(Path(pth).read_text())
        r = d["result"] if "result" in d else d
        seed = r["seed"]
        vd = d.get("verdict", {})
        seeds.append({"seed": seed, "noreplay": r["noreplay"]["frac_recalled"],
                      "min_budget": r["replay_curve"][0]["frac_recalled"],
                      "max_budget": r["replay_curve"][-1]["frac_recalled"],
                      "best": vd.get("best_frac"), "gap_to_ceiling": vd.get("gap_to_ceiling"),
                      "outcome": vd.get("ws2_outcome")})
        for rc in r["replay_curve"]:
            by_work.setdefault(rc["work"], {"epochs": rc["replay_epochs"], "per_fact": rc["replay_per_fact"],
                                            "fracs": []})["fracs"].append(rc["frac_recalled"])
    print("\n<!--derived-->")
    print("6-SEED SLEEP-REPLAY BUDGET SWEEP -- retention vs replay budget")
    print(f"{'work':>6} {'(re_e,re_pf)':>14} {'mean_frac':>10} {'sd':>6} {'n':>3}")
    for w in sorted(by_work):
        e = by_work[w]; f = np.array(e["fracs"])
        print(f"{w:>6} {'('+str(e['epochs'])+','+str(e['per_fact'])+')':>14} "
              f"{f.mean():>10.3f} {f.std():>6.3f} {len(f):>3}")
    nore = np.array([s["noreplay"] for s in seeds]); maxb = np.array([s["max_budget"] for s in seeds])
    best = np.array([s["best"] for s in seeds if s["best"] is not None])
    print(f"\nnoreplay mean {nore.mean():.3f} | max-budget mean {maxb.mean():.3f} | "
          f"best mean {best.mean():.3f} | ceiling 0.8")
    print("per-seed:", [(s["seed"], round(s["noreplay"], 2), round(s["max_budget"], 2), s["outcome"]) for s in seeds])
    reach = sum(1 for s in seeds if s["best"] is not None and s["best"] >= 0.8 - 1e-9)
    print(f"\nAGGREGATE: {reach}/{len(seeds)} seeds reach the 0.8 ceiling at best budget. "
          f"{'BUDGET SUFFICES' if reach >= 5 else 'BUDGET DOES NOT SUFFICE -> store-fidelity (WS-1) is the lever'}")
    return {"seeds": seeds, "by_work": {str(k): {"epochs": v["epochs"], "per_fact": v["per_fact"],
            "mean": float(np.mean(v["fracs"])), "sd": float(np.std(v["fracs"])), "n": len(v["fracs"])}
            for k, v in by_work.items()}, "seeds_reaching_ceiling": reach}


def main():
    ap = argparse.ArgumentParser(description="Sleep-replay BUDGET SWEEP (WS-2): does MORE replay on the host store "
                                             "reach the 0.8 ceiling, isolating replay-quantity from store-fidelity?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", choices=["baseline", "full"], default="full")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40, help="per-fact WAKE teaching epochs (FIXED = baseline)")
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--pool", type=int, default=1, help="parallel budget-point workers (numpy, 1 core each)")
    ap.add_argument("--aggregate", nargs="+", default=None, help="aggregate per-seed JSONs into the 6-seed curve")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.aggregate:
        _aggregate(a.aggregate)
        return 0

    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.grid, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_noise, a.pool)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_sleep_replay_budget_sweep", "seed": a.seed, "grid": a.grid,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[budget-sweep] seed {a.seed} grid={a.grid} | noreplay {result['noreplay']['frac_recalled']:.2f} "
          f"| chance {result['chance']:.2f} | ceiling 0.80", flush=True)
    print("  retention vs replay budget (work = re_epochs * re_per_fact):", flush=True)
    for r in result["replay_curve"]:
        print(f"    work {r['work']:>5} (re_e={r['replay_epochs']:>3}, re_pf={r['replay_per_fact']:>3}): "
              f"frac {r['frac_recalled']:.2f} ({r['n_recalled']}/{r['largest_N']}) | "
              f"imm-acq {r['mean_immediate_acq']:.3f} | {r['wall_seconds']:.0f}s", flush=True)
    for s in result["scramble"]:
        print(f"    SCRAMBLE work {s['work']:>5}: frac {s['frac_recalled']:.2f}", flush=True)
    print(f"  => WS-2 OUTCOME: {verdict['ws2_outcome']}", flush=True)
    print(f"     rise(max-min) {verdict['rise_max_minus_min']:+.2f} | rho {verdict['spearman_work_frac']:+.2f} "
          f"| best {verdict['best_frac']:.2f} at {verdict['best_point']} | gap-to-0.8 "
          f"{verdict['gap_to_ceiling']:+.2f} | min-acq {verdict['min_immediate_acq']:.3f} "
          f"| scramble@max {verdict['scramble_max_frac']:.2f} | VERDICT {verdict['status']}", flush=True)
    print(f"[budget-sweep] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
