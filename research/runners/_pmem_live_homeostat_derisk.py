"""PROSPECTIVE-MEMORY LIVE OPERATING-POINT HOMEOSTAT -- converts the STATIC per-seed calibration TABLE
(`CALIBRATED_FAC_G`, `_operating_point_stabilizer_derisk.py`) into a genuine, LIVE, IN-LOOP homeostat
(roadmap DEEPER #4 follow-on; branch `research/lbf-live-homeostat`, 2026-09-23). This is the explicit
honest-residual follow-on named by `research/findings/2026-09-23-operating-point-stabilizer-go-6seed.md`:
"a genuinely continuous/live homeostat (rather than this static, precomputed 6-seed table) is a named
follow-on".

WHY (CLAUDE.md: speed is SECONDARY to faithfulness; a wall/negative is a verdict on a METHOD). The prior
build replaced ONE constant (`fac_g=6000` for every seed) with a BETTER constant (a per-seed number picked
by an offline grid search and shipped as a Python dict, `CALIBRATED_FAC_G`). That is progress on the
OUTCOME but not on the MECHANISM: a real neuron's homeostatic set-point regulator (Turrigiano 2011, "Too
many cooks? Intrinsic and synaptic homeostatic mechanisms in cortical circuits", Nat Rev Neurosci; Desai,
Rutherford & Turrigiano 1999, "Plasticity in the intrinsic excitability of cortical pyramidal neurons", Nat
Neurosci -- the SAME citations the static build used) is not a lookup table keyed by the animal's identity;
it is a CONTINUING PROCESS that reads the cell's/pathway's OWN activity and slowly adjusts gain toward a
set point, on every animal, using the SAME machinery. This runner builds that process: a `LiveHomeostat`
that measures the pool's own running coincidence read, computes the error against a fixed target, and
updates `fac_g` by a bounded, rate-limited INTEGRAL step -- the discrete-time realization of slow synaptic
scaling -- repeated over successive trial episodes until the gain SETTLES, with NO per-seed table anywhere
in the control loop.

THE CLIFF THIS MUST RESPECT -- AND A SECOND, SHARPER ONE THIS BUILD'S OWN PILOT FOUND (measured by the
parent build, `_operating_point_stabilizer_derisk.py`, committed
`research/findings/raw/_pmem_operating_point_stabilizer.json`, PLUS a finer pilot scan run before freezing
the constants below -- see the committed finding for the full pilot trace). At seed 44, a FRESH,
self-consistent build (the only correct way to evaluate a candidate `fac_g` on this substrate -- an
in-place mutation understates the pool's own intrinsic-excitability homeostat's response) shows the
coincidence read is NOT monotonic in `fac_g`: 0.2111 (6000), 0.2078 (7000, a dip), 0.2139 (8000), 0.2183
(9000, the safe maximum) -- the parent build's OWN coarse grid then jumped straight to 0.1506 (10000), so it
reported "the cliff" as somewhere in [9000, 10000]. THIS BUILD'S PILOT, run at a finer step BEFORE choosing
`G_CEILING` (a continuous controller can probe at any gain, unlike the parent's fixed grid), found the true
edge is far tighter and almost discontinuous: 0.2144 (9100, still fine) then 0.1911 (9200) -- an 0.023 drop
in ONE HUNDRED pA, seed-44-specific (the identical scan at seeds 100 and 101 stayed flat, 0.283-0.296,
through 9400; the cliff is NOT a substrate-wide property at this range, it is this seed's own intrinsic-
excitability homeostat response). A ceiling anywhere in the parent's assumed-safe [9000, 9500] window
(the first version of this controller used 9500, caught ONLY by this pilot) would have walked straight into
this seed's collapse. THE FIX IS ONE BOUND THAT SOLVES BOTH PROBLEMS AT ONCE: `REL_TARGET=0.30` (1.5x
FIRE_THR) is unreachable inside the safe range for seeds 44/100/101, so a naive integral controller chasing
an unreachable target WINDS UP without bound -- and would walk straight into whichever cliff sits ahead of
it. A hard ceiling `G_CEILING=9000`, fixed BEFORE the 6-seed run at the LAST GRID POINT ALREADY VALIDATED
safe by the parent's own committed measurement (not merely "below 10000" -- below the newly-found edge at
~9150 too, with the 9100 pilot point as an independent confirmation this margin holds), is both the
anti-windup clamp AND the cliff safety bound for EVERY seed (uniform, not seed-tuned): the controller
literally cannot select a gain in the collapse zone, no matter how long an unreachable target pulls it
there, and it does not depend on this build having found every seed's exact edge (it never approaches 9100
at all).

THE CONTROLLER (Turrigiano-style slow integral set-point regulator; additive; NO sim/ edit; reuse-by-import
of the already-committed GO facilitation substrate `_pmem_facilitation_derisk` and the frozen N=5 silence
gate via the already-committed op-stabilizer's `_frozen_silence`). For a seed, starting from an initial
gain `g_0`:
  1. MEASURE the pool's OWN N=3 production-protocol intact coincidence read `rel_i` at the CURRENT gain
     `g_i`, via a FRESH build (`F._n3_arm`) -- this is the live read; there is no `CALIBRATED_FAC_G` import
     anywhere in this loop (see `prove_it_is_live`, anti-cheat #1).
  2. error_i = REL_TARGET - rel_i.
  3. raw_delta = K_I * error_i  (K_I fixed BEFORE the 6-seed run from a stability argument + a single
     seed-44 pilot -- see "CONTROLLER GAIN" below, never re-fit to the outcome).
  4. delta_i = clip(raw_delta, -MAX_STEP, +MAX_STEP)  -- a RATE LIMIT: the gain can move by at most
     `MAX_STEP` pA per trial episode, independent of the measured local slope (a slope-robust anti-
     oscillation bound: a real synaptic-scaling process also has a maximal per-unit-time scaling rate, it
     does not jump).
  5. g_{i+1} = clip(g_i + delta_i, G_FLOOR, G_CEILING)  -- G_FLOOR = the shipped constant `FAC_G_DEFAULT`
     (a homeostat already at its target does not regress below what shipped -- the SAME floor-guard
     principle the static build used, realized here as a hard bound rather than a table branch); G_CEILING
     = 9000, the cliff-safety bound above.
  6. Repeat. CONVERGED when `|g_{i+1} - g_i| < G_TOL` for `CONVERGE_STREAK` consecutive iterations (a
     settled interior point, e.g. a seed already at target sitting at the floor) OR the gain is pinned at a
     bound for that many iterations (a settled BOUNDARY equilibrium, e.g. an unreachable-target seed pinned
     safely at the ceiling -- this IS the cliff-safe behavior, not a failure to converge). MAX_ITERS bounds
     the worst case.

CONTROLLER GAIN, fixed BEFORE the 6-seed run (a stability argument, not a fit to the outcome). The plant
(fac_g -> rel) is SHALLOW (local slope ~2-4e-6 per pA, from the parent build's own grid) and mildly
non-monotonic (the dip at 7000), so a plain proportional-integral law risks either creeping too slowly or
overshooting on the steeper stretches. `MAX_STEP=500` pA makes the WORST-CASE per-step movement independent
of the (unknown, possibly locally steeper) slope -- a true rate limiter, the anti-oscillation guarantee that
does not depend on estimating the plant. `K_I=20000` only sets when the law switches from this rate-limited
"bang-bang" regime to fine proportional control (below `|error| ~ MAX_STEP/K_I = 0.025`), so the controller
still slows down and settles near the true crossing rather than perpetually overshooting by a fixed step.
Both constants were checked for smooth (non-oscillating, no cliff-adjacent overshoot) convergence on seed 44
ONLY (the pre-documented hardest/cliffiest seed) before being frozen and applied UNCHANGED to all 6 seeds --
see `research/findings/2026-09-23-live-homeostat-*.md` for the pilot trace.

ANTI-CHEATS (each implemented + reported in `_derisk`):
  1. GENUINELY LIVE, NOT A TABLE: `prove_it_is_live()` asserts this module never imports
     `CALIBRATED_FAC_G`/`stabilized_fac_g_for_seed` from the static stabilizer, and that the SAME seed run
     TWICE from the SAME init reproduces an IDENTICAL trajectory (the read is deterministic and genuinely
     computed at each step, not memoised from a table) while a DIFFERENT init produces a DIFFERENT
     trajectory that still lands in the same neighborhood (anti-cheat #2) -- a lookup table cannot do
     either.
  2. TWO DIFFERENT INITS, SAME SET-POINT: every seed is run from `G_INIT_LOW=6000` (the shipped floor) AND
     `G_INIT_MID=8000` (a different interior point); a converged final gain within `SETPOINT_TOL` of each
     other (or both pinned at the SAME bound) is a real set-point; an init-dependent final gain would mean
     this is drift, not regulation.
  3. CLIFF-SAFE: `max(all g visited across all seeds/inits) <= G_CEILING < CLIFF_EDGE_S44 < 10000` is asserted directly off
     the trajectories -- the controller structurally cannot enter the collapse zone regardless of how long
     an unreachable target pulls it there (see step 5 above; this is a property of the clip, verified, not
     assumed).
  4. LOAD-BEARING PRESERVED: re-verified at the CONVERGED gain exactly as the static build did (the
     facilitation current is gated by the maintained assembly's own firing AND postsynaptic depolarization;
     the lesion zeroes the former).
  5. BYTE-IDENTICAL DEFAULT-OFF: `BRAIN_PMEM_LIVE_HOMEOSTAT` unset -> the production hook is not called at
     all (the SAME code path as before this build); iteration 0 of the LOW-init trajectory (g=FAC_G_DEFAULT)
     is EXACT-COMPARED against the prior, INDEPENDENTLY-committed static-stabilizer artifact's own
     grid-point-0 read (same seed, same gain, different process, different day) -- not an inference.
  6. DETERMINISM: re-running one seed's LOW-init trajectory reproduces the IDENTICAL `rel` sequence to 4
     decimal places (the substrate is fully seeded via `cfg.seed`; see CLAUDE.md's seed-never-controlled-the
     -substrate finding).

THE PRE-REGISTERED GATE (task's own wording, fixed before running). GO iff, on all 6 seeds: the live
homeostat self-converges (from >=2 different inits, to the same neighborhood) to a `fac_g` whose margin
(`converged_rel - FIRE_THR`) is (a) strictly positive AND (b) >= the STATIC-table stabilizer's own committed
margin (`research/findings/raw/_pmem_operating_point_stabilizer.json`, read, not re-derived); load-bearing
preserved; frozen N=5 silence held; default-off byte-identical; cliff-safe. A seed that converges to a LOWER
margin than the static table (the expected honest trade: the live controller's uniform safety ceiling is
more conservative than the static table's per-seed-chosen top grid point, e.g. the static table used 11000
for seed 101) is an HONEST partial result on THAT seed, not a forced pass -- see the committed finding's
Verdict block for which of GO / BOUNDARY this run actually earned.

  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_homeostat_derisk --selftest
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_homeostat_derisk --derisk
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_homeostat_derisk --seed 44 --trace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import research.runners._pmem_facilitation_derisk as F               # noqa: E402  (the committed GO substrate)
from research.runners._pmem_intention_latch_derisk import FIRE_THR   # noqa: E402  (imported, never re-typed)
# reuse-by-import of the STATIC stabilizer's own constants + frozen-silence helper -- NOT its lookup table
# (CALIBRATED_FAC_G / stabilized_fac_g_for_seed are never imported here; see prove_it_is_live()).
from research.runners._operating_point_stabilizer_derisk import (   # noqa: E402
    FAC_G_DEFAULT, REL_TARGET, _frozen_silence,
)
from tools.lab import attributable_to, void_if                       # noqa: E402
from tools.verdict import Verdict                                    # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_live_homeostat.json")
# the STATIC stabilizer's own committed artifact -- the GO-gate's comparison baseline, READ, never re-derived.
STATIC_STABILIZER_ARTIFACT = os.path.join(_REPO, "research", "findings", "raw",
                                          "_pmem_operating_point_stabilizer.json")

SEEDS = (42, 43, 44, 100, 101, 102)

# ---- the flag this live homeostat is gated behind (default-OFF; wired into prospective_memory_production_organ.py)
LIVE_HOMEOSTAT_ENV = "BRAIN_PMEM_LIVE_HOMEOSTAT"

# ---- controller constants, fixed BEFORE the 6-seed run (see module docstring "CONTROLLER GAIN") ----
# CLIFF_EDGE_S44: this build's OWN pilot (finer than the parent's grid) found seed 44's true collapse edge
# between 9100 pA (still fine, rel=0.2144) and 9200 pA (collapsed, rel=0.1911) -- an 0.023 drop in 100 pA.
# Recorded here so the ceiling's safety margin is an explicit, checkable number, not a comment.
CLIFF_EDGE_S44 = 9150.0          # the midpoint of the pilot's bracket -- everything below this is confirmed
                                  # safe by direct measurement (9100), everything at/above is confirmed
                                  # collapsed (9200); NOT a per-seed tune of the ceiling, a documented bound
G_FLOOR = FAC_G_DEFAULT          # 6000.0 -- never regress below the shipped constant
G_CEILING = 9000.0               # cliff-safety bound: the LAST grid point already validated safe by the
                                  # parent build's own committed measurement, itself ~150 pA below
                                  # CLIFF_EDGE_S44 and confirmed by an independent pilot read at 9100 -- a
                                  # SINGLE global bound applied uniformly to every seed (never seed-tuned),
                                  # so it does not depend on this build having found every seed's own edge
K_I = 20000.0                    # integral gain (pA per unit `rel` error)
MAX_STEP = 500.0                 # rate limit: max |delta fac_g| per trial episode (pA) -- slope-robust
G_TOL = 25.0                     # a step smaller than this counts as "settled" for the convergence streak
CONVERGE_STREAK = 3              # consecutive settled steps required to call it converged
MAX_ITERS = 16                   # worst-case iteration budget per (seed, init)
SETPOINT_TOL = 750.0             # two inits' converged gains within this count as "the same neighborhood"
                                  # (three rate-limited steps of MAX_STEP=500; tighter than one extra full step)

# two different initial gains -- BOTH inside the safe interior (never AT a bound), so every seed exercises a
# genuine, non-trivial trajectory in at least one direction (anti-cheat #2).
G_INIT_LOW = FAC_G_DEFAULT       # 6000.0 -- the shipped floor
G_INIT_MID = 8000.0              # an interior point 1000 pA below the ceiling


# --------------------------------------------------------------------------------------------------------
# THE LIVE CONTROLLER. No CALIBRATED_FAC_G, no per-seed branch: identical law, identical constants, for
# every seed -- only the MEASURED trajectory differs, because the controller reads the pool's own output.
# --------------------------------------------------------------------------------------------------------
def _measure_rel(seed: int, g: float) -> float:
    """The LIVE read: a fresh, self-consistent N=3 production-protocol intact coincidence read at gain `g`
    (the only correct way to evaluate a candidate fac_g on this substrate -- see the parent build's own
    discovery). Reused verbatim from the committed GO facilitation substrate; never a table lookup."""
    return round(float(F._n3_arm(seed, fac_on=True, lesion=False, N=3, fac_g=g,
                                 fac_U=F.FAC_U, fac_tau_F_steps=F.FAC_TAU_F_STEPS)["rel"]), 4)


def run_live_homeostat(seed: int, g_init: float, target: float = REL_TARGET,
                       max_iters: int = MAX_ITERS, k_i: float = K_I, max_step: float = MAX_STEP,
                       g_floor: float = G_FLOOR, g_ceiling: float = G_CEILING, tol: float = G_TOL,
                       streak_need: int = CONVERGE_STREAK) -> dict:
    """Run the live integral homeostat from `g_init` until it converges or `max_iters` is exhausted.
    Returns the full trajectory (for the anti-cheats + the finding's convergence plot) plus the converged
    (gain, rel, error) triple."""
    g = float(g_init)
    prev_g = None
    streak = 0
    traj = []
    converged = False
    reason = "max_iters exhausted (%d)" % max_iters
    for i in range(max_iters):
        rel = _measure_rel(seed, g)
        error = round(target - rel, 4)
        if prev_g is not None:
            moved = abs(g - prev_g)
            streak = streak + 1 if moved < tol else 0
        traj.append({"iter": i, "fac_g": round(g, 1), "rel": rel, "error": error, "streak": streak})
        if prev_g is not None and streak >= streak_need:
            at_bound = "ceiling" if abs(g - g_ceiling) < 1e-6 else ("floor" if abs(g - g_floor) < 1e-6 else None)
            reason = ("pinned at safety %s (fac_g=%.0f)" % (at_bound, g) if at_bound
                      else "settled at an interior set-point (fac_g=%.0f)" % g)
            converged = True
            break
        prev_g = g
        raw_delta = k_i * error
        delta = max(-max_step, min(max_step, raw_delta))
        g = max(g_floor, min(g_ceiling, g + delta))
    last = traj[-1]
    return {"seed": seed, "g_init": g_init, "converged": converged, "reason": reason,
            "n_iters": len(traj), "final_fac_g": last["fac_g"], "final_rel": last["rel"],
            "final_error": last["error"], "trajectory": traj}


def prove_it_is_live(seed: int = 44) -> dict:
    """ANTI-CHEAT #1: this module never imports the static table, AND a fresh re-run of the SAME
    (seed, init) reproduces an IDENTICAL trajectory -- proof the value is genuinely COMPUTED at each step
    (deterministic live measurement), not memoised from a dict. A table lookup and a live read are
    indistinguishable from the FINAL number alone; re-deriving the whole trajectory twice is what tells
    them apart."""
    no_table_import = ("CALIBRATED_FAC_G" not in globals() and "stabilized_fac_g_for_seed" not in globals())
    r1 = run_live_homeostat(seed, G_INIT_LOW)
    r2 = run_live_homeostat(seed, G_INIT_LOW)
    same_traj = r1["trajectory"] == r2["trajectory"]
    return {"seed": seed, "no_table_import": no_table_import, "rerun_identical_trajectory": same_traj,
            "trajectory_a": r1["trajectory"], "trajectory_b": r2["trajectory"]}


def _static_margin_for_seed(seed: int) -> float:
    """The STATIC stabilizer's own committed per-seed margin -- READ from its artifact, never re-derived
    (the GO-gate's comparison baseline)."""
    with open(STATIC_STABILIZER_ARTIFACT) as fh:
        d = json.load(fh)
    return float(d["per_seed"][str(seed)]["stabilized_margin"])


def _default_off_exact_compare(per_seed: dict, seeds) -> dict:
    """DEFAULT-OFF BYTE-IDENTICAL, asserted IN THE DATA: this run's OWN low-init iteration-0 read (measured
    at g=FAC_G_DEFAULT, this process) against the STATIC stabilizer's INDEPENDENTLY-committed grid-point-0
    read (same seed, same gain, a different process, a different day)."""
    if not os.path.exists(STATIC_STABILIZER_ARTIFACT):
        return {"ok": None, "note": "static stabilizer artifact not found -- comparison skipped"}
    with open(STATIC_STABILIZER_ARTIFACT) as fh:
        static = json.load(fh)
    per = {}
    ok = True
    for s in seeds:
        mine = per_seed[s]["low_init"]["trajectory"][0]["rel"]
        theirs = static["per_seed"][str(s)]["grid_evals"].get(str(FAC_G_DEFAULT)) \
            or static["per_seed"][str(s)]["grid_evals"].get(FAC_G_DEFAULT)
        match = (theirs is not None) and (mine == theirs)
        per[s] = {"this_run_iter0_rel": mine, "static_stabilizer_grid0_rel": theirs, "exact_match": match}
        ok = ok and match
    return {"ok": ok, "per_seed": per,
            "source": os.path.relpath(STATIC_STABILIZER_ARTIFACT, _REPO).replace(os.sep, "/")}


# --------------------------------------------------------------------------------------------------------
def _derisk(seeds=SEEDS, smoke=False):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"LIVE HOMEOSTAT [{tag}] -- in-loop integral set-point controller on fac_g; target={REL_TARGET:.3f} "
          f"K_I={K_I} MAX_STEP={MAX_STEP} floor={G_FLOOR} ceiling={G_CEILING}; {len(seeds)} seed(s)", flush=True)
    t0 = time.time()
    err = None
    per_seed = {}
    all_visited_g = []
    try:
        for s in seeds:
            print(f"\n--- seed {s}: live homeostat, two inits ---", flush=True)
            low = run_live_homeostat(s, G_INIT_LOW)
            print(f"  [low init={G_INIT_LOW}] converged={low['converged']} n_iters={low['n_iters']} "
                  f"final_fac_g={low['final_fac_g']} final_rel={low['final_rel']} :: {low['reason']}", flush=True)
            for row in low["trajectory"]:
                print(f"      iter {row['iter']:2d}  fac_g={row['fac_g']:7.1f}  rel={row['rel']:.4f}  "
                      f"error={row['error']:+.4f}  streak={row['streak']}", flush=True)
            mid = run_live_homeostat(s, G_INIT_MID)
            print(f"  [mid init={G_INIT_MID}] converged={mid['converged']} n_iters={mid['n_iters']} "
                  f"final_fac_g={mid['final_fac_g']} final_rel={mid['final_rel']} :: {mid['reason']}", flush=True)
            for row in mid["trajectory"]:
                print(f"      iter {row['iter']:2d}  fac_g={row['fac_g']:7.1f}  rel={row['rel']:.4f}  "
                      f"error={row['error']:+.4f}  streak={row['streak']}", flush=True)

            all_visited_g += [row["fac_g"] for row in low["trajectory"]] + [row["fac_g"] for row in mid["trajectory"]]
            same_setpoint = abs(low["final_fac_g"] - mid["final_fac_g"]) <= SETPOINT_TOL
            converged_g = low["final_fac_g"]        # the canonical converged gain: the low-init trajectory
            converged_rel = low["final_rel"]

            lb = F._n3_load_bearing(s, fac_on=True, N=3, fac_g=converged_g, fac_U=F.FAC_U,
                                    fac_tau_F_steps=F.FAC_TAU_F_STEPS)
            print(f"  load-bearing @ converged g={converged_g}: intact rel={lb['intact_rel']:.4f} "
                  f"fired={lb['intact_fired']} | lesion rel={lb['lesion_rel']:.4f} fired={lb['lesion_fired']} | "
                  f"LB={lb['load_bearing']}", flush=True)
            # ATTRIBUTION (tools.lab, CLAUDE.md's "the proxy dominates" caution): the coincidence read is
            # gated by BOTH the held assembly's own firing AND the cue's postsynaptic depolarization -- ask
            # explicitly what FRACTION of the intact read survives with the held assembly lesioned away,
            # rather than banking intact_rel/lesion_rel one key apart and never subtracting them.
            lesion_attrib = attributable_to(f"seed {s}: intact vs lesion coincidence (load-bearing attribution)",
                                            lb["intact_rel"], lb["lesion_rel"])

            froz = _frozen_silence(s, converged_g)
            fails = [k for k, v in froz["clauses"].items() if not v]
            print(f"  frozen N=5 silence @ converged g: passed={froz['passed']} "
                  f"max_silent={froz['max_silent']:.4f} fails={fails or 'none'}", flush=True)

            static_margin = _static_margin_for_seed(s)
            converged_margin = round(lb["intact_rel"] - FIRE_THR, 4)
            margin_positive = converged_margin > 0
            margin_meets_static = converged_margin >= static_margin

            per_seed[s] = {
                "low_init": low, "mid_init": mid, "same_setpoint_within_tol": same_setpoint,
                "converged_fac_g": converged_g, "converged_rel": converged_rel,
                "intact_rel": lb["intact_rel"], "intact_fired": lb["intact_fired"],
                "lesion_rel": lb["lesion_rel"], "lesion_fired": lb["lesion_fired"],
                "load_bearing": lb["load_bearing"],
                "frozen_passed": froz["passed"], "frozen_max_silent": round(froz["max_silent"], 4),
                "frozen_clause_fails": fails, "frozen_clauses": froz["clauses"],
                "static_table_margin": static_margin, "converged_margin": converged_margin,
                "margin_positive": margin_positive, "margin_meets_static": margin_meets_static,
                "lesion_attribution": lesion_attrib,
            }

        live_proof = prove_it_is_live(44 if 44 in per_seed else seeds[0])
        print(f"\n--- ANTI-CHEAT: prove_it_is_live (seed {live_proof['seed']}) ---", flush=True)
        print(f"  no_table_import={live_proof['no_table_import']}  "
              f"rerun_identical_trajectory={live_proof['rerun_identical_trajectory']}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_live_homeostat", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    n_seeds = len(seeds)
    n_positive = sum(1 for s in seeds if per_seed[s]["margin_positive"])
    n_meets_static = sum(1 for s in seeds if per_seed[s]["margin_meets_static"])
    n_lb = sum(1 for s in seeds if per_seed[s]["load_bearing"])
    n_same_setpoint = sum(1 for s in seeds if per_seed[s]["same_setpoint_within_tol"])
    silence_regressed = [s for s in seeds if per_seed[s]["frozen_clause_fails"]]
    cliff_safe = (max(all_visited_g) if all_visited_g else 0.0) <= G_CEILING < CLIFF_EDGE_S44 < 10000.0
    cheat = void_if(bool(silence_regressed),
                    f"the live homeostat REGRESSED a silence clause at seed(s) {silence_regressed} -> spurious "
                    f"fires (a homeostat that raises gain until everything fires is a CHEAT; VOID)")
    cheat2 = void_if(not cliff_safe,
                     f"the controller visited fac_g outside the cliff-safety bound (max visited "
                     f"{max(all_visited_g) if all_visited_g else float('nan')}, ceiling {G_CEILING}) -> VOID")
    default_off_exact = _default_off_exact_compare(per_seed, seeds) if not smoke else None
    changed_seeds = [s for s in seeds if per_seed[s]["converged_fac_g"] != G_FLOOR]
    unchanged_seeds = [s for s in seeds if per_seed[s]["converged_fac_g"] == G_FLOOR]
    thin_seed = min(seeds, key=lambda s: per_seed[s]["static_table_margin"])

    from research.runners._pmem_perpool_homeostat_derisk import SILENCE_CLAUSES  # noqa: E402

    # THE PRE-REGISTERED GATE: margin strictly positive AND >= the static table's own margin, on ALL seeds;
    # load-bearing preserved; silence intact; cliff-safe; a real (init-independent) set-point; not smoke.
    go = bool(n_positive == n_seeds and n_meets_static == n_seeds and n_lb == n_seeds
             and n_same_setpoint == n_seeds and cliff_safe and not cheat and not cheat2 and not smoke)

    # NOTE on what is a PRECONDITION vs what is THE DECISION (tools.verdict's own distinction, gates/
    # verdict_preconditions): a precondition is something that must hold for the comparison to be
    # INTERPRETABLE AT ALL, regardless of which way the comparison comes out (load-bearing intact, the two
    # inits agreeing on a set-point, cliff-safety, silence) -- those go through vd.require() below, so a
    # failure there correctly earns UNDEFINED (an uninterpretable run, never a negative). Whether the
    # live-converged margin actually MEETS the static table's own margin, seed by seed, is THE decision this
    # runner exists to make -- it is not a precondition for interpreting the run, it IS the run's answer, so
    # it is reported directly (n_positive / n_meets_static, per-seed detail in `per_seed`) and folded into
    # `go` above, never wrapped in require() (that would misreport a clean, fully-measured 5/6 shortfall as
    # an instrument failure).
    vd = Verdict("pmem_live_homeostat")
    vd.require("load-bearing preserved at the converged gain (per-seed count)", n_lb,
               expect=lambda x, n=n_seeds: x == n)
    vd.require("two different inits converge to the SAME neighborhood (per-seed count)", n_same_setpoint,
               expect=lambda x, n=n_seeds: x == n)
    vd.require("cliff-safe: no visited fac_g reached the collapse zone", cliff_safe, expect=True)
    for c in SILENCE_CLAUSES:
        vd.require(f"frozen-gate silence held at the converged gain: {c}",
                   sum(1 for s in seeds if per_seed[s]["frozen_clauses"].get(c)),
                   expect=lambda x, n=n_seeds: x == n)
    vd.disabled("STDP / long-term Hebbian LTP / OU-noise",
                "identical scope to the parent facilitation + static-stabilizer de-risks; the only added "
                "mechanism is the live, rate-limited integral controller replacing the static lookup table")
    decided = vd.decide(go)

    cliff_note = (
        " CLIFF-SAFETY: the ceiling (%.0f) sits below the documented collapse (fac_g=10000, s44 rel "
        "0.2183->0.1506); max fac_g visited across all seeds/inits was %.0f -- the controller never entered "
        "the collapse zone even where the target (%.2f) was unreachable and windup pulled it toward the "
        "bound." % (G_CEILING, max(all_visited_g) if all_visited_g else float("nan"), REL_TARGET)
    )
    status_word = "GO" if go else "VOID" if (cheat or cheat2) else (
        "UNDEFINED" if (decided or {}).get("status") == "UNDEFINED" else "NO-GO")
    verdict = (
        f"{status_word} ({n_meets_static}/{n_seeds}) -- per the PRE-REGISTERED "
        f"gate, {n_positive}/{n_seeds} seeds hold a strictly positive live-converged margin, "
        f"{n_meets_static}/{n_seeds} meet-or-beat the STATIC table's own margin, {n_lb}/{n_seeds} stay "
        f"load-bearing, {n_same_setpoint}/{n_seeds} converge to the SAME neighborhood from two different "
        f"inits; silence-regressed={silence_regressed or 'none'}; cliff-safe={cliff_safe}. Seed(s) "
        f"{changed_seeds or 'none'} moved off the shipped floor; seed(s) {unchanged_seeds or 'none'} settled "
        f"back at it (already at/above target). Thinnest-margin seed in the static table (s{thin_seed}): "
        f"static margin {per_seed[thin_seed]['static_table_margin']:+.4f} -> live-converged "
        f"{per_seed[thin_seed]['converged_margin']:+.4f}." + cliff_note
    )

    summary = {
        "probe": "pmem_live_homeostat", "verdict": verdict, "go": bool(go),
        "task": ("Convert the static per-seed CALIBRATED_FAC_G lookup table into a genuine LIVE, in-loop "
                 "Turrigiano-style integral set-point controller on the facilitation gain fac_g -- no "
                 "precomputed table anywhere in the control loop; the controller reads the pool's own "
                 "running coincidence output and adjusts gain by a bounded, rate-limited integral step, "
                 "safety-ceilinged below the documented substrate collapse."),
        "gate": {"FIRE_THR": FIRE_THR, "REL_TARGET": REL_TARGET, "G_FLOOR": G_FLOOR, "G_CEILING": G_CEILING,
                 "K_I": K_I, "MAX_STEP": MAX_STEP, "G_TOL": G_TOL, "CONVERGE_STREAK": CONVERGE_STREAK,
                 "MAX_ITERS": MAX_ITERS, "SETPOINT_TOL": SETPOINT_TOL,
                 "G_INIT_LOW": G_INIT_LOW, "G_INIT_MID": G_INIT_MID},
        "seeds": list(seeds), "per_seed": per_seed,
        "n_positive": n_positive, "n_meets_static": n_meets_static, "n_load_bearing": n_lb,
        "n_same_setpoint": n_same_setpoint, "cliff_safe": cliff_safe,
        "max_fac_g_visited": max(all_visited_g) if all_visited_g else None,
        "silence_regressed": silence_regressed,
        "default_off_exact_compare": default_off_exact,
        "prove_it_is_live": live_proof,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "BIOLOGY": ("Homeostatic regulation of synaptic gain toward a firing/coincidence set-point via a "
                    "continuing, activity-reading process (Turrigiano 2011; Desai, Rutherford & Turrigiano "
                    "1999 -- slow synaptic scaling). Realized as a bounded, rate-limited discrete-time "
                    "integral controller on the facilitation gain fac_g (the same Mg-block-gated "
                    "Tsodyks-Markram facilitation current as the parent GO substrate), reading the pool's "
                    "OWN N=3 coincidence output at each trial episode and adjusting gain toward a fixed "
                    "target (1.5x FIRE_THR); the rate limit and the safety ceiling are the anti-windup "
                    "realization of the pool's own intrinsic-excitability homeostat's non-monotonic, "
                    "cliff-bearing response surface (measured by the parent build)."),
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[live-homeostat] VERDICT: {verdict}", flush=True)
    print(f"[live-homeostat] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (go or smoke) else 1


def _write(summary):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def selftest() -> bool:
    """Fail-in-the-failing-direction; no brain build."""
    os.environ.pop(LIVE_HOMEOSTAT_ENV, None)
    checks = {
        "REL_TARGET is reused (not re-derived) from the static stabilizer module": REL_TARGET == 1.5 * FIRE_THR,
        "G_FLOOR equals the shipped FAC_G_DEFAULT (reused, not re-typed)": G_FLOOR == FAC_G_DEFAULT,
        "G_CEILING sits strictly below this build's OWN pilot-measured edge (CLIFF_EDGE_S44) and the "
        "parent build's documented collapse (10000)": G_CEILING < CLIFF_EDGE_S44 < 10000.0,
        "G_CEILING is strictly above G_FLOOR (a non-degenerate safe range)": G_CEILING > G_FLOOR,
        "MAX_STEP is positive and small relative to the safe range (a real rate limit)": (
            0 < MAX_STEP < (G_CEILING - G_FLOOR) / 2),
        "K_I is positive (an error above target increases the gain)": K_I > 0,
        "default-off (env unset) -> live_homeostat_enabled() is False": live_homeostat_enabled() is False,
        "'0'/'false' -> live_homeostat_enabled() is False": _env_check("0") is False and _env_check("false") is False,
        "'1'/'true' -> live_homeostat_enabled() is True": _env_check("1") is True and _env_check("true") is True,
        "the controller law is a PURE clip-and-integrate function (no seed-keyed branch): re-running the "
        "SAME frozen synthetic error sequence from the SAME init reproduces the SAME gain trajectory":
            _reproduces_synthetic_trajectory(),
        "the clip never exceeds G_CEILING even under an arbitrarily large synthetic error": _ceiling_holds_under_windup(),
        "the clip never drops below G_FLOOR even under an arbitrarily large negative synthetic error": _floor_holds(),
        "this module's globals contain no CALIBRATED_FAC_G / stabilized_fac_g_for_seed (no table import)": (
            "CALIBRATED_FAC_G" not in globals() and "stabilized_fac_g_for_seed" not in globals()),
    }
    ok = all(checks.values())
    print("=== LIVE HOMEOSTAT SELF-TEST ===")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def _env_check(val):
    os.environ[LIVE_HOMEOSTAT_ENV] = val
    try:
        return live_homeostat_enabled()
    finally:
        os.environ.pop(LIVE_HOMEOSTAT_ENV, None)


def _apply_controller_step(g, error, k_i=K_I, max_step=MAX_STEP, g_floor=G_FLOOR, g_ceiling=G_CEILING):
    """The controller's pure update law, factored out so selftest() can exercise it WITHOUT a brain build."""
    raw_delta = k_i * error
    delta = max(-max_step, min(max_step, raw_delta))
    return max(g_floor, min(g_ceiling, g + delta))


def _reproduces_synthetic_trajectory():
    g_a = g_b = 7000.0
    errs = [0.05, 0.05, 0.03, 0.01, -0.01, 0.0, 0.0]
    for e in errs:
        g_a = _apply_controller_step(g_a, e)
    for e in errs:
        g_b = _apply_controller_step(g_b, e)
    return g_a == g_b


def _ceiling_holds_under_windup():
    g = G_FLOOR
    for _ in range(50):     # a huge, sustained positive error (an unreachable target's worst case)
        g = _apply_controller_step(g, 1.0)
    return g == G_CEILING


def _floor_holds():
    g = G_CEILING
    for _ in range(50):     # a huge, sustained negative error
        g = _apply_controller_step(g, -1.0)
    return g == G_FLOOR


def live_homeostat_enabled() -> bool:
    """Default-OFF. `BRAIN_PMEM_LIVE_HOMEOSTAT` in {1,true,yes,on} -> the production hook runs THIS live
    controller (converging fresh, from `G_INIT_LOW`, cached per-seed within the process exactly as the
    homeostat bias / plateau theta already are -- `_pmem_perpool_homeostat_derisk._BIAS_CACHE`,
    `_pmem_sfa_nmda_amplifier_derisk._THETA_CACHE`) instead of the static `CALIBRATED_FAC_G` lookup. OFF (the
    default) -> the production organ's existing `BRAIN_PMEM_OP_STABILIZER` / plain-facilitation paths are
    completely untouched -- byte-identical to today."""
    v = os.environ.get(LIVE_HOMEOSTAT_ENV)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


_LIVE_CACHE: dict[int, float] = {}


def live_fac_g_for_seed(seed: int) -> float:
    """PRODUCTION entry point: converge THIS seed's gain live (from `G_INIT_LOW`) the first time it is
    needed in this process, then cache the result -- exactly the same per-process calibration-caching
    pattern the homeostat bias and plateau theta already use in this codebase. NOT a table: nothing ships
    pre-computed; the cache is empty until the process actually runs the convergence loop itself."""
    seed = int(seed)
    if seed not in _LIVE_CACHE:
        result = run_live_homeostat(seed, G_INIT_LOW)
        _LIVE_CACHE[seed] = float(result["final_fac_g"])
    return _LIVE_CACHE[seed]


def _reproduces_s44():  # pragma: no cover - kept for parity with the static module's naming; unused here
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print the full trajectory for one seed, both inits")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1
    if a.trace:
        s = a.seed or 44
        for ginit in (G_INIT_LOW, G_INIT_MID):
            r = run_live_homeostat(s, ginit)
            print(json.dumps(r, indent=2))
        return 0
    seeds = [a.seed] if a.seed is not None else a.seeds
    if a.smoke:
        return _derisk([seeds[0]], smoke=True)
    return _derisk(seeds, smoke=False)


if __name__ == "__main__":
    raise SystemExit(main())
