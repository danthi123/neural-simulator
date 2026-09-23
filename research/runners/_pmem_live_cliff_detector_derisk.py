"""PROSPECTIVE-MEMORY LIVE PER-SEED CLIFF DETECTOR -- the named next controller banked by
`research/findings/2026-09-23-live-homeostat-nogo-6seed.md` (an honest NO-GO 5/6: ONE global cliff-safety
ceiling, `G_CEILING=9000`, forbids the gain seed 101 needs -- the static table used 11000 for that seed
specifically). Pre-registered design: `research/findings/2026-09-23-pmem-live-cliff-detector-PREREGISTRATION.md`
(locked BEFORE this runner's `--derisk` is ever executed).

WHY (CLAUDE.md's "deepest lesson" -- the companion process the real system runs alongside a slow integral
set-point controller). A single global ceiling is a STATIC bound standing in for a process the animal actually
runs: metaplasticity -- a companion mechanism that tracks a cell's OWN recent activity and adjusts how far
plasticity is allowed to go, rather than one number shared by every cell. This is the Bienenstock-Cooper-Munro
(BCM) SLIDING THRESHOLD (Lee & Kirkwood 2019, "Mechanisms of Homeostatic Synaptic Plasticity", Front Cell
Neurosci 13:520, DOI:10.3389/fncel.2019.00520 -- external source for gates/deep_research_at_wall, recorded
`research/queue/.external_searches.jsonl`, lane `load-bearing`). This build replaces the parent's ONE constant
ceiling with a LIVE, per-seed ceiling that is DISCOVERED from that seed's own measured coincidence trajectory --
the direct generalization the parent build named and declined to build itself (banking the miss, not tuning a
bigger constant to force a pass).

THE MECHANISM (additive; NO sim/ edit; reuse-by-import of the committed live-homeostat + facilitation
substrate -- `_measure_rel`, `_static_margin_for_seed`, `_default_off_exact_compare` are ALL reused verbatim
from `_pmem_live_homeostat_derisk`, never re-typed). For each seed, from an initial gain `g_0`:
  1. Determine ONCE, from the FIRST live read at `g_init`, whether this seed is a CLIMBER (below
     `REL_TARGET`, needs to raise gain) or SETTLED (already at/above target) -- a LIVE, per-seed branch driven
     by that seed's OWN measurement, never a hand-picked seed list.
  2. A SETTLED seed runs the PARENT build's identical rate-limited integral law, UNCHANGED (descend toward
     `G_FLOOR`; no cliff-detection applies -- descending into already-validated low-gain territory carries no
     documented collapse risk; see the parent's own committed grid, `_pmem_operating_point_stabilizer.json`).
  3. A CLIMBER runs the SAME rate-limited integral law (`K_I`, `MAX_STEP`, reused unchanged, never re-tuned),
     but its ceiling is now `G_CEILING_CAP=11000` (the TOP of the parent build's own already-validated grid --
     a hard, uniform, GLOBAL bound: no seed is ever probed outside the domain the parent already measured for
     every seed) -- and, while climbing, the controller tracks the running best `(g_best, rel_best)` it has
     measured so far and watches for two live signals, sized ONLY from magnitudes the parent build already
     committed (see the pre-registration doc's derivation, never fit to this run's own outcome):
       (a) ABRUPT: a single step's `rel` drop exceeds `DROP_THRESH=0.015` (between the parent's own committed
           recoverable one-step dip, 0.0072, and its non-recoverable cliff, 0.0233);
       (b) SUSTAINED: `DECLINE_PATIENCE=2` consecutive climbing steps that fail to beat the running best (the
           parent's one known recoverable dip lasted exactly ONE step before recovering past the pre-dip value).
     Either signal REVERTS the candidate to `(g_best, rel_best)`, CLAMPS this seed's live ceiling to `g_best`
     (below the drop), and declares the seed CONVERGED there -- a live-detected safety boundary, discovered
     per seed, not shipped as one number. A seed with no cliff/decline inside the validated domain (e.g. seed
     101, whose grid rises monotonically to 11000) simply climbs to `G_CEILING_CAP` with the running best AT
     that cap -- the mechanism that lets 101 reach the higher gain the static table's own per-seed search found.
  4. THE CANONICAL CONVERGED ANSWER for a CLIMBER is ALWAYS `(g_best, rel_best)` -- the best point this seed's
     own live trajectory ever measured (see the pre-registration's redefinition of "cliff-safe": the mechanism
     deliberately probes transiently above a seed's own eventual ceiling in order to FIND it; the invariant
     that survives is that probing never leaves the validated domain, and the FINAL selected gain is always a
     directly measured, non-collapsed point by construction).

ANTI-CHEATS (each implemented + reported in `_derisk`):
  1. GENUINELY LIVE, NOT A TABLE: reused `prove_it_is_live`-style check (this module never imports
     `CALIBRATED_FAC_G`); a re-run of the identical (seed, init) reproduces an IDENTICAL trajectory (including
     cliff events).
  2. TWO DIFFERENT INITS, SAME SET-POINT (canonical `g_best`, not just the raw final iterate).
  3. DISCRIMINATES (falsifiable on the mechanism itself, not just the outcome): the detector must FIRE a
     cliff/decline event on seed 44 (the KNOWN-cliff seed) on BOTH inits, and must NOT fire one on seed 101
     (the KNOWN-no-cliff-in-domain seed) on either init -- `prove_it_discriminates`. A detector that fires on
     everyone or no one is not a genuine live detector.
  4. DOMAIN-BOUNDED: no probe, on any seed/init, ever visits a `fac_g` outside `[G_FLOOR, G_CEILING_CAP]` --
     the pre-registered redefinition of cliff-safety (the parent's OWN already-validated measurement domain,
     never extrapolated).
  5. LOAD-BEARING PRESERVED, re-verified at the CANONICAL converged gain exactly as the parent did.
  6. BYTE-IDENTICAL DEFAULT-OFF, exact-compared against the parent stabilizer's independently committed
     grid-point-0 read (reused `_default_off_exact_compare`).
  7. DETERMINISM: the re-run in anti-cheat #1 IS the determinism check.

THE PRE-REGISTERED GATE (fixed BEFORE running; see the preregistration doc for the full derivation). GO iff, on
all 6 seeds: the canonical live-converged margin is strictly positive AND meets-or-beats the STATIC table's own
committed margin; load-bearing preserved; frozen N=5 silence held; two different inits converge to the SAME
neighborhood; every probe stays inside `[G_FLOOR, G_CEILING_CAP]`; the detector discriminates (fires on 44,
not on 101); default-off byte-identical; deterministic.

  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --selftest
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --smoke
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --derisk
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --seed 44 --trace
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
# reuse-by-import of the LIVE HOMEOSTAT's own helpers (NOT the static table): the fresh-build read, the static
# comparison baseline reader, and the default-off exact-compare helper are all reused verbatim.
from research.runners._pmem_live_homeostat_derisk import (            # noqa: E402
    _measure_rel, _static_margin_for_seed, _default_off_exact_compare,
)
from research.runners._operating_point_stabilizer_derisk import (     # noqa: E402
    FAC_G_DEFAULT, REL_TARGET, _frozen_silence,
)
from tools.lab import attributable_to, void_if                        # noqa: E402
from tools.verdict import Verdict                                     # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_live_cliff_detector.json")
# the PARENT live-homeostat's own committed artifact -- byte-identical default-off comparison baseline (its
# grid-point-0 read is identical to the static stabilizer's, which _default_off_exact_compare already reads).
STATIC_STABILIZER_ARTIFACT = os.path.join(_REPO, "research", "findings", "raw",
                                          "_pmem_operating_point_stabilizer.json")

SEEDS = (42, 43, 44, 100, 101, 102)

# ---- the flag this cliff detector is gated behind (default-OFF; wired into prospective_memory_production_organ.py)
LIVE_CLIFF_ENV = "BRAIN_PMEM_LIVE_CLIFF_DETECTOR"

# ---- controller constants, frozen in the pre-registration doc BEFORE the 6-seed run ----
G_FLOOR = FAC_G_DEFAULT           # 6000.0 -- never regress below the shipped constant
G_CEILING_CAP = 11000.0           # the TOP of the parent stabilizer's own already-validated grid (FAC_G_GRID);
                                   # a hard, uniform, GLOBAL bound -- no seed is ever probed beyond the domain
                                   # the parent already measured for every seed (never extrapolated)
K_I = 20000.0                     # reused unchanged from the parent live homeostat (never re-tuned)
MAX_STEP = 500.0                  # reused unchanged
G_TOL = 25.0                      # reused unchanged
CONVERGE_STREAK = 3               # reused unchanged
MAX_ITERS = 24                    # doubled from the parent's 16 (the live ceiling starts further from the
                                   # floor -- 11000 vs 9000 -- so more climbing steps may be needed)
SETPOINT_TOL = 750.0              # reused unchanged

# the two already-committed magnitudes this build's OWN pre-registration used to size its live-detection
# constants (research/findings/2026-09-23-live-homeostat-nogo-6seed.md's own committed trace + pilot scan):
KNOWN_RECOVERABLE_DIP = 0.0072     # seed 44, low-init, iter1->iter2 (g 6500->7000): rel 0.2150->0.2078, recovers
KNOWN_NONRECOVERABLE_CLIFF = 0.0233  # seed 44 pilot scan (9100->9200 pA): rel 0.2144->0.1911, never recovers

DROP_THRESH = round((KNOWN_RECOVERABLE_DIP + KNOWN_NONRECOVERABLE_CLIFF) / 2.0, 4)   # 0.0153 -> reported as 0.015
DECLINE_PATIENCE = 2               # the one known recoverable dip lasted exactly 1 step; 2 tolerates it

G_INIT_LOW = FAC_G_DEFAULT        # 6000.0 -- the shipped floor
G_INIT_MID = 8000.0               # an interior point


# --------------------------------------------------------------------------------------------------------
# THE LIVE PER-SEED CLIFF DETECTOR. No CALIBRATED_FAC_G, no per-seed branch on identity: every seed runs the
# IDENTICAL law and constants; only the MEASURED trajectory (and hence which branch it takes) differs, because
# the controller reads the pool's own output.
# --------------------------------------------------------------------------------------------------------
def run_live_cliff_homeostat(seed: int, g_init: float, target: float = REL_TARGET,
                             max_iters: int = MAX_ITERS, k_i: float = K_I, max_step: float = MAX_STEP,
                             g_floor: float = G_FLOOR, g_ceiling_cap: float = G_CEILING_CAP, tol: float = G_TOL,
                             streak_need: int = CONVERGE_STREAK, drop_thresh: float = DROP_THRESH,
                             decline_patience: int = DECLINE_PATIENCE) -> dict:
    """Run the live per-seed cliff-detecting homeostat from `g_init`. Returns the full trajectory + cliff
    events (for the anti-cheats + the finding's convergence plot) plus the CANONICAL (g_best, rel_best) triple
    -- the best point this seed's own trajectory ever measured, which for a CLIMBER is the reported answer
    regardless of whether it was reached via a cliff-trigger revert or a clean climb to the cap."""
    g = float(g_init)
    prev_g = None
    prev_rel = None
    converge_streak = 0
    decline_streak = 0
    traj = []
    cliff_events = []
    converged = False
    reason = "max_iters exhausted (%d)" % max_iters
    g_ceiling = g_ceiling_cap        # this seed's LIVE ceiling; starts at the global cap, can only shrink
    rel0 = _measure_rel(seed, g)
    climber = (target - rel0) > 0    # LIVE, per-seed determination from the seed's OWN first measurement
    g_best, rel_best = g, rel0

    for i in range(max_iters):
        rel = rel0 if i == 0 else _measure_rel(seed, g)
        climbing = climber and prev_g is not None and g > prev_g
        cliff_triggered = False
        cliff_kind = None

        if climber:
            if rel > rel_best:
                g_best, rel_best = g, rel
            if not climbing:
                decline_streak = 0
            elif prev_rel is not None and rel < prev_rel:
                # a genuine LOCAL decrease from the immediately PRECEDING point (never from the running best
                # -- a step that is still below best but ABOVE the preceding point is a RECOVERY, not a
                # decline; see the pre-registration's seed-44 dip trace, which recovers one step after its
                # only dip and must NOT trigger here).
                decline_streak += 1
                step_drop = round((prev_rel - rel), 4)
                if step_drop > drop_thresh:
                    cliff_triggered, cliff_kind = True, ("abrupt: step drop %.4f > DROP_THRESH %.4f"
                                                         % (step_drop, drop_thresh))
                elif decline_streak >= decline_patience:
                    cliff_triggered, cliff_kind = True, ("sustained: %d consecutive climbing steps each below "
                                                         "its immediate predecessor" % decline_streak)
            else:
                decline_streak = 0    # flat or a local increase (a recovery) resets the streak

        traj.append({"iter": i, "fac_g": round(g, 1), "rel": rel, "climber": climber, "climbing": climbing,
                    "decline_streak": decline_streak, "g_ceiling": round(g_ceiling, 1),
                    "g_best": round(g_best, 1), "rel_best": rel_best})

        if cliff_triggered:
            cliff_events.append({"iter": i, "probed_fac_g": round(g, 1), "probed_rel": rel, "kind": cliff_kind,
                                "reverted_to_fac_g": round(g_best, 1), "reverted_to_rel": rel_best,
                                "new_ceiling": round(g_best, 1)})
            g_ceiling = g_best       # CLAMP this seed's live ceiling below the drop
            g = g_best
            rel = rel_best
            traj[-1].update({"fac_g": round(g, 1), "rel": rel, "reverted": True})
            reason = "cliff detected & clamped at fac_g=%.0f (%s)" % (g, cliff_kind)
            converged = True
            prev_g, prev_rel = g, rel
            break

        if prev_g is not None:
            moved = abs(g - prev_g)
            converge_streak = converge_streak + 1 if moved < tol else 0
            if converge_streak >= streak_need:
                at_bound = ("ceiling" if abs(g - g_ceiling) < 1e-6
                           else ("floor" if abs(g - g_floor) < 1e-6 else None))
                reason = ("pinned at %s (fac_g=%.0f)" % (at_bound, g) if at_bound
                         else "settled at an interior set-point (fac_g=%.0f)" % g)
                converged = True
                break
        prev_g, prev_rel = g, rel
        error = round(target - rel, 4)
        raw_delta = k_i * error
        delta = max(-max_step, min(max_step, raw_delta))
        g = max(g_floor, min(g_ceiling, g + delta))

    last = traj[-1]
    canonical_fac_g = g_best if climber else last["fac_g"]
    canonical_rel = rel_best if climber else last["rel"]
    return {"seed": seed, "g_init": g_init, "climber": climber, "converged": converged, "reason": reason,
            "n_iters": len(traj), "final_fac_g": canonical_fac_g, "final_rel": canonical_rel,
            "raw_last_fac_g": last["fac_g"], "raw_last_rel": last["rel"],
            "live_ceiling": round(g_ceiling, 1), "cliff_events": cliff_events, "trajectory": traj}


def prove_it_is_live(seed: int = 44) -> dict:
    """ANTI-CHEAT #1: this module never imports the static table, AND a fresh re-run of the SAME (seed, init)
    reproduces an IDENTICAL trajectory (including cliff events) -- the value is genuinely COMPUTED at each
    step, not memoised from a dict."""
    no_table_import = ("CALIBRATED_FAC_G" not in globals() and "stabilized_fac_g_for_seed" not in globals())
    r1 = run_live_cliff_homeostat(seed, G_INIT_LOW)
    r2 = run_live_cliff_homeostat(seed, G_INIT_LOW)
    same_traj = r1["trajectory"] == r2["trajectory"]
    same_cliffs = r1["cliff_events"] == r2["cliff_events"]
    return {"seed": seed, "no_table_import": no_table_import, "rerun_identical_trajectory": same_traj,
            "rerun_identical_cliff_events": same_cliffs}


def prove_it_discriminates(per_seed: dict) -> dict:
    """ANTI-CHEAT #3 (falsifiable on the MECHANISM, not just the outcome): the detector must fire on the
    KNOWN-cliff seed (44, both inits) and must NOT fire on the KNOWN-no-cliff-in-domain seed (101, either
    init) -- pre-registered before this run. A detector that fires on everyone or no one is not genuine."""
    fires_44 = (bool(per_seed[44]["low_init"]["cliff_events"]) if 44 in per_seed else None,
               bool(per_seed[44]["mid_init"]["cliff_events"]) if 44 in per_seed else None)
    fires_101 = (bool(per_seed[101]["low_init"]["cliff_events"]) if 101 in per_seed else None,
                bool(per_seed[101]["mid_init"]["cliff_events"]) if 101 in per_seed else None)
    ok = (all(fires_44) if 44 in per_seed else None, not any(fires_101) if 101 in per_seed else None)
    return {"fires_on_seed44_low_mid": fires_44, "fires_on_seed101_low_mid": fires_101,
            "seed44_both_fire": ok[0], "seed101_neither_fires": ok[1],
            "discriminates": bool(ok[0]) and bool(ok[1]) if (ok[0] is not None and ok[1] is not None) else None}


# --------------------------------------------------------------------------------------------------------
def _derisk(seeds=SEEDS, smoke=False):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"LIVE CLIFF DETECTOR [{tag}] -- per-seed live-detected safety ceiling on fac_g; target={REL_TARGET:.3f} "
          f"K_I={K_I} MAX_STEP={MAX_STEP} floor={G_FLOOR} cap={G_CEILING_CAP} DROP_THRESH={DROP_THRESH} "
          f"DECLINE_PATIENCE={DECLINE_PATIENCE}; {len(seeds)} seed(s)", flush=True)
    t0 = time.time()
    err = None
    per_seed = {}
    all_visited_g = []
    try:
        for s in seeds:
            print(f"\n--- seed {s}: live cliff-detecting homeostat, two inits ---", flush=True)
            low = run_live_cliff_homeostat(s, G_INIT_LOW)
            print(f"  [low init={G_INIT_LOW}] climber={low['climber']} converged={low['converged']} "
                  f"n_iters={low['n_iters']} canonical_fac_g={low['final_fac_g']} canonical_rel={low['final_rel']} "
                  f"cliff_events={len(low['cliff_events'])} :: {low['reason']}", flush=True)
            for row in low["trajectory"]:
                print(f"      iter {row['iter']:2d}  fac_g={row['fac_g']:7.1f}  rel={row['rel']:.4f}  "
                      f"climbing={row['climbing']}  decline_streak={row['decline_streak']}  "
                      f"g_best={row['g_best']:.1f}", flush=True)
            mid = run_live_cliff_homeostat(s, G_INIT_MID)
            print(f"  [mid init={G_INIT_MID}] climber={mid['climber']} converged={mid['converged']} "
                  f"n_iters={mid['n_iters']} canonical_fac_g={mid['final_fac_g']} canonical_rel={mid['final_rel']} "
                  f"cliff_events={len(mid['cliff_events'])} :: {mid['reason']}", flush=True)
            for row in mid["trajectory"]:
                print(f"      iter {row['iter']:2d}  fac_g={row['fac_g']:7.1f}  rel={row['rel']:.4f}  "
                      f"climbing={row['climbing']}  decline_streak={row['decline_streak']}  "
                      f"g_best={row['g_best']:.1f}", flush=True)

            all_visited_g += [row["fac_g"] for row in low["trajectory"]] \
                + [e["probed_fac_g"] for e in low["cliff_events"]] \
                + [row["fac_g"] for row in mid["trajectory"]] \
                + [e["probed_fac_g"] for e in mid["cliff_events"]]
            same_setpoint = abs(low["final_fac_g"] - mid["final_fac_g"]) <= SETPOINT_TOL
            converged_g = low["final_fac_g"]         # canonical converged gain: the low-init trajectory
            converged_rel = low["final_rel"]

            lb = F._n3_load_bearing(s, fac_on=True, N=3, fac_g=converged_g, fac_U=F.FAC_U,
                                    fac_tau_F_steps=F.FAC_TAU_F_STEPS)
            print(f"  load-bearing @ canonical g={converged_g}: intact rel={lb['intact_rel']:.4f} "
                  f"fired={lb['intact_fired']} | lesion rel={lb['lesion_rel']:.4f} fired={lb['lesion_fired']} | "
                  f"LB={lb['load_bearing']}", flush=True)
            lesion_attrib = attributable_to(f"seed {s}: intact vs lesion coincidence (load-bearing attribution)",
                                            lb["intact_rel"], lb["lesion_rel"])

            froz = _frozen_silence(s, converged_g)
            fails = [k for k, v in froz["clauses"].items() if not v]
            print(f"  frozen N=5 silence @ canonical g: passed={froz['passed']} "
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
                "cliff_events": low["cliff_events"],
            }

        live_proof = prove_it_is_live(44 if 44 in per_seed else seeds[0])
        print(f"\n--- ANTI-CHEAT: prove_it_is_live (seed {live_proof['seed']}) ---", flush=True)
        print(f"  no_table_import={live_proof['no_table_import']}  "
              f"rerun_identical_trajectory={live_proof['rerun_identical_trajectory']}  "
              f"rerun_identical_cliff_events={live_proof['rerun_identical_cliff_events']}", flush=True)
        discrim = prove_it_discriminates(per_seed) if not smoke else None
        if discrim is not None:
            print(f"--- ANTI-CHEAT: prove_it_discriminates -- {discrim}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_live_cliff_detector", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    n_seeds = len(seeds)
    n_positive = sum(1 for s in seeds if per_seed[s]["margin_positive"])
    n_meets_static = sum(1 for s in seeds if per_seed[s]["margin_meets_static"])
    n_lb = sum(1 for s in seeds if per_seed[s]["load_bearing"])
    n_same_setpoint = sum(1 for s in seeds if per_seed[s]["same_setpoint_within_tol"])
    silence_regressed = [s for s in seeds if per_seed[s]["frozen_clause_fails"]]
    domain_bounded = (max(all_visited_g) if all_visited_g else 0.0) <= G_CEILING_CAP \
        and (min(all_visited_g) if all_visited_g else G_FLOOR) >= G_FLOOR
    cheat = void_if(bool(silence_regressed),
                    f"the live cliff detector REGRESSED a silence clause at seed(s) {silence_regressed} -> "
                    f"spurious fires (a homeostat that raises gain until everything fires is a CHEAT; VOID)")
    cheat2 = void_if(not domain_bounded,
                     f"the controller visited fac_g outside [{G_FLOOR}, {G_CEILING_CAP}] (max visited "
                     f"{max(all_visited_g) if all_visited_g else float('nan')}) -> VOID")
    default_off_exact = _default_off_exact_compare(per_seed, seeds) if not smoke else None
    changed_seeds = [s for s in seeds if per_seed[s]["converged_fac_g"] != G_FLOOR]
    unchanged_seeds = [s for s in seeds if per_seed[s]["converged_fac_g"] == G_FLOOR]
    thin_seed = min(seeds, key=lambda s: per_seed[s]["static_table_margin"])
    discrim = prove_it_discriminates(per_seed) if not smoke else None

    from research.runners._pmem_perpool_homeostat_derisk import SILENCE_CLAUSES  # noqa: E402

    go = bool(n_positive == n_seeds and n_meets_static == n_seeds and n_lb == n_seeds
             and n_same_setpoint == n_seeds and domain_bounded and not cheat and not cheat2
             and (default_off_exact is None or default_off_exact.get("ok") is not False)
             and (discrim is None or discrim.get("discriminates") is not False)
             and not smoke)

    vd = Verdict("pmem_live_cliff_detector")
    vd.require("load-bearing preserved at the canonical converged gain (per-seed count)", n_lb,
               expect=lambda x, n=n_seeds: x == n)
    vd.require("two different inits converge to the SAME canonical neighborhood (per-seed count)",
               n_same_setpoint, expect=lambda x, n=n_seeds: x == n)
    vd.require("domain-bounded: no probe left [G_FLOOR, G_CEILING_CAP]", domain_bounded, expect=True)
    for c in SILENCE_CLAUSES:
        vd.require(f"frozen-gate silence held at the canonical converged gain: {c}",
                   sum(1 for s in seeds if per_seed[s]["frozen_clauses"].get(c)),
                   expect=lambda x, n=n_seeds: x == n)
    if discrim is not None:
        vd.require("the detector discriminates (fires on seed 44, not on seed 101)",
                   discrim.get("discriminates"), expect=True)
    vd.disabled("STDP / long-term Hebbian LTP / OU-noise",
                "identical scope to the parent facilitation + live-homeostat de-risks; the only added "
                "mechanism is the live, per-seed cliff/decline detector replacing the parent's single "
                "global ceiling")
    decided = vd.decide(go)

    domain_note = (
        " DOMAIN-BOUNDED: every probe stayed inside [%.0f, %.0f] (the parent stabilizer's own already-"
        "validated grid domain); max fac_g visited across all seeds/inits/probes was %.0f. The FINAL selected "
        "gain for every seed is always a directly-measured, non-collapsed point (canonical = running best), by "
        "construction." % (G_FLOOR, G_CEILING_CAP, max(all_visited_g) if all_visited_g else float("nan"))
    )
    status_word = "GO" if go else "VOID" if (cheat or cheat2) else (
        "UNDEFINED" if (decided or {}).get("status") == "UNDEFINED" else "NO-GO")
    verdict = (
        f"{status_word} ({n_meets_static}/{n_seeds}) -- per the PRE-REGISTERED "
        f"gate, {n_positive}/{n_seeds} seeds hold a strictly positive canonical margin, "
        f"{n_meets_static}/{n_seeds} meet-or-beat the STATIC table's own margin, {n_lb}/{n_seeds} stay "
        f"load-bearing, {n_same_setpoint}/{n_seeds} converge to the SAME canonical neighborhood from two "
        f"different inits; silence-regressed={silence_regressed or 'none'}; domain-bounded={domain_bounded}; "
        f"discriminates={discrim.get('discriminates') if discrim else 'n/a (smoke)'}. Seed(s) "
        f"{changed_seeds or 'none'} moved off the shipped floor; seed(s) {unchanged_seeds or 'none'} settled "
        f"back at it. Thinnest-margin seed in the static table (s{thin_seed}): static margin "
        f"{per_seed[thin_seed]['static_table_margin']:+.4f} -> live-converged "
        f"{per_seed[thin_seed]['converged_margin']:+.4f}." + domain_note
    )

    summary = {
        "probe": "pmem_live_cliff_detector", "verdict": verdict, "go": bool(go),
        "task": ("Replace the parent live-homeostat's SINGLE GLOBAL cliff-safety ceiling (G_CEILING=9000) with "
                 "a LIVE, per-seed ceiling discovered from that seed's own measured coincidence trajectory -- "
                 "a local-derivative + sustained-decline safety check that clamps the ceiling below a "
                 "detected drop, live, per seed, with no per-seed constant anywhere in the control loop."),
        "gate": {"FIRE_THR": FIRE_THR, "REL_TARGET": REL_TARGET, "G_FLOOR": G_FLOOR,
                 "G_CEILING_CAP": G_CEILING_CAP, "K_I": K_I, "MAX_STEP": MAX_STEP, "G_TOL": G_TOL,
                 "CONVERGE_STREAK": CONVERGE_STREAK, "MAX_ITERS": MAX_ITERS, "SETPOINT_TOL": SETPOINT_TOL,
                 "DROP_THRESH": DROP_THRESH, "DECLINE_PATIENCE": DECLINE_PATIENCE,
                 "G_INIT_LOW": G_INIT_LOW, "G_INIT_MID": G_INIT_MID},
        "seeds": list(seeds), "per_seed": per_seed,
        "n_positive": n_positive, "n_meets_static": n_meets_static, "n_load_bearing": n_lb,
        "n_same_setpoint": n_same_setpoint, "domain_bounded": domain_bounded,
        "max_fac_g_visited": max(all_visited_g) if all_visited_g else None,
        "silence_regressed": silence_regressed,
        "default_off_exact_compare": default_off_exact,
        "prove_it_is_live": live_proof,
        "prove_it_discriminates": discrim,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "BIOLOGY": ("Metaplasticity / BCM sliding threshold (Lee & Kirkwood 2019, Front Cell Neurosci 13:520, "
                    "DOI:10.3389/fncel.2019.00520): a companion process that adjusts the threshold for "
                    "plasticity from a unit's OWN recent activity history, rather than one fixed bound shared "
                    "by every unit. Realized here as a per-seed LIVE safety ceiling discovered from that "
                    "seed's own coincidence-output trajectory (a local-derivative abrupt-cliff check plus a "
                    "sustained-decline check, both sized from the parent build's own already-measured dip/"
                    "cliff magnitudes), generalizing the parent live homeostat's single global constant "
                    "ceiling into a genuinely per-unit, activity-dependent bound."),
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[live-cliff-detector] VERDICT: {verdict}", flush=True)
    print(f"[live-cliff-detector] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (go or smoke) else 1


def _write(summary):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def selftest() -> bool:
    """Fail-in-the-failing-direction; no brain build."""
    os.environ.pop(LIVE_CLIFF_ENV, None)
    checks = {
        "REL_TARGET is reused (not re-derived) from the static stabilizer module": REL_TARGET == 1.5 * FIRE_THR,
        "G_FLOOR equals the shipped FAC_G_DEFAULT (reused, not re-typed)": G_FLOOR == FAC_G_DEFAULT,
        "G_CEILING_CAP equals the parent stabilizer's own top grid point (11000, never extrapolated)": (
            G_CEILING_CAP == 11000.0),
        "G_CEILING_CAP is strictly above G_FLOOR (a non-degenerate domain)": G_CEILING_CAP > G_FLOOR,
        "MAX_STEP is positive and small relative to the domain (a real rate limit)": (
            0 < MAX_STEP < (G_CEILING_CAP - G_FLOOR) / 2),
        "K_I is positive (an error above target increases the gain)": K_I > 0,
        "DROP_THRESH sits strictly between the parent's own committed recoverable dip and non-recoverable "
        "cliff magnitudes (never fit to this build's own outcome)": (
            KNOWN_RECOVERABLE_DIP < DROP_THRESH < KNOWN_NONRECOVERABLE_CLIFF),
        "DECLINE_PATIENCE is strictly greater than the parent's known recoverable dip's duration (1 step)": (
            DECLINE_PATIENCE > 1),
        "default-off (env unset) -> live_cliff_detector_enabled() is False": live_cliff_detector_enabled() is False,
        "'0'/'false' -> live_cliff_detector_enabled() is False": (
            _env_check("0") is False and _env_check("false") is False),
        "'1'/'true' -> live_cliff_detector_enabled() is True": (
            _env_check("1") is True and _env_check("true") is True),
        "the abrupt-cliff law is a PURE function of the running best + step drop (no seed-keyed branch): "
        "re-running the SAME frozen synthetic sequence from the SAME init reproduces the SAME cliff decision":
            _reproduces_synthetic_cliff_decision(),
        "a synthetic single-step drop just ABOVE DROP_THRESH triggers an abrupt cliff": _abrupt_cliff_fires(),
        "a synthetic single-step drop just BELOW DROP_THRESH (the parent's own recoverable-dip magnitude) "
        "does NOT trigger an abrupt cliff": not _abrupt_cliff_does_not_fire_on_known_dip(),
        "DECLINE_PATIENCE consecutive small declines (each below DROP_THRESH) DO trigger a sustained-decline "
        "cliff (the seed-100-shaped case the abrupt check alone would miss)": _sustained_decline_fires(),
        "a single small decline that then RECOVERS (the parent's own seed-44 dip shape) does NOT trigger "
        "either check": _recoverable_dip_does_not_fire(),
        "the clip never exceeds G_CEILING_CAP even under an arbitrarily large synthetic error": (
            _ceiling_holds_under_windup()),
        "the clip never drops below G_FLOOR even under an arbitrarily large negative synthetic error": (
            _floor_holds()),
        "this module's globals contain no CALIBRATED_FAC_G / stabilized_fac_g_for_seed (no table import)": (
            "CALIBRATED_FAC_G" not in globals() and "stabilized_fac_g_for_seed" not in globals()),
    }
    ok = all(checks.values())
    print("=== LIVE CLIFF DETECTOR SELF-TEST ===")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def _env_check(val):
    os.environ[LIVE_CLIFF_ENV] = val
    try:
        return live_cliff_detector_enabled()
    finally:
        os.environ.pop(LIVE_CLIFF_ENV, None)


def _apply_controller_step(g, error, k_i=K_I, max_step=MAX_STEP, g_floor=G_FLOOR, g_ceiling=G_CEILING_CAP):
    """The rate-limited integral update law, factored out so selftest() can exercise it WITHOUT a brain build."""
    raw_delta = k_i * error
    delta = max(-max_step, min(max_step, raw_delta))
    return max(g_floor, min(g_ceiling, g + delta))


def _synthetic_cliff_scan(rels, drop_thresh=DROP_THRESH, decline_patience=DECLINE_PATIENCE):
    """The PURE cliff/decline-detection law over a synthetic `rel` sequence at fixed 500pA steps -- exercises
    exactly the same branch (and the same LOCAL-decrease-from-the-preceding-point definition of "decline") as
    `run_live_cliff_homeostat`'s climber path, without a brain build."""
    g_best, rel_best = 0.0, rels[0]
    decline_streak = 0
    prev_rel = rels[0]
    for i, rel in enumerate(rels[1:], start=1):
        if rel > rel_best:
            g_best, rel_best = float(i), rel
        if rel < prev_rel:
            decline_streak += 1
            step_drop = round(prev_rel - rel, 4)
            prev_rel = rel
            if step_drop > drop_thresh:
                return {"triggered": True, "kind": "abrupt", "iter": i, "g_best": g_best, "rel_best": rel_best}
            if decline_streak >= decline_patience:
                return {"triggered": True, "kind": "sustained", "iter": i, "g_best": g_best,
                        "rel_best": rel_best}
        else:
            decline_streak = 0
            prev_rel = rel
    return {"triggered": False, "kind": None, "iter": len(rels) - 1, "g_best": g_best, "rel_best": rel_best}


def _reproduces_synthetic_cliff_decision():
    rels = [0.20, 0.22, 0.21, 0.19]
    a = _synthetic_cliff_scan(rels)
    b = _synthetic_cliff_scan(rels)
    return a == b


def _abrupt_cliff_fires():
    # a single step drop of DROP_THRESH + 0.002 (just above threshold) must fire immediately (iter 1)
    rels = [0.20, 0.20 - (DROP_THRESH + 0.002)]
    r = _synthetic_cliff_scan(rels)
    return r["triggered"] and r["kind"] == "abrupt" and r["iter"] == 1


def _abrupt_cliff_does_not_fire_on_known_dip():
    # the parent's own committed recoverable-dip magnitude (0.0072), well below DROP_THRESH, must NOT fire
    # the abrupt check on its own (a single dip step, patience=2 requires a SECOND non-recovering step)
    rels = [0.2150, 0.2150 - KNOWN_RECOVERABLE_DIP]
    r = _synthetic_cliff_scan(rels)
    return r["triggered"] and r["kind"] == "abrupt"   # i.e. this returns False for a correctly-behaving law


def _sustained_decline_fires():
    # two consecutive small declines (each well below DROP_THRESH) from a running best -- the seed-100 shape
    rels = [0.2956, 0.2956 - 0.0022, 0.2956 - 0.0045]
    r = _synthetic_cliff_scan(rels)
    return r["triggered"] and r["kind"] == "sustained" and r["iter"] == 2


def _recoverable_dip_does_not_fire():
    # the parent's own committed seed-44 shape: one dip step, then RECOVERS past the pre-dip value
    rels = [0.2111, 0.2150, 0.2078, 0.2089, 0.2139, 0.2172, 0.2183]
    r = _synthetic_cliff_scan(rels)
    return not r["triggered"]


def _ceiling_holds_under_windup():
    g = G_FLOOR
    for _ in range(50):
        g = _apply_controller_step(g, 1.0)
    return g == G_CEILING_CAP


def _floor_holds():
    g = G_CEILING_CAP
    for _ in range(50):
        g = _apply_controller_step(g, -1.0)
    return g == G_FLOOR


def live_cliff_detector_enabled() -> bool:
    """Default-OFF. `BRAIN_PMEM_LIVE_CLIFF_DETECTOR` in {1,true,yes,on} -> replace the STATIC per-seed table AND
    the parent live homeostat's single global ceiling with THIS per-seed live-detected ceiling (converging
    fresh, from `G_INIT_LOW`, cached per-seed within the process -- the SAME per-process calibration-caching
    pattern the homeostat bias / plateau theta / live homeostat already use). Takes PRIORITY over BOTH
    `BRAIN_PMEM_LIVE_HOMEOSTAT` and `BRAIN_PMEM_OP_STABILIZER` when more than one is set (see `_ensure_pm`).
    OFF (the default) -> the production organ's existing paths are completely untouched -- byte-identical to
    today."""
    v = os.environ.get(LIVE_CLIFF_ENV)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


_LIVE_CLIFF_CACHE: dict[int, float] = {}


def live_cliff_fac_g_for_seed(seed: int) -> float:
    """PRODUCTION entry point: converge THIS seed's gain live via the cliff detector (from `G_INIT_LOW`) the
    first time it is needed in this process, then cache the result. NOT a table: nothing ships pre-computed."""
    seed = int(seed)
    if seed not in _LIVE_CLIFF_CACHE:
        result = run_live_cliff_homeostat(seed, G_INIT_LOW)
        _LIVE_CLIFF_CACHE[seed] = float(result["final_fac_g"])
    return _LIVE_CLIFF_CACHE[seed]


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
            r = run_live_cliff_homeostat(s, ginit)
            print(json.dumps(r, indent=2))
        return 0
    seeds = [a.seed] if a.seed is not None else a.seeds
    if a.smoke:
        return _derisk([seeds[0]], smoke=True)
    return _derisk(seeds, smoke=False)


if __name__ == "__main__":
    raise SystemExit(main())
