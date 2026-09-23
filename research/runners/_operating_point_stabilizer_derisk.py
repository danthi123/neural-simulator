"""PROSPECTIVE-MEMORY OPERATING-POINT STABILIZER -- hardening the THIN s44 margin with a genuine per-seed
homeostatic set-point search (roadmap DEEPER #4; research/lbf-operating-point-stabilizer, 2026-09-23).

WHY (CLAUDE.md's deepest lesson). `2026-09-22-prospective-memory-facilitation-load-bearing-6seed.md` made
prospective-memory load-bearing 6/6 via short-term (Tsodyks-Markram) facilitation of the maintained-intention
drive, but at ONE global constant `FAC_G=6000` -- and that finding's own honest residual says s44's margin above
the crossing is "MODEST ... not the wide plateau a prior draft claimed". The fixed FAC_G is exactly the kind of
CONSTANT CLAUDE.md flags: biology does not hold synaptic gain fixed across individuals -- it runs a slow
HOMEOSTATIC SET-POINT process (Turrigiano 2011, "Too many cooks? Intrinsic and synaptic homeostatic mechanisms";
Desai, Rutherford & Turrigiano 1999) that regulates each cell's/pathway's own gain toward a population operating
point. This runner replaces the single constant with that companion process: a per-seed CALIBRATED facilitation
gain that targets a principled set-point (REL_TARGET, 1.5x FIRE_THR -- "50% headroom over the release threshold"),
using the SAME algorithm and target for every seed (never a seed-specific hand pick), while a fragile seed like
s44 converges to a DIFFERENT gain than an already-robust seed precisely because it reads its OWN output.

THE DISCOVERY THIS RUNNER MADE (an honest, load-bearing finding in its own right; read it before assuming a
knob is safe to turn). The naive expectation -- "more fac_g -> a proportionally bigger coincidence read" -- is
FALSE once the substrate's OWN pre-existing per-pool intrinsic-excitability homeostat and NMDA-plateau
calibration (`_pmem_perpool_homeostat_derisk` / `_pmem_sfa_nmda_amplifier_derisk`, both ALREADY live under
facilitation, `fac_calib=True`) are left to recalibrate SELF-CONSISTENTLY against the new gain (i.e., on a FRESH
build, which is the only correct way to evaluate a candidate fac_g -- a stale-bias in-place mutation UNDERSTATES
the true safety response and would have reported this substrate as safe up to fac_g=22000; a fresh, self-
consistent build shows it is NOT). At seed 44, fresh builds across a probe grid gave `rel` = 0.2111 (g=6000),
0.2078 (7000, a DIP), 0.2139 (8000), 0.2183 (9000, the safe maximum), 0.1506 (10000, a CLIFF -- the pool's own
homeostat hyperpolarizes it enough in response to the higher single-input drive that the COINCIDENCE collapses
too), 0.1589 (11000, still collapsed). This is a genuine bifurcation in a recurrent NMDA-accumulator + spike-
frequency-adaptation + plateau system, not measurement noise (every read is exactly reproducible at a fixed seed
+ fac_g). CONSEQUENCE FOR THE DESIGN: a homeostat here cannot be a blind proportional/gradient controller (it
would walk straight into the cliff); it must be a bounded, PRE-REGISTERED candidate search that only ever moves
to a candidate at least as good as where it started (a floor-guarded arg-max), which is what is built below. This
finding is banked as the honest characterization of WHY a naive "just turn the gain up" lever fails here.

THE MECHANISM (additive; NO sim/ edit; reuse-by-import of the already-committed GO facilitation substrate
`_pmem_facilitation_derisk.FacilitatedHebbianProspectiveMemory` / its `_n3_arm` / `_n3_load_bearing` / the frozen
N=5 silence gate via `_pmem_intention_latch_derisk.run_seed`). For each seed:
  1. Evaluate the N=3 production-protocol intact coincidence read `rel` on a FIXED, pre-registered grid of fresh
     builds, FAC_G_GRID = (6000, 7000, 8000, 9000, 10000, 11000) pA -- bounded specifically at the discovered
     cliff (10000-11000), not swept further; grid[0]=6000 IS today's shipped constant, so it doubles as the
     "un-stabilized" (facilitation-only) baseline read at no extra cost.
  2. IF the baseline (g=6000) already clears REL_TARGET=0.30 (1.5x FIRE_THR): keep g=6000 -- a homeostat that
     is already at its set-point does not move (this is the ANTI-CHEAT for "not tuning to force a margin": it
     only acts on seeds that need it).
  3. ELSE: search the grid for the smallest g that reaches REL_TARGET; if none reach it (the target is
     unreachable inside the safe grid -- true for every seed below target here), take the floor-guarded arg-max
     (the best-performing candidate that is NOT WORSE than the baseline -- never regress).
This is IDENTICAL machinery/knobs for every seed (the grid, the target, the selection rule); only the OUTCOME
differs, because it reads each seed's own coincidence output -- see `prove_it_adapts()` for the direct proof
this is target-directed, not a per-seed constant in disguise.

ANTI-CHEATS
  1. PROVES IT ADAPTS (not a shifted constant): `prove_it_adapts()` runs the IDENTICAL seed-44 grid data at TWO
     different REL_TARGET values (0.30 vs 0.21) and shows the selection rule returns a DIFFERENT calibrated
     fac_g (9000 vs 6000) -- the outcome moves with the requested set-point, which a fixed number cannot do.
  2. LOAD-BEARING PRESERVED: the lesion arm (BRAIN_PMEM_LESION collapses the held assembly, h_a~0) still reads
     `rel`~0 at the CHOSEN fac_g on every seed -- structurally guaranteed by the Mg-block-gated facilitation
     current (`I_fac` requires h_a>0 AND F>0; the lesion zeroes h_a), and empirically reverified below.
  3. SILENCE HELD: the frozen N=5 gate (both A/B pools, all clauses, `_pmem_intention_latch_derisk.run_seed`) is
     re-run at the CHOSEN per-seed gain -- a homeostat that fixed the coincidence by breaking a silence clause
     is void (`SILENCE_CLAUSES`, imported, never re-typed).
  4. DEFAULT-OFF BYTE-IDENTICAL: the production hook (`prospective_memory_production_organ.pmem_op_stabilizer_enabled`)
     only changes the `fac_g` KWARG already accepted by the committed facilitation class; OFF, it passes the
     SAME `FAC_G_DEFAULT=6000.0` the shipped organ already defaults to -- an EXACT compare against grid[0]'s own
     reads (not an inference) proves this, plus `selftest()`.
  5. NO METRIC-TUNING: REL_TARGET=0.30 is fixed BEFORE the 6-seed run (1.5x FIRE_THR, a round biologically-
     neutral headroom -- not derived from, or swept to fit, any one seed's outcome); FAC_G_GRID is bounded at the
     empirically-discovered cliff, not chosen to make a particular seed pass.

ATTRIBUTION (tools.lab.attributable_to; CLAUDE.md's "the proxy dominates" caution, run in the other direction).
The pre-registered GO-GATE compares the stabilized read against the DIAGNOSIS baseline (`op_s*.json`, no
facilitation at all) -- against THAT baseline, most of seeds 42/43/102's enlargement is owed to the ALREADY-
COMMITTED facilitation fix, not to this session's build (this runner leaves their gain at the shipped constant).
This runner's OWN marginal contribution is isolated and reported honestly per seed (`stabilizer_own_share`): it
is the seeds BELOW the target (44, 100, 101) where this build adds a genuine, further, measured enlargement on
top of facilitation's own.

  SIM_BACKEND=numpy .venv/bin/python -m research.runners._operating_point_stabilizer_derisk --selftest
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._operating_point_stabilizer_derisk --derisk
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
import research.runners._pmem_intention_latch_derisk as base         # noqa: E402  (the frozen N=5 silence gate)
from research.runners._pmem_intention_latch_derisk import FIRE_THR   # noqa: E402  (imported, never re-typed)
from research.runners._pmem_perpool_homeostat_derisk import SILENCE_CLAUSES  # noqa: E402
from tools.lab import attributable_to, void_if                       # noqa: E402
from tools.verdict import Verdict                                    # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_operating_point_stabilizer.json")
DIAG_DIR = os.path.join(_REPO, "research", "findings", "raw", "_lbf_borderline")
# the PRIOR session's committed facilitation-only artifact -- an INDEPENDENT (different process, different day)
# exact-compare target for the default-OFF byte-identical claim (stronger than re-deriving the same computation
# twice in this process, which only proves idempotency).
PRIOR_FACILITATION_ARTIFACT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_facilitation.json")

SEEDS = (42, 43, 44, 100, 101, 102)
FAC_G_DEFAULT = F.FAC_G                              # 6000.0 -- today's shipped constant; grid[0] below
FAC_G_GRID = (6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0)   # pre-registered; bounded at the found cliff
REL_TARGET = 1.5 * FIRE_THR                          # 0.30 -- 50% headroom over the release threshold, fixed
                                                      # BEFORE the 6-seed run (not derived from any one seed)

# the flag this stabilizer is gated behind (default-OFF; wired into prospective_memory_production_organ.py)
STABILIZER_ENV = "BRAIN_PMEM_OP_STABILIZER"

# the per-seed calibration this runner's --derisk produces (see the committed finding for the run that made
# this table). Any seed NOT in the table falls back to FAC_G_DEFAULT (an honest, explicit scope limit -- this
# is a STATIC table from a 6-seed calibration run, not a live/continuous in-session homeostat; see "Next").
CALIBRATED_FAC_G = {42: 6000.0, 43: 6000.0, 44: 9000.0, 100: 9000.0, 101: 11000.0, 102: 6000.0}


def stabilizer_enabled() -> bool:
    """Default-OFF. BRAIN_PMEM_OP_STABILIZER in {1,true,yes,on} -> the production hook looks up this seed's
    calibrated fac_g (CALIBRATED_FAC_G, falling back to FAC_G_DEFAULT) instead of the shipped constant. OFF (the
    default) -> FAC_G_DEFAULT everywhere -- byte-identical to today's facilitation-only path."""
    v = os.environ.get(STABILIZER_ENV)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def stabilized_fac_g_for_seed(seed: int, fallback: float = FAC_G_DEFAULT) -> float:
    return float(CALIBRATED_FAC_G.get(int(seed), fallback))


# --------------------------------------------------------------------------------------------------------
# THE CALIBRATION -- a bounded, floor-guarded, target-seeking grid search (the discrete realization of a
# homeostatic set-point regulator; see the module docstring for why a blind proportional controller is unsafe
# on this substrate). Every fresh build reuses F._n3_arm, which is SELF-CONSISTENT (bias + plateau + facilitation
# all calibrate against the SAME candidate fac_g at construction, exactly as the shipped organ does).
# --------------------------------------------------------------------------------------------------------
def _grid_evals(seed: int, grid=FAC_G_GRID) -> dict:
    """Fresh-build N=3 intact coincidence read at each grid candidate (pool A; symmetric algorithm, pool A only
    exercised here for compute scope -- the same reuse-by-import scope as the parent facilitation de-risk's own
    N=3 verification)."""
    return {g: round(float(F._n3_arm(seed, fac_on=True, lesion=False, N=3, fac_g=g,
                                     fac_U=F.FAC_U, fac_tau_F_steps=F.FAC_TAU_F_STEPS)["rel"]), 4)
            for g in grid}


def calibrate_fac_g(seed: int, grid=FAC_G_GRID, target: float = REL_TARGET, evals: dict | None = None):
    """The selection rule (identical for every seed): keep the shipped constant if it already clears `target`;
    otherwise take the smallest grid candidate reaching `target`, or -- if none do -- the floor-guarded arg-max
    (never worse than the shipped constant). Returns (chosen_g, evals, reason)."""
    evals = evals if evals is not None else _grid_evals(seed, grid)
    baseline_g = grid[0]
    baseline_rel = evals[baseline_g]
    if baseline_rel >= target:
        return baseline_g, evals, ("baseline g=%s already clears target %.3f (rel=%.4f) -- no change "
                                    "(a set-point regulator does not move what is already at its target)"
                                    % (baseline_g, target, baseline_rel))
    eligible = {g: r for g, r in evals.items() if r >= baseline_rel}     # the floor: never regress
    reaching = sorted(g for g, r in eligible.items() if r >= target)
    if reaching:
        chosen = reaching[0]
        return chosen, evals, "reached target %.3f at fac_g=%s (rel=%.4f)" % (target, chosen, evals[chosen])
    chosen = max(eligible, key=lambda g: eligible[g])
    return chosen, evals, ("target %.3f UNREACHABLE inside the safe grid (max %.4f) -- took the floor-guarded "
                            "arg-max at fac_g=%s" % (target, eligible[chosen], chosen))


def prove_it_adapts(seed: int = 44, evals: dict | None = None) -> dict:
    """ANTI-CHEAT #1: the SAME seed's SAME grid data, calibrated against two different targets, must select a
    DIFFERENT fac_g -- proof the mechanism is target-directed (a genuine set-point search), not a per-seed
    constant standing in for one. Reuses ALREADY-MEASURED grid evals when the caller has them (no duplicate
    brain builds); only measures fresh if none are passed."""
    evals = evals if evals is not None else _grid_evals(seed)
    g_hi, _, r_hi = calibrate_fac_g(seed, target=0.30, evals=evals)
    g_lo, _, r_lo = calibrate_fac_g(seed, target=0.21, evals=evals)
    adapts = bool(g_hi != g_lo)
    return {"seed": seed, "evals": evals, "target_0.30": {"chosen_g": g_hi, "reason": r_hi},
            "target_0.21": {"chosen_g": g_lo, "reason": r_lo}, "adapts": adapts}


def _diagnosis_margin(seed: int) -> float:
    """The pre-registered 'un-stabilized baseline': the ORIGINAL (no facilitation at all) operating-point
    diagnosis margin, READ from the committed artifact (never re-derived/hardcoded)."""
    path = os.path.join(DIAG_DIR, "op_s%d.json" % seed)
    with open(path) as fh:
        d = json.load(fh)
    return float(d["per_faculty"]["prospective-memory"]["margin_to_threshold"])


def _default_off_exact_compare(per_seed: dict, seeds) -> dict:
    """DEFAULT-OFF BYTE-IDENTICAL, asserted IN THE DATA (docs/TERMS.md 'byte-identical'): this run's OWN
    fac_g=FAC_G_DEFAULT grid point (measured fresh, THIS process) against the PRIOR session's INDEPENDENTLY
    committed facilitation-only artifact (a different process, a different day) -- an exact compare across two
    separate runs, not a re-derivation inside one."""
    if not os.path.exists(PRIOR_FACILITATION_ARTIFACT):
        return {"ok": None, "note": "prior artifact not found -- comparison skipped"}
    with open(PRIOR_FACILITATION_ARTIFACT) as fh:
        prior = {r["seed"]: r["intact_rel"] for r in json.load(fh)["n3_per_seed_ON"]}
    per = {}
    ok = True
    for s in seeds:
        mine = per_seed[s]["grid_evals"][FAC_G_GRID[0]]
        theirs = prior.get(s)
        match = (theirs is not None) and (mine == theirs)
        per[s] = {"this_run": mine, "prior_committed_artifact": theirs, "exact_match": match}
        ok = ok and match
    return {"ok": ok, "per_seed": per,
            "source": os.path.relpath(PRIOR_FACILITATION_ARTIFACT, _REPO).replace(os.sep, "/")}


def _frozen_silence(seed: int, g: float) -> dict:
    """The frozen N=5 gate (both A/B pools, all clauses) at gain `g` -- the anti-cheat that a stabilized
    coincidence lift did not come from breaking a single-input silence condition."""
    base.ProspectiveMemory = F.FacilitatedProspectiveMemory
    d = base.run_seed(seed, N=5, n_distractors=4, homeostat_on=True, sfa_on=True, plateau_on=True,
                       fac_on=True, fac_g=float(g), fac_U=F.FAC_U, fac_tau_F_steps=F.FAC_TAU_F_STEPS)
    return d


# --------------------------------------------------------------------------------------------------------
def _derisk(seeds=SEEDS, smoke=False):
    tag = "SMOKE" if smoke else "DE-RISK"
    print(f"OPERATING-POINT STABILIZER [{tag}] -- per-seed floor-guarded set-point search on fac_g; "
          f"target={REL_TARGET:.3f} grid={FAC_G_GRID}; {len(seeds)} seed(s)", flush=True)
    t0 = time.time()
    err = None
    per_seed = {}
    try:
        for s in seeds:
            print(f"\n--- seed {s}: grid search ---", flush=True)
            chosen_g, evals, reason = calibrate_fac_g(s)
            print(f"  evals={evals}", flush=True)
            print(f"  chosen fac_g={chosen_g} :: {reason}", flush=True)

            lb = F._n3_load_bearing(s, fac_on=True, N=3, fac_g=chosen_g, fac_U=F.FAC_U,
                                    fac_tau_F_steps=F.FAC_TAU_F_STEPS)
            print(f"  load-bearing @ chosen g: intact rel={lb['intact_rel']:.4f} fired={lb['intact_fired']} | "
                  f"lesion rel={lb['lesion_rel']:.4f} fired={lb['lesion_fired']} | LB={lb['load_bearing']}",
                  flush=True)

            froz = _frozen_silence(s, chosen_g)
            fails = [k for k, v in froz["clauses"].items() if not v]
            print(f"  frozen N=5 silence @ chosen g: passed={froz['passed']} max_silent={froz['max_silent']:.4f} "
                  f"fails={fails or 'none'}", flush=True)

            diag_margin = _diagnosis_margin(s)
            baseline_rel = evals[FAC_G_GRID[0]]
            baseline_margin = round(baseline_rel - FIRE_THR, 4)
            stabilized_margin = round(lb["intact_rel"] - FIRE_THR, 4)
            enlarged_vs_diagnosis = stabilized_margin > diag_margin
            enlarged_vs_production = stabilized_margin > baseline_margin
            total_gain_over_diag = stabilized_margin - diag_margin
            facilitation_share = (baseline_margin - diag_margin)
            stabilizer_share = attributable_to(
                f"seed {s}: stabilizer's OWN share of the enlargement over the diagnosis baseline",
                total_gain_over_diag, facilitation_share) if abs(total_gain_over_diag) > 1e-9 else None

            per_seed[s] = {
                "grid_evals": evals, "chosen_fac_g": chosen_g, "selection_reason": reason,
                "intact_rel": lb["intact_rel"], "intact_fired": lb["intact_fired"],
                "lesion_rel": lb["lesion_rel"], "lesion_fired": lb["lesion_fired"],
                "load_bearing": lb["load_bearing"],
                "frozen_passed": froz["passed"], "frozen_max_silent": round(froz["max_silent"], 4),
                "frozen_clause_fails": fails, "frozen_clauses": froz["clauses"],
                "diagnosis_margin": round(diag_margin, 4), "production_margin": baseline_margin,
                "stabilized_margin": stabilized_margin,
                "enlarged_vs_diagnosis": enlarged_vs_diagnosis, "enlarged_vs_production": enlarged_vs_production,
                "stabilizer_own_share_of_diagnosis_gain": stabilizer_share,
            }
        adapt_seed = 44 if 44 in per_seed else seeds[0]
        adapt = prove_it_adapts(adapt_seed, evals=per_seed[adapt_seed]["grid_evals"])
        print(f"\n--- ANTI-CHEAT: proves it adapts (seed {adapt_seed}, target 0.30 vs 0.21) ---", flush=True)
        print(f"  target=0.30 -> fac_g={adapt['target_0.30']['chosen_g']}  "
              f"target=0.21 -> fac_g={adapt['target_0.21']['chosen_g']}  adapts={adapt['adapts']}", flush=True)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        traceback.print_exc()

    if err is not None:
        summary = {"probe": "pmem_operating_point_stabilizer", "verdict": f"ERROR -- {err}", "go": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}
        _write(summary)
        return 1

    n_seeds = len(seeds)
    n_positive = sum(1 for s in seeds if per_seed[s]["stabilized_margin"] > 0)
    n_enlarged_diag = sum(1 for s in seeds if per_seed[s]["enlarged_vs_diagnosis"])
    n_enlarged_prod = sum(1 for s in seeds if per_seed[s]["enlarged_vs_production"])
    n_lb = sum(1 for s in seeds if per_seed[s]["load_bearing"])
    silence_regressed = [s for s in seeds if per_seed[s]["frozen_clause_fails"]]
    cheat = void_if(bool(silence_regressed),
                    f"the stabilizer REGRESSED a silence clause at seed(s) {silence_regressed} -> spurious fires "
                    f"(a homeostat that raises gain until everything fires is a CHEAT; VOID)")
    default_off_exact = _default_off_exact_compare(per_seed, seeds) if not smoke else None
    changed_seeds = [s for s in seeds if per_seed[s]["chosen_fac_g"] != FAC_G_GRID[0]]
    unchanged_seeds = [s for s in seeds if per_seed[s]["chosen_fac_g"] == FAC_G_GRID[0]]
    # the THINNEST seed in THIS run (smallest production margin) anchors the reaches()/control() single-seed
    # checks -- generically, not a hardcoded 44 (so --smoke / --seed single-seed runs do not crash).
    thin_seed = min(seeds, key=lambda s: per_seed[s]["production_margin"])

    # THE PRE-REGISTERED GATE (task's own wording): margin strictly positive AND enlarged vs the DIAGNOSIS
    # baseline, on ALL seeds; load-bearing preserved; silence intact; not smoke.
    go = bool(n_positive == n_seeds and n_enlarged_diag == n_seeds and n_lb == n_seeds
             and not cheat and not smoke)

    vd = Verdict("pmem_operating_point_stabilizer")
    vd.require("stabilized margin strictly positive (per-seed count, vs FIRE_THR)", n_positive,
               expect=lambda x, n=n_seeds: x == n)
    vd.require("stabilized margin ENLARGED vs the diagnosis baseline (op_s*.json, pre-registered)", n_enlarged_diag,
               expect=lambda x, n=n_seeds: x == n)
    vd.require("load-bearing preserved at the chosen gain (per-seed count)", n_lb, expect=lambda x, n=n_seeds: x == n)
    for c in SILENCE_CLAUSES:
        vd.require(f"frozen-gate silence held at the chosen gain: {c}",
                   sum(1 for s in seeds if per_seed[s]["frozen_clauses"].get(c)),
                   expect=lambda x, n=n_seeds: x == n)
    vd.reaches(f"thinnest seed in this run (s{thin_seed}): production constant -> stabilized",
              per_seed[thin_seed]["production_margin"], per_seed[thin_seed]["stabilized_margin"])
    vd.control(f"seed {thin_seed} stabilized vs diagnosis margin", per_seed[thin_seed]["stabilized_margin"],
              per_seed[thin_seed]["diagnosis_margin"], min_separation=0.0)
    vd.disabled("STDP / long-term Hebbian LTP / OU-noise",
                "identical scope to the parent facilitation de-risk; the only added mechanism is the bounded "
                "per-seed fac_g set-point search")
    decided = vd.decide(go)

    thin_pct = ((per_seed[thin_seed]['stabilized_margin'] / per_seed[thin_seed]['production_margin'] - 1) * 100
                if per_seed[thin_seed]['production_margin'] else float("nan"))
    thin_share = per_seed[thin_seed]["stabilizer_own_share_of_diagnosis_gain"]
    cliff_note = (
        " DISCOVERED RESIDUAL: fac_g is NOT a safely-widenable operating point -- a fresh, self-consistent build "
        "shows a real cliff at fac_g>=10000 for seed 44 (rel collapses 0.2183->0.1506), so the grid is bounded "
        "there, not swept further; REL_TARGET=%.2f is UNREACHABLE inside the safe grid on %s -- an honest "
        "residual, not forced to GO." % (REL_TARGET, changed_seeds or "the seed(s) below target")
    ) if 44 in per_seed else ""
    verdict = (
        f"{'GO' if go else 'VOID' if cheat else 'BOUNDARY/PARTIAL'} -- per the PRE-REGISTERED gate (margin vs the "
        f"diagnosis baseline op_s*.json), {n_positive}/{n_seeds} seeds hold a strictly positive stabilized margin "
        f"and {n_enlarged_diag}/{n_seeds} are enlarged vs that baseline; load-bearing preserved {n_lb}/{n_seeds}; "
        f"silence-regressed={silence_regressed or 'none'}. HONEST SCOPE NOTE: compared against the CURRENT "
        f"production baseline (facilitation @ fac_g={FAC_G_DEFAULT}, already shipped/committed), only "
        f"{n_enlarged_prod}/{n_seeds} seeds actually CHANGE under this stabilizer (seed(s) {changed_seeds or 'none'} "
        f"-- the ones below REL_TARGET={REL_TARGET:.2f}); seed(s) {unchanged_seeds or 'none'} are left at the "
        f"shipped constant BY DESIGN (already at/above target -- a set-point regulator that is already there does "
        f"not move). The thinnest seed in this run (s{thin_seed}): production margin "
        f"{per_seed[thin_seed]['production_margin']:+.4f} -> stabilized {per_seed[thin_seed]['stabilized_margin']:+.4f} "
        f"(~{thin_pct:.0f}% larger), of which "
        f"{'UNDEFINED' if thin_share is None else thin_share} is this session's OWN share (the rest, if any, is the "
        f"already-committed facilitation fix's credit)." + cliff_note
    )

    summary = {
        "probe": "pmem_operating_point_stabilizer", "verdict": verdict, "go": bool(go),
        "task": ("Prospective-memory operating-point stabilizer: replace the single global facilitation gain "
                 "constant (fac_g=6000, uniform across seeds) with a per-seed, floor-guarded, target-seeking "
                 "grid search over a PRE-REGISTERED candidate set -- a discrete realization of a Turrigiano-style "
                 "homeostatic set-point regulator, bounded at an empirically-discovered non-monotonic cliff. "
                 "Identical algorithm/target/grid for every seed; the OUTCOME differs per seed because the search "
                 "reads the pool's own coincidence output."),
        "gate": {"FIRE_THR": FIRE_THR, "REL_TARGET": REL_TARGET, "FAC_G_GRID": list(FAC_G_GRID),
                 "FAC_G_DEFAULT": FAC_G_DEFAULT},
        "seeds": list(seeds), "per_seed": per_seed,
        "n_positive": n_positive, "n_enlarged_vs_diagnosis": n_enlarged_diag,
        "n_enlarged_vs_production": n_enlarged_prod, "n_load_bearing": n_lb,
        "silence_regressed": silence_regressed,
        "default_off_exact_compare": default_off_exact,
        "calibrated_fac_g_table": {s: per_seed[s]["chosen_fac_g"] for s in seeds},
        "prove_it_adapts": adapt,
        "preconditions": (decided or {}).get("preconditions"),
        "disabled_processes": (decided or {}).get("disabled_processes"),
        "verdict_status": (decided or {}).get("status"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "BIOLOGY": ("Homeostatic regulation of synaptic/intrinsic gain toward a firing-rate set-point (Turrigiano "
                    "2011; Desai, Rutherford & Turrigiano 1999). Realized here as a bounded per-seed calibration "
                    "of the facilitation gain fac_g (the same Mg-block-gated Tsodyks-Markram facilitation current "
                    "as the parent GO substrate) toward a fixed target (1.5x FIRE_THR); the search is discretized "
                    "and safety-bounded because a fresh, self-consistent evaluation of this substrate shows a real "
                    "non-monotonic cliff at high gain (the pool's own pre-existing intrinsic-excitability homeostat "
                    "+ NMDA-plateau calibration over-compensates), which a blind continuous/proportional controller "
                    "would walk into."),
    }
    _write(summary)
    print("\n" + "=" * 118, flush=True)
    print(f"[op-stabilizer] VERDICT: {verdict}", flush=True)
    print(f"[op-stabilizer] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (go or smoke) else 1


def _write(summary):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def selftest() -> bool:
    """Fail-in-the-failing-direction; no brain build."""
    os.environ.pop(STABILIZER_ENV, None)
    checks = {
        "REL_TARGET is 1.5x FIRE_THR": abs(REL_TARGET - 1.5 * FIRE_THR) < 1e-9,
        "FAC_G_GRID[0] equals the shipped FAC_G_DEFAULT": FAC_G_GRID[0] == FAC_G_DEFAULT,
        "FAC_G_GRID is sorted and bounded at the discovered cliff (<=11000)": (
            list(FAC_G_GRID) == sorted(FAC_G_GRID) and max(FAC_G_GRID) <= 11000.0),
        "default-off (env unset) -> stabilizer_enabled() is False": stabilizer_enabled() is False,
        "'0'/'false' -> stabilizer_enabled() is False": _env_check("0") is False and _env_check("false") is False,
        "'1'/'true' -> stabilizer_enabled() is True": _env_check("1") is True and _env_check("true") is True,
        "unseeded fallback returns FAC_G_DEFAULT": stabilized_fac_g_for_seed(999999) == FAC_G_DEFAULT,
        "every tabulated seed's calibrated gain is in the grid": all(
            g in FAC_G_GRID for g in CALIBRATED_FAC_G.values()),
        "every SEED is tabulated": set(CALIBRATED_FAC_G) == set(SEEDS),
        "selection rule is a pure function of (evals, target) -- re-deriving seed 44 at target=0.30 from cached "
        "evals reproduces the committed CALIBRATED_FAC_G entry": _reproduces_s44(),
    }
    ok = all(checks.values())
    print("=== OPERATING-POINT STABILIZER SELF-TEST ===")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def _env_check(val):
    os.environ[STABILIZER_ENV] = val
    try:
        return stabilizer_enabled()
    finally:
        os.environ.pop(STABILIZER_ENV, None)


def _reproduces_s44():
    # a FROZEN copy of seed 44's committed grid evals (from the --derisk artifact) -- selftest must not build a
    # brain, so the selection RULE is re-run over already-measured numbers, not re-measured.
    frozen_evals = {6000.0: 0.2111, 7000.0: 0.2078, 8000.0: 0.2139, 9000.0: 0.2183, 10000.0: 0.1506, 11000.0: 0.1589}
    chosen, _, _ = calibrate_fac_g(44, evals=frozen_evals)
    return chosen == CALIBRATED_FAC_G[44] == 9000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prove-adapts", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1
    if a.prove_adapts:
        print(json.dumps(prove_it_adapts(a.seed or 44), indent=2))
        return 0
    seeds = [a.seed] if a.seed is not None else a.seeds
    if a.smoke:
        return _derisk([seeds[0]], smoke=True)
    return _derisk(seeds, smoke=False)


if __name__ == "__main__":
    raise SystemExit(main())
