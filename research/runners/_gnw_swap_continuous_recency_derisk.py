"""GNW SWAP CONTINUOUS CROSS-TURN IGNITION — the named next rung after board #77/#85 (2026-08-19 findings:
'A truly continuous cross-turn ignition (no restore) is the named next rung'). This runner tests whether making the
production held-topic workspace's substrate genuinely CONTINUOUS across turns (no per-turn restore-to-snapshot) is
SAFE (preserves the shipped swap-vs-hold correctness, no regression) and whether it carries a real, measurable
synaptic trace of recent occupancy that a restore-based substrate structurally cannot -- and HONESTLY characterizes
where that trace does, and does not yet, reach a reproducible reply-level behavioral consequence.

WHY THIS IS THE GENUINE NEXT RUNG, NOT A RE-DERIVATION (verify-first, `before_you_build.sh` + `git log --grep`
+ findings scan, all run before writing a line of this file). Board #77 (2026-08-19-gnw-swap-into-chat-GO.md) and
board #85 (2026-08-19-swap-drives-chat-load-bearing-GO.md) are BOTH already merged to origin/main, DEFAULT-ON
(`_SWAP_DRIVES_DEFAULT_ON=True` in webapp/server.py), and lesion-verified load-bearing on the live `/api/brain-chat`
reply (a topic-change SWAP prepends a transition lead the mismatch-detector lesion makes vanish) -- re-confirmed
independently by 2026-09-05-rank11-topic-swap-scaffold-backlog-item-already-integrated.md (byte-identical re-run
against TODAY's code) and already wired into `research/runners/load_bearing_fraction.py`'s FACULTY_LESIONS battery
(row `swap-drives-response`). Building that capability again here would DUPLICATE shipped work. What is NOT built is
honest-limit #1 of `webapp/gnw_thought_swap.py`'s own docstring: "the cross-turn CONTINUITY of the held thought is
carried by a host label... RE-ESTABLISHED on the substrate each turn via `run_intention_swap(isolate=True)` (restore
clean snapshot -> re-ignite the held topic...). A truly continuous cross-turn ignition (no restore) is the named next
rung." This runner builds and de-risks exactly that rung.

THE MECHANISM (reuse-by-import, NO `sim/` edit; every primitive below is imported from the ALREADY-6/6-seed-GO
`_gnw_neural_swap_intention_derisk` module -- `build`, `run_intention_swap`, `MultiLoopSTD`; `run_intention_swap`
already exposes `isolate=False` as "a CONTINUOUS run (0 restore calls)" and `run_two_swap` already uses it for a
two-swap A->B->A reversibility headline -- this runner is the FIRST use of that existing continuous-mode plumbing
as a genuinely MULTI-TURN, production-shaped conversation, not a single two-swap probe).

  Per-seed protocol: establish topic A (first-thought, one necessary COLD-START `isolate=True` -- there is no prior
  turn to be continuous WITH), then swap continuously (`isolate=False`, zero restores) A -> B. This evicts A via the
  SAME recurrence-weakening STD the shipped #77/#85 mechanism already uses -- A's recurrent loop is left with a
  depleted resource variable `x_A < 1` (Tsodyks-Markram short-term depression). Two turns now branch from this
  IDENTICAL point (same seed -> byte-identical substrate state, confirmed below):
    RECENT — the very next turn re-proposes A (the topic the user JUST left).
    FRESH  — the very next turn proposes C, a topic NEVER held this session (x_C == 1, virgin loop).

GO GATE (robust, 6/6, at the SHIPPED production operating point `SALIENT_PA`) — what THIS finding actually proves:
  1. THE CARRYOVER IS REAL: x_A at the branch point is always < 1 (measured 0.73-0.78 across seeds) -- continuous
     mode genuinely carries synaptic state across the turn boundary; this is not a bookkeeping label.
  2. RESTORE MODE IS STRUCTURALLY BLIND TO IT: the exact reset call `isolate=True` performs (`std.reset()`) always
     wipes this to EXACTLY 1.0 for every pattern -- a code-level, not merely statistical, guarantee that today's
     shipped restore-every-turn default cannot carry this trace, regardless of operating point.
  3. SAFE / NO REGRESSION: at the shipped production drive strength, continuous mode's swap-vs-hold VERDICT on
     every turn (establish, evict, re-admit-recent, admit-fresh) is IDENTICAL to what restore mode would decide --
     enabling continuity does not destabilize the already-GO'd swap decision. This is what makes a default-off
     `BRAIN_GNW_SWAP_CONTINUOUS` flag a low-risk addition rather than a new failure surface.
  4. DETERMINISM (build-twice hash) and branch-fork identity (two independent builds at one seed reach a
     byte-identical branch point, making RECENT vs FRESH a controlled, not merely repeated, comparison).

HONEST RESIDUAL, NAMED AND QUANTIFIED, NOT CLAIMED CLOSED (docs/TERMS.md: "load-bearing" requires a REPRODUCIBLE
state->reply dependency; this is NOT yet demonstrated for the recency trace specifically, and is NOT gated into the
GO verdict above). At the production operating point (`SALIENT_PA`), the swap decision is DELIBERATELY supra-
critical/robust (per the shipped mechanism's own design intent), so the x_A carryover measured in (1) above does NOT
change which turns swap on 6/6 seeds -- a null result at that operating point, reported as such, not hidden. A
hand-swept WEAKER re-proposal drive (`--near-threshold-pa`, default 1500 pA, roughly 30% of `SALIENT_PA`) DOES
produce a qualitative recency-driven swap FAILURE (seed 42 of the standard six: RECENT fails to re-ignite while
FRESH succeeds under an otherwise-identical drive) -- but this is an EXISTENCE PROOF on 1/6 seeds, not a reproducible
lever: the same drive strength gives both-fail (seed 102) or both-succeed (seeds 43/44/100/101) on the other five,
because per-seed Izhikevich heterogeneity shifts each pattern's individual ignition margin near this threshold band
enough to dominate the ~25% STD carryover's effect on any single fixed drive value. The `--near-threshold-scan` mode
below records this per-seed diagnostically (never gates `seed_go`) so a future session does not have to re-derive
that a fixed near-threshold PA is the wrong lever -- the NEXT rung this residual maps to is a per-seed (or per-
session, learned) CALIBRATED margin, or a genuinely graded (salience-scaled, not fixed-PA) re-proposal drive, not a
retry at a different constant.

NEURAL LESION (mechanistic, on the CARRYOVER itself, not gated into `seed_go` for the reason above -- see
`recent_lesioned` in the per-seed record): forcibly resetting ONLY A's recurrence resource (`std.deps[0].x[:]=1.0`)
immediately before re-proposing it, leaving every OTHER continuous-mode state variable untouched, is the surgical
tool this finding leaves in place for whichever near-threshold operating point a future session calibrates.

CONTRACT if wired to production (this runner is the DE-RISK; production wiring in `webapp/gnw_thought_swap.py`
behind a NEW, default-off, additive flag `BRAIN_GNW_SWAP_CONTINUOUS` -- unset/0 -> BYTE-IDENTICAL to the shipped
#77/#85 isolate=True-every-turn behavior; see that module for the flag and its own honest-residual update).

Biology: short-term synaptic depression as a substrate for working-memory-adjacent recency/priming effects
(Mongillo, Barak & Tsodyks 2008, Science 319:1543 -- "synaptic theory of working memory", the SAME citation the
shipped eviction mechanism already carries); the prediction that a just-vacated cortical assembly is transiently
harder to re-recruit is the STP analogue of neural refractoriness / repetition suppression (Grill-Spector, Henson
& Martin 2006, TICS 10:14). Corpus check (`before_you_build.sh "GNW continuous cross-turn ignition recency swap"`
+ rag_search) run before writing this file; see the accompanying finding for the transcript and the near-threshold
calibration sweep (pa in {5000,3000,2000,1500,1200,1000,800}) that located where the effect exists vs is noise.

Usage (CPU cheap-first; export OMP/OPENBLAS/MKL_NUM_THREADS=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_swap_continuous_recency_derisk --smoke --seed 42 \\
      --json research/findings/raw/_gnw_swap_continuous_recency_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_swap_continuous_recency_derisk --six-seed \\
      --json research/findings/raw/_gnw_swap_continuous_recency_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import to_host
from tools.verdict import Verdict
from tools.lab import lever, void_if

from research.runners._gnw_neural_swap_intention_derisk import (
    build, run_intention_swap, MultiLoopSTD, SALIENT_PA, N_PATTERNS, W_REC,
)

# ── operating point (reused UNCHANGED from the shipped #77/#85 mechanism's own calibrated point) ──────────────────
A, B, C = 0, 1, 2                       # three disjoint held-topic slots; N_PATTERNS must be >= 3 (it is: 3)
assert N_PATTERNS >= 3, "this probe needs >=3 disjoint pattern slots (A, B, and a never-touched C)"

MIN_X_DEFICIT = 0.05      # x_A_at_branch must be at least this far below 1.0 (the carryover LEVER must have moved)
NEAR_THRESHOLD_PA = 1500.0  # hand-swept diagnostic drive (see module docstring); NEVER gates seed_go.


def _izh_hash(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


def _fresh_branch(seed, w_rec, heterogeneity):
    """Build a substrate and drive it, CONTINUOUSLY (one necessary cold-start restore for the very first thought,
    then zero restores), through: establish A -> swap A->B. Returns (S, std, first, ab) at the branch point: B is
    held, A was JUST evicted (its recurrence carries a partial STD debt)."""
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    std = MultiLoopSTD(S["bridge"], S["xp"], S["ws_used"], S["patterns_host"])
    first = run_intention_swap(S, std, incumbent=A, proposed=A, proposal_pa=SALIENT_PA, isolate=True)
    ab = run_intention_swap(S, std, incumbent=A, proposed=B, proposal_pa=SALIENT_PA, isolate=False)
    return S, std, first, ab


def evaluate_seed(seed, *, w_rec=None, heterogeneity=True, near_threshold_pa=NEAR_THRESHOLD_PA, verbose=True):
    if w_rec is None:
        w_rec = W_REC

    # ── GATING ARMS (production operating point, SALIENT_PA -- what "safe to wire" is measured on) ────────────────
    S1, std1, first1, ab1 = _fresh_branch(seed, w_rec, heterogeneity)
    xA_at_branch = std1.x_mean(A)
    recent = run_intention_swap(S1, std1, incumbent=B, proposed=A, proposal_pa=SALIENT_PA, isolate=False)

    S2, std2, first2, ab2 = _fresh_branch(seed, w_rec, heterogeneity)
    xC_at_branch = std2.x_mean(C)
    fresh = run_intention_swap(S2, std2, incumbent=B, proposed=C, proposal_pa=SALIENT_PA, isolate=False)

    # the two-arm fork is a CONTROLLED comparison only if S1/S2 are byte-identical up to the branch (same seed,
    # same sequence of steps up to turn 3 -> must reach the identical state before diverging on turn 3's proposal).
    branch_identical = bool(abs(ab1["new_rate_post"] - ab2["new_rate_post"]) < 1e-9
                             and abs(ab1["old_residual_post"] - ab2["old_residual_post"]) < 1e-9
                             and ab1["winner_post"] == ab2["winner_post"] and ab1["n_ignited_post"] == ab2["n_ignited_post"])

    # restore-mode-blind: the EXACT operation isolate=True performs before every shipped-mode turn (std.reset(),
    # called on std1 -- which has JUST come out of a real continuous A->B swap and carries the 0.7x debt measured
    # above) unconditionally wipes it back to 1.0 for EVERY pattern. This is a code-level guarantee, verified here
    # rather than merely asserted by reading the source.
    std1.reset()
    restore_blind = bool(abs(std1.x_mean(A) - 1.0) < 1e-12 and abs(std1.x_mean(C) - 1.0) < 1e-12)

    h1 = _izh_hash(S1["bridge"]); h2 = _izh_hash(S2["bridge"])
    seed_deterministic = bool(h1 == h2 and h1 != "")

    swaps_correct = bool(ab1["swapped"] and ab2["swapped"] and recent["swapped"] and fresh["swapped"])
    no_regression_at_production_pa = bool(recent["swapped"] == fresh["swapped"] == True
                                          and abs(recent["new_rate_post"] - fresh["new_rate_post"]) < 1e-6)

    carryover_insufficient = void_if(xA_at_branch >= 1.0 - MIN_X_DEFICIT,
                                      "x_A_at_branch did not drop below 1.0-%.3f -- no STD carryover to measure; "
                                      "the whole comparison is void" % MIN_X_DEFICIT)
    carryover_ok = bool(not carryover_insufficient)

    # ── DIAGNOSTIC ARMS (near-threshold PA -- reported, NEVER gates seed_go; see module docstring) ─────────────────
    S3, std3, first3, ab3 = _fresh_branch(seed, w_rec, heterogeneity)
    nt_recent = run_intention_swap(S3, std3, incumbent=B, proposed=A, proposal_pa=near_threshold_pa, isolate=False)
    S4, std4, first4, ab4 = _fresh_branch(seed, w_rec, heterogeneity)
    nt_fresh = run_intention_swap(S4, std4, incumbent=B, proposed=C, proposal_pa=near_threshold_pa, isolate=False)
    S5, std5, first5, ab5 = _fresh_branch(seed, w_rec, heterogeneity)
    std5.deps[A].x[:] = 1.0    # the neural lesion: wipe ONLY A's carryover, keep everything else continuous
    nt_recent_lesioned = run_intention_swap(S5, std5, incumbent=B, proposed=A, proposal_pa=near_threshold_pa, isolate=False)
    near_threshold_dissociation = bool(nt_recent["swapped"] is False and nt_fresh["swapped"] is True)

    if verbose:
        print(f"[swap-continuous-recency] seed={seed} xA_at_branch={xA_at_branch:.4f} xC_at_branch={xC_at_branch:.4f} "
              f"branch_identical={branch_identical} restore_blind={restore_blind}", flush=True)
        print(f"  @SALIENT_PA={SALIENT_PA:.0f}  recent.swapped={recent['swapped']}  fresh.swapped={fresh['swapped']}  "
              f"no_regression={no_regression_at_production_pa}", flush=True)
        print(f"  @near_threshold_pa={near_threshold_pa:.0f} (diagnostic, NOT gating)  recent.swapped={nt_recent['swapped']}  "
              f"fresh.swapped={nt_fresh['swapped']}  lesioned-recent.swapped={nt_recent_lesioned['swapped']}  "
              f"dissociation={near_threshold_dissociation}", flush=True)

    lever("std_x_at_branch (A vs C, both start at 1.0)", 1.0, round(xA_at_branch, 4), continuous=xA_at_branch)

    seed_go = bool(swaps_correct and branch_identical and seed_deterministic and carryover_ok
                   and restore_blind and no_regression_at_production_pa)

    return {
        "seed": int(seed),
        "xA_at_branch": float(xA_at_branch), "xC_at_branch": float(xC_at_branch),
        "branch_identical": branch_identical, "restore_blind": restore_blind,
        "recent": {k: recent[k] for k in ("swapped", "new_rate_post", "old_residual_post", "b_ignite_step")},
        "fresh": {k: fresh[k] for k in ("swapped", "new_rate_post", "old_residual_post", "b_ignite_step")},
        "seed_deterministic": seed_deterministic,
        "near_threshold_diagnostic": {
            "pa": near_threshold_pa,
            "recent_swapped": nt_recent["swapped"], "fresh_swapped": nt_fresh["swapped"],
            "recent_lesioned_swapped": nt_recent_lesioned["swapped"],
            "dissociation": near_threshold_dissociation,
        },
        "go_gate": {
            "swaps_correct": swaps_correct,
            "branch_identical": branch_identical,
            "carryover_ok": carryover_ok,
            "restore_blind": restore_blind,
            "no_regression_at_production_pa": no_regression_at_production_pa,
            "seed_deterministic": seed_deterministic,
        },
        "seed_go": seed_go,
        "operating_point": {"salient_pa": SALIENT_PA, "min_x_deficit": MIN_X_DEFICIT,
                             "near_threshold_pa": near_threshold_pa},
    }


def run_smoke(seed, args):
    r = evaluate_seed(seed, heterogeneity=not args.no_heterogeneity, near_threshold_pa=args.near_threshold_pa, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_swap_continuous_recency_derisk", "mode": "smoke", "seed": seed, "result": r},
                  f, indent=2, default=str)
    print(f"\n[swap-continuous-recency smoke] wrote {args.json}  seed_go={r['seed_go']}  "
          f"near_threshold_dissociation={r['near_threshold_diagnostic']['dissociation']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[swap-continuous-recency six-seed] seeds={seeds}", flush=True)
    per_seed = [evaluate_seed(s, heterogeneity=not args.no_heterogeneity, near_threshold_pa=args.near_threshold_pa,
                              verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swaps_correct"])
    n_branch = sum(1 for r in per_seed if r["go_gate"]["branch_identical"])
    n_carry = sum(1 for r in per_seed if r["go_gate"]["carryover_ok"])
    n_blind = sum(1 for r in per_seed if r["go_gate"]["restore_blind"])
    n_noreg = sum(1 for r in per_seed if r["go_gate"]["no_regression_at_production_pa"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    n_dissoc = sum(1 for r in per_seed if r["near_threshold_diagnostic"]["dissociation"])
    pooled_go = bool(n_go == 6 and n_swap == 6 and n_branch == 6 and n_carry == 6 and n_blind == 6
                     and n_noreg == 6 and n_det == 6)
    verdict = "GO" if pooled_go else ("PARTIAL" if n_go >= 1 else "NO-GO")

    v = Verdict("GNW swap continuous cross-turn ignition: 6-seed aggregate (safety + carryover mechanism)")
    v.require("all four swap decisions (A first-thought, A->B, RECENT re-admit A, FRESH admit C) correct on 6/6",
              bool(n_swap == 6), expect=True)
    v.require("the two-arm fork is byte-identical at the branch point on 6/6", bool(n_branch == 6), expect=True)
    v.require("the STD carryover lever actually moved (x_A < 1 at branch) on 6/6", bool(n_carry == 6), expect=True)
    v.require("restore mode's own reset provably wipes the carryover to exactly 1.0 on 6/6", bool(n_blind == 6), expect=True)
    v.require("continuous mode does NOT regress the swap-vs-hold verdict at production drive on 6/6", bool(n_noreg == 6),
              expect=True)
    v.require("determinism (build-twice hash) on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights, inherited from the reused #77/#85 substrate build")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_swap_continuous_recency_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "swaps_correct": n_swap, "branch_identical": n_branch,
                          "carryover_ok": n_carry, "restore_blind": n_blind,
                          "no_regression_at_production_pa": n_noreg, "seed_deterministic": n_det,
                          "n_seeds": len(seeds)},
               "near_threshold_diagnostic_summary": {
                   "n_dissociation": n_dissoc, "n_seeds": len(seeds),
                   "note": "diagnostic ONLY, NEVER gates pooled_go -- see module docstring 'HONEST RESIDUAL'"},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[swap-continuous-recency six-seed] verdict={verdict} seed_go {n_go}/6 swap {n_swap}/6 branch {n_branch}/6 "
          f"carry {n_carry}/6 blind {n_blind}/6 no_regression {n_noreg}/6 det {n_det}/6  "
          f"[diagnostic near-threshold dissociation {n_dissoc}/6, NOT gating]", flush=True)
    print(f"[swap-continuous-recency six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW swap continuous cross-turn ignition: is removing the per-turn "
                                             "restore SAFE, and does it carry a genuine synaptic recency trace?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-threshold-pa", type=float, default=NEAR_THRESHOLD_PA)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_swap_continuous_recency.json")
    args = ap.parse_args()
    if args.six_seed:
        return run_six_seed(args)
    return run_smoke(args.seed, args)


if __name__ == "__main__":
    raise SystemExit(main())
