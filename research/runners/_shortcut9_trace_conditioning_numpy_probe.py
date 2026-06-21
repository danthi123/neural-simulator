"""SHORTCUT #9 -- the GENUINE close (numpy GATE): a TRACE-CONDITIONING task where the
value is PROVABLY LOAD-BEARING (the validate-by-function close the nav deploy lacked).

Scoping: research/findings/2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md.
The #9 dendrite-graded value is a GO in isolation (delta=1.33, 6/6) but its nav DEPLOY is a
qualified-NEGATIVE: the moving-goal gridworld is IMMEDIATE-REWARD-SOLVABLE, so the value's
distinctive function (credit assignment over a temporal GAP) is never exercised, and lesioning
the value barely moves navigation (dendcritic 8.47 ~= value-lesion 9.08, Delta 7.2%). The genuine
close = validate the value BY ITS FUNCTION on a task where it IS load-bearing: TRACE CONDITIONING
(catalog F.22/F.23; Hesslow-Yeo 2002; the H.M. trace-vs-delay dissociation).

THE DECISIVE DESIGN -- the delay-vs-trace 2x2 factorial (the catalog's own validation logic):
    arm (TRACE / DELAY)  x  value (full-bootstrap / value-LESION)
  * G2 (load-bearing, the HEADLINE): lesion the value -> the TRACE-arm conditioned value/response
    COLLAPSES. (This is the gate the nav deploy FAILED; here it must PASS because the trace gap
    NEEDS the value to bridge it.)
  * G3 (the discriminating IMMEDIATE-REWARD control): the SAME value-lesion on the DELAY arm
    (gap=0, US overlaps CS) does NOT collapse it -> proves the task DISCRIMINATES "needs the
    critic" (trace) from "immediate-reward-solvable" (delay) = directly answers the deploy confound.
  * plus the standard anti-cheats: NO-LEARNING collapses, PERMUTED-CS-US collapses.

THE NUMPY REALIZATION (the cheap-first RL-sanity GATE, before any spiking):
  The CSC (complete-serial-compound) TD critic from sim.td_value_critic. The cue is represented
  by a tapped-delay state (one tap per time-since-CS), so a learned V can RIDE the taps across a
  CS-free gap to the US.  The trace/delay/gap is encoded in WHERE the US lands relative to the CS:
    - DELAY arm: t_us = onset + 0  (US co-active with CS onset, no gap; the SHORTEST possible delay)
    - TRACE arm: t_us = onset + GAP (a CS-free interval of GAP taps before the US)
  The "value lesion" is the no_bootstrap mode (delta = r - V; the gamma*V(s') bootstrap REMOVED) --
  the numpy analogue of silencing the dendrite-graded value: WITHOUT the bootstrap the only credit
  path back to the CS is the raw eligibility trace (which decays over the gap, so a longer trace
  STARVES the un-bootstrapped critic -> the value at the CS COLLAPSES). WITH the bootstrap the
  value at each tap targets gamma*V(next tap), so V propagates back across the whole gap regardless
  of the eligibility window.

THE DEPENDENT VARIABLE (what "the value is doing work" looks like):
  V(CS_onset) = the learned value AT the cue (the prediction that bridges the gap). A working critic
  acquires V at the CS (Schultz cue-shift: the prediction migrates onto the CS); a value-lesioned
  critic on the TRACE arm CANNOT (the gap starves the eligibility), so V(CS) collapses to ~0.

GO = G2 collapses (value load-bearing on the TRACE arm) AND G3 does NOT collapse (the DELAY-arm
control discriminates) AND the no-learning + permuted controls collapse. This is the pure-RL proof
that the TASK discriminates -- the prerequisite before lifting onto the spiking limbic core.

CPU only. Run under SIM_BACKEND=numpy (no GPU; pure array math).

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._shortcut9_trace_conditioning_numpy_probe \
        --seeds 42,43,44 --out research/findings/raw/_shortcut9_trace_numpy.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sim.kernels import fused_eligibility_trace_decay  # REUSED UNMODIFIED (the eligibility carrier)
from sim.td_value_critic import csc_features, GAMMA, ALPHA, LAMBDA  # the validated CSC TD core


# ---------------------------------------------------------------------------
# The trace-conditioning critic run. A thin specialization of run_pavlovian:
#  * the gap is a knob (DELAY=0 / TRACE=gap>0); t_us = onset + gap.
#  * value_lesion (the #9 analogue): no_bootstrap (delta = r - V), the value/bootstrap REMOVED.
#  * no_learning / permuted: the standard anti-cheats.
# Returns the dependent variables: V(CS_onset) (the gap-bridging prediction) + the late RPEs.
# ---------------------------------------------------------------------------
def run_trace(seed, *, gap, value_lesion=False, no_learning=False, permuted=False,
              T=24, cs_onsets=(3, 4, 5, 6, 7), n_trials=1500, n_late=150, lam=0.5):
    """One CSC-TD trace-conditioning run with a CS->gap->US schedule.

    gap=0 is the DELAY arm (US co-active with the CS onset); gap>0 is the TRACE arm (a CS-free
    interval before the US).

    value_lesion is the FAITHFUL numpy analogue of silencing the dendrite-graded value at the SNc
    (the bridge --...-graded-strength 0): the value's CONTRIBUTION TO THE TEACHING SIGNAL is zeroed,
    so delta = r (a pure reward signal, no -V(s) subtraction and no +gamma*V(s') bootstrap; bounded
    in [0,1], unlike the DIVERGENT delta=r-V Monte-Carlo rule whose 1e9 blow-up would mask the gap
    effect -- confirmed: sim.td_value_critic's own no_bootstrap mode diverges, vrmse 178). The critic
    synapses + STDP eligibility are UNCHANGED; only the DA teaching signal loses the value. Under
    delta=r the ONLY credit path back to the CS is the raw eligibility trace, which decays by
    (gamma*lambda)^gap over the CS-free interval -- so a longer gap STARVES the value-lesioned critic
    (V(CS)/V(US) -> (gamma*lambda)^gap) while a gap=0 (DELAY) arm keeps the CS-onset eligibility
    maximal at the US (V(CS)/V(US) -> 1, survives WITHOUT the value). With the value intact,
    delta = r + gamma*V(s') - V(s) bootstraps V back across the WHOLE gap regardless of the
    eligibility window (V(CS)/V(US) -> gamma^gap, the Schultz cue-shift). A SHORT eligibility window
    (lam ~0.5, the biological ~0.2-2s reward-DA window, Yagishita 2014) is what makes the bootstrap
    and the trace dissociate: the bootstrap bridges an arbitrarily long gap, the eligibility cannot.

    permuted: the US time is drawn at RANDOM each trial (no CS->US contingency at all), so the cue
    carries no predictive information -> no value should acquire at the CS.

    Returns the DEPENDENT VARIABLES: v_cs (V at the cue onset), v_us (V at the US tap), and the
    BOUNDED value_transfer = V(CS)/V(US) (the fraction of US-value that reached the CS across the
    gap -- the gap-bridging measure, trial-count-independent), + the late dCS/dUS RPEs."""
    rng = np.random.default_rng(int(seed))
    n_feat = T + 1
    w = np.zeros(n_feat)
    decay = GAMMA * float(lam)
    onsets = tuple(int(o) for o in cs_onsets)
    late_dCS, late_dUS = [], []
    for trial in range(int(n_trials)):
        onset = int(rng.choice(onsets))
        if permuted:
            # NO contingency: the US lands at a RANDOM tap, uncorrelated with the cue onset (the
            # cue predicts nothing). Drawn over the same span the paired US could occupy.
            t_us = int(rng.integers(onsets[0], min(onsets[-1] + int(gap) + 1, T)))
        else:
            t_us = onset + int(gap)
        if t_us >= T:                     # keep the US inside the window
            t_us = T - 1
        X = csc_features(onset, T)
        e = np.zeros(n_feat)
        for t in range(T):
            r = 1.0 if t == t_us else 0.0
            v_t = X[t] @ w
            v_tp1 = (X[t + 1] @ w) if t + 1 < T else 0.0
            if value_lesion:
                delta = r                 # the value is SILENCED at the teaching signal (delta=r)
            else:
                delta = r + GAMMA * v_tp1 - v_t
            e = np.asarray(fused_eligibility_trace_decay(e, decay)) + X[t]
            if not no_learning:           # no_learning: freeze w (the value must be LEARNED)
                w = w + ALPHA * delta * e
            if trial >= n_trials - n_late:
                if t == onset:            # the RPE AT the cue onset (the gap-bridging prediction)
                    late_dCS.append(delta)
                if t == t_us:
                    late_dUS.append(abs(delta))
    # V on the canonical cue-anchored timeline: V(CS_onset) is tap=0, V(US) is tap=gap. The
    # value_transfer (the gap-bridging DV) reads the CUE-SPECIFIC TAP weights ALONE (NO bias): the
    # shared bias feature (active at every tap of every trial) accumulates the most under the
    # bias-free delta=r lesion and would WASH OUT the gap effect (V(CS)/V(US) -> 1) if included. The
    # cue-tap weight w[0] is the prediction the CUE specifically carries to the onset -- exactly the
    # quantity the dendrite-graded value supplies. (v_cs/v_us WITH the bias are reported for context.)
    bias = w[T]
    w_cs = float(w[0]); w_us = float(w[min(int(gap), T - 1)])
    v_cs = w_cs + bias
    v_us = w_us + bias
    value_transfer = float(max(w_cs, 0.0) / w_us) if w_us > 1e-9 else 0.0
    l_cs = float(np.mean(np.abs(late_dCS))) if late_dCS else 0.0
    l_us = float(np.mean(late_dUS)) if late_dUS else 0.0
    return dict(seed=int(seed), gap=int(gap), value_lesion=bool(value_lesion),
                no_learning=bool(no_learning), permuted=bool(permuted), lam=float(lam),
                v_cs=v_cs, v_us=v_us, w_cs=w_cs, w_us=w_us, bias=float(bias),
                value_transfer=value_transfer, late_dCS=l_cs, late_dUS=l_us)


def _collapse(transfer_full, transfer_lesion, *, frac=0.40):
    """The value is LOAD-BEARING iff lesioning it drops the value-transfer-to-CS (V(CS)/V(US)) to
    <= frac of the full-value transfer (and the full-value transfer is itself meaningfully > 0).
    Returns (collapses_bool, ratio_of_transfers)."""
    if transfer_full <= 1e-2:
        return False, float("nan")     # nothing transferred even with the value -> not testable
    ratio = max(transfer_lesion, 0.0) / transfer_full
    return bool(ratio <= frac), float(ratio)


def run_factorial(seeds, *, gap, T=24, n_trials=1500, lam=0.5, verbose=True):
    """The TRACE-vs-DELAY x value-ON-vs-LESION 2x2 factorial + the no-learning/permuted controls,
    multi-seed. The dependent variable is value_transfer = V(CS)/V(US) (the fraction of US-value
    that reached the CS across the gap). Returns the per-seed table + the aggregate gate verdict."""
    per_seed = {}
    kw = dict(T=T, n_trials=n_trials, lam=lam)
    for s in seeds:
        # the 2x2 factorial
        trace_full = run_trace(s, gap=gap, **kw)
        trace_les = run_trace(s, gap=gap, value_lesion=True, **kw)
        delay_full = run_trace(s, gap=0, **kw)
        delay_les = run_trace(s, gap=0, value_lesion=True, **kw)
        # anti-cheats (on the TRACE arm, where the value is load-bearing)
        trace_nolearn = run_trace(s, gap=gap, no_learning=True, **kw)
        trace_perm = run_trace(s, gap=gap, permuted=True, **kw)

        g2_collapse, g2_ratio = _collapse(trace_full["value_transfer"], trace_les["value_transfer"])
        g3_collapse, g3_ratio = _collapse(delay_full["value_transfer"], delay_les["value_transfer"])
        # the no-learning + permuted controls: the value-transfer to CS must be ~0 (no acquisition)
        nl_ref = max(trace_full["value_transfer"], 1e-6)
        nolearn_floor = bool(trace_nolearn["value_transfer"] <= 0.15 * nl_ref)
        perm_floor = bool(trace_perm["value_transfer"] <= 0.40 * nl_ref)

        per_seed[s] = dict(
            trace_full=trace_full, trace_lesion=trace_les,
            delay_full=delay_full, delay_lesion=delay_les,
            trace_nolearn=trace_nolearn, trace_permuted=trace_perm,
            g2_trace_value_collapses=g2_collapse, g2_ratio=g2_ratio,
            g3_delay_value_survives=(not g3_collapse), g3_ratio=g3_ratio,
            nolearn_collapses=nolearn_floor, permuted_collapses=perm_floor,
        )
        if verbose:
            print(f"  [seed {s}] gap={gap}  "
                  f"transfer V(CS)/V(US): TRACE full={trace_full['value_transfer']:.3f} "
                  f"lesion={trace_les['value_transfer']:.3f} (ratio {g2_ratio:.2f} -> "
                  f"collapses={g2_collapse}) | DELAY full={delay_full['value_transfer']:.3f} "
                  f"lesion={delay_les['value_transfer']:.3f} (ratio {g3_ratio:.2f} -> "
                  f"survives={not g3_collapse})")
            print(f"           controls: no-learning transfer={trace_nolearn['value_transfer']:.3f} "
                  f"(floor={nolearn_floor}) | permuted transfer={trace_perm['value_transfer']:.3f} "
                  f"(floor={perm_floor})")
    return per_seed


def _aggregate(per_seed, seeds, *, min_pass=None):
    n = len(seeds)
    maj = (n + 1) // 2 if min_pass is None else int(min_pass)
    g2 = sum(1 for s in seeds if per_seed[s]["g2_trace_value_collapses"])
    g3 = sum(1 for s in seeds if per_seed[s]["g3_delay_value_survives"])
    nl = sum(1 for s in seeds if per_seed[s]["nolearn_collapses"])
    pm = sum(1 for s in seeds if per_seed[s]["permuted_collapses"])
    go = (g2 >= maj and g3 >= maj and nl >= maj and pm >= maj)
    return dict(n=n, maj=maj, g2_count=g2, g3_count=g3, nolearn_count=nl, permuted_count=pm,
                GO=bool(go))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--gap", type=int, default=6, help="TRACE-arm CS-free gap in taps (DELAY=0)")
    ap.add_argument("--T", type=int, default=24, help="CSC timeline length (must exceed onset+gap)")
    ap.add_argument("--n-trials", type=int, default=1500)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="eligibility lambda (the ~0.2-2s reward-DA window; short so the bootstrap "
                         "and the trace dissociate). Default 0.5.")
    ap.add_argument("--gap-sweep", action="store_true",
                    help="sweep gap in {0,2,4,6,8} to show the trace-length dose-response")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    if args.gap_sweep:
        print("=== GAP-LENGTH DOSE-RESPONSE (value-lesion collapse grows with the trace gap) ===")
        sweep = {}
        for g in (0, 2, 4, 6, 8):
            ps = run_factorial(seeds, gap=g, T=args.T, n_trials=args.n_trials, lam=args.lam,
                               verbose=False)
            agg = _aggregate(ps, seeds)
            # mean lesion/full transfer ratio across seeds (the trace arm)
            ratios = [ps[s]["g2_ratio"] for s in seeds if np.isfinite(ps[s]["g2_ratio"])]
            mr = _st.mean(ratios) if ratios else float("nan")
            sweep[g] = dict(agg=agg, mean_trace_lesion_ratio=mr)
            print(f"  gap={g:2d}: TRACE value-lesion/full transfer ratio={mr:.2f}  "
                  f"G2(collapse {agg['g2_count']}/{agg['n']})  G3(survive {agg['g3_count']}/{agg['n']})")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(dict(mode="trace_conditioning_numpy_gap_sweep", seeds=seeds,
                               sweep={str(g): {"g2_count": v["agg"]["g2_count"],
                                               "g3_count": v["agg"]["g3_count"],
                                               "mean_trace_lesion_ratio": v["mean_trace_lesion_ratio"]}
                                      for g, v in sweep.items()}), f, indent=2, default=float)
            print(f"  wrote {args.out}")
        return

    print(f"##### SHORTCUT #9 TRACE-CONDITIONING numpy GATE (gap={args.gap}, T={args.T}, "
          f"lam={args.lam}, n_trials={args.n_trials}) #####")
    print("  The 2x2 factorial: arm (TRACE/DELAY) x value (full-bootstrap / value-LESION).")
    print("  DV = value_transfer V(CS)/V(US) (fraction of US-value that reached the CS across the gap).")
    print("  G2 = TRACE value-lesion COLLAPSES transfer (value load-bearing); G3 = DELAY survives.\n")
    per_seed = run_factorial(seeds, gap=args.gap, T=args.T, n_trials=args.n_trials, lam=args.lam,
                             verbose=True)
    agg = _aggregate(per_seed, seeds)

    print("\n" + "=" * 104)
    print("=== G2/G3 FACTORIAL TABLE (DV = value_transfer V(CS)/V(US), the gap-bridging prediction) ===")
    print("=" * 104)
    print(f"  {'seed':>5} | {'TRACE full':>10} {'TRACE les':>9} {'G2 ratio':>8} {'collapse':>8} | "
          f"{'DELAY full':>10} {'DELAY les':>9} {'G3 ratio':>8} {'survive':>7}")
    for s in seeds:
        p = per_seed[s]
        print(f"  {s:>5} | {p['trace_full']['value_transfer']:>10.3f} "
              f"{p['trace_lesion']['value_transfer']:>9.3f} "
              f"{p['g2_ratio']:>8.2f} {('Y' if p['g2_trace_value_collapses'] else 'n'):>8} | "
              f"{p['delay_full']['value_transfer']:>10.3f} "
              f"{p['delay_lesion']['value_transfer']:>9.3f} "
              f"{p['g3_ratio']:>8.2f} {('Y' if p['g3_delay_value_survives'] else 'n'):>7}")

    print("\n" + "=" * 104)
    print("=== GATE (validate-by-function: value load-bearing on TRACE, NOT on DELAY) ===")
    print("=" * 104)
    print(f"  (G2) TRACE value-lesion COLLAPSES transfer (<=0.40x): {agg['g2_count']}/{agg['n']}  "
          f"<- the HEADLINE (the value is load-bearing on the gap)")
    print(f"  (G3) DELAY value-lesion SURVIVES transfer            : {agg['g3_count']}/{agg['n']}  "
          f"<- the DISCRIMINATOR (the no-gap control does NOT need the value)")
    print(f"  (AC) NO-LEARNING collapses transfer                 : {agg['nolearn_count']}/{agg['n']}")
    print(f"  (AC) PERMUTED CS-US collapses transfer              : {agg['permuted_count']}/{agg['n']}")

    verdict = "GO" if agg["GO"] else "NEGATIVE"
    if agg["GO"]:
        note = ("the TRACE-arm value is LOAD-BEARING (lesion collapses the value-transfer to the CS) "
                "AND the DELAY-arm control does NOT need it (lesion survives) AND the no-learning + "
                "permuted controls collapse -> the task DISCRIMINATES 'needs the critic' (trace) from "
                "'immediate-reward-solvable' (delay). The numpy GATE for the #9 genuine close is GREEN "
                "-> lift onto the spiking limbic core.")
    else:
        why = []
        if agg["g2_count"] < agg["maj"]:
            why.append(f"G2 the TRACE value-lesion did NOT collapse the transfer ({agg['g2_count']}/{agg['n']})")
        if agg["g3_count"] < agg["maj"]:
            why.append(f"G3 the DELAY value-lesion DID collapse the transfer ({agg['g3_count']}/{agg['n']}) "
                       f"-> the task does NOT cleanly discriminate trace from delay")
        if agg["nolearn_count"] < agg["maj"]:
            why.append(f"the no-learning control did NOT collapse ({agg['nolearn_count']}/{agg['n']})")
        if agg["permuted_count"] < agg["maj"]:
            why.append(f"the permuted control did NOT collapse ({agg['permuted_count']}/{agg['n']})")
        note = "; ".join(why) + ". Characterize + (per the SURPASS workflow) the next move."

    print(f"\n=== TRACE-CONDITIONING numpy GATE VERDICT: {verdict} ===")
    print(f"=== {note} ===")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(mode="trace_conditioning_numpy_gate", gap=args.gap, T=args.T, lam=args.lam,
                           n_trials=args.n_trials, seeds=seeds,
                           per_seed={str(s): per_seed[s] for s in seeds},
                           aggregate=agg, verdict=verdict, verdict_note=note), f, indent=2, default=float)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
