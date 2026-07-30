#!/usr/bin/env python3
"""gap#5 NEURAL READER, mechanism 3: sequence selectivity in the WEIGHTS (STDP resonance).

TWO PRIOR MECHANISMS ARE ELIMINATED ON MEASURED GROUNDS (2026-07-29):
  * conduction delays  -- clean discrimination needs 63-125 ms of spread; axonal delay is ~1-30 ms;
  * slow synapses      -- physiological, but only ~12% separation AND it MISRANKS (a travelling-widening
                          front reads MORE directional than clean travel).
Both tried to ALIGN arrivals in time. This does not: the reader learns the EXPECTED ORDER in its own
recurrent weights via forward-asymmetric STDP, so a forward sweep RESONATES with the learned recurrence and
a reverse sweep does not. Precedent: the Mehta forward-asymmetric mechanism is already 6-seed GO in this
project for growing the replay band itself (2026-07-25-gap5-learned-band-emergence-...-6seed-GO.md).

WHY THIS FILE IS STRUCTURED CONTROLS-FIRST. A previous attempt at this mechanism was VOID for two reasons,
both mine: (a) it compared amplification through TWO separately-normalised recurrence matrices, so the ratio
was dominated by their scale rather than by the input's order; (b) its "lesion" divided an arm by ITSELF and
so returned 1.000 and could not fail. Fixes, by construction:
  * ONE recurrence matrix. The ratio varies the INPUT (forward vs reverse), never the matrix -- no
    cross-matrix normalisation can leak in.
  * The LESION is a genuinely DIFFERENT quantity: recurrence trained on ORDER-DESTROYED (time-shuffled)
    wake activity. It cannot be reached by editing one argument of the treatment.
  * tools/lab.py is imported AT THE TOP and the engagement/lever checks run BEFORE any score is believed.

ENGAGEMENT (the thing six void arms this session lacked): asymmetry(R) = ||R-R^T||/||R||. Forward-asymmetric
STDP MUST produce an asymmetric matrix; if asymmetry ~ 0 the mechanism never engaged and every ratio below is
meaningless. Reported first, and the run refuses to score if it is degenerate.

    .venv/bin/python -m research.runners._gap5_reader_stdp_resonance_derisk --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import numpy as np

from tools.lab import lever, void_if, undefined_if_empty


# ---------------------------------------------------------------- stimuli (controls defined FIRST)
def sweep(N, T, width, rng, direction=+1, widen=False, static=False):
    """One generator, four conditions, so the conditions cannot differ by accident.

    static=True is THE decisive control (fixed centre, growing width, NO travel): a reader that is genuinely
    reading TRAVEL must be neutral on it. A previous rung's "spread" control travelled while widening and so
    could not test that at all.
    """
    F = np.zeros((T, N))
    for t in range(T):
        c = (N - 1) / 2.0 if static else (t / (T - 1) if direction > 0 else 1 - t / (T - 1)) * (N - 1)
        w = width * (1 + 4 * t / (T - 1)) if (widen or static) else width
        p = np.exp(-0.5 * ((np.arange(N) - c) / w) ** 2)
        F[t] = (rng.random(N) < np.clip(p, 0, 1) * 0.9).astype(float)
    return F


def learn_tuning(F, n_read, rng, lr=0.05):
    """Reader tuning: a random position SEED plus k-WTA Hebbian.

    ⛔ THE ORIGINAL DOCSTRING HERE WAS FALSE AND ITS CLAIM IS RETRACTED (2026-07-29). It read "Reader
    ACQUIRES tuning by k-WTA Hebbian. Handed nothing -- that is the whole point", and that was quoted
    repeatedly as evidence the read-out architecture was de-risked. MEASURED: selectivity is 9.376 /
    9.363 / 9.334 at **lr=0** (seeds 42/43/44) versus 9.177 / 9.163 / 9.157 when trained -- i.e. the
    ENTIRE ~9.1x peak-to-mean is supplied by the `seeds`/`+= 0.05` bump below, BEFORE any learning, and
    the Hebbian loop makes it slightly WORSE on 3/3 seeds.

    So each reader IS handed a place field. This function is a legitimate way to obtain tuned readers
    for testing a DOWNSTREAM read-out, and every order-reading result in this arc used hand-set timing
    and said so. It is NOT evidence that tuning can be ACQUIRED -- that question is open, and any claim
    about it must carry an lr=0 arm (see `.claude/skills/verify-go/SKILL.md`).
    """
    N = F.shape[1]
    W = rng.uniform(0, 0.01, size=(n_read, N))
    seeds = rng.permutation(N)[:n_read]
    for j in range(n_read):
        W[j, max(0, seeds[j] - 2):seeds[j] + 3] += 0.05
    for t in range(F.shape[0]):
        a = W @ F[t]
        if a.max() <= 0:
            continue
        w = int(np.argmax(a))
        W[w] += lr * F[t] * (1.0 - W[w])
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)


def learn_recurrence(act, tau, lr=0.02):
    """Forward-asymmetric STDP: post(now) x pre(recent trace) -> pre-before-post strengthens pre->post."""
    n, T = act.shape
    R = np.zeros((n, n))
    trace = np.zeros(n)
    decay = np.exp(-1.0 / tau)
    for t in range(T):
        trace = trace * decay + act[:, t]
        R += lr * np.outer(act[:, t], trace)
    np.fill_diagonal(R, 0.0)
    return R


def resonance(act, R, gain, leak=0.7):
    """Total positive recurrent drive the learned weights generate for THIS input.

    ONE matrix, so the comparison is purely about the input's ORDER. High when the input's order matches the
    order stored in R (resonance); low when it does not.
    """
    n, T = act.shape
    h = np.zeros(n)
    tot = 0.0
    for t in range(T):
        rec = np.maximum(gain * (R @ h), 0.0)
        h = leak * h + act[:, t] + rec
        tot += float(rec.sum())
    return tot


def predictive_alignment(act, R, leak=0.7):
    """TIMING-SENSITIVE read: does the recurrent drive POINT AT the cells that fire NEXT?

    Replaces the summed-drive read, which was direction-BLIND: a reverse sweep activates the same connected
    pairs as forward, merely in reverse temporal order, so the SUM is identical (measured FWD/REV = 0.843 at
    0.3%-matched activity). Alignment asks a timing question instead -- if R stores the forward order, then
    for a forward input the recurrent drive at t should anticipate the input at t+1.

    Cosine is used deliberately: it is scale-free, so this ALSO removes the activity-volume confound that
    invalidated the static/shuffle controls (their reader activity was 3.4x forward's).
    """
    n, T = act.shape
    h = np.zeros(n)
    cos_sum = 0.0
    cnt = 0
    for t in range(T - 1):
        rec = np.maximum(R @ h, 0.0)
        h = leak * h + act[:, t]
        nxt = act[:, t + 1]
        nr = np.linalg.norm(rec)
        nn = np.linalg.norm(nxt)
        if nr > 1e-9 and nn > 1e-9:
            cos_sum += float(rec @ nxt / (nr * nn))
            cnt += 1
    return cos_sum / cnt if cnt else 0.0


def pairwise_order_vote(act, pref, lag=3):
    """LOCAL pairwise order detection -- the fix for the delay-line physiology problem.

    The delay-line read worked (2.2x separation) but needed 63-125 ms of spread, because it tried to align the
    WHOLE population to one common time -- a span equal to the sweep duration. A global ORDER statistic does
    not need that: it can be assembled from many LOCAL pairwise comparisons, and ADJACENT cells in a 250 ms
    sweep over ~40 reader cells peak only ~6 ms apart. That IS within axonal range (~1-30 ms).

    Each adjacent pair (by LEARNED preferred position) votes: did the lower-position cell lead the
    higher-position one by `lag`? Coincidence of (earlier cell delayed by lag) with (later cell now) is a
    standard pairwise sequence detector. Votes are summed, then normalised by the number of active pairs, so
    the read is scale-free and cannot be confounded by activity volume (the flaw that invalidated the
    summed-drive controls).
    """
    order = np.argsort(pref)
    fwd = rev = 0.0
    npair = 0
    for k in range(len(order) - 1):
        lo, hi = order[k], order[k + 1]
        a_lo, a_hi = act[lo], act[hi]
        if a_lo.sum() <= 0 or a_hi.sum() <= 0:
            continue
        npair += 1
        # forward detector: lo delayed by lag, coincident with hi
        fwd += float(np.dot(np.roll(a_lo, lag), a_hi))
        # reverse detector: hi delayed by lag, coincident with lo
        rev += float(np.dot(np.roll(a_hi, lag), a_lo))
    if not npair:
        return 0.0, 0.0, 0
    return fwd / npair, rev / npair, npair


def run(seed, N=200, T=120, n_read=40, width=6.0, tau=6.0, gain=1.0, verbose=False):
    rng = np.random.default_rng(seed)
    wake = sweep(N, T, width, rng, direction=+1)
    W = learn_tuning(wake, n_read, rng)
    act_wake = W @ wake.T

    R = learn_recurrence(act_wake, tau)
    # LESION: a genuinely different quantity -- recurrence trained on ORDER-DESTROYED wake activity.
    R_les = learn_recurrence(act_wake[:, rng.permutation(T)], tau)

    # ---- ENGAGEMENT FIRST. Nothing below is believable if the STDP produced no asymmetry.
    asym = float(np.linalg.norm(R - R.T) / (np.linalg.norm(R) + 1e-12))
    asym_les = float(np.linalg.norm(R_les - R_les.T) / (np.linalg.norm(R_les) + 1e-12))
    if void_if(asym < 0.05, "STDP produced a near-symmetric R (asym=%.4f): the mechanism never engaged" % asym):
        return None
    if verbose:
        lever("R asymmetry (learned vs order-destroyed)", round(asym_les, 4), round(asym, 4), required=False)

    scores = {}
    for name, kw in (("forward", dict(direction=+1)), ("reverse", dict(direction=-1)),
                     ("widen_travel", dict(direction=+1, widen=True)), ("static_widen", dict(static=True))):
        a = W @ sweep(N, T, width, rng, **kw).T
        scores[name] = resonance(a, R, gain)
        scores[name + "_align"] = predictive_alignment(a, R)
        _pf, _pr, _np = pairwise_order_vote(a, np.argmax(W, axis=1))
        scores[name + "_pair"] = _pf / (_pr + 1e-9)
        scores[name + "_npair"] = _np
        scores[name + "_spikes"] = float(a.sum())

    a_fwd = W @ sweep(N, T, width, rng, direction=+1).T
    scores["forward_LESION"] = resonance(a_fwd, R_les, gain)
    scores["forward_LESION_align"] = predictive_alignment(a_fwd, R_les)
    a_shuf = W @ sweep(N, T, width, rng, direction=+1)[rng.permutation(T)].T
    scores["shuffle"] = resonance(a_shuf, R, gain)
    scores["shuffle_align"] = predictive_alignment(a_shuf, R)

    base = scores["reverse"] + 1e-9
    out = {"seed": seed, "asym": round(asym, 4), "asym_lesion": round(asym_les, 4),
           "fwd_over_rev": round(scores["forward"] / base, 3),
           "widen_over_rev": round(scores["widen_travel"] / base, 3),
           "static_over_rev": round(scores["static_widen"] / base, 3),
           "lesion_over_rev": round(scores["forward_LESION"] / base, 3),
           "shuffle_over_rev": round(scores["shuffle"] / base, 3),
           "fwd_spikes": round(scores["forward_spikes"], 1),
           "align_fwd": round(scores["forward_align"], 4),
           "align_rev": round(scores["reverse_align"], 4),
           "align_static": round(scores["static_widen_align"], 4),
           "align_lesion": round(scores["forward_LESION_align"], 4),
           "align_shuffle": round(scores["shuffle_align"], 4),
           "pair_fwd": round(scores["forward_pair"], 3),
           "pair_rev": round(scores["reverse_pair"], 3),
           "pair_static": round(scores["static_widen_pair"], 3),
           "npair": scores["forward_npair"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    rows = [r for r in (run(s, tau=a.tau, gain=a.gain, verbose=(i == 0))
                        for i, s in enumerate(a.seeds)) if r]
    n = undefined_if_empty("stdp-resonance", len(rows), len(rows), len(a.seeds))
    if not n:
        return 1

    print("\n%-8s %-8s %-9s %-13s %-14s %-13s %s"
          % ("seed", "asym", "FWD/REV", "widen/REV", "STATIC/REV", "LESION/REV", "shuffle/REV"))
    for r in rows:
        print("%-8d %-8.3f %-9.3f %-13.3f %-14.3f %-13.3f %.3f"
              % (r["seed"], r["asym"], r["fwd_over_rev"], r["widen_over_rev"],
                 r["static_over_rev"], r["lesion_over_rev"], r["shuffle_over_rev"]))

    m = {k: float(np.mean([r[k] for r in rows])) for k in
         ("fwd_over_rev", "widen_over_rev", "static_over_rev", "lesion_over_rev", "shuffle_over_rev", "asym")}
    print("\nmean  asym=%.3f  FWD/REV=%.3f  widen=%.3f  STATIC=%.3f  LESION=%.3f  shuffle=%.3f"
          % (m["asym"], m["fwd_over_rev"], m["widen_over_rev"], m["static_over_rev"],
             m["lesion_over_rev"], m["shuffle_over_rev"]))

    # GO needs BOTH: a real forward preference, AND every non-directional/destroyed control near neutral.
    sel = m["fwd_over_rev"] > 1.5
    ctrl = (abs(m["static_over_rev"] - 1.0) < 0.25 and abs(m["lesion_over_rev"] - 1.0) < 0.25
            and abs(m["shuffle_over_rev"] - 1.0) < 0.25)
    print("\nVERDICT: %s  (selectivity FWD/REV>1.5: %s | controls near 1.0: %s)"
          % ("GO" if (sel and ctrl) else "NO-GO", sel, ctrl))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"rows": rows, "mean": m, "go": bool(sel and ctrl), "argv": sys.argv},
                  open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
