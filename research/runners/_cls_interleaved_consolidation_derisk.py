"""CLS interleaved-consolidation de-risk -- does a FAST hippocampal store +
SLOW interleaved neocortical store resist CATASTROPHIC INTERFERENCE?

SERVES #66 KNOWLEDGE SCALE. Sequentially teaching new facts (set B) must NOT
overwrite old ones (set A) when set A is CONSOLIDATED into the slow cortical
store via interleaved SWR-replay -- vs a single-store baseline that DOES forget.

BIOLOGY (read + bound at research/biology/cls-interleaved-consolidation.md):
  Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995).
  Catalog N.14: repeated coordinated hippocampal reactivation "gradually
  transfers memory from HC-dependent (recent) to neocortex-dependent (remote)
  state." Buzsaki two-stage model: waking encoding writes the fast HC store;
  sleep SWR-replay drives the SAME content into a SLOW neocortex where late-LTP
  builds a durable trace. The escape from catastrophic interference is the
  ARCHITECTURE: A lives in the fast store, so it can be REPLAYED INTERLEAVED
  with new B experience, and a slow learner never forgets what it keeps seeing.
  The replay content the cortex consumes is what the GO swr-sequence-replay
  organ (_gap5_ecker_recurrent_replay.py) emits -- ordered, weight-borne
  reactivation of the stored episode. This runner de-risks the CLS FUNCTION.

MODEL. The neocortical store is a rate-coded three-factor Hebbian associator
(delta rule = error-gated plasticity: a synapse changes with pre-activity times
a post-side teaching signal -- a documented idealization of the slow cortex).
Neurons compute the readout (W @ x, WTA over category pools); the synapses W
carry the memory; plasticity is local. The hippocampal store is a fast,
pattern-separated buffer that holds set-A items and REPLAYS them (input pattern
re-instated -> cortex re-trained on the true A mapping).

ARMS (all share the SAME set A, SAME set B, SAME cortex_lr, SAME B-exposure):
  seq            : train A, then train B cortex-only (NO replay). Single-store
                   baseline. Expected: A forgotten (catastrophic).
  interleaved    : train A -> hippocampus stores A -> train B with A-replay
                   interleaved. Expected: A retained.
  exposure_match : seq + the SAME number of EXTRA steps as interleaved's replay
                   count, but spent on MORE B (not replay). Matches total updates
                   AND rules out "more training / lower effective lr" -> A still
                   forgotten. (closes the lr/exposure confound)
  shuffled       : interleaved, but replayed A-inputs are paired with WRONG
                   (deranged) A-targets. Matches total updates -> NO protection
                   (proves the CORRECT replay CONTENT is load-bearing, not
                   generic extra activity).

ANTI-CHEATS (wired into the printed VERDICT; a GO whose anti-cheats fail is a
NO-GO):
  AC1 replay-load-bearing : seq (replay lesioned) forgets A  (retention_seq LOW)
  AC2 content-specific    : shuffled replay does NOT protect  (retention_shuf LOW)
  AC3 exposure-matched    : more-B-not-replay does NOT protect (retention_exp LOW)
  AC4 new-learning-works  : B is learned in ALL arms          (B acc HIGH)
The lesion-load-bearing GO: A-retention CHANGES with replay ON (interleaved) and
the change VANISHES under the replay lesion (seq / shuffled / exposure_match).

Usage:
  numpy smoke (1 seed, foreground):
    python -m research.runners._cls_interleaved_consolidation_derisk --seed 42
  6-seed sweep is driven by the pool (see the reported pool_cmd).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.lab import attributable_to
from tools.verdict import GO, Verdict

# CLS constraint (research/biology/cls-interleaved-consolidation.md): the
# neocortical store must be SLOW. A fast cortical learner catastrophically
# overwrites regardless of replay (McClelland, McNaughton & O'Reilly 1995).
cortex_lr = 0.02


# ----------------------------------------------------------------------------
# Task: sparse random input patterns -> one of C category pools.
# Sets A and B SHARE the input feature space (overlapping features -> different
# categories), which is what makes sequential B-training interfere with A.
# ----------------------------------------------------------------------------
def make_items(rng, n_items, dim, n_cat, active):
    """n_items sparse binary input patterns each mapped to a random category."""
    items = []
    for _ in range(n_items):
        x = np.zeros(dim, dtype=np.float64)
        on = rng.choice(dim, size=active, replace=False)
        x[on] = 1.0
        c = int(rng.integers(n_cat))
        items.append((x, c))
    return items


def readout(W, x):
    """Cortex neurons compute pool activations; WTA over category pools."""
    return int(np.argmax(W @ x))


def accuracy(W, items):
    if not items:
        return 0.0
    return float(np.mean([readout(W, x) == c for (x, c) in items]))


def train_step(W, x, c, lr, n_cat):
    """One three-factor Hebbian (delta-rule) update: dW = lr * (target - out) x^T.

    target is the teacher-clamped one-hot over category pools; out is the
    softmax cortical response. Plasticity is local (pre-activity x times a
    post-side error). Returns the updated W in place.
    """
    a = W @ x
    a = a - a.max()
    p = np.exp(a)
    p = p / p.sum()
    t = np.zeros(n_cat)
    t[c] = 1.0
    W += lr * np.outer(t - p, x)


def train_phase(W, items, lr, n_cat, rng, reps):
    """reps sweeps over items in random order."""
    for _ in range(reps):
        order = rng.permutation(len(items))
        for i in order:
            x, c = items[i]
            train_step(W, x, c, lr, n_cat)


def run_seed(seed, args):
    rng = np.random.default_rng(seed)
    dim, n_cat, active = args.dim, args.n_cat, args.active
    A = make_items(rng, args.n_a, dim, n_cat, active)
    B = make_items(rng, args.n_b, dim, n_cat, active)

    # Hippocampal store of A: the fast buffer re-instates the TRUE A mapping.
    # (In the full system these samples are what the GO SWR organ replays.)
    hippo_A = list(A)
    # Shuffled (deranged) A targets for the content-specificity control.
    cats = [c for (_, c) in A]
    sh = rng.permutation(len(cats))
    # ensure a genuine derangement of labels where possible
    if len(cats) > 1:
        for _ in range(8):
            if all(sh[i] != i for i in range(len(cats))):
                break
            sh = rng.permutation(len(cats))
    hippo_A_shuf = [(A[i][0], cats[sh[i]]) for i in range(len(A))]

    lr = args.cortex_lr
    reps_a = args.reps_a
    reps_b = args.reps_b

    def fresh_cortex():
        return np.zeros((n_cat, dim), dtype=np.float64)

    def phase_a():
        W = fresh_cortex()
        train_phase(W, A, lr, n_cat, np.random.default_rng(seed + 1), reps_a)
        return W

    # Baseline A accuracy after phase A (before any B) -- shared reference.
    W0 = phase_a()
    acc_A_baseline = accuracy(W0, A)

    def train_B_with_replay(W, replay_items, replay_ratio, r_rng):
        """Interleave replay_items among the B stream. For every B item, with
        prob replay_ratio also present one replayed item. Returns replay count."""
        n_replay = 0
        for _ in range(reps_b):
            order = r_rng.permutation(len(B))
            for i in order:
                x, c = B[i]
                train_step(W, x, c, lr, n_cat)
                if replay_items is not None and r_rng.random() < replay_ratio:
                    j = int(r_rng.integers(len(replay_items)))
                    rx, rc = replay_items[j]
                    train_step(W, rx, rc, lr, n_cat)
                    n_replay += 1
        return n_replay

    # --- interleaved (true CLS): replay TRUE A content -----------------------
    W = W0.copy()
    n_rep = train_B_with_replay(W, hippo_A, args.replay_ratio,
                                np.random.default_rng(seed + 2))
    ret_interleaved = accuracy(W, A) / max(acc_A_baseline, 1e-9)
    accB_interleaved = accuracy(W, B)

    # --- seq (single-store baseline, replay LESIONED) ------------------------
    W = W0.copy()
    train_B_with_replay(W, None, 0.0, np.random.default_rng(seed + 2))
    ret_seq = accuracy(W, A) / max(acc_A_baseline, 1e-9)
    accB_seq = accuracy(W, B)

    # --- shuffled replay (content-specificity control) -----------------------
    W = W0.copy()
    train_B_with_replay(W, hippo_A_shuf, args.replay_ratio,
                        np.random.default_rng(seed + 2))
    ret_shuf = accuracy(W, A) / max(acc_A_baseline, 1e-9)
    accB_shuf = accuracy(W, B)

    # --- exposure-matched (extra steps -> MORE B, not replay) ----------------
    # match total update count: seq + n_rep extra B updates.
    W = W0.copy()
    er = np.random.default_rng(seed + 2)
    train_B_with_replay(W, None, 0.0, er)
    for _ in range(n_rep):
        i = int(er.integers(len(B)))
        x, c = B[i]
        train_step(W, x, c, lr, n_cat)
    ret_exp = accuracy(W, A) / max(acc_A_baseline, 1e-9)
    accB_exp = accuracy(W, B)

    return dict(
        seed=seed, acc_A_baseline=acc_A_baseline, n_replay=n_rep,
        ret_interleaved=ret_interleaved, accB_interleaved=accB_interleaved,
        ret_seq=ret_seq, accB_seq=accB_seq,
        ret_shuf=ret_shuf, accB_shuf=accB_shuf,
        ret_exp=ret_exp, accB_exp=accB_exp,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="multi-seed sweep; overrides --seed when given")
    # Defaults put the SLOW cortical store OVER CAPACITY (60 items in 24 dims):
    # this is the regime where a single-store learner is FORCED to reuse weights
    # and therefore catastrophically forgets A -- i.e. where the CLS protection
    # is meaningful. In an under-capacity regime a lone linear associator does
    # not forget and the de-risk is vacuous (measured: no forgetting at dim=100).
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--n-cat", type=int, default=8)
    ap.add_argument("--active", type=int, default=8,
                    help="active input units per pattern (pattern separation)")
    ap.add_argument("--n-a", type=int, default=30, help="set-A items")
    ap.add_argument("--n-b", type=int, default=30, help="set-B items")
    ap.add_argument("--cortex-lr", type=float, default=cortex_lr,
                    help="SLOW neocortical rate (CLS constraint; see biology)")
    ap.add_argument("--reps-a", type=int, default=120)
    ap.add_argument("--reps-b", type=int, default=120)
    ap.add_argument("--replay-ratio", type=float, default=1.0,
                    help="P(replay one A item | per B item)")
    # GO thresholds (pre-registered)
    ap.add_argument("--retain-hi", type=float, default=0.80,
                    help="interleaved retention must be >= this (frac of baseline)")
    ap.add_argument("--forget-hi", type=float, default=0.60,
                    help="lesioned arms retention must be <= this")
    ap.add_argument("--b-hi", type=float, default=0.80,
                    help="new-learning (B acc) must be >= this in all arms")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    rows = [run_seed(s, args) for s in seeds]

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    ri = mean("ret_interleaved")
    rs = mean("ret_seq")
    rsh = mean("ret_shuf")
    re_ = mean("ret_exp")
    bi = mean("accB_interleaved")
    bs = mean("accB_seq")
    bsh = mean("accB_shuf")
    be = mean("accB_exp")

    print("=" * 70)
    print(f"CLS interleaved-consolidation de-risk | seeds={seeds}")
    print(f"  baseline A acc (post-phase-A) : {mean('acc_A_baseline'):.3f}")
    print(f"  avg replay steps interleaved  : {mean('n_replay'):.0f}")
    print("-" * 70)
    print("  arm             A-retention(/baseline)   B-acc(new learning)")
    print(f"  interleaved     {ri:6.3f}                   {bi:6.3f}")
    print(f"  seq (no-replay) {rs:6.3f}                   {bs:6.3f}")
    print(f"  shuffled-replay {rsh:6.3f}                   {bsh:6.3f}")
    print(f"  exposure-match  {re_:6.3f}                   {be:6.3f}")
    print("-" * 70)

    # ---- anti-cheats (each must hold) --------------------------------------
    ac1 = rs <= args.forget_hi                 # replay lesion -> forgets A
    ac2 = rsh <= args.forget_hi                # shuffled -> no protection
    ac3 = re_ <= args.forget_hi                # more-B -> no protection
    ac4 = min(bi, bs, bsh, be) >= args.b_hi    # B learned in all arms
    protect = ri >= args.retain_hi             # interleaved protects A
    # lesion-load-bearing: interleaved must beat EVERY lesioned arm clearly
    margin = ri - max(rs, rsh, re_)
    ac5 = margin >= 0.20

    print(f"  AC1 replay-load-bearing (seq forgets, <= {args.forget_hi:.2f}) : "
          f"{'PASS' if ac1 else 'FAIL'} (seq ret={rs:.3f})")
    print(f"  AC2 content-specific (shuffled no-protect)          : "
          f"{'PASS' if ac2 else 'FAIL'} (shuf ret={rsh:.3f})")
    print(f"  AC3 exposure-matched (more-B no-protect)            : "
          f"{'PASS' if ac3 else 'FAIL'} (exp ret={re_:.3f})")
    print(f"  AC4 new-learning-works (B acc >= {args.b_hi:.2f} all arms)  : "
          f"{'PASS' if ac4 else 'FAIL'} (minB={min(bi,bs,bsh,be):.3f})")
    print(f"  AC5 lesion-load-bearing margin >= 0.20              : "
          f"{'PASS' if ac5 else 'FAIL'} (margin={margin:.3f})")
    print(f"  PROTECT interleaved retains A (>= {args.retain_hi:.2f})       : "
          f"{'PASS' if protect else 'FAIL'} (ret={ri:.3f})")
    print("-" * 70)

    # ---- CREDIT ATTRIBUTION (whose is the retained-A?) ---------------------
    # Treatment = interleaved (replay ON); control = seq (replay LESIONED, the
    # single-store baseline). If most of the retention is ALSO present in the
    # no-replay control, replay is not what protects A. (gap#5 lesson: measure
    # both arms AND subtract them.)
    frac = attributable_to("A-retention -> interleaved replay", ri, rs)
    ac6 = frac is not None and frac >= 0.50  # replay owns >= half the retention
    print(f"  AC6 replay owns >= 50% of retained-A                : "
          f"{'PASS' if ac6 else 'FAIL'} "
          f"(frac={('n/a' if frac is None else f'{frac:.3f}')})")
    print("-" * 70)

    # ---- EARNED verdict: preconditions travel with it (tools.verdict.Verdict) --
    # The anti-cheats ARE the preconditions/controls; the hypothesis DECIDED is
    # whether interleaved retention clears retain_hi. If any control fails to
    # behave (a lesion that does not lesion, an unlearned B, unexecuted replay),
    # decide() returns UNDEFINED -- an uninterpretable run, never a negative.
    v = Verdict("CLS interleaved consolidation defeats catastrophic interference")
    v.disabled("spiking neocortex",
               why="rate-coded delta-rule associator = documented idealization of the SLOW "
                   "cortical store; this de-risks the CLS FUNCTION, not a spiking cortex")
    # precondition: A must be learned to ceiling before B, else 'forgetting' is undefined.
    # (This is why acc_A_baseline is at ceiling BY DESIGN -- a precondition, not the
    #  discriminating metric; the discriminating quantity is the retention spread below.)
    v.require("A learned to ceiling before B (precondition)", measured=mean("acc_A_baseline"),
              expect=lambda a: a >= 0.99)
    v.require("replay branch actually executed", measured=mean("n_replay"),
              expect=lambda n: n > 0)
    v.floor("AC4 new-learning: min B-acc over arms beats floor",
            measured=min(bi, bs, bsh, be), floor=args.b_hi)
    v.require("AC1 seq (replay-lesioned) forgets A", measured=rs,
              expect=lambda r: r <= args.forget_hi)
    v.require("AC2 shuffled replay does NOT protect", measured=rsh,
              expect=lambda r: r <= args.forget_hi)
    v.require("AC3 exposure-matched (more-B) does NOT protect", measured=re_,
              expect=lambda r: r <= args.forget_hi)
    v.control("AC5 interleaved vs best lesion arm", treatment=ri,
              control=max(rs, rsh, re_), min_separation=0.20)
    v.require("AC6 replay owns >= 50% of retained-A",
              measured=(frac if frac is not None else -1.0), expect=lambda f: f >= 0.50)
    decided = v.decide(go=protect, verbose=False)
    verdict = decided["status"]
    go = verdict == GO
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(seeds=seeds, rows=rows, backend="numpy",
                 means=dict(ret_interleaved=ri, ret_seq=rs, ret_shuf=rsh,
                            ret_exp=re_, accB_interleaved=bi, accB_seq=bs,
                            accB_shuf=bsh, accB_exp=be, margin=margin),
                 attribution_frac=frac,
                 anti_cheats=dict(ac1=ac1, ac2=ac2, ac3=ac3, ac4=ac4, ac5=ac5,
                                  ac6=ac6, protect=protect),
                 verdict=verdict, status=verdict,
                 preconditions=decided["preconditions"],
                 disabled_processes=decided["disabled_processes"]), indent=2))
        print(f"wrote {args.out}")

    sys.exit(0 if go else 1)


if __name__ == "__main__":
    main()
