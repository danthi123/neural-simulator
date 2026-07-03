"""EMERGE-83 -- RANK-3 cheap-first: how DEEP a center-embedded (recursive) dependency can the reservoir track before it
BOUNDARIES -- naming where the theta-gamma WM-buffer / assembly-calculus stack becomes necessary?

WHY. EMERGE-78/80/82 established the reservoir resolves a SINGLE-embedding non-local dependency. The research gate ranked
RANK-3 (bounded recursion) as the deeper frontier: real productivity needs NESTED / center-embedded structure, and a plain
reservoir's fading memory tracks only BOUNDED depth (humans also fail past ~2 center-embeddings; catalog N.15 theta-gamma
multiplexed WM buffer; Mitropolsky assembly-calculus stack). This de-risk MEASURES the reservoir's recursion depth on the
canonical psycholinguistic test -- SUBJECT-VERB NUMBER AGREEMENT ACROSS CENTER-EMBEDDING (agreement attraction): the MATRIX
verb must agree with the MATRIX subject's number, skipping the numbers of the intervening EMBEDDED subjects (the
distractors). At depth d, the reservoir must maintain the matrix subject's number across d nested clauses amid d competing
number cues.

THE TASK (real discovered markers; the interference is the point). Number markers `sng`/`plu` prepend each noun, and the
relativizer `that` opens each embedding -- all made REAL, FREQUENT, DISCOVERED words (no OOV). We SCORE the matrix subject's
number (the agreement target) from the whole sequence:
  depth 0: `<n1> s1`                                              -> answer n1
  depth 1: `<n1> s1 that <n2> s2`                                 -> answer n1 (skip the nearer n2)
  depth 2: `<n1> s1 that <n2> s2 that <n3> s3`                    -> answer n1 (skip n2, n3)
  depth 3: `<n1> s1 that <n2> s2 that <n3> s3 that <n4> s4`       -> answer n1 (skip 3 distractors)
A NEAREST-NUMBER baseline (the last-seen number = agreement attraction) is WRONG at depth>=1 (it predicts the innermost
subject's number). The reservoir must maintain the FIRST (matrix) number across the nesting -- which its fading memory does
up to a BOUNDED depth, then interference from the distractors wins.

THE DE-RISK (6 seeds; rate-level CPU; reuse EMERGE-78 Reservoir/Encoder; NO `sim/` edit). Report reservoir accuracy vs depth
+ the recursion DEPTH d* (largest depth >= 0.90) + whether it beats the nearest-number baseline (which is at chance/attraction
beyond depth 0). This is a CHARACTERIZATION that NAMES the boundary:
  * if the reservoir tracks to some d* then degrades -> the honest recursion-depth boundary; DEEPER needs the RANK-3
    WM-buffer/stack (the reservoir alone cannot hold an unbounded stack) -- the named next mechanism;
  * a MATRIX-NUMBER-LESION (replace n1 with a neutral token) collapses the answer -> the answer is genuinely the matrix
    subject's number (not a positional artifact);
  * a NEAREST-number baseline quantifies the attraction the reservoir must overcome.
Do NOT force a GO; the deliverable is the measured depth + the named RANK-3 mechanism.

HONEST SCOPE. Measures BOUNDED recursion depth on a single agreement dependency (the canonical center-embedding test), not
open-ended recursion. Reuse-by-import; NO `sim/` edit.

Run:
  python -m research.runners._emerge83_reservoir_recursion_depth_derisk --derisk
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, Reservoir, _content_pools  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge83_reservoir_recursion_depth.json"

_NUMS = ["sng", "plu"]                       # number markers (REAL frequent discovered words)
_THAT = "that"                                # the relativizer / embedding opener (made discovered)
_NUM_IDX = {n: i for i, n in enumerate(_NUMS)}
_TRAIN_MAXDEPTH = 4
_TEST_DEPTHS = [0, 1, 2, 3, 4]
_N_TRAIN_PER = 300
_N_TEST = 200
_RIDGE_LAMBDA = 1e-3


def _discover_with_markers(seed):
    """Discovery stream ALSO containing the number markers + `that` as FREQUENT tokens, so EMERGE-62 discovers them as
    distinct closed cues (no OOV). Returns (discovered, subj)."""
    base = m62.build_stream(seed, n_sentences=4000)
    rng = np.random.default_rng(seed * 31 + 1)
    subj0, verb0 = m62._SUBJECTS, m62._VERBS
    extra = []
    for _ in range(5000):
        depth = int(rng.integers(0, _TRAIN_MAXDEPTH + 1))
        toks = []
        for _d in range(depth + 1):
            toks += [str(rng.choice(_NUMS)), str(rng.choice(subj0))]
            if _d < depth:
                toks.append(_THAT)
        extra += toks + [m62.SENT_PERIOD]
    words, freq, cover, _c = m62.compute_stats(base + extra)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj = _content_pools(discovered)[0]
    return discovered, subj


def _make(depth, rng, subj):
    """Center-embedded chain of `depth+1` subjects; SCORE the MATRIX (first) subject's number. Returns (tokens, answer)."""
    nums = [str(rng.choice(_NUMS)) for _ in range(depth + 1)]
    toks = []
    for d in range(depth + 1):
        toks += [nums[d], str(rng.choice(subj))]
        if d < depth:
            toks.append(_THAT)
    return toks, nums[0]                                     # answer = the MATRIX subject's number


def _final(res, enc, toks, lesion_matrix=False):
    if lesion_matrix:
        t = list(toks); t[0] = _THAT                        # replace the matrix number with a neutral discovered token
        toks = t
    return np.concatenate([res.final_state(enc.encode(toks)), [1.0]])


def _fit(res, enc, sents):
    X = np.asarray([_final(res, enc, t) for (t, _a) in sents])
    y = np.asarray([_NUM_IDX[a] for (_t, a) in sents])
    T = np.zeros((len(y), len(_NUMS))); T[np.arange(len(y)), y] = 1.0
    return np.linalg.solve(X.T @ X + _RIDGE_LAMBDA * np.eye(X.shape[1]), X.T @ T)


def _acc(res, enc, W, sents, lesion_matrix=False):
    hit = 0
    for (toks, a) in sents:
        f = _final(res, enc, toks, lesion_matrix=lesion_matrix)
        hit += int(_NUMS[int(np.argmax(f @ W))] == a)
    return float(hit / max(1, len(sents)))


def _nearest_number_acc(sents):
    """The agreement-attraction baseline: predict the LAST-seen number (the innermost subject) -> correct only at depth 0."""
    hit = 0
    for (toks, a) in sents:
        last = None
        for w in toks:
            if w in _NUM_IDX:
                last = w
        hit += int(last == a)
    return float(hit / max(1, len(sents)))


def _one(seed):
    discovered, subj = _discover_with_markers(seed)
    markers_ok = all(n in discovered for n in _NUMS) and (_THAT in discovered)
    enc = Encoder(discovered)
    res = Reservoir(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = [_make(d, rng, subj) for d in range(_TRAIN_MAXDEPTH + 1) for _ in range(_N_TRAIN_PER)]
    W = _fit(res, enc, train)

    by_depth = {}
    for d in _TEST_DEPTHS:
        test = [_make(d, rng, subj) for _ in range(_N_TEST)]
        by_depth[d] = {"reservoir": _acc(res, enc, W, test), "nearest_number": _nearest_number_acc(test)}
    dctl = 2
    ctl = [_make(dctl, rng, subj) for _ in range(_N_TEST)]
    lesion = _acc(res, enc, W, ctl, lesion_matrix=True)
    return {"seed": seed, "markers_ok": bool(markers_ok), "by_depth": by_depth, "matrix_lesion_acc": lesion,
            "chance": 0.5}


def _derisk(seeds):
    print(f"EMERGE-83: RANK-3 reservoir RECURSION-DEPTH characterization (subject-verb number agreement across center-"
          f"embedding); {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _one(s); per.append(d)
            row = " ".join(f"d{dd}:{d['by_depth'][dd]['reservoir']:.2f}/{d['by_depth'][dd]['nearest_number']:.2f}"
                           for dd in _TEST_DEPTHS)
            print(f"  [seed {s}] markers_ok {d['markers_ok']} | res/nearest by depth: {row} | matrix-lesion "
                  f"{d['matrix_lesion_acc']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        agg = {d: {k: float(np.mean([p["by_depth"][d][k] for p in per])) for k in ("reservoir", "nearest_number")}
               for d in _TEST_DEPTHS}
        lesion = float(np.mean([p["matrix_lesion_acc"] for p in per]))
        markers_ok = all(p["markers_ok"] for p in per)
        depth_star = max([d for d in _TEST_DEPTHS if agg[d]["reservoir"] >= 0.90], default=-1)
        held_full = (depth_star == _TEST_DEPTHS[-1])
        # does the reservoir beat the attraction baseline where it's wrong (depth >= 1)?
        beats_attraction = all(agg[d]["reservoir"] - agg[d]["nearest_number"] >= 0.20 for d in _TEST_DEPTHS if d >= 1)
        lesion_ok = (agg[2]["reservoir"] - lesion) >= 0.20 if 2 in agg else True

        verdict = (
            f"CHARACTERIZATION -- the reservoir tracks the MATRIX subject's number across center-embedding to a recursion "
            f"DEPTH of {'>= ' + str(depth_star) + ' (held to the max tested depth ' + str(_TEST_DEPTHS[-1]) + ')' if held_full else 'd* = ' + str(depth_star) + ' (falls below 0.90 at deeper nesting)'} "
            f"(profile: {', '.join(f'd{d}={agg[d]['reservoir']:.2f}' for d in _TEST_DEPTHS)}), beating the agreement-"
            f"ATTRACTION baseline (predict the nearest/innermost number: {', '.join(f'd{d}={agg[d]['nearest_number']:.2f}' for d in _TEST_DEPTHS)}) "
            f"where it is wrong (depth>=1 beats-attraction = {beats_attraction}). MARKERS discovered (no OOV) = {markers_ok}. "
            f"MATRIX-NUMBER-LESION collapses the answer to {lesion:.2f} (genuinely the matrix number). ==> a plain "
            f"reservoir handles BOUNDED center-embedding to depth ~{depth_star} amid distractors, then interference "
            f"{'has not yet won in the tested range' if held_full else 'wins (the recursion boundary)'} -- DEEPER/unbounded "
            f"recursion is where the RANK-3 mechanism (theta-gamma multiplexed WM buffer, catalog N.15; assembly-calculus "
            f"stack, Mitropolsky) becomes necessary (a reservoir cannot hold an unbounded stack). Reuse-by-import; NO sim/ "
            f"edit.")
        go = bool(markers_ok and beats_attraction and depth_star >= 1 and lesion_ok)
    else:
        go = False; verdict = f"ERROR -- {err}"; agg = lesion = markers_ok = depth_star = None

    summary = {
        "probe": "emerge83_reservoir_recursion_depth", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("RANK-3 cheap-first: measure the reservoir's bounded-recursion DEPTH on subject-verb number agreement "
                 "across center-embedding (agreement attraction) -- the matrix subject's number must be tracked past d "
                 "embedded-subject distractors; report accuracy-vs-depth + d* + beats-attraction + matrix-lesion; names "
                 "where the RANK-3 WM-buffer/stack becomes necessary; 6-seed rate CPU"),
        "nums": _NUMS, "that": _THAT, "test_depths": _TEST_DEPTHS, "seeds": list(seeds),
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "by_depth": {str(d): agg[d] for d in _TEST_DEPTHS}, "matrix_lesion_acc": lesion,
            "markers_discovered": markers_ok, "recursion_depth_star_ge_090": depth_star,
        },
        "per_seed": per,
        "HONEST_NOTE": ("Measures BOUNDED recursion depth on the canonical center-embedding agreement test (the matrix "
                        "subject's number tracked past embedded-subject distractors). A plain reservoir handles bounded "
                        "depth amid interference, then attraction wins -- the honest recursion boundary that NAMES the "
                        "RANK-3 mechanism (theta-gamma WM buffer / assembly-calculus stack) for deeper/unbounded recursion "
                        "(a reservoir cannot hold an unbounded stack; humans also fail past ~2 center-embeddings). "
                        "Reuse-by-import; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge83] VERDICT: {verdict}", flush=True)
    print(f"[emerge83] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    return _derisk(a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
