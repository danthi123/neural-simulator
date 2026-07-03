"""EMERGE-84 -- RANK-3, the GENUINE stack-recursion test: nested subject-verb PAIR-MATCHING across center-embedding, where
grammaticality requires a PUSH/POP STACK a reservoir provably cannot maintain -- measuring the depth at which a plain
reservoir BOUNDARIES, naming where the theta-gamma WM-buffer / assembly-calculus stack becomes necessary.

WHY (the honest follow-on to EMERGE-83). EMERGE-83 showed the reservoir tracks the MATRIX subject's number across
center-embedding to depth >= 4 -- but that is RETENTION (the first cue is at the START; the reservoir just holds it), NOT
the stack-requiring core of recursion. THIS test requires genuine nested MATCHING: a center-embedded structure
`<n1> s1 that <n2> s2 ... that <n_{d+1}> s_{d+1}  <m_{d+1}> v_{d+1} ... <m_1> v_1` where the verbs appear in REVERSE
pairing order and verb j must AGREE with subject j (`m_j == n_j`). Judging GRAMMATICALITY requires matching every verb to
its subject across the nesting -- a PUSH (subjects) / POP (verbs, reversed) stack. A reservoir has fading memory, NOT a
stack, so it can judge shallow nesting but must FAIL as depth grows (humans also fail past ~2 center-embeddings).

THE COUNT SHORTCUT IS DEFEATED. The ungrammatical case is made by SWAPPING two verbs' numbers (chosen to differ), so the
number MULTISET is UNCHANGED between grammatical and ungrammatical -- a "count the sng/plu" shortcut is at chance, forcing
genuine per-pair MATCHING (the stack).

THE DE-RISK (6 seeds; rate reservoir; NO `sim/` edit). Report reservoir grammaticality accuracy vs depth + the stack-
recursion DEPTH d* (largest depth >= 0.90) + the count-multiset baseline (at chance by construction) + a POSITION-SHUFFLE
control (destroys the pairing structure -> chance). This is a CHARACTERIZATION that NAMES the boundary: the depth at which
the reservoir falls to chance is its stack-recursion limit; DEEPER needs the RANK-3 mechanism (theta-gamma multiplexed WM
buffer, catalog N.15; assembly-calculus stack, Mitropolsky arXiv:2206.13217). A reservoir GO at shallow depth + a BOUNDARY
at deeper depth is the expected, honest signature (a plain reservoir is not a stack machine). Do NOT force a GO past the
boundary; the deliverable is the measured stack depth + the named next mechanism.

HONEST SCOPE. The canonical stack-recursion (center-embedded agreement matching) on a bounded corpus. Reuse-by-import
(EMERGE-78 Reservoir/Encoder); NO `sim/` edit.

Run:
  python -m research.runners._emerge84_reservoir_stack_recursion_derisk --derisk
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
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, Reservoir, _content_pools  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge84_reservoir_stack_recursion.json"

_NUMS = ["sng", "plu"]
_THAT = "that"
_TRAIN_MAXDEPTH = 3
_TEST_DEPTHS = [1, 2, 3]
_N_TRAIN_PER = 400
_N_TEST = 200
_RIDGE_LAMBDA = 1e-3


def _discover(seed):
    base = m62.build_stream(seed, n_sentences=4000)
    rng = np.random.default_rng(seed * 31 + 1)
    subj0, verb0 = m62._SUBJECTS, m62._VERBS
    extra = []
    for _ in range(6000):
        depth = int(rng.integers(1, _TRAIN_MAXDEPTH + 1))
        toks = []
        for d in range(depth + 1):
            toks += [str(rng.choice(_NUMS)), str(rng.choice(subj0))]
            if d < depth:
                toks.append(_THAT)
        for j in range(depth, -1, -1):
            toks += [str(rng.choice(_NUMS)), str(rng.choice(verb0))]
        extra += toks + [m62.SENT_PERIOD]
    words, freq, cover, _c = m62.compute_stats(base + extra)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj, verb = _content_pools(discovered)[0], _content_pools(discovered)[1]
    return discovered, subj, verb


def _make(depth, grammatical, rng, subj, verb):
    """Center-embedded chain of depth+1 (subject, verb) pairs; verbs in REVERSE pairing order. Grammatical: verb j's
    number == subject j's number. Ungrammatical: SWAP two verbs' numbers (chosen to differ -> multiset unchanged, at least
    one pair mismatched). Returns (tokens, label 1=grammatical/0=not)."""
    # require >= 2 DISTINCT subject numbers (BOTH classes) so a multiset-PRESERVING swap always exists -> the count
    # shortcut is fully defeated (grammatical + ungrammatical share the identical number multiset + the same distribution).
    while True:
        n_subj = [str(rng.choice(_NUMS)) for _ in range(depth + 1)]
        if len(set(n_subj)) >= 2:
            break
    n_verb = list(n_subj)                                        # grammatical pairing: verb j number == subject j number
    if not grammatical:
        pairs = [(i, j) for i in range(depth + 1) for j in range(i + 1, depth + 1) if n_subj[i] != n_subj[j]]
        i, j = pairs[int(rng.integers(0, len(pairs)))]          # swap two differing verb numbers (multiset preserved)
        n_verb[i], n_verb[j] = n_verb[j], n_verb[i]
    toks = []
    for d in range(depth + 1):
        toks += [n_subj[d], str(rng.choice(subj))]
        if d < depth:
            toks.append(_THAT)
    for j in range(depth, -1, -1):                              # verbs in REVERSE pairing order (center-embedding)
        toks += [n_verb[j], str(rng.choice(verb))]
    return toks, (1 if grammatical else 0)


def _final(res, enc, toks, shuffle_rng=None):
    t = list(toks)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(t)
    return np.concatenate([res.final_state(enc.encode(t)), [1.0]])


def _fit(res, enc, sents):
    X = np.asarray([_final(res, enc, t) for (t, _y) in sents])
    y = np.asarray([lab for (_t, lab) in sents])
    T = np.zeros((len(y), 2)); T[np.arange(len(y)), y] = 1.0
    return np.linalg.solve(X.T @ X + _RIDGE_LAMBDA * np.eye(X.shape[1]), X.T @ T)


def _acc(res, enc, W, sents, shuffle_rng=None):
    hit = 0
    for (toks, y) in sents:
        f = _final(res, enc, toks, shuffle_rng=shuffle_rng)
        hit += int(int(np.argmax(f @ W)) == y)
    return float(hit / max(1, len(sents)))


def _count_multiset_baseline_acc(sents):
    """Predict grammatical from the (sng,plu) COUNT only -> at chance by construction (the swap preserves the multiset)."""
    from collections import Counter
    table = Counter(); tot = Counter()
    for (toks, y) in sents:
        key = tuple(sorted(Counter(w for w in toks if w in _NUMS).items()))
        table[(key, y)] += 1; tot[key] += 1
    hit = 0
    for (toks, y) in sents:
        key = tuple(sorted(Counter(w for w in toks if w in _NUMS).items()))
        pred = 1 if table[(key, 1)] >= table[(key, 0)] else 0
        hit += int(pred == y)
    return float(hit / max(1, len(sents)))


def _gen(depth, n, rng, subj, verb):
    return [_make(depth, bool(i % 2), rng, subj, verb) for i in range(n)]


def _one(seed):
    discovered, subj, verb = _discover(seed)
    # the NUMBER markers must be distinct discovered cues (essential); `that` may abstract to OPEN (a neutral embedding
    # marker) -- the agreement dependency is over the NUMBERS, so that-discovery is not required.
    markers_ok = all(n in discovered for n in _NUMS)
    enc = Encoder(discovered)
    res = Reservoir(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = [x for d in range(1, _TRAIN_MAXDEPTH + 1) for x in _gen(d, _N_TRAIN_PER, rng, subj, verb)]
    W = _fit(res, enc, train)

    by_depth = {}
    for d in _TEST_DEPTHS:
        test = _gen(d, _N_TEST, rng, subj, verb)
        scr = np.random.default_rng(seed * 613 + d)
        by_depth[d] = {"reservoir": _acc(res, enc, W, test),
                       "count_baseline": _count_multiset_baseline_acc(test),
                       "shuffle": _acc(res, enc, W, test, shuffle_rng=scr)}
    return {"seed": seed, "markers_ok": bool(markers_ok), "by_depth": by_depth, "chance": 0.5}


def _derisk(seeds):
    print(f"EMERGE-84: RANK-3 GENUINE stack-recursion (nested subject-verb PAIR-MATCHING grammaticality across center-"
          f"embedding); {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _one(s); per.append(d)
            row = " ".join(f"d{dd}:{d['by_depth'][dd]['reservoir']:.2f}/cnt{d['by_depth'][dd]['count_baseline']:.2f}/"
                           f"shf{d['by_depth'][dd]['shuffle']:.2f}" for dd in _TEST_DEPTHS)
            print(f"  [seed {s}] markers_ok {d['markers_ok']} | res/count/shuffle by depth: {row}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        agg = {d: {k: float(np.mean([p["by_depth"][d][k] for p in per]))
                   for k in ("reservoir", "count_baseline", "shuffle")} for d in _TEST_DEPTHS}
        markers_ok = all(p["markers_ok"] for p in per)
        depth_star = max([d for d in _TEST_DEPTHS if agg[d]["reservoir"] >= 0.90], default=0)
        count_defeated = all(agg[d]["count_baseline"] <= 0.65 for d in _TEST_DEPTHS)
        shuffle_ok = all(agg[d]["shuffle"] <= 0.65 for d in _TEST_DEPTHS)
        boundaries = (depth_star < _TEST_DEPTHS[-1])              # the reservoir falls below 0.90 at some tested depth

        verdict = (
            f"CHARACTERIZATION -- genuine STACK-recursion (nested subject-verb pair-matching grammaticality) exposes the "
            f"reservoir's recursion LIMIT: it judges grammaticality to a STACK-DEPTH d* = {depth_star} (profile "
            f"{', '.join(f'd{d}={agg[d]['reservoir']:.2f}' for d in _TEST_DEPTHS)}), then "
            f"{'FALLS toward chance at deeper nesting -- the recursion BOUNDARY (a reservoir has fading memory, NOT a stack)' if boundaries else 'holds across the tested range (a wider reservoir / deeper sweep would find the limit)'}. "
            f"The COUNT-multiset shortcut is DEFEATED (baseline "
            f"{', '.join(f'd{d}={agg[d]['count_baseline']:.2f}' for d in _TEST_DEPTHS)} ~chance -- the swap preserves the "
            f"multiset, so the reservoir must do genuine PER-PAIR matching); POSITION-SHUFFLE collapses "
            f"({', '.join(f'd{d}={agg[d]['shuffle']:.2f}' for d in _TEST_DEPTHS)} -> reads structure). markers discovered "
            f"= {markers_ok}. ==> {'the plain reservoir BOUNDARIES on genuine stack-recursion at depth ' + str(depth_star + 1) + ' -- this is where the RANK-3 mechanism (theta-gamma multiplexed WM buffer, catalog N.15; assembly-calculus stack, Mitropolsky arXiv:2206.13217) becomes NECESSARY (a reservoir cannot hold a push/pop stack); the next de-risk ADDS that mechanism.' if boundaries else 'the reservoir handled the tested stack depths; extend the depth sweep / narrow the reservoir to find the limit, then add the RANK-3 stack.'} "
            f"Reuse-by-import; NO sim/ edit.")
        # GO here = the test is VALID (count defeated, shuffle collapses, markers ok) AND it either shows the mechanism at
        # shallow depth OR cleanly boundaries -- a valid characterization either way; the deliverable is the measured depth.
        go = bool(markers_ok and count_defeated and shuffle_ok and depth_star >= 1)
    else:
        go = False; verdict = f"ERROR -- {err}"; agg = markers_ok = depth_star = None

    summary = {
        "probe": "emerge84_reservoir_stack_recursion", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("RANK-3 genuine stack-recursion: judge grammaticality of nested subject-verb pair-matching across "
                 "center-embedding (verbs in reversed pairing order; verb j must agree with subject j -> a push/pop "
                 "stack); the ungrammatical case swaps two verb numbers (multiset preserved -> count shortcut defeated); "
                 "report reservoir accuracy vs stack-depth + count-baseline + position-shuffle; names the recursion "
                 "boundary + the RANK-3 mechanism; 6-seed rate CPU"),
        "nums": _NUMS, "that": _THAT, "test_depths": _TEST_DEPTHS, "seeds": list(seeds),
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "by_depth": {str(d): agg[d] for d in _TEST_DEPTHS}, "markers_discovered": markers_ok,
            "reservoir_stack_depth_star_ge_090": depth_star,
        },
        "per_seed": per,
        "HONEST_NOTE": ("The GENUINE stack-recursion test (vs EMERGE-83's retention): nested pair-MATCHING requires a "
                        "push/pop stack a reservoir lacks. The count-multiset shortcut is defeated by a multiset-preserving "
                        "swap; position-shuffle collapses. The depth at which the reservoir falls to chance is its stack-"
                        "recursion limit; DEEPER needs the RANK-3 mechanism (theta-gamma WM buffer N.15 / assembly-calculus "
                        "stack). A shallow-GO + deeper-BOUNDARY is the honest, expected signature (a reservoir is not a "
                        "stack machine); the next de-risk ADDS the stack mechanism. Reuse-by-import; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge84] VERDICT: {verdict}", flush=True)
    print(f"[emerge84] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    return _derisk(a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
