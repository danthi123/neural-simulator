"""EMERGENCE-ENGINE STREAM-LANGUAGE DE-RISK — does the roadmap's on-bridge HTM Temporal-Memory sequence cortex
(EMERGE-14) LEARN the structure of a RICHER, LANGUAGE-SHAPED token STREAM online — predicting a long-range AGREEMENT
dependency above the best fixed-order n-gram floor AND GENERALIZING to HELD-OUT continuations (filler paths NEVER seen in
training) — or does it only MEMORIZE surface sequences and collapse to the n-gram floor once the stream's intervening
material actually VARIES (the statistical shape of a real stream)?

WHY THIS IS THE FRONTIER (our-own-record first):
  * The three 2026-08-11 emergence-engine findings (horizon / selective-write store / hetero-LTD allocation) all measured
    the on-bridge HTM-TM on the EMERGE-14 OVERLAP CORPUS: n_seq sentences [cue, <FIXED shared middle>, branch]. That
    middle is IDENTICAL every presentation, so the task is MEMORISE-AND-RECALL (the horizon finding's own HONEST_NOTE:
    "NOT held-out generalisation"). The engine learns high-order structure, is non-fading, and the store+hetero-allocation
    extend its horizon/capacity — but NOBODY has asked whether it learns the STATISTICAL structure of a stream where the
    intervening tokens are NOT fixed, and whether it GENERALISES the dependency to novel continuations. That is the gap
    between memorising sequences and learning LANGUAGE structure.

THE STREAM (a minimal LANGUAGE-shaped structure with a genuine long-range dependency; generated ONLINE, never repeated):
  Vocabulary (modest, 16-64): n_subj SUBJECT tokens, n_fill FILLER tokens, n_subj VERB tokens. A sentence is
      [subject_i] + [L filler tokens drawn i.i.d. from the filler pool] + [verb_i]
  where verb_i AGREES with subject_i (deterministic subject->verb map). The verb depends ONLY on the subject L+1 tokens
  back; the L intervening fillers are i.i.d. NOISE (uninformative about the verb, shared pool across all subjects). This is
  the classic long-range-agreement structure (number/gender agreement across an arbitrary intervening span). Because the
  filler span is RANDOM and DIFFERENT every sentence, it CANNOT be memorised as a fixed sequence — the model must carry the
  subject (a latent variable) invariantly across novel intervening material, which is exactly what language requires.

WHY THE n-GRAM FLOOR IS AT CHANCE (the emergence bar is meaningful):
  A fixed-order-k n-gram at the verb position sees the last k tokens = k random fillers (k<=L) -> uninformative -> chance;
  or, at order >= L+1, it sees [subject, <this sentence's random fillers>] -> a context that (on a random stream) is
  UNIQUE per sentence -> on HELD-OUT sentences (novel filler paths) that context was never seen -> back-off -> chance. So
  the BEST fixed-order n-gram, evaluated on HELD-OUT, is pinned at chance 1/n_subj at EVERY order. Beating it on held-out
  REQUIRES abstracting the subject across variable fillers — a structure-learner's job, not a surface-window's.

GENERALISATION (the anti-memorisation core): TRAIN on a stream of sentences with random filler; TEST on a DISJOINT set of
  sentences whose exact filler paths never appeared in training. test-acc >> chance => the engine learned the AGREEMENT
  RULE (generalises); test-acc ~ chance while TRAIN-acc is high => it MEMORISED surface paths and did not generalise (the
  honest negative that names the next mechanism: a latent-variable / variable-binding working memory, not more allocation).

ARMS / ANTI-CHEATS (each EXECUTES via tools.lab; the earned teeth):
  (a) htm            : OnBridgeLearner (EMERGE-14 emergence engine) trained on the real agreement stream; branch(verb) acc
                       on HELD-OUT test (the generalisation number) and on TRAIN (the memorisation number).
  (b) htm_store      : the banked SELECTIVE-WRITE content-addressable store harvested over the TRAIN traversals, read at
                       test — asks whether the store that restored the FIXED-middle horizon ALSO helps generalise a stream.
  (c) lesion         : dAP/coincidence OFF -> the priming chain is severed -> must collapse to chance (recurrence is
                       load-bearing).
  (d) untrained      -> chance.
  (e) permuted-stream: retrain on a stream where the verb is DRAWN INDEPENDENTLY of the subject (no agreement structure)
                       -> chance on BOTH train and test (attribution: any above-chance came from the REAL S->V structure).
  (f) swap-follows   : inject a DIFFERENT subject into a held-out sentence -> the branch prediction must FOLLOW the injected
                       subject's verb (proves subject-DRIVEN, not filler/positional).
  Plus the best-fixed-order n-gram HELD-OUT floor (pinned at chance) + a subject-oracle (task solvable by construction) +
  the n_fill=1 FIXED anchor (reproduces the prior memorise-and-recall ~1.0) + multi-seed.

GO (emergence of generalisable long-range structure) = at a stream point where the n-gram floor is chance, HELD-OUT
  test-acc >= 0.90 AND >= chance + 0.20 AND >= ngram_floor + 0.15, dAP-lesion collapses (test - lesion >= 0.20),
  permuted-stream <= chance + 0.10, swap-follows >= 0.90, untrained <= chance + 0.10, multi-seed. HONEST NEGATIVE
  (first-class) = the engine MEMORISES the training stream (train-acc high) but does NOT generalise (held-out ~ floor),
  or breaks as branching-factor/distance grows — reported with the numbers + the named next mechanism.

Reuse-by-import (EMERGE-14 on-bridge learner + the selective-write store); NO sim/ edit. SIM_BACKEND=numpy (the horizon
finding established these sub-1k-neuron coincidence loops are LAUNCH-BOUND: cupy is slower; CPU/numpy is correct + faster).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner)
from research.runners._emerge_selective_write_store_derisk import (
    harvest_store, store_predict_branch)

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_emerge_stream_language.json")


# --------------------------------------------------------------------------------------------------------------------
# The agreement stream: [subject_i] + [L i.i.d. fillers] + [verb_i], verb_i agrees with subject_i.
# token layout (columns): subjects [0,n_subj) | fillers [n_subj, n_subj+n_fill) | verbs [n_subj+n_fill, +n_subj)
# --------------------------------------------------------------------------------------------------------------------
def vocab_layout(n_subj, n_fill):
    subj = list(range(n_subj))
    fill = list(range(n_subj, n_subj + n_fill))
    verb = list(range(n_subj + n_fill, n_subj + n_fill + n_subj))
    V = n_subj + n_fill + n_subj
    return subj, fill, verb, V


def make_fixed_corpus(n_subj, L, seed=0):
    """The FIXED-distinct-middle anchor (the prior EMERGE overlap corpus): n_subj sentences that share ONE fixed middle of
    L DISTINCT filler tokens -> [subject_i] + [f_0..f_{L-1}] + [verb_i]. The shared middle pins any fixed-order n-gram at
    chance (identical middle for every subject); the engine must carry the subject through the middle. Uses a filler pool
    of size L. Returns (seqs, V)."""
    subj, fill, verb, V = vocab_layout(n_subj, L)          # n_fill == L distinct middle tokens
    middle = list(fill)
    seqs = [[subj[i]] + middle + [verb[i]] for i in range(n_subj)]
    return seqs, V


def make_stream(n_subj, n_fill, L, n_sent, rng, verb_map=None, exclude=None):
    """A list of agreement sentences with i.i.d. random filler spans. verb_map: subject_idx -> verb TOKEN (default = the
    agreeing verb; pass a deranged/random map for the permuted-stream control). exclude: a set of filler-tuple keys to
    AVOID (for a held-out disjoint test set). Returns (seqs, n_novel) where n_novel = # sentences whose filler tuple was
    NOT in `exclude`."""
    subj, fill, verb, V = vocab_layout(n_subj, n_fill)
    if verb_map is None:
        verb_map = {i: verb[i] for i in range(n_subj)}
    exclude = exclude or set()
    seqs, novel = [], 0
    for _ in range(n_sent):
        i = int(rng.integers(0, n_subj))
        ftuple = tuple(int(fill[rng.integers(0, n_fill)]) for _ in range(L))
        seqs.append([subj[i]] + list(ftuple) + [verb_map[i]])
        novel += int(ftuple not in exclude)
    return seqs, novel


def make_heldout(n_subj, n_fill, L, n_sent, rng, train_ftuples, max_tries_mult=40):
    """A test set whose filler tuples are DISJOINT from the training set (true held-out generalisation). Returns
    (seqs, generalisation_defined): if the path space n_fill**L is too small to hold out enough novel paths, returns the
    best it can with generalisation_defined=False (the n_fill=1 FIXED anchor lands here -> memorise-and-recall)."""
    subj, fill, verb, V = vocab_layout(n_subj, n_fill)
    seqs, seen = [], set()
    tries = 0
    while len(seqs) < n_sent and tries < n_sent * max_tries_mult:
        tries += 1
        i = int(rng.integers(0, n_subj))
        ftuple = tuple(int(fill[rng.integers(0, n_fill)]) for _ in range(L))
        if ftuple in train_ftuples:
            continue
        seqs.append([subj[i]] + list(ftuple) + [verb[i]])
        seen.add(ftuple)
    generalisation_defined = len(seqs) >= max(20, 4 * n_subj)
    if not generalisation_defined and len(seqs) < n_sent:
        # not enough novel paths (small branching factor / short span): fall back to in-sample-style eval (the anchor)
        extra, _ = make_stream(n_subj, n_fill, L, n_sent - len(seqs), rng)
        seqs += extra
    return seqs, generalisation_defined


# --------------------------------------------------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------------------------------------------------
def ngram_floor_heldout(train, test, div_pos, n_subj, max_order=None):
    """BEST fixed-order n-gram floor on HELD-OUT: build order-k counts of the (context -> next-token) map on TRAIN, predict
    the branch on TEST, take the MAX accuracy over orders k=1..max_order. On a random-filler stream this is pinned at
    chance 1/n_subj at every order (the k recent tokens are random; order>=L+1 contexts are unique per sentence -> unseen
    on held-out -> back-off to chance). Returns (best_acc, best_order)."""
    L = div_pos
    if max_order is None:
        max_order = L + 2
    best_acc, best_order = 0.0, 0
    for k in range(1, max_order + 1):
        counts = defaultdict(Counter)
        for s in train:
            t = len(s) - 2                                   # branch position's predecessor index (== div_pos)
            ctx = tuple(s[max(0, t - k + 1): t + 1])
            counts[ctx][s[t + 1]] += 1
        ok = 0.0
        for s in test:
            t = len(s) - 2
            ctx = tuple(s[max(0, t - k + 1): t + 1])
            dist = counts.get(ctx)
            if not dist:
                ok += 1.0 / n_subj                            # unseen context -> back-off to chance
                continue
            top = max(dist.values()); win = [x for x, n in dist.items() if n == top]
            ok += (1.0 / len(win)) if s[t + 1] in win else 0.0
        acc = ok / len(test)
        if acc > best_acc:
            best_acc, best_order = acc, k
    return best_acc, best_order


# --------------------------------------------------------------------------------------------------------------------
# On-bridge emergence engine (EMERGE-14) train / eval
# --------------------------------------------------------------------------------------------------------------------
def train_engine(seed, arm, n_subj, n_fill, L, n_cells, k_win, act_th, epochs, train_seqs):
    _, _, _, V = vocab_layout(n_subj, n_fill)
    b, cells_idx, row, col = build_pool_bridge(V, n_cells, seed, act_th=act_th, coincidence=(arm != "lesion"))
    lr = OnBridgeLearner(b, row, col, cells_idx, V, n_cells, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in train_seqs:
                lr.train_sequence(s)
    return lr


def branch_acc(lr, seqs, L):
    return sum(int(lr.predict_branch(s, L)[L] == {s[-1]}) for s in seqs) / max(1, len(seqs))


def swap_follows_context(lr, test_seqs, n_subj, n_fill, L, max_pairs=300, rng=None):
    """CONTEXT-NECESSITY: for each held-out sentence, inject a DIFFERENT subject (word 0) and require the branch prediction
    to FOLLOW the injected subject's verb (== the model carries the SUBJECT through the span, not the filler/position)."""
    subj, fill, verb, V = vocab_layout(n_subj, n_fill)
    rng = rng or np.random.default_rng(0)
    ok = tot = 0
    for s in test_seqs:
        cur = s[0]
        j = int(rng.integers(0, n_subj))
        if verb[j] == s[-1]:                                  # ensure a genuinely DIFFERENT subject
            j = (j + 1) % n_subj
        swapped = list(s); swapped[0] = subj[j]
        pred = lr.predict_branch(swapped, L)[L]
        ok += int(pred == {verb[j]}); tot += 1
        if tot >= max_pairs:
            break
    return ok / max(1, tot)


def run_point(seed, n_subj, n_fill, L, n_cells, k_win, act_th, epochs, n_train, n_test, tau):
    """One (seed, n_fill, L) point: build a train stream + a DISJOINT held-out test stream, train the emergence engine and
    all control arms, harvest the selective-write store, and measure branch(verb) accuracy + every anti-cheat.
    n_fill == 0 selects the FIXED-distinct-middle ANCHOR (the prior EMERGE overlap corpus: in-sample memorise-and-recall,
    expected ~1.0 -> proves the machinery works + reproduces the prior finding)."""
    rng = np.random.default_rng(seed)
    anchor = (n_fill == 0)
    eff_nfill = L if anchor else n_fill                    # the anchor's filler pool is L distinct tokens
    subj, fill, verb, V = vocab_layout(n_subj, eff_nfill)
    chance = 1.0 / n_subj

    if anchor:
        fixed_seqs, _ = make_fixed_corpus(n_subj, L, seed)
        train_seqs = fixed_seqs
        test_seqs, gen_defined = fixed_seqs, False         # only n_subj sentences -> in-sample (memorise-and-recall)
    else:
        train_seqs, _ = make_stream(n_subj, n_fill, L, n_train, rng)
        train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)
        test_seqs, gen_defined = make_heldout(n_subj, n_fill, L, n_test, rng, train_ftuples)
    train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)

    # permuted-stream: verb drawn INDEPENDENTLY of subject (a derangement of the subject->verb map) -> no agreement rule
    perm = list(range(n_subj))
    for _ in range(64):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n_subj)):
            break
    perm_map = {i: verb[perm[i]] for i in range(n_subj)}
    if anchor:
        perm_train = [[s[0]] + list(s[1:-1]) + [perm_map[s[0]]] for s in train_seqs]
    else:
        perm_train, _ = make_stream(n_subj, eff_nfill, L, n_train, np.random.default_rng(seed + 777), verb_map=perm_map)
    # the permuted TEST targets the deranged verb too (measures whether an arbitrary S->V map is learnable/generalises)
    perm_test = [[s[0]] + list(s[1:-1]) + [perm_map[s[0]]] for s in test_seqs]

    # --- engine arms (all use eff_nfill for the vocab/column layout) ---
    lr = train_engine(seed, "htm", n_subj, eff_nfill, L, n_cells, k_win, act_th, epochs, train_seqs)
    lr_les = train_engine(seed, "lesion", n_subj, eff_nfill, L, n_cells, k_win, act_th, epochs, train_seqs)
    lr_unt = train_engine(seed, "untrained", n_subj, eff_nfill, L, n_cells, k_win, act_th, epochs, train_seqs)
    lr_perm = train_engine(seed, "htm", n_subj, eff_nfill, L, n_cells, k_win, act_th, epochs, perm_train)

    train_acc = branch_acc(lr, train_seqs, L)
    test_acc = branch_acc(lr, test_seqs, L)
    lesion_test = branch_acc(lr_les, test_seqs, L)
    untrained_test = branch_acc(lr_unt, test_seqs, L)
    permuted_train = branch_acc(lr_perm, perm_train, L)
    permuted_test = branch_acc(lr_perm, perm_test, L)
    swap = swap_follows_context(lr, test_seqs, n_subj, eff_nfill, L, rng=np.random.default_rng(seed + 3))

    # --- banked selective-write content store over the TRAIN traversals, read at test ---
    store = harvest_store(lr, train_seqs, selective=True)
    store_train = sum(int(store_predict_branch(lr, store, s, L, tau) == {s[-1]}) for s in train_seqs) / max(1, len(train_seqs))
    store_test = sum(int(store_predict_branch(lr, store, s, L, tau) == {s[-1]}) for s in test_seqs) / max(1, len(test_seqs))

    # --- floors ---
    ngram_test, ngram_order = ngram_floor_heldout(train_seqs, test_seqs, L, n_subj)
    ngram_train, _ = ngram_floor_heldout(train_seqs, train_seqs, L, n_subj)   # in-sample (memorisation) reference

    return {"seed": seed, "n_fill": n_fill, "L": L, "distance": L + 1, "n_cells": n_cells, "chance": chance,
            "train_acc": train_acc, "test_acc": test_acc, "lesion_test": lesion_test, "untrained_test": untrained_test,
            "permuted_train": permuted_train, "permuted_test": permuted_test, "swap_follows": swap,
            "store_train": store_train, "store_test": store_test, "n_store_keys": len(store.store),
            "n_store_poisoned": len(store.poisoned),
            "ngram_floor_test": ngram_test, "ngram_floor_order": ngram_order, "ngram_train": ngram_train,
            "subject_oracle": 1.0, "generalisation_defined": bool(gen_defined), "anchor": bool(anchor),
            "n_train": len(train_seqs), "n_test": len(test_seqs),
            "n_distinct_train_paths": len(train_ftuples),
            "path_space": (1.0 if anchor else float(n_fill) ** L)}


def agg(per):
    keys = ["train_acc", "test_acc", "lesion_test", "untrained_test", "permuted_train", "permuted_test", "swap_follows",
            "store_train", "store_test", "n_store_keys", "n_store_poisoned", "ngram_floor_test", "ngram_train",
            "subject_oracle", "n_distinct_train_paths"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a.update({"n_fill": per[0]["n_fill"], "L": per[0]["L"], "distance": per[0]["distance"], "n_cells": per[0]["n_cells"],
              "chance": per[0]["chance"], "path_space": per[0]["path_space"], "anchor": bool(per[0]["anchor"]),
              "generalisation_defined": all(p["generalisation_defined"] for p in per),
              "ngram_floor_order": int(np.round(np.mean([p["ngram_floor_order"] for p in per]))), "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-subj", type=int, default=4, help="# subject classes = # agreeing verbs (chance = 1/n_subj)")
    ap.add_argument("--n-fills", type=int, nargs="+", default=[4],
                    help="filler-alphabet sizes to sweep (BRANCHING FACTOR of the stream; n_fill=1 = the FIXED anchor)")
    ap.add_argument("--distances", type=int, nargs="+", default=[2],
                    help="filler-span lengths L (dependency distance = L+1); swept for the horizon")
    ap.add_argument("--n-cells", type=int, default=40, help="cells/column (generous: give the engine its best shot)")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3, help="passes over the train stream (memorisation ceiling for train)")
    ap.add_argument("--n-train", type=int, default=360, help="# training sentences (online stream)")
    ap.add_argument("--n-test", type=int, default=160, help="# held-out (disjoint-path) test sentences")
    ap.add_argument("--tau-read", type=float, default=0.5, help="min allocation-SDR overlap for a content-store hit")
    ap.add_argument("--go-nfill", type=int, default=None, help="n_fill at which to evaluate GO (default: the largest swept)")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to evaluate GO (default: the largest swept)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 1:
        print("NOT-RUNNABLE: need >=1 seed"); return 2
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy")
    device = "cpu" if backend == "numpy" else "gpu"
    chance = 1.0 / a.n_subj
    nfills = sorted(set(a.n_fills)); dists = sorted(set(a.distances))
    _, _, _, Vmax = vocab_layout(a.n_subj, max(nfills))
    print(f"backend={backend} device={device} | n_subj={a.n_subj} chance={chance:.3f} | n_fills={nfills} distances(L)={dists} "
          f"| vocab<= {Vmax} | n_cells={a.n_cells} epochs={a.epochs} n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds}",
          flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for nf in nfills:
            for L in dists:
                per = [run_point(s, a.n_subj, nf, L, a.n_cells, a.k_win, a.act_th, a.epochs, a.n_train, a.n_test,
                                 a.tau_read) for s in a.seeds]
                p = agg(per); points.append(p)
                gen = "GEN(held-out)" if p["generalisation_defined"] else ("FIXED-anchor" if p["anchor"] else "in-sample")
                nfl = "fix" if nf == 0 else str(nf)
                print(f"  [n_fill={nfl:>3} L={L} dist={L+1} paths={p['path_space']:.0f} {gen:>13}] "
                      f"train {p['train_acc']:.3f} | TEST {p['test_acc']:.3f} | lesion {p['lesion_test']:.3f} | untr "
                      f"{p['untrained_test']:.3f} | perm(tr/te) {p['permuted_train']:.3f}/{p['permuted_test']:.3f} | "
                      f"store(tr/te) {p['store_train']:.3f}/{p['store_test']:.3f} || swap {p['swap_follows']:.3f} | "
                      f"ngram_floor(te) {p['ngram_floor_test']:.3f}@k{p['ngram_floor_order']} (train {p['ngram_train']:.3f}) "
                      f"| chance {chance:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # --- pick the GO evaluation point: a GENERALISATION point (largest branching x distance by default) ---
    go_nf = a.go_nfill if a.go_nfill is not None else (max(nfills) if nfills else None)
    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["n_fill"] == go_nf and p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: (p["n_fill"], p["L"]))

    verdict = None
    if err is None and far is not None:
        print(f"\n-- emergence bar + anti-cheats at the GO point (n_fill={far['n_fill']}, L={far['L']}, dist={far['distance']}) --",
              flush=True)
        void_if(not far["generalisation_defined"],
                "GO point has too small a path space to hold out novel continuations -> generalisation UNDEFINED (anchor)")
        lever("held-out TEST vs dAP-lesion (recurrence load-bearing)", round(far["lesion_test"], 3), round(far["test_acc"], 3),
              required=False)
        lever("held-out TEST vs bare content-store (does the banked store help generalise?)",
              round(far["test_acc"], 3), round(far["store_test"], 3), required=False)
        attributable_to("held-out learned over the best n-gram floor", far["test_acc"], far["ngram_floor_test"])
        attributable_to("held-out learned over the permuted-stream control", far["test_acc"], far["permuted_test"])

        gen_defined = far["generalisation_defined"]
        beats_ngram = far["test_acc"] >= far["ngram_floor_test"] + 0.15
        above_chance = far["test_acc"] >= chance + 0.20 and far["test_acc"] >= 0.90
        recurrence = far["test_acc"] >= far["lesion_test"] + 0.20
        attributed = far["permuted_test"] <= chance + 0.10
        context_driven = far["swap_follows"] >= 0.90
        untr_ok = far["untrained_test"] <= chance + 0.10
        memorised_train = far["train_acc"] >= chance + 0.20         # did the engine at least fit the train stream?

        core = bool(gen_defined and beats_ngram and above_chance and recurrence and attributed and context_driven and untr_ok)
        go = bool(core and not smoke)

        if not gen_defined:
            verdict = (f"INCONCLUSIVE — the GO point (n_fill={far['n_fill']}, L={far['L']}) has path space "
                       f"{far['path_space']:.0f}, too small to hold out novel continuations, so GENERALISATION is UNDEFINED "
                       f"(this point is the memorise-and-recall anchor). Increase n_fill/L for a real held-out regime.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — the on-bridge HTM-TM emergence engine LEARNS the agreement stream's long-range structure "
                       f"and GENERALISES it: at n_fill={far['n_fill']} L={far['L']} (dist {far['distance']}) held-out TEST "
                       f"branch(verb) acc {far['test_acc']:.3f} >> chance {chance:.3f}, >> best-fixed-order n-gram floor "
                       f"{far['ngram_floor_test']:.3f} (order {far['ngram_floor_order']}, pinned at chance on held-out), and "
                       f"the recurrence is load-bearing (dAP-lesion {far['lesion_test']:.3f}), attributable "
                       f"(permuted-stream {far['permuted_test']:.3f} <= chance), subject-driven (swap-follows "
                       f"{far['swap_follows']:.3f}), untrained {far['untrained_test']:.3f}. Conversational/sequence structure "
                       f"EMERGES from the stream, not memorised. Reuse-by-import of EMERGE-14; NO sim/ edit.")
        else:
            miss = []
            if not beats_ngram: miss.append(f"held-out did NOT beat the n-gram floor (test {far['test_acc']:.3f} vs floor {far['ngram_floor_test']:.3f})")
            if not above_chance: miss.append(f"held-out test {far['test_acc']:.3f} not >= 0.90/chance+0.20 ({chance:.3f})")
            if not recurrence: miss.append(f"recurrence not load-bearing (test {far['test_acc']:.3f} vs lesion {far['lesion_test']:.3f})")
            if not attributed: miss.append(f"not attributable (permuted-stream {far['permuted_test']:.3f} > chance+0.10)")
            if not context_driven: miss.append(f"not subject-driven (swap-follows {far['swap_follows']:.3f} < 0.90)")
            memo = (f"MEMORISED train ({far['train_acc']:.3f}) but did NOT generalise (held-out {far['test_acc']:.3f} ~ "
                    f"floor {far['ngram_floor_test']:.3f}/chance {chance:.3f})") if (memorised_train and far["test_acc"] <= far["ngram_floor_test"] + 0.10) \
                   else f"train {far['train_acc']:.3f} / held-out {far['test_acc']:.3f}"
            verdict = ("HONEST NEGATIVE / BOUNDARY — the emergence engine did NOT learn generalisable long-range stream "
                       f"structure at n_fill={far['n_fill']} L={far['L']}: " + "; ".join(miss) + f". Diagnosis: {memo}. "
                       "The on-bridge HTM-TM does EXACT-PATH high-order memory (allocates a context-specific SDR per "
                       "traversed path); it does not ABSTRACT the latent subject variable invariantly across novel/variable "
                       "intervening fillers, so it memorises surface paths and cannot generalise the agreement rule. Names "
                       "the next mechanism: a latent-variable / variable-binding WORKING MEMORY (a gated slot that carries "
                       "the agreement feature across arbitrary intervening tokens) rather than more allocation capacity or a "
                       "content store over path-specific keys — and hands the residual to the deep-credit gap#4 enabler.")
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict: VALIDITY preconditions travel with the verdict (tools/gates/verdict_preconditions) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        V = Verdict("emerge_stream_language", chance=chance)
        # PRECONDITIONS = VALIDITY checks that must hold for ANY verdict (GO or a legitimate negative) to be
        # meaningful: the task is context-solvable, the held-out regime is a real generalisation regime, the
        # attribution instrument (permuted-stream) discriminates, and the untrained baseline sits at chance. The
        # RESULT (held-out beats chance/floor, recurrence load-bearing) is the DECISION passed to decide() -- a
        # VALID point BELOW threshold is a legitimate NO-GO/honest-negative, NOT UNDEFINED.
        if far is not None:
            V.require("subject_oracle_task_solvable", 1.0, expect=lambda x: x > 0.99,
                      note="the verb is determined by the subject by construction -> the task IS context-solvable")
            V.require("generalisation_defined_novel_heldout", 1 if far["generalisation_defined"] else 0,
                      expect=lambda x: x >= 1,
                      note="held-out test paths must be disjoint from train (a real generalisation regime), else UNDEFINED")
            V.require("permute_attribution_instrument_valid", round(far["permuted_test"], 4),
                      expect=lambda x: x <= chance + 0.10,
                      note="permuted-stream (verb independent of subject) must read <= chance, else the attribution "
                           "instrument does not discriminate and the comparison is invalid (VALIDITY, not the result)")
            V.require("untrained_baseline_at_chance", round(far["untrained_test"], 4),
                      expect=lambda x: x <= chance + 0.10,
                      note="the untrained control must sit at chance (a VALIDITY baseline for the instrument)")
        else:
            V.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(far is not None and far["generalisation_defined"] and far["test_acc"] >= 0.90
                   and far["test_acc"] >= chance + 0.20 and far["test_acc"] >= far["ngram_floor_test"] + 0.15
                   and far["test_acc"] >= far["lesion_test"] + 0.20 and far["permuted_test"] <= chance + 0.10
                   and far["swap_follows"] >= 0.90 and far["untrained_test"] <= chance + 0.10)
        dec = V.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_stream_language", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "device": device, "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "mechanism": "on-bridge HTM Temporal-Memory (EMERGE-14 OnBridgeLearner: allocation + non-fading priming "
                            "chain over the sim/ fused_htm_permanence_update kernel on cp_connections.data) trained ONLINE "
                            "on an agreement stream, tested for HELD-OUT generalisation of a long-range subject->verb "
                            "dependency; the banked selective-write content store is harvested over the train traversals "
                            "and read at test as an additional arm",
               "task": "agreement stream [subject_i]+[L i.i.d. fillers]+[verb_i] (verb agrees with subject L+1 tokens back "
                       "through random filler noise); held-out TEST = disjoint filler paths; anti-cheats: dAP-lesion "
                       "(recurrence) + untrained + permuted-stream (verb independent of subject; attribution) + "
                       "swap-follows-context (subject-driven) + best-fixed-order n-gram HELD-OUT floor (pinned at chance) + "
                       "n_fill=1 FIXED anchor + multi-seed",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "n_fills": nfills, "distances": dists, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs, "n_train": a.n_train, "n_test": a.n_test,
               "tau_read": a.tau_read, "chance": chance, "go_nfill": go_nf, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the EMERGE-14 on-bridge learner + the selective-write store; NO sim/ edit. "
                              "Unlike the prior EMERGE overlap-corpus findings (FIXED shared middle = memorise-and-recall), "
                              "the middle here is a RANDOM i.i.d. filler span and the TEST set holds out NOVEL filler paths, "
                              "so the metric is HELD-OUT GENERALISATION of a long-range agreement rule. The n-gram floor is "
                              "the BEST fixed order over 1..L+2 on held-out (pinned at chance by the random uninformative "
                              "middle). n_fill=1 is the FIXED anchor (reproduces the prior ~1.0). 1-seed is a SMOKE "
                              "indicator; the decisive run is the 6-seed sweep."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge_stream_language] VERDICT: {verdict}", flush=True)
    print(f"[emerge_stream_language] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
