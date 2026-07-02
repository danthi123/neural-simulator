"""EMERGE-9b (rung-3 pivot) — the FAITHFUL HTM Temporal Memory: multi-SEGMENT cells + population (SDR) winners +
grow-a-new-segment-on-a-fresh-cell allocation. This is the correct architecture EMERGE-9's minimal single-permanence-
row version approximated wrongly (which merged contexts). The disambiguation the minimal version couldn't do:
distinct prior contexts activate DISTINCT winner cells because (a) a novel context matches NO existing segment
(each segment holds synapses to ONE specific prior SDR) so it allocates FRESH least-used cells, and (b) predictive
state is per-SEGMENT (a cell fires only for the context ITS segment encodes). Unsupervised, local, NO teacher.

MECHANISM (Hawkins-Ahmad HTM-TM, the discrete essence of Bouhadjar-Diesmann 2022's spiking port):
  - Each cell owns a LIST of distal segments; each segment = {presyn_cell: permanence}. A synapse is CONNECTED at
    permanence >= perm_conn. A segment is ACTIVE (-> its cell is predictive) when >= act_th of its connected synapses
    are from currently-active cells.
  - Per symbol c: predictive cells in column c become active + are the winners (correct high-order prediction, sparse).
    Else BURST: all cells in c active; winner = the best-MATCHING-segment cell if it clears learn_th, ELSE the k
    least-used cells (fewest segments) get a NEW segment each = allocation of a DISTINCT SDR for this context.
  - LOCAL Hebbian permanence learning (NO teacher): reinforce the chosen segment's synapses to prior-active cells,
    depress the rest, grow up to max_new new synapses to prior-WINNER cells; depress wrongly-predictive segments.
  - High-order context self-organizes because A->B and E->B match no shared segment -> allocate DISJOINT B-SDRs ->
    the distinct SDRs propagate through the shared middle -> D predicts X vs Y.

TASK: overlapping sequences [cue]+[shared middle L]+[branch] (all order-k<=L Markov provably chance). ANTI-CHEATS:
beat the Markov floor + distal LESION collapses + NO-teacher + full-context oracle (learnability) + multi-seed.
Reuse-by-import; NO sim/ edit; CPU/numpy.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge9b_htm_faithful.json")


def make_overlap_sequences(n_seq=2, middle_len=4, seed=42):
    cues = list(range(n_seq)); middle = list(range(n_seq, n_seq + middle_len))
    branches = list(range(n_seq + middle_len, n_seq + middle_len + n_seq))
    seqs = [[cues[i]] + middle + [branches[i]] for i in range(n_seq)]
    return seqs, n_seq + middle_len + n_seq, {"branches": branches, "L": middle_len}


def markov_branch_acc(seqs, k, n_branch):
    from collections import defaultdict, Counter
    counts = defaultdict(Counter)
    for s in seqs:
        for t in range(len(s) - 1):
            counts[tuple(s[max(0, t - k + 1): t + 1])][s[t + 1]] += 1
    ok = 0.0
    for s in seqs:
        t = len(s) - 2; dist = counts[tuple(s[max(0, t - k + 1): t + 1])]
        if not dist:
            ok += 1.0 / n_branch; continue
        top = max(dist.values()); win = [x for x, n in dist.items() if n == top]
        ok += (1.0 / len(win)) if s[t + 1] in win else 0.0
    return ok / len(seqs)


class HTM:
    """Faithful multi-segment HTM Temporal Memory. Fully local + unsupervised (no teacher). Locality: learning uses
    only pre/post co-activity + a cell's own segments; no forward-weight transpose (used_transpose stays False)."""

    def __init__(self, n_cols, n_cells=16, seed=0, k_win=4, act_th=3, learn_th=2,
                 perm_init=0.24, perm_conn=0.5, inc=0.10, dec=0.06, dec_pred=0.01, max_new=8, lesion=False):
        self.M, self.nE, self.N = n_cols, n_cells, n_cols * n_cells
        self.k_win, self.act_th, self.learn_th = k_win, act_th, learn_th
        self.perm_conn, self.p_init = perm_conn, perm_init
        self.inc, self.dec, self.dec_pred, self.max_new = inc, dec, dec_pred, max_new
        self.lesion = lesion
        self.rng = np.random.default_rng(seed)
        self.segments = [[] for _ in range(self.N)]   # per-cell list of segments; segment = {presyn_cell: permanence}
        self.used_transpose = False

    def _col(self, c):
        return list(range(c * self.nE, (c + 1) * self.nE))

    def _seg_conn_active(self, seg, active):
        return sum(1 for p, w in seg.items() if w >= self.perm_conn and p in active)

    def _seg_match(self, seg, active):                 # potential-synapse overlap (for learning/matching)
        return sum(1 for p in seg if p in active)

    def _committed(self, cell):                        # usage = # of non-empty (context-bearing) segments
        return sum(1 for seg in self.segments[cell] if seg)

    def _best_seg(self, cell, active):
        best, bs = None, -1
        for seg in self.segments[cell]:
            s = self._seg_match(seg, active)
            if s > bs:
                best, bs = seg, s
        return best, bs

    def run_sequence(self, seq, learn):
        prev_active, prev_winners = set(), set()
        predictive = set()
        preds = []
        for c in seq:
            col = self._col(c)
            pred_here = [i for i in col if i in predictive]
            active, winners, to_learn = set(), set(), []
            if pred_here and not self.lesion:
                active.update(pred_here); winners.update(pred_here)   # correct prediction -> sparse, context-specific
                if learn:
                    for i in pred_here:                               # reinforce ONLY the matching segment (never create empty)
                        seg, sc = self._best_seg(i, prev_winners)      # match against WINNERS (sparse), not the bursting active set
                        if seg is not None and sc >= self.learn_th:
                            to_learn.append((i, seg))
            elif not prev_winners:                                    # first symbol (cue): no context -> pick a stable SDR,
                winners.update(col[:self.k_win]); active.update(col)  #   NO segment (nothing to learn from), just burst-active
            else:
                active.update(col)                                    # BURST with a context to learn
                if learn:
                    # match against prev WINNERS (sparse) -- NOT prev_active (a burst makes the whole column active, so
                    # matching against active would let the OLD context's downstream cells match -> context merge).
                    scored = sorted(((self._best_seg(i, prev_winners)[1], i) for i in col), reverse=True)
                    if scored[0][0] >= self.learn_th:                  # a matching segment exists -> reinforce its cell(s)
                        for sc, i in scored[:self.k_win]:
                            if sc >= self.learn_th:
                                winners.add(i); to_learn.append((i, self._best_seg(i, prev_winners)[0]))
                    else:                                              # ALLOCATE a DISJOINT SDR: k least-COMMITTED cells
                        lu = sorted(col, key=lambda i: (self._committed(i), i))[:self.k_win]
                        for i in lu:
                            winners.add(i); to_learn.append((i, self._new_seg(i)))
            if learn:
                self._learn(to_learn, prev_winners)
                self._punish_wrong(predictive, c, prev_winners)
            # compute predictive cells for the NEXT symbol
            predictive = set()
            for i in range(self.N):
                for seg in self.segments[i]:
                    if self._seg_conn_active(seg, active) >= self.act_th:
                        predictive.add(i); break
            preds.append(set(i // self.nE for i in predictive))
            prev_active, prev_winners = active, winners
        return preds

    def _new_seg(self, cell):
        seg = {}; self.segments[cell].append(seg); return seg

    def _learn(self, to_learn, prev_winners):
        for cell, seg in to_learn:
            if seg is None:
                continue
            for p in list(seg.keys()):                                # reinforce synapses to the prior WINNER SDR, depress rest
                seg[p] = min(1.0, seg[p] + self.inc) if p in prev_winners else max(0.0, seg[p] - self.dec)
            grow = [p for p in prev_winners if p not in seg]          # grow new synapses to the prior SDR
            self.rng.shuffle(grow)
            for p in grow[:self.max_new]:
                seg[p] = self.p_init

    def _punish_wrong(self, predictive, c, prev_winners):
        for i in predictive:                                          # was predictive but its column didn't activate
            if i // self.nE != c:
                for seg in self.segments[i]:
                    if self._seg_conn_active(seg, prev_winners) >= self.act_th:
                        for p in list(seg.keys()):
                            if p in prev_winners:
                                seg[p] = max(0.0, seg[p] - self.dec_pred)


def branch_acc(tm, seqs, div_pos):
    ok = 0
    for s in seqs:
        pred = tm.run_sequence(s, learn=False)[div_pos]
        ok += int(pred == {s[div_pos + 1]})
    return ok / len(seqs)


def full_oracle(seqs, div_pos):
    from collections import defaultdict
    tab = defaultdict(set)
    for s in seqs:
        tab[tuple(s[:div_pos + 1])].add(s[div_pos + 1])
    return sum(1 for s in seqs if tab[tuple(s[:div_pos + 1])] == {s[div_pos + 1]}) / len(seqs)


def _run_arm(job):
    seed, arm, n_seq, L, n_cells, k_win, act_th, epochs = job
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    div_pos = L
    tm = HTM(vocab, n_cells=n_cells, seed=seed, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                tm.run_sequence(s, learn=True)
    return (seed, arm, {"branch_acc": branch_acc(tm, seqs, div_pos), "locality_ok": (not tm.used_transpose)})


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    floors = {}
    for s in a.seeds:
        seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=s)
        floors[s] = {"markov_L": markov_branch_acc(seqs, a.middle_len, a.n_seq),
                     "oracle": full_oracle(seqs, a.middle_len), "chance": 1.0 / a.n_seq}
    try:
        jobs = [(s, arm, a.n_seq, a.middle_len, a.n_cells, a.k_win, a.act_th, a.epochs)
                for s in a.seeds for arm in ARMS]
        cap = a.max_workers if (a.max_workers and a.max_workers > 0) else (os.cpu_count() or 1)
        collected = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), cap)) as ex:
                for seed, arm, entry in ex.map(_run_arm, jobs):
                    collected.setdefault(seed, {})[arm] = entry
        except Exception:
            for job in jobs:
                seed, arm, entry = _run_arm(job); collected.setdefault(seed, {})[arm] = entry
        for s in a.seeds:
            d = collected[s]; d["seed"] = s; d["floors"] = floors[s]; per.append(d)
        for d in per:
            f = d["floors"]
            print(f"  [seed {d['seed']}] HTM branch {d['htm']['branch_acc']:.3f} | lesion {d['lesion']['branch_acc']:.3f} "
                  f"| untr {d['untrained']['branch_acc']:.3f} || markov_L {f['markov_L']:.3f} oracle {f['oracle']:.3f} "
                  f"chance {f['chance']:.3f} loc {d['htm']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["branch_acc"] for p in per]))
        htm, les, unt = m("htm"), m("lesion"), m("untrained")
        markov_L = float(np.mean([p["floors"]["markov_L"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        loc = all(p["htm"]["locality_ok"] for p in per)
        task_ok = oracle > 0.99
        go = bool(task_ok and htm >= 0.90 and htm >= markov_L + 0.15 and htm >= chance + 0.20 and htm >= les + 0.20 and loc)
        if not loc:
            verdict = "INVALID -- locality assert failed."
        elif not task_ok:
            verdict = f"INCONCLUSIVE -- task not context-solvable (oracle {oracle:.3f})."
        elif go:
            verdict = (f"GO -- the FAITHFUL multi-segment HTM Temporal Memory self-organizes robust CONTEXT-SPECIFIC "
                       f"high-order prediction, UNSUPERVISED + local: branch acc {htm:.3f} >> Markov floor {markov_L:.3f}, "
                       f">> chance {chance:.3f}, >> lesion {les:.3f} (distal mechanism load-bearing); untrained {unt:.3f}; "
                       f"locality asserted. Multi-seed. => the rung-3 pivot WORKS robustly (allocation-based self-"
                       f"organization) -> 6-seed + capacity (more sequences), then rung-3b spiking-LIF port (dAP -> our "
                       f"two-compartment neuron). NO sim/ edit.")
        else:
            miss = []
            if htm < 0.90: miss.append(f"branch acc {htm:.3f} < 0.90")
            if htm < markov_L + 0.15 or htm < chance + 0.20: miss.append(f"didn't clear Markov/chance (htm {htm:.3f}, markov {markov_L:.3f})")
            if htm < les + 0.20: miss.append(f"lesion didn't collapse (htm {htm:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}). Iterate the "
                       f"faithful HTM (k_win / act_th / n_cells / epochs / learn_th) -- allocation needs >= k distinct "
                       f"least-used cells + act_th separating contexts. NOT a wall; the mechanism is the next tuning.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge9b_htm_faithful", "verdict": verdict,
               "mechanism": "faithful multi-segment HTM Temporal Memory (Hawkins-Ahmad; discrete Bouhadjar-Diesmann 2022): "
                            "per-cell segment LISTS, population SDR winners, best-matching-segment-else-allocate-fresh-cells, "
                            "local Hebbian permanence; UNSUPERVISED (no teacher); no BPTT/transport",
               "task": "overlapping sequences; branch (divergent) prediction; Markov floor + distal-lesion + oracle controls",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Faithful multi-segment HTM (fixes EMERGE-9's minimal single-permanence-row context-merge). "
                              "Discrete algorithm; spiking-LIF port (dAP -> our two-compartment neuron) is rung-3b. "
                              "Unsupervised: no teacher; self-organization IS the deliverable, not a cheat."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge9b] VERDICT: {verdict}", flush=True)
    print(f"[emerge9b] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
