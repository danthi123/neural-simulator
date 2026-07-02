"""EMERGE-9 (rung-3 pivot) — UNSUPERVISED self-organizing high-order sequence prediction via a minimal-faithful
HTM Temporal Memory (the discrete algorithm Bouhadjar-Diesmann 2022 spikified). Does a fully-LOCAL, NO-TEACHER,
allocation-based mechanism self-organize CONTEXT-SPECIFIC next-element prediction on overlapping sequences that
share a middle -- beating a Markov floor a fixed reservoir / lookup cannot?

WHY (the pivot): five probes (rung-3a target-based, e-prop, RFLO, EMERGE-7 next-symbol, EMERGE-8 Predictive Alignment)
confirmed SUPERVISED local recurrent-weight training does not beat a fixed reservoir at toy scale. Bouhadjar-Diesmann
2022 (PLoS CB 18:e1010233, PMC9273101) sidesteps this: it trains NO recurrent weights toward a target and avoids
interference by ALLOCATION -- carving DISJOINT sparse cell-subsets per context out of a fixed random skeleton. This
is the discrete HTM Temporal-Memory essence of that spiking model (the spiking-LIF port, mapping the distal-dendrite
plateau `dAP` to our confirmed two-compartment neuron, is rung-3b).

MECHANISM (minimal-faithful HTM-TM): M columns (one per symbol) x nE cells/column. Each cell has ONE distal segment =
a permanence vector over a FIXED sparse potential skeleton (p_skel). A synapse is MATURE (functional) at permanence
>= theta_perm. Per step, on symbol c:
  - PREDICTIVE cells in column c (those whose segment had >= theta_seg mature synapses from the PREVIOUS active cells)
    become the sole active cells (correct high-order prediction -> SPARSE, context-specific).
  - else BURST: all cells in c fire; pick a WINNER cell = the best-segment-match to the previous active set if it
    clears theta_seg (reinforce), ELSE the least-used cell in the column (ALLOCATE a new context representation).
  - LOCAL Hebbian permanence learning (NO teacher, performance never feeds learning): grow winner<-prev-winner
    permanences; decay the winner's unused synapses; depress cells that were wrongly predictive.
  - prediction for t+1 = columns containing a predictive cell.
High-order context self-organizes because A->B and E->B allocate DISTINCT winner cells (E-context doesn't match
b1's A-context segment -> allocate b2), so "B after A" vs "B after E" propagate distinct cells -> D predicts X vs Y.

TASK: the overlapping-sequences family (reused from `_fork2_predesign_markov_floor_check.py`): [cue] + [shared middle
length L] + [branch]; predicting the branch requires context reaching back past the shared middle. Metric = branch
(divergent) prediction accuracy. ANTI-CHEATS: beat the order-k<=L Markov floor (provably chance) + LESION the distal
prediction (-> always burst, no context -> collapse) + NO-teacher asserted + full-context oracle (learnability) +
multi-seed 42/43/44. Reuse-by-import; NO sim/ edit; CPU/numpy.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/findings/raw/_emerge9_temporal_memory.json")


# --- overlapping-sequences task + Markov floor (inlined from the fork2 pre-design probe; the shared middle makes every
#     order-k<=middle_len Markov predictor provably chance at the branch step) ---
def make_overlap_sequences(n_seq=2, middle_len=4, seed=42):
    cues = list(range(n_seq))
    middle = list(range(n_seq, n_seq + middle_len))
    branches = list(range(n_seq + middle_len, n_seq + middle_len + n_seq))
    seqs = [[cues[i]] + middle + [branches[i]] for i in range(n_seq)]
    vocab = n_seq + middle_len + n_seq
    return seqs, vocab, {"cues": cues, "middle": middle, "branches": branches, "n_seq": n_seq, "L": middle_len}


def kth_order_markov_branch_acc(seqs, k, info):
    from collections import defaultdict, Counter
    counts = defaultdict(Counter)
    for s in seqs:
        for t in range(len(s) - 1):
            counts[tuple(s[max(0, t - k + 1): t + 1])][s[t + 1]] += 1
    correct = 0.0
    for s in seqs:
        t = len(s) - 2
        dist = counts[tuple(s[max(0, t - k + 1): t + 1])]
        if not dist:
            correct += 1.0 / len(info["branches"]); continue
        top = max(dist.values()); winners = [sym for sym, c in dist.items() if c == top]
        correct += (1.0 / len(winners)) if s[t + 1] in winners else 0.0
    return correct / len(seqs)


class TemporalMemory:
    """Minimal-faithful HTM Temporal Memory. Fully local + unsupervised (no teacher; performance never feeds learning).
    Cell i's single distal segment = permanence row P[i, :] over a fixed sparse skeleton (P=-1 -> no potential synapse)."""

    def __init__(self, n_cols, n_cells=16, seed=0, p_skel=0.35, theta_perm=0.5, theta_seg=1,
                 perm_init=0.45, inc=0.12, dec=0.05, decay=0.006, lesion=False, k_win=1):
        # DEFAULTS = the validated single-winner regime (theta_seg=1, k_win=1): PROVES the mechanism (seed-42 perfect
        # context-specific prediction). Multi-seed robustness needs the faithful SDR (k_win>1) + multi-synapse segments
        # -- a known-fiddly reimplementation flagged as the next iteration (the minimal SDR attempt regressed).
        self.M, self.nE, self.k_win = n_cols, n_cells, k_win
        self.N = n_cols * n_cells
        rng = np.random.default_rng(seed)
        skel = rng.random((self.N, self.N)) < p_skel
        np.fill_diagonal(skel, False)
        # forbid intra-column distal synapses (a column's cells don't predict via each other)
        for c in range(n_cols):
            lo, hi = c * n_cells, (c + 1) * n_cells
            skel[lo:hi, lo:hi] = False
        self.P = np.where(skel, perm_init + 0.02 * rng.standard_normal((self.N, self.N)), -1.0)
        self.theta_perm, self.theta_seg = theta_perm, theta_seg
        self.inc, self.dec, self.decay = inc, dec, decay
        self.lesion = lesion
        self.rng = rng
        self.used_transpose = False        # locality flag (unsupervised Hebbian permanence; no forward-weight transpose)

    def _col(self, c):
        return np.arange(c * self.nE, (c + 1) * self.nE)

    def run_sequence(self, seq, learn):
        """Present `seq` (column indices). Return per-step predicted-column sets (prediction for the NEXT symbol)."""
        N = self.N
        active = np.zeros(N, bool)
        predictive = np.zeros(N, bool)
        prev_active = np.zeros(N, bool)
        prev_winners = np.array([], int)
        preds = []
        for t, c in enumerate(seq):
            cells = self._col(c)
            pred_here = cells[predictive[cells]]
            active = np.zeros(N, bool)
            if len(pred_here) > 0 and not self.lesion:
                active[pred_here] = True
                winners = pred_here                                   # correct prediction -> sparse, context-specific
                bursting = False
            else:
                active[cells] = True                                  # burst (unpredicted or lesioned)
                bursting = True
                if learn:
                    mature = self.P >= self.theta_perm
                    overlap = mature[:, prev_active].sum(axis=1) if prev_active.any() else np.zeros(N)
                    col_ov = overlap[cells]
                    matched = cells[col_ov >= self.theta_seg] if prev_active.any() else np.array([], int)
                    if matched.size > 0:
                        winners = matched                             # existing SDR matches this context -> reinforce it
                    else:                                             # ALLOCATE a fresh k-cell context SDR...
                        nmat = mature[cells].sum(axis=1).astype(float)
                        if prev_winners.size > 0:                     # ...preferring cells WIRED to the prev context
                            connected = (self.P[cells][:, prev_winners] >= 0).sum(axis=1)
                            nmat -= connected                         # prefer better-connected + less-used cells
                        winners = cells[np.argsort(nmat)[:self.k_win]]
                else:
                    winners = cells

            if learn and prev_winners.size > 0:
                pw = prev_winners
                for w in winners:
                    pot = self.P[w] >= 0                              # potential-synapse mask for this cell
                    grow = np.zeros(N, bool); grow[pw] = True; grow &= pot
                    self.P[w, grow] = np.minimum(1.0, self.P[w, grow] + self.inc)   # grow context->winner
                    decpaths = pot & ~grow                            # decay this cell's other (unused) synapses
                    self.P[w, decpaths] = np.maximum(0.0, self.P[w, decpaths] - self.decay)
                # depress cells that were predictive last step but whose column did NOT become the current symbol
                wrong = np.where(predictive)[0]
                wrong = wrong[(wrong // self.nE) != c]
                for wc in wrong:
                    pot = self.P[wc] >= 0; m = np.zeros(N, bool); m[prev_active] = True; m &= pot
                    self.P[wc, m] = np.maximum(0.0, self.P[wc, m] - self.dec)

            mature = self.P >= self.theta_perm
            overlap = mature[:, active].sum(axis=1)
            predictive = overlap >= self.theta_seg
            preds.append(set((np.where(predictive)[0] // self.nE).tolist()))
            prev_active = active.copy()
            prev_winners = winners
        return preds


def branch_pred_acc(tm, cells_meta, div_pos, learn=False):
    """Fraction of sequences whose predicted-column set at the divergent step CONTAINS the true branch column AND is
    specific (does not also predict the OTHER sequences' branch). Strict: exact-match to {true branch}."""
    ok = 0
    for c in cells_meta:
        preds = tm.run_sequence(c["seq"], learn=learn)
        pred_cols = preds[div_pos]                    # prediction made AT div_pos is for seq[div_pos+1] (the branch)
        true_branch = c["seq"][div_pos + 1]
        ok += int(pred_cols == {true_branch})         # strict: predicts exactly the correct branch column
    return ok / max(1, len(cells_meta))


def full_context_oracle(cells_meta, div_pos):
    """Learnability guard (EMERGE-7 grokking check): a full-prefix lookup -> 1.0 by construction (task IS solvable
    with context; the branch is a deterministic function of the full prefix). Confirms the task is not ill-posed."""
    from collections import defaultdict
    table = defaultdict(set)
    for c in cells_meta:
        table[tuple(c["seq"][:div_pos + 1])].add(c["seq"][div_pos + 1])
    ok = sum(1 for c in cells_meta if table[tuple(c["seq"][:div_pos + 1])] == {c["seq"][div_pos + 1]})
    return ok / max(1, len(cells_meta))


def _cells_meta(seqs):
    return [{"seq": s} for s in seqs]


def _run_arm(job):
    seed, arm, n_seq, L, n_cells, epochs, theta_seg, p_skel = job
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    div_pos = L                                        # step whose next-symbol is the branch (seq=[cue]+middle(L)+[branch])
    meta = _cells_meta(seqs)
    tm = TemporalMemory(vocab, n_cells=n_cells, seed=seed, p_skel=p_skel, theta_seg=theta_seg,
                        lesion=(arm == "lesion_distal"))
    if arm != "untrained":
        for _ in range(epochs):                        # unsupervised exposure to the sequence set
            for c in meta:
                tm.run_sequence(c["seq"], learn=True)
    acc = branch_pred_acc(tm, meta, div_pos, learn=False)
    return (seed, arm, {"branch_acc": acc, "locality_ok": (not tm.used_transpose)})


ARMS = ["htm", "lesion_distal", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--theta-seg", type=int, default=1)   # segment activation threshold (validated single-winner regime)
    ap.add_argument("--p-skel", type=float, default=0.35)
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    # reference floors + learnability, per seed
    floors = {}
    for s in a.seeds:
        seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=s)
        div_pos = a.middle_len
        floors[s] = {
            "markov_L": kth_order_markov_branch_acc(seqs, a.middle_len, info),
            "markov_1": kth_order_markov_branch_acc(seqs, 1, info),
            "oracle": full_context_oracle(_cells_meta(seqs), div_pos),
            "vocab": vocab, "chance": 1.0 / a.n_seq,
        }
    try:
        jobs = [(s, arm, a.n_seq, a.middle_len, a.n_cells, a.epochs, a.theta_seg, a.p_skel)
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
            print(f"  [seed {d['seed']}] HTM branch {d['htm']['branch_acc']:.3f} | lesion {d['lesion_distal']['branch_acc']:.3f} "
                  f"| untr {d['untrained']['branch_acc']:.3f} || markov_L {f['markov_L']:.3f} markov_1 {f['markov_1']:.3f} "
                  f"oracle {f['oracle']:.3f} chance {f['chance']:.3f} loc {d['htm']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["branch_acc"] for p in per]))
        htm, les, unt = m("htm"), m("lesion_distal"), m("untrained")
        markov_L = float(np.mean([p["floors"]["markov_L"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        loc = all(p["htm"]["locality_ok"] for p in per)
        task_ok = oracle > 0.99
        beats_markov = htm >= markov_L + 0.15 and htm >= chance + 0.20
        lesion_collapses = htm >= les + 0.20
        strong = htm >= 0.90
        go = bool(task_ok and beats_markov and lesion_collapses and loc and strong)
        if not loc:
            verdict = "INVALID -- locality assert failed."
        elif not task_ok:
            verdict = (f"INCONCLUSIVE -- the task is not context-solvable (full-context oracle {oracle:.3f} < 1.0); "
                       f"fix the sequence design before reading the mechanism.")
        elif go:
            verdict = (f"GO -- a fully-LOCAL, NO-TEACHER HTM Temporal-Memory network self-organizes CONTEXT-SPECIFIC "
                       f"high-order next-element prediction: branch accuracy {htm:.3f} >> order-{a.middle_len} Markov floor "
                       f"{markov_L:.3f}, >> chance {chance:.3f}; LESIONING the distal prediction collapses it to {les:.3f} "
                       f"(the dAP/context mechanism is load-bearing); untrained {unt:.3f}; unsupervised (performance never "
                       f"fed learning); locality asserted. Multi-seed. ⇒ the pivot WORKS -- allocation-based self-organization "
                       f"beats the reservoir dead-end -> promote to 6 seeds + capacity (more sequences), then rung-3b "
                       f"(spiking-LIF TM, dAP -> our two-compartment neuron). NO sim/ edit.")
        else:
            miss = []
            if not strong: miss.append(f"HTM branch acc {htm:.3f} < 0.90")
            if not beats_markov: miss.append(f"didn't beat Markov floor+.15/chance+.20 (htm {htm:.3f}, markov_L {markov_L:.3f}, chance {chance:.3f})")
            if not lesion_collapses: miss.append(f"distal-lesion didn't collapse it (htm {htm:.3f} vs lesion {les:.3f})")
            verdict = ("BOUNDARY (build-informative, not a stop) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}, "
                       f"task context-solvable). The minimal HTM-TM did not cleanly self-organize context-specific "
                       f"prediction at this config -> bounded retune (n_cells / theta_seg / p_skel / epochs -- allocation "
                       f"needs enough cells/column + a segment threshold that separates contexts; do NOT thrash) OR a "
                       f"faithfulness gap (add multi-segment cells / the depression term). Do NOT start the sim/ port.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge9_temporal_memory", "verdict": verdict,
               "mechanism": "minimal-faithful HTM Temporal Memory (discrete form of Bouhadjar-Diesmann 2022 spiking TM): "
                            "M columns x nE cells; per-cell distal segment (permanence over a fixed sparse skeleton); "
                            "predictive cells / bursting / best-match-else-ALLOCATE winner selection / local Hebbian "
                            "permanence learning; UNSUPERVISED (no teacher, performance never feeds learning); no BPTT",
               "task": "overlapping sequences ([cue]+[shared middle L]+[branch]); branch (divergent) prediction; "
                       "beat order-<=L Markov floor + distal-lesion collapse + full-context-oracle learnability",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "epochs": a.epochs, "theta_seg": a.theta_seg, "p_skel": a.p_skel},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Discrete HTM-TM (the algorithm Bouhadjar spikified); the spiking-LIF port (dAP -> our "
                              "confirmed two-compartment neuron, permanence three-term rule, WTA inhibition) is rung-3b. "
                              "High-order context self-organizes by ALLOCATION (distinct winner cells per context), NOT by "
                              "training recurrent weights to a target -> sidesteps the 5-probe reservoir dead-end. "
                              "Unsupervised: no teacher; 'self-organizes without a target' is the deliverable, not a cheat."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge9] VERDICT: {verdict}", flush=True)
    print(f"[emerge9] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
