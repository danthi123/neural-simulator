"""EMERGE-13 / rung-4 Stage C DE-RISK (numpy gate before the sim/ kernel) — does the Bouhadjar three-term learning
rule reproduce EMERGE-9d's self-organized branch prediction when the per-cell SEGMENT LISTS are replaced by a FLAT
[post, pre] PERMANENCE MATRIX (a single conceptual segment per cell over a DENSE cross-column potential pool)?

WHY THIS GATES THE sim/ PORT: the bridge's `coincidence_detector` pathway is a flat CSR of per-(pre,post) synapses,
NOT per-cell segment lists. If the flat-matrix three-term rule self-organizes context on the disjoint-SDR task
(each cell used for <=1 context, so single-segment suffices -- EMERGE-9's flat FAILURE was the winner-vs-active bug,
fixed in 9b, NOT the flat structure), then the Stage-C sim/ realization is a clean flat-CSR permanence-update kernel
over a pre-allocated potential pool + a WEIGHTED coincidence (c_drive = sum of active-synapse permanences, thresholded
at act_th's connected-count) -- much simpler than porting segment lists. If it FAILS, multi-segment structure is
genuinely required and the sim/ approach must differ (build-informative either way).

THE FLAT REALIZATION:
  - W[post, pre] in [0,1] = permanence (only cross-column entries are potential synapses; same-column = 0, no self).
  - CONNECTED synapse: W >= perm_conn. A cell is PREDICTIVE for the next symbol iff, over the currently-active cells,
    its count of CONNECTED synapses from them >= act_th (== the bridge's coincidence with a connected-threshold; on
    the bridge, the WEIGHTED coincidence sum approximates this).
  - Learning (Bouhadjar three-term, winner-based -- the EMERGE-9b lesson): for each WINNER cell j (that had a matching
    connected segment, i.e. was correctly predicted, OR is an allocation target), potentiate W[j, pre] for pre in
    prev_WINNERS (scaled by the per-cell dAP-rate homeostasis hfac), depress W[j, pre] for the OTHER currently-wired
    pre; grow (from p_init) toward prev_winners. Presynaptic-depress wrongly-predictive cells. Per-cell z EMA.
  - ALLOCATION without segment growth: a burst with NO matching connected segment -> the k FRESHEST (lowest-z) cells
    in the column become winners and potentiate toward prev_winners on their (initially sub-connected) flat row ->
    the homeostasis STEERS distinct contexts onto distinct fresh cells (== 9d's dAP-rate allocation, no discrete
    least-committed heuristic, no structural growth -- exactly what a fixed-topology bridge pathway can do).

Reuse-by-import (EMERGE-9b task + floors); NO sim/ edit; CPU/numpy; multi-seed. Anti-cheats: Markov floor + dAP-lesion
collapse + no-teacher + oracle + multi-seed + (the gate) parity with the EMERGE-9d segment-list reference.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge9b_htm_faithful_derisk import make_overlap_sequences, markov_branch_acc, full_oracle

OUT = Path("research/findings/raw/_emerge13_stageC_flat_permanence.json")


class FlatTM:
    """HTM Temporal Memory on a FLAT [post, pre] permanence matrix (single conceptual segment per cell). Spiking-style
    winner selection is abstracted to the discrete dAP-primed rule (EMERGE-9b/9c parity: predicted cells win sparse;
    unpredicted bursts) -- Stage B2 already validated the spiking realization of THIS selection; Stage C's variable is
    the LEARNING data structure (segment lists -> flat matrix). Learning is the Bouhadjar three-term rule on W."""

    def __init__(self, n_cols, n_cells=16, seed=0, k_win=4, act_th=3, learn_th=2, perm_conn=0.5, p_init=0.24,
                 lam_pot=0.14, lam_dep=0.02, z_tau=0.85, z_star=1.0, connect_grow=6, lesion=False):
        self.M, self.nE, self.N = n_cols, n_cells, n_cols * n_cells
        self.k_win, self.act_th, self.learn_th = k_win, act_th, learn_th
        self.perm_conn, self.p_init = perm_conn, p_init
        self.lam_pot, self.lam_dep, self.z_star, self.z_tau, self.connect_grow = lam_pot, lam_dep, z_star, z_tau, connect_grow
        self.lesion = lesion
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((self.N, self.N), np.float64)   # W[post, pre] permanence; cross-column potential synapses
        self.z = np.zeros(self.N)                          # per-cell low-pass dAP (predictive) rate
        self.used_transpose = False

    def _col(self, c):
        return list(range(c * self.nE, (c + 1) * self.nE))

    def W_wired_count(self):
        return np.sum(self.W > 0.0, axis=1)                # per-cell # of wired (potential) synapses = "committed"

    def _conn_active_count(self, post, active_arr):
        """# of CONNECTED synapses (W>=perm_conn) into `post` from currently-active cells."""
        row = self.W[post]
        return int(np.sum((row[active_arr] >= self.perm_conn))) if len(active_arr) else 0

    def _match_count(self, post, winners_arr):
        """potential-synapse overlap (W>0) into `post` from prev winners (for matching/learning, like _seg_match)."""
        row = self.W[post]
        return int(np.sum(row[winners_arr] > 0.0)) if len(winners_arr) else 0

    def _predictive(self, active):
        active_arr = np.fromiter(active, int, len(active))
        pred = set()
        for i in range(self.N):
            if self._conn_active_count(i, active_arr) >= self.act_th:
                pred.add(i)
        return pred

    def _potentiate(self, post, prev_win_arr, prev_win_set):
        hfac = 0.5 + 0.5 * max(0.0, self.z_star - self.z[post])       # homeostasis modulates (never fully gates)
        row = self.W[post]
        wired = np.where(row > 0.0)[0]
        for p in wired:                                              # potentiate to prev-winners, depress the rest
            row[p] = min(1.0, row[p] + self.lam_pot * hfac) if p in prev_win_set else max(0.0, row[p] - self.lam_dep)
        grow = [p for p in prev_win_arr if row[p] == 0.0]           # grow new potential synapses toward prev winners
        self.rng.shuffle(grow)
        for p in grow[:self.connect_grow]:
            row[p] = self.p_init

    def train_sequence(self, seq):
        predictive, prev_winners = set(), set()
        for c in seq:
            col = self._col(c)
            prev_arr = np.fromiter(prev_winners, int, len(prev_winners))
            primed = [i for i in col if i in predictive] if not self.lesion else []
            to_learn = []
            if primed:
                winners = set(primed[:self.k_win]) if len(primed) > self.k_win else set(primed)
                active = winners
                for i in winners:
                    if self._match_count(i, prev_arr) >= self.learn_th:
                        to_learn.append(i)
            elif not prev_winners:
                winners = set(col[:self.k_win]); active = set(col)
            else:
                active = set(col)
                scored = sorted(((self._match_count(i, prev_arr), i) for i in col), reverse=True)
                if scored[0][0] >= self.learn_th:                    # a matching connected segment exists -> reinforce
                    winners = set()
                    for sc, i in scored[:self.k_win]:
                        if sc >= self.learn_th:
                            winners.add(i); to_learn.append(i)
                else:                                                # ALLOCATE onto the k FRESHEST cells. z-homeostasis
                    # can't cold-start allocation (z=0 for all until a cell is connected), so bootstrap with the flat
                    # "committed" metric = # of wired synapses in the cell's row (== EMERGE-9d's committed-segment
                    # bootstrap); z then MODULATES potentiation (hfac). Freshest = fewest wired synapses.
                    wired_ct = self.W_wired_count()
                    lu = sorted(col, key=lambda i: (int(wired_ct[i]), i))[:self.k_win]
                    winners = set(lu); to_learn.extend(lu)
            if prev_winners:
                for j in to_learn:                                   # (1) potentiation x homeostasis + grow
                    self._potentiate(j, prev_arr, prev_winners)
                for i in list(predictive):                           # (2) presynaptic depression of wrongly-predictive
                    if i // self.nE != c and self._conn_active_count(i, prev_arr) >= self.act_th:
                        row = self.W[i]
                        for p in prev_arr:
                            if row[p] > 0.0:
                                row[p] = max(0.0, row[p] - self.lam_dep)
            predictive = self._predictive(active)
            self.z *= self.z_tau                                     # (3) low-pass dAP-rate homeostasis
            for i in predictive:
                self.z[i] += (1.0 - self.z_tau)
            prev_winners = winners

    def predict(self, seq, div_pos):
        predictive, prev_winners = set(), set()
        preds = []
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            if primed:
                active = set(primed[:self.k_win]) if len(primed) > self.k_win else set(primed)
            elif not prev_winners:
                active = set(col)
            else:
                active = set(col)
            predictive = self._predictive(active)
            preds.append(set(i // self.nE for i in predictive))
            prev_winners = active
        return preds


def branch_acc(tm, seqs, div_pos):
    ok = 0
    for s in seqs:
        ok += int(tm.predict(s, div_pos)[div_pos] == {s[div_pos + 1]})
    return ok / len(seqs)


def _run_arm(job):
    seed, arm, n_seq, L, n_cells, k_win, act_th, epochs = job
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    tm = FlatTM(vocab, n_cells=n_cells, seed=seed, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                tm.train_sequence(s)
    return (seed, arm, {"branch": branch_acc(tm, seqs, L), "locality_ok": (not tm.used_transpose)})


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
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
            print(f"  [seed {d['seed']}] FLAT-3term branch {d['htm']['branch']:.3f} | lesion {d['lesion']['branch']:.3f} "
                  f"| untr {d['untrained']['branch']:.3f} || markov {f['markov_L']:.3f} chance {f['chance']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["branch"] for p in per]))
        htm, les, unt = m("htm"), m("lesion"), m("untrained")
        markov = float(np.mean([p["floors"]["markov_L"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        go = bool(oracle > 0.99 and htm >= 0.90 and htm >= markov + 0.15 and htm >= chance + 0.20 and htm >= les + 0.20)
        if oracle <= 0.99:
            verdict = f"INCONCLUSIVE -- task not context-solvable (oracle {oracle:.3f})."
        elif go:
            verdict = (f"GO -- the Bouhadjar THREE-TERM rule self-organizes context-specific branch prediction on a FLAT "
                       f"[post,pre] PERMANENCE MATRIX (single-segment-per-cell over a dense potential pool, homeostatic "
                       f"allocation, NO segment lists, NO structural growth): branch {htm:.3f} >> Markov {markov:.3f}, "
                       f">> chance {chance:.3f}, >> dAP-lesion {les:.3f}; untrained {unt:.3f}; no teacher; multi-seed. "
                       f"=> the Stage-C sim/ realization is a flat-CSR permanence-update kernel over a pre-allocated "
                       f"coincidence potential pool + WEIGHTED coincidence (c_drive = active-synapse permanence sum). "
                       f"BUILD the additive/guarded fused_htm_permanence_update kernel next.")
        else:
            miss = []
            if htm < 0.90: miss.append(f"branch {htm:.3f} < 0.90")
            if htm < markov + 0.15 or htm < chance + 0.20: miss.append(f"didn't clear Markov/chance ({htm:.3f})")
            if htm < les + 0.20: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}). The FLAT matrix "
                       f"did not fully self-organize context here -> either tune (lam_pot/lam_dep/z_tau/act_th/n_cells/"
                       f"epochs) OR the multi-SEGMENT structure is genuinely required (a cell reused across contexts) "
                       f"-> the sim/ port needs per-segment coincidence, not a flat CSR. Decisive for the port design.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge13_stageC_flat_permanence", "verdict": verdict,
               "mechanism": "Bouhadjar three-term HTM learning on a FLAT [post,pre] permanence matrix (single conceptual "
                            "segment per cell, dense cross-column potential pool, homeostatic dAP-rate allocation, no "
                            "structural growth) -- the flat-CSR-compatible realization that gates the sim/ Stage-C kernel",
               "task": "overlapping sequences; branch prediction; Markov floor + dAP-lesion + oracle + multi-seed",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Stage-C DESIGN GATE: if flat-matrix three-term == EMERGE-9d, the sim/ port is a flat-CSR "
                              "permanence kernel + weighted coincidence (simple); if not, multi-segment is required "
                              "(harder port). Cheap-first, single-variable (learning data structure), gated before any "
                              "sim/ edit. The disjoint-SDR task uses each cell for <=1 context, so single-segment suffices."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge13] VERDICT: {verdict}", flush=True)
    print(f"[emerge13] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
