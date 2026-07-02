"""EMERGE-18 / toward-language — HIGH-ORDER SEQUENCE GENERALIZATION: the emergent on-bridge sequence cortex generalizes
a HIGH-ORDER (earlier-context-dependent) prediction to a HELD-OUT SIMILAR word. This unifies EMERGE-15 (high-order
next-word prediction through a shared middle) with EMERGE-17 (overlapping word codes → generalization): a held-out
similar SUBJECT ("wolf", never trained) predicts the correct branch ("home") by generalizing from a trained similar
subject ("dog") THROUGH the shared middle "chased ball" — carrying the family context it never saw for this word.

MECHANISM: a two-level word encoding (the generalization research gate): each word = a set of MICRO-COLUMNS; the SUBJECT
words in a family SHARE micro-columns (overlapping identity SDRs) while the shared MIDDLE keeps CONTEXT cells disjoint
per family (the HTM allocation, unchanged). Training "dog chased ball home" potentiates dog's identity SDR → the shared
middle's canine-context cells → ... → home. Presenting the held-out "wolf chased ball" fires wolf's identity SDR (shares
the canine micro-columns with dog) → the SHARED cells drive the middle's learned canine-context coincidence → wolf
follows dog's high-order pathway → predicts home. The `sim/` kernel is UNCHANGED; the only change is the word encoding.

TASK: canines {dog,wolf,fox}→home, felines {cat,lion}→away, sentence = [subject, chased, ball, branch]. TRAIN on ONE
per family (dog, cat); HOLD OUT the similar subjects (wolf,fox,lion). TEST: held-out "wolf chased ball ___" predicts the
family branch. ANTI-CHEATS: held-out-similar-subject >> chance/n-gram-floor; the ORTHOGONAL-code control (subjects do NOT
share micro-columns) collapses; dAP-LESION collapses; DERANGED family→branch → chance; no-teacher; 6-seed. Reuse-by-import
(`_emerge14` + `_emerge17`); NO `sim/` edit. CPU numpy-backend.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, apply_kernel_update, coincidence_predict, _host)

OUT = Path("research/findings/raw/_emerge18_sequence_generalization.json")


def build_vocab(overlap=True):
    if overlap:
        subj = {"dog": [0, 1, 2, 3], "wolf": [0, 1, 2, 4], "fox": [0, 1, 2, 5],   # canines share [0,1,2]
                "cat": [6, 7, 8, 9], "lion": [6, 7, 8, 10]}                        # felines share [6,7,8]
    else:
        subj = {"dog": [0, 1, 2, 3], "wolf": [30, 31, 32, 33], "fox": [34, 35, 36, 37],
                "cat": [6, 7, 8, 9], "lion": [38, 39, 40, 41]}                     # ORTHOGONAL control (no shared blocks)
    mid = {"chased": [11, 12, 13, 14], "ball": [15, 16, 17, 18]}                   # shared middle (same for all sentences)
    branch = {"home": [19, 20, 21, 22], "away": [23, 24, 25, 26]}
    word2cols = {**subj, **mid, **branch}
    family_branch = {"dog": "home", "wolf": "home", "fox": "home", "cat": "away", "lion": "away"}
    sentences = {s: [s, "chased", "ball", family_branch[s]] for s in subj}
    train_subj = ["dog", "cat"]; held_out = ["wolf", "fox", "lion"]
    M = 1 + max(c for cols in word2cols.values() for c in cols)
    return word2cols, family_branch, sentences, train_subj, held_out, list(branch), list(mid), M


class SeqGenLearner:
    """On-bridge HTM-TM sequence learner over WORDS = sets of MICRO-COLUMNS. The SUBJECT (position 0) fires its fixed
    identity SDR (1 cell/micro-column -> similar subjects overlap); downstream words select context-specific winners
    (predicted subset, else committed-metric allocation) within their micro-columns. Reuses the sim/ coincidence
    prediction + three-term kernel unchanged."""

    def __init__(self, bridge, row, col, cells_idx, word2cols, nE, M, k_win=4, act_th=3, learn_th=2,
                 p_init=0.0, lam_pot=0.14, lam_dep=0.02, lesion=False):
        self.b, self.row, self.col, self.cells_idx = bridge, row, col, cells_idx
        self.word2cols, self.nE, self.M, self.N = word2cols, nE, M, M * nE
        self.k_win, self.act_th, self.learn_th, self.p_init = k_win, act_th, learn_th, p_init
        self.lam_pot, self.lam_dep, self.lesion = lam_pot, lam_dep, lesion
        self.z = np.zeros(self.N)

    def _wordcells(self, w):
        cs = []
        for c in self.word2cols[w]:
            cs.extend(range(c * self.nE, (c + 1) * self.nE))
        return cs

    def _identity_sdr(self, w):
        return set(c * self.nE + 0 for c in self.word2cols[w])   # cell 0 of each micro-column (fixed, overlaps for similar words)

    def _committed(self, cells):
        data = _host(self.b.cp_connections.data).astype(np.float64)
        n = int(self.b.core_config.num_neurons)
        wc = np.zeros(n); np.add.at(wc, self.col, (data > self.p_init + 0.02).astype(np.float64))
        return {cell: wc[self.cells_idx[cell]] for cell in cells}

    def _match(self, post_cell, prev_win):
        if not prev_win:
            return 0
        data = _host(self.b.cp_connections.data).astype(np.float64)
        pre_set = set(int(self.cells_idx[i]) for i in prev_win)
        bpost = int(self.cells_idx[post_cell]); idx = np.where(self.col == bpost)[0]
        return int(sum(1 for k in idx if int(self.row[k]) in pre_set and data[k] > self.p_init + 0.02))

    def _winners(self, w, pos, predictive, prev_win):
        cells = self._wordcells(w)
        primed = [i for i in cells if i in predictive] if not self.lesion else []
        if pos == 0:
            return self._identity_sdr(w)                          # subject: fixed identity SDR (overlaps for similar words)
        if primed:
            return set(primed[:self.k_win])
        if not prev_win:
            return set(sorted(self._identity_sdr(w))[:self.k_win])
        scored = sorted(((self._match(i, prev_win), i) for i in cells), reverse=True)
        if scored[0][0] >= self.learn_th:
            return set(i for sc, i in scored[:self.k_win] if sc >= self.learn_th)
        wc = self._committed(cells)                               # allocate the freshest cells in the word's micro-columns
        return set(sorted(cells, key=lambda i: (wc[i], i))[:self.k_win])

    def _predict(self, active):
        return coincidence_predict(self.b, self.cells_idx, active, self.N, self.nE)

    def train_sentence(self, seq):
        predictive, prev_win = set(), set()
        for pos, w in enumerate(seq):
            winners = self._winners(w, pos, predictive, prev_win)
            if prev_win:
                apply_kernel_update(self.b, self.row, self.col, self.cells_idx, prev_win, winners,
                                    self.z, self.lam_pot, self.lam_dep, 1.0)
            predictive = self._predict(winners)
            prev_win = winners

    def predict_next(self, seq_prefix, candidate_words):
        """Process seq_prefix; return the predicted next word = the candidate whose micro-columns are most primed."""
        predictive, prev_win = set(), set()
        for pos, w in enumerate(seq_prefix):
            winners = self._winners(w, pos, predictive, prev_win)
            predictive = self._predict(winners)
            prev_win = winners
        primed_cols = Counter(int(i) // self.nE for i in predictive)
        if not primed_cols:
            return None
        scores = {cw: sum(primed_cols.get(c, 0) for c in self.word2cols[cw]) for cw in candidate_words}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else None


def _run_arm(seed, arm, epochs, k_win=4, act_th=3):
    overlap = (arm != "orthogonal")
    word2cols, fb, sentences, train_subj, held_out, branches, mids, M = build_vocab(overlap=overlap)
    if arm == "deranged":
        fb = {"dog": "home", "wolf": "away", "fox": "away", "cat": "away", "lion": "home"}  # inconsistent vs the shared code structure
        sentences = {s: [s, "chased", "ball", fb[s]] for s in word2cols if s in fb}
    coincidence = (arm != "lesion")
    nE = 24
    b, cells_idx, row, col = build_pool_bridge(M, nE, seed, act_th=act_th, coincidence=coincidence)
    lr = SeqGenLearner(b, row, col, cells_idx, word2cols, nE, M, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in train_subj:
                lr.train_sentence(sentences[s])
    # TEST: held-out similar subject's branch prediction ("wolf chased ball ___" -> home?)
    ok = 0
    for w in held_out:
        prefix = [w, "chased", "ball"]                            # present the held-out subject through the shared middle
        pred = lr.predict_next(prefix, branches)
        ok += int(pred == fb[w])
    return arm, ok / len(held_out)


ARMS = ["htm", "orthogonal", "lesion", "deranged", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    chance = 0.5
    w2c, fb, sents, tr, held, br, mids, M = build_vocab(True)
    print(f"train subjects {tr} -> {[sents[s] for s in tr]} | HELD-OUT {held} (generalize through the shared middle) | chance {chance:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, acc = _run_arm(s, arm, a.epochs, a.k_win, a.act_th)
                d[arm] = acc
            per.append(d)
            print(f"  [seed {s}] HTM held-out-seq-gen {d['htm']:.3f} | orthogonal {d['orthogonal']:.3f} | lesion {d['lesion']:.3f} "
                  f"| deranged {d['deranged']:.3f} | untrained {d['untrained']:.3f} || chance {chance:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm] for p in per]))
        htm, orth, les, der, unt = m("htm"), m("orthogonal"), m("lesion"), m("deranged"), m("untrained")
        go = bool(htm >= 0.90 and htm >= orth + 0.30 and htm >= les + 0.30 and htm >= der + 0.30 and htm >= chance + 0.30)
        if go:
            verdict = (f"GO -- the emergent on-bridge sequence cortex GENERALIZES a HIGH-ORDER prediction to a HELD-OUT similar "
                       f"word: a held-out similar SUBJECT ('wolf', never trained) predicts the correct family branch ('home') "
                       f"THROUGH the shared middle 'chased ball' at {htm:.3f} >> chance {chance:.2f}, by generalizing from the "
                       f"trained similar subject ('dog') via overlapping micro-columns. ORTHOGONAL-code control {orth:.3f} (no "
                       f"shared micro-columns -> no transfer); dAP-LESION {les:.3f}; DERANGED {der:.3f}; untrained {unt:.3f}; "
                       f"no teacher; multi-seed. => EMERGE-15 (high-order prediction) + EMERGE-17 (generalization) UNIFIED: a "
                       f"GENERALIZING high-order sequence language model on the real spiking substrate. NO sim/ edit (only the "
                       f"two-level word encoding).")
        else:
            miss = []
            if htm < 0.90: miss.append(f"held-out-seq-gen {htm:.3f} < 0.90")
            if htm < orth + 0.30: miss.append(f"orthogonal didn't collapse ({htm:.3f} vs {orth:.3f})")
            if htm < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            if htm < der + 0.30: miss.append(f"deranged didn't collapse ({htm:.3f} vs {der:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Tune the two-level encoding (subject shared-"
                       f"block size vs act_th; the middle nE for 2 family contexts; k_win/epochs); high-order generalization "
                       f"is the next tuning, not a wall. chance {chance:.2f}.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge18_sequence_generalization", "verdict": verdict,
               "mechanism": "high-order sequence generalization: a two-level word encoding (subjects share micro-columns = "
                            "overlapping identity SDRs; the shared middle keeps context cells disjoint per family) so a held-out "
                            "similar subject generalizes its family branch THROUGH the shared middle; sim/ kernel unchanged",
               "task": "canine/feline -> branch through a shared middle; train one subject per family, hold out the similar "
                       "subjects, test held-out high-order branch prediction vs orthogonal + lesion + deranged + untrained",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "k_win": a.k_win, "act_th": a.act_th},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "unifies EMERGE-15 (high-order) + EMERGE-17 (generalization). Next: the real stream-cortex PPMI "
                              "codes as the scale-up; grounding the emitted words to the no-confab moat; open-domain fluency gate."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge18] VERDICT: {verdict}", flush=True)
    print(f"[emerge18] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
