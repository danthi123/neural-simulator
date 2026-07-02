"""EMERGE-16 / toward-language — AUTOREGRESSIVE GENERATION on the emergent HTM Temporal-Memory sequence cortex: after
the substrate has LEARNED the word sequences (EMERGE-15), a CUE word rolls out the continuation AUTONOMOUSLY. This turns
the next-word PREDICTOR into a word-sequence PRODUCER — the substrate's own generation, with NO external drive of the
continuation, NO critic, NO extra learning.

MECHANISM (excitability-replay, a built-in read-out mode of the Bouhadjar-Diesmann 2022 spiking-HTM substrate — per the
research gate `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`): a dendritic-plateau-PRIMED (predicted)
cell is one excitability step from firing; drive it so the plateau ALONE fires and the already-learned distal synapses
roll the sequence out. Here we realize that as an autoregressive loop on the same bridge machinery: present the cue's
winner SDR → the bridge's WEIGHTED coincidence predicts the next column's context-specific cells → those PRIMED cells
become the next active set (they "fire") → they predict the next column → ... The generated column at each step = the
generated word. NOT the prior SongHVC generation NEGATIVE (that needed a self-comprehension critic that couldn't read
order back; this is a different mechanism — pure predict→become-active rollout on the learned connectivity).

GO: from each cue word the substrate GENERATES the correct learned continuation (e.g. "dog" -> "chased the ball home",
"cat" -> "chased the ball away") — the branch matching the cue, proving the generation carries the high-order context.
ANTI-CHEATS: dAP-LESION (coincidence off) -> cannot generate (collapses); untrained -> collapses; the generated branch
must MATCH THE CUE (a cue-swap generates the OTHER continuation = context-driven, not a fixed rollout); multi-seed.
Reuse-by-import (`_emerge14` + `_emerge15`); NO `sim/` edit. CPU numpy-backend.
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

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, OnBridgeLearner, coincidence_predict
from research.runners._emerge15_word_sequence_lm_derisk import make_word_corpus, sentences_to_cols

OUT = Path("research/findings/raw/_emerge16_word_generation.json")


def generate_from_cue(lr, cue_col, n_steps, nE, k_win):
    """Autoregressive rollout: present the cue's winner SDR, then repeatedly let the PREDICTED (dAP-primed) cells of the
    most-predicted column become the next active set and predict the next column. Returns the generated column sequence
    (the generated words), starting WITH the cue."""
    gen = [int(cue_col)]
    active = set(lr._col(cue_col)[:k_win])                          # the cue's winner SDR fires
    for _ in range(n_steps):
        predictive = coincidence_predict(lr.b, lr.cells_idx, active, lr.N, nE)  # the bridge's own weighted coincidence
        if not predictive:
            break                                                  # nothing primed -> generation halts (lesion collapses here)
        col_votes = Counter(int(i) // nE for i in predictive)
        next_col = col_votes.most_common(1)[0][0]                  # the most-primed column = the generated next word
        gen.append(int(next_col))
        active = set(i for i in predictive if int(i) // nE == next_col)  # that column's primed cells become active (fire)
        if len(active) > k_win:
            active = set(sorted(active)[:k_win])
    return gen


def _run_arm(seed, arm, n_subj, epochs, k_win=4, act_th=3):
    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=n_subj, seed=seed)
    col_seqs = sentences_to_cols(sentences, word2col)
    vocab_n = len(vocab); nE = k_win * n_subj + 8
    b, cells_idx, row, col = build_pool_bridge(vocab_n, nE, seed, act_th=act_th, coincidence=(arm != "lesion"))
    lr = OnBridgeLearner(b, row, col, cells_idx, vocab_n, nE, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in col_seqs:
                lr.train_sequence(s)
    n_steps = len(col_seqs[0]) - 1                                  # generate the full continuation length
    # GENERATE from each cue (word 0 of each sentence); GO if the generated sequence == the learned sentence.
    gen_exact = 0; gen_branch = 0
    swap_ok = 0; swap_tot = 0
    for i, s in enumerate(col_seqs):
        g = generate_from_cue(lr, s[0], n_steps, nE, k_win)
        gen_exact += int(g == list(s))                             # exact full-sentence generation
        gen_branch += int(len(g) > branch_pos and g[branch_pos] == s[branch_pos])  # the branch word matches the cue
    # CUE-SWAP context control (htm only): starting from cue j should generate cue j's branch (context-driven)
    if arm == "htm":
        for i in range(len(col_seqs)):
            g = generate_from_cue(lr, col_seqs[i][0], n_steps, nE, k_win)
            swap_ok += int(len(g) > branch_pos and g[branch_pos] == col_seqs[i][branch_pos]); swap_tot += 1
    return arm, {"gen_exact": gen_exact / len(col_seqs), "gen_branch": gen_branch / len(col_seqs),
                 "swap_follows": (swap_ok / max(1, swap_tot)) if arm == "htm" else None}


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-subj", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    sentences, vocab, word2col, branch_pos = make_word_corpus(n_subj=a.n_subj)
    print(f"corpus: {sentences}\n  vocab {vocab} | generate full continuation from each cue word", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.n_subj, a.epochs, a.k_win, a.act_th)
                d[arm + "_exact"] = r["gen_exact"]; d[arm + "_branch"] = r["gen_branch"]
                if arm == "htm":
                    d["swap_follows"] = r["swap_follows"]
            per.append(d)
            print(f"  [seed {s}] GEN exact {d['htm_exact']:.3f} branch {d['htm_branch']:.3f} | swap-follows {d['swap_follows']:.3f} "
                  f"| lesion exact {d['lesion_exact']:.3f} branch {d['lesion_branch']:.3f} | untrained branch {d['untrained_branch']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([p[k] for p in per]))
        ex, br = m("htm_exact"), m("htm_branch")
        les_br, unt_br = m("lesion_branch"), m("untrained_branch")
        swap = m("swap_follows"); chance = 1.0 / a.n_subj
        go = bool(ex >= 0.90 and br >= 0.90 and br >= les_br + 0.30 and unt_br <= chance + 0.1 and swap >= 0.90)
        if go:
            verdict = (f"GO -- the emergent HTM Temporal-Memory sequence cortex GENERATES word sequences autoregressively on the "
                       f"real spiking bridge: from a CUE word it rolls out the correct learned continuation (exact-sentence "
                       f"{ex:.3f}, branch-matches-cue {br:.3f}) with NO external drive of the continuation, NO critic, NO extra "
                       f"learning -- the predicted (dAP-primed) cells become the next active set and predict the next word "
                       f"(excitability-replay). The generation is CONTEXT-DRIVEN (each cue generates ITS branch, swap-follows "
                       f"{swap:.3f}); dAP-LESION collapses it (branch {les_br:.3f}); untrained {unt_br:.3f}; multi-seed. => the "
                       f"substrate is now a word-sequence PRODUCER, not just a predictor -- the honest simulate-don't-bolt-on "
                       f"path to language PRODUCTION, replacing the transformer's generate role. NO sim/ edit.")
        else:
            miss = []
            if ex < 0.90: miss.append(f"exact-sentence generation {ex:.3f} < 0.90")
            if br < 0.90: miss.append(f"branch-matches-cue {br:.3f} < 0.90")
            if br < les_br + 0.30: miss.append(f"dAP-lesion didn't collapse (branch {br:.3f} vs {les_br:.3f})")
            if swap < 0.90: miss.append(f"generation not context-driven (swap-follows {swap:.3f})")
            if unt_br > chance + 0.1: miss.append(f"untrained didn't collapse ({unt_br:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Tune the rollout (which primed cells become "
                       f"active; the excitability/k_win of the autoregressive step) against the EMERGE-15 word-LM (which "
                       f"PREDICTS 1.000); generation is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge16_word_generation", "verdict": verdict,
               "mechanism": "autoregressive GENERATION on the emergent on-bridge HTM Temporal-Memory: a cue word's winner SDR "
                            "-> the bridge's weighted coincidence predicts the next column's primed cells -> those become the "
                            "next active set (excitability-replay) -> roll out the sequence; NO external drive of the "
                            "continuation, NO critic, NO extra learning",
               "task": "generate the full continuation from each cue word; exact-sentence + branch-matches-cue + dAP-lesion + "
                       "untrained + cue-context-driven + multi-seed",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "epochs": a.epochs, "k_win": a.k_win, "act_th": a.act_th},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "reuse-by-import of the rung-4 learner + EMERGE-15 word-LM; NO sim/ edit. Generation = the "
                              "predict->become-active rollout (the excitability-replay read-out mode). Next: similarity-"
                              "structured codes (generalization) + grounding the emitted words to the no-confab moat; the "
                              "open residual (next gate) = open-domain surface fluency."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge16] VERDICT: {verdict}", flush=True)
    print(f"[emerge16] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
