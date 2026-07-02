"""EMERGE-17 / toward-language — GENERALIZING word codes: the emergent on-bridge learning GENERALIZES across SIMILAR
words. When words are encoded as OVERLAPPING sparse codes (similar words share micro-columns), an association learned for
one word TRANSFERS to a HELD-OUT similar word it was never trained on — the hallmark of a real (non-memorizing) language
representation. This surpasses the EMERGE-14/15/16 orthogonal-column encoding (which can only memorize).

MECHANISM (the generalization research gate `2026-07-02-sequence-cortex-generalizing-word-codes-research-gate.md`): the
`sim/` coincidence kernel + three-term update are UNCHANGED; the ONLY change is the word->active-set ENCODING. Each word
= a fixed sparse code over a shared MICRO-COLUMN pool (its identity SDR = one cell per micro-column). Similar words SHARE
micro-columns (their SDRs OVERLAP). Learning "dog -> home" potentiates dog's SDR cells onto home's cells (distal
coincidence synapses); presenting a HELD-OUT similar word "wolf" (whose SDR shares the family micro-columns with dog)
drives home's coincidence from the SHARED cells -> if >= act_th shared cells fire, home is PREDICTED -> the association
GENERALIZES to wolf without ever training on wolf. Canonical HTM (Numenta semantic folding; Ahmad-Hawkins SDR overlap ~
similarity): overlap in the code IS semantic similarity.

CHEAP-FIRST (this de-risk): CONTROLLED SYNTHETIC similarity (defined families) isolates the generalization-from-overlap
claim; a bigram (word->word) association isolates it from the high-order sequence machinery (which EMERGE-15 already
validated). The real stream-cortex PPMI codes (similarity-structured, verified) are the SCALE-UP after this GOes.

TASK: families canines {dog,wolf,fox} -> "home", felines {cat,lion} -> "away". TRAIN on ONE per family (dog->home,
cat->away); HELD OUT the rest (wolf,fox,lion). TEST: does the held-out similar word predict its FAMILY's branch (wolf
-> home, generalizing from dog)? ANTI-CHEATS: held-out-generalization >> chance; the ORTHOGONAL-code control (families
do NOT share micro-columns) -> held-out collapses (isolates OVERLAP as the cause); dAP-LESION (coincidence off) ->
collapses; DERANGED family->branch (inconsistent) -> chance; no-teacher; multi-seed. Reuse-by-import (`_emerge14`
`build_pool_bridge`/`apply_kernel_update`/`coincidence_predict`); NO `sim/` edit. CPU numpy-backend.
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

OUT = Path("research/findings/raw/_emerge17_generalizing_word_codes.json")


def build_vocab(overlap=True):
    """Micro-column layout. overlap=True: family members SHARE a family micro-column block (similar codes). overlap=False
    (the orthogonal control): every word gets DISJOINT micro-columns (no similarity). Returns word2cols + the branch words
    + train/held-out split + family->branch map."""
    # canine family shared block [0,1,2]; feline shared [6,7,8]; the rest are word-unique / branch micro-columns.
    if overlap:
        word2cols = {
            "dog": [0, 1, 2, 3], "wolf": [0, 1, 2, 4], "fox": [0, 1, 2, 5],     # canines SHARE [0,1,2]
            "cat": [6, 7, 8, 9], "lion": [6, 7, 8, 10],                          # felines SHARE [6,7,8]
            "home": [11, 12, 13, 14], "away": [15, 16, 17, 18],                  # branch words (targets)
        }
    else:
        word2cols = {                                                           # ORTHOGONAL control: no shared blocks
            "dog": [0, 1, 2, 3], "wolf": [19, 20, 21, 22], "fox": [23, 24, 25, 26],
            "cat": [6, 7, 8, 9], "lion": [27, 28, 29, 30],
            "home": [11, 12, 13, 14], "away": [15, 16, 17, 18],
        }
    family_branch = {"dog": "home", "wolf": "home", "fox": "home", "cat": "away", "lion": "away"}
    train = [("dog", "home"), ("cat", "away")]                                   # ONE trained example per family
    held_out = ["wolf", "fox", "lion"]                                           # never trained; must generalize
    branches = ["home", "away"]
    M = 1 + max(c for cols in word2cols.values() for c in cols)
    return word2cols, family_branch, train, held_out, branches, M


def word_sdr(word2cols, w, cells_idx, nE):
    """The word's fixed identity SDR = cell 0 of each of its micro-columns (EMERGE cell indices). Similar words (shared
    micro-columns) therefore share those cells -> overlapping SDRs."""
    return set(int(c) * nE + 0 for c in word2cols[w])          # EMERGE cell index = micro-col*nE + cell0


def _run_arm(seed, arm, epochs, k_win=4, act_th=3, lam_pot=0.14, lam_dep=0.02):
    overlap = (arm not in ("orthogonal",))
    deranged = (arm == "deranged")
    word2cols, family_branch, train, held_out, branches, M = build_vocab(overlap=overlap)
    if deranged:                                               # break the family->branch consistency -> unlearnable structure
        train = [("dog", "away"), ("cat", "home")]             # (still one per family, but the mapping is deranged vs the held-out)
    coincidence = (arm != "lesion")
    nE = 8
    b, cells_idx, row, col = build_pool_bridge(M, nE, seed, act_th=act_th, coincidence=coincidence)
    z = np.zeros(M * nE)
    if arm != "untrained":
        for _ in range(epochs):
            for a, tgt in train:
                apply_kernel_update(b, row, col, cells_idx, word_sdr(word2cols, a, cells_idx, nE),
                                    word_sdr(word2cols, tgt, cells_idx, nE), z, lam_pot, lam_dep, 1.0)
    # TEST generalization on the HELD-OUT similar words: present the held-out word, predict the branch.
    branch_cols = {br: set(word2cols[br]) for br in branches}
    ok = 0
    for w in held_out:
        primed = coincidence_predict(b, cells_idx, word_sdr(word2cols, w, cells_idx, nE), M * nE, nE)
        primed_cols = Counter(int(i) // nE for i in primed)
        if not primed_cols:
            continue                                           # nothing primed -> no prediction (lesion collapses here)
        # predicted branch = the branch word whose micro-columns are most primed
        scores = {br: sum(primed_cols.get(c, 0) for c in cols) for br, cols in branch_cols.items()}
        pred_br = max(scores, key=scores.get) if max(scores.values()) > 0 else None
        ok += int(pred_br == family_branch[w])
    return arm, ok / len(held_out)


ARMS = ["htm", "orthogonal", "lesion", "deranged", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    chance = 0.5
    w2c, fb, train, held, branches, M = build_vocab(True)
    print(f"vocab {list(w2c)} | train {train} | held-out {held} | families -> {set(fb.values())} | chance {chance:.2f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, acc = _run_arm(s, arm, a.epochs, a.k_win, a.act_th)
                d[arm] = acc
            per.append(d)
            print(f"  [seed {s}] HTM held-out-gen {d['htm']:.3f} | orthogonal {d['orthogonal']:.3f} | lesion {d['lesion']:.3f} "
                  f"| deranged {d['deranged']:.3f} | untrained {d['untrained']:.3f} || chance {chance:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm] for p in per]))
        htm, orth, les, der, unt = m("htm"), m("orthogonal"), m("lesion"), m("deranged"), m("untrained")
        go = bool(htm >= 0.90 and htm >= orth + 0.30 and htm >= les + 0.30 and htm >= der + 0.30 and htm >= chance + 0.30)
        if go:
            verdict = (f"GO -- the emergent on-bridge learning GENERALIZES across SIMILAR words: an association learned for a "
                       f"TRAINED word (dog->home, cat->away) TRANSFERS to HELD-OUT similar words never trained on "
                       f"(wolf/fox->home, lion->away) at {htm:.3f} >> chance {chance:.2f}, because similar words share "
                       f"micro-columns (overlapping SDRs) so the held-out word drives the learned coincidence pathway from the "
                       f"SHARED cells. The ORTHOGONAL-code control collapses to {orth:.3f} (no shared micro-columns -> no "
                       f"transfer: OVERLAP is the cause); dAP-LESION {les:.3f}; DERANGED family->branch {der:.3f}; untrained "
                       f"{unt:.3f}; no teacher; multi-seed. => the sequence cortex now has GENERALIZING word representations "
                       f"(not just memorized codes) -- similar words transfer learning, the hallmark of a real lexical cortex. "
                       f"NO sim/ edit; the ONLY change is the word->code encoding (overlapping SDRs).")
        else:
            miss = []
            if htm < 0.90: miss.append(f"held-out-gen {htm:.3f} < 0.90")
            if htm < orth + 0.30: miss.append(f"orthogonal control didn't collapse ({htm:.3f} vs {orth:.3f} -- overlap not the cause)")
            if htm < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            if htm < der + 0.30: miss.append(f"deranged didn't collapse ({htm:.3f} vs {der:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Tune the encoding (shared-block size vs "
                       f"act_th so a held-out word's SHARED cells clear the coincidence threshold; k_win/epochs); the "
                       f"generalizing-word-code encoding is the next tuning, not a wall. chance {chance:.2f}.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge17_generalizing_word_codes", "verdict": verdict,
               "mechanism": "generalizing word representations: words = fixed sparse codes over a shared micro-column pool "
                            "(similar words share micro-columns -> overlapping SDRs); the emergent on-bridge coincidence "
                            "learning transfers a learned association to a HELD-OUT similar word via the SHARED cells; the "
                            "sim/ kernel is unchanged, the only change is the word->code encoding",
               "task": "families -> branch; train one per family, hold out the similar words, test held-out generalization vs "
                       "orthogonal-code + dAP-lesion + deranged + untrained controls",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "k_win": a.k_win, "act_th": a.act_th},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "cheap-first: controlled synthetic similarity + bigram association isolates generalization-from-"
                              "overlap. Next: the real stream-cortex PPMI codes (verified similarity-structured) as the scale-up; "
                              "then high-order sequence generalization (EMERGE-15 corpus with similar words); then grounding."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge17] VERDICT: {verdict}", flush=True)
    print(f"[emerge17] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
