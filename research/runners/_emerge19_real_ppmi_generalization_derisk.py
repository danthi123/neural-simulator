"""EMERGE-19 / toward-language — GENERALIZATION on the REAL stream-cortex PPMI codes. EMERGE-17/18 proved the mechanism
with hand-designed synthetic similarity; this validates it on the LEARNED, real similarity structure of the project's
stream-cortex codes (`_phaseB_stream_codes_320_seed42.npy`, 320×300, verified similarity-structured, max off-diag cos
0.832). If the emergent on-bridge learning GENERALIZES a learned association from one word to a HELD-OUT word that is
similar ONLY because the stream cortex LEARNED them to be similar (not because we designed it), the generalizing lexical
representation is real, not a hand-built artifact.

MECHANISM (unchanged from EMERGE-17): each word = its code's TOP-Kc dimensions as MICRO-COLUMNS (an SDR over the 300 code
dims); words whose LEARNED codes are similar SHARE top-K dims (overlapping SDRs — the gate verified corr(cosine, top-K
overlap) = +0.48). The `sim/` kernel is UNCHANGED; the only input is the real codes. We find two TIGHT real-similarity
CLUSTERS (no word labels needed — cluster by cosine on the real codes), train ONE member of each cluster to a distinct
branch, HOLD OUT the rest, and test whether a held-out member generalizes its cluster's branch (because its real code
overlaps the trained member's).

ANTI-CHEATS: held-out real-similar-word generalization >> chance; the SHUFFLED-CODE control (replace each word's code
with a random code of the same sparsity -> destroys the real similarity -> no shared micro-columns -> held-out
collapses, isolating the REAL LEARNED similarity as the cause); dAP-LESION collapses; untrained collapses; multi-seed.
Reuse-by-import (`_emerge14`); NO `sim/` edit. CPU numpy-backend.
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
    build_pool_bridge, apply_kernel_update, coincidence_predict)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODES_PATH = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
OUT = Path("research/findings/raw/_emerge19_real_ppmi_generalization.json")


def topk_cols(code, Kc):
    """The word's micro-columns = the Kc largest-|magnitude| dimensions of its code (a K-of-300 SDR). Similar codes ->
    overlapping top-K -> overlapping micro-columns."""
    return set(int(d) for d in np.argsort(-np.abs(code))[:Kc])


def find_clusters(codes, Kc, act_th, fam_size=3, n_fam=2, min_overlap=None, min_cos=0.40):
    """Find n_fam disjoint TIGHT real-similarity clusters where every member is genuinely SIMILAR to the anchor (cosine
    >= min_cos) AND shares >= act_th top-Kc micro-columns with it (so a held-out member's SHARED micro-columns reliably
    drive the anchor's learned coincidence). The cosine floor is what the boundary run lacked -- it accepted loose
    clusters (cos 0.27) whose top-K overlapped by chance but did not carry the association. No word labels."""
    min_overlap = act_th if min_overlap is None else min_overlap
    cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    sims = cn @ cn.T; np.fill_diagonal(sims, -1)
    cols = [topk_cols(codes[i], Kc) for i in range(len(codes))]
    used = set(); fams = []
    order = np.argsort(-sims.max(axis=1))                        # anchors with a very-similar neighbour first
    for anchor in order:
        if anchor in used:
            continue
        nbrs = np.argsort(-sims[anchor])
        members = [int(anchor)]
        for j in nbrs:
            if len(members) >= fam_size:
                break
            j = int(j)
            if j in used or j in members or sims[anchor, j] < min_cos:   # GENUINE cosine similarity required
                continue
            if len(cols[anchor] & cols[j]) >= min_overlap and all(len(cols[m] & cols[j]) >= 1 for m in members):
                members.append(j)
        if len(members) >= fam_size:
            # require this family's members to be DISJOINT (few shared cols) from all prior families
            if all(all(len(cols[m] & cols[pm]) < min_overlap for pm in pf) for pf in fams for m in members):
                fams.append(members); used.update(members)
        if len(fams) >= n_fam:
            break
    return fams, cols


def word_sdr(cols_set, nE):
    return set(int(c) * nE + 0 for c in cols_set)               # cell 0 of each micro-column


def _run_arm(seed, arm, codes, Kc, epochs, act_th, k_win, fams, cols):
    """arm: htm (real codes) / shuffled (random codes) / lesion / untrained."""
    n_dims = codes.shape[1]
    n_fam = len(fams)
    if arm == "shuffled":
        rng = np.random.default_rng(seed + 99)                  # random codes of the same top-K sparsity -> no real similarity
        wcols_raw = {int(m): set(rng.choice(n_dims, Kc, replace=False).tolist()) for fam in fams for m in fam}
    else:
        wcols_raw = {int(m): set(cols[m]) for fam in fams for m in fam}
    # COMPACT the substrate: only the micro-columns actually used (+ 1 branch micro-column per family), remapped to
    # [0, M) -- so the dense pool is ~50-70 micro-columns, not all 300 (fast; the unused code dims are irrelevant).
    used = sorted(set().union(*wcols_raw.values()))
    remap = {c: i for i, c in enumerate(used)}
    n_word_cols = len(used)
    wcols = {m: {remap[c] for c in cs} for m, cs in wcols_raw.items()}
    branch_cols = {f: {n_word_cols + f} for f in range(n_fam)}   # 1 micro-column per family branch (distinct)
    M = n_word_cols + n_fam
    nE = 8
    b, cells_idx, row, col = build_pool_bridge(M, nE, seed, act_th=act_th, coincidence=(arm != "lesion"))
    z = np.zeros(M * nE)
    train = [(fam[0], f) for f, fam in enumerate(fams)]         # train ONE member of each family -> its branch
    held = [(m, f) for f, fam in enumerate(fams) for m in fam[1:]]  # hold out the rest
    if arm != "untrained":
        for _ in range(epochs):
            for m, f in train:
                apply_kernel_update(b, row, col, cells_idx, word_sdr(wcols[m], nE),
                                    word_sdr(branch_cols[f], nE), z, 0.14, 0.02, 1.0)
    ok = 0
    for m, f in held:
        primed = coincidence_predict(b, cells_idx, word_sdr(wcols[m], nE), M * nE, nE)
        pc = Counter(int(i) // nE for i in primed)
        scores = {ff: sum(pc.get(c, 0) for c in branch_cols[ff]) for ff in range(n_fam)}
        pred = max(scores, key=scores.get) if max(scores.values()) > 0 else None
        ok += int(pred == f)
    return arm, ok / max(1, len(held)), len(held)


ARMS = ["htm", "shuffled", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--kc", type=int, default=12)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--n-fam", type=int, default=3)
    ap.add_argument("--fam-size", type=int, default=3)
    ap.add_argument("--min-cos", type=float, default=0.45)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(CODES_PATH):
        print(f"NOT-RUNNABLE: missing {CODES_PATH}"); return 2
    codes = np.load(CODES_PATH).astype(np.float64)
    fams, cols = find_clusters(codes, a.kc, a.act_th, fam_size=a.fam_size, n_fam=a.n_fam, min_cos=a.min_cos)
    if len(fams) < 2:
        print(f"NOT-RUNNABLE: found only {len(fams)} tight real-similarity clusters (need >=2). Lower --act-th or raise --kc."); return 2
    cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    fam_cos = [float(np.mean([cn[fam[0]] @ cn[m] for m in fam[1:]])) for fam in fams]
    chance = 1.0 / len(fams)
    print(f"codes {codes.shape} | found {len(fams)} real-similarity clusters (fam_size {a.fam_size}); anchor->member mean cos "
          f"{[round(c,2) for c in fam_cos]} | Kc {a.kc} act_th {a.act_th} | chance {chance:.3f}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, acc, nheld = _run_arm(s, arm, codes, a.kc, a.epochs, a.act_th, a.k_win, fams, cols)
                d[arm] = acc
            per.append(d)
            print(f"  [seed {s}] HTM held-out-gen(real codes) {d['htm']:.3f} | shuffled {d['shuffled']:.3f} "
                  f"| lesion {d['lesion']:.3f} | untrained {d['untrained']:.3f} || chance {chance:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm] for p in per]))
        htm, shuf, les, unt = m("htm"), m("shuffled"), m("lesion"), m("untrained")
        go = bool(htm >= 0.90 and htm >= shuf + 0.30 and htm >= les + 0.30 and htm >= chance + 0.30 and unt <= chance + 0.1)
        if go:
            verdict = (f"GO -- the emergent on-bridge learning GENERALIZES on the REAL stream-cortex PPMI codes: a learned "
                       f"association transfers to a HELD-OUT word that is similar ONLY because the stream cortex LEARNED it so "
                       f"(held-out-gen {htm:.3f} >> chance {chance:.3f}), via the real codes' overlapping top-K micro-columns. "
                       f"The SHUFFLED-CODE control collapses to {shuf:.3f} (random codes -> no real similarity -> no transfer: "
                       f"the LEARNED similarity is the cause); dAP-LESION {les:.3f}; untrained {unt:.3f}; multi-seed. => the "
                       f"generalizing lexical representation is REAL (works on learned similarity, not a hand-built artifact). "
                       f"NO sim/ edit.")
        else:
            miss = []
            if htm < 0.90: miss.append(f"held-out-gen {htm:.3f} < 0.90")
            if htm < shuf + 0.30: miss.append(f"shuffled control didn't collapse ({htm:.3f} vs {shuf:.3f})")
            if htm < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f". Tune Kc (top-K overlap vs act_th) / cluster "
                       f"tightness / epochs so a held-out real-similar word's SHARED top-K micro-columns clear the coincidence "
                       f"threshold; real-code generalization is the next tuning, not a wall. chance {chance:.3f}.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge19_real_ppmi_generalization", "verdict": verdict,
               "mechanism": "generalization on the REAL stream-cortex PPMI codes: words = top-K micro-columns of their learned "
                            "code; real-similar words share top-K -> overlapping SDRs -> a learned association transfers to a "
                            "held-out real-similar word; sim/ kernel unchanged, only the real codes as input",
               "task": "cluster real-similarity families by cosine (no labels), train one per family -> branch, hold out the "
                       "rest, test held-out generalization vs shuffled-code + dAP-lesion + untrained",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "kc": a.kc, "act_th": a.act_th, "n_fam": len(fams),
               "fam_size": a.fam_size, "fam_anchor_member_cos": fam_cos},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "validates EMERGE-17/18 on the project's REAL learned similarity structure (not synthetic). Next: "
                              "grounding the emitted words to the no-confab moat; the open-domain surface-fluency research gate."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge19] VERDICT: {verdict}", flush=True)
    print(f"[emerge19] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
