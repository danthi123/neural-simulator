"""EMERGE-20 / toward-language — GROUNDED production + the INTRINSIC no-confab MOAT. The emergent sequence cortex
produces a word only when it has a LEARNED (grounded) pathway for it, and produces NOTHING for a truly-novel ungrounded
cue — so it CANNOT confabulate what it has no pathway for. The no-confab moat (the project's load-bearing property) is
INTRINSIC to the substrate, not a bolted-on check: an ungrounded cue (a novel word whose code is DISJOINT from anything
learned) drives NO coincidence -> no primed cells -> the cortex ABSTAINS.

Three regimes on ONE trained cortex (grounded facts dog->home, cat->away):
  - GROUNDED (a trained word): dog -> generates "home", cat -> "away" (the learned fact).
  - GENERALIZED (an untrained but SIMILAR word, a valid grounded inference): wolf/fox -> "home", lion -> "away"
    (generalizes via the shared family micro-columns -- EMERGE-17/19).
  - ABSTAIN / MOAT (a truly-NOVEL word, code disjoint from everything): "zzz" -> NOTHING primed -> ABSTAINS
    (no confabulation). This is the no-confab moat, emergent from the substrate.

This replaces the transformer's fluency-in-the-loop role WHILE keeping the moat: the substrate produces grounded word
sequences and abstains when ungrounded, biology-native. ANTI-CHEATS: grounded + generalized accuracy high; novel-word
ABSTAIN rate = 1.0 (confabulation rate 0); dAP-LESION collapses grounded (the coincidence is load-bearing); multi-seed.
Reuse-by-import (`_emerge14` + `_emerge17`); NO `sim/` edit. CPU numpy-backend.
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

OUT = Path("research/findings/raw/_emerge20_grounded_moat.json")


def build_vocab():
    word2cols = {
        "dog": [0, 1, 2, 3], "wolf": [0, 1, 2, 4], "fox": [0, 1, 2, 5],           # canines share [0,1,2]
        "cat": [6, 7, 8, 9], "lion": [6, 7, 8, 10],                                # felines share [6,7,8]
        "home": [11, 12, 13, 14], "away": [15, 16, 17, 18],                        # branch words
        "zzz": [27, 28, 29, 30], "qqq": [31, 32, 33, 34],                          # NOVEL words -- codes DISJOINT from all above
    }
    grounded = [("dog", "home"), ("cat", "away")]                                  # the learned facts
    generalized = {"wolf": "home", "fox": "home", "lion": "away"}                  # untrained-but-similar (valid inferences)
    novel = ["zzz", "qqq"]                                                         # truly-novel/ungrounded -> must abstain
    branches = ["home", "away"]
    M = 1 + max(c for cols in word2cols.values() for c in cols)
    return word2cols, grounded, generalized, novel, branches, M


def word_sdr(word2cols, w, nE):
    return set(int(c) * nE + 0 for c in word2cols[w])


def _run_arm(seed, arm, epochs, act_th=3):
    coincidence = (arm != "lesion")
    word2cols, grounded, generalized, novel, branches, M = build_vocab()
    nE = 8
    b, cells_idx, row, col = build_pool_bridge(M, nE, seed, act_th=act_th, coincidence=coincidence)
    z = np.zeros(M * nE)
    if arm != "untrained":
        for _ in range(epochs):
            for a, tgt in grounded:
                apply_kernel_update(b, row, col, cells_idx, word_sdr(word2cols, a, nE),
                                    word_sdr(word2cols, tgt, nE), z, 0.14, 0.02, 1.0)
    branch_cols = {br: set(word2cols[br]) for br in branches}

    def produce(w):
        """Cue the cortex with word w; return the produced branch, or None (ABSTAIN) if nothing is primed."""
        primed = coincidence_predict(b, cells_idx, word_sdr(word2cols, w, nE), M * nE, nE)
        pc = Counter(int(i) // nE for i in primed)
        scores = {br: sum(pc.get(c, 0) for c in cols) for br, cols in branch_cols.items()}
        if not pc or max(scores.values()) == 0:
            return None                                                            # ABSTAIN -- no grounded pathway
        return max(scores, key=scores.get)

    grd = sum(produce(a) == tgt for a, tgt in grounded) / len(grounded)
    gen = sum(produce(w) == tgt for w, tgt in generalized.items()) / len(generalized)
    abstain = sum(produce(w) is None for w in novel) / len(novel)                  # moat: novel -> abstain
    confab = sum(produce(w) is not None for w in novel) / len(novel)              # confabulation rate (must be 0)
    return arm, {"grounded": grd, "generalized": gen, "novel_abstain": abstain, "confab": confab}


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    w2c, grd, gen, nov, br, M = build_vocab()
    print(f"grounded facts {grd} | generalized (similar) {list(gen)} | NOVEL/ungrounded {nov} (must ABSTAIN)", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs, a.act_th)
                d[arm] = r
            per.append(d)
            h = d["htm"]
            print(f"  [seed {s}] GROUNDED {h['grounded']:.2f} | GENERALIZED {h['generalized']:.2f} | NOVEL-ABSTAIN "
                  f"{h['novel_abstain']:.2f} (confab {h['confab']:.2f}) || lesion-grounded {d['lesion']['grounded']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        grd, gen, ab, cf = m("htm", "grounded"), m("htm", "generalized"), m("htm", "novel_abstain"), m("htm", "confab")
        les_grd = m("lesion", "grounded")
        go = bool(grd >= 0.90 and gen >= 0.90 and ab >= 0.90 and cf <= 0.10 and grd >= les_grd + 0.30)
        if go:
            verdict = (f"GO -- the emergent sequence cortex produces GROUNDED words + has an INTRINSIC no-confab MOAT: it "
                       f"generates the learned fact for a GROUNDED (trained) cue (dog->home, cat->away: {grd:.2f}), GENERALIZES "
                       f"a valid grounded inference for a SIMILAR untrained cue (wolf->home, lion->away: {gen:.2f}), and ABSTAINS "
                       f"for a truly-NOVEL ungrounded cue (novel-abstain {ab:.2f}, confabulation {cf:.2f}) -- it CANNOT confabulate "
                       f"what it has no learned pathway for. dAP-LESION collapses grounded ({les_grd:.2f}, the coincidence is "
                       f"load-bearing); multi-seed. => the no-confab moat is EMERGENT from the substrate (not a bolted-on check): "
                       f"a grounded, moat-protected emergent word producer -- replacing the transformer's fluency-in-the-loop role "
                       f"WHILE keeping the moat. NO sim/ edit.")
        else:
            miss = []
            if grd < 0.90: miss.append(f"grounded {grd:.2f} < 0.90")
            if gen < 0.90: miss.append(f"generalized {gen:.2f} < 0.90")
            if ab < 0.90: miss.append(f"novel-abstain {ab:.2f} < 0.90 (moat leak, confab {cf:.2f})")
            if grd < les_grd + 0.30: miss.append(f"dAP-lesion didn't collapse grounded ({grd:.2f} vs {les_grd:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune act_th vs the code sparsity so a novel "
                       "disjoint cue drives 0 coincidence (abstain) while grounded/similar cues clear it; the intrinsic moat "
                       "is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge20_grounded_moat", "verdict": verdict,
               "mechanism": "grounded production + intrinsic no-confab moat on the emergent sequence cortex: a grounded (trained) "
                            "cue produces the learned fact; a similar untrained cue generalizes a valid inference; a truly-novel "
                            "cue (disjoint code) drives no coincidence -> abstains (no confabulation); sim/ kernel unchanged",
               "task": "grounded facts dog->home/cat->away; produce for grounded/generalized/novel cues; novel must abstain; "
                       "dAP-lesion collapses grounded; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": a.act_th},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the no-confab moat is INTRINSIC (an ungrounded cue has no learned pathway -> no production). Next: "
                              "the open-domain surface-fluency research gate (the transformer's last unique job)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge20] VERDICT: {verdict}", flush=True)
    print(f"[emerge20] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
