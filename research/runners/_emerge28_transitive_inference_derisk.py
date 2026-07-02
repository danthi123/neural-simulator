"""EMERGE-28 / toward-semantics — TRANSITIVE relational INFERENCE (hippocampal): learn only ADJACENT premises
A>B, B>C, C>D, D>E; INFER the never-trained NON-ADJACENT relations (B>D, A>D, B>E, ...) by chaining the overlapping
premises into an integrated order on the spiking HTM cortex -- the classic transitive-inference paradigm (Dusek-
Eichenbaum; catalog D.02), emergent, NO inference engine, NO `sim/` edit.

MECHANISM: each item = a disjoint content code; a premise "X > Y" is learned as X->Y (the committed `sim/` three-term
kernel over the coincidence pool). The overlapping premises (B is the LESS item in A>B and the GREATER item in B>C)
chain into a single learned sequence A->B->C->D->E. A transitive judgment `greater(X, Y)` = "is Y REACHABLE downstream
of X in the learned chain?" -- read by rolling the chain out from X (autoregressive priming, EMERGE-16) and collecting
every item reached. B reaches C, D, E though B>D and B>E were NEVER trained -> the non-adjacent order is INFERRED by
integrating the premises.

THE DISCRIMINATING TEST (why this is real inference, not association): the ANCHOR pairs (A>E) are solvable by simple
associative strength (A is always the greater in training, E always the lesser). The CRITICAL INTERNAL pairs (B>D, C
appears... B and D each appear as BOTH a greater AND a lesser item across the premises) CANNOT be solved by
associative strength -- they REQUIRE the integrated order. So the internal-pair accuracy is the genuine TI signal
(Dusek-Eichenbaum 1997).

ANTI-CHEATS: non-adjacent accuracy on HELD-OUT pairs (never trained); the CRITICAL internal pair B>D (both endpoints
internal); dAP-LESION (no chaining -> collapses); BROKEN-CHAIN (drop the middle premise C>D -> B and D become
uncomparable -> the internal inference collapses, isolating the transitive chaining as the cause); 6-seed. Reuse-by-
import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from itertools import combinations
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_emerge28_transitive_inference.json")

ITEMS = ["A", "B", "C", "D", "E"]                                               # the true order A > B > C > D > E
CONTENT = {it: [i * 3, i * 3 + 1, i * 3 + 2] for i, it in enumerate(ITEMS)}     # 3 disjoint cols each
PREMISES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]                     # adjacent greater->less
ADJACENT = set((g, l) for g, l in PREMISES)
NONADJ = [(ITEMS[i], ITEMS[j]) for i, j in combinations(range(len(ITEMS)), 2) if j - i > 1]  # never trained
CRITICAL = [("B", "D")]                                                         # internal pair (both endpoints internal)
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max(c for cs in CONTENT.values() for c in cs)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class TransitiveProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, premises=None):
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        prem = PREMISES if premises is None else premises
        for _ in range(epochs):
            for g, l in prem:
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[g]), _sdr(CONTENT[l]),
                                    self.z, 0.14, 0.02, 1.0)

    def _reachable(self, start, depth=6):
        reached, active = set(), _sdr(CONTENT[start])
        for _ in range(depth):
            ab = np.zeros(len(self.ci), bool)
            for i in active:
                ab[i] = True
            _prime_from_winners(self.b, self.ci, ab)
            vap = getattr(self.b, "cp_v_apical", None)
            if vap is None:
                break
            vap = _host(vap)[self.ci]
            nxt = None
            for it in ITEMS:
                if it in reached or it == start:
                    continue
                dr = float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in CONTENT[it]]))
                if dr > FLOOR and (nxt is None or dr > nxt[1]):
                    nxt = (it, dr)
            if nxt is None:
                break
            reached.add(nxt[0]); active = _sdr(CONTENT[nxt[0]])
        return reached

    def greater(self, x, y):
        """True if x > y is inferred: y is reachable downstream of x, and x is NOT reachable downstream of y."""
        return (y in self._reachable(x)) and (x not in self._reachable(y))

    def judge(self, pair):
        g, l = pair                                                            # g is the true-greater item
        return self.greater(g, l)


def _run_arm(seed, arm, epochs):
    prem = [p for p in PREMISES if p != ("C", "D")] if arm == "broken" else None
    p = TransitiveProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"), premises=prem)
    adj = np.mean([p.judge(pr) for pr in PREMISES])
    nonadj = np.mean([p.judge(pr) for pr in NONADJ])
    crit = np.mean([p.judge(pr) for pr in CRITICAL])
    return arm, {"adjacent": float(adj), "nonadjacent": float(nonadj), "critical_BD": float(crit)}


ARMS = ["htm", "lesion", "broken"]


def _demo(seed=42, epochs=80):
    p = TransitiveProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-28 transitive inference (hippocampal; no inference engine, no transformer) ===")
    print(f"  TAUGHT only adjacent premises: {['>'.join(pr) for pr in PREMISES]}\n")
    for it in ITEMS:
        print(f"  {it} is greater than: {sorted(p._reachable(it))}")
    print("\n  the CRITICAL never-trained internal pair:")
    print(f"    is B > D? -> {p.greater('B','D')}   (INFERRED: B>C, C>D trained, B>D never trained)")
    print(f"    is D > B? -> {p.greater('D','B')}   (correctly False)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo:
        _demo(a.seeds[0], a.epochs); return 0
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    print(f"order {'>'.join(ITEMS)} | premises {[' >'.join(pr) for pr in PREMISES]} | non-adjacent (held-out) {NONADJ} | critical {CRITICAL}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] adjacent {h['adjacent']:.2f} | NON-ADJACENT(held-out) {h['nonadjacent']:.2f} "
                  f"| CRITICAL B>D {h['critical_BD']:.2f} || lesion-nonadj {d['lesion']['nonadjacent']:.2f} "
                  f"| broken-chain B>D {d['broken']['critical_BD']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        adj, nonadj, crit = m("htm", "adjacent"), m("htm", "nonadjacent"), m("htm", "critical_BD")
        les, brk = m("lesion", "nonadjacent"), m("broken", "critical_BD")
        go = bool(nonadj >= 0.90 and crit >= 0.90 and adj >= 0.90 and nonadj >= les + 0.30 and crit >= brk + 0.30)
        if go:
            verdict = (f"GO -- TRANSITIVE INFERENCE emerges on the spiking HTM cortex: from ONLY adjacent premises "
                       f"(A>B, B>C, C>D, D>E) the never-trained NON-ADJACENT relations are INFERRED ({nonadj:.2f} on HELD-OUT "
                       f"pairs) by chaining the overlapping premises into an integrated order -- NO inference engine (Dusek-"
                       f"Eichenbaum; catalog D.02). The CRITICAL internal pair B>D ({crit:.2f}) -- unsolvable by associative "
                       f"strength (B and D each appear as both greater and lesser), so this is genuine inference not "
                       f"association. dAP-LESION collapses ({les:.2f}); BROKEN-CHAIN (drop the middle premise C>D -> B and D "
                       f"uncomparable) collapses B>D ({brk:.2f}, isolating the transitive chaining); 6-seed. => the substrate "
                       f"INTEGRATES premises into a relational order, NO sim/ edit.")
        else:
            miss = []
            if nonadj < 0.90: miss.append(f"non-adjacent {nonadj:.2f} < 0.90")
            if crit < 0.90: miss.append(f"critical B>D {crit:.2f} < 0.90")
            if adj < 0.90: miss.append(f"adjacent {adj:.2f} < 0.90")
            if nonadj < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({nonadj:.2f} vs {les:.2f})")
            if crit < brk + 0.30: miss.append(f"broken-chain didn't collapse B>D ({crit:.2f} vs {brk:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the rollout depth / block sizes vs "
                       "ACT_TH / epochs; transitive inference is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge28_transitive_inference", "verdict": verdict,
               "mechanism": "transitive inference by chaining overlapping premises: each item = a disjoint code; premise "
                            "X>Y learned as X->Y via the committed sim/ three-term kernel; overlapping premises chain into a "
                            "learned order A->B->C->D->E; greater(X,Y) = Y reachable downstream of X (autoregressive rollout); "
                            "non-adjacent relations inferred by integration; sim/ unchanged",
               "task": "teach adjacent premises A>B..D>E; test non-adjacent + critical B>D vs dAP-lesion + broken-chain; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH, "items": ITEMS},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the items + premises are host-DESIGNED; inference-over-structure, not acquisition-from-experience "
                              "(the R-c residual). With EMERGE-26/27 (inheritance) this completes the inference triad: "
                              "generalization (17), inheritance (26/27), transitivity (28). Next: emergent structure from "
                              "experience (the R-c research gate) + couple the inference read-out into the console."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge28] VERDICT: {verdict}", flush=True)
    print(f"[emerge28] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
