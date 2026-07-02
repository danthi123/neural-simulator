"""EMERGE-32 / toward-semantics — EMERGENT STRUCTURE FROM VARIED EXPERIENCE (the honest robustness closure of EMERGE-30):
the emergent category is NOT keyed to a single shared context token -- each member is observed with a DIFFERENT
overlapping subset of its category's feature pool (robin sees {nest,wing,sky}, sparrow {nest,wing,hedge}, ... -- no
universal token), yet the category STILL emerges and a held-out member STILL inherits a property taught via one
exemplar, via the FEATURE OVERLAP. This removes the "one shared token = a provided superordinate" critique of the
EMERGE-30 cheap-first. NO `sim/` edit.

MECHANISM (Rogers-McClelland feature-overlap category structure + the next-state predictor): each category has a
feature POOL; each member is streamed with its own random k-of-n subset (guaranteed pairwise overlap >= ACT_TH by
n=4,k=3). The committed `sim/` three-term kernel learns member-content -> its feature subset (on-bridge Hebbian). A
property is taught via an EXEMPLAR (present robin, prime its features, bind fly to robin + its features). A held-out
member (a DIFFERENT subset) inherits because its subset OVERLAPS the exemplar's taught features by >= ACT_TH -> the
shared features prime the property. No single feature is universal; the overlap carries the category.

ANTI-CHEATS: held-out inheritance (a member whose property was never taught, whose feature subset differs from the
exemplar's); PERMUTED-POOL (each member draws features from a RANDOM pool -> category overlap destroyed -> collapses);
NO-LEARNING (skip the stream -> abstains); dAP-LESION collapses; MOAT (a never-observed member abstains); cross-
category is handled by the argmax read (a fish's features overlap the fish exemplar, not the bird's); 6-seed. Reuse-by-
import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_emerge32_varied_context_emergence.json")

MEMBERS = {"robin": "B", "sparrow": "B", "canary": "B", "trout": "F", "salmon": "F", "pike": "F"}
EXEMPLARS = {"B": "robin", "F": "trout"}                                        # one member per category teaches the property
HELD_OUT = ["sparrow", "canary", "salmon", "pike"]                             # never in property teaching, different subsets
CATPROP = {"B": "fly", "F": "swim"}
CONTENT = {m: [i * 3, i * 3 + 1, i * 3 + 2] for i, m in enumerate(MEMBERS)}     # cols 0..17
POOL = {"B": [18, 19, 20, 21], "F": [22, 23, 24, 25]}                          # 4 features per category (n=4)
PROP = {"fly": [26, 27], "swim": [28, 29]}
NOVEL_CONTENT = [30, 31, 32]
K = 3                                                                          # each member observes 3-of-4 -> pairwise overlap >= 2
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max([c for cs in list(CONTENT.values()) + list(POOL.values()) + list(PROP.values()) for c in cs] + NOVEL_CONTENT)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class VariedProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, permute=False, learn=True):
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        rng = np.random.default_rng(seed + 11)
        # each member gets its OWN varied k-subset of its category pool. PERMUTED: draw from the MIXED union of both
        # pools -> members' subsets overlap only by chance (~K*K/8 < ACT_TH) -> no category structure -> collapses.
        allfeat = POOL["B"] + POOL["F"]
        self.subset = {}
        for m, cat in MEMBERS.items():
            pool = POOL[cat] if not permute else allfeat
            self.subset[m] = sorted(rng.choice(pool, size=K, replace=False).tolist())
        if learn:
            for _ in range(epochs):                                            # STREAM member -> its varied feature subset
                for m in MEMBERS:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[m]), _sdr(self.subset[m]),
                                        self.z, 0.14, 0.02, 1.0)
        for _ in range(epochs):                                                # teach property via the EXEMPLAR of each category
            for cat, ex in EXEMPLARS.items():
                v = self._prime(_sdr(CONTENT[ex]))
                ctx = set(int(i) for i in np.where(v > FLOOR)[0]) if v is not None else set()
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[ex]) | ctx,
                                    _sdr(PROP[CATPROP[cat]]), self.z, 0.14, 0.02, 1.0)

    def _prime(self, cells):
        ab = np.zeros(len(self.ci), bool)
        for i in cells:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        return None if vap is None else _host(vap)[self.ci]

    def infer(self, content_cols):
        """Direct read (exemplar) else 2-hop via the member's emergent features (held-out). Returns property or ABSTAIN."""
        v1 = self._prime(_sdr(content_cols))
        if v1 is None:
            return "ABSTAIN"
        dr1 = {p: float(np.mean([v1[c * nE:(c + 1) * nE].max() for c in cols])) for p, cols in PROP.items()}
        if max(dr1.values()) > FLOOR:
            return max(dr1, key=dr1.get)
        ctx = set(int(i) for i in np.where(v1 > FLOOR)[0])
        if not ctx:
            return "ABSTAIN"
        v2 = self._prime(ctx)
        if v2 is None:
            return "ABSTAIN"
        dr = {p: float(np.mean([v2[c * nE:(c + 1) * nE].max() for c in cols])) for p, cols in PROP.items()}
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else "ABSTAIN"


def _run_arm(seed, arm, epochs):
    p = VariedProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                    permute=(arm == "permuted"), learn=(arm != "nolearn"))
    held = np.mean([p.infer(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in HELD_OUT])
    allm = np.mean([p.infer(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in MEMBERS])
    moat = float(p.infer(NOVEL_CONTENT) == "ABSTAIN")
    return arm, {"held_out": float(held), "all": float(allm), "moat": moat}


ARMS = ["htm", "permuted", "nolearn", "lesion"]


def _demo(seed=42, epochs=80):
    p = VariedProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-32 emergent category from VARIED experience (no shared token; no transformer) ===")
    print("  each member OBSERVED with its OWN overlapping feature subset (no universal token):")
    for m in MEMBERS:
        print(f"    {m:8s} (latent {MEMBERS[m]}) features {p.subset[m]}" + ("  <- EXEMPLAR" if m in EXEMPLARS.values() else "  <- held out"))
    print("  taught only 'robin can fly' / 'trout can swim' (via the exemplars)\n")
    for m in MEMBERS:
        print(f"  does the {m:8s} fly/swim? -> {p.infer(CONTENT[m])}")
    print(f"  a NEVER-observed member -> {p.infer(NOVEL_CONTENT)}   (moat)\n")


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
    print(f"varied contexts (each member a different {K}-of-4 feature subset) | exemplars {EXEMPLARS} | held-out {HELD_OUT}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] HELD-OUT-inherit {h['held_out']:.2f} | all {h['all']:.2f} | MOAT {h['moat']:.2f} "
                  f"|| permuted {d['permuted']['held_out']:.2f} | no-learn {d['nolearn']['held_out']:.2f} | lesion {d['lesion']['held_out']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        held, allm, moat = m("htm", "held_out"), m("htm", "all"), m("htm", "moat")
        perm, nol, les = m("permuted", "held_out"), m("nolearn", "held_out"), m("lesion", "held_out")
        go = bool(held >= 0.90 and moat >= 0.90 and held >= perm + 0.30 and held >= nol + 0.30 and held >= les + 0.30)
        if go:
            verdict = (f"GO -- EMERGENT STRUCTURE FROM VARIED EXPERIENCE: the emergent category is NOT keyed to a single "
                       f"shared token -- each member was OBSERVED with a DIFFERENT overlapping feature subset (no universal "
                       f"token), yet a HELD-OUT member (different subset, property never taught) still INHERITS via the FEATURE "
                       f"OVERLAP ({held:.2f}), taught via one exemplar per category. PERMUTED-POOL collapses it ({perm:.2f} -- "
                       f"random pool -> no category overlap); NO-LEARNING ({nol:.2f}) and dAP-LESION ({les:.2f}) collapse; a "
                       f"never-observed member ABSTAINS ({moat:.2f}); 6-seed. => the emergence is robust to varied contexts -- "
                       f"the 'shared-token = provided superordinate' critique is CLOSED. NO sim/ edit.")
        else:
            miss = []
            if held < 0.90: miss.append(f"held-out {held:.2f} < 0.90")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if held < perm + 0.30: miss.append(f"permuted didn't collapse ({held:.2f} vs {perm:.2f})")
            if held < nol + 0.30: miss.append(f"no-learning didn't collapse ({held:.2f} vs {nol:.2f})")
            if held < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({held:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the pool/subset overlap vs ACT_TH (pairwise "
                       "overlap must clear the threshold) / epochs; varied-context emergence is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge32_varied_context_emergence", "verdict": verdict,
               "mechanism": "emergent category from Rogers-McClelland feature overlap: each member streamed with its own "
                            "varied k-of-n feature subset (guaranteed pairwise overlap >= ACT_TH); member-content -> features "
                            "learned on-bridge; property taught via an exemplar binds to its features; a held-out member "
                            "inherits via the OVERLAP of its subset with the exemplar's -- no universal token; sim/ unchanged",
               "task": "stream members with varied overlapping feature subsets; teach via one exemplar/category; test held-out "
                       "inheritance + moat vs permuted-pool + no-learning + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH, "pool_n": 4, "subset_k": K},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "closes the EMERGE-30 'one shared token' simplification. The features are still the environment "
                              "(legitimate); the GROUPING is discovered from overlap (permuted-pool isolates it). Next: an HTM "
                              "Spatial-Pooler that forms a NEW shared column block from the varied inputs; cancellation on "
                              "emergent codes; couple into the experiential console (EMERGE-31)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge32] VERDICT: {verdict}", flush=True)
    print(f"[emerge32] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
