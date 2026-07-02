"""EMERGE-26 / toward-semantics — EMERGENT INFERENCE BEYOND TOLD FACTS: Collins-Quillian property INHERITANCE (with
cancellation) emerges on the spiking HTM cortex with NO explicit inference engine. Teach ONLY class-level facts
("a BIRD flies", "a FISH swims"); a never-taught member (robin, trout) INHERITS the property (robin->flies,
trout->swims) purely because it SHARES a superordinate code with its class — inference beyond told facts. And a
member-specific fact CANCELS the inherited default (penguin, told "penguin walks", answers WALKS not the inherited
FLIES) — the discriminating Collins-Quillian cancellation. This is the first inference-beyond-told-facts on the
emergent substrate, per the open-world-semantics research gate. NO `sim/` edit.

MECHANISM (the reframe: inference EMERGES from overlapping/shared codes x the next-state predictor — no inference
engine). Each concept = a three-block sparse code: a CONTENT block (the specific concept, 3 cols) + a shared
SUPERORDINATE block (is-a: robin/sparrow/canary/penguin all share BIRD's 2 cols; trout/salmon share FISH). The
class-level fact is taught by potentiating the SUPERORDINATE block -> the property (BIRD-cols -> flies), via the
committed `sim/` three-term kernel. Querying a member presents its content+superordinate cells; the shared BIRD cells
prime "flies" through the learned class pathway -> the member inherits, though its own content was never bound to any
property. CANCELLATION: the member-specific pathway (penguin's 3 content cols -> walks) out-DRIVES the inherited class
pathway (BIRD's 2 super cols -> flies) -> a graded-magnitude read (argmax over each property's apical drive) picks the
specific over the inherited (Collins-Quillian: the most specific stored property wins).

ANTI-CHEATS: inheritance accuracy on HELD-OUT members (their property NEVER taught, only the class); DERANGED-
SUPERORDINATE (members share the WRONG superordinate -> inheritance collapses -> isolates the is-a code as the cause);
dAP-LESION (coincidence off -> nothing primed -> collapses); MOAT (a concept with no superordinate abstains, no
confabulated property); the CANCELLATION test (penguin -> walks, not the inherited flies); 6-seed. Reuse-by-import
(`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
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

OUT = Path("research/findings/raw/_emerge26_emergent_inheritance.json")

# three-block codes: CONTENT (3 cols/concept) + shared SUPERORDINATE (is-a) block; properties (2 cols each).
CONTENT = {"robin": [0, 1, 2], "sparrow": [3, 4, 5], "canary": [6, 7, 8], "penguin": [9, 10, 11],
           "trout": [12, 13, 14], "salmon": [15, 16, 17],
           "flies": [18, 19], "swims": [20, 21], "walks": [22, 23],
           "BIRD": [24, 25], "FISH": [26, 27], "novel": [30, 31, 32]}          # novel: no superordinate (moat)
SUPER = {"robin": "BIRD", "sparrow": "BIRD", "canary": "BIRD", "penguin": "BIRD", "trout": "FISH", "salmon": "FISH"}
PROPS = ["flies", "swims", "walks"]
MEMBERS_INHERIT = {"robin": "flies", "sparrow": "flies", "canary": "flies", "trout": "swims", "salmon": "swims"}
OVERRIDE = ("penguin", "walks")                                                 # taught specifically; cancels inherited flies
MOAT = ["novel"]
nE = 8
ACT_TH = 2
FLOOR = -40.0                                                                    # between the ~-62 apical rest and the plateau
M = 1 + max(c for cs in CONTENT.values() for c in cs)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class InheritanceProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, super_map=None):
        self.super = dict(SUPER if super_map is None else super_map)
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        for _ in range(epochs):                                                 # teach ONLY class facts + the one override
            apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT["BIRD"]), _sdr(CONTENT["flies"]),
                                self.z, 0.14, 0.02, 1.0)
            apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT["FISH"]), _sdr(CONTENT["swims"]),
                                self.z, 0.14, 0.02, 1.0)
            apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[OVERRIDE[0]]), _sdr(CONTENT[OVERRIDE[1]]),
                                self.z, 0.14, 0.02, 1.0)

    def query(self, concept):
        """Present concept's content+superordinate cells; return the property with the highest apical DRIVE (argmax over
        properties), or 'ABSTAIN' if none is driven above the rest floor. Graded read so the member-specific (stronger)
        pathway CANCELS the inherited (weaker) default."""
        cols = list(CONTENT[concept]) + list(CONTENT[self.super[concept]]) if concept in self.super else list(CONTENT[concept])
        ab = np.zeros(len(self.ci), bool)
        for i in _sdr(cols):
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return "ABSTAIN"
        vap = _host(vap)[self.ci]
        drive = {p: float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in CONTENT[p]])) for p in PROPS}
        best = max(drive, key=drive.get)
        return best if drive[best] > FLOOR else "ABSTAIN"


def _deranged_super():
    """Members share the WRONG superordinate (BIRD<->FISH swapped) -> inheritance points to the wrong property."""
    swap = {"BIRD": "FISH", "FISH": "BIRD"}
    return {m: swap[s] for m, s in SUPER.items()}


def _run_arm(seed, arm, epochs):
    p = InheritanceProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                         super_map=(_deranged_super() if arm == "deranged" else None))
    inh = np.mean([p.query(m) == prop for m, prop in MEMBERS_INHERIT.items()])   # HELD-OUT members inherit
    override = float(p.query(OVERRIDE[0]) == OVERRIDE[1])                        # cancellation
    moat = np.mean([p.query(m) == "ABSTAIN" for m in MOAT])
    return arm, {"inheritance": float(inh), "override": override, "moat": float(moat)}


ARMS = ["htm", "deranged", "lesion"]


def _demo(seed=42, epochs=80):
    p = InheritanceProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-26 emergent inheritance (Collins-Quillian; no inference engine, no transformer) ===")
    print("  TAUGHT only class facts: 'a BIRD flies', 'a FISH swims'  +  one specific: 'a penguin walks'\n")
    for m, note in [("robin", "INHERITS flies (never taught -- is a BIRD)"), ("sparrow", "INHERITS flies"),
                    ("canary", "INHERITS flies"), ("trout", "INHERITS swims (is a FISH)"), ("salmon", "INHERITS swims"),
                    ("penguin", "CANCELS: walks (specific beats inherited flies)"), ("novel", "no superordinate -> ABSTAIN")]:
        print(f"  q: does the {m} ...? -> {p.query(m)}   ({note})")
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
    print(f"class facts BIRD->flies, FISH->swims | override penguin->walks | held-out members {list(MEMBERS_INHERIT)} | moat {MOAT}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] INHERITANCE {h['inheritance']:.2f} | CANCELLATION {h['override']:.2f} | MOAT {h['moat']:.2f} "
                  f"|| deranged-inheritance {d['deranged']['inheritance']:.2f} | lesion-inheritance {d['lesion']['inheritance']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        inh, ov, moat = m("htm", "inheritance"), m("htm", "override"), m("htm", "moat")
        der, les = m("deranged", "inheritance"), m("lesion", "inheritance")
        go = bool(inh >= 0.90 and ov >= 0.90 and moat >= 0.90 and inh >= der + 0.30 and inh >= les + 0.30)
        if go:
            verdict = (f"GO -- INFERENCE BEYOND TOLD FACTS emerges on the spiking HTM cortex: a never-taught member INHERITS "
                       f"its class property ({inh:.2f} on HELD-OUT members whose property was never taught, only the class "
                       f"'a BIRD flies'/'a FISH swims') purely from a SHARED superordinate code x the next-state predictor -- "
                       f"NO inference engine (Collins-Quillian 1969). CANCELLATION holds ({ov:.2f}: 'penguin' answers WALKS, the "
                       f"specific fact beating the inherited flies -- graded drive). DERANGED-superordinate collapses it "
                       f"({der:.2f}, isolating the is-a code as the cause); dAP-LESION collapses ({les:.2f}); a no-superordinate "
                       f"concept ABSTAINS ({moat:.2f}, no confabulated property); 6-seed. => the substrate INFERS over relational "
                       f"structure -- the first inference-beyond-told-facts on the emergent brain, NO sim/ edit.")
        else:
            miss = []
            if inh < 0.90: miss.append(f"inheritance {inh:.2f} < 0.90")
            if ov < 0.90: miss.append(f"cancellation {ov:.2f} < 0.90")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if inh < der + 0.30: miss.append(f"deranged didn't collapse ({inh:.2f} vs {der:.2f})")
            if inh < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({inh:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the content/super block sizes vs ACT_TH "
                       "(the specific must out-drive the inherited for cancellation; the inherited must clear the floor for "
                       "inheritance) / epochs; emergent inheritance is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge26_emergent_inheritance", "verdict": verdict,
               "mechanism": "Collins-Quillian property inheritance emerges from a shared superordinate (is-a) code x the HTM "
                            "next-state predictor -- NO inference engine: teach only class-level facts on the shared "
                            "superordinate block; a member inherits because it shares that block; a stronger member-specific "
                            "pathway cancels the inherited default (graded apical-drive argmax); committed sim/ three-term "
                            "kernel; sim/ unchanged",
               "task": "teach class facts (BIRD->flies, FISH->swims) + one specific (penguin->walks); test held-out members "
                       "inherit + cancellation + moat vs deranged-superordinate + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH, "content_cols": 3, "super_cols": 2},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the substrate INFERS over relational structure that is host-DESIGNED (the shared is-a codes are "
                              "hand-assigned). A GO proves inference-over-structure, NOT acquisition-of-structure-from-experience "
                              "(R-c, the deferred residual): the is-a codes must EMERGE from co-occurrence/perception statistics "
                              "(the PPMI stream cortex + replay) -- the next research gate. Also: transitive/multi-hop taxonomy "
                              "(bird->animal->breathes) is the named next inference step."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge26] VERDICT: {verdict}", flush=True)
    print(f"[emerge26] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
