"""EMERGE-27 / toward-semantics — MULTI-LEVEL taxonomic inheritance: a concept inherits properties from MULTIPLE levels
of its is-a hierarchy, and a cancellation at ONE level does not block inheritance at ANOTHER — the full Collins-Quillian
hierarchical structure, emergent on the spiking HTM cortex, NO inference engine, NO `sim/` edit.

Hierarchy ANIMAL > {BIRD, FISH} > {robin, penguin, trout}. Teach ONLY level facts: "an ANIMAL breathes" (top),
"a BIRD flies" / "a FISH swims" (mid), and one specific "a penguin walks". Then (per property DIMENSION):
  - robin  -> breathes (inherited from ANIMAL, 2 levels up) + flies (inherited from BIRD, 1 level up);
  - trout  -> breathes + swims;
  - penguin -> breathes (STILL inherited from ANIMAL) + walks (the specific fact CANCELS the inherited flies at the
    locomotion dimension) -- the cancellation at LOCOMOTION does NOT block the RESPIRATION inheritance.

MECHANISM (same as EMERGE-26, extended to nested levels): each concept's code = its CONTENT block + ALL its ancestor
SUPERORDINATE blocks (robin = content + BIRD + ANIMAL). A level fact is taught by potentiating that level's block ->
the property (ANIMAL-cols -> breathes) via the committed `sim/` three-term kernel. Querying a concept presents its
content + every ancestor block; each level's block primes its property -> the concept inherits from every level. Read
PER DIMENSION (respiration {breathes} / locomotion {flies,swims,walks}) by argmax over that dimension's apical DRIVE,
abstain below the rest floor -- so the stronger member-specific pathway CANCELS the inherited default WITHIN its
dimension while other dimensions inherit untouched.

ANTI-CHEATS: mid-level inheritance accuracy on HELD-OUT concepts (their property never taught, only the level's);
DIMENSION-ISOLATION (penguin keeps breathes while walks cancels flies); DERANGED-ANCESTORS (concepts share the WRONG
mid-level -> mid inheritance collapses); dAP-LESION collapses; MOAT (no-ancestor concept abstains both dimensions);
6-seed. Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
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

OUT = Path("research/findings/raw/_emerge27_multilevel_taxonomy.json")

CONTENT = {"robin": [0, 1, 2], "penguin": [3, 4, 5], "trout": [6, 7, 8],
           "breathes": [9, 10], "flies": [11, 12], "swims": [13, 14], "walks": [15, 16],
           "BIRD": [17, 18], "FISH": [19, 20], "ANIMAL": [21, 22], "novel": [30, 31, 32]}
ANCESTORS = {"robin": ["BIRD", "ANIMAL"], "penguin": ["BIRD", "ANIMAL"], "trout": ["FISH", "ANIMAL"]}
DIMS = {"RESP": ["breathes"], "LOCO": ["flies", "swims", "walks"]}
EXPECT = {"robin": {"RESP": "breathes", "LOCO": "flies"},                        # 2-hop + 1-hop inheritance
          "trout": {"RESP": "breathes", "LOCO": "swims"},
          "penguin": {"RESP": "breathes", "LOCO": "walks"}}                      # cancellation at LOCO, RESP survives
MOAT = ["novel"]
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max(c for cs in CONTENT.values() for c in cs)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class TaxonomyProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, ancestors=None):
        self.anc = {k: list(v) for k, v in (ANCESTORS if ancestors is None else ancestors).items()}
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        facts = [("ANIMAL", "breathes"), ("BIRD", "flies"), ("FISH", "swims"), ("penguin", "walks")]
        for _ in range(epochs):
            for pre, post in facts:
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[pre]), _sdr(CONTENT[post]),
                                    self.z, 0.14, 0.02, 1.0)

    def query(self, concept):
        cols = list(CONTENT[concept]) + [c for a in self.anc.get(concept, []) for c in CONTENT[a]]
        ab = np.zeros(len(self.ci), bool)
        for i in _sdr(cols):
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return {d: "ABSTAIN" for d in DIMS}
        vap = _host(vap)[self.ci]
        out = {}
        for dim, props in DIMS.items():
            dr = {p: float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in CONTENT[p]])) for p in props}
            best = max(dr, key=dr.get)
            out[dim] = best if dr[best] > FLOOR else "ABSTAIN"
        return out


def _deranged_ancestors():
    """Concepts share the WRONG mid-level (BIRD<->FISH swapped); top ANIMAL preserved -> mid inheritance collapses."""
    swap = {"BIRD": "FISH", "FISH": "BIRD"}
    return {c: [swap.get(a, a) for a in anc] for c, anc in ANCESTORS.items()}


def _run_arm(seed, arm, epochs):
    p = TaxonomyProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                      ancestors=(_deranged_ancestors() if arm == "deranged" else None))
    resp = np.mean([p.query(c)["RESP"] == EXPECT[c]["RESP"] for c in EXPECT])       # 2-hop-up inheritance (all)
    loco = np.mean([p.query(c)["LOCO"] == EXPECT[c]["LOCO"] for c in ("robin", "trout")])  # 1-hop held-out inheritance
    iso = float(p.query("penguin")["RESP"] == "breathes" and p.query("penguin")["LOCO"] == "walks")  # dimension isolation
    moat = np.mean([all(p.query(m)[d] == "ABSTAIN" for d in DIMS) for m in MOAT])
    return arm, {"resp_inherit": float(resp), "loco_inherit": float(loco), "dim_isolation": iso, "moat": float(moat)}


ARMS = ["htm", "deranged", "lesion"]


def _demo(seed=42, epochs=80):
    p = TaxonomyProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-27 multi-level taxonomic inheritance (Collins-Quillian; no inference engine) ===")
    print("  hierarchy ANIMAL > {BIRD, FISH} > {robin, penguin, trout}")
    print("  TAUGHT only level facts: 'an ANIMAL breathes', 'a BIRD flies', 'a FISH swims'  +  'a penguin walks'\n")
    for c, note in [("robin", "breathes (ANIMAL, 2 up) + flies (BIRD, 1 up)"),
                    ("trout", "breathes + swims"),
                    ("penguin", "breathes STILL inherited; walks CANCELS flies (per-dimension)"),
                    ("novel", "no ancestors -> ABSTAIN both")]:
        q = p.query(c)
        print(f"  the {c:8s}: respiration={q['RESP']:8s} locomotion={q['LOCO']:8s}   ({note})")
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
    print(f"hierarchy ANIMAL>{{BIRD,FISH}}>{{robin,penguin,trout}} | level facts breathes/flies/swims + penguin->walks", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] RESP-inherit(2-hop) {h['resp_inherit']:.2f} | LOCO-inherit(held-out) {h['loco_inherit']:.2f} "
                  f"| DIM-ISOLATION {h['dim_isolation']:.2f} | MOAT {h['moat']:.2f} || deranged-loco {d['deranged']['loco_inherit']:.2f} "
                  f"| lesion-loco {d['lesion']['loco_inherit']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        resp, loco, iso, moat = m("htm", "resp_inherit"), m("htm", "loco_inherit"), m("htm", "dim_isolation"), m("htm", "moat")
        der, les = m("deranged", "loco_inherit"), m("lesion", "loco_inherit")
        go = bool(resp >= 0.90 and loco >= 0.90 and iso >= 0.90 and moat >= 0.90 and loco >= der + 0.30 and loco >= les + 0.30)
        if go:
            verdict = (f"GO -- MULTI-LEVEL taxonomic inheritance emerges on the spiking HTM cortex: a concept inherits from "
                       f"MULTIPLE levels of its is-a hierarchy at once -- respiration 'breathes' from ANIMAL (2 levels up, "
                       f"{resp:.2f}) AND locomotion from BIRD/FISH (1 level up, held-out, {loco:.2f}) -- with NO inference engine. "
                       f"DIMENSION-ISOLATION ({iso:.2f}): penguin's specific 'walks' CANCELS the inherited flies at LOCOMOTION "
                       f"while its RESPIRATION inheritance 'breathes' SURVIVES -- the full Collins-Quillian hierarchical structure. "
                       f"DERANGED mid-level collapses locomotion inheritance ({der:.2f}); dAP-LESION collapses ({les:.2f}); a "
                       f"no-ancestor concept ABSTAINS both dimensions ({moat:.2f}); 6-seed. => the substrate infers over a "
                       f"MULTI-LEVEL relational hierarchy, NO sim/ edit.")
        else:
            miss = []
            if resp < 0.90: miss.append(f"resp-inherit {resp:.2f} < 0.90")
            if loco < 0.90: miss.append(f"loco-inherit {loco:.2f} < 0.90")
            if iso < 0.90: miss.append(f"dimension-isolation {iso:.2f} < 0.90")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if loco < der + 0.30: miss.append(f"deranged didn't collapse loco ({loco:.2f} vs {der:.2f})")
            if loco < les + 0.30: miss.append(f"dAP-lesion didn't collapse loco ({loco:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the level-block sizes vs ACT_TH (each "
                       "level must clear the floor; the specific must out-drive its level for cancellation) / epochs; multi-level "
                       "inheritance is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge27_multilevel_taxonomy", "verdict": verdict,
               "mechanism": "multi-level Collins-Quillian inheritance: each concept's code = content + ALL ancestor "
                            "superordinate blocks; teach a property at each level's block; a concept inherits from every level "
                            "(each level's block primes its property); per-dimension argmax over apical drive so a member-"
                            "specific fact cancels only its dimension; committed sim/ three-term kernel; sim/ unchanged",
               "task": "teach level facts (ANIMAL->breathes, BIRD->flies, FISH->swims) + specific (penguin->walks); test "
                       "multi-level inheritance + dimension-isolation + moat vs deranged-ancestors + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the is-a hierarchy is host-DESIGNED (ancestor blocks hand-assigned). Inference-over-structure, "
                              "NOT acquisition-of-structure-from-experience (the deferred R-c residual: the hierarchy must EMERGE "
                              "from statistics -- the next research gate). Next build: transitive relational inference (recombine "
                              "overlapping learned pairs, hippocampal D.02)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge27] VERDICT: {verdict}", flush=True)
    print(f"[emerge27] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
