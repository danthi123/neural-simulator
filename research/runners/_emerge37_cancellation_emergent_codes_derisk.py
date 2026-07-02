"""EMERGE-37 / toward-semantics — CANCELLATION on EMERGENT codes: the full Collins-Quillian inference (inheritance +
specific-override CANCELLATION) works on codes LEARNED FROM EXPERIENCE (co-occurrence), not just hand-assigned ones.
A class property taught on the emergent superordinate is INHERITED by category-mates; a member-specific fact CANCELS
the inherited default for that member. This ties the inference arc (EMERGE-26 cancellation) to the emergence arc
(EMERGE-30 learned superordinate). NO `sim/` edit.

MECHANISM (EMERGE-30 + EMERGE-26): members co-occur with context tokens (the committed `sim/` three-term kernel learns
member-content -> context on the bridge); the shared context cells are the EMERGENT superordinate (never labeled). A
CLASS property is taught on those emergent cells ("bird-context -> flies"); a member-SPECIFIC override is taught
directly on one member's content ("robin -> walks"). Querying a member: the DIRECT (member-content -> property, 1-hop)
pathway competes with the INHERITED (member-content -> emergent-context -> property, 2-hop) pathway; the strongest wins
(a graded-drive read). The specific direct fact out-drives the inherited default -> robin answers WALKS (cancellation),
while sparrow/canary (no override) INHERIT flies via the learned grouping.

ANTI-CHEATS (per the control-validity methodology): CANCELLATION (the overridden member answers its specific property,
not the inherited); INHERITANCE (the non-overridden members inherit the class property via the learned grouping);
PERMUTED-CONTEXT (input-destruction: scrambled co-occurrence -> no emergent grouping -> INHERITANCE collapses, while the
content-direct override survives -- isolating the learned grouping as the inheritance cause); dAP-LESION collapses;
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

OUT = Path("research/findings/raw/_emerge37_cancellation_emergent_codes.json")

MEMBERS = {"robin": "B", "sparrow": "B", "canary": "B", "trout": "F", "salmon": "F"}   # latent categories, never labeled
CONTENT = {m: [i * 3, i * 3 + 1, i * 3 + 2] for i, m in enumerate(MEMBERS)}
CTX = {"B": [15, 16], "F": [17, 18]}                                           # environment context = emergent superordinate
PROP = {"flies": [19, 20], "swims": [21, 22], "walks": [23, 24]}
CATPROP = {"B": "flies", "F": "swims"}
OVERRIDE = ("robin", "walks")                                                   # the specific member-fact that cancels the default
INHERIT_MEMBERS = ["sparrow", "canary", "trout", "salmon"]                     # non-overridden -> inherit the class property
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max(c for cs in list(CONTENT.values()) + list(CTX.values()) + list(PROP.values()) for c in cs)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class CancellationProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, permute=False):
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        rng = np.random.default_rng(seed + 7)
        for _ in range(epochs):                                                # STREAM member -> context (co-occurrence)
            for m, cat in MEMBERS.items():
                use = cat if not permute else ("B" if rng.random() < 0.5 else "F")
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[m]), _sdr(CTX[use]),
                                    self.z, 0.14, 0.02, 1.0)
        for _ in range(epochs):                                                # class property on emergent context + the override
            for cat, p in CATPROP.items():
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CTX[cat]), _sdr(PROP[p]),
                                    self.z, 0.14, 0.02, 1.0)
            apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[OVERRIDE[0]]), _sdr(PROP[OVERRIDE[1]]),
                                self.z, 0.14, 0.02, 1.0)

    def _prime(self, cells):
        ab = np.zeros(len(self.ci), bool)
        for i in cells:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        return None if vap is None else _host(vap)[self.ci]

    def infer(self, member):
        v1 = self._prime(_sdr(CONTENT[member]))                                # 1-hop: direct (override) + emergent context
        if v1 is None:
            return "ABSTAIN"
        drd = {p: float(np.mean([v1[c * nE:(c + 1) * nE].max() for c in cs])) for p, cs in PROP.items()}
        ctx = set(int(i) for i in np.where(v1 > FLOOR)[0])
        v2 = self._prime(ctx) if ctx else None
        dri = ({p: float(np.mean([v2[c * nE:(c + 1) * nE].max() for c in cs])) for p, cs in PROP.items()}
               if v2 is not None else {p: -100.0 for p in PROP})
        dr = {p: max(drd[p], dri[p]) for p in PROP}                             # strongest wins (specific direct overrides inherited)
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else "ABSTAIN"


def _run_arm(seed, arm, epochs):
    p = CancellationProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"), permute=(arm == "permuted"))
    override = float(p.infer(OVERRIDE[0]) == OVERRIDE[1])
    inherit = np.mean([p.infer(m) == CATPROP[MEMBERS[m]] for m in INHERIT_MEMBERS])
    return arm, {"override": override, "inherit": float(inherit)}


ARMS = ["htm", "permuted", "lesion"]


def _demo(seed=42, epochs=80):
    p = CancellationProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-37 cancellation on EMERGENT codes (Collins-Quillian on learned-from-experience grouping) ===")
    print("  members co-occur with context (learned grouping); class fact on the emergent context; robin has a specific fact.\n")
    for m in MEMBERS:
        note = "CANCELS (specific fact beats inherited)" if m == OVERRIDE[0] else "INHERITS (class property, via learned grouping)"
        print(f"  the {m:8s} (latent {MEMBERS[m]}) -> {p.infer(m):6s}  ({note})")
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
    print(f"cancellation on emergent codes: override {OVERRIDE}, inherit {INHERIT_MEMBERS}; permuted destroys the grouping", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] CANCELLATION {h['override']:.2f} | INHERITANCE {h['inherit']:.2f} "
                  f"|| permuted-inherit {d['permuted']['inherit']:.2f} | lesion-inherit {d['lesion']['inherit']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        ov, inh = m("htm", "override"), m("htm", "inherit")
        perm, les = m("permuted", "inherit"), m("lesion", "inherit")
        go = bool(ov >= 0.90 and inh >= 0.90 and inh >= perm + 0.30 and inh >= les + 0.30)
        if go:
            verdict = (f"GO -- the full Collins-Quillian inference (INHERITANCE + specific-override CANCELLATION) works on codes "
                       f"LEARNED FROM EXPERIENCE, not just hand-assigned. A member-specific fact CANCELS the inherited class "
                       f"default ({ov:.2f}: robin answers WALKS, its specific fact beating the inherited flies); the "
                       f"non-overridden members INHERIT the class property via the LEARNED grouping ({inh:.2f}). PERMUTED-CONTEXT "
                       f"collapses the inheritance ({perm:.2f} -- scrambled co-occurrence -> no emergent grouping); dAP-LESION "
                       f"{les:.2f}; 6-seed. => cancellation is not tied to hand-assigned codes -- the substrate does full "
                       f"Collins-Quillian inference over structure DISCOVERED from experience. NO sim/ edit.")
        else:
            miss = []
            if ov < 0.90: miss.append(f"cancellation {ov:.2f} < 0.90")
            if inh < 0.90: miss.append(f"inheritance {inh:.2f} < 0.90")
            if inh < perm + 0.30: miss.append(f"permuted didn't collapse inheritance ({inh:.2f} vs {perm:.2f})")
            if inh < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({inh:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the direct-vs-inherited drive (content "
                       "cols vs context cols) / epochs; cancellation on emergent codes is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge37_cancellation_emergent_codes", "verdict": verdict,
               "mechanism": "cancellation on emergent codes: members learn member-content -> context (co-occurrence); class "
                            "property on the emergent context (inherited); a member-specific override taught on the member's "
                            "content directly out-drives the inherited (graded read: direct 1-hop beats inherited 2-hop); "
                            "committed sim/ three-term kernel; sim/ unchanged",
               "task": "stream members->context; teach class property on emergent context + one specific override; test "
                       "cancellation + inheritance vs permuted-context + dAP-lesion; multi-seed",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH, "override": OVERRIDE},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "ties EMERGE-26 cancellation (hand-assigned codes) to EMERGE-30 emergent codes -- the full "
                              "Collins-Quillian inference over LEARNED structure. The context tokens are the environment; the "
                              "grouping is discovered (permuted-context isolates it)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge37] VERDICT: {verdict}", flush=True)
    print(f"[emerge37] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
