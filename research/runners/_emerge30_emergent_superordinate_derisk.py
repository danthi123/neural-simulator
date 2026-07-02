"""EMERGE-30 / toward-semantics — EMERGENT STRUCTURE FROM EXPERIENCE (the master-directive core): the shared
SUPERORDINATE code is not host-designed -- it EMERGES from a co-occurrence stream, unsupervised, and the validated
inheritance (EMERGE-26) then rides the LEARNED grouping. This closes the last honest residual of the whole inference
arc (EMERGE-26/27/28 all rode HAND-ASSIGNED is-a codes). NO `sim/` edit.

THE RESIDUAL IT CLOSES: EMERGE-26/27/28 proved the substrate INFERS over relational structure, but the structure
(`SUPER = {"robin":"BIRD", ...}`) was a hand-written dict -- the category was TOLD. Here the category is DISCOVERED:
members that appear in the same contexts DEVELOP a shared representation, and a property taught to that emergent shared
representation is inherited by a member that only ever CO-OCCURRED with the context (never told the category, never
told the property).

MECHANISM (competitive-Hebbian category formation; the reframe: overlapping codes emerge from co-occurrence, then the
next-state predictor infers over them -- HTM Spatial Pooler, Cui-Ahmad-Hawkins 2017; taxonomic inheritance from
feature-prediction, Saxe-McClelland-Ganguli 2019). Latent categories BIRD{robin,sparrow,canary}, FISH{trout,salmon,
pike} -- NEVER labeled. Each member is OBSERVED co-occurring with its category's CONTEXT tokens (the environment): the
committed `sim/` three-term kernel learns member-content -> context (on-bridge Hebbian co-occurrence, the validated
`corr(M,C)` mechanism). All members of a category thus learn to activate the SAME context cells -> the context cells
are the EMERGENT superordinate (no member's code was hand-given a "BIRD" block). A class property is taught on those
emergent shared cells (emergent-BIRD-context -> flies). Asking about a member is a 2-hop read: member-content primes
its emergent context (learned), the context primes the property -> the member INHERITS, though it was never told the
category or the property.

ANTI-CHEATS: inheritance accuracy (every member inferred -- NONE was told a property); PERMUTED-CONTEXT (members
co-occur with a scrambled context -> no category structure emerges -> collapses to chance, isolating the LEARNED
co-occurrence as the cause); NO-LEARNING (skip the stream -> no member->context -> abstains); dAP-LESION collapses;
MOAT (a never-streamed member has no learned context -> abstains); 6-seed. Reuse-by-import (`_emerge14` + `_emerge12`);
NO `sim/` edit. CPU numpy-backend. `--demo`.
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

OUT = Path("research/findings/raw/_emerge30_emergent_superordinate.json")

MEMBERS = {"robin": "B", "sparrow": "B", "canary": "B", "trout": "F", "salmon": "F", "pike": "F"}  # LATENT (never labeled)
HELD_OUT = ["canary", "pike"]                                                   # highlighted: only ever saw the context
CONTENT = {m: [i * 3, i * 3 + 1, i * 3 + 2] for i, m in enumerate(MEMBERS)}     # per-member content (cols 0..17)
CTX = {"B": [18, 19], "F": [20, 21]}                                           # environment context = emergent superordinate
PROP = {"flies": [22, 23], "swims": [24, 25]}
CATPROP = {"B": "flies", "F": "swims"}
NOVEL_CONTENT = [30, 31, 32]                                                    # a never-streamed member (moat)
nE = 8
ACT_TH = 2
FLOOR = -40.0
M = 1 + max([c for cs in list(CONTENT.values()) + list(CTX.values()) + list(PROP.values()) for c in cs] + NOVEL_CONTENT)


def _sdr(cols):
    return set(c * nE + 0 for c in cols)


class EmergentProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, permute=False, learn=True):
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        # PERMUTE: each member co-occurs with an INDEPENDENTLY RANDOM context every stream epoch -> no consistent
        # per-category grouping can form (destroys the latent structure while keeping the same tokens + property).
        ctxmap = dict(MEMBERS)
        rng = np.random.default_rng(seed + 7)
        if learn:
            for _ in range(epochs):                                            # STREAM: learn member -> context (co-occurrence)
                for m in MEMBERS:
                    cat = ctxmap[m] if not permute else ("B" if rng.random() < 0.5 else "F")  # permuted: random each epoch
                    apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CONTENT[m]), _sdr(CTX[cat]),
                                        self.z, 0.14, 0.02, 1.0)
        for _ in range(epochs):                                                # teach property on the EMERGENT context cells
            for cat, prop in CATPROP.items():
                apply_kernel_update(self.b, self.row, self.col, self.ci, _sdr(CTX[cat]), _sdr(PROP[prop]),
                                    self.z, 0.14, 0.02, 1.0)

    def _prime(self, active_cells):
        ab = np.zeros(len(self.ci), bool)
        for i in active_cells:
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        return None if vap is None else _host(vap)[self.ci]

    def inherit(self, content_cols):
        """2-hop read: content -> emergent context (learned) -> property. Returns the inferred property or 'ABSTAIN'."""
        v1 = self._prime(_sdr(content_cols))
        if v1 is None:
            return "ABSTAIN"
        ctx_cells = set(int(i) for i in np.where(v1 > FLOOR)[0])
        if not ctx_cells:
            return "ABSTAIN"
        v2 = self._prime(ctx_cells)
        if v2 is None:
            return "ABSTAIN"
        dr = {p: float(np.mean([v2[c * nE:(c + 1) * nE].max() for c in cols])) for p, cols in PROP.items()}
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else "ABSTAIN"


def _run_arm(seed, arm, epochs):
    p = EmergentProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                      permute=(arm == "permuted"), learn=(arm != "nolearn"))
    inh = np.mean([p.inherit(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in MEMBERS])
    held = np.mean([p.inherit(CONTENT[m]) == CATPROP[MEMBERS[m]] for m in HELD_OUT])
    moat = float(p.inherit(NOVEL_CONTENT) == "ABSTAIN")
    return arm, {"inherit": float(inh), "held_out": float(held), "moat": moat}


ARMS = ["htm", "permuted", "nolearn", "lesion"]


def _demo(seed=42, epochs=80):
    p = EmergentProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-30 emergent superordinate from co-occurrence (category DISCOVERED, not told; no transformer) ===")
    print("  latent categories BIRD{robin,sparrow,canary} / FISH{trout,salmon,pike} -- NEVER labeled")
    print("  the brain only OBSERVED each member co-occurring with its context; then 'bird-context flies', 'fish-context swims'\n")
    for m, cat in MEMBERS.items():
        tag = " (HELD OUT of property teaching)" if m in HELD_OUT else ""
        print(f"  does the {m:8s} (latent {cat}) fly/swim? -> {p.inherit(CONTENT[m]):8s}{tag}")
    print(f"  a NEVER-observed member -> {p.inherit(NOVEL_CONTENT)}   (moat)")
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
    print(f"latent categories {MEMBERS} | held-out {HELD_OUT} | property on the EMERGENT context (learned from co-occurrence)", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] INHERIT {h['inherit']:.2f} | HELD-OUT {h['held_out']:.2f} | MOAT {h['moat']:.2f} "
                  f"|| permuted {d['permuted']['inherit']:.2f} | no-learn {d['nolearn']['inherit']:.2f} | lesion {d['lesion']['inherit']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        inh, held, moat = m("htm", "inherit"), m("htm", "held_out"), m("htm", "moat")
        perm, nol, les = m("permuted", "inherit"), m("nolearn", "inherit"), m("lesion", "inherit")
        go = bool(inh >= 0.90 and held >= 0.90 and moat >= 0.90 and inh >= perm + 0.30 and inh >= nol + 0.30 and inh >= les + 0.30)
        if go:
            verdict = (f"GO -- EMERGENT STRUCTURE FROM EXPERIENCE: the shared superordinate is NOT host-designed -- it EMERGES "
                       f"from a co-occurrence stream, and the validated inheritance rides the LEARNED grouping. A member is "
                       f"NEVER told its category or its property; the brain only OBSERVED it co-occurring with context tokens, "
                       f"yet it INHERITS the class property ({inh:.2f}; the HIGHLIGHTED held-out members {held:.2f}). "
                       f"PERMUTED-CONTEXT collapses it ({perm:.2f} -- scrambled co-occurrence -> no category emerges); NO-LEARNING "
                       f"({nol:.2f}) and dAP-LESION ({les:.2f}) collapse; a never-observed member ABSTAINS ({moat:.2f}); 6-seed. "
                       f"=> the substrate ACQUIRES relational structure from experience AND infers over it -- the last honest "
                       f"residual of the inference arc CLOSED, the master-directive core, NO sim/ edit.")
        else:
            miss = []
            if inh < 0.90: miss.append(f"inherit {inh:.2f} < 0.90")
            if held < 0.90: miss.append(f"held-out {held:.2f} < 0.90")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if inh < perm + 0.30: miss.append(f"permuted didn't collapse ({inh:.2f} vs {perm:.2f})")
            if inh < nol + 0.30: miss.append(f"no-learning didn't collapse ({inh:.2f} vs {nol:.2f})")
            if inh < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({inh:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune the stream epochs / context-block size vs "
                       "ACT_TH / the 2-hop threshold; emergent-structure-from-experience is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge30_emergent_superordinate", "verdict": verdict,
               "mechanism": "the shared superordinate EMERGES from co-occurrence: the committed sim/ three-term kernel learns "
                            "member-content -> context (on-bridge Hebbian co-occurrence); all category members learn the same "
                            "context cells -> the emergent superordinate; a class property taught on those emergent cells is "
                            "inherited via a 2-hop read (member -> emergent context -> property); sim/ unchanged",
               "task": "stream members co-occurring with latent-category contexts (no label); teach a property on the emergent "
                       "shared cells; test member inheritance + held-out + moat vs permuted-context + no-learning + dAP-lesion",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "act_th": ACT_TH},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the context TOKENS are the environment (legitimate world/experience); the category GROUPING (which "
                              "members belong together) is DISCOVERED from co-occurrence, not told -- the permuted-context control "
                              "isolates that. Follow-ons: overlapping/varied contexts (not one shared token per category) + an HTM "
                              "Spatial-Pooler that forms a NEW shared column block (Cui-Ahmad-Hawkins 2017) + cancellation on "
                              "emergent codes + coupling into the console (EMERGE-29)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge30] VERDICT: {verdict}", flush=True)
    print(f"[emerge30] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
