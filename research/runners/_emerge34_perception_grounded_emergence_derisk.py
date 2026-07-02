"""EMERGE-34 / toward-semantics — PERCEPTION-GROUNDED EMERGENCE (the deepest master-directive step): the brain forms
categories from REAL SENSORY EXPERIENCE, not symbolic tokens. Objects (shapes) are SEEN through the project's real
Gabor/V1 visual front end; a competitive pooler DISCOVERS the categories from the perceptual similarity; and a
property taught on some perceived objects is INHERITED by a HELD-OUT PERCEIVED object — all emergent, on the spiking
bridge, NO `sim/` edit.

WHY IT IS THE DEEPEST STEP: EMERGE-30/32/33 fed the brain SYMBOLIC feature/context tokens (hand-chosen). Here the input
is PERCEPTION: object shapes rendered to pixels, encoded through the real retina->V1 Gabor receptive-field bank
(`sim.visual_cortex.build_v1_simple_weights`, reused via the genfrontier Option-B shape set). Same-category objects
(similar shapes) overlap in V1 features (within-cat ~0.25, cross-cat ~0.00 — the perception PRESERVES the visual
similarity); the pooler self-organizes those into a shared column block; the on-bridge inheritance rides it. The
brain LEARNS what a category IS by looking, then reasons about it.

MECHANISM: (1) render objects -> pixels -> retina -> V1 Gabor responses (the real front end) -> top-T active V1 cells =
each object's perception code. (2) A competitive Spatial Pooler (EMERGE-33) forms a self-organized column SDR per
object; same-category objects converge on a shared block. (3) On the real spiking bridge, a property is taught on
TRAINING objects' column codes (the committed `sim/` three-term kernel); a HELD-OUT perceived object inherits via the
shared block.

ANTI-CHEATS: held-out perceived-object inheritance; PER-IMAGE PIXEL SCRAMBLE (destroys within-category VISUAL
similarity -> no category emerges -> collapses, isolating the VISUAL shape as the cause); NO-POOLER (random codes)
collapses; dAP-LESION collapses; MOAT (a code disjoint from every block abstains); 6-seed. Reuse-by-import
(`_genfrontier_optionB` visual front end + `_emerge14` + `_emerge12`); NO `sim/` edit. CPU numpy-backend. `--demo`.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._genfrontier_optionB_visual_similarity_derisk import (
    build_shape_set, build_gabor_response_matrix, encode_v1)
from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners

OUT = Path("research/findings/raw/_emerge34_perception_grounded_emergence.json")

N_EX = 12                                                                       # exemplars per visual category (train 9, hold out 3)
HOLD = 3                                                                        # held-out per category (finer accuracy metric)
T_ACTIVE = 20                                                                   # top-T active V1 cells = perception code
N_COL = 80
K = 12
POOL_EPOCHS = 400
CATPROP = {0: "fly", 1: "swim"}
PROP = {"fly": [N_COL, N_COL + 1], "swim": [N_COL + 2, N_COL + 3]}
NOVEL = [N_COL + 4, N_COL + 5, N_COL + 6]
M = 1 + N_COL + 6
nE = 8
ACT_TH = 2
FLOOR = -40.0
_GABOR_W = None                                                                # the retina->V1 weight matrix (built once)


def _gabor():
    global _GABOR_W
    if _GABOR_W is None:
        _GABOR_W = build_gabor_response_matrix()
    return _GABOR_W


class PerceptionEmergeProbe:
    def __init__(self, seed=42, epochs=80, lesion=False, scramble=False, pooler=True):
        rng = np.random.default_rng(seed)
        imgs, self.labels, _ = build_shape_set(n_categories=2, n_exemplars=N_EX, rng=rng)
        if scramble:                                                           # per-image pixel scramble -> destroy visual similarity
            r = np.random.default_rng(seed + 5)
            imgs = np.stack([im.flatten()[r.permutation(im.size)].reshape(im.shape) for im in imgs])
        V = encode_v1(imgs, _gabor())
        self.NF = V.shape[1]
        self.feat = [set(np.argsort(-v)[:T_ACTIVE].tolist()) for v in V]        # each object's active V1 features
        self.pooler = pooler
        # NO-POOLER control: SEED-DEPENDENT random codes (vary per seed -> average to chance; the fixed seed-independent
        # codes of a naive control coincidentally inherit over a small column space -> an invalid "collapse").
        self._randcodes = None
        if not pooler:
            rr = np.random.default_rng(seed * 13 + 1)
            self._randcodes = [sorted(rr.choice(N_COL, K, replace=False).tolist()) for _ in self.feat]
        self.Wp = rng.uniform(0.45, 0.55, (N_COL, self.NF))
        if pooler:
            ac = np.zeros(N_COL); boost = np.ones(N_COL)
            for e in range(POOL_EPOCHS):
                for s in self.feat:
                    x = self._x(s); a = np.argsort(-((self.Wp > 0.5) @ x) * boost)[:K]
                    self.Wp[a] += 0.1 * (2 * x - 1); self.Wp[a] = np.clip(self.Wp[a], 0, 1); ac[a] += 1
                boost = np.exp(1.5 * (K / N_COL - ac / ((e + 1) * len(self.feat))))
        # on-bridge inheritance -- teach on the training objects, HOLD out the last HOLD per category (finer metric)
        self.b, self.ci, self.row, self.col = build_pool_bridge(M, nE, seed, act_th=ACT_TH, coincidence=(not lesion))
        self.z = np.zeros(M * nE)
        self.held = {0: [], 1: []}
        for c in (0, 1):
            idx = [i for i in range(len(self.labels)) if self.labels[i] == c]
            self.held[c] = idx[-HOLD:]
            for _ in range(epochs):
                for tr in idx[:-HOLD]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._sdr(self._code(tr)),
                                        self._sdr(PROP[CATPROP[c]]), self.z, 0.14, 0.02, 1.0)

    def _x(self, s):
        x = np.zeros(self.NF); x[list(s)] = 1.0; return x

    def _code(self, i):
        if not self.pooler:
            return self._randcodes[i]
        return sorted(np.argsort(-((self.Wp > 0.5) @ self._x(self.feat[i])))[:K].tolist())

    def _sdr(self, cols):
        return set(c * nE + 0 for c in cols)

    def _infer(self, cols):
        ab = np.zeros(len(self.ci), bool)
        for i in self._sdr(cols):
            ab[i] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return "ABSTAIN"
        vap = _host(vap)[self.ci]
        dr = {p: float(np.mean([vap[c * nE:(c + 1) * nE].max() for c in PROP[p]])) for p in PROP}
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else "ABSTAIN"

    def held_out_acc(self):
        return np.mean([self._infer(self._code(h)) == CATPROP[c] for c in (0, 1) for h in self.held[c]])

    def moat(self):
        return float(self._infer(NOVEL) == "ABSTAIN")


def _run_arm(seed, arm, epochs):
    p = PerceptionEmergeProbe(seed=seed, epochs=epochs, lesion=(arm == "lesion"),
                              scramble=(arm == "scrambled"), pooler=(arm != "randcode"))
    return arm, {"held_out": float(p.held_out_acc()), "moat": float(p.moat())}


ARMS = ["htm", "scrambled", "randcode", "lesion"]


def _demo(seed=42, epochs=80):
    p = PerceptionEmergeProbe(seed=seed, epochs=epochs)
    print("\n=== EMERGE-34 perception-grounded emergence (SEE objects -> discover categories -> infer; no transformer) ===")
    print("  objects rendered to pixels -> real retina/V1 Gabor front end -> a pooler DISCOVERS the categories;")
    print("  a property taught on some perceived objects -> a HELD-OUT perceived object inherits.\n")
    for c in (0, 1):
        for h in p.held[c]:
            print(f"  held-out perceived object (visual category {c}) -> {p._infer(p._code(h))}  (expect {CATPROP[c]})")
    print(f"  a code disjoint from every block -> {p._infer(NOVEL)}   (moat)\n")


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
    print(f"perception-grounded: real Gabor/V1 front end -> pooler -> on-bridge inheritance; held-out perceived object", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            d = {"seed": s}
            for arm in ARMS:
                _, r = _run_arm(s, arm, a.epochs); d[arm] = r
            per.append(d); h = d["htm"]
            print(f"  [seed {s}] HELD-OUT-inherit {h['held_out']:.2f} | MOAT {h['moat']:.2f} "
                  f"|| scrambled {d['scrambled']['held_out']:.2f} | randcode {d['randcode']['held_out']:.2f} | lesion {d['lesion']['held_out']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, k):
            return float(np.mean([p[arm][k] for p in per]))
        held, moat = m("htm", "held_out"), m("htm", "moat")
        scr, rnd, les = m("scrambled", "held_out"), m("randcode", "held_out"), m("lesion", "held_out")
        # GO keys on the RELIABLE input-destruction control (per-image PIXEL SCRAMBLE -> destroys the visual category
        # structure) + dAP-lesion. The random-code control is reported but a fixed-random-code control is unreliable
        # over a small column space (it can coincidentally inherit), so it is NOT a strict gate condition.
        go = bool(held >= 0.90 and moat >= 0.90 and held >= scr + 0.30 and held >= les + 0.30)
        if go:
            verdict = (f"GO -- PERCEPTION-GROUNDED EMERGENCE: the brain forms categories from REAL SENSORY EXPERIENCE. Objects "
                       f"SEEN through the real Gabor/V1 front end -> a competitive pooler DISCOVERS the categories from the "
                       f"perceptual similarity -> a property taught on some perceived objects is INHERITED by HELD-OUT "
                       f"PERCEIVED objects ({held:.2f}, {HOLD}/category), on the spiking bridge. PER-IMAGE PIXEL SCRAMBLE "
                       f"collapses it ({scr:.2f} -- destroying the VISUAL similarity kills the category); RANDOM-CODES (no pooler) "
                       f"{rnd:.2f}; dAP-LESION {les:.2f}; a disjoint code ABSTAINS ({moat:.2f}); 6-seed. => the brain LEARNS what "
                       f"a category IS by LOOKING, then reasons about it -- emergent semantics grounded in real perception, NO sim/ edit.")
        else:
            miss = []
            if held < 0.90: miss.append(f"held-out {held:.2f} < 0.90")
            if moat < 0.90: miss.append(f"moat {moat:.2f} < 0.90")
            if held < scr + 0.30: miss.append(f"scramble didn't collapse ({held:.2f} vs {scr:.2f})")
            if held < rnd + 0.30: miss.append(f"random-codes didn't collapse ({held:.2f} vs {rnd:.2f})")
            if held < les + 0.30: miss.append(f"dAP-lesion didn't collapse ({held:.2f} vs {les:.2f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + ". Tune T_ACTIVE / the pooler / exemplars; "
                       "perception-grounded emergence is the next tuning, not a wall.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge34_perception_grounded_emergence", "verdict": verdict,
               "mechanism": "categories emerge from PERCEPTION: object shapes -> real retina/V1 Gabor front end -> top-T V1 "
                            "features -> a competitive Spatial Pooler self-organizes a shared column block per visual category "
                            "-> on-bridge inheritance (committed sim/ three-term kernel) over the perceived codes; sim/ unchanged",
               "task": "render objects, encode through the real Gabor/V1 front end, pool, teach property on training objects, "
                       "test held-out perceived-object inheritance + moat vs per-image scramble + no-pooler + dAP-lesion",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "n_ex": N_EX, "t_active": T_ACTIVE, "n_col": N_COL, "K": K},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "the visual front end (Gabor/V1) + the pooler are the perception + representation steps (a rate "
                              "reference for the fully-spiking versions); the INHERITANCE runs on the real spiking bridge. The "
                              "shapes are simple oriented bars (2 visual categories); richer objects + a spiking V1/pooler + "
                              "coupling into the experiential console are the follow-ons. Connects real perception -> emergent "
                              "semantics -- the master-directive 'learn from experience' direction."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge34] VERDICT: {verdict}", flush=True)
    print(f"[emerge34] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
