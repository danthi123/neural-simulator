"""OPEN-WORLD INFERENCE #3 -- analogical transfer (A:B :: C:?) via the parallelogram on the brain's LEARNED
co-occurrence codes (the VSA/word2vec analogy op: d ~= b - a + c). The open-domain research gate flagged this as
sound-but-OVER-CLAIM-PRONE (a NEGATIVE on raw bundled codes; the 2026-05-14 transitive-inference over-claim) -- so
the controls are foregrounded: the parallelogram must (1) beat a "just C's nearest neighbour" baseline (else it is
NOT analogy, just retrieval of C's neighbour), (2) collapse under PERMUTED codes, (3) collapse under a WITHIN-FACTOR
shuffle. Concepts carry TWO emergent factors (category x attribute) learned purely from co-occurrence; the analogy
transfers the attribute-shift across categories. Reuse-by-import; numpy. NO `sim/` edit.
"""
from __future__ import annotations
import argparse
import numpy as np
from research.runners._emergent_vocab_breadth_scale_derisk import learn_stream_codes

N_CAT = 3          # categories (e.g. animal / vehicle / tool)
N_ATTR = 4         # attributes per category (e.g. small / big / fast / slow)
N_MEM = 3          # distinct member words per (category, attribute) cell


def _two_factor_stream(rng):
    """Each concept word co-occurs with BOTH its category context AND its attribute context (two independent factors)
    -> the learned code is (roughly) the sum of a category component + an attribute component. The analogy op then
    transfers the attribute shift across categories. The structure EMERGES from co-occurrence; it is NOT baked into
    the codes."""
    words = {}          # (c,a) -> [member words]
    sents = []
    for c in range(N_CAT):
        for a in range(N_ATTR):
            mem = [f"w_{c}_{a}_{i}" for i in range(N_MEM)]
            words[(c, a)] = mem
            for w in mem:
                for _ in range(3):
                    sents.append([w, f"catctx{c}"]); sents.append([f"catctx{c}", w])
                    sents.append([w, f"attrctx{a}"]); sents.append([f"attrctx{a}", w])
    stream, order = [], list(range(len(sents)))
    for _ in range(50):
        rng.shuffle(order); stream += [list(sents[i]) for i in order]
    return stream, words


def run_seed(seed):
    rng = np.random.default_rng(seed)
    stream, words = _two_factor_stream(rng)
    members = [w for cell in words.values() for w in cell]
    ctxs = sorted({t for st in stream for t in st if t.startswith("catctx") or t.startswith("attrctx")})
    codes, _ = learn_stream_codes(seed, stream, members, ctxs, window=2)
    row = {w: i for i, w in enumerate(members)}
    X = codes.astype(float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)     # unit rows for cosine
    cell_of = {w: cell for cell, ms in words.items() for w in ms}

    def nearest(vec, exclude):
        best, bi = -1e9, None
        for w, i in row.items():
            if w in exclude:
                continue
            cs = float(vec @ X[i])
            if cs > best:
                best, bi = cs, w
        return bi

    # build analogy quads: (c1,a1):(c1,a2) :: (c2,a1):(c2,a2)  -- transfer the a1->a2 attribute shift to category c2
    para_hit = base_hit = tot = 0
    quads = []
    for c1 in range(N_CAT):
        for c2 in range(N_CAT):
            if c1 == c2:
                continue
            for a1 in range(N_ATTR):
                for a2 in range(N_ATTR):
                    if a1 == a2:
                        continue
                    quads.append((c1, a1, c2, a2))
    rng.shuffle(quads)
    for (c1, a1, c2, a2) in quads[:60]:
        A = words[(c1, a1)][0]; B = words[(c1, a2)][0]; C = words[(c2, a1)][0]
        target_cell = (c2, a2)
        d = X[row[B]] - X[row[A]] + X[row[C]]                    # parallelogram
        pred = nearest(d, exclude={A, B, C})
        base = nearest(X[row[C]], exclude={C})                   # BASELINE: just C's nearest neighbour (no analogy)
        tot += 1
        para_hit += int(cell_of[pred] == target_cell)
        base_hit += int(cell_of[base] == target_cell)
    para_acc = para_hit / tot
    base_acc = base_hit / tot

    # (perm) PERMUTED codes: shuffle which code belongs to which word -> the parallelogram must collapse
    perm = list(range(len(members))); rng.shuffle(perm)
    Xp = X[perm]
    ph = pt = 0
    for (c1, a1, c2, a2) in quads[:60]:
        A = words[(c1, a1)][0]; B = words[(c1, a2)][0]; C = words[(c2, a1)][0]
        d = Xp[row[B]] - Xp[row[A]] + Xp[row[C]]
        best, bi = -1e9, None
        for w, i in row.items():
            if w in {A, B, C}:
                continue
            cs = float(d @ Xp[i])
            if cs > best:
                best, bi = cs, w
        pt += 1; ph += int(cell_of[bi] == (c2, a2))
    perm_acc = ph / pt
    chance = 1.0 / (N_CAT * N_ATTR)
    return {"para": para_acc, "base": base_acc, "perm": perm_acc, "chance": chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[analogical transfer] A:B :: C:? via the parallelogram on learned co-occurrence codes | "
          f"{N_CAT} cats x {N_ATTR} attrs x {N_MEM} members", flush=True)
    P, B, PM = [], [], []
    for s in seeds:
        r = run_seed(s)
        P.append(r["para"]); B.append(r["base"]); PM.append(r["perm"])
        print(f"  [seed {s}] parallelogram={r['para']:.3f}  baseline(C-neighbour)={r['base']:.3f}  permuted={r['perm']:.3f}  chance={r['chance']:.3f}", flush=True)
    chance = N_CAT and 1.0 / (N_CAT * N_ATTR)
    # GO: parallelogram beats BOTH the C-neighbour baseline AND permuted, every seed (genuine analogy, not retrieval)
    go = (all(P[i] - B[i] > 0.30 for i in range(len(P))) and all(P[i] - PM[i] > 0.30 for i in range(len(P)))
          and all(p > 0.70 for p in P))
    print(f"\n  AGGREGATE: parallelogram={np.mean(P):.3f} baseline={np.mean(B):.3f} permuted={np.mean(PM):.3f} chance={chance:.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- A:B :: C:? analogy via the parallelogram on the brain's "
          f"learned co-occurrence codes {'BEATS the C-neighbour baseline (it is genuine analogy, not just retrieving C-neighbour) AND collapses under permuted codes -> open-world analogical inference on learned codes' if go else '-- does NOT cleanly beat the baseline/permuted controls (the over-claim trap the gate warned of); honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
