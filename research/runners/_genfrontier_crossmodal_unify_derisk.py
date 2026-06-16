"""Generalization frontier cheap-first — does cross-modal Hebbian UNIFICATION transfer the conversation cortex's
generalization to the PERCEPTION channel, and is Option B (similarity-structured perception input) a prerequisite?

Scoping: `research/findings/2026-06-16-generalization-frontier-scoping.md` §4 (Option A, the top-ranked move).
The conversation cortex already generalizes across SIMILAR concepts (the PPMI stream cortex: held-out 0.86-0.91,
on the spiking substrate). The perception channel does NOT (flat-distinct orthogonal object codes). Option A:
co-activate perception(X) + the conversation cortex(word "X") so Hebbian plasticity converges them onto ONE concept
code -> perception inherits the generalizing codes. THE LOAD-BEARING QUESTION: does that convergence TRANSFER the
category-similarity structure to perception so a HELD-OUT (never-converged) concept's perception code lands in its
correct semantic CATEGORY -- and does it need the perception input to itself carry shared-feature similarity (B)?

This isolates the convergence+transfer MECHANISM in numpy (minutes, NO sim/ edit). The word codes here are
category-structured (a faithful stand-in for the validated PPMI codes -- which provably carry category structure,
`_pearson_vs_Strue ~ +0.52`, gen 0.86-0.91; cited `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md`).
The MEASURED VARIABLE is the PERCEPTION input: FLAT-DISTINCT (the current nav regime) vs SIMILARITY-STRUCTURED
(Option B, its own INDEPENDENT category basis -- no leakage of the word codes). The convergence map is learned on a
TRAIN split; the transfer is tested on HELD-OUT concepts. Anti-cheats: flat-baseline, no-leakage, permuted-pairing,
the no-confab moat.

GATE (3 seeds 42/43/44):
  GO       : STRUCTURED-perception held-out same-category margin >= 0.15 AND held-out lands in the right CATEGORY
             >> chance (1/n_cat) AND the FLAT-distinct arm is ~chance (localizes B as the prerequisite) AND the
             permuted control collapses AND the no-confab moat abstains on a novel concept.
  PARTIAL  : structured transfers weakly (margin 0.05-0.15).
  NEGATIVE : even structured-perception convergence does not transfer (margin ~0) -> convergence alone insufficient.

  SIM_BACKEND=numpy python -m research.runners._genfrontier_crossmodal_unify_derisk
No `sim/` edit; pure numpy.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

N_CAT = 4
N_PER_CAT = 4
F = N_CAT * N_PER_CAT          # 16 concepts
D = 64                         # code dimension
CAT_FRAC = 0.7                 # fraction of a code that is the shared CATEGORY component (vs the unique component)
RIDGE = 1e-2


def _orthonormal(n, d, rng):
    """n orthonormal-ish d-dim basis vectors (n <= d): QR of a random gaussian."""
    q, _ = np.linalg.qr(rng.standard_normal((d, max(n, d))))
    return q[:, :n].T                                   # (n, d)


def _structured_codes(cat_ids, seed, frac=CAT_FRAC):
    """Category-structured codes: code[i] = normalize(frac*cat_basis[cat[i]] + (1-frac)*unique[i]). Same-category
    concepts share the category component -> within-cat cosine high, between-cat low (the PPMI / Option-B structure).
    The basis is seed-specific so the WORD basis and the PERCEPTION basis are INDEPENDENT (no leakage)."""
    rng = np.random.default_rng(seed)
    cat_basis = _orthonormal(N_CAT, D, rng)             # (N_CAT, D) -- the shared category directions
    uniq = rng.standard_normal((F, D))
    uniq /= np.linalg.norm(uniq, axis=1, keepdims=True)
    codes = frac * cat_basis[np.asarray(cat_ids)] + (1.0 - frac) * uniq
    return codes / np.linalg.norm(codes, axis=1, keepdims=True)


def _flat_codes(seed):
    """Flat-distinct orthogonal codes (the current nav perception regime): between-cos ~ 0, NO category structure."""
    rng = np.random.default_rng(seed)
    return _orthonormal(F, D, rng)                       # (F, D), orthonormal -> between-cos ~ 0


def _fit_convergence(perc_train, word_train):
    """Ridge map W: perception_code -> word(concept) code, fit on TRAIN concepts only (the Hebbian/regression
    convergence in its simplest linear form). W = (Y^T X)(X^T X + lambda I)^-1, maps a D_perc vector to D_word."""
    X, Y = perc_train, word_train
    return (Y.T @ X) @ np.linalg.inv(X.T @ X + RIDGE * np.eye(D))


def _cos(a, B):
    a = a / (np.linalg.norm(a) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B @ a


def _heldout_transfer(W, perc, word, cat_ids, held_out):
    """For each HELD-OUT concept: map its perception code through W, measure (a) the same-category margin (mean cos to
    same-cat word codes minus mean cos to other-cat) -- the GENERALIZATION signature -- and (b) whether it lands in
    its correct CATEGORY (argmax word code's category == its own)."""
    margins, cat_hits, own_hits = [], [], []
    for j in held_out:
        pred = W @ perc[j]
        sims = _cos(pred, word)
        same = [k for k in range(F) if cat_ids[k] == cat_ids[j] and k != j]
        other = [k for k in range(F) if cat_ids[k] != cat_ids[j]]
        margins.append(float(np.mean(sims[same]) - np.mean(sims[other])))
        cat_hits.append(int(cat_ids[int(np.argmax(sims))] == cat_ids[j]))
        own_hits.append(int(int(np.argmax(sims)) == j))
    return float(np.mean(margins)), float(np.mean(cat_hits)), float(np.mean(own_hits))


def run_seed(seed):
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)
    # WORD codes (the validated-PPMI stand-in): category-structured.
    word = _structured_codes(cat_ids, seed * 17 + 1)
    # leakage-free TRAIN/HELD-OUT split: hold out 1 concept per category (so each held-out has same-cat train peers).
    rng = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng.choice(np.where(cat_ids == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: train and held-out overlap"

    out = {"seed": seed}
    for arm, perc in (("flat", _flat_codes(seed * 19 + 3)),
                      ("structured", _structured_codes(cat_ids, seed * 23 + 7))):
        # ANTI-CHEAT (no-leakage): the perception category basis is seed-INDEPENDENT of the word basis (different
        # seeds) -> any transferred category similarity is LEARNED by W, not pre-shared between the code sets.
        between = float(np.mean([abs(np.dot(perc[a], perc[b])) for a in range(F) for b in range(F)
                                 if a != b and cat_ids[a] != cat_ids[b]]))
        W = _fit_convergence(perc[train], word[train])
        margin, cat_hit, own_hit = _heldout_transfer(W, perc, word, cat_ids, held_out)
        # ANTI-CHEAT (category-DERANGEMENT permuted control): give W a CONSISTENT but WRONG category mapping -- pair
        # each train concept's perception code with a word code from a DERANGED category (no category maps to itself).
        # If the transfer is the LEARNED perc-cat<->word-cat correspondence, held-out must now land in the WRONG
        # category (cat-acc -> chance, margin -> negative). (A within-set random shuffle leaks: with few concepts a
        # random permutation preserves some category alignment by chance -- the derangement is the clean control.)
        derange = (np.arange(N_CAT) + 1) % N_CAT                    # 0->1->2->3->0: a fixed derangement of categories
        train_by_cat = {c: [t for t in train if cat_ids[t] == c] for c in range(N_CAT)}
        Y_perm = np.zeros((len(train), D))
        for idx, t in enumerate(train):
            c = int(cat_ids[t]); k = train_by_cat[c].index(t)
            donor = train_by_cat[int(derange[c])][k % len(train_by_cat[int(derange[c])])]
            Y_perm[idx] = word[donor]                              # paired with a WRONG-category word code
        Wp = _fit_convergence(perc[train], Y_perm)
        pmargin, _, _ = _heldout_transfer(Wp, perc, word, cat_ids, held_out)
        out[arm] = {"heldout_margin": margin, "heldout_cat_acc": cat_hit, "heldout_own_acc": own_hit,
                    "permuted_margin": pmargin, "between_cat_cos": between}

    # ANTI-CHEAT (no-confab MOAT): a NOVEL concept (random perception code, no category) mapped through the structured
    # W -> its best word-code cosine must be BELOW the familiarity of a real held-out concept (abstain, not confabulate).
    perc_s = _structured_codes(cat_ids, seed * 23 + 7)
    W_s = _fit_convergence(perc_s[train], word[train])
    novel = rng.standard_normal(D); novel /= np.linalg.norm(novel)
    novel_fam = float(np.max(_cos(W_s @ novel, word)))
    heldout_fam = float(np.mean([np.max(_cos(W_s @ perc_s[j], word)) for j in held_out]))
    moat_ok = bool(heldout_fam > novel_fam + 0.10)       # the familiar (held-out) concept is clearly more familiar
    out["moat"] = {"heldout_familiarity": heldout_fam, "novel_familiarity": novel_fam, "moat_ok": moat_ok}

    chance_cat = 1.0 / N_CAT
    s, fl = out["structured"], out["flat"]
    print(f"  [seed {seed}] STRUCTURED held-out margin {s['heldout_margin']:+.3f} cat-acc {s['heldout_cat_acc']:.2f} "
          f"(chance {chance_cat:.2f}) own-acc {s['heldout_own_acc']:.2f} permuted {s['permuted_margin']:+.3f} | "
          f"FLAT margin {fl['heldout_margin']:+.3f} cat-acc {fl['heldout_cat_acc']:.2f} | moat {'OK' if moat_ok else 'BREACH'} "
          f"(fam {out['moat']['heldout_familiarity']:.2f} vs novel {out['moat']['novel_familiarity']:.2f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="research/findings/raw/_genfrontier_crossmodal_unify.json")
    args = ap.parse_args()
    print("[genfrontier] does cross-modal Hebbian convergence TRANSFER the word cortex's category-generalization to "
          "perception? (flat-distinct vs similarity-structured perception input)", flush=True)
    rows = [run_seed(s) for s in args.seeds]
    chance_cat = 1.0 / N_CAT

    def m(arm, k):
        return float(np.mean([r[arm][k] for r in rows]))
    s_margin, s_cat, s_perm = m("structured", "heldout_margin"), m("structured", "heldout_cat_acc"), m("structured", "permuted_margin")
    f_margin, f_cat = m("flat", "heldout_margin"), m("flat", "heldout_cat_acc")
    moat_all = all(r["moat"]["moat_ok"] for r in rows)

    go = (all(r["structured"]["heldout_margin"] >= 0.15 for r in rows)
          and all(r["structured"]["heldout_cat_acc"] >= 0.5 for r in rows)
          and s_perm < 0.05 and f_margin < 0.05 and moat_all)
    partial = (s_margin >= 0.05 and s_margin > f_margin + 0.05 and moat_all)
    verdict = "GO" if go else ("PARTIAL" if partial else "NEGATIVE")

    print(f"\n{'='*100}\n  MEAN ({len(rows)} seeds): STRUCTURED held-out margin {s_margin:+.3f} cat-acc {s_cat:.2f} "
          f"(chance {chance_cat:.2f}) permuted {s_perm:+.3f} | FLAT margin {f_margin:+.3f} cat-acc {f_cat:.2f} | "
          f"moat {'INTACT' if moat_all else 'BREACH'}  ==> {verdict}\n{'='*100}", flush=True)
    if verdict == "GO":
        print(f"  GO: cross-modal convergence TRANSFERS the word cortex's category-generalization to perception when "
              f"the perception input carries shared-feature similarity (Option B) -- held-out (never-converged) "
              f"concepts land in their correct semantic category ({s_cat:.0%} >> chance {chance_cat:.0%}, margin "
              f"{s_margin:+.3f}); the FLAT-distinct arm is ~chance ({f_cat:.0%}, margin {f_margin:+.3f}) -> Option B "
              f"(similarity-structured perception) is the PREREQUISITE, and the permuted control collapses "
              f"({s_perm:+.3f}) + the no-confab moat survives. ==> greenlight the GPU build: co-activate perception + "
              f"conversation cortex on the merged bridge, with a similarity-structured perception front end. Dendritic "
              f"NOT required.", flush=True)
    elif verdict == "PARTIAL":
        print(f"  PARTIAL: structured transfers weakly ({s_margin:+.3f}) above flat ({f_margin:+.3f}) -- the "
              f"convergence carries some category structure but not cleanly; tune the structure fraction / map.", flush=True)
    else:
        print(f"  NEGATIVE: even similarity-structured-perception convergence does not transfer category structure "
              f"(margin {s_margin:+.3f}) -> linear cross-modal convergence alone is insufficient; the transfer needs a "
              f"richer (e.g. shared-assembly / nonlinear) convergence. Honest negative that localizes the next step.", flush=True)
    with open(args.out, "w") as f:
        json.dump({"verdict": verdict, "structured_margin": s_margin, "structured_cat_acc": s_cat,
                   "flat_margin": f_margin, "permuted_margin": s_perm, "moat_intact": moat_all,
                   "chance_cat": chance_cat, "per_seed": rows}, f, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
