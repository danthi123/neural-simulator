"""gap-#2 POINTER/INDIRECTION EMBEDDED-CLAUSE de-risk (2026-07-22) — step 2/2 of fully retiring the FHRR
exact-inverse algebra.

Research gate: `research/findings/2026-07-22-recursive-slotbinder-research-gate.md` (MOVE 3, "#1 RECOMMENDED
recursion — Pointer/indirection slot"). The biology is POINT-DON'T-COPY (Neural Blackboard Architecture; assembly
projections + multilevel pointers; Frankland-Greene factored registers): a depth-1 embedded clause is stored as its
OWN slot-group `j`; the matrix fact's patient slot binds a dedicated POINTER pool `CLAUSE_j` (indirection, NOT a copy
-> no clause-level superposition, the gap-#2 win preserved one level down). Read = scan the matrix -> recover the
pointer -> FOLLOW it to group `j`'s slots (the same 6-seed-GO neural scan).

The plain pointer-pool read is a filler read exactly like a noun filler (already 6-seed GO), and the follow-hop is
another slot scan -- so NO D3 persistent-slot attractor is needed (the pointer is just another filler pool). This
de-risk confirms that empirically.

6 seeds [42,43,44,100,101,102], CPU/numpy. NO `sim/` edit; additive default-preserving composer change only.

Gates:
  MAIN GO       : embedded-clause roles (inner a/v/p triple) recovered >= 0.90 AND matrix-clause roles (outer a/v)
                  recovered >= 0.90 on >= 5/6 seeds; flat SVO un-regressed.
  ANTI-CHEAT 1  : permuted-pointer (matrix pointers deranged across inner groups) -> embedded-vs-TRUE collapses to
                  ~0 (indirection is load-bearing: the pointer decides which group is read); embedded-vs-PERMUTED
                  stays high (faithful follow of whatever the pointer names).
  ANTI-CHEAT 2  : lesion-the-second-hop (read the matrix patient WITHOUT following) -> returns the pointer CODE/label,
                  NOT the inner-clause content (the follow-hop is the mechanism).
  ANTI-CHEAT 3  : wrong-clause distractor (>= 2 clauses stored) -> each matrix pointer selects the RIGHT group
                  (query_clause returns its OWN inner, never another clause's).
  ANTI-CHEAT 4  : moat -- a matrix fact whose pointer names no stored group -> abstain (None); never-stored cues -> None.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU, deterministic

import numpy as np

from research.runners.slotbinder_composer import SlotBinderComposer
from research.runners.core_sim_composition import Clause

SEEDS = [42, 43, 44, 100, 101, 102]

VOCAB = ["dog", "cat", "bird", "fish", "apple", "river",
         "see", "hear", "chase", "eat", "know", "tell"]

# depth-1 embedded-clause facts: (matrix_agent, matrix_verb, (inner_agent, inner_verb, inner_patient)).
# fact 1 = object-relative shape ("dog sees [that] cat chase bird"); fact 3's inner is a subject-relative content
# ("[the] dog [that] chase fish"); at the composer level both collapse to matrix+inner routed by the pointer.
CLAUSE_FACTS = [
    ("dog",  "see",  ("cat",  "chase", "bird")),
    ("fish", "hear", ("bird", "eat",   "apple")),
    ("cat",  "know", ("dog",  "chase", "fish")),
]
# each embedded-clause patient is passed as a DIFFERENT accepted form (Clause / plain 3-tuple / dict) to prove the
# store contract accepts all three.
def _as_operand(inner, k):
    a, v, p = inner
    if k == 0:
        return Clause(a, v, p)
    if k == 1:
        return (a, v, p)
    return {"agent": a, "action": v, "patient": p}

FLAT_FACTS = [
    ("river", "hear", "dog"),
    ("apple", "see",  "fish"),
]
DANGLING_GROUP = 9   # a pointer id whose group is never stored (moat)


def _build_main(seed):
    c = SlotBinderComposer(seed=seed, vocab=list(VOCAB), max_facts=12)
    for k, (ma, mv, inner) in enumerate(CLAUSE_FACTS):
        assert c.store(ma, mv, _as_operand(inner, k)) is True, "clause store must accept Clause/3-tuple/dict"
    for a, v, n in FLAT_FACTS:
        assert c.store(a, v, n) is True
    return c


def main_seed(seed):
    c = _build_main(seed)
    emb_ok, mat_ok = [], []
    for ma, mv, inner in CLAUSE_FACTS:
        emb_ok.append(c.query_clause(ma, mv) == inner)         # inner (a,v,p) triple recovered by following pointer
        mi = c._match(cue_a=ma, cue_v=mv)                       # matrix roles: read the matrix fact's own slots back
        mat_ok.append(mi is not None and c._read_word(mi, 0) == ma and c._read_word(mi, 1) == mv)
    flat_ok = [c.query_patient(a, v) == n for a, v, n in FLAT_FACTS]
    return {
        "embedded": float(np.mean(emb_ok)),
        "matrix": float(np.mean(mat_ok)),
        "flat": float(np.mean(flat_ok)),
    }


def ac1_permuted_seed(seed):
    """Store the 3 inner clauses as flat facts (groups 0,1,2), then store 3 matrix facts with a DERANGED pointer
    (matrix i points to inner perm[i]). Following the pointer must return inner[perm[i]] (faithful), which != the
    TRUE inner[i] (collapse) -- proving indirection (the pointer, not position) decides which group is read."""
    c = SlotBinderComposer(seed=seed, vocab=list(VOCAB), max_facts=12)
    inners = [inner for _, _, inner in CLAUSE_FACTS]
    for a, v, p in inners:
        c.store(a, v, p)                                       # groups 0,1,2
    matrix_cues = [(ma, mv) for ma, mv, _ in CLAUSE_FACTS]
    perm = [1, 2, 0]                                           # a derangement (no fact keeps its own group)
    for (ma, mv), pg in zip(matrix_cues, perm):
        c._store_matrix_with_pointer(ma, mv, pg)               # groups 3,4,5 -> deranged pointers
    vs_true, vs_perm = [], []
    for i, (ma, mv) in enumerate(matrix_cues):
        got = c.query_clause(ma, mv)
        vs_true.append(got == inners[i])                       # should collapse (~0): deranged pointer
        vs_perm.append(got == inners[perm[i]])                 # should stay high: faithful follow of the pointer
    return float(np.mean(vs_true)), float(np.mean(vs_perm))


def ac2_lesion_seed(c):
    """Reading the matrix patient WITHOUT the follow-hop (query_patient) returns the pointer CODE, not inner content."""
    ok = []
    for ma, mv, inner in CLAUSE_FACTS:
        p = c.query_patient(ma, mv)                            # the raw patient slot (no follow)
        is_pointer = isinstance(p, str) and p.startswith("__CLAUSE")
        not_content = p not in inner                           # not any inner-clause word
        ok.append(bool(is_pointer and not_content))
    return float(np.mean(ok))


def ac3_distractor_seed(c):
    """With >= 2 clauses stored, each matrix pointer selects the RIGHT group -- query_clause returns its OWN inner,
    never another clause's inner (right_group), and the three reads are not a fixed single clause (distinct)."""
    inners = [inner for _, _, inner in CLAUSE_FACTS]
    right, wrong = [], []
    got_all = []
    for i, (ma, mv, inner) in enumerate(CLAUSE_FACTS):
        got = c.query_clause(ma, mv)
        got_all.append(got)
        right.append(got == inner)
        wrong.append(any(got == inners[k] for k in range(len(inners)) if k != i and got is not None))
    distinct = len({g for g in got_all if g is not None}) == len([g for g in got_all if g is not None])
    return float(np.mean(right)), float(np.mean(wrong)), bool(distinct)


def ac4_moat_seed(seed):
    c = SlotBinderComposer(seed=seed, vocab=list(VOCAB), max_facts=12)
    c.store("dog", "see", Clause("cat", "chase", "bird"))      # group 0 inner, group 1 matrix
    # a matrix whose pointer names a group that is never stored -> query_clause must abstain
    c._store_matrix_with_pointer("fish", "hear", DANGLING_GROUP)   # group 2, dangling pointer
    dangling_none = c.query_clause("fish", "hear") is None
    # never-stored cues -> None on both the follow AND the flat read
    novel_clause_none = c.query_clause("river", "tell") is None
    novel_patient_none = c.query_patient("river", "tell") is None
    # a genuine flat fact is NOT a clause -> query_clause returns None (not a confabulated triple)
    real_clause_ok = c.query_clause("dog", "see") == ("cat", "chase", "bird")
    return {
        "dangling_pointer_None": dangling_none,
        "novel_cue_clause_None": novel_clause_none,
        "novel_cue_patient_None": novel_patient_none,
        "real_clause_recovered": real_clause_ok,
    }


def run():
    print(f"gap#2 pointer/indirection embedded-clause de-risk | {len(SEEDS)} seeds | "
          f"{len(CLAUSE_FACTS)} depth-1 clause facts + {len(FLAT_FACTS)} flat facts\n")

    print("--- MAIN GO (embedded inner-triple AND matrix outer-roles recovery) ---")
    rows = []
    for s in SEEDS:
        r = main_seed(s)
        rows.append(r)
        print(f"  seed {s:>3}: embedded={r['embedded']:.3f}  matrix={r['matrix']:.3f}  flat={r['flat']:.3f}")
    mean_emb = float(np.mean([r["embedded"] for r in rows]))
    mean_mat = float(np.mean([r["matrix"] for r in rows]))
    mean_flat = float(np.mean([r["flat"] for r in rows]))
    n_emb_ge = sum(r["embedded"] >= 0.90 for r in rows)
    n_mat_ge = sum(r["matrix"] >= 0.90 for r in rows)
    print(f"  MEAN   : embedded={mean_emb:.3f}  matrix={mean_mat:.3f}  flat={mean_flat:.3f}  "
          f"| seeds>=0.90: embedded={n_emb_ge}/6 matrix={n_mat_ge}/6")

    print("\n--- ANTI-CHEAT 1: permuted-pointer (derangement) ---")
    a1t, a1p = [], []
    for s in SEEDS:
        vt, vp = ac1_permuted_seed(s)
        a1t.append(vt)
        a1p.append(vp)
        print(f"  seed {s:>3}: embedded-vs-TRUE={vt:.3f} (want ~0)  embedded-vs-PERMUTED={vp:.3f} (want high)")
    mean_a1t, mean_a1p = float(np.mean(a1t)), float(np.mean(a1p))
    print(f"  MEAN   : embedded-vs-TRUE={mean_a1t:.3f}  embedded-vs-PERMUTED={mean_a1p:.3f}")

    print("\n--- ANTI-CHEAT 2 + 3: lesion-second-hop + wrong-clause distractor (on the MAIN composer) ---")
    a2, a3r, a3w, a3d = [], [], [], []
    for s in SEEDS:
        c = _build_main(s)
        a2.append(ac2_lesion_seed(c))
        r, w, d = ac3_distractor_seed(c)
        a3r.append(r)
        a3w.append(w)
        a3d.append(d)
        print(f"  seed {s:>3}: lesion_returns_pointer_not_content={a2[-1]:.3f}  "
              f"distractor_right_group={r:.3f}  distractor_wrong_group={w:.3f}  distinct={d}")
    mean_a2 = float(np.mean(a2))
    mean_a3r, mean_a3w = float(np.mean(a3r)), float(np.mean(a3w))
    print(f"  MEAN   : lesion={mean_a2:.3f}  right_group={mean_a3r:.3f}  wrong_group={mean_a3w:.3f}  "
          f"all_distinct={all(a3d)}")

    print("\n--- ANTI-CHEAT 4: moat (dangling pointer + never-stored cue) ---")
    m_rows = []
    for s in SEEDS:
        m = ac4_moat_seed(s)
        m_rows.append(m)
        print(f"  seed {s:>3}: {m}")
    moat_all = all(all(m.values()) for m in m_rows)
    print(f"  moat holds all seeds: {moat_all}")

    # verdicts
    go_main = (n_emb_ge >= 5 and n_mat_ge >= 5 and mean_flat >= 0.90)
    go_ac1 = (mean_a1t <= 0.05 and mean_a1p >= 0.90)
    go_ac2 = (mean_a2 >= 0.90)
    go_ac3 = (mean_a3r >= 0.90 and mean_a3w <= 0.05 and all(a3d))
    go_ac4 = moat_all
    print("\n=== VERDICT ===")
    print(f"  MAIN GO (emb>=0.90 & mat>=0.90 on >=5/6, flat un-regressed): {'GO' if go_main else 'NO'} "
          f"(emb {n_emb_ge}/6, mat {n_mat_ge}/6, flat={mean_flat:.3f})")
    print(f"  ANTI-CHEAT 1 (permuted->0, faithful): {'GO' if go_ac1 else 'NO'} "
          f"(vs_true={mean_a1t:.3f}, vs_perm={mean_a1p:.3f})")
    print(f"  ANTI-CHEAT 2 (lesion returns pointer code): {'GO' if go_ac2 else 'NO'} ({mean_a2:.3f})")
    print(f"  ANTI-CHEAT 3 (right-group select): {'GO' if go_ac3 else 'NO'} "
          f"(right={mean_a3r:.3f}, wrong={mean_a3w:.3f})")
    print(f"  ANTI-CHEAT 4 (moat): {'GO' if go_ac4 else 'NO'}")
    overall = go_main and go_ac1 and go_ac2 and go_ac3 and go_ac4
    print(f"  OVERALL: {'GO' if overall else 'PARTIAL/NEGATIVE'}")


if __name__ == "__main__":
    run()
