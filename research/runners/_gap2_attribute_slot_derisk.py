"""gap-#2 ATTRIBUTE-SLOT de-risk (2026-07-22) — the first step of fully retiring the FHRR exact-inverse algebra.

Research gate: `research/findings/2026-07-22-recursive-slotbinder-research-gate.md` (MOVE 3, "#2-difficulty but
DO-FIRST — Attribute slot"). The `SlotBinderComposer` gains a 5th flat `attribute` role so a SINGLE-attribute entity
("big apple") stores + recalls BOTH its noun (patient slot) AND its adjective (attribute slot). NOT recursion; NOT
2-attribute (the FHRR's own ~29% boundary). An un-attributed fact writes a NOATTR filler -> query_attribute -> None
(no confabulated adjective = the moat, by construction).

6 seeds [42,43,44,100,101,102], CPU/numpy. NO `sim/` edit; additive default-preserving composer change only.

Gates:
  MAIN GO       : recover BOTH patient AND attribute for the attributed facts >= 0.90 (joint), flat facts un-regressed.
  ANTI-CHEAT 1  : permuted-attribute (adjectives deranged across facts) -> attribute-read-vs-TRUE collapses to chance
                  (proves the attribute slot is fact-specific, not a fixed/derived default); read-vs-PERMUTED stays
                  high (proves faithful storage of whatever was taught).
  ANTI-CHEAT 2  : moat -- an un-attributed fact's query_attribute is None; a never-stored cue abstains (patient+attr).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU, deterministic

import numpy as np

from research.runners.slotbinder_composer import SlotBinderComposer

SEEDS = [42, 43, 44, 100, 101, 102]

# vocab: agents/patients + verbs + adjectives (+ the 3 internal polarity/noattr fillers appended by the composer)
VOCAB = ["dog", "cat", "bird", "fish", "apple", "river", "north", "south",
         "see", "eat", "chase", "hear", "go",
         "big", "small", "hot", "cold"]

# (agent, verb, adjective, noun) -- 3 attributed facts with DISTINCT adjectives
ATTR_FACTS = [
    ("dog",  "see",   "big",   "apple"),
    ("cat",  "eat",   "small", "fish"),
    ("bird", "chase", "hot",   "river"),
]
# (agent, verb, noun) -- 2 un-attributed facts (must recover patient + read NO adjective)
FLAT_FACTS = [
    ("fish", "hear", "north"),
    ("dog",  "go",   "south"),
]
# a proper derangement of the 3 adjectives (no fact keeps its own) for the permuted anti-cheat
PERM_ADJ = ["small", "hot", "big"]   # aligned to ATTR_FACTS order; index i gets PERM_ADJ[i]


def _build(seed):
    c = SlotBinderComposer(seed=seed, vocab=list(VOCAB), max_facts=8)
    for a, v, adj, n in ATTR_FACTS:
        assert c.store(a, v, (adj, n)) is True, "attributed store must accept a (adj, noun) tuple patient"
    for a, v, n in FLAT_FACTS:
        assert c.store(a, v, n) is True
    return c


def main_seed(seed):
    c = _build(seed)
    pat_ok, attr_ok, joint_ok = [], [], []
    for a, v, adj, n in ATTR_FACTS:
        p = c.query_patient(a, v)
        at = c.query_attribute(a, v)
        pat_ok.append(p == n)
        attr_ok.append(at == adj)
        joint_ok.append(p == n and at == adj)
    # flat facts: patient recovered AND no confabulated adjective
    flat_pat_ok, flat_attr_none = [], []
    for a, v, n in FLAT_FACTS:
        flat_pat_ok.append(c.query_patient(a, v) == n)
        flat_attr_none.append(c.query_attribute(a, v) is None)
    return {
        "patient": float(np.mean(pat_ok)),
        "attribute": float(np.mean(attr_ok)),
        "joint": float(np.mean(joint_ok)),
        "flat_patient": float(np.mean(flat_pat_ok)),
        "flat_attr_none": float(np.mean(flat_attr_none)),
    }


def permuted_seed(seed):
    c = SlotBinderComposer(seed=seed, vocab=list(VOCAB), max_facts=8)
    for (a, v, _adj, n), padj in zip(ATTR_FACTS, PERM_ADJ):
        c.store(a, v, (padj, n))   # taught the PERMUTED adjective
    vs_true, vs_perm = [], []
    for (a, v, tadj, _n), padj in zip(ATTR_FACTS, PERM_ADJ):
        got = c.query_attribute(a, v)
        vs_true.append(got == tadj)   # should collapse (fact-specific, not the original mapping)
        vs_perm.append(got == padj)   # should stay high (faithful storage of what was taught)
    return float(np.mean(vs_true)), float(np.mean(vs_perm))


def moat_seed(seed):
    c = _build(seed)
    # (1) un-attributed fact -> query_attribute None (already checked in main; re-assert cleanly)
    unattr_none = c.query_attribute("fish", "hear") is None
    # (2) never-stored cue -> abstain on BOTH patient and attribute
    novel_pat_none = c.query_patient("cat", "see") is None      # cat's verb is 'eat', not 'see'
    novel_attr_none = c.query_attribute("cat", "see") is None
    novel2_pat_none = c.query_patient("bird", "go") is None     # bird's verb is 'chase'
    novel2_attr_none = c.query_attribute("bird", "go") is None
    return {
        "unattr_query_attribute_is_None": unattr_none,
        "novel_cue_patient_None": novel_pat_none,
        "novel_cue_attribute_None": novel_attr_none,
        "novel_cue2_patient_None": novel2_pat_none,
        "novel_cue2_attribute_None": novel2_attr_none,
    }


def run():
    chance = 1.0 / 3.0  # 3 adjectives
    print(f"gap#2 attribute-slot de-risk | {len(SEEDS)} seeds | {len(ATTR_FACTS)} attributed + "
          f"{len(FLAT_FACTS)} flat facts | attr-chance={chance:.2f}\n")

    print("--- MAIN GO (patient AND attribute recovery for attributed facts) ---")
    main_rows = []
    for s in SEEDS:
        r = main_seed(s)
        main_rows.append(r)
        print(f"  seed {s:>3}: patient={r['patient']:.3f}  attribute={r['attribute']:.3f}  "
              f"JOINT={r['joint']:.3f}  | flat: patient={r['flat_patient']:.3f} attr_None={r['flat_attr_none']:.3f}")
    mean_pat = float(np.mean([r["patient"] for r in main_rows]))
    mean_attr = float(np.mean([r["attribute"] for r in main_rows]))
    mean_joint = float(np.mean([r["joint"] for r in main_rows]))
    mean_flat_pat = float(np.mean([r["flat_patient"] for r in main_rows]))
    mean_flat_none = float(np.mean([r["flat_attr_none"] for r in main_rows]))
    print(f"  MEAN   : patient={mean_pat:.3f}  attribute={mean_attr:.3f}  JOINT={mean_joint:.3f}  "
          f"| flat: patient={mean_flat_pat:.3f} attr_None={mean_flat_none:.3f}")

    print("\n--- ANTI-CHEAT 1: permuted-attribute (derangement) ---")
    at_true, at_perm = [], []
    for s in SEEDS:
        vt, vp = permuted_seed(s)
        at_true.append(vt)
        at_perm.append(vp)
        print(f"  seed {s:>3}: attr-vs-TRUE={vt:.3f} (want ~chance {chance:.2f})  "
              f"attr-vs-PERMUTED={vp:.3f} (want high)")
    mean_true = float(np.mean(at_true))
    mean_perm = float(np.mean(at_perm))
    print(f"  MEAN   : attr-vs-TRUE={mean_true:.3f}  attr-vs-PERMUTED={mean_perm:.3f}")

    print("\n--- ANTI-CHEAT 2: moat ---")
    moat_rows = []
    for s in SEEDS:
        m = moat_seed(s)
        moat_rows.append(m)
        print(f"  seed {s:>3}: {m}")
    moat_all = all(all(m.values()) for m in moat_rows)
    print(f"  moat holds all seeds: {moat_all}")

    # verdicts
    go_main = mean_joint >= 0.90 and mean_flat_pat >= 0.90 and mean_flat_none >= 0.90
    go_ac1 = mean_true <= chance + 1e-9 and mean_perm >= 0.90
    go_ac2 = moat_all
    print("\n=== VERDICT ===")
    print(f"  MAIN GO (joint>=0.90, flat un-regressed): {'GO' if go_main else 'NO'} "
          f"(joint={mean_joint:.3f}, flat_pat={mean_flat_pat:.3f}, flat_none={mean_flat_none:.3f})")
    print(f"  ANTI-CHEAT 1 (permuted->chance, faithful): {'GO' if go_ac1 else 'NO'} "
          f"(vs_true={mean_true:.3f}<= {chance:.2f}, vs_perm={mean_perm:.3f})")
    print(f"  ANTI-CHEAT 2 (moat): {'GO' if go_ac2 else 'NO'}")
    print(f"  OVERALL: {'GO' if (go_main and go_ac1 and go_ac2) else 'PARTIAL/NEGATIVE'}")


if __name__ == "__main__":
    run()
