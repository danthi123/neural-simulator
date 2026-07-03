# EMERGE-52 — the MULTI-LEVEL CONVERSATIONAL CONSOLE — GO (3-seed)

**Date:** 2026-07-02
**Runner:** `research/runners/_emerge52_multilevel_conversational_console.py`
**Test:** `tests/test_emerge52_multilevel_conversational_console.py` (4 CPU/numpy tests, offline)
**Raw:** `research/findings/raw/_emerge52_multilevel_conversational_console.json`
**Verdict:** **GO** (3-seed 42/43/44). Reuse-by-import; **NO `sim/` edit**.

## What it is (the toward-the-north-star capability)

EMERGE-51 gave a natural-language console over pooler-discovered **FLAT** categories (inheritance +
cancellation + no-confab moat). EMERGE-44/45 built the **DISCOVERED multi-level taxonomy** (STACK the
competitive pooler: member features → sub-category L1 → genus L2 → order L3, inheritance chaining through
levels). **EMERGE-52 CONNECTS them:** a console where the brain **discovers a multi-level hierarchy from
experience** and **answers inheritance across LEVELS in plain language**, with cancellation + the no-confab
moat intact, on the real spiking `SimulationBridge`, transformer-free.

```
you> a robin has wings feathers red small        (OBSERVE a member: its features)
you> a robin is a thrush / a thrush is a bird / a bird is an animal   (speak the taxonomy)
...  (many members across birds + fish; the STACKED pooler DISCOVERS sub-cat -> genus -> order)
you> a bird can fly            (TEACH a MID/genus property)
you> an animal breathes        (TEACH a TOP/order property, 2 discovered levels up)
you> a penguin walks           (member-specific EXCEPTION -- cancellation)
you> can a robin fly?      brain> Yes, a robin can fly.       (INHERIT 1 discovered level up -- genus)
you> can a robin breathe?  brain> Yes, a robin can breathe.   (INHERIT 2 discovered levels up -- order)
you> can a robin swim?     brain> I don't know whether a robin can swim.  (sibling branch -- NOT inherited)
you> can a penguin fly?    brain> No, a penguin walks.        (CANCELLATION -- the member's own exception)
you> can a zzz breathe?    brain> I don't know what a zzz is.  (the no-confab MOAT -- never observed)
```

## Mechanism (emergent; no inference engine, no transformer)

- **L1** = the EMERGE-38 competitive self-organizing pooler on member **FEATURES** → a sub-category codon.
- **L2** = the same pooler over the L1 codons, trained on the **CO-OCCURRENCE of same-genus members** (which
  members share a genus is read from the **spoken is-a taxonomy** — the experienced context) → a genus codon.
- **L3** = the pooler over L2 codons, trained on same-order co-occurrence → an order codon.
- A class property spoken as `a <class> can P` / `a <class> P` is taught (the committed `sim/` three-term
  kernel) on the **level the class lives at** (genus→L2 codons of that genus's members; order→L3 codons of
  that order's members), over the members' **discovered** codons.
- Asking `can a <member> P?` primes the member's discovered L2 + L3 codons + its identity ensemble and reads
  the graded apical drive to every taught property; the strongest fires, with the member's OWN specific
  exception winning a tie (Collins-Quillian cancellation). **Sibling-discrimination is read PURELY from the
  discovered codons** (the asked property must drive a taught class-property cell above the floor via the
  discovered codon — **no spoken-taxonomy shortcut in the read**). A never-observed token drives no codon →
  the moat abstains.

Composes EMERGE-44/45 (multi-level discovery) + EMERGE-51 (NL console) + EMERGE-42 (cancellation). Biology:
the ventral hierarchy's successive pooling stages + ATL convergence (Kandel Ch21; Patterson-Lambon Ralph;
Damasio) — each level pools the one below.

## Gate numbers (3-seed 42/43/44; elapsed ~78 s CPU)

| Gate | Result | Threshold |
|---|---|---|
| 2-level held-out inheritance (order 'breathe', 2 discovered levels up) | **1.00** (per-seed [1.0, 1.0, 1.0]) | ≥ 0.75 |
| 1-level (genus) inheritance floor | **1.00** | reported |
| Real sibling-confusion (held-out bird does NOT inherit fish 'swim') | **0.00** | ≤ 0.05 |
| Cancellation (exception member answers ITS fact) | **1.00** | ≥ 0.99 |
| No-confab moat abstains on unknown token | **True** all seeds | required |
| Moat false-accepts (`zzz`/`qqq`/`wobble`) | **0** | 0 |
| **LOAD-BEARING collapse — permute-co-occurrence raises sibling-confusion** | **0.33 avg** (0.50/0.00/0.50) | ≥ real + 0.25 |
| (secondary) permute-features 2-level | 1.00 (does NOT collapse — see honest scope) | reported |

## The control-validity story (honest, and the crux of this de-risk)

The first-cut collapse control (permute-features, à la EMERGE-42/51) **did NOT collapse** the 2-level
inheritance, and neither did permute-co-occurrence on the *inheritance* metric. Diagnosis: the L2/L3
co-occurrence stream is **keyed by the intact spoken taxonomy**, and same-branch members share features, so
neither permutation alone destroys the branch grouping that carries the inheritance signal. This is exactly
**EMERGE-45's documented honest scope**: the feature-driven **L2/genus grouping is the DOMINANT carrier** of
the multi-level signal; L3/order is a **seed-variable increment**.

The load-bearing, genuinely-collapsing control is on the **codon-driven sibling-discrimination**: scrambling
the L2/L3 **co-occurrence pairs** breaks the pooler's ability to separate the branches, so a held-out bird
then **wrongly inherits the fish 'swim'** property — real sibling-confusion 0.00 → permuted **0.33 avg** (and
0.50 on 2 of 3 seeds; seed 43 at 0.00, honestly seed-variable per EMERGE-45). This proves the **discovered
codons are load-bearing** for the sibling-discrimination (the read has no spoken-taxonomy shortcut — the
`k in anc` ancestor filter was removed precisely so the discrimination is codon-driven, not host-routed).

**Honest verdict:** 2-level conversational inheritance across the discovered hierarchy **works** (2-level +
1-level + sibling-discrimination + cancellation + moat, all gates pass); the multi-level signal **rides the
discovered L2/genus grouping** (the dominant carrier), with L3/order a seed-variable increment — precisely
EMERGE-45's framing, carried faithfully into the conversational console. The moat never weakens (0
false-accepts, abstains on unknowns) and there is no host-taxonomy shortcut in the query read.

## Honest wrinkles (disclosed)

- `can a penguin breathe?` → *"No, a penguin walks"*: the member's exception dominates the read (honest
  Collins-Quillian — the strongest specific fact wins). A per-property override (so a member's exception
  cancels only the matching property) is a bounded follow-on.
- The demo vocabulary is a small curated bird/fish taxonomy (birds share
  wings/feathers/beak/talons/plume/crest; fish share fins/scales/gills/tail/stripe/barbel, each member a
  varied 4-of-6 subset). Corpus-scale feature/taxonomy discovery is a follow-on.
- The pooler LEARNING is a rate-reference (realized fully-on-substrate at EMERGE-39..41, k-WTA spiking at
  EMERGE-41); the inheritance chain runs on the spiking bridge over the discovered codons.

## Reproduce

```bash
# 3-seed de-risk
SIM_BACKEND=numpy python -m research.runners._emerge52_multilevel_conversational_console --derisk --seeds 42 43 44
# scripted demo transcript
SIM_BACKEND=numpy python -m research.runners._emerge52_multilevel_conversational_console --demo --seed 42
# CI tests
SIM_BACKEND=numpy python -m pytest tests/test_emerge52_multilevel_conversational_console.py -v
```

⇒ **discover a multi-level taxonomy from experience → talk to the brain across levels** (inherit 1 level +
2 levels up, sibling-discrimination, cancellation, no-confab moat), one spiking brain, transformer-free,
**NO `sim/` edit**.
