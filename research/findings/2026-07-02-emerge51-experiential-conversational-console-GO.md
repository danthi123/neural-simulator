# EMERGE-51 — the EXPERIENTIAL CONVERSATIONAL CONSOLE: GO

**Date:** 2026-07-02
**Verdict:** **GO (3-seed, all gates clean, NO `sim/` edit)**
**Runner:** `research/runners/_emerge51_experiential_conversational_console.py`
**Tests:** `tests/test_emerge51_experiential_conversational_console.py` (3 CPU/numpy tests, all PASS)
**Raw:** `research/findings/raw/_emerge51_experiential_conversational_console.json`

## What it demonstrates (the north star: a brain you TALK TO)

The competitive self-organizing pooler arc (EMERGE-38..50) discovered overlapping/multi-level categories from
experience and reasoned over them (inheritance/cancellation/transitivity) fully on-substrate. EMERGE-51 **connects that
emergent semantic substrate to a NATURAL-LANGUAGE console**: the brain OBSERVES members with features in plain
language, the competitive pooler **DISCOVERS** the overlapping categories from that experience (not hand-assigned), and
the user **TEACHES** a class property + a member-specific exception and **ASKS in natural language** — answered by the
on-substrate inheritance/cancellation over the DISCOVERED codes, with the honest no-confab moat.

> "discover categories from experience → talk to the brain about them."

This is a **composition of validated pieces** (NO new mechanism, NO `sim/` edit): EMERGE-38/42 competitive pooler +
EMERGE-42/43 inheritance/cancellation on the spiking bridge + EMERGE-29/31 natural-language console.

## The demo transcript (seed 42, abridged)

```
--- OBSERVE members with features (the competitive pooler DISCOVERS the categories) ---
you> a robin has feathers beak talons crest        brain> ok -- I've seen a robin with feathers beak talons crest.
you> a sparrow has wings feathers beak plume       brain> ok -- I've seen a sparrow with wings feathers beak plume.
...  (9 birds share wings/feathers/beak/talons/plume/crest; 9 fish share fins/scales/gills/tail/stripe/barbel)
--- bind members to their class name ---
you> a robin is a bird                              brain> ok -- a robin is a bird.
...
--- TEACH the class property via a few exemplars + member-specific EXCEPTIONS ---
you> a robin can fly                                brain> ok -- a robin can fly.        (bird exemplar)
you> a sparrow can fly / a eagle can fly / ...      (6 bird exemplars taught the class property)
you> a trout can swim / ...                         (6 fish exemplars)
you> a penguin walks                               brain> ok -- a penguin walks.        (member-specific EXCEPTION)
you> a pike lurks                                  brain> ok -- a pike lurks.           (member-specific EXCEPTION)
--- ASK in natural language (answered by the on-substrate inference over the DISCOVERED codes) ---
you> can a owl fly?                                brain> Yes, an owl can fly.          (INHERIT -- never told)
you> can a wren fly?                               brain> Yes, a wren can fly.          (INHERIT -- never told)
you> can a minnow swim?                            brain> Yes, a minnow can swim.       (INHERIT -- never told)
you> can a gar swim?                               brain> Yes, a gar can swim.          (INHERIT -- never told)
you> can a penguin fly?                            brain> No, a penguin walks.          (CANCEL -- member exception)
you> can a pike swim?                              brain> No, a pike lurks.             (CANCEL -- member exception)
you> can a zzz fly?                                brain> I don't know what a zzz is.   (the no-confab MOAT)
```

`owl`, `wren`, `minnow`, `gar` were **never named in a `can`/exception sentence** — they inherit the class property
ONLY via the shared codon the competitive pooler discovered from their features.

## The de-risk gates (3-seed 42/43/44)

| gate | seed 42 | seed 43 | seed 44 | mean | pass |
|---|---|---|---|---|---|
| **held-out inheritance** (never-taught member inherits via the discovered codon; chance ~0.12) | 1.00 | 1.00 | 1.00 | **1.00** | ≥ 0.80 ✓ |
| **cancellation** (exception member answers ITS specific fact) | 1.00 | 1.00 | 1.00 | **1.00** | == 1.0 ✓ |
| **moat — unknown token abstains** | ✓ | ✓ | ✓ | all | ✓ |
| **moat false-accepts** (never-observed tokens: zzz/qqq/wobble) | 0 | 0 | 0 | **0** | == 0 ✓ |
| **PERMUTED control** (scrambled features → no discoverable categories → inheritance collapses) | 0.00 | 0.00 | 0.00 | **0.00** | inh ≥ perm+0.30 ✓ |

All five gates pass on all three seeds. Held-out inheritance (1.00) is decisively above the permuted control (0.00):
the result rides the **discovered category structure**, not a teaching artifact.

## Mechanism (emergent; no inference engine, no transformer)

- **Observe** `"a X has f1 f2 f3"` → member `X` gets a feature vector over a named feature vocabulary.
- **Discover** — the competitive HTM Spatial Pooler (EMERGE-38: winners potentiate active inputs + depress inactive +
  homeostatic boosting, k-WTA) self-organizes a codon per member; members sharing features converge on **overlapping
  codons = the emergent categories**. Cross-category codon overlap is 0; within-category overlap is reliable.
- **Teach class** `"a <exemplar> can P"` → potentiates the `codon → P` coincidence pool on the spiking bridge (the
  committed `sim/` three-term kernel `fused_htm_permanence_update`). Taught on several exemplars per category so the
  shared columns are broadly potentiated → held-out members **inherit**.
- **Teach exception** `"a <member> P"` → potentiates a **member-identity ensemble → P**, a stronger direct fact.
- **Ask** `"can a X P?"` → prime the member's discovered codon + its identity ensemble, read the graded apical drive;
  the member's own exception (specific fact) wins over the inherited class default (Collins-Quillian cancellation,
  including a saturated-plateau tie); an unknown/never-observed token drives no codon → the moat abstains.

The only host code is the world/keyboard interface: a tiny regex NL front end + presenting features + reading the
answer. The category discovery, the inheritance, the cancellation, and the moat all run on the spiking
`SimulationBridge`.

## Files

- **Created:** `research/runners/_emerge51_experiential_conversational_console.py` (console class + `handle()`/`ask_can()`
  + `--demo` scripted transcript + `--derisk` gates + `--script`/interactive).
- **Created:** `tests/test_emerge51_experiential_conversational_console.py` (demo self-check · 3-seed inference gates ·
  permuted control; all CPU/numpy, offline, PASS).
- **Created:** `research/findings/raw/_emerge51_experiential_conversational_console.json` (de-risk record).

## Honest scope

- **Composition, not new mechanism.** Reuse-by-import of EMERGE-38 (competitive pooler; the learning is a rate-reference
  realized fully-on-substrate at EMERGE-39..41) + EMERGE-42/43 (inheritance/cancellation on the spiking bridge over the
  discovered codons) + EMERGE-29/31 (natural-language console). NO `sim/` edit.
- **Multi-exemplar class teaching** — per EMERGE-42's validated inheritance protocol, the class property is taught via
  several exemplars (6 of the 8 per-category members named in `can` sentences) so the shared discovered columns are
  broadly potentiated. The 2 held-out members per category are **never** named in a `can`/exception sentence and
  inherit only via the shared codon — a genuine generalization test. (A *single*-exemplar class teach also works ad-hoc
  for a small vocabulary, as the `--script` smoke shows, but is per-member fragile; multi-exemplar is the robust,
  validated regime.)
- **Curated demo vocabulary** — a small feature set (birds vs fish, each member a varied 4-of-6 subset of its
  category's 6-feature pool, EMERGE-42 style). Corpus-scale feature discovery and **multi-level taxonomy in natural
  language** (inherit from ANIMAL 2-up AND BIRD 1-up conversationally) are the next follow-ons.
- **Cancellation tie rule** — when a member's exception and its inherited class default both saturate the coincidence
  plateau, the read resolves to the **specific** fact (the correct Collins-Quillian semantics: the member exception
  overrides the class default). This was a genuine mechanism nuance caught by the CI test (a seed-44 pike saturated
  tie), fixed by preferring the member's own exception on a tie — not a test hack.

## Verdict

**GO.** The emergent semantic substrate is now **conversationally queryable**: observe experience → the competitive
pooler discovers the categories → teach a class property + a member exception → ask in natural language and get
inheritance / cancellation / an honest abstention, all on one spiking brain, transformer-free, NO `sim/` edit. This
closes the "discover categories from experience → talk to the brain about them" loop end-to-end.
