---
type: finding
status: live
date: 2026-06-23
mechanism: generative-replay
---

# Generative-frontier (b2) — the BRAIN'S OWN generative replay INVENTS novel-but-plausible propositions: GO (2026-06-23)

**The cheapest-first de-risk of "can the brain INVENT content" (tier G2) — the FIRST brain-mechanism
novel-composition > 0, beating the measured 0.0 retrieval baseline.** Per the generative-frontier scoping
(`2026-06-23-generative-frontier-scoping.md`, Path (b) option b2 — the most "brain's own" route to G2 novel
propositions). The owner wants genuine generation that is THE BRAIN'S (not the LLM's); this probes whether the
brain can recombine what it learned into never-told but graph-supported propositions. Reuse-by-import, **NO
`sim/` edit**, CPU (`SIM_BACKEND=numpy`).

Runner: `research/runners/_genfrontier_b2_generative_replay_derisk.py`
Raw: `research/findings/raw/_genfrontier_b2_generative_replay_derisk.json`

---

## Result — GO (6 seeds, D=128)

| measurement | result | bar / baseline |
|---|---|---|
| **(1) novel-composition score** | **0.752 mean** (0.756/0.800/0.578/0.822/0.844/0.711; min **26** distinct novel proposed) | vs measured **0.0** retrieval baseline (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) |
| **(2) plausibility ADVANTAGE** (replay vs random recombination) | replay plausible-frac **0.328** vs random **0.0195** → **17.0× mean** (min **14.9×**) | ≥ 3× ratio, all seeds ✓ |
| **(3) SHUFFLED-GRAPH control** | shuffled-graph TRUE plausible-frac **0.018** (≈ random floor 0.020, vs replay 0.328) → collapses ≤ 0.5× replay, all seeds ✓ | the learned structure is load-bearing, not a string/template artifact |
| **(4) PROPOSER-MOAT** | **0** proposal→known-fact leaks + **0** explicitly-negated facts re-proposed, all 6 seeds | the load-bearing b2 moat ✓ |
| (4-sanity) composer baseline abstention floor | 0.983 mean / **0.95** min | the documented RF code-fidelity tail (NOT a b2 gate) |
| **LESION (ablate plausibility gate)** | gate ON → **100%** of accepted truly-plausible; gate OFF → 278 accepted, **only 12%** plausible (nonsense floods) | the gate is causally responsible for sensibility |

**VERDICT: GO.** The brain proposes novel-but-plausible propositions far ABOVE chance (17× the random floor),
the learned graph is load-bearing (shuffled collapses to the random floor), proposals are honestly flagged
plausible-not-known (the no-confab moat is preserved), and the plausibility gate is causally responsible
(lesion floods nonsense). **This is the FIRST time the project scores > 0 on novel-COMPOSITION from a BRAIN
mechanism** — the direct refutation of the measured 0.0 retrieval ceiling.

Example novel-but-plausible propositions the brain INVENTED (never told, but graph-supported): `duck eat soup`,
`cat sleep house`, `dog run road`, `bird sing white`, `mouse eat cake`, `frog jump green`, `sister look eye`.

---

## What was built (the mechanism)

A **hippocampal generative-replay proposer** (catalog G.09 constructive imagination; Stoianov/Maisto/Pezzulo
2022; Barry/Love Nat Hum Behav 2023 — "generative replay resamples FICTIVE sequences INCLUDING never-experienced
recombinations") that RESAMPLES role-filler bindings from the **learned association graph** to PROPOSE novel SVO
triples, GATED by (i) graph-plausibility and (ii) non-contradiction. Reuse-by-import:

- **The learned graph / plausibility signal** = the project's PPMI co-occurrence cortex over REAL TinyStories
  (`option_c_real_cooccurrence_derisk.build_real_cooccurrence` + the 8×8 a-priori taxonomy). A symmetric PPMI
  word-relatedness matrix `P[i,j]` is the brain's learned "how related are these two words" signal — the
  generative model's likelihood the replay samples from.
- **The known-fact store + the no-confab moat** = the RF phasor composer (`rf_phasor_composer.RFPhasorComposer`:
  `store(agent, action, patient, polarity)`, `query_patient`, `ask_yes_no`). The proposer reads it (and must
  not contradict it).
- **The proposer** (`GenerativeReplayProposer`): pick a seed agent, SAMPLE an action weighted by its
  graph-relatedness (PPMI) to the agent, SAMPLE a patient weighted by its relatedness to {agent, action} — so
  replay is BIASED toward graph-plausible recombinations (it samples from the learned likelihood, not
  uniformly). Accept a sample iff NOVEL (never stored, either polarity), PLAUSIBLE (gate i:
  selectional-preference — agent~action AND action~patient graph-related), and NON-CONTRADICTORY (gate ii: the
  brain wasn't told it is FALSE — `ask_yes_no != "no"`).

**Plausibility = selectional preference**, not a single-valued cue→patient map. The verb selects its arguments
(agent plausibly DOES the action; action plausibly TAKES the patient). An agent doing MULTIPLE plausible things
("dog plays ball" AND "dog plays toy") is **not** a contradiction — assuming single-valued cue→patient was the
first design's bug (it forbade the very recombinations that constitute generation; see "Two design iterations"
below).

---

## The 4 measurements + anti-cheats (per the scoping §5)

1. **Novel-composition score** = distinct novel-plausible triples proposed / the discoverable
   novel-plausible universe (≈45 triples at the tested scale). Mean 0.752 (all seeds > 0) vs the measured 0.0
   retrieval baseline. **The headline: > 0 from a brain mechanism.**
2. **Random-recombination baseline** = uniform vocab triples (no graph bias). Plausible-frac 0.0195 — the chance
   rate. The brain's biased replay beats it **17×** (the scale-robust RATIO; conjunctive plausibility makes both
   absolute fractions small, so the ratio — not an absolute margin — is the honest advantage signal).
3. **Shuffled-graph control** (the load-bearing anti-cheat): permute the off-diagonal PPMI entries (preserves
   the marginal edge-weight distribution, destroys every word's neighborhood → the category structure is gone).
   The shuffled-graph replay's TRUE-graph plausibility collapses to **0.018** (≈ the random floor 0.020, vs the
   real replay's 0.328). ⇒ the **learned co-occurrence statistics**, not the SVO template / a string artifact,
   drive the proposals.
4. **Moat-honesty** (the owner-sanctioned moat/generativity trade, `feedback_moat_not_hard_lossy_memory_ok`):
   - proposals are a SEPARATE, honestly-flagged HYPOTHESIS channel — a proposal must NEVER pass the composer's
     KNOWN-fact retrieval (`query_patient`/`ask_yes_no` still abstain on it): **0 leaks** across all seeds (the
     brain distinguishes "I know X" from "X is plausible");
   - the non-contradiction gate caught **every** explicitly-negated fact (the brain stored 12 NEGATED facts like
     "duck does NOT eat brown"; each is itself a plausible recombination — a tempting nonsense — and the gate
     re-proposed **0** of them).
   - **Lesion** (scoping §5.4): ablating the plausibility gate floods nonsense (278 accepted, 12% plausible) —
     the gate is causally responsible. (The non-contradiction gate's effect is the 0-negated-re-proposed metric.)
5. **Multi-seed** (6 seeds: 42/43/44/100/101/102) per `feedback_6seed_validation` — the novel-composition score
   and advantage ratio are variable effects.

---

## Two design iterations (the honest record)

The first two attempts returned `HONEST_NEGATIVE` for instructive reasons; the fixes are part of the result:

1. **Strict 3-way plausibility starved the universe.** Requiring all three pairs (agent~action AND
   action~patient AND agent~patient) graph-related made the discoverable universe tiny (1–3 triples) — the
   direct agent~patient relation is rarely high-PPMI even in a sensible scenario ("frog" ↔ "head" is weak even
   when "frog jump" and "jump head" are fine). FIX: **selectional-preference plausibility** (agent~action AND
   action~patient as the hard gate; agent~patient as a graded ranking bonus, reported as a stricter secondary
   "strongly-plausible" count). This is the standard subject-verb-object semantics.
2. **Single-valued cue→patient forbade recombination.** Defining contradiction as "the agent does a DIFFERENT
   patient under this action" rejected every plausible recombination (all stored cues saturated the high-PPMI
   agent-action pairs, so every novel triple reused a stored cue → flagged contradictory → 0 accepted despite
   43% being plausible). FIX: the brain stores some **NEGATED facts** ("X does NOT Y", via the composer's
   polarity tag); a contradiction is ONLY re-asserting an explicitly-negated triple. Multiple affirmative
   patients per cue are allowed. This makes the recombination space rich AND gives the non-contradiction gate
   real, testable work.

These iterations are exactly the "the comfortable verdict is the START of the research" discipline — the early
negatives were measurement/world-construction artifacts, not the mechanism's verdict.

---

## Scope + honesty (where this sits, what it is NOT)

- **What this is:** the first brain-mechanism novel-COMPOSITION (G2 propositions), validated multi-seed,
  graph-load-bearing, moat-preserved. It directly attacks the measured 0.0 retrieval ceiling with a
  biology-faithful mechanism (generative replay / constructive imagination).
- **What this is NOT (the scoping's honest framing):** it produces **propositions (G2), not fluent open
  discourse (G3)** — a proposed triple would FEED a renderer (the neural serial-order renderer / the grounded
  faculty) for the surface form, still gated/verified. The plausibility signal is a learned co-occurrence
  statistic the brain has (PPMI cortex); the recombination BOOKKEEPING (the loop over samples) is host code, as
  is legitimate for any de-risk harness — the load-bearing pieces (the plausibility likelihood + the fact store
  + the moat) are the brain's. A fully-spiking generative-replay sampler (resampling on the substrate via SWR-
  gated CA3 + the engram/replay machinery the project already has) is the natural follow-on if this direction is
  prioritized.
- **The moat/generativity trade is owner-sanctioned and exercised honestly:** a pure no-confab moat ("never
  assert the un-stored") FORBIDS G2 by construction; generative replay DELIBERATELY asserts never-told (but
  plausible, graph-supported) propositions. The gate becomes "plausible given the learned graph + not
  contradicting a stored fact." The WEAKER-but-still-strong guarantee is preserved: a proposal never passes as a
  known fact (0 leaks), and an explicitly-false recombination is never proposed (0 negated re-proposed). This is
  the reconstructive-memory stance — biologically correct (real memory confabulates *plausibly*).

---

## Reproduce

```bash
# the decisive 6-seed run (CPU, ~140 s)
SIM_BACKEND=numpy python -u -m research.runners._genfrontier_b2_generative_replay_derisk \
    --seeds 42,43,44,100,101,102 --D 128 --n-facts 24 --n-negated 12 --n-attempts 3000 --tau-pct 50 \
    --out research/findings/raw/_genfrontier_b2_generative_replay_derisk.json
```

Knobs: `--tau-pct` (the learned graph-related threshold = percentile of positive PPMI), `--n-facts` /
`--n-negated` (affirmed / negated facts the brain is told), `--advantage-bar` (the replay-vs-random ratio gate,
default 3×), `--shuffle-collapse-frac` (the shuffled-graph collapse gate, default 0.5×). `--store-floor-bar`
(0.95) is the composer's baseline-fidelity SANITY tolerance, NOT a b2 gate.

## Catalog / literature
G.09 (constructive imagination / future simulation — was "missing"; this is its first concrete probe), D.50/D.51
+ N.* (replay), D.14 (engram). Hippocampal formation as a hierarchical generative model / generative replay
(Stoianov-Maisto-Pezzulo 2022); A generative model of memory construction and consolidation (Nat Hum Behav 2023).
