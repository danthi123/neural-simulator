# `MultiTurnAgentV2.narrate(topics)` — coherent multi-sentence narration PROMOTED to production (GO, 6/6)

**Date:** 2026-06-17
**Verdict:** **GO** — unanimous 6/6 seeds, all four load-bearing controls at 1.000.
**What shipped:** a production `narrate()` method on `MultiTurnAgentV2` (`research/runners/multi_turn_agent_v2.py`)
that composes two separately-validated, multi-seed-GO de-risk mechanisms into one method, plus
`tests/test_multiturn_narrate.py`. **NO `sim/` edit; reuse-by-import only; no git commit.**

## The capability

Given an ordered list of topics the agent has stored facts about, `narrate(topics)` returns a COHERENT
MULTI-SENTENCE string. It holds the topics in the order-encoded working memory (gamma-slot POSITION phasors on the
project's spiking resonate-and-fire substrate), emits one sentence per slot IN SLOT ORDER, AND pronominalizes a
recurring subject — the pronoun resolving on the substrate to its correct antecedent slot.

```
narrate(["dog", "bird", "dog"])  ->  "dog ran north. bird ate worm. then it ran north."
                                       the recurring 'dog' is pronominalized; "it" resolves
                                       (spiking slot-anaphora) to 'dog' at gamma-slot 0.
```

## Mechanism (reuse-by-composition; no new machinery)

Two validated GO mechanisms, lifted verbatim and composed:

| piece | role in `narrate` | provenance |
|---|---|---|
| ORDERED EMISSION — `OrderedPositionWM.encode_sequence` / `read_slot` + `describe()` (neural word order) | hold the topics in gamma-slot position phasors; emit one sentence per slot in slot order | `2026-06-17-multisentence-ordered-emission-derisk.md` (GO 6/6) |
| CROSS-SENTENCE COHERENCE — `MultiTurnAgentV2.referent_at(antecedent_slot)` (by-slot spiking slot-anaphora) | a recurring subject → pronoun resolving to its EARLIEST (antecedent) slot, not the most-recent | `2026-06-17-cross-sentence-coherence-derisk.md` (GO 6/6) |

The coherence loop (the de-risk's `CoherentDiscourse`) is promoted into the module as an internal
`_CoherentNarration` helper that drives the agent's own `_window` / `_composite` / `wm` on the spiking substrate:
accumulate referents in surface order, track `_slot_of[ref]` = the EARLIEST slot each referent occupied (its
**antecedent slot**); a first mention is the validated full-noun `describe` path; a recurrence emits a pronoun and
resolves it by reading the antecedent slot (`referent_at`). Content comes from the composer's own flat fact memory
(`composer.kb`); order comes from the WM slots; intra-sentence word order is the neural serial-order renderer
(`enable_neural_render=True`). **Why antecedent-slot, not most-recent-slot:** after "dog ran north" the surface
window is `[dog(slot0), north(slot1)]` — a recurring "dog" must resolve to slot 0, NOT the most-recent slot ("north")
— the by-slot addressing the rate-attractor buffer structurally lacked.

The two existing de-risk runners (`_phaseB_multisentence_ordered_emission_derisk.py`,
`_phaseB_cross_sentence_coherence_derisk.py`) keep their own copies and **still run GO at seed 42** after the module
edit (verified).

### No-confab moat (held)
A topic with NO stored fact → the slot ABSTAINS (no sentence, skipped from the surface string), never a
confabulated sentence. (A topic whose WM read does not even ground — the familiarity gate — is likewise skipped.)

### Side-effect-free
`narrate()` uses a FRESH discourse buffer per call: it SAVES the agent's standing `_window`/`_composite` on entry
and RESTORES them on exit (a `try/finally`), so a narration does not perturb an in-progress multi-turn dialogue.
Verified: a `hear` → `narrate` → resolve sequence leaves `most_recent_referent()`, the in-progress Q&A, and the
held window unchanged. The existing `MultiTurnAgentV2` capabilities (multi-referent resolution, single-referent
anaphora, Q&A, reason-chain) are untouched.

## Tests + per-seed results (6 seeds: 42 43 44 100 101 102, CPU/numpy)

`tests/test_multiturn_narrate.py`. Frozen bars: per-seed per-control accuracy ≥ 0.80; GO bar = a FRACTIONAL
≥ 5/6 of seeds per control (scaled to the seed count, never a hardcoded absolute — `_go_thresh`).

| seed | ordered narration | coherence | order-control (permute) | order-control (flip) | no-confab moat |
|------|:----------------:|:---------:|:-----------------------:|:--------------------:|:--------------:|
| 42   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 43   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 44   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 100  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 101  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 102  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **mean** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **pass** | **6/6** | **6/6** | **6/6** | **6/6** | **6/6** |

GO bar = ≥ 5/6 seeds. Trials/seed: ordered 40, coherence 40, order-permute 40, order-flip 40, no-confab 30.

- **Test 1 — ORDERED NARRATION.** `narrate` of a random ORDERED K=3 subset (all distinct → no pronouns) emits
  exactly the 3 stored-fact sentences in the topic order (exact content per topic). **1.000, all seeds.**
- **Test 2 — COHERENCE.** Discourse `[sA, sB, sA]`: the 3rd sentence is pronominalized AND its "it" resolves
  (spiking slot-anaphora) to sA at its antecedent slot. **1.000, all seeds.**
- **Test 3 — ORDER-CONTROL (two parts, both load-bearing).** (a) permuting the topics permutes the emitted
  sentences (`emitted(perm) == perm(emitted)` — proves the inter-sentence order comes from the WM slots, not a
  fixed storage order; a storage-dump would fail). (b) swapping WHICH referent recurs FLIPS the resolved antecedent
  (sA↔sB — proves resolution is by the recurring referent's own slot, not a fixed entity). **Both 1.000, all
  seeds** (6/6 seeds also pass the combined permute-AND-flip).
- **Test 4 — NO-CONFAB MOAT.** A length-3 narration with one unknown topic (no stored fact) at a random slot →
  that slot abstains (skipped), the known slots emit their correct sentences in order, the surface has no
  confabulated unknown sentence. **1.000, all seeds.**

Plus: `test_narrate_is_side_effect_free` (seeds 42, 100), `test_narrate_empty_and_all_unknown` (empty topics → `""`;
an agent with no facts → `""`), and `test_fixed_coherent_transcript_seed42` (the exact validated surface string).

**Companion regression:** `tests/test_multi_turn_ordered_wm.py` (the 31-assertion MultiTurnAgentV2 gate) stays
green — **31/31 passed** after the edit (multi-referent resolution, the order-control flip, the no-confab moat, the
single-referent regression, and the code-parity guard all intact).

## Example narrated transcripts (seed 42)

```
ordered:    narrate(["bird", "hawk", "frog"])  ->  "bird ate worm. hawk chased mouse. frog crossed road."
coherence:  narrate(["dog",  "cat",  "dog"])   ->  "dog ran north. cat saw river. then it ran north."
                 the recurring 'dog' is pronominalized; "it" RESOLVED on the substrate to
                 antecedent 'dog' (read from gamma-slot 0) — not the most-recent slot ("river").
```

## Honest scope

- **Capability + all four controls: robustly GO, 6/6, 1.000.** This promotes the two de-risked mechanisms verbatim;
  the per-seed fidelity matches the de-risks (both were 6/6).
- **Inherited operating envelope.** The ordered-WM bundle capacity caps clean multi-sentence turns at K≈4 at the
  agent's `D=128` (the de-risk's documented ceiling; 5+ slots erode, seed-variable, needing a larger D). `narrate`
  is validated at K∈{2,3} (the de-risk loads) and at vocab 10 (6 subjects + objects + an unknown probe).
- **Anaphora policy is the antecedent heuristic.** "Recurring subject → its first-introduction slot" is the
  resolution rule. Richer anaphora (a pronoun for a recurring OBJECT, number/gender beyond singular "it", binding
  by syntactic role rather than antecedent slot) is a bounded follow-on — any held slot is readable
  (`referent_at`), so the substrate supports it; the selection policy is the open part.
- **Substrate is the deployed one.** Bind/unbind/bundle/cleanup are the production composer's spiking RF operations
  (resonate-and-fire neurons + complex synapses); narration reuses them by import. No new mechanism, no `sim/` edit.

## Reproduce

```bash
SIM_BACKEND=numpy python -m pytest tests/test_multiturn_narrate.py tests/test_multi_turn_ordered_wm.py -v
```

Deliverables: the extended `research/runners/multi_turn_agent_v2.py` (`narrate()` + internal `_CoherentNarration`
helper + `_join_sentences`), `tests/test_multiturn_narrate.py`, this findings doc. The two de-risk runners are
unchanged and still GO. No `sim/` edit; no git commit (controller commits after verifying).
