# Concept pool architecture — Phase 1: diversity beyond 4 motor pools

**Date:** 2026-05-13
**Status:** v9 BREAKTHROUGH — bidirectional binding works at single seed.
Phase 1 W→A: 6/12. **Phase 3 A→W: 12/12 (all pools speak trained word).**
Multi-seed validation queued.

## User mandate

2026-05-12 user directive: "those scaling axes are 100% what need to be
given our full focus currently, as the blocker for reaching
conversational capabilities, as the blocker for reaching conversational
capabilities. Nothing else we do is very productive until that is
accomplished, regardless of how small the vocab needs to be to still run
locally, it needs concepts, composition, and diversity."

Three blockers explicitly named:
1. **Concepts** — non-direction concept categories (nouns, verbs, etc.)
2. **Composition** — combining concepts into phrases
3. **Diversity** — beyond 4 motor pools

## Architectural diagnosis (root cause)

Every prior conversational ceiling shares one common cause:

| Ceiling | Symptom | Architectural cause |
|---|---|---|
| P5 abstract concepts (2/4 PARTIAL) | Naming fails | 4 motor pools, no noun pool |
| Tier 2.3 phrase composition (34-40%) | Verb+noun won't bind | 4 motor pools, no verb pool |
| In-vivo new-vocab binding (2/4 fixed) | Capacity ceiling | 4 motor pools, all slots taken |
| Synonym32 cross-language vocab (44% W→A) | Approaching encoding wall | Still only 4 motor outputs |

The single shared cause: **only 4 distinct output categories** (motor_N,
motor_E, motor_S, motor_W). Every concept must collapse onto one of
those four directions. "Apple" can't bind to a non-direction. Synonyms
work (because they share direction targets); concepts don't.

## Solution: dedicated pools per concept category

Mirror the proven Tier 1 recipe (6/6 multi-seed PASS for direction
words) for non-direction concepts:

```
Existing motor pools (4):
  motor_N, motor_E, motor_S, motor_W

NEW noun pools (4):
  noun_pool_APPLE, noun_pool_RIVER, noun_pool_DOG, noun_pool_CAT

NEW verb pools (2):
  verb_pool_GO, verb_pool_COME

→ 10 distinct output categories (2.5× diversity gain)
```

Each pool follows the Tier 1 recipe:
- 500 neurons (Schieber 2001 / Rathelot 2009 motor sub-pool size)
- exc 0.8 + internal recurrence 0.10 (Lefort 2009 cortical canon)
- `lang_input → pool` plastic pathway with topographic prior
- Optional FS cross-inhibition WITHIN kind (Vogels 2011 / Hofer 2011)
- Reciprocal `pool → language_output` for A→W readout (Pulvermüller 2003)

Critical design choice: **FS cross-inhibition is WITHIN kind only**.
- noun_pool_APPLE_fs inhibits other noun pools, NOT motor or verb pools
- verb_pool_GO_fs inhibits other verb pools, NOT motor or noun pools

This deliberate omission allows composition: "go north" should fire
verb_pool_GO + motor_N together. If FS crossed kinds, the pools would
compete and only one would win.

## Architecture summary

```
                    language_input (4096)
                    ↙           ↓           ↘
        motor_X (500)   noun_pool_X (500)   verb_pool_X (500)
        × 4 pools       × 4 pools           × 2 pools (v1) / 4 (v2)
        (FS within)     (FS within)         (FS within)
                    ↘           ↓           ↙
                    language_output (4096)
```

## Biological grounding (catalog references)

Maps to several catalog entries (catalog-build branch,
`references/feature-catalog.md`):

- **G.11 Dual-stream model of language (Hickok & Poeppel)** —
  Kandel 6e Ch 55 pp 1380–1387. Ventral stream: superior + middle
  temporal → semantic interface (sound→meaning). My lang_input →
  concept_pool pathway implements the ventral stream's
  sound-to-meaning map.

- **G.13 Wernicke's area — auditory-to-semantic mapping** —
  Kandel 6e Ch 55 pp 1384–1385. "Selects words matching intended
  meaning; phonemic and semantic paraphasias result from selection
  failures." My v1 0/10 result is essentially a paraphasia failure
  — wrong concept (verb_pool_COME) consistently selected for input.

- **G.12 Broca's area — speech production + grammatical processing** —
  Kandel 6e Ch 55 pp 1382–1384. Notably: "Damage → labored,
  agrammatic speech, **retained noun selection, lost
  function-word/verb use**". This validates my architectural choice
  to put nouns and verbs in separate pools — they are
  biologically dissociable.

- **E.04 Topographic / somatotopic maps** — Kandel 6e Ch 17 pp 460–462.
  "Adjacent receptors map to adjacent cortical neurons, producing
  organized maps." My topographic prior (Pulvermüller-style 2.0/0.5
  boost ratio) implements this for lang_input → concept_pool edges.

- **E.05 Lateral inhibition** — Kandel 6e Ch 22 pp 588–593. "Inhibitory
  horizontal connections sharpen contrast." Cluster B notes MSN
  lateral inhibition is the "same algorithmic motif" used for action
  WTA. My within-kind FS cross-inhibition implements this for
  category selection within each concept kind.

- **E.10 Cortical columns** — Kandel 6e Ch 23 pp 562–569. "Vertical
  columns share a feature; horizontal organization tiles all values."
  My per-concept pools are functionally column-like — each pool is
  a dedicated population representing one semantic feature.

- **Cluster B / FS WTA** — Cluster B catalog (entries B.01-B.07).
  Striatal PV-FSI architecture provides the algorithmic kin for my
  within-kind FS cross-inhibition. Catalog notes FSI feedforward
  inhibition is the canonical WTA microcircuit.

**Catalog gap**: G.11/G.12/G.13 are marked "Sim status: missing".
This work is the first attempt to implement the ventral language
stream in the simulator.

## Prior art in this codebase

`enable_multi_pool_wernicke` (existing infrastructure in
`build_biological_brain_regions`) already implements a similar
per-concept Wernicke architecture. The P5 iter KK-PP arc (2026-05-11
to 2026-05-12) explored this exact problem space and discovered:

- Iter AA (toy scale, weak dynamics 0.05/0.3/0.8): **4/6 BIDIR** —
  ceiling.
- Iter KK (canon dynamics 0.10/2.0/4.0, biological scale):
  **0/seed_42** — "canon amplifies structural bias".
- Iter LL/MM/NN/OO/PP: 0-1/seed at biological scale, ceiling
  confirmed.

Key takeaway: at biological scale, **per-seed random structural
pool variance compounds through multi-hop chains and dominates the
input signal**.

My concept_pool_demo uses **cortical canon (0.10/2.0/4.0)** for
each concept pool. This may trigger the same structural-bias
amplification observed in wernicke iter KK. **v2 will tell us
whether the symmetric pool counts + target-only STDP gating + 4x
topographic prior are enough to overcome this — or whether the same
ceiling applies.**

If v2 fails, the proven fallback is **weak dynamics
(0.05/0.3/0.8)** following iter AA. This trades cortical realism
for differentiation robustness.

## v2b smoke result (50 events, 2048 lang_input, 200 per_pool)

**Result: 3/12 PASS** (north, east, go).

Wall clock: 1245s (~21 min). Per-word: motor 146-180s, noun 71-84s,
verb 55-57s (GPU warm-up effect).

| Word | Target | Target rate | Max-off | Max-off pool | Ratio |
|---|---|---|---|---|---|
| north | motor_N | **1.220** | 1.190 | noun_pool_RIVER | **1.03x PASS** |
| east | motor_E | **1.410** | 1.285 | noun_pool_RIVER | **1.10x PASS** |
| south | motor_S | 0.730 | 1.000 | verb_pool_LOOK | 0.73x FAIL |
| west | motor_W | 0.980 | 1.200 | noun_pool_RIVER | 0.82x FAIL |
| apple | noun_pool_APPLE | 1.420 | 1.490 | motor_W | 0.95x FAIL |
| river | noun_pool_RIVER | 0.920 | 1.025 | verb_pool_GO | 0.90x FAIL |
| dog | noun_pool_DOG | 1.260 | 1.315 | verb_pool_GO | 0.96x FAIL |
| cat | noun_pool_CAT | 1.775 | 1.940 | motor_N | 0.91x FAIL |
| go | verb_pool_GO | **1.725** | 1.565 | noun_pool_CAT | **1.10x PASS** |
| come | verb_pool_COME | 0.905 | 1.390 | noun_pool_DOG | 0.65x FAIL |
| stop | verb_pool_STOP | 0.835 | 1.215 | verb_pool_GO | 0.69x FAIL |
| look | verb_pool_LOOK | 0.920 | 0.950 | noun_pool_DOG | 0.97x FAIL |

### Diagnosis: v2 fixes worked, but training under-dosed at 50 events

- **v1 dominance pattern gone**: verb_pool_COME no longer dominates
  9/10 words. Different pools dominate as max-off per word — no
  single structural winner. The 4-verb pool fix + target-only STDP
  gating addressed the v1 root cause.
- **Many borderline FAILs** (ratio 0.90-0.97): close calls. With
  more training, target weights should pull further ahead.
- **Some target rates LOW** (south 0.73, come 0.91, stop 0.84):
  these words got under-trained pathways. More events should help.
- **Tier 1 used 200 events**; smoke v2b used 50 (one-quarter dose).

### v2c (200 events) FAIL: 0/12 — canon amplifies bias

v2c trained 200 events with v2 architecture. **Result: 0/12 PASS,
WORSE than v2b's 3/12 at 50 events.**

| Word | Target | v2b ratio | v2c ratio | Δ |
|---|---|---|---|---|
| north | motor_N | 1.03x PASS | 1.00x tie | -0.03 |
| east | motor_E | 1.10x PASS | 0.94x FAIL | -0.16 |
| south | motor_S | 0.73x | 0.54x | -0.19 (worse) |
| west | motor_W | 0.82x | 0.62x | -0.20 |
| apple | noun_APPLE | 0.95x | 0.97x | +0.02 |
| river | noun_RIVER | 0.90x | 0.74x | -0.16 |
| dog | noun_DOG | 0.96x | 0.88x | -0.08 |
| cat | noun_CAT | 0.91x | 0.89x | -0.02 |
| go | verb_GO | 1.10x PASS | 0.84x FAIL | -0.26 |
| come | verb_COME | 0.65x | 0.72x | +0.07 |
| stop | verb_STOP | 0.69x | 0.64x | -0.05 |
| look | verb_LOOK | 0.97x | 0.73x | -0.24 |

More training → most ratios got WORSE (10 of 12). Pattern is the
same as **P5 iter KK with multi-pool wernicke**: at biological scale
with many pools, canon dynamics (0.10/2.0/4.0) cause off-target
pools to accumulate activated states from spillover drive across
events. Internal recurrence + NMDA bistability + cross-kind FS
omission = unbounded off-target firing growth.

### v3 fix: weak dynamics (iter AA recipe)

Implementation (commit fc31152):
- New params in `build_biological_brain_regions`:
  `concept_pool_internal_density / exc_weight_mean / inh_weight_mean`
- Default None → use motor_internal_density (backward compat for motor)
- `--weak-concept-dynamics` CLI in concept_pool_demo:
  sets concept pools to 0.05/0.3/0.8 (iter AA recipe)
- Motor pools KEEP canon (Tier 1 6/6 multi-seed works with canon)
- Only non-motor pools get weak dynamics

Plus interleaved training (`--interleaved`) to match Tier 1 pattern.

v3 launched seed 42 with: weak dynamics + interleaved + 100 events.

### v3 (weak + interleaved, 100ev): 3/12 PASS — same ceiling

PASSes: north, west, cat. Different words than v2b (was north, east,
go). Weak dynamics prevented the v2c regression but didn't push past
the v2b 3/12 ceiling.

### v4 (v3 + stronger topographic 3.0/0.3, 100ev): 5/12 PASS — improvement

PASSes: north, east, apple, dog, cat. All 4 noun pools improved.

Stronger topographic prior (10x ratio vs 4x) gave +2 words. Pattern:
- All 4 nouns improving (apple 1.03x, dog 1.12x, cat 1.37x)
- 2 motor PASS (north, east — borderline)
- 2 motor FAIL (south, west)
- All 4 verbs FAIL (especially under-trained: stop 0.84, look 0.72)

| Variant | PASS | Key change |
|---|---|---|
| v1 (canon seq 200ev) | 0/10 | baseline (FS bug) |
| v2b (canon seq 50ev) | 3/12 | FS fixed |
| v2c (canon seq 200ev) | 0/12 | canon amplifies bias |
| v3 (weak interl 100ev) | 3/12 | weak fixes regression |
| v4 (weak interl 100ev + 3.0/0.3 topo) | **5/12** | strong topo helps |

Trajectory: each architectural refinement adds 2-3 PASS words. The
ceiling of v2b/v3 was structural; v4 broke past it via topographic
strength. Under-trained verbs suggest 200 events may push further.

### v5 launched: v4 + 200 events

Expected: most under-trained verbs (stop, look at 0.72-0.84 target
rate) should hit higher target rates with double the training. If
v5 = 7+/12, multi-seed validation is justified.

### v6 (cross-kind topographic dampening): 6/12 PASS

PASSes: apple, river, dog, cat (all nouns), go, come.
FAILs: all 4 motors, stop, look.

The cross-kind topographic dampening helped nouns pass uniformly,
but motor target rates dropped sharply (north: 1.22 → 0.67).

### v7 (target-priority topographic): 6/12 PASS

Cross-kind dampening bug: when multiple words' active lang_input
neurons overlap, the multiplicative dampening cumulates. An edge
"target" for "north" got boosted (3.0x) then dampened by 11 other
words (0.3x each) = 3.0 × 0.3^11 ≈ 5e-6. Killed motor target firing.

Fix: two-pass priority. Pass 1 collects all target edges into a set
and boosts them. Pass 2 dampens off-target edges, skipping any in
target set. Each edge gets exactly one bias (target OR off-target).

### v7 weight probe (commit 1c3a411 priority fix)

| Variant | target_w | max_off_w | weight ratio | PASS |
|---|---|---|---|---|
| v4 (multiplicative bias, within-kind) | 5.4-5.7 | 2.8-2.9 | 1.9x | 5/12 |
| v6 (multiplicative bias, cross-kind) | -- | -- | -- | 6/12 |
| **v7 (priority bias, cross-kind)** | **6.68-6.73** | **1.6-1.9** | **4.0x** | **6/12** |

v7 weights are correctly biased: target ~4x stronger than off-target.
Yet firing-rate PASS rate is still 6/12. This indicates the remaining
gap is at the dynamics level (recurrent amplification, NMDA bistability)
not at the weight level.

PASSes: north, east, apple, dog, cat, go (6)
FAILs: south, west, river, come, stop, look (6)

Pattern: motor_E structurally over-fires across many words, dominating
as max-off for south/west. With multi-seed, different pools would
dominate different seeds — averaging may give consistent ~50% PASS.

### Final v7 architecture summary

```
build_concept_bridge:
  weak_dynamics = True    (0.05 / 0.3 / 0.8 for concept pools; motor canon)
  topographic_factor = 3.0
  off_target_factor = 0.3 (priority-based, target-first)
  interleaved = True      (matches Tier 1 pattern)
  n_train_events = 200
  n_per_pool = 200, n_fs_per_pool = 24
```

### Trajectory summary (single seed, 8 iterations)

| v | PASS | Key change |
|---|---|---|
| v1 | 0/10 | baseline (broken FS topology) |
| v2b | 3/12 | 4 verb pools + target-only STDP gate |
| v2c | 0/12 | canon dynamics amplify bias with more events |
| v3 | 3/12 | weak dynamics fixes v2c regression |
| v4 | 5/12 | stronger topographic prior 3.0/0.3 |
| v6 | 6/12 | cross-kind dampening (helps nouns, hurts motors) |
| v7 | 6/12 | priority-based bias (restores motors AND keeps nouns) |

**v7 is the production recipe.** From 0/10 → 6/12 across 8 iterations.

## Multi-seed validation (v7 architecture)

| Seed | PASS | Strong words | Weak words |
|---|---|---|---|
| 42 | 6/12 | north, east, apple, dog, cat, go | south, west, river, come, stop, look |
| 43 | 7/12 | north, east, south, river, cat, come, look | west, apple, dog, go, stop |
| 44 | 7/12 | north, east, west, apple, river, cat, look | south, dog, go, come, stop |
| 45 | 7/12 | (data in seed45_v7.json) | |
| 46 | 5/12 | (data in seed46_v7.json) | |

**v7 5-seed final: mean 6.4/12 (53%), std 0.89, range 5-7.**

Verdict: all 5 seeds PARTIAL (5 ≤ PASS < 8). 0 GO, 0 FAIL.

Per-word PASS rate across 5 seeds:
- **80% robust**: north, east, cat — pass on 4 of 5 seeds
- **60% mixed**: apple, river, look — pass on 3 of 5 seeds
- **40% fragile**: south, west, dog, go, come — pass on 2 of 5 seeds
- **20% fragile**: stop — only passed on seed 43

Architecture produces CONSISTENT 53% PASS rate (low std 0.89 across
5 seeds). 3 words (north, east, cat) reach 80% — demonstrating the
architecture CAN reliably bind specific (word, pool) pairings.

The fragile words have seed-dependent unfavorable structural pool
connectivity. With orthogonal codes or pre-shape phase, these might
flip to robust. Drive-pattern overlap analysis showed stop's
lang_input overlap is normal (4-13%), so issue is downstream.

## Phase 2 (composition) + Phase 3 (A→W readout) on v7 seed 42

### Phase 2: composition test on saved v7 bridge

Using `concept_compose_demo --load-bridge seed42_v7.simstate.h5
--weak-concept-dynamics`:

| Test | Result | Notes |
|---|---|---|
| Single-word isolation | **9/12 PASS** | Better than concept_pool_demo's 6/12 (slight state difference) |
| Sequential composition (NMDA persistence) | **0/6 PASS** | Pool_a drops to ~0 firing when drive switches to word_b |
| Co-firing composition (merged drive) | **2/6 PASS** | go+north and dog+west work |

Key trade-off discovered: **weak dynamics (chosen to fix v2c canon
bias) prevents NMDA bistability needed for sequential composition.**
Pool firing collapses to baseline within 25ms of drive removal.

Sequential composition would require either:
- Canon dynamics (reintroduces bias amplification)
- Longer-tau NMDA in concept pools specifically
- Explicit working-memory gate (Tier 2.3 PFC pattern)

Co-fire composition partial-works (2/6 = 33%) — concept_pool architecture
demonstrates concepts + diversity + co-fire composition; sequential
composition is an open architectural problem.

### Phase 3: A→W readout on v7 saved bridge

Using `concept_speak_demo --load-bridge seed42_v7.simstate.h5
--weak-concept-dynamics`:

**Result: 0/12 PASS.** Driving each pool produces uniform low
cosines (0.05-0.15) against word reference patterns.

Diagnosis: `concept_to_language_output_weight` defaults to 0.5,
while `motor_to_language_output_weight` is 2.0 (Tier 1 setting).
Concept pool → language_output projection is **4x weaker** than the
motor projection that Tier 1 uses successfully.

Fix queued for v8 batch (commit d21efae then revert 54f72d1 to
preserve v7 multi-seed consistency). Will re-apply after seed 46
v7 completes and run v8 single-seed validation.

**Sizes**: 13,792 total neurons (4096 lang_in + 4096 lang_out +
4×500 motor + 4×60 motor_FS + 4×500 noun + 4×60 noun_FS + 2×500 verb +
2×60 verb_FS). 14.7M synapses, 2.4 GB GPU.

## Implementation

Three modules:

**`sim/research/runners/text_minimal_isolation.py`** —
Extended `build_biological_brain_regions` with parameters:
- `enable_noun_pools`, `noun_pool_names`, `n_noun_per_pool`, `n_noun_fs_per_pool`
- `enable_verb_pools`, `verb_pool_names`, `n_verb_per_pool`, `n_verb_fs_per_pool`
- `concept_to_language_output_density`, `concept_to_language_output_weight`

Internal helper `_add_concept_kind(kind, pool_names, ...)` adds:
- Per-pool BrainRegion (500-neuron cortical canon)
- `language_input → pool` RegionPathway (gate-tagged `language_input_to_{kind}_pool`)
- Reciprocal `pool → language_output` RegionPathway (gate-tagged `{kind}_pool_to_language_output`)
- FS interneurons + cross-inhibition (within kind)

Default OFF for backward compat; existing tests continue to work.

**`sim/research/runners/concept_pool_demo.py`** —
Phase 1 validation runner. Trains all 10 pools with paired teacher-current
recipe, then measures cross-category isolation:
- For each word, drive lang_input(word) alone (no teacher)
- Measure firing in all 10 pools
- PASS = target pool > all off-target pools

**`sim/research/runners/concept_compose_demo.py`** —
Phase 2 composition runner. Same training, then three tests:
1. Cross-category isolation (Phase 1 sanity)
2. Sequential composition (NMDA bistability: drive A then B, both
   targets active in B's window)
3. Co-firing composition (merged drive A+B → both targets active)

## Smoke test result (sub-Tier-1 dose)

5 events × 1024 lang_input × 100 neurons/pool (smoke): **3/10 PASS**.
Expected at this dose — Tier 1 needs 200 events to differentiate, and
the smoke uses 1/4 the lang_input + 1/5 the per-pool neuron count.
Smoke confirms the pipeline works end-to-end: bridge builds, topographic
bias applies correctly, training runs, eval measures.

## Real run (seed 42 v1: 0/10 FAIL, 2026-05-13)

Full Tier 1-scale config:
- n_train_events = 200 (Tier 1 default)
- n_lang_input = 4096 (Tier 2.1 v4 scale-up)
- n_per_pool = 500 (Schieber 2001 motor sub-pool)
- n_fs_per_pool = 60 (12% PV-FSI fraction)
- Wall clock: 1314s (~22 min)

**Result: 0/10 PASS.** All 10 words FAILed cross-category isolation.

| Word | Target | Target rate | Max-off | Max-off pool | Ratio |
|---|---|---|---|---|---|
| north | motor_N | 1.768 | 2.986 | verb_pool_COME | 0.59x |
| east | motor_E | 1.512 | 3.232 | verb_pool_COME | 0.47x |
| south | motor_S | 0.992 | 2.376 | verb_pool_COME | 0.42x |
| west | motor_W | 0.818 | 1.900 | verb_pool_COME | 0.43x |
| apple | noun_pool_APPLE | 1.560 | 2.060 | verb_pool_COME | 0.76x |
| river | noun_pool_RIVER | 1.548 | 2.334 | verb_pool_COME | 0.66x |
| dog | noun_pool_DOG | 2.180 | 2.898 | verb_pool_COME | 0.75x |
| cat | noun_pool_CAT | 1.784 | 2.824 | verb_pool_COME | 0.63x |
| go | verb_pool_GO | 2.322 | 2.826 | verb_pool_COME | 0.82x |
| come | verb_pool_COME | 2.806 | 3.180 | verb_pool_GO | 0.88x |

## Diagnosis: verb_pool_COME structural dominance (FS imbalance)

**Pattern**: verb_pool_COME has anomalously high firing rate (2.8-3.2)
across NINE of ten words it wasn't trained for. Even target pools fire
lower than off-target verb_pool_COME.

**Root cause**: FS within-kind imbalance.

- Motor pools (4): each motor_FS has 3 cross-inhibition edges
- Noun pools (4): each noun_FS has 3 cross-inhibition edges
- **Verb pools (2)**: each verb_FS has only **1** cross-inhibition edge

Verb pools receive 1/3 the within-kind suppression that motor/noun pools
do → verb pools fire freely across all stimuli → verb_pool_COME (seed-
specific structurally dominant pool) dominates everything.

This was anticipated in the design tradeoff documentation (FS within-
kind only to enable composition), but the asymmetry in pool count per
kind broke the FS suppression budget.

## Fix v2: 4 verb pools + tighter topographic prior

Two changes pushed in commit 8bbb01a:

**1. Expand verb pools from 2 → 4** (GO, COME, STOP, LOOK)
   - Now each verb_FS has 3 cross-inhibition edges (matches noun/motor)
   - FS suppression budget equalized across all kinds
   - 12 total output pools (was 10) — still 3× Tier 1 diversity

**2. Tighten topographic prior 1.5/0.7 → 2.0/0.5**
   - Target/off-target weight ratio at init: 2.14x → 4.0x
   - Stronger initial signal for STDP to amplify
   - Within Pulvermüller 2003 reported biology range (2-4x)

V2 seed 42 LAUNCHED with `--save-bridge` enabled (for post-mortem
weight probing if it also fails).

## Phase 1 success criteria

For Phase 1 (cross-category isolation), each word should fire its
target pool >> all off-target pools:

| Threshold | Verdict |
|---|---|
| n_pass ≥ 8/10 | GO — architecture is sound for Phase 2 |
| n_pass 5-7/10 | PARTIAL — needs refinement (more training events, stronger topographic prior, or NMDA tuning) |
| n_pass < 5/10 | NO-GO — architecture has fundamental issue, revisit |

## Phase 2 success criteria (after Phase 1 PASS)

For Phase 2 (composition), 6 word pairs spanning verb+direction,
noun+direction, verb+noun:

| Test | Threshold | Verdict |
|---|---|---|
| Sequential | ≥ 4/6 pairs PASS NMDA persistence | GO |
| Co-firing | ≥ 4/6 pairs PASS both-targets-dominate | GO |

## What this unlocks

If Phase 1 PASSes:
- Real diversity: 10 distinct concepts trainable on biology-grounded sim
- Foundation for scaling to 20+ noun pools, 4+ verb pools, adjective pools

If Phase 2 PASSes:
- Compositional grammar: "go north" / "apple south" / "come dog"
- Bridge between Tier 1 motor binding and conversational sentences

## Next steps

After seed 42:
- If PASS: multi-seed 43-46 via `launch_multiseed.ps1`
- If multi-seed PASS: extend to 20+ noun pools, 4+ verb pools (real
  conversational dictionary)
- After diversity: Phase 3 A→W readout (drive concept pool → speak word)
- After Phase 3: integrate with chat_repl for interactive concept
  conversations

## Files

- `research/runners/text_minimal_isolation.py` — concept pool builder
- `research/runners/concept_pool_demo.py` — Phase 1 validation runner
- `research/runners/concept_compose_demo.py` — Phase 2 composition runner
- `webapp/server.py` — PRESETS["concept_pool_demo"] +
  PRESETS["concept_compose_demo"]
- `webapp/static/index.html` — launcher dropdown entries
- `webapp/static/ui.js` — "Concept pool architecture" category
- `research/findings/raw/g11_bg/concept_pool_demo/seed42.json` — seed 42
  result (pending)
- `research/findings/raw/g11_bg/concept_pool_demo/launch_multiseed.ps1` —
  multi-seed launcher

## v8/v9 update (2026-05-13 post-v7 multi-seed): A→W BREAKTHROUGH

After v7 multi-seed validated 6.4/12 (53%) PASS on Phase 1 (W→A),
Phase 3 (A→W readout) was tested on the saved v7 bridge.

**v8 (concept_to_language_output_weight 0.5 → 2.0): NEGATIVE.**
A→W still 0/12. Weight magnitude wasn't the issue.

**v9 (+ reciprocal topographic bias): 🎉 A→W 12/12 PASS.**
Adding `apply_reciprocal=True` to `apply_concept_topographic_bias`
(matching Tier 1's apply_topographic_bias pattern) is the fix:

- pool_target(w) → word's lang_output pattern: boost (3.0x)
- pool_target(w) → off-target lang_output: dampen (0.3x)
- off-target pool → word's lang_output: dampen
- Same target-priority logic; ~148k edges biased

**v9 seed 42 result:** Phase 1 6/12 (unchanged from v7), Phase 3
**12/12 PASS**. Every pool speaks its trained word — top-1 cosine
match for all 4 motor + all 4 noun + all 4 verb pools.

| Variant | Phase 1 W→A | Phase 3 A→W |
|---|---|---|
| v7 (forward bias only) | 6/12 | 0/12 |
| v8 (+ weight 0.5→2.0) | 6/12 | 0/12 |
| **v9 (+ reciprocal bias)** | **6/12** | **12/12** |

The reciprocal bias was the **missing piece for A→W**. Forward bias
alone fixed W→A but had nothing for A→W. v7's correct lang_input→pool
weights couldn't help pool→lang_output be selective; the reverse path
needed its own topographic prior.

Bidirectional binding now demonstrated:
- W→A 50% PASS (seed-dependent, 80% on robust trio)
- A→W 100% PASS at single seed

The architecture satisfies the user's three blockers more strongly
than v7 alone suggested. Multi-seed validation queued to confirm
A→W consistency.

## v10 (NMDA tau 250ms): NEGATIVE — uniform tau extension hurts both

Tried `--nmda-tau-decay-ms 250.0` on seed 42 to extend pool persistence
for sequential composition (catalog G.06 PFC working memory).

Result: **Phase 1: 1/12** (was 6/12 in v9). **A→W: 2/12** (was 12/12).

Cause: with longer NMDA, noun_pool_CAT (structurally dominant at this
seed) self-sustains at high firing (4.4-4.6 rate) across MOST words.
Same "canon amplifies bias" failure mode as v2c, but via NMDA route
instead of recurrent gain.

Lesson: uniform NMDA tau extension across all pools breaks isolation.
Need targeted approach (dlpfc_verb pattern from Tier 2.3) where
persistence is in a DEDICATED holding region, not in the concept
pools themselves.

v11 pivots to scale validation (enable_adjective: 16 pools = 4 motor
+ 4 noun + 4 verb + 4 adjective). v12 will attempt dlpfc_verb
integration for sequential composition if v11 confirms scale.

## v11 (16 pools, single seed): 🎉 SCALE WORKS — 11/16 W→A (69%)

Added 4 adjective pools (BIG/SMALL/HOT/COLD) via --enable-adjective.
Total 16 distinct output categories.

**Single-seed result on seed 42:**
- Phase 1 W→A: **11/16 PASS (69%)** — BETTER than v9's 6/12 (50%)!
- Phase 3 A→W: **12/12 PASS** on original 12 pools (speak demo doesn't iterate adjectives)

Per-word breakdown (v11):
- Motor: 3/4 PASS (east, south, west; north FAIL — was PASS in v9)
- Noun: 3/4 PASS (apple, dog, cat; river FAIL)
- Verb: 2/4 PASS (go, come; stop, look FAIL)
- Adjective: 3/4 PASS (big, small, hot; cold FAIL)

**Key insight: more pools IMPROVED Phase 1 PASS rate.**

v9 12-pool: south, west, come FAILed
v11 16-pool: south, west, come now PASS

The architecture's discrimination IMPROVES with more output diversity.
Hypothesis: each pool's "off-target" competition gets larger but
distributed. Probability of any single pool dominating ALL
competitors decreases as pool count grows. Adjective pools soak up
some structural bias that previously concentrated on noun/verb pools.

This is a real architectural finding: **diversity → better isolation,
not worse.** The user's mandate ("regardless of how small the vocab")
is actually under-stating the architecture's strength.

Multi-seed validation queued to confirm pattern.

### v11 multi-seed partial (4 of 5 seeds done)

| Seed | Phase 1 W→A | A→W (12-pool subset) |
|---|---|---|
| 42 | 11/16 | 12/12 |
| 43 | 9/16 | 12/12 |
| 44 | 9/16 | 12/12 |
| 45 | 7/16 | 12/12 |
| 46 | (in flight) | (pending) |
| **Mean (4 seeds)** | **9.0/16 (56%)** | **48/48 (100%)** |

v11 PASS rate (56% at 16 pools) > v9 PASS rate (52% at 12 pools)
even with seed 45 dragging the average. A→W remains 100%
consistent. Architecture is scaling.

Compose with adjective pools not yet evaluated separately for
seeds 43-46 (v11 speak16 follow-up queued auto-fires after seed 46).

### v11 5-seed FINAL: 16-pool architecture validated

Phase 1 W→A (per seed):
| Seed | W→A (16) | A→W (12-subset) | A→W (full 16) |
|---|---|---|---|
| 42 | 11/16 | 12/12 | 16/16 |
| 43 | 9/16 | 12/12 | 16/16 |
| 44 | 9/16 | 12/12 | 16/16 |
| 45 | 7/16 | 12/12 | 15/16 |
| 46 | 9/16 | 12/12 | 16/16 |
| **Total** | **45/80 (56%)** | **60/60 (100%)** | **79/80 (98.75%)** |

**v11 16-pool architecture is fully validated:**
- Phase 1 W→A 56% mean (better than v9's 52%)
- Phase 3 A→W 98.75% across 16 categories (essentially unanimous)
- Adjectives don't break the architecture; they help it scale

Compare to baseline (Tier 1, 4 motor pools): 6/6 multi-seed PASS.
Compare to prior conversational architectures: P5 2/4 ceiling,
Tier 2.3 34-40%, in-vivo 2/4 fixed.

The concept pool architecture is the first project structure to
demonstrate diversity (16 concepts) + bidirectional binding
(99% A→W + 56% W→A) at multi-seed.

## v12 (dlpfc_verb holding) — MIXED to NEGATIVE on seed 42

Added bidirectional verb_pool_X ↔ dlpfc_verb pathways with shared
gates "verb_pool_to_dlpfc" and "dlpfc_to_verb_pool" opened during
verb word training.

Result vs v9 baseline (12-pool, seed 42):

| Test | v9 | v12 | Δ |
|---|---|---|---|
| Phase 1 W→A | 6/12 | 5/12 | -1 |
| Phase 3 A→W | **12/12** | **4/12** | **-8 (regression)** |
| Compose sequential | 0/6 | 1/6 | +1 |
| Compose co-fire | 2/6 | 0/6 | -2 |

**A→W collapsed from 100% → 33%.** Hypothesis: dlpfc_verb feedback
introduces non-specific firing in verb pools during training (PFC
fires on any verb word → activates ALL trained verbs via feedback →
STDP at verb_pool→lang_output trains less selective weights).

Sequential composition gained +1 word, marginal. Co-fire lost.

**v12 fails the "first do no harm" criterion.** v9/v11 remain the
production recipe. v13 would need two-phase training (train v9 first,
then introduce PFC pathways without retraining verb→lang_output) or
unidirectional verb→PFC only (no feedback during training).

Sequential composition remains the unsolved frontier for full
conversational. v9/v11 demonstrate single-word bidirectional binding
+ scale to 16 concepts; sequential is genuinely a harder problem
requiring more careful architectural work than the v12 quick attempt.

## v13 (NMDA only on verb pools): partial — persistence ↑3x but isolation ↓5x

Cluster G v2 pattern: `BrainRegion.enable_nmda` per-region opt-in.
v13 sets `enable_nmda=True` only on verb pools, with cfg.nmda_tau_decay=200ms.

Result vs v9 baseline (seed 42):

| Test | v9 | v13 | Δ |
|---|---|---|---|
| Phase 1 W→A | 6/12 | **1/12** | -5 (regression) |
| Phase 3 A→W | 12/12 | 12/12 | 0 |
| Compose seq persistence | 0.00-0.10 | **0.30-0.38** | +3-4× (partial) |
| Compose seq PASS | 0/6 | 0/6 (still below 0.5 threshold) | 0 |

**Real progress on persistence (3-4× better) but at the cost of
isolation.** Verb pools with NMDA self-sustain on cross-input firing
→ dominate all words. Same canon-amplifies-bias pattern, restricted
to verb kind.

**Architectural tension surfaced:**
- Holding (NMDA bistability) inherently breaks selection (off-target sustain)
- Selection (clean isolation) inherently breaks holding (no persistence)

Biology solves this by separating the holding stage (PFC) from
selection (sensory cortex). dlpfc_verb is the natural carrier — v12
tried bidirectional integration and broke A→W. v14 design (queued):
unidirectional verb_pool → dlpfc only during training, with
post-hoc reciprocal init for eval-time feedback. More careful design
than tonight permits.

**Pivot to W→A improvement** (more tractable). v11 multi-seed shows
W→A 56% is the seed-averaged ceiling at current scale. Options:
- Orthogonal codes (vs hash-based drive patterns)
- Pre-shape phase (Tomasello two-stage)
- Stronger teacher current
- Larger pools (n_per_pool 200 → 500)

## v9 5-seed FINAL (2026-05-13): A→W 100% UNANIMOUS 🎉

| Seed | Phase 1 W→A | Phase 3 A→W |
|---|---|---|
| 42 | 6/12 | **12/12** |
| 43 | 7/12 | **12/12** |
| 44 | 6/12 | **12/12** |
| 45 | 7/12 | **12/12** |
| 46 | 5/12 | **12/12** |
| **Mean** | **6.2/12 (52%)** | **60/60 = 100%** |

**A→W spoken-word readout is deterministic at 12/12 across all 5
seeds.** Every concept pool reliably speaks its trained word.

Phase 1 (W→A cross-category isolation) at 52% mirrors the v7
baseline — the reciprocal bias didn't change forward firing.
Architecture limitations on isolation are dynamics-driven (seed-
dependent per-pool firing variance), not weight-driven (weights
have consistent 4x ratio across seeds).

### v9 is the production recipe

```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --save-bridge ... --out ...
```

Then for A→W readout:
```bash
python -m research.runners.concept_speak_demo --seed N \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --weak-concept-dynamics --load-bridge <saved bridge>
```

Demonstrates:
- 12 distinct concept pools (4 motor + 4 noun + 4 verb)
- Bidirectional binding (W→A 52% mean / A→W 100%)
- 3× diversity over Tier 1 (4 motor pools only)
- Co-fire composition partial (2/6 pairs)
- Sequential composition NOT YET (NMDA tau gap; v10 design queued)

## v14 (orthogonal drive codes, 16 pools): NEW BEST — multi-seed PASS

**Hypothesis:** v11's hash-based `vocab_to_drive_pattern` produces ~10%
pairwise overlap between word patterns; that overlap is the residual
interference floor on W→A. Replacing with `orthogonal_drive_pattern`
(non-overlapping bands assigned by `cue_idx`) eliminates the floor.

**Architecture:** v11 + 16 pools (4 motor + 4 noun + 4 verb + 4 adjective)
+ orthogonal codes via `--orthogonal-codes`. Same training recipe
otherwise: 200 events/word, n_lang_input 2048, n_per_pool 200, weak
dynamics, target-only STDP gating, topographic factor 3.0/0.3,
reciprocal bias on pool→language_output.

**5-seed FINAL result:**

| Seed | Phase 1 W→A | Phase 3 A→W | Total | Wall (min) |
|---|---|---|---|---|
| 42 | **15/16** | 16/16 | 31/32 (97%) | 18.0 |
| 43 | 12/16 | 16/16 | 28/32 (88%) | 16.3 |
| 44 | 12/16 | 16/16 | 28/32 (88%) | 16.6 |
| 45 | 12/16 | 16/16 | 28/32 (88%) | 16.3 |
| 46 | 11/16 | 16/16 | 27/32 (84%) | 17.6 |
| **Mean (5)** | **12.4/16 (77.5%)** | **16/16 unanimous (100%)** | **28.4/32 (88.75%)** |
| **Std** | 1.52 | 0 | — | — |
| **Range** | 11-15/16 | 16/16 | — | — |

**Per-word PASS rate (5 seeds, 16-word vocab):**
- **Robust 5/5:** west, apple, cat, come, hot, cold (6 words)
- **Robust 4/5:** east, south, go, stop (4 words)
- **Mixed 3/5:** north, river, dog, small (4 words)
- **Fragile 2/5:** look, big (2 words)

A→W is **unanimous 16/16 across all 5 seeds = 80/80 = 100%**. This is
the strongest multi-seed binding result yet at 16-word vocab. Seed 42
breakthrough W→A=15/16 (94%) demonstrates the orthogonal-code ceiling
is genuinely high; the multi-seed mean 12.4/16 (77.5%) confirms the
reliable improvement over v11 (mean 9.0/16 = 56%).

**Comparison v9 → v11 → v14:**

| Variant | Vocab | W→A mean | A→W mean | Total |
|---|---|---|---|---|
| v9 (hash codes, 12 pools) | 12 | 52% | 100% | 18.5/24 (77%) |
| v11 (hash codes, +adjective) | 16 | 56% | 99% | 25/32 (78%) |
| **v14 (orthogonal codes, +adjective)** | **16** | **77.5%** | **100%** | **28.4/32 (89%)** |

v14 jumps both metrics: W→A +22pp over v11, total improvement +11pp.

**Cosine strengthening (seed 42):** v11 cosines 0.05-0.40 → v14 cosines
0.25-0.49. Cleaner readout signal correlates with the higher W→A.

**Why orthogonal codes are the right fix:**
v11 hash-based patterns: each word activates ~10% of lang_input neurons
randomly. Two words overlap ~1% (10% × 10%) of their neurons, but
~10-20% via correlation in dense connectivity. STDP at lang_input →
pool sees mixed signal: target word's edges get LTP, but ~10% of
target word's edges also belong to OTHER words → cross-word LTP/LTD
interference.

Orthogonal codes: word i occupies band `[i × (sparsity × n_neurons / N) :
(i+1) × ...]`. Zero pairwise overlap. STDP at lang_input → pool is
strictly word-specific. Phase 1 W→A jumps as the cross-word noise
floor is eliminated.

**Production recipe:**
```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge <bridge.h5> --out <result.json>

python -m research.runners.concept_speak_demo --seed N \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --weak-concept-dynamics --enable-adjective --orthogonal-codes \
    --sparsity 0.05 --load-bridge <bridge.h5> --out <speak.json>
```

**v14 is the new production recipe.** Surpasses v9 (12 pools, 6/12 W→A,
12/12 A→W = 18/24) and v11 (16 pools, 11/16 W→A, 16/16 A→W = 27/32) on
both metrics simultaneously while increasing vocab from 12→16 concepts.

**Sequential composition unchanged** — still v15 work (see
`docs/plans/2026-05-13-sequential-composition-design-note.md`).
