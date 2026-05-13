# Concept pool architecture — Phase 1: diversity beyond 4 motor pools

**Date:** 2026-05-13
**Status:** v1 architecture SHIPPED + seed 42 v1 FAIL (0/10) + diagnosis +
v2 fix LAUNCHED (4 verb pools, tighter topographic prior).

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
