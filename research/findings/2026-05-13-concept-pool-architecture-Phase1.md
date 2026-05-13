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
        × 4 pools       × 4 pools           × 2 pools
        (FS within)     (FS within)         (FS within)
                    ↘           ↓           ↙
                    language_output (4096)
```

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
