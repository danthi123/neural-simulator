# Biologically-grounded user↔sim language communication — 3-tier plan

**Date:** 2026-05-05
**Status:** Tier 1 design + implementation kicking off this session.
Tier 2/3 designs scoped; implementation conditional on Tier 1 success.

**Core insight (from today's verdict):** Today's W→A verdict showed
3-factor reward-based learning fails at biological scale, but
SUPERVISED GRADIENT under the same biology canon got 3/3 NESW aligned.
The architecture works. The task design (flashcard cue→action with
scalar reward) is the wrong paradigm. Real animal language acquisition
uses **embodied Hebbian co-firing during shared experience** —
no scalar feedback required.

This plan implements the embodied paradigm at three increasing levels
of capability, all within biological accuracy.

---

## Tier 1 — Single-word embodied binding (~2 weeks)

**Goal:** Bidirectional 4-word binding: `north/east/south/west` ↔
motor execution. User types a word → agent moves; agent moves → user
sees the word.

### Mechanism

**Embodied Hebbian co-firing during action execution**:
1. Drive `language_input` with word "north" sparsely
2. Drive `language_output` with same word sparsely (TEACHER signal)
3. Drive `motor_N` with elevated current (TEACHER signal — "the parent
   demonstrates moving north while saying the word")
4. Forward-propagate for `stim_steps_per_event`
5. STDP at all 3 co-active sites strengthens connections:
   - `language_input → motor_X` (input understanding)
   - `motor_X → language_output` (NEW reciprocal pathway, output gen)
   - `language_input ↔ language_output` (auto-association via shared
     teacher)
6. After N trials per word, freeze cross-projections (critical period)

**No scalar reward.** Co-activity IS the teacher signal. Matches
Pulvermüller 2001-2012 somatotopic semantic theory.

### Architecture changes

```python
# Add reciprocal pathway to build_biological_brain_regions:
for action in ['N', 'E', 'S', 'W']:
    pathways.append(RegionPathway(
        from_region=f"motor_{action}",
        to_region="language_output",
        density=0.3,
        weight_mean=2.0,
        weight_jitter=0.5,
        plastic=True,
        plasticity_gate="motor_to_language_output",  # new gate name
    ))
```

Biological basis: Felleman & Van Essen 1991 reciprocal cortical
connectivity; Broca's area interleaved with premotor cortex (Pulvermüller
2003).

### CLI changes

```python
ap.add_argument("--embodied-hebbian", action="store_true",
                help="Tier 1: embodied Hebbian word↔motor binding "
                     "(no scalar reward; co-activity teaches).")
ap.add_argument("--embodied-motor-teacher-pA", type=float, default=300.0,
                help="External current pushed to target motor pool "
                     "during embodied trial. Default 300pA = strong "
                     "teacher.")
ap.add_argument("--embodied-trials-per-word", type=int, default=200,
                help="Embodied training trials per word (default 200; "
                     "x4 words = 800 total trials).")
```

### Training loop changes

Replace the 3-factor reward update with:

```python
def run_embodied_hebbian_trial(bridge, token, target_action, ...):
    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()

    # SIMULTANEOUS teacher signals on 3 sites
    drive = vocab_to_drive_pattern(token, ...)
    bridge.cp_external_input_current[lang_input_idx] = drive  # input
    bridge.cp_external_input_current[lang_output_idx] = drive  # output (teacher)
    motor_target_idx = motor_idx[target_action]
    bridge.cp_external_input_current[motor_target_idx] += teacher_pA  # motor (teacher)

    # Forward — STDP fires in all co-active synapses
    for _ in range(stim_steps_per_event):
        bridge._run_one_simulation_step()
```

### Tests

After training, freeze plasticity. Two evals:

1. **Word → Action**: drive only `language_input` with word, measure
   motor pool activations. Expected: motor_target dominates.
   - Permuted-label control: 6/6 aligned to NESW (vs today's 1/6
     with 3-factor)

2. **Action → Word**: drive only motor pool, read `language_output`
   activity → cosine match to vocab. Expected: correct word top-1.
   - Permuted-label control: 6/6 aligned

### Pass criterion

Either direction at >=4/6 permuted-label aligned. Both at 6/6 = full
success.

### Tier 1 risks

- **Reciprocal pathway might cause runaway feedback** during training
  (motor → language_output → ... → motor loop). Mitigation: gate
  pathway weights to small mean (2.0); freeze gate during eval.
- **Topographic prior might not be needed or might over-help.**
  Test with and without; biology canon defaults pre-bias N→N etc.
- **Inter-trial bleed**: if rest_steps too short, NMDA carryover
  contaminates. Already handled in eval (n_reset_steps=100).

---

## Tier 2 — Compositional understanding (~1 month after Tier 1)

**Goal:** Two-word phrases. "Go to goal" → goal-directed behavior.
~20-30 word vocabulary. Object references, verbs, simple adjectives.

### New mechanisms

1. **Object-word binding**: bind "goal" to `goal_cells` representation
   via co-firing during goal exposure (similar to motor binding but
   for perceptual concepts).
2. **Visual binding**: word "red" binds to specific Gabor/color
   features in visual cortex. Need color channel in retina (currently
   only ON/OFF).
3. **Phrase parsing**: maintain context across timesteps via PFC
   working memory. Two-word command "go" + "north" requires holding
   "go" while parsing "north."
4. **Semantic composition**: PFC binds (verb, object) pair → produces
   action plan.

### Architecture additions

- **Color retina channels** (3 channels: R, G, B in addition to ON/OFF)
- **Persistent language buffer** in PFC: holds recent word activations
  for ~500ms working memory window (NMDA-tau aligned)
- **Verb/object segregation**: distinct sub-pools in language_input
  for verbs vs nouns vs adjectives. Hebbian over time clusters them
  by usage statistics.

### Training paradigm

Embodied scenarios:
- "go" + walking → bind verb to action sequence
- "stop" + halt → bind verb to inhibition
- "goal" + see-goal → bind noun to perceived object
- "red" + see-red → bind adjective to feature

Then composition trials: "go to goal" presented during goal-directed
navigation → PFC binding emerges.

### Tier 2 risks

- **Working memory capacity**: PFC NMDA bistability needs tuning to
  hold word patterns for 500-1000ms across word sequences.
- **Compositional generalization**: bound objects might not combine
  with novel verbs (binding problem). May need dendritic learning
  for compositional generalization.

### Tier 2 abort condition

If after 4 weeks of work, two-word phrase comprehension fails
permuted-label control on novel combinations, abort and either
go to Tier 3 (dendritic) or accept Tier 1 ceiling.

---

## Tier 3 — Genuine compositional semantics (~2-3 months after Tier 2)

**Goal:** Sentence-level understanding. Subject-verb-object. Abstract
concepts (happy, tired, hungry). Multi-step plans from language input.

### Required infrastructure

This is where Tier 1+2 hit their ceiling. Compositional generalization
requires per-region error signals — i.e., dendritic learning OR
predictive coding.

**Option 3A: Dendritic learning (Bono & Clopath 2017)**
- Multi-compartment Izhikevich neurons (V_basal, V_apical, V_soma)
- Apical compartment receives top-down feedback from "expectation"
- Plasticity at basal synapses gated by apical activity
- 1.5-2 mo implementation cost
- See `docs/plans/2026-05-05-dendritic-learning-design.md`

**Option 3B: Predictive coding (Rao & Ballard 1999)**
- Each region predicts inputs from internal state
- Errors propagate as signals
- Mathematically equivalent to backprop under certain conditions
- 2-3 mo, more architectural changes

### Tier 3 capabilities

After completion:
- Parse "go to the red square" — subject (agent), verb (go),
  prepositional phrase (to red square)
- Bind abstract concepts: "happy" → high-reward state activation
- Multi-step plans: "go up then left" → execute sequence
- Question answering: "where are you?" → produce position description

### Tier 3 risks

- Architectural complexity may break existing 32×32 navigation result
- Multi-compartment kernels are 3-4× slower per neuron
- Need to validate every existing test still passes

---

## Sequencing strategy

```
Tier 1 (2 weeks)
   ├── Pass: ≥4/6 aligned bidirectional
   │   └── Move to Tier 2 (1 month)
   │       ├── Pass: 2-word phrases work
   │       │   └── Move to Tier 3 (2-3 months, dendritic)
   │       └── Fail: stop at Tier 2 ceiling, keep Tier 1
   └── Fail: investigate (likely needs Tier 3-style fix immediately)
```

Each tier has a clear pass/fail signal via permuted-label control.
No tier locks in the next; user can stop at any tier.

## Cost summary

| Tier | Time | Risk | Capability |
|---|---|---|---|
| Tier 1 | 2 weeks | Low | Bidirectional 4-word↔motor |
| Tier 2 | 1 month | Medium | Two-word phrases, ~20-30 words |
| Tier 3 | 2-3 months | Higher | Sentences, abstract concepts |

**Total to full Tier 3:** ~4-5 months focused work. **Tier 1 alone:**
2 weeks for real but limited user↔sim communication.

## Why this beats the alternatives

vs. **dendritic learning for W→A flashcards** (today's recommended
fix for the verdict):
- Tier 1 takes 2 weeks vs 1.5-2 months
- Same architecture (no rewrite)
- Uses what already works (Hebbian + STDP at biology canon)
- Hits 80% of practical user↔sim utility immediately

vs. **scaling-only direction** (32×32 → 64×64 etc):
- 32×32 already at peak; 64×64 hits phase budget ceiling
- Adding embodied language is orthogonal — doesn't compete with
  scaling axes (obstacles, real visual, etc)
- Different dimension of progress

vs. **pivoting away from text I/O entirely**:
- This plan IS the pivot — but to a working biological paradigm
  rather than abandoning the goal
- W→A flashcard task is dead; embodied co-firing is alive

## Implementation starting now

Tier 1 implementation begins in this session:

1. Add `motor_X → language_output` reciprocal pathway to
   `build_biological_brain_regions` in
   `research/runners/text_minimal_isolation.py`
2. Add `--embodied-hebbian` flag and training loop to
   `research/runners/bio_three_factor.py` (or new runner if cleaner)
3. Smoke test 1 seed (~10 min)
4. 6-seed validation if smoke passes (~30 min at parallel=3)
5. Permuted-label control on results
6. Findings doc + Tier 2 decision

Pass criterion for tonight: bidirectional aligned ≥4/6 in either
direction. Stretch: 6/6 in both.
