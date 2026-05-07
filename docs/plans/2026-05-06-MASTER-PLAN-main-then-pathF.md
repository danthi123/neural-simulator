# Master Plan — Biology-grounded `main` work + Path F hybrid branch

**Date:** 2026-05-06
**Author:** Claude (autonomous run mandate)
**Status:** AUTHORITATIVE plan for the next 6-10 weeks of work.
This document is the source of truth for the autonomous arc.

---

## Overall goal

Build a spiking neural network simulator that:
1. **Learns continually** — new experiences absorbed online without catastrophic forgetting
2. **Adapts and grows** — capabilities expand naturally with use, not frozen at deploy
3. **Becomes conversational** — even if worse than LLMs, can hold a meaningful exchange
4. **Maintains biology fidelity where it matters** — `main` stays pure biology;
   `path-f-hybrid` branch allows backprop-as-developmental-shortcut for cortex
   pretraining only

The intended distinction:
- **`main`:** all learning mechanisms biology-faithful (STDP, embodied co-firing,
  hippocampal replay, plasticity gates). Weights can be set via STDP only, never
  directly via gradient descent.
- **`path-f-hybrid`:** allows surrogate-gradient backprop ONCE for "developmental"
  cortex pretraining (mirroring real cortex's slow maturational learning).
  All POST-pretraining learning still uses biology-grounded mechanisms
  (continual learning preserved).

## Why this order

Path F's core thesis is that biology-grounded mechanisms can handle ALL
post-pretraining continual learning. **If that thesis is wrong, Path F is
worse than LLMs because we'd just have a less-capable Nord clone.**

Therefore: before adding backprop, we MUST validate that biology-grounded
mechanisms genuinely support continual learning at the scale we need. That
means proving:
1. Multi-modal grounded language (Tier 2.2) — words bind to perceptual concepts
2. Compositional binding (Tier 2.3) — multi-word phrases work
3. Sleep consolidation actually transfers hippocampus → cortex
4. New learning doesn't catastrophically erase old learning

Only after these are validated does Path F's premise hold.

---

# Phase 1 — Biology-grounded `main` work

Total estimated time: **6-9 weeks** of autonomous compute + engineering.

## Phase 1.1 — Tier 2.2: Visual+language binding (~1 week)

### Goal

Word "goal" binds to visual concept of goal during nav. Bidirectional:
- W→I: drive `language_input["goal"]` → `cortex_it` activates spatial pattern
- I→W: agent perceives goal in visual field → `language_output` emits "goal"

This extends the embodied paradigm from text-only (Tier 1+2.1) to multimodal.

### Implementation steps

Concrete plan exists at
`docs/plans/2026-05-06-Tier2.2-implementation-plan-CONCRETE.md`. Key elements:

1. Add `--embodied-language` CLI flag to g11_bg_runner.py
2. During each nav step:
   - When agent executes action `a`: drive
     `language_input[ACTION_TO_WORD[a]]` and `language_output[ACTION_TO_WORD[a]]`
     simultaneously (Tier 1 paradigm)
   - When agent's `cortex_it` fires above threshold (perceiving goal): also drive
     `language_input["goal"]` and `language_output["goal"]`
3. Open language plasticity gates during nav training; freeze for eval
4. Add evals:
   - W→A (already exists in text_eval): drive direction word → motor
   - W→I (NEW): drive "goal" → measure cortex_it pattern, cosine vs perceiving-goal pattern
   - I→W (NEW): agent at goal_pos → read language_output, cosine match to vocab
   - A→W (already exists): port from bio_three_factor inline eval

### Internal debate: thresholds for goal perception

When does the agent "perceive" the goal vs just being in the world?

Options:
- **A)** When goal is within visual field (dist < some radius) — biologically
  reasonable, fires often enough for STDP
- **B)** When `cortex_it` activity exceeds threshold — uses the actual visual
  cortex output, not a heuristic
- **C)** Always (every step) — simple but dilutes the binding signal
- **D)** Only when adjacent to goal (dist <= 1) — clean signal but rare

**Choosing (B)** because it's the most biology-grounded — uses what the network
actually computes from visual input. Threshold tuned empirically (start at mean
+ 1σ of cortex_it firing rate during random walk).

**Would change to (A) if:** `cortex_it` proves too sparse to fire reliably during
nav (would need to crank up visual cortex weights), in which case (A) is
mechanically simpler and biologically defensible (visual attention).

### Pass criteria

6-seed validation. Pass if:
- W→A ≥ 4/6 aligned to NESW (biology-canon Tier 1 baseline)
- W→I cosine > 0.5 in ≥ 4/6 seeds (drive "goal" → cortex_it pattern matches
  perceiving-goal pattern)
- I→W: in ≥ 4/6 seeds, agent at goal_pos → language_output top-1 = "goal"
- 32×32 nav performance preserved within 1σ of baseline (2.57 ± 0.11)

### Risks + mitigation

| Risk | Mitigation |
|---|---|
| Visual cortex + BG cascade dynamics interfere with language STDP | Strong language drive (200 pA); tune relative to other inputs |
| Goal perception trigger too narrow → too few binding events | Start with cortex_it firing-rate threshold (option B); fall back to dist-based (A) if sparse |
| Reward-driven plasticity conflicts with embodied teachers | Tune embodied teacher pA; freeze reward modulation during embodied phases if needed |
| Nav performance regresses | Monitor 32×32 baseline as control; abort if mean degrades by >0.5 |

### Validation order

1. Smoke test (1 seed, 600 nav steps): does it run cleanly? Do language regions
   actually fire? Does training complete?
2. 3-seed mid validation: do the new evals show signal?
3. 6-seed final validation: pass criteria above

### Deliverables

- Updated g11_bg_runner.py with `--embodied-language` flag
- 6-seed validation YAML
- Findings doc: `2026-05-XX-Tier2.2-visual-language-binding-results.md`
- CLAUDE.md update if successful

---

## Phase 1.2 — Tier 2.3: Compositional 2-word phrases (~2-3 weeks)

### Goal

Agent processes "go north" as a compositional unit, not as separate "go" and
"north." PFC working memory holds verb context (~500ms) while the next word
parses; combined activity gates motor selection.

### Implementation steps

Concrete plan exists at `docs/plans/2026-05-06-Tier2.3-two-word-phrases-design.md`.

1. **Add PFC verb pool** as new BrainRegion:
   - 200 neurons, NMDA-bistable for ~500ms persistent activity
   - High recurrence (internal_density 0.15)
2. **Pathway: language_input → dlpfc_verb** (verb words activate pool)
3. **Action gate via neuromodulator:**
   - PFC verb pool activity drives an `action_gate` modulator
   - Modulator increases motor pool excitability when verb context active
   - Without verb context, motor pools are inhibited (won't act on direction alone)
4. **Training curriculum (3 trial types):**
   - 60%: phrase trials ("go" + direction → motor target)
   - 30%: direction-only trials (Tier 1 backward compat)
   - 10%: verb-only trials (verb alone → no motor action)
5. **Add evals:**
   - Phrase eval: "go north" → motor_N? (composition test)
   - Direction-only: "north" alone → motor_N? (Tier 1 compat)
   - Verb-only: "go" alone → motor pools quiet? (anti-action)

### Internal debate: how to gate motor selection

Three options for the verb-context-required-for-action mechanism:

- **A)** Neuromodulator `action_gate` (excitability_drive scope=group:motor_X) —
  PFC activity boosts motor pool excitability
- **B)** Inhibitory PFC → motor pathway (inhibits motor unless PFC also gates
  release) — more like real BG indirect pathway
- **C)** PFC → striatum → BG cascade modulation — most biologically accurate
  but most complex

**Choosing (A)** for first pass. Cheapest implementation, uses existing
neuromodulator subsystem, biologically defensible (PFC excitability modulation
of motor cortex is well-attested).

**Would change to (B) or (C) if:** (A) doesn't produce clean verb-only
inhibition. Real biology has multiple gating mechanisms; if neuromodulator
alone is too coarse, add inhibitory pathway.

### Internal debate: word order

Real biology processes "go north" left-to-right. Two options:

- **A)** Sequential drive: drive "go" for 100ms, then drive "north" for 100ms,
  with PFC NMDA holding "go" across the gap
- **B)** Simultaneous drive: drive both at once, let recurrence sort out

**Choosing (A)** because it matches real auditory processing temporal structure
and tests PFC's bistability claim (which is the whole point of Tier 2.3).

### Pass criteria

6-seed validation. Pass if all three conditions held:
- Phrase ≥ 4/6 seeds correctly execute "go [direction]" → motor target
- Direction-only ≥ 4/6 seeds preserve Tier 1 behavior
- Verb-only ≥ 4/6 seeds keep motor pools quiet

### Risks + mitigation

| Risk | Mitigation |
|---|---|
| PFC NMDA bistability not tuned for 500ms | Sweep `nmda_tau_ms` in [200, 500, 1000]; pick best |
| Action gate neuromodulator too coarse | Fall back to inhibitory pathway (option B) |
| Phrase trials hurt direction-only performance | Adjust trial mix proportions; ensure direction-only baseline preserved |

### Validation order

1. Build PFC verb pool + connectivity. Smoke: drive "go", verify PFC pool fires + persists ~500ms
2. Add action gate neuromodulator. Smoke: PFC active → motor excitability up; PFC quiet → down
3. Train with 3 trial types (single seed). Smoke: do all three eval modes show signal?
4. 6-seed validation

### Deliverables

- New `dlpfc_verb` region builder
- `--enable-tier-2-3` flag with all infrastructure
- 6-seed validation YAML
- Findings doc

---

## Phase 1.3 — Hippocampus → cortex consolidation closed loop (~1-2 weeks)

### Goal

After learning new word-concept bindings (Tier 1/2.1/2.2), a "sleep" phase
should transfer those bindings from hippocampus to cortex. After sleep,
hippocampus can be cleared and cortex retains the binding (Buzsaki/McClelland
two-stage memory model).

### Why this is critical

Without consolidation, all learning lives in hippocampus, which has limited
capacity. Real animals consolidate during NREM sleep via SWR replay. This is
THE mechanism that makes continual learning possible without catastrophic
forgetting at scale.

### Implementation steps

We have Cluster D v2 (DG/CA3/CA1 + SWR-gated CA3 plasticity). What we DON'T
have: the closed loop where hippocampal patterns drive cortex during sleep.

1. **Identify "what's in hippocampus":** During training, mark patterns that
   were stored in CA3 (high recent firing). Capture as templates for replay.
2. **SWR replay generator:** During quiet phases (no input), drive CA3
   sub-populations (templates) at SWR rate (~150 Hz, 100ms bursts).
3. **CA1 → cortex pathway:** ensure hippocampus output projects to relevant
   cortex regions (motor cortex, visual cortex, language cortex). Add
   pathways if missing.
4. **Cortex plasticity during sleep:** open cortex plasticity gates during
   replay, freeze hippocampus internal weights. STDP at hippo → cortex
   synapses transfers the pattern.
5. **Eval:** train new pattern → sleep → erase hippocampus → does cortex
   still produce the pattern when tested?

### Internal debate: replay schedule

How often / how long should replay happen?

- **A)** Continuous: replay throughout training (interleaved)
- **B)** Periodic: every N training events, do M sleep events
- **C)** Post-training only: train fully, then enter sleep mode for K events
- **D)** Triggered by reward: after success, replay the trajectory (Foster &
  Wilson 2006)

**Choosing (C) for first pass + (D) as enhancement.** (C) is simplest to
validate the consolidation mechanism. (D) adds reward-anchored prioritization
later.

### Pass criteria

- Train word-action binding (Tier 1 paradigm) → 100% accuracy
- Run sleep phase (200 SWR replay events)
- Erase hippocampus state (or freeze hippo plasticity, drive cortex only)
- Test: drive language_input → does motor pool still fire correctly?
- Pass if ≥ 4/6 seeds show preserved binding after sleep + hippo erase

### Risks + mitigation

| Risk | Mitigation |
|---|---|
| Replay doesn't drive cortex strongly enough | Increase replay drive amplitude; tune CA1→cortex weights |
| Cortex develops conflicting patterns | Use pattern separation in DG to prevent interference |
| Sleep phase erases existing cortex knowledge | Freeze cortex internal recurrence during sleep; only allow hippo→cortex updates |

### Deliverables

- SWR replay scheduler in g11_bg_runner.py
- CA1 → cortex pathways
- Sleep-phase plasticity gating
- New eval: post-sleep retention test
- Findings doc

---

## Phase 1.4 — Catastrophic forgetting eval suite (~3-5 days)

### Goal

Quantify whether biology-grounded learning catastrophically forgets when new
information is added. THE foundational test for Path F's premise.

### Test design

**Sequential vocabulary expansion:**
1. Train Tier 1 (4 words: north/east/south/west). Validate aligned. Save state.
2. Continue training with 4 NEW words (not synonyms — different actions/concepts,
   e.g., specific object names). Validate.
3. Re-test ORIGINAL 4 words. Did they degrade?
4. Repeat for 12, 16 words.
5. Plot retention curves.

**Interference test:**
1. Train all 8 words with random interleaved presentation
2. Compare to A→B sequential ordering
3. Does interleaved preserve all? Does sequential cause forgetting?

**Long-tail test:**
1. Train 8 words with imbalanced frequency (some 200x more common than others)
2. Do rare words still get learned? Do they survive consolidation?

### Implementation

- Add `--continual-eval-mode` flag that runs the sequential expansion automatically
- Each phase: train, eval-on-current, eval-on-prior, save curves
- Generate retention curve plot (matplotlib output)

### Pass criteria

For Path F's premise to hold, we need:
- After learning vocab B (new), vocab A retention ≥ 50% of original
- After consolidation, vocab A retention ≥ 80%
- Long-tail words: learned at ≥ 30% accuracy (vs 25% chance baseline)

### Risks + mitigation

| Risk | Mitigation |
|---|---|
| Catastrophic forgetting really does happen | If yes, Path F's premise needs adjustment (add EWC?). If severe, biology-grounded continual learning fails — important negative finding |
| Test methodology flawed | Cross-check with permuted-label control on each retention test |

### Deliverables

- Continual eval runner mode
- Retention curve plotting
- Findings doc with quantitative forgetting curves
- Decision: if catastrophic forgetting found, add Phase 1.4b (EWC mitigation)
  before Phase 2

---

## Phase 1.5 — Continual-learning eval suite (~3-5 days)

### Goal

Unified benchmark suite for continual learning, used as regression check during
Phase 2 development.

### Components

1. **Sequential learning:** Phase 1.4 sequential expansion test
2. **Interference test:** Phase 1.4 interleaved vs sequential
3. **Long-tail test:** Phase 1.4 imbalanced frequency
4. **Retention over time:** Train, wait N silence steps, retest
5. **Multi-modality interaction:** Train word→motor (Tier 1), word→percept
   (Tier 2.2), do both still work?
6. **Compositional preservation:** Train phrases (Tier 2.3), do single-word
   bindings still work?

### Implementation

`research/runners/continual_eval.py` — single runner that executes all 6
benchmarks against a model checkpoint.

### Pass criteria

Each test produces a retention score [0, 1]. Aggregate score (mean) ≥ 0.7
for "biology-grounded continual learning works" claim.

### Deliverables

- continual_eval.py
- Standard benchmark output format
- Baseline measurement against current main branch state
- Findings doc

---

## Phase 1 Summary

Total: ~6-9 weeks autonomous work

After Phase 1:
- Multimodal grounded language (Tier 2.2)
- Compositional binding (Tier 2.3)
- Working sleep consolidation
- Quantified continual learning capability
- Validated Path F's premise

If Phase 1.4 reveals catastrophic forgetting:
- Insert Phase 1.4b: investigate biology-grounded mitigations
  (heterosynaptic LTD, replay-based interleaving, slower learning rates)
- May extend Phase 1 by 1-2 weeks
- If unfixable with biology-grounded methods, Path F needs EWC or similar
  (which we'd add even though it's "non-strictly biological" — this exception
  would be discussed with user before implementing)

---

# Phase 2 — `path-f-hybrid` branch

Total estimated time: **6-10 weeks** (begins only after Phase 1 complete).

## Phase 2.1 — Branch setup + surrogate-grad backprop mode (~1-2 weeks)

### Goal

Add the ability to train cortex regions via surrogate-gradient backprop on text
sequences. NOT enabled on main; only available on `path-f-hybrid`.

### Implementation

1. Create branch: `git checkout -b path-f-hybrid`
2. Add `cfg.enable_surrogate_grad_pretraining: bool = False` to CoreSimConfig
3. Implement surrogate-gradient framework:
   - ATan or Heaviside surrogate (Zenke 2018, used by SuperSpike + Nord)
   - Backward pass through spike sequences
   - Per-cortex-region pretraining loop
4. Reference Project Nord's `nord_core.py` for the gradient implementation
5. Test: small SNN trains on a toy sequence task (predict next token in
   "ABCABC...")

### Pass criteria

- Surrogate-grad trains a 1-layer SNN to >80% accuracy on toy sequence task
- Existing main-branch tests still pass (gated by config flag)
- No interference with main-branch biology-grounded paths

### Deliverables

- Branch with surrogate-grad infrastructure
- Test suite for the new gradient code
- Toy task demonstration

---

## Phase 2.2 — Cortex pretraining on small corpus (~2-3 weeks)

### Goal

Pretrain the language cortex regions (cortex_it, language regions, motor
cortex word-vocabulary mapping) on a real text corpus. Get to "GPT-2-ish but
SNN" baseline.

### Corpus selection

- **Tiny Shakespeare** (~1MB) for first proof — fast iteration
- **FineWeb-Edu sample (~100MB)** for next pass — Project Nord's corpus
- **Full FineWeb-Edu (~1TB)** for final pass — only if 100MB version works

Start with Tiny Shakespeare for the same reason Project Nord did — proves the
approach works at small scale before committing to large compute.

### Architecture

Map our brain regions onto Project Nord's zonal structure:
- `language_input` ← Nord "Sensory Zone" (text input encoding)
- `cortex_it` ← Nord "Association Zone" (semantic/lexical)
- `dlpfc_wm` ← Nord "Memory Cortex" (working memory)
- `cortex_motor_X` + `language_output` ← Nord "Executive Zone"

Adopt Nord's:
- AssociativeLIF neurons (vs our Izhikevich; AssociativeLIF is purpose-built
  for backprop training)
- Multi-scale temporal (T_fast=8 + T_slow=2)
- EMA temporal readout
- Spike-Driven MoE (optional, for capacity)

Keep our:
- Brain region framework
- BG cascade for response selection
- Hippocampus (Cluster D v2)
- PFC working memory

### Pretraining schedule

- Tiny Shakespeare: 5K-10K steps, 4-8 hours on local 3090
- FineWeb-Edu 100MB: 20K-50K steps, ~2-3 days on local 3090 (or hours on H100)
- Full FineWeb-Edu: only if needed; cloud H100 ~ $300-500

### Pass criteria

- Loss < 5.0 on validation (Project Nord hit 4.4 after 27K steps)
- Sample generation produces coherent grammatical text (not random)
- Can answer simple questions ("what color is grass?" — green, etc.)

---

## Phase 2.3 — Wire continual-learning loop (~1-2 weeks)

### Goal

Connect the pretrained cortex to the biology-grounded hippocampus + PFC + BG
machinery. New experiences absorbed via Tier 1+2.2 paradigm continue to update
hippocampus; sleep consolidation transfers to cortex via Phase 1.3 mechanism.

### Implementation

1. After pretraining completes, freeze cortex weights at the gross level but
   keep STDP active for fine-tuning
2. New experiences during deployment:
   - User input → language_input (frozen pretrained processing)
   - Hippocampus binds new context (Tier 1+2.2 paradigm)
   - PFC holds dialog state
   - BG selects responses based on combined cortex + hippocampus + PFC
   - During quiet periods, SWR-driven replay consolidates hippocampus →
     cortex (Phase 1.3 mechanism)
3. Expose a chat interface (`chat_pathf.py`) that:
   - Takes user input
   - Drives language_input
   - Reads language_output
   - Decodes via cosine match (or Nord-style EMA readout)

### Pass criteria

- After pretraining, ask "what color is grass?" → "green" (or similar)
- During chat, tell sim "Daisy is my dog" → 800 STDP exposures
- 1 day later (or 200 sleep events), ask "what is Daisy?" → "dog" (consolidated
  via Phase 1.3 mechanism)
- Tier 1 word-action bindings still work (regression check)

---

## Phase 2.4 — First conversational demo (~1 week)

### Goal

End-to-end conversation demo showing continual learning.

### Demo script

```
User: Hello, what's your name?
Sim: [grammatical response, possibly arbitrary name based on training]

User: I'll call you Spike. Spike, what color is grass?
Sim: green.

User: Good. Spike, my dog's name is Daisy.
Sim: ok / nice / acknowledgment.

[Wait for sleep consolidation, e.g., 2 minutes of quiet]

User: Spike, what is Daisy?
Sim: dog.

User: What's my dog's name?
Sim: Daisy.

[Test that pretraining knowledge survives]
User: What color is grass?
Sim: green.
```

If this works, Path F is validated.

### Pass criteria

- Sim responds grammatically (not random tokens)
- Acquired facts ("Daisy is my dog") survive sleep + are retrievable
- Pretrained knowledge (basic facts) preserved after acquiring new facts
- Holds 5+ turn conversation without complete coherence loss

### Deliverables

- chat_pathf.py interactive script
- Findings doc with full transcript
- Project status update / wiki sync

---

# Phase 3 — Forward planning (post-Phase 2)

If Phase 2 succeeds:
- **Validate at scale** — train on FineWeb-Edu full, target Project Nord
  capability levels
- **Tier 4** — sentence-level processing (subject-verb-object)
- **Tier 5** — multi-turn dialog with persistent personality

If Phase 2 fails (cortex doesn't pretrain to useful baseline OR continual
learning loop breaks pretrained knowledge):
- Investigate failure mode
- Either fix or accept Path A pure biology trajectory

---

# Engineering practices throughout

## Branch policy
- All Phase 1 work on `main`
- Phase 2 work on `path-f-hybrid` only
- Periodic merges from main → path-f-hybrid (one-way) to keep biology
  improvements
- NEVER merge path-f-hybrid → main (would pollute pure biology line)

## Commit cadence
- Every meaningful chunk → commit + push to both remotes (github + gitea)
- After each phase deliverable → wiki-sync
- Findings docs alongside code commits

## Validation gating
- No phase advances until pass criteria met or explicit decision-with-debate
  to proceed despite incomplete result
- Each phase has its own validation; phases don't depend on subsequent phases'
  validation

## Internal debate documentation
- Every non-trivial decision documented inline in design docs
- Pros/cons explicit
- "Would change if X" criteria explicit

## Anti-cheat discipline
- Permuted-label control on all alignment claims
- Sleep consolidation tested with fresh seeds (not cherry-picked)
- Continual learning eval as regression check after every phase

---

# Decision log

## 2026-05-06 ~17:30 EDT — Tier 2.2 v1 disrupted nav

**Issue:** First Tier 2.2 implementation drove language regions
every step at 200pA. At 16×16 grid, nav degraded from baseline
1.03 to 6.50 (6× worse), at_goal dropped from 695 to 1/1800 steps.

**Diagnosis:** Language drive too strong + too frequent. Language
input → cortex_X → motor pathway propagated language activity
into motor pools, distorting BG cascade action selection.

**Decision:** Reduce drive amplitude and frequency.
- 200pA → 80pA (supplements rather than dominates)
- Every step → every 5 steps (sporadic, biologically plausible)

**Biological justification:** Real language acquisition is
EPISODIC, not continuous. Children hear words paired with
specific actions/scenes occasionally, not at every microsecond
of motor command. Pulvermüller's framework specifies co-firing
at MOMENTS of word-action coincidence.

**Outcome:** v2 preserved nav (1.05 vs 1.03 baseline) but W→A
13%, I→W 25% with strong directional bias (all words → one
wrong direction).

## 2026-05-06 ~17:55 EDT — Tier 2.2 v3: warmup gate

**Issue:** v2 with sporadic drive preserved nav but bound
language to wrong actions.

**Diagnosis:** Random-walk phase (early nav, before agent
learns to navigate) was binding language to random actions.
~360 sporadic events in random walk dominated the bindings
even after agent learned competent nav.

**Decision:** Add `--embodied-language-warmup-steps 600`.
Skip embodied-language until nav has converged. Mirrors child
language acquisition: words heard during INTENTIONAL action,
not random flailing.

**Biological justification:** Children don't learn "go" until
they can already locomote intentionally. Vocabulary acquisition
follows competent action production.

**Outcome:** v3 preserved nav (1.02), W→A 26% (single seed),
"west" → W bound perfectly (25/25). Other directions still
biased. Single seed insufficient to evaluate.

**Next:** 6-seed validation at 32×32 (validated baseline) to
see if pattern holds across seeds.

## 2026-05-06 ~18:20 EDT — Phase 1.3 scope reduction

**Discovery:** Sleep replay infrastructure already exists in
g11_bg_runner (`--sleep-replay-after-step`, `--sleep-replay-steps`).
Cluster D v2 SWR-gated plasticity already shipped.

**Decision:** Phase 1.3 budget reduces from "1-2 weeks
implementation" to "primarily validation + minor wiring."
The infrastructure for hippocampus → cortex consolidation
exists; need to:
1. Validate it actually transfers patterns (Phase 1.3 main task)
2. Add post-sleep retention test
3. Wire it into Phase 1.4 catastrophic forgetting eval

This shrinks Phase 1.3 from 1-2 weeks to ~1 week.

## 2026-05-06 ~20:30 EDT — Phase 1.4 smoke launch + parallel pre-staging

**Status:** Phase 1.4 catastrophic forgetting smoke launched
(PID 33756, ETA ~21:25 EDT). Used parallel time during the
~55min wait to pre-stage downstream work.

**Pre-staged artifacts (all committed to main):**

1. `research/runners/continual_eval_suite.py` -- Phase 1.5
   unified eval dispatcher. 4 of 6 benchmarks live:
   - sequential_expansion (wraps Phase 1.4 logic)
   - retention_over_time (silent-step retention with frozen
     plasticity to isolate passive retention)
   - interference (interleaved 8-word vocab)
   - long_tail (4 common + 4 rare words at 20:1 frequency
     ratio)
   - multimodality (stub, depends on Tier 2.2)
   - composition (stub, depends on Tier 2.3)

2. `experiments/continual_forgetting_validation.yaml` --
   Phase 1.4 6-seed validation YAML, ready to launch when
   smoke confirms protocol.

3. `experiments/continual_eval_suite.yaml` -- Phase 1.5
   4-benchmark suite x 6 seeds.

4. `docs/plans/2026-05-06-Phase-1.4-decision-tree.md` --
   pre-codified next-step branching based on smoke result
   (Branch A >=80%, B 50-80%, C <50%, D crash). Defines
   default action if user unavailable.

5. `docs/plans/2026-05-06-Phase-1.3-consolidation-design.md`
   -- detailed Phase 1.3 implementation design. Reuses
   Cluster D infrastructure; adds ca1->motor and
   ca1->language_output consolidation pathways. Sleep loop
   alternates encoding (awake) and consolidation (sleep)
   plasticity gates. Three eval modes specified: standard
   W->A, hippo-OFF (consolidation proof), sleep-recovery.

6. `docs/plans/2026-05-06-Phase-2.1-surrogate-grad-design.md`
   -- detailed Phase 2.1 design for path-f-hybrid branch.
   Manual BPTT with ATan surrogate gradient (Zenke 2018),
   T=20 unrolled timesteps, 4-layer SNN cortex stack. Toy
   task: predict next token in ABCABC... sequence.

**Decision:** Pre-staging everything makes the autonomous
arc more robust to context switching and provides clear
go-paths regardless of Phase 1.4 outcome. The decision tree
ensures progression continues without user intervention.

**Next:** wait for Phase 1.4 smoke result, follow decision
tree to next phase.

## 2026-05-06 ~21:30-22:10 EDT -- Phase 1.4 smoke iteration v1->v2->v3

Three smoke runs needed to nail down the correct config:

**v1 (scale-up arch):** Phase A primary W->A 14% -- BELOW 25% chance.
Architecture mismatch: scale-up (4096/1000/120) was for 8-word
synonyms; with 4-word vocab, motor pools didn't get enough
discriminative training. Killed at 21:33.
KEY POSITIVE despite the bug: primary retention went from 14%
post-A to 18% post-B (129% retention ratio). Synonym training
actually IMPROVED primary accuracy. No catastrophic forgetting
visible at v1.

**v2 (standard arch, no NMDA):** Phase A primary W->A 25%
(exactly chance). Better than v1 but still ~8pp below validated
Tier 1 baseline of 33%. Diagnosis: BREAKTHROUGH config used
enable_nmda=True; my call was missing it (defaulted to False).
NMDA bistability is the critical training mechanism for embodied
Hebbian binding -- without it motor pools don't develop
attractor dynamics. Killed at 22:00.

**v3 (standard arch + NMDA):** in flight (PID 27368, started
22:00:58). Expect Phase A ~33-45% per Tier 1 baseline.

**Pre-staged during wait (parallel work):**
- Phase A sanity check (auto-abort if below chance, saves 30 min)
- Phase 1.5 dispatcher unit tests (6 tests passing)
- Tier 2.3 PFC verb pool builder + 7 unit tests (opt-in)
- Tier 2.3 action_gate neuromodulator helper + 3 unit tests
- forgetting_summarize tool + 6 unit tests
- Master plan decision log updates (this entry)

**Lesson:** copy ALL config from validated baseline JSON. NMDA
looked like an architectural detail but was the critical
training mechanism. Future continual-learning tests should
follow validated baselines exactly, then deviate only with
clear rationale.

**Next:** wait for v3 Phase A eval result. If >= 33%, launch
6-seed validation. If still 25%, deeper investigation needed
(maybe topographic_bias_factor or similar).
