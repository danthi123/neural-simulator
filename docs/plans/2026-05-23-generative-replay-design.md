# Generative replay: design — the hippocampal-prefrontal replay loop with mode-unification, the third leg of the conversational-path reframe

## Status

Design, pre-registered framing. The third leg of the owner's
2026-05-19 conversational-path reframe (1. SPEAR temporal
multiplexing — built, convergent ceiling; 2. theta-gamma mode-
unification — built, both readouts VALIDATED biologized via
parallel matching; 3. **generative replay** — the next pre-registered
build). All foundational subsystems are validated; this design
records the GENUINELY-NEW integration and proposes the pre-registered
test. Substantial multi-week build to follow; the owner has
instructed: write the design doc for review, then owner-steered
execution.

## Background and motivation

The owner's 2026-05-19 scientific reframe named the biological
resolution of the recent/remote retrieval conflict the earlier
necessity-instrument line went terminal on: conversation = a
generative hippocampal-prefrontal replay loop, PFC holding the
ordered compositional frame, hippocampal replay proposing-and-
pattern-completing against the consolidated cortical schema. Three
mechanisms load-bearing:

- **SPEAR temporal multiplexing** (Hasselmo): one shared theta
  rhythm time-multiplexes encode vs retrieve phases (high vs low
  ACh). The project built this; the decisive run hit a convergent
  ceiling because the static two-store framing doesn't fit
  biology.
- **Theta-gamma mode-unification** (Lisman-Idiart N.16): order-
  bearing AND order-invariant readout from one theta-gamma encoded
  code. The project built this in 2026-05-23 with parallel-
  population-matching identification (VALIDATED; both readouts
  PASS multi-seed at 32-concept tier).
- **Generative replay** (Buzsaki sharp-wave ripples;
  complementary-learning-systems / McClelland 1995): during quiet
  wake and slow-wave sleep, hippocampal sharp-wave ripples (SWRs)
  replay encoded episodes; the replays drive cortical pattern-
  completion via consolidated CLS pathways. Behaviourally these
  replays support prospective planning and offline learning. The
  project has validated SWR consolidation (Phase 1.3, 3/3 strict
  anti-cheat) but has NOT built the generative loop — the
  prospective-planning side where replay PROPOSES continuations
  against a held PFC frame.

With mode-unification's order-bearing readout now biologized, the
PFC can hold an ORDERED compositional frame (a sequence of bound
concepts at specific gamma slots in a theta cycle). The
generative-replay loop closes by: PFC frame → triggers SWR replay
against the consolidated cortical schema → replay activates
pattern-completed continuation candidates → mode-unification
decodes them → loop integrates the strongest candidates back into
the PFC frame for the next iteration.

This loop is the project's catalog-named conversational substrate.

## What is reused (validated, byte-unchanged)

- **Substrate**: trained sparse-distributed concept ensemble
  (G.20 sparse; 32-concept per bridge, 5-bridge ensemble validated
  at 160 unique concepts).
- **Hippocampus**: trisynaptic loop builder
  `build_biological_brain_regions(enable_hippocampus_consolidation
   =True)` (validated 2026-05-11: D.12 pattern separation; D.13
  pattern completion; D.14 engram tagging start/commit/stimulate;
  Phase 1.3 SWR consolidation 3/3 strict anti-cheat multi-seed).
- **PFC working memory**: `dlpfc_wm` NMDA bistable maintenance
  region (validated; per-region NMDA in v2.5 stack).
- **Mode-unification**: per-slot binding via gamma-slot position
  phasors; per-slot decoding via parallel-population-matching
  (VALIDATED capability pillar n=93).
- **FHRR-biologization layer**: resonate-and-fire bind/unbind/
  bundle on the substrate's grounded symbols.
- **Neuromodulator subsystem**: ACh gating for encode/retrieve
  phases (per SPEAR's faithful fix at commit f1292a0).
- **No-confab moat**: abstention gate (7/7 stays green throughout).

## What is genuinely new

A **generative-replay controller** that orchestrates the loop on
top of the validated subsystems. The controller:

1. **Holds a PFC frame**: an ordered K-tuple of bound (item,
   gamma-slot-position) pairs in dlpfc_wm; encoded via
   ResonateFireFHRR.encode; held by NMDA bistability.

2. **Gates a hippocampal SWR replay event**: triggers an
   SWR-window in the hippocampus (the validated SWR-replay-phase
   mechanism); the replay activates a consolidated pattern in the
   cortical schema (via the validated CA1→cortex consolidation
   pathway).

3. **Decodes the replayed continuation**: the cortical activity
   pattern post-replay is captured; mode-unification's per-slot
   parallel-matching decoder identifies the replayed item at the
   target slot.

4. **Updates the PFC frame**: the decoded continuation candidate
   is integrated into the PFC frame for the next iteration (e.g.,
   extends the frame by one slot at the next gamma position; OR
   shifts the frame forward consuming the earliest slot).

5. **Repeats** for N iterations, producing a sequence of replay-
   completed continuations from the consolidated schema.

The controller is the genuinely-new wiring. Components 1-4 each
reuse validated subsystems; the LOOP is the new integration.

## Pre-registered test (proposed; to be refined in TDD plan)

The cleanest pre-registered test of "generative replay" is **partial-
sequence completion via replay**:

1. **Train the substrate** on K stored sequences of ordered (item,
   slot) bindings — e.g., 32 ordered triples (A, B, C) where A is
   bound to slot 1, B to slot 2, C to slot 3. Use the validated
   G.20 training + the existing engram-tagging mechanism (D.14) so
   the sequences are stored in the consolidated cortical schema
   (post-Phase-1.3 consolidation).

2. **Initialise the PFC frame** with a partial cue — e.g., the
   first 2 slots filled (A bound to slot 1, B to slot 2), slot 3
   empty.

3. **Run the generative-replay loop** for N iterations: each
   iteration triggers SWR replay against the schema, decodes the
   continuation, updates the frame.

4. **Measure** whether the loop completes the partial cue to the
   correct stored sequence (slot 3 = C). Multi-seed (42, 43, 44);
   200 trials per K (sequence count); pre-registered frozen 0.80
   completion accuracy bar.

PASS iff multi-seed-mean completion accuracy ≥ 0.80 at every K in
a fixed K-ladder (e.g., {4, 8, 16, 32} sequences in the schema).
NEGATIVE if the replay-decoded continuation does NOT match the
stored sequence's slot-3 item.

The 0.80 bar matches the project's frozen compositional bar
verbatim. Multi-seed matches the project's discipline.

## Soundness considerations

The generative-replay loop's load-bearing properties (to be
pinned by adversarial review BEFORE any decisive run):

1. **No oracle leak**: the test scoring uses the true slot-3 item
   ONLY for post-hoc comparison; the replay-decoded continuation
   MUST be a function only of (PFC frame, SWR replay activity,
   parallel-matching decoder output over the full vocabulary). The
   true slot-3 item NEVER influences the replay or decode.

2. **Genuine SWR replay**: the SWR-window mechanism (the
   validated `enable_hippocampus_consolidation=True` builder's
   replay phase) is reused byte-unchanged; the replay is driven by
   the held PFC frame, not by a hand-supplied seed.

3. **The PFC frame is genuinely held by NMDA bistability** (not
   re-injected each iteration). The dlpfc_wm region maintains the
   bound pairs via the validated NMDA bistability across the
   replay window.

4. **The consolidated schema is genuinely the substrate's
   trained content** (post-Phase-1.3 consolidation), not a hand-
   supplied lookup. The schema lives in the cortical
   weights post-consolidation.

5. **The parallel-matching decoder is the validated one** (the
   2026-05-23 VALIDATED capability pillar n=93), used unchanged
   for the per-slot decoding of replay outputs.

6. **Reuse-by-import only**: no protected/frozen/moat module
   modified. The genuinely-new code is the controller wiring.

7. **No automatic differentiation**: all learning is via the
   reused validated local rules (STDP, eligibility, Hebbian,
   reward modulation).

8. **The no-confab moat remains 7/7 green throughout**.

9. **Frozen 0.80 bar never tuned**.

## Implementation outline (TDD plan to follow separately, OWNER-STEERED)

Task 0: grounding pin (constants, frozen bar, K=16 recipe, mode-
unification recipe, hippo/dlpfc subsystems all unchanged; loop
controller module exists; red until Task 2).

Task 1: small helper for the K-stored-sequence vocabulary
generation (per-seed deterministic, reuses g20_vocab_spec patterns
+ gamma-slot positions).

Task 2: the generative-replay loop controller +
runner — `research/findings/raw/generative_replay_runner.py`.
Loads trained substrate; encodes K stored sequences via mode-
unification's encoding; runs Phase-1.3-style consolidation to
embed them in cortex; per trial initialises PFC frame with partial
cue and runs N iterations of the loop; per-iteration triggers SWR,
captures replay-driven cortical activity, decodes via parallel-
matching, updates frame. Smoke mode for end-to-end validation.

Task 3: soundness tests (PFC frame is held; SWR is genuine; decoder
is unchanged; no oracle leak in the loop).

Task 4: dedicated adversarial review BEFORE the decisive run.

Task 5: controller-only decisive GPU run (substantial; multi-seed;
K-ladder; multi-hour to multi-day GPU depending on substrate +
consolidation time).

## Wall-clock estimate

The substrate side reuses the validated trained-substrate cache
(~58 min/seed for fresh substrate; 0 min if reused). The
consolidation phase reuses Phase 1.3 (the existing consolidation
trainer; ~25 min single-seed at the standard config; multi-seed
multiplies). The replay-loop CPU is cheap.

Realistic per-seed estimate: ~1.5-2 hours including substrate +
consolidation + replay loop trials. Multi-seed × 3 = ~5-6 hours.
Plus the K-ladder multiplies modestly. Probably ~10-15 hours total
for a comprehensive multi-seed K-ladder run.

Significantly cheaper than the 14+ hour 160-ensemble run (which
required 5 separate bridges × training). The replay loop reuses
one bridge with engram-tagged sequences.

## Honest scope

This design is the third leg of the owner's conversational-path
reframe. Whatever the verdict on the decisive run:

- **PASS**: the project has a validated biology-grounded
  generative-replay loop on top of mode-unification, on top of
  the consolidated trained substrate. This IS the conversational
  substrate the owner's reframe described. Subject to honest
  caveats (oracle-adjacency of the parallel-matching decoder; the
  PFC frame's compositional capacity at this scale; etc.).
- **NEGATIVE**: an honest finding about which biological
  component fails to scale. Each subsystem has been validated
  individually; if the integration fails, the failure mode
  precisely localises which integration property is missing.

This is NOT a claim that the project would have "conversation" in
the natural-language sense (no NLP, no language model, no tokens).
It IS a claim that the project would have the BRAIN-FAITHFUL
substrate for conversation — the same substrate biology uses to
generate context-appropriate continuations from consolidated
schematic content.

The standing oracle-adjacency caveat from mode-unification carries
forward: parallel matching's per-slot decoding is structurally
closer to "argmax over a stored vocabulary" than TPAM's recurrent
attractor; the "vocabulary" is the substrate's own derived
grounded symbols (biology-grounded), but the caveat is recorded.

Frozen 0.80 bar never tuned. Reuse-by-import only (the substrate,
hippocampus, PFC, consolidation, mode-unification, abstention gate
are all reused unchanged). No automatic differentiation. The no-
confab moat must remain 7/7 green throughout.

## Files

- This design doc:
  `docs/plans/2026-05-23-generative-replay-design.md`
- TDD implementation plan (to be written next, owner-steered):
  `docs/plans/2026-05-23-generative-replay-implementation.md`
- Reused subsystem references:
  - Substrate: `vocabulary_scaling_run_trained.py`,
    `concept_pool_sparse_distributed.py`,
    `vocabulary_scaling_160ensemble_helpers.py`
  - Hippocampus + consolidation:
    `research/runners/text_minimal_isolation.py` (`build_biological_brain_regions(enable_hippocampus_consolidation=True)`),
    `research/runners/consolidation_trainer.py`
  - PFC dlpfc_wm: integrated in build_biological_brain_regions
  - Mode-unification:
    `research/findings/raw/biologized_spiking_mode_unification_parallel_matching_runner.py`
  - Abstention gate / no-confab moat:
    `research/runners/abstention_gate.py`
- The owner's 2026-05-19 conversational-path reframe (deepens the
  current objective in AUTONOMOUS_STATE.md).
