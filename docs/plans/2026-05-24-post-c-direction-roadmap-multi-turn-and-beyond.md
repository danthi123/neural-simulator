---
type: plan
status: live
date: 2026-05-24
---

# Post-(c) direction roadmap — what advances toward conversation AFTER the generative-replay loop is validated (2026-05-24)

> Planning sketch written during the DLPFC-extension GPU wait. Incorporates the
> Schwartenbeck 2023 biology + PFC-SWR timing biology shipped today. Owner-
> authorised overnight autonomy: "continue working all night and morning";
> "consideration of our goals for this project (artificial life and conversational
> capabilities)"; "make sure we stick to biological realism ultimately"; "reference
> the knowledge catalog as needed."

## Context

The (c) generative-replay arc (in flight; substrate-readiness chain n=96/n=97 + n=98 DLPFC-extension pending) builds the SECOND major mechanism of the 2026-05-19 conversational-path reframe (after the FIRST: SPEAR temporal multiplexing, convergent ceiling; the SECOND IS theta-gamma mode-unification, biologized end-to-end via n=92-97; and the THIRD IS generative replay, the (c) build).

What comes AFTER (c) toward FULL conversational capability? Three nested directions, each with its own pre-registered fixed-bar test, each building on validated predecessors. Sketched here for owner-steered execution after (c) validates.

## Direction 1: Iterative-refinement characterisation of the (c) loop

**Motivation**: Schwartenbeck 2023 (Cell) observes three-stage iterative refinement in hippocampal-prefrontal generative replay: early non-selective → middle selective → late converged (3500ms = ~30 theta cycles in their data).

**Test**: After (c) base PASS, run the (c) loop for ~30+ iterations on partial-cue tasks; characterise the iteration-by-iteration decoded continuation distribution. The biology-translatable expectation: distribution should narrow over iterations (entropy decreases monotonically; cosine to correct continuation increases monotonically).

**Pre-registered metric (descriptive; not a PASS criterion)**: iteration-by-iteration accuracy curve shape; entropy decay rate; convergence iteration index.

**Cost**: ~hours; reuses (c) substrate + loop runner; characterisation extension to (c) pillar.

## Direction 2: Multi-turn dialog dynamics

**Motivation**: Conversation requires MULTI-TURN interaction (queries refer back to earlier turns; PFC working memory persists across turns; hippocampus binds prior dialog turns as episodic memories). The biology: extended-time-scale hippocampus-prefrontal interaction; the project's existing infrastructure provides the components.

**Components needed (all validated independently)**:
- (c) generative-replay loop (after (c) PASS)
- Engram tagging (D.14) for prior-turn episodic binding (already validated)
- dlpfc_wm NMDA bistability for cross-turn frame persistence (already validated component)
- Phase 1.3 SWR consolidation for prior-turn → schema integration (already validated)

**Genuinely-new code**: a multi-turn dialog controller — accepts a sequence of N user queries, each one engram-tags into hippocampus, runs the (c) loop on each, accumulates dialog state in dlpfc_wm, replays prior-turn engrams during current-turn loop to provide context.

**Pre-registered test**: 3-turn dialog where each turn's partial cue includes a back-reference to a prior turn's bound concepts. PASS iff multi-seed-mean completion accuracy ≥ 0.80 on the back-reference resolution.

**Cost estimate**: ~1-2 weeks subagent-driven build + ~6-12 hr decisive GPU run.

## Direction 3: Larger vocabulary scaling (32 / 64 / 160-concept tier on bio_brain_regions)

**Motivation**: The project's existing G.20 sparse substrate validates up to 320-concept tier (5 bridges × 64). The bio_brain_regions substrate validated at 16-concept tier (this session). Conversation requires 100+ concept vocabularies.

**Test**: extend the OPTION 3 / HIPPO-OPTION3 / DLPFC-extension chain to 32, 64, 160 concepts on bio_brain_regions. Per-concept dynamics may need re-tuning (weak dynamics at higher count); concept-pool size may need adjustment.

**Pre-registered**: at each tier, parallel-matching mode-unification must PASS multi-seed ≥ 0.80 on both readouts; (c) loop must continue PASSing the partial-sequence completion bar.

**Cost**: each tier ~hours GPU; iterative scaling; characterisation curve.

## Direction 4: Cross-bridge composition on bio_brain_regions (the 160-ensemble pattern)

**Motivation**: G.20 sparse cross-bridge composition validated at OB perfect / OI L=5 boundary (n=95). The bio_brain_regions analog hasn't been built. If multiple bio_brain_regions substrates are trained on different vocabulary categories, cross-bridge composition extends the conversational vocabulary substantially.

**Test**: train 5 bio_brain_regions bridges on noun/verb/adj/spatial/functional vocabularies; ensemble them; run parallel-matching mode-unification cross-bridge.

**Cost**: per-bridge ~30 min train; full ensemble ~3 hours; cross-bridge probe ~10 min CPU.

## Direction 5: Goal-directed generation (basal ganglia integration)

**Motivation**: The project's BG cascade (g11_bg_runner) is the validated basal ganglia infrastructure. For goal-directed conversation, BG selects among (c) loop's candidate continuations based on reward / dopamine modulation.

**Components needed**:
- (c) loop (after (c) PASS)
- BG cascade with per-action striatal D1/D2 + GPi + thalamus + cortex (already validated for navigation)
- Neuromodulator subsystem (DA modulation; already validated)

**Genuinely-new code**: a BG-(c) interface layer — (c) loop's candidate continuations feed into BG cortex; BG selects the highest-DA-weighted continuation; chosen continuation feeds back into (c) loop's next iteration.

**Pre-registered test**: goal-conditioned partial-cue completion. The agent has a goal (e.g., "answer about color"); BG selects from (c) candidates based on goal-relevance. PASS iff goal-relevant continuations are selected at multi-seed ≥ 0.80.

**Cost**: ~2-4 week subagent-driven build + multi-day decisive GPU.

## Direction 6: Continual learning + episodic memory integration over time

**Motivation**: McClelland 1995 / Buzsaki 2013 CLS thesis: rapid hippocampal episodic + slow cortical schema; the project's Phase 1.3 SWR consolidation already validates the rapid → slow transfer. Conversation needs BOTH simultaneously: new dialog turn → engram-tag → consolidate into schema while preserving prior dialog as accessible context.

**Test**: run a multi-day "dialog session" simulation; periodically run Phase 1.3 consolidation cycles to transfer recent dialog into schema; verify the schema captures the dialog content (queryable via parallel-matching mode-unification) AND the recent engrams still discriminate distinct turns (no catastrophic forgetting).

**Pre-registered**: schema-recall accuracy + engram-distinctness simultaneously ≥ 0.80 multi-seed at simulated day boundaries (consolidation cycles).

**Cost**: longest GPU run of the chain; ~1-2 weeks total wall-clock; substantial substrate-level investigation.

## Honest framing throughout

These directions advance the BIOLOGY-GROUNDED substrate for conversation. They are NOT LLM-fluent prose. The validated project asset remains the trustworthy continual memory + no-confabulation abstention; conversation-substrate validation across these directions extends the biological coverage without claiming language fluency. Any direction that succeeds is a biology-translatable insight; any that fails honestly is also biology-translatable (precisely identifies which biological component fails to scale).

## Suggested execution order (post-(c) base PASS)

1. **Iterative-refinement characterisation** (direction 1; ~hours; sharpens (c) result)
2. **Multi-turn dialog** (direction 2; 1-2 weeks; the natural conversational arc)
3. **Cross-bridge bio_brain_regions** (direction 4; ~hours; extends vocabulary cleanly)
4. **Larger vocab scaling** (direction 3; iterative scaling; uses both pillars 1+4 if validated)
5. **Goal-directed generation** (direction 5; 2-4 weeks; integrates BG)
6. **Continual learning + EM** (direction 6; longest; the McClelland CLS integration)

Each direction has its own pre-registered fixed-bar test, adversarial review, and capability pillar. Standing discipline preserved throughout: reuse-by-import, no protected-module modification, no autograd, no-confab moat 7/7, frozen 0.80 bar.

## Standing constraints

- Reuse-by-import only across all directions.
- Each direction's adversarial review is mandatory before pillar.
- Honest scope: biology-grounded substrate, not LLM-fluent prose.
- Both remotes propagated at every commit.
- The owner steers the execution order (planning sketch is for reference, not unilateral commitment).

## Files

- This roadmap: `docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`
- (c) design: `docs/plans/2026-05-23-generative-replay-design.md`
- (c) TDD plan: `docs/plans/2026-05-24-generative-replay-implementation.md`
- Schwartenbeck biology: `research/findings/2026-05-24-Schwartenbeck-2023-biology-reference-for-c-generative-replay-three-stage-iterative-refinement.md`
- PFC-SWR biology: `research/findings/2026-05-24-biology-references-PFC-SWR-replay-30-50ms-window-selective-trajectory-encoding.md`
