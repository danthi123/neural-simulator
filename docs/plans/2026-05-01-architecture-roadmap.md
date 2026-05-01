# Sim Architecture Roadmap — Post-Cheat-5 Closure

**Date:** 2026-05-01
**Trigger:** F v2 + `--heuristic-single-pool` reached 4.55 ± 0.28 on cheat-5 multi-goal det (35% over A+E ceiling, biology-grounded, no cheat flags). Cluster-stacking yielded diminishing returns. Time to broaden scope per user directive: "plan additions to the sim architecture/features based on the catalog/academic/research papers we compiled to flesh out the sim".
**Author:** Autonomous overnight planning pass.

## Strategic priorities (from user directive)

1. **Biology-grounded** — every addition must cite real biological references in the code+docs. No hallucinations, no engineering-shortcut clusters that drift from the catalog.
2. **Scientific accuracy in naming** — use canonical glossary terms; never invent terms that aren't in `references/glossary.md`.
3. **Multimodal capability** — sim should accept richer input than the current 7×7 (dx,dy) sensor grid: visual frames, auditory streams, proprioceptive feedback.
4. **Dataset / behavior training** — sim should be able to learn from external data (expert trajectories, pre-recorded experiences) to emulate behaviors beyond the current cheat-5 navigation task.
5. **LLM-style interaction** — bidirectional communication: sim emits behaviors, external entity (user or LLM) provides feedback/commentary/reward shaping.

## Tier 0: Architecture priorities (next 2-4 weeks)

### G.1 Cluster G — PFC working memory (Wang 2002 attractor)

**Status:** Already partially scaffolded as `dlpfc_wm` region (cluster D wiring). Currently feedforward; needs NMDA-mediated recurrent excitation for true persistent activity.

**Biology source:** Kandel 6e Ch 17–29; Wang 2002 *J Neurosci* "Synaptic basis of cortical persistent activity"; Goldman-Rakic 1995 "Cellular basis of working memory"; Funahashi 1989 "Mnemonic coding of visual space".

**Implementation:**
- Add NMDA-component to `dlpfc_wm` synapses (currently AMPA-only via cp_conductance_g_e)
- Tune NMDA Mg2+ block + recurrent strength to give bistable attractor states
- Test on delayed-response task: hold goal across goal_silence_after_step window
- **Expected outcome:** PFC maintains goal info during silence, agent navigates correctly post-silence (vs current behavior: PFC drift)

**Test plan:**
- Unit test: NMDA conductance scales with V via Mg2+ block (already in `fused_nmda_update_and_current`)
- Integration test: PFC firing rate stays elevated during silence period
- Cheat-5 with `--goal-silence-after-step 1500 --goal-silence-duration 100`: agent should still reach phase-3 goal

**Effort:** ~1-2 days. Existing kernels handle NMDA; just need to wire it into the PFC region.

### K.1 Cluster K — Visual cortex hierarchy (V1→V2→V4→IT)

**Status:** NOT IMPLEMENTED. Current sensory is a 7×7 (dx,dy) goal-relative grid encoding.

**Biology source:** Kandel 6e Ch 22 (visual processing); Hubel & Wiesel 1962 "Receptive fields, binocular interaction"; Felleman & Van Essen 1991 "Distributed hierarchical processing"; Tanaka 1996 "Inferotemporal cortex and object vision".

**Implementation:**
- New `BrainRegion` types: `retina_v1` (Gabor-tuned simple cells), `cortex_v1_complex` (orientation-pooled), `cortex_v2` (illusory contours), `cortex_v4` (color+form), `cortex_it` (object identity)
- Pathway: image_input → retina_v1 → cortex_v1 → cortex_v2 → cortex_v4 → cortex_it → cortex_X (motor planning)
- Initial weights: Gabor-tuned for V1 (matching Hubel-Wiesel), then plastic for higher layers
- Frame input: 8×8 grid rendered to 64×64 pixels (8x upsample), fed as cortex drive

**Test plan:**
- V1 receptive fields after init match Gabor pattern (orientation tuning)
- IT readout decodes object identity above chance after STDP training
- Compose with existing cortex (cortex_X) for motor: visual_it → cortex motor planning

**Effort:** ~1 week. Needs: image-input driver, region types, initial weight Gabor patches, validation tests.

### G.2 + K.1 Compose: visual working memory

After G.1 and K.1 land, test their composition:
- Visual scene → retina_v1 → cortex_v1...IT → dlpfc_wm
- Delayed match-to-sample task (Funahashi 1989 paradigm)
- Test: PFC holds visual identity across delay; behavioral readout decodes

**Effort:** ~2-3 days after dependencies land.

### Train.1 Dataset / trajectory training infrastructure

**Goal:** Train sim on pre-recorded expert trajectories instead of (or in addition to) live RL.

**Implementation:**
- New runner: `research/runners/g11_bg_trajectory_train.py`
- Input: JSON file with `[{state: (x, y), action: 0-3, reward: float, next_state: (x, y)}, ...]`
- Loop: for each step, drive cortex_X corresponding to `action`, set reward signal, run stim window with plasticity
- Bypasses heuristic, bypasses BG action selection (we don't sample action; we impose it)
- This is **imitation learning via STDP + reward**

**Test plan:**
- Generate expert trajectories from a hand-coded policy on cheat-5
- Train sim on those trajectories for N epochs
- Eval: agent runs cheat-5 fresh; does it generalize / behave like the expert?

**Effort:** ~3-5 days. Self-contained; doesn't require new biology.

## Tier 1: Smaller incremental additions

### O.1 Cluster O — Amygdala for valence learning

**Status:** Not implemented. Reward currently scalar; emotional valence not decomposed.

**Biology:** Kandel Ch 49–50; LeDoux 2000 "Emotion circuits in the brain".

**Implementation:**
- BLA (basolateral amygdala) region: sensory + reward → BLA → NAcc shell
- CeA (central amygdala) region: BLA → CeA → autonomic effectors

**Use case:** Distinguish appetitive (NAcc shell) from aversive (CeA) responses. Compose with two-component DA (Schultz 2016).

**Effort:** ~3-4 days.

### N.1 Cluster N — Slow oscillation + ACh REM gating

**Status:** Sleep replay exists but is episodic (NREM trajectory replay). Real sleep has 1Hz slow oscillation, K-complexes, spindles.

**Biology:** Kandel Ch 49; Buzsáki 2006 *Rhythms of the Brain* Cycle 9 + 11.

**Implementation:**
- Add slow_osc neuromodulator with 1 Hz sinusoidal concentration
- Couple to membrane excitability_drive
- Up-state / down-state alternation gates plasticity (B.3 TANs ACh window-gating fits here)

**Effort:** ~2-3 days.

### J.1 Cluster J — Three-factor learning rule explicit

**Status:** Already implemented as eligibility traces × reward modulation. Not explicitly named or documented as "three-factor".

**Biology:** Sutton & Barto; Schultz 2007.

**Implementation:** Documentation + test fixtures only (mechanism is in place).

**Effort:** ~1 day.

## Tier 2: Major undertakings (next month)

### Multimodal sensory integration (composes K + auditory + somatosensory)

After visual cortex (K.1) lands, add:
- Auditory cortex: cochlea (hair cells per Kandel Ch 23) → auditory_cortex_a1
- Somatosensory cortex: Pacinian/Meissner/Merkel/Ruffini afferents → s1
- Multimodal integration: V/A/S → posterior parietal cortex → cortex_X

**Goal:** Sim consumes (image, audio_clip, touch_state) tuples and learns from them.

**Effort:** ~2-3 weeks for full multimodal stack.

### Continuous-time / continuous-action environment

Currently env is discrete: 1 step per env tick, 4 cardinal actions. Real motor control is continuous.

**Implementation:** new env class with velocity-based control; agent reads from motor regions as analog signals.

**Effort:** ~2 weeks. Major refactor of env step.

### LLM-as-supervisor loop

User explicitly mentioned this. Build infrastructure where:
- Sim emits behavior to a buffer
- LLM reads buffer, emits commentary/preference labels/reward shaping
- Sim consumes these as additional reward / curriculum signals

**Implementation:** stdin/stdout protocol or socket-based. LLM runs externally.

**Effort:** ~1 week for a basic stub; ~3 weeks for a useful loop.

## Tier 3: Long-term research directions

- **Spinal CPGs (Cluster H)** for true motor sequences
- **Disease models (Cluster P)** — simulate Parkinson's via DA depletion
- **Glia (Cluster Q)** — slow modulators, K+ buffering
- **Embodied learning** — sim as agent in an OpenAI gym env

## Sequencing recommendation

**Week 1-2:** G.1 (PFC working memory). Highest value-per-effort.

**Week 2-3:** K.1 (visual cortex hierarchy). Foundational for multimodal.

**Week 3-4:** Train.1 (dataset training). Unlocks new experimental paradigms.

**Month 2:** G.2+K.1 compose, then O.1 amygdala, then start multimodal stack.

**Month 3+:** continuous-time env + LLM supervisor + multimodal full stack.

## What this roadmap is NOT

- **Not a cluster-stacking exercise.** We've established the cluster-stacking has diminishing returns past F v2. Major-feature additions take precedence over more cluster permutations.
- **Not engineering-shortcut work.** Every feature must be biology-grounded with citations.
- **Not "polish" work.** Webapp / docs / tests are valuable but not the priority — major architecture additions are.

## Files

- This plan: `docs/plans/2026-05-01-architecture-roadmap.md`
- Cluster G design: TBD (`docs/plans/2026-05-01-cluster-g-pfc-wm-wang2002.md` to be written)
- Cluster K design: TBD (`docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md` to be written)
- Train.1 design: TBD (`docs/plans/2026-05-01-trajectory-training-infrastructure.md` to be written)
