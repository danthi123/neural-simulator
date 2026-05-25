# Prior mechanism-class audit + direction selection guide (2026-05-25)

> Built during the P-v3 rediscovery cleanup. Purpose: prevent future
> duplicate-direction launches by maintaining an explicit index of
> which mechanism classes are CLOSED (convergent NEGATIVE) vs OPEN
> (untested) vs IN-FLIGHT. Read this file BEFORE proposing any new
> direction to confirm the proposed mechanism is genuinely net-new.

## How to use this file

Each section is a MECHANISM CLASS (not a substrate variant). When
proposing a new direction:

1. Find the mechanism class the direction falls into.
2. Check the convergent status (CLOSED / OPEN / IN-FLIGHT).
3. If CLOSED: do NOT re-derive a fix in that class; pick from OPEN
   classes or scale-axis directions.
4. If IN-FLIGHT: coordinate with the in-flight work.
5. If OPEN: proceed with cheap-first probe per discipline.

## Mechanism class index

### CLASS 1: Dynamics-gating / wiring / amplification (CLOSED - 5+ convergent NEGATIVE)

Hypothesis: compositional capability emerges from adding the right
network dynamics (gating, lateral inhibition, recurrence, etc.) or
wiring (extra pathways, cross-projections).

Convergent NEGATIVE findings:

| Direction / Probe | Date | Mechanism | Result |
|---|---|---|---|
| SPEAR ACh phase-separation | 2026-05-19/20 | global synaptic-gain modulation across theta cycle | NEGATIVE (full_acc 0.00 every rung) |
| ca1-variant substrate | 2026-05-22 | 12 plastic ca1->concept_pool wires | NEGATIVE (post-install pool rate 0.0023 vs pre 0.0024) |
| Staged-recurrence variant | 2026-05-22 | ACh-staged recurrent excitation ON ca1-variant | NEGATIVE (consolidation flat at noise floor; structural-effect check verified active) |
| Difference-readout probe | 2026-05-21 | sub-pool firing-rate difference at readout | NEGATIVE |
| 8 prior architectures | 2026-04 -> 2026-05-21 | gating, theta-multiplexing, disinhibition, per-regime monitoring, cue-suppression, aggressive consolidation, pool-readout substitution | All NEGATIVE on compositional readout |
| Direction P (multitag + SWR) | 2026-05-24 | cortex-only multitag + SWR sleep | TRIVIAL PASS (SWR not load-bearing) |
| Direction P-v2 (hippo-only engram) | 2026-05-24 | hippocampal-only engram + SWR transfer | HIPPO_ENCODING_INSUFFICIENT (pre-A 0.167 < 0.50) |
| Direction P-v3 (CLS architectural fix) | 2026-05-25 | hippo-only engram + 12 ca1->concept pathways | DUPLICATE of 2026-05-22 ca1-variant; killed |
| (c) generative-replay loop | 2026-05-24 | encode PFC frame -> SWR -> capture cortex -> decode -> update | pillar n=99 NEGATIVE (104/1800 = 5.78% vs 6.25% chance) |
| (c) loop diagnostic | 2026-05-24 | direct measurement of SWR-driven cortical reactivation | REPLAY_DOESNT_REACTIVATE (selectivity +0.006) |

**Status: CLOSED.** Verdict from 2026-05-22 (still binding): "The
compositional fix is not in the network dynamics. It is in the
REPRESENTATION."

**Discipline: do NOT propose any new dynamics-gating / wiring /
amplification mechanism without first explaining why it ISN'T a
duplicate of the 10+ already-closed findings in this class.**

### CLASS 2: Phase-coded vector-symbolic composition (REPRESENTATION) (CHARACTERIZED - bounded at substrate scale)

Hypothesis: composition is carried by phase-coded structured
representation (FHRR, spiking phasor, theta-gamma multiplexing),
NOT by network dynamics.

Convergent CHARACTERIZED findings:

| Direction / Probe | Date | Mechanism | Result |
|---|---|---|---|
| FHRR algebra | various | pure numpy FHRR bind/unbind/bundle | PASS multi-seed |
| FHRR + substrate-grounded | 2026-05-22 | substrate concept-pool activity grounds FHRR phasors | PASS (multiple pillars n=84-94) |
| Direction K teacher | 2026-05-24 | FHRR + substrate, teacher current | 1.000 (artifact) |
| Direction K no-teacher | 2026-05-24 | FHRR + substrate, fair test | 1.000 BUT substrate not load-bearing (reviewer BLOCK; random phasors also PASS at N_DIM=3200) |
| Direction K biologized | 2026-05-24 | FHRR + biologized pipeline | 0.000 (too strict at scale) |
| Theta-gamma ALGEBRA | 2026-05-24 | numpy probe of Lisman-Idiart multiplexing | PASS multi-seed (pillar n=103 VALIDATED, reviewer CLEAR) |
| Direction E Task 1 substrate | 2026-05-24 | cortical + theta-gamma at substrate | 0.250 BOUNDARY |
| Direction G HIPPO + theta-gamma | 2026-05-24 | HIPPO substrate + theta-gamma | 0.333 BOUNDARY |

**Status: ALGEBRA PASS / SUBSTRATE BOUNDED.** The mechanism works
at numpy/algebra scale but is BOUNDED at substrate scale. Same
ceiling as Class 1 - substrate is the bottleneck.

### CLASS 3: Substrate-scale extension (OPEN - pre-registered, not built)

Hypothesis: the convergent substrate-scale ceiling (both classes
above hit it) can be raised by extending substrate scale toward
biological values.

Pre-registered options (none yet implemented):

| Direction | Pre-registered cost | Test |
|---|---|---|
| Q: dlpfc_wm 60 -> 1000 neurons + dense recurrent | 1-2 weeks build | Wang 2002 NMDA persistent activity at proper scale; closes Direction I bound |
| Q-prime: ALL substrate region neuron-counts scaled 10x (~80K total) | 2-4 weeks build; substantial GPU | Tests whether substrate-scale is THE bottleneck (vs mechanism) |

**Status: OPEN.** Direction Q (dlpfc_wm scale-up) is the cheapest
substrate-scale extension and is explicitly pre-registered. Q-prime
(uniform 10x scale-up) is more substantial but most directly tests
the substrate-scale hypothesis.

### CLASS 4: Cross-bridge composition (OPEN - per 2026-05-24 post-c roadmap)

Hypothesis: composition is achievable across MULTIPLE substrates
(each one validated) rather than within a single substrate at scale.

Existing status:

| Substrate | Cross-bridge tested? | Result |
|---|---|---|
| G.20 sparse 5-bridge (160-concept) | YES - cross_bridge_mode_unification_probe.py | OB perfect / OI L=5 boundary (pillar n=95) |
| G.20 sparse 5-bridge (320-concept) | YES - Direction M deliverable | working multi-bridge chat (98.4% per-bridge; sentence parser; abstention gate) |
| bio_brain_regions cross-bridge | NO - Direction 4 from post-c roadmap | NOT BUILT |

**Status: G.20 sparse VALIDATED (multiple pillars); bio_brain_regions
cross-bridge OPEN.** Direction 4 = build cross-bridge composition on
bio_brain_regions substrates. Per-bridge ~30 min train; full ensemble
~3 hr; cross-bridge probe ~10 min CPU. Needs design first.

### CLASS 5: Vocab scaling within single substrate (OPEN - per 2026-05-24 post-c roadmap)

Hypothesis: the validated bio_brain_regions substrate can handle
larger vocabularies without re-architecting.

Existing status:

| Substrate | Vocab tested | Result |
|---|---|---|
| bio_brain_regions OPTION 3 | V=16 | OI L=7 0.900 multi-seed |
| bio_brain_regions HIPPO-OPTION3 | V=16 | OI L=7 0.895 multi-seed |
| bio_brain_regions DLPFC-extension | V=16 | OI L=7 0.935 multi-seed |
| bio_brain_regions V=32/64/160 | NOT TESTED | OPEN |

**Status: V=16 fully characterized (pillars n=96/n=97/n=98 + load
ceiling map); V=32/64/160 OPEN.** Direction 3 = vocab scaling on
bio_brain_regions. Per tier ~1.5-2 hr GPU. Smaller change than
Direction 4 (no cross-bridge architecture needed).

### CLASS 6: Goal-directed generation (BG integration) (OPEN)

Hypothesis: composition under goal/reward selection via basal
ganglia integration produces conversational utility.

Components ALL VALIDATED INDEPENDENTLY:
- (c) loop: NEGATIVE on partial-sequence completion (so this class
  may inherit the same dynamics-class ceiling)
- BG cascade (g11_bg_runner): validated for navigation
- Neuromodulator subsystem: validated

**Status: NEEDS REDESIGN given (c) NEGATIVE.** The original
proposal in 2026-05-24 post-c roadmap assumed (c) PASS. Now needs
re-thinking with the convergent dynamics-class CLOSED finding.

### CLASS 7: Continual learning + episodic memory integration (OPEN)

Hypothesis: simultaneous dialog-state engram + schema consolidation
without catastrophic forgetting produces multi-day conversational
artifact.

Status: NOT BUILT. Per 2026-05-24 post-c roadmap. Longest GPU run
of the chain; ~1-2 weeks total wall-clock; substantial substrate-
level investigation.

## Cheapest-first ranking (autonomous direction selection)

For the next concrete action (per autonomous-runs principle "highest
expected information gain, fewest dependencies, fastest cycle time"):

1. **Class 5 Direction 3** (vocab scaling on bio_brain_regions; V=32
   tier first). ~1.5-2 hr GPU. Tests whether the validated substrate
   has vocab headroom beyond V=16. Compounds on validated pillars
   n=96/n=97/n=98. Pre-registered: parallel-matching mode-unification
   PASS multi-seed >= 0.80 on both readouts. NEGATIVE would inform
   the vocab-capacity scaling exponent (cf. FHRR n=87 algebra cap
   N_dim / V).

2. **Class 4 Direction 4** (cross-bridge bio_brain_regions). ~3 hr
   per ensemble + probe time. More substantial; needs design first.

3. **Class 3 Direction Q** (dlpfc_wm scale-up). 1-2 weeks; closes
   Direction I bound + tests the substrate-scale-is-bottleneck
   hypothesis directly. Most substantial; biggest payoff.

## Discipline rules (binding)

1. **Pre-launch grep** of prior findings dir for the proposed
   mechanism class + architectural substrate. If a similar mechanism
   was already characterized, do NOT proceed without explicitly
   addressing why this is genuinely different.
2. **Bar UNCHANGED**: 0.80 multi-seed strict top-1; 0.50/0.30/0.30
   for CLS arc; no bar tuning.
3. **No protected/frozen/moat modification.** The validated set is
   byte-empty diff; the no-confab moat is 7/7 green.
4. **No autograd.** Reuse validated local-learning rules only.
5. **GPU/CuPy for real runs; numpy only for cheap-first probes.**
6. **Honest propagation EVERY outcome** to both git remotes; a
   negative is a scientific finding, not a failure.
7. **Pre-registered tags** for every direction (PASS / NEGATIVE /
   BOUNDARY / DUPLICATE) so the verdict is determined by the
   recorded outcome, not by interpretation.

## Files referenced

- 2026-05-22 staged-recurrence NEGATIVE: `research/findings/2026-05-22-staged-recurrence-variant-NEGATIVE-verified-active-dynamics-gating-class-exhausted-converges-with-SPEAR.md`
- 2026-05-24 (c) loop diagnostic: `research/findings/2026-05-24-c-loop-diagnostic-REPLAY_DOESNT_REACTIVATE-Phase-1-3-SWR-consolidation-validated-for-direct-binding-not-sequence-completion.md`
- 2026-05-24 (c) loop n=99 pillar: `research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-loop-at-n-iterations-1-doesnt-produce-above-chance-completion-pivot-direction-identified.md`
- 2026-05-24 bio_brain_regions load ceiling map: `research/findings/2026-05-24-bio_brain_regions-load-ceiling-map-ALL-3-substrates-PASS-every-load-L2-to-L7-the-c-NEGATIVE-is-not-substrate-bounded.md`
- 2026-05-24 post-c roadmap: `docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`
- 2026-05-24 Direction M deliverable: `research/findings/2026-05-24-DIRECTION-M-COMPLETE-320-concept-multi-bridge-chat-deliverable-VALIDATED.md`
- 2026-05-25 P-v3 rediscovery: `research/findings/2026-05-25-DIRECTION-P-v3-DUPLICATE-REDISCOVERY-ca1-variant-arc-CONVERGENT-NEGATIVE-pivot-to-representation-class.md`
