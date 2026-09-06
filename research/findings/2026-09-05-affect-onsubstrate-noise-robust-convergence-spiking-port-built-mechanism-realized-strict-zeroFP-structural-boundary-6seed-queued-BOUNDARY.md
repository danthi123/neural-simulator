---
type: finding
status: boundary
claim_check: measured
date: 2026-09-05
mechanism: the numpy-GO noise-robust grounded affect convergence realized ON A REAL spiking SimulationBridge -- code_in (graded per-neuron current) -> a plastic rate-Hebbian FF -> an assembly of excitatory NMDA neurons in soft-WTA competition with a shared FS interneuron pool (Wong-Wang/Grossberg on-center/off-surround = competition + divisive normalization); the learned affect concept code READ OFF cp_firing_states (real assembly spikes, snapshot/restore-isolated per concept), read by the SAME validated separability CEILING. The point-neuron assembly SPIKES and is grounding-modulated, but under the strict zero-FP criterion at full-partition SCALE it does NOT reproduce the numpy separation: false-grounded neutrals fire as strongly as true affect, so the substrate loses the comfort/discomfort STRUCTURAL discrimination the numpy rate+ridge idealization retains
lane: affect-learned-gate-retirement (rank-7)
seeds: [42]
runner: research/runners/_affect_onsubstrate_noise_robust_convergence_derisk.py
artifacts:
  - research/findings/raw/_affect_onsubstrate_noise_robust_convergence_1seed_full.json
builds_on:
  - research/findings/2026-09-05-affect-noise-robust-homeostatic-three-factor-convergence-clears-strict-bar-derisk-GO.md
  - research/findings/2026-09-05-affect-grounded-experience-stream-hebbian-convergence-teaches-separable-code-derisk-PARTIAL.md
  - research/findings/2026-08-19-embodied-affect-interoception-GO.md
verdict: >
  BOUNDARY / the spiking port is BUILT and the mechanism is REALIZED, but at 1-seed on the full partition it does NOT
  reproduce the numpy GO's separation under the strict zero-FP criterion; the 6-seed GPU verify is QUEUED to confirm
  robustly (controller harvests). The numpy noise-robust grounded affect convergence GO named a fully-spiking
  on-substrate convergence as its next rung. This runner BUILDS it: the COMPETITIVE convergence runs on a real
  SimulationBridge (code_in graded current -> plastic rate-Hebbian FF -> an excitatory NMDA assembly in soft-WTA
  competition with a shared FS interneuron pool) and the learned affect concept code is READ OFF cp_firing_states, not
  a numpy matrix. The MECHANISM works: the assembly SPIKES (24.3 spikes/concept) and is GROUNDING-MODULATED (grounded
  concepts fire ~32.6 spikes, the no-body-state LESION and the shuffle-binding control both collapse), and the
  separability instrument is valid (synthetic clean at ceiling, text-code ceiling well below the discrimination floor).
  BUT the learned spiking code does NOT clear the strict worst-case zero-FP bar on the full 164-word partition: SPIKING
  recall@FP0 at the realistic noisy point reads ~0, collapsing to the LESION baseline, where the numpy GO reads 0.600.
  MECHANISM of the boundary (measured): at realistic
  interoceptive noise ~17 of 62 neutral concepts receive "false grounding" (afferent noise passes the relay
  noise-floor) and drive the point-neuron assembly AS STRONGLY as true affect (neutral max 48 vs grounded mean 32.6
  spikes); the numpy rate+ridge readout separates these false-grounded neutrals from true affect by their
  comfort/discomfort SIGN STRUCTURE, but the shared-WTA point-neuron assembly collapses that structure into an
  undifferentiated "grounded/not" response, so the strict zero-FP threshold (set by the worst false-grounded neutral)
  craters recall. A brain-based lever -- an OUTPUT-side homeostatic floor (the downstream reader's intrinsic threshold,
  Turrigiano) on the read spike code -- was tested and does NOT help (the false-grounded neutrals fire too strongly to
  floor). This is an HONEST NEGATIVE under strict biology (the deliverable: it maps what the point-neuron substrate can
  and cannot do on its own vs the numpy idealization). NO-DEFER named surpass: an OPPONENT / columnar assembly (separate
  comfort- and discomfort-selective sub-pools, the _affect_composed_selforganized_opponent structure) that PRESERVES the
  sign structure the zero-FP discrimination needs. ADDITIVE, default-OFF (--spiking is opt-in; OFF delegates to the
  imported numpy GO verbatim, byte-identical -- asserted), NO sim/ edit, NOT wired (_STRONG_MARGIN==2.0 asserted). The
  6-seed spiking verify (SIM_BACKEND=cupy) is queued behind the mouth-training run; this finding is FINALIZED (boundary
  confirmed across seeds, or localized) when it lands.
lane_wall: affect salience gate (which words may move mood) -- rank-7 / affect-learned-gate-retirement; the fully-spiking on-substrate convergence rung
---

# Affect salience gate: the noise-robust grounded convergence on the spiking substrate -- BUILT + mechanism realized, but a strict-zero-FP structural boundary at scale (6-seed queued)

## The question this answers
`2026-09-05-affect-noise-robust-homeostatic-three-factor-convergence-clears-strict-bar-derisk-GO.md` is a NUMPY de-risk
GO: an emergent competitive rate-Hebbian convergence over a grounded interoceptive body-state stream + four biological
companion processes TEACHES a separable, generalizing affect concept code that clears the strict worst-case zero-FP bar
at realistic interoceptive noise. Its named next rung: **a FULLY-SPIKING on-substrate convergence -- reuse
`_genfrontier` build_propagation_bridge, so the concept code is READ OFF cp_firing_states, not a numpy matrix.** This
runner builds that and asks: **does the convergence PRESERVE its GO when the learning + competition run on a real
spiking SimulationBridge and the code is read off real spikes?** The measured answer at 1-seed on the full partition is
NO under the strict zero-FP criterion, for a specific and understood reason -- an honest boundary, with a named surpass.

## What is on the substrate vs the world/body boundary (brain-based-only)
Host is legit ONLY for the world/body US delivery. The interoceptive relay POPULATION + its pooled/adapted read
(companions 1+2 = pooling + the homeostatic noise-floor, the numpy GO's load-bearing pair) stay at the body boundary,
reused by import. What runs ON the spiking substrate is the piece the rung is about:
- **code_in** (Din neurons): the convergence input `[L2(text)*text_gain | POP-population-coded cleaned-intero*intero_gain]`
  as a GRADED per-neuron current. Population-coding each cleaned channel to POP neurons gives the assembly enough
  afferents to spike (a handful of channel neurons cannot drive a point neuron -- measured).
- **assembly** (M excitatory NMDA neurons): a plastic rate-Hebbian FF `code_in->assembly` -- the convergence the
  substrate LEARNS. NMDA integrates the sparse graded drive to spikes.
- **assembly_fs** (FS interneurons): a shared inhibitory pool (assembly -> FS exc, FS -> assembly gaba_a) = Wong-Wang /
  Grossberg soft-WTA competition + divisive normalization, spiking-native.
- **The learned concept code is READ OFF cp_firing_states**: drive each concept's code_in alone, snapshot/restore-isolate
  the read (kill the slow-NMDA carryover -- the decisive fix without which read-order artifacts masquerade as signal),
  accumulate assembly SPIKES per neuron -> an (n x M) spike-rate code -> the SAME validated ceiling.

## Derived -- the measured numbers (1-seed full partition + operating-point probes)
<!--derived: full-partition values are direct reads of research/findings/raw/_affect_onsubstrate_noise_robust_convergence_1seed_full.json (seed 42); the tiny-corpus and n=40 synthetic values are scratch operating-point probes -->
- **The mechanism is realized (G0 pass):** the assembly SPIKES **24.3 spikes/concept** at the realistic point (real
  cp_firing_states) and is GROUNDING-MODULATED -- grounded concepts fire ~32.6, the no-body-state LESION reads 0.000
  and the shuffle-binding control 0.000 (both collapse).
- **The instrument is valid (G3 pass):** synthetic clean-code ceiling 1.000; the text-code ceiling reads 0.029 (< 0.2)
  on the full 164-word partition -- the boundary is reproduced.
- **The spiking code does NOT clear the strict zero-FP bar (G1 FAIL, the boundary):** SPIKING recall@FP0 at the
  realistic noisy point reads **0.010**, vs the numpy GO **0.600** and collapsing to the LESION baseline 0.000. The
  clean/full point reads 0.200 and held-out(clean) 0.097.
- **MECHANISM of the boundary (measured):** at the realistic point ~**17 of 62 neutral** concepts receive false
  grounding (afferent noise passes the relay noise-floor) and drive the assembly AS STRONGLY as true affect (neutral
  total-spike max 48 vs grounded mean 32.6). The strict zero-FP threshold is set by the worst false-grounded neutral,
  so recall craters. The numpy rate+ridge readout separates false-grounded neutrals from true affect by their
  comfort/discomfort SIGN structure; the shared-WTA point-neuron assembly collapses that structure.
- **The output-homeostatic-floor lever does NOT help (measured):** an output-side floor (median + k*MAD of the
  total-activity distribution, k swept 1-10) leaves SPIKING recall@FP0 at 0.010 -- the false-grounded neutrals fire too
  strongly to floor without also zeroing true affect.
- **The tiny-corpus smoke was MISLEADING:** on a 24-word slice the spiking code read 0.500 (cleared the bar), an
  artifact of few neutrals; the full 164-word partition (62 neutrals under zero-FP) is the real test and reads 0.010.
  The n=40 synthetic probe (16 neutrals) read 0.583 for the same reason -- scale (neutral count under zero-FP) is the
  binding variable, which is why a single full-partition seed, not a smoke, is the honest evidence.
- **Byte-identical-when-off (asserted):** with `--spiking` OFF the pipeline delegates every ceiling to the imported
  numpy `robust_learned_code_ceiling` verbatim and reproduces it EXACTLY; the imported population stream at n_relay=4 is
  byte-identical to the imported grounded stream; `_STRONG_MARGIN==2.0` (production organs byte-unchanged).

## Reading it (no-defer)
The mechanism is realized -- an emergent competitive rate-Hebbian convergence on a real spiking bridge, reading a
grounding-modulated concept code off cp_firing_states -- but it does NOT preserve the numpy GO's separation under the
strict zero-FP criterion at full-partition scale, and the reason is measured, not mysterious: the point-neuron
shared-WTA assembly collapses the comfort/discomfort SIGN structure, so false-grounded neutrals (an unavoidable
~17/62 at realistic afferent noise) fire as strongly as true affect and poison the zero-FP threshold. This is an honest
negative under strict biology -- exactly the deliverable the brain-based-only standard asks for (it maps what the
point-neuron substrate can and cannot do vs the numpy rate+ridge idealization). It is a verdict on THIS METHOD (a
shared-WTA assembly + a magnitude readout), NOT on the CAPABILITY. The named surpass, not deferred: an OPPONENT /
columnar assembly with separate comfort- and discomfort-selective sub-pools (the `_affect_composed_selforganized_opponent`
structure already in the affect lane) that PRESERVES the sign structure a zero-FP discrimination needs -- a
false-grounded neutral drives BOTH sub-pools, a true affect drives ONE, so the opponent read distinguishes them where
the shared-WTA magnitude read cannot.

## The scoped next steps (named, not deferred)
1. **Harvest the queued 6-seed spiking verify** (the `_affect_onsubstrate_noise_robust_convergence_6seed` artifact under
   the raw findings dir, written by the queued run) -> confirm the boundary is robust across seeds (or find a clearing
   seed), and finalize this finding.
2. **The named surpass: an OPPONENT / columnar spiking assembly** -- separate comfort- and discomfort-selective
   sub-pools (reuse the `_affect_composed_selforganized_opponent` opponent structure) so the read preserves the sign
   structure the zero-FP discrimination needs; re-test on the full partition.
3. **A real grounded world** (coverage) + production wire-in + a production lesion test -- the standing arc residuals,
   after a spiking read that clears the bar.

## Honest scope + residuals
Additive, default-OFF (`--spiking` opt-in; OFF = the imported numpy GO verbatim, byte-identical), NO sim/ edit,
reuse-by-import; `_STRONG_MARGIN==2.0` asserted, nothing wired. (1) the body-state US is a declared ORACLE STAND-IN
(the SAME stand-in the numpy GO used). (2) companions 1+2 (pooling + noise-floor) are at the world/body relay boundary
(host-legit US delivery); the COMPETITIVE CONVERGENCE + the divisive-norm soft-WTA + the code READ run on the substrate;
the three-factor US gate is applied as drive-scaling; homeostatic synaptic scaling is available but default-off (probed
to over-suppress this point). (3) the ceiling is a linear supervised upper bound. (4) the 164-word closed partition is
inherited. (5) the clean/full (sigma=0) point additionally does not transfer to spikes because the noise-floor
over-subtracts the arousal channel at high coverage and redundant clean input collapses the soft-WTA -- a second
measured spiking/idealization boundary, secondary to the zero-FP structural one. (6) this finding's numbers are 1-seed
(full partition) + probes; the 6-seed is queued to confirm the boundary robustly. (7) the spiking operating point
(POP / ff_init / perc_scale / nmda / n_fs / epochs) is documented and probed, never hand-set to the labels.

## Sources (external -- deep_research_at_wall, lane affect-learned-gate-retirement; reused from the numpy GO + the affect lane as apt)
- Carandini & Heeger (2012, Nat Rev Neurosci) "Normalization as a canonical neural computation" -- divisive
  normalization / the shared-FS soft-WTA (competition on the assembly side).
- Grossberg (1973) on-center/off-surround competitive networks; Wong & Wang (2006, J Neurosci) -- the
  recurrent-inhibition soft-WTA the shared FS pool realizes.
- Turrigiano (2008, Cell) "The Self-Tuning Neuron" -- homeostatic synaptic scaling / the intrinsic homeostatic
  threshold the output-floor lever tested.
- Namburi, Tye et al. (2015, Nature) -- opponent valence populations bound to a real US: the biological basis for the
  named OPPONENT-assembly surpass (separate comfort/discomfort pools preserving the sign structure).
- Fremaux & Gerstner (2016, Front Neural Circuits) -- neuromodulator-gated three-factor eligibility (companion 3).

## Reproduce
```
# byte-identical-off smoke (default; delegates to the numpy GO verbatim):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_onsubstrate_noise_robust_convergence_derisk --smoke
# 1-seed FULL partition (the honest evidence; a tiny --smoke slice is misleading under zero-FP):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_onsubstrate_noise_robust_convergence_derisk --spiking \
    --seeds 42 --out _affect_onsubstrate_noise_robust_convergence_1seed_full.json
# the 6-seed spiking verify (GPU -- QUEUED on gpu_queue.sh, runs after the mouth train; basename shown):
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._affect_onsubstrate_noise_robust_convergence_derisk --spiking \
    --seeds 42 43 44 100 101 102 --out _affect_onsubstrate_noise_robust_convergence_6seed.json
```
