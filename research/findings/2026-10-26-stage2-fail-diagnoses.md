---
type: finding
status: live
date: 2026-10-26
lane: integration-first (Stage-2 exit, re-attempt due 11-21)
mechanism: diagnosis of the four Stage-2 exit failures (C2, C3, C4) and the BG step-1 membrane breach, each verified by an independent skeptic
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/2026-10-26-stage2-exit-gate-RESULT.md
  - research/findings/raw/_stage2_exit/gate/aggregate.json
  - research/findings/raw/_stage2_exit/gate/s42/ask__intact_a.json
  - research/findings/raw/_stage2_exit/gate/s42/emo__intact_a.json
---

# Why the Stage-2 exit attempt failed: four diagnoses (2026-10-26)

Claude ran one diagnoser per failure (code plus the recorded gate shards at pin 9bdb92ee8, plus small CPU probes),
then one skeptic per diagnosis that re-checked every citation and tried to refute it. All four diagnoses SURVIVED.
The skeptics corrected parts of each; the corrected statements are the ones below. Probe code, probe outputs and the structured
reports are kept on the workstation (folder diag_1026 under sim-rev), diagnostic and uncommitted.
The gate data read are research/findings/raw/_stage2_exit/gate/aggregate.json and the per-seed shards beside it
(for example research/findings/raw/_stage2_exit/gate/s42/ask__intact_a.json). No gate-seed run was made for selection.

## The common cause behind C4 and the BG breach: the 1 ms explicit-Euler step

Two of the four defects are the same family. The substrate advances membranes with an explicit Euler step at
dt = 1 ms. A neuron whose heterogeneity draw gives it a low capacitance becomes numerically unstable under strong
conductance. Its voltage leaves the physiological range (below -110 mV, down to about -1900 mV) and it emits spikes
that biology would not produce. Those spikes then change what the organ decides. This is the instrument (the
integrator) acting as part of the emulation, not a property of the cells. The real membrane is stable: the
continuous-time cell at rest has eigenvalues of about -0.003 and -2.26 per ms (workstation probe). <!--derived-->

## C4 (emotion): seed 101 never names joy

- **Cause (verified, reproduced on a second host).** The emotion-category organ's draw gave sadness-appraisal
  neuron 405 a capacitance of 53.64 pF (mean about 100). When the joy channel wins, its inhibitory pool drives that
  neuron's GABA conductance to about 100-136 nS. With g_i*dt/C above about 2, the explicit step overshoots
  E_i = -75 mV and oscillates with growing amplitude (-96, -23.6, -116, +18, ... -137 mV). The neuron emits five
  spurious spikes (steps 51-67). They recruit the sadness quench pool (steps 89-105), which silences the joy rungs
  (74/49/8 spikes during exposure, 0/0/0 in the last 20 settle steps), so joy never latches.
- **Single-neuron counterfactual.** Changing only C[405] on s101 flips the outcome between 53.0 pF (joy None) and
  56.0 pF (joy named, 93 slot spikes).
- **It is a near-certain miss, not a near-miss.** Readout, gate and slot counts are exactly 0 on s101 joy (s42: 69/63/83
  against a latch bar of 18). The other three channels on s101 are normal.
- **Family.** The same excursion is present at all five pinned failures found: s101, s202, s242, s308, s348. The
  pinned organ fails about 4% of builds (held-out seeds 301-360: 58/60). Excursions are common and usually harmless:
  s42 passes with two. A failure needs the excursion plus a spurious spike that recruits the quench.
- **The C4 lesion and permuted shortfalls are the same event,** not independent failures.
- **Fix candidate (organ level only).** The repo already has the guard for this overshoot class:
  `cfg.enable_bounded_conductance_integration`, default OFF, used by other organs. The emotion organ never set it.
  With it set at construction: s101, s202 and s242 pass intact and permuted, the out-of-range count goes to 0, the
  held-out battery is 60/60 and dev 1-8 is 8/8. NOT yet run through the live chat handler.
- **The mechanism label is conductance-term overshoot.** It is not an Izhikevich quadratic-term v_crit; the
  diagnoser's first label was wrong and the skeptic corrected it.
- **Fear reads 'neutral' on 6/6 for a separate reason, and the scorer excludes it from `all`.** A host dead zone
  (LADDER_NEUTRAL_TOL = 0.03) is applied to the spiking ladder's differential. Fear lands on the ladder's first rung
  (about -0.021), which sits inside the dead zone. The brain's own opponent pools have the right sign on 5/6 seeds
  (fear negative pool 66-98 spikes, positive pool 0; s102 only 3). The fear mood itself is host arithmetic: a
  Warriner/DR-2 word mean through a host EMA, identical (-0.3587; s42 emo shard) on all six seeds. This is a host-decided label,
  not a C4 pass condition.

## BG step 1: the membrane breach on dev seeds 72, 75, 77

- **Cause (verified by bit-exact single-cell replay, max |dv| = 0).** One striatal MSN per breaching seed drew a
  capacitance about half the 50 pF preset: s72 idx 439 (str_d2) 24.42 pF, s75 idx 145 (str_d1) 18.43 pF, s77 idx 418
  (str_d1) 24.12 pF. The Euler map's rest multiplier is 1 - 55/C, so C below 27.5 pF is unstable at rest and below
  45 pF under strong GABA_A. A selector_reset GABA_A volley during the build's commit-settle steps (g_i about 145 nS)
  starts a growing period-2 oscillation (s75 starts on its own). The cell then locks into an absorbing two-step
  cycle (spike, reset to -55 mV, deep dip, spike) and fires every second step for the organ's lifetime. The
  recovery variable u runs away through b < 0, which sets the dip depth (predicted -1928/-1624/-438 mV, observed
  -1918/-1633/-460). The cached organ carries the cycle across turns, which is why the minima are identical across
  the eight tonic sets.
- **What is necessary.** Bounded conductance integration is ON in this organ and is needed (without it 207-234
  cells breach per seed). It bounds the synaptic term but not the -u/C or quadratic terms, where the runaway is.
  The refractory timer and b are not necessary.
- **Hermes's observe run** (`research/coordination/bg-breach-observe-RESULT.md`) found the same cell (s72
  str_d2_1 idx 439) independently.
- **Fix candidate (dev seeds 71-78 only).** Sub-stepping the membrane update (4 steps of 0.25 ms) keeps every
  membrane in [-90.4, +40.0] mV on 8/8 dev seeds with 0 breach cells. Not built yet.
- **Seed note.** A build-only parameter read puts the gate seeds' lowest MSN capacitance at 25.8-29.9 pF, above
  every observed lock-in threshold except borderline s102. So the breach may be mostly a property of the dev block
  71-78. The defect is still latent substrate-wide (FS cells draw 11-20 pF). The seed-to-cell map is specific to
  the numpy CPU random stream.

## C2 (inference): the permuted exposure was never acquired

- **Cause of 0/6 (recorded on 7/7 seeds).** The first permuted teach, 'the heron hunts the gnat', is refused by the
  default-ON D4 comprehension gate before it is stored. The gate treats a sentence as fully out-of-vocabulary when
  its verb is outside the hand VERB_SELECTS table ('hunt' is absent) and both nouns are absent from the hand
  ANIMACY table and the stored vocabulary (heron, gnat). The spiking margin reads 0.0 against a 0.2 threshold and a
  host-lexical clarification replaces the teach. Only 8 facts are stored instead of 9. The intact teach contains
  'frog', a hand-ANIMACY word, so it is outside the gate's scope and is stored.
- **What is not yet shown.** That the permuted arm would give 'frog' if the teach were stored has been shown only on
  a stub composer. Claude is running the decisive check on dev seed 7 (the permuted arm with
  BRAIN_COMPREHENSION_GATE=0).
- **The same gate refuses other valid teaches:** 'the goat eats the hay' in all nine ask arms (role branch, margin
  0.01667 on s42) and 'the crab eats the leaf' in ask permuted. This is why every C3 seed lists m_goat as an undefined moat.
- **Who decides the conclusion in the passing arms: host code, as the preregistration declared.** The possessive
  chain route reads both hops by scanning a Python list (`OneBrainComposer.kb`, `_distinct_patients`). The spiking
  `query_patient` is only a fallback that the production composer never reaches. Evidence: under qp_lesion the
  answer, derived flag and derived_from are unchanged on 7/7 seeds. No intervention on the list has been run yet
  (the planned kb_scramble arm). The module docstring still says the hops are spiking reads; that is stale.
- **Instrument notes.** `score_infer` has no acquisition precondition, unlike the ask group's A4 rule, so an
  acquisition refusal is scored as 'not exposure-driven'. The `transitive_chat` informational arm is void: its flag
  is already default-ON at the pin and the route sits only on the single-fact path, never on the production rich
  path. All infer arms are identical across the seven seeds in every non-text field, so 6/6 is one deterministic
  measurement repeated, not six replications.

## C3 (asking): a session-global host gain shuts asking off, and there is no per-topic stop

- **Layer 1 (the measured cause of most clauses; replayed 693/693 decisions with 0 mismatches).** Every ask decision
  is `want_hz * clip(0.2, 3.0, 1 + 1.5*(DA - 0.5)) >= threshold` in host code (webapp/da_curiosity_drives_chat.py).
  want_hz (38.2-38.9 Hz) and the threshold (19.10 Hz) are constant. DA is a session-global host EMA of a host
  engagement mix. The probes all have the form 'what does the X chase', and the carrier word habituates, so DA falls
  to its floor (0.046163 on s42) within 4-6 probes. At the floor the gain is 0.31924 and the effective want is 12.4 Hz, below
  threshold, so nothing is asked. The ask cutoff is DA about 0.161 (computed from want and threshold). <!--derived--> This causes lp_learnable 0/6, the reversed gap
  rate 0/6, the unasked first toad probe, and the s102 discrimination miss. It also makes reversed lp_unlearnable
  pass 6/6 for the wrong reason.
- **Layer 2.** With the gain pinned (the da_lesion arm, engaged on 11/11 traces), toad is asked on all four probes on
  6/6 seeds, so lp_unlearnable still fails. The production handler has no per-topic stop. An organ-level, count-based
  per-topic stop exists (method 2, noise-off, dev 8/8 provisional) and is not integrated.
- **Answer rate.** Pinning the gain changed abstained/recalled_svo on 0 of 133 turns; it changes only the appended
  follow-up question.
- **Correction to the RESULT doc.** It lists the global DA threshold as a 'known architecture gap (not the measured
  cause)'. The data show it IS the measured cause of every C3 clause failure except the per-topic stop.
- **Host share.** The ask/no-ask decision on every C3 probe is host arithmetic over spiking inputs. The
  preregistration's host audit does not list the gain, so C3's host-decision share is under-reported.

## What follows (ruling stage2-fixes-1026)

The verdict of the 10-26 attempt stands. These are methods for the 11-21 re-attempt, in order of readiness:
- **C4.** Build a default-OFF flag that sets bounded conductance in the emotion organ. Then the handler-level s101
  pair, the held-out battery, a quench-necessity arm, and flag-off byte identity. Flip to default-ON before the
  re-pin only if all hold. The fear-sign dead zone goes in a separate, lower-priority packet.
- **BG.** Build a default-OFF membrane sub-step option in the substrate. Then the step-1 grid is re-run on the step-1
  block 71-78; gate seeds are never run in step 1.
- **C2.** First the dev-7 gate-off check and a kb_scramble arm (causal proof of who owns the conclusion). Then two
  separate methods:
  - acquisition of well-formed novel transitives through a learned word-order cue, not by adding words to a table
    (this is the C1 word-role family, upstream of which is the cs-race persistent-trace fix);
  - hop reads moved onto the spiking substrate, with a multi-valued-hop abstain arm.
- **C3.** First the gain-pinned baseline in both orders on dev seeds. Then integrate the noise-off per-topic stop
  behind a default-OFF flag. Then replace the host gain with a topic-locked phasic DA read, with its tonic term and
  sensitivity registered before data.
- **Substrate.** The integrator defect class may be latent in other organs that combine strong inhibition with
  low-capacitance draws. An out-of-range membrane census across production organs is owed before any claim about
  them.
