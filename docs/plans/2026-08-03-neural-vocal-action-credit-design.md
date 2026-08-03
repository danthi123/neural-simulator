---
type: plan
status: active
date: 2026-08-03
---

# Neural Vocal Action Selection And Credit

## Purpose

The immediate blocker is not vocabulary size. The brain cannot yet reliably
tell which of several self-generated vocal actions should receive credit from a
later social consequence. Until that works, adding words or a larger language
circuit would scale an unstable learning loop.

This increment will build a small, fully neural selector whose executed action
is distinct from losing candidates and whose local synapses alone retain
eligibility for a later dopamine signal. It must work before it is connected to
the full two-intent by two-referent reversal task.

## Observed Boundary

The first shared-arousal implementation passed only one of four development
seeds. Its recorded decision events showed near-universal co-firing of both
competitors in each factor bank. Three conclusions follow:

1. More exploration current cannot solve the attribution problem.
2. Global dopamine is usable only when recent synaptic eligibility is local to
   the action that actually won.
3. A host-selected action, channel-specific reward injection, or Python latch
   would hide the missing brain mechanism and is forbidden.

## Biological Basis

- A dedicated anterior forebrain pathway generates vocal variability in young
  songbirds; silencing its LMAN output sharply reduces exploratory song
  variability ([Olveczky, Andalman & Fee 2005](https://doi.org/10.1371/journal.pbio.0030153)).
- The same broad circuit can alter vocal output moment by moment, supporting a
  separation between exploratory variability and the stable motor pathway
  ([Kao, Doupe & Brainard 2005](https://doi.org/10.1038/nature03127)).
- Contingent reinforcement can shape naturally occurring vocal variation,
  whereas non-contingent feedback does not produce the same adaptation
  ([Tumer & Brainard 2007](https://doi.org/10.1038/nature06390)).
- Area-X-projecting dopamine neurons encode better- and worse-than-expected
  vocal outcomes at the relevant moment
  ([Gadagkar et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5464363/)).
- Manipulating VTA input to the song basal ganglia is sufficient to guide
  syllable-specific learning
  ([Hisey et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29483664/);
  [Xiao et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29551492/)).
- Corticostriatal synapses can retain a silent, seconds-long local eligibility
  trace that later dopamine converts into plasticity
  ([Shindou et al. 2019](https://doi.org/10.1111/ejn.13921)).

These studies constrain the design; they do not establish that the exact
proposed simulator circuit exists as written in an animal brain.

## Proposed Circuit

Keep the existing shared arousal population as a practice-state signal. Replace
its direct drive into vocal outputs with three parallel two-channel selection
loops: speak/silence, intent 0/1, and referent 0/1.

Each channel contains:

```text
state or perception cortex + variable premotor input
    -> D1 and D2 striatal populations
    -> GPe / STN / GPi competition
    -> thalamic disinhibition of one channel
    -> vocal motor output
    -> action collateral back to that channel's local eligibility population
```

The direct and indirect pathways provide selection and suppression. The
thalamic return sustains the winner long enough to commit an action. Only the
winning vocal output sends an action collateral to its own channel. State or
perception synapses onto that channel can then carry a strong local eligibility
trace. The listener's consequence remains a sensory event that changes the
shared SNc/RMTg dopamine signal; it never identifies the desired channel.

The three selectors remain factorized so a learned intent and referent can form
combinations absent from training. The host may present the world, deliver the
listener's real consequence, and measure spikes. It may not choose a channel,
reset a winner, inject output current, or scope dopamine by the answer.

## Cheap-First Gates

### Gate A: Selection Physiology

Build one isolated two-channel selector on the production bridge and run seeds
42, 43, 44, and 100 before any convention learning.

GO requires all of the following at every seed:

- exactly one thalamic/motor channel commits on at least 95% of trials;
- loser motor spikes are at most 25% of winner spikes on at least 95% of
  committed trials;
- each channel wins at least 25% of 100 target-independent exploration trials;
- removing shared arousal sharply reduces exploration;
- removing GPi-to-thalamus disinhibition prevents commitment;
- no host channel input, host argmax decision, or output current is present.

If this fails, stop at the selector. Do not add the language task or tune on
held-out seeds.

**Gate A v1 result, 2026-08-03:** NO-GO. Seeds 43, 44, and 100 passed; seed 42
produced 92% clean commits against the fixed 95% requirement. Choices were
balanced, loser spikes were zero at commit, and both the arousal and direct-path
lesions reduced commits to zero in every seed. Seed 42's omissions showed
bilateral striatal-interneuron activity with too little D1 activity to release
thalamus. Revise that competition as a new version and rerun all four
development seeds. Gates B and C and held-out seeds 101/102 remain locked. See
[`research/findings/2026-08-03-neural-vocal-selector-gateA-4seed-NO-GO.md`](../../research/findings/2026-08-03-neural-vocal-selector-gateA-4seed-NO-GO.md).

**Gate A v2 preregistration:** remove the two striatal FSI populations and their
proposal, cross-MSN, and reset pathways. Keep every current, remaining weight,
population size, decision rule, threshold, duration, seed, and GO criterion
unchanged. V1 showed that when both candidate channels recruited their FSI
pools, both D1 routes could be suppressed and no action occurred. V2 tests the
narrow hypothesis that downstream commit competition is sufficient and that
the extra striatal competition is counterproductive at this scale. The runner
preserves v1 and exposes v2 through `--selector-version v2`; the topology change
reduces the probe from 632 to 600 neurons and from 44 to 36 declared pathways.
The full v2 evidence must rerun all 100 main, 100 no-arousal, and 100
direct-path-lesion trials for every development seed before Gate A can pass.

**Gate A v2 result, 2026-08-03:** GO. All four development seeds passed the
unchanged criteria with 98-100% clean commits, at least 32.7% of commits from
the smaller channel, zero losing motor spikes at commitment, and zero commits
under either lesion. Gate B is unlocked for development seeds 42, 43, 44, and
100. Held-out seeds 101 and 102 remain locked until Gate C. See the
[`Gate A v2 finding`](../../research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md).

### Gate B: Local Credit

Add plastic cue-to-striatum routes and delay the consequence beyond action
commitment.

**Gate B v1 calibration boundary:** calibration may use seeds 7 and 11 only.
Add one shared cortical cue, one D1 actor/eligibility population per action, a
local motor-to-actor collateral, fixed actor-to-GPi direct routes, and a
reward-US-to-SNc pathway producing one shared dopamine broadcast. Only the two
cue-to-actor pathways are plastic. Cue input is identical for both actions;
the host may respond to an observed motor action but may not inject, tag, or
reward a neural channel directly. Tune population sizes, weights, and time
constants on calibration seeds, then freeze them before running development
seeds 42, 43, 44, and 100. Do not inspect seeds 101 or 102.

The causal comparison uses four separately initialized conditions with the
same seed: contingent reward for action 0, the contingent run's reward schedule
circularly shifted by one third of training so its count and temporal pattern
are preserved but trial-level action contingency is broken, action-collateral
lesion, and dopamine-path lesion.
The lesions must preserve initial selector physiology. Record eligibility
immediately before delayed reward and compare all synapses in the executed and
losing cue-to-actor routes; do not clear or assign traces by channel from the
host.

Warmup is followed by the same neural reset and washout used between trials
before any baseline measurement. Record motor and actor spikes during the cue
lead separately from the action period so startup activity or an early motor
commit cannot be mistaken for learned cue-driven bias.

GO requires all of the following at every development seed:

- executed-route eligibility exceeds the losing route by at least 10:1 before
  dopamine arrives on at least 90% of rewarded trials;
- the rewarded action becomes more likely while a yoked reward schedule does
  not produce the same preference;
- an action-collateral lesion preserves initial selection but removes the
  learned preference;
- dopamine lesion preserves selection but removes learning;
- changed synapses remain within declared corticostriatal vocal routes.

If eligibility remains nonlocal, revise the circuit rather than masking
synapses from Python.

**Gate B v1 calibration result, 2026-08-03:** NO-GO. Contingent reward produced
fully local eligibility, action-0 preference 1.00, and zero outside-route
changes on calibration seeds 7 and 11. Collateral and dopamine-path lesions
preserved selection but prevented weight change. The yoked schedule also
produced action-0 preference 1.00 on seed 11, however, because every raw reward
remained a positive dopamine event and reinforced whichever route was locally
active. Gate B therefore requires an action-conditioned spiking value critic
that subtracts expected value at SNc before development seeds are opened. See
the [Gate B v1 finding](../../research/findings/2026-08-03-neural-vocal-credit-gateB-v1-yoked-NO-GO.md).

**Gate B v4 smoke result, 2026-08-03:** Gate B v3 subsequently localized
the remaining failure to bounded action value not persisting to outcome time.
The sealed v4 candidate reused v3 and the existing graded dendritic-plateau
substrate to test an action-local expectation trace plus a symmetric generic
outcome read. The dendritic state was real and causal, but adversarial review
found a load-bearing Python winner/timing latch and label-derived selectivity
checks; the CuPy smoke also exceeded the firing ceiling because late motor
activity normalized a different value channel. V4 is retired, every formal
seed remains unused and sealed, and the successor must use a neural commit
event plus independently measured neural selectivity. See the
[v4 smoke plan](2026-08-03-neural-vocal-action-credit-gateB-v4-smoke.md) and
[NO-GO finding](../../research/findings/2026-08-03-neural-vocal-credit-gateB-v4-smoke-NO-GO.md).

### Gate C: Same-Brain Convention Reversal

Only after Gates A and B pass all four development seeds, connect three copies
to the existing grounded listener loop. Use the existing identity convention,
negative-only extinction, swapped convention, held-out cross-combinations, and
old-convention evaluation.

Development GO requires all four seeds to reach:

- initial joint and held-out accuracy 1.00;
- reversed joint and held-out accuracy 1.00 in the same brain;
- old-convention accuracy 0.00 after reversal;
- all four composite vocal actions explored;
- zero changed synapses outside declared vocal-learning routes.

Only then unlock untouched seeds 101 and 102 and run the six-seed promotion
battery: no consequence, yoked reward, dopamine lesion, arousal lesion, RMTg
error-path lesion, action-collateral lesion, context lesion, perception lesion,
and no-reason silence.

## Performance Boundary

The selector should use small sparse populations and reuse existing basal
ganglia neuron presets and pathway helpers. Record added neurons, synapses,
milliseconds per trial, peak GPU memory, and two-run concurrency throughput.
Reject a design that relies on dense all-to-all growth or makes the current
30,000-neuron bridge impractical on a 24 GB consumer GPU without a measured
scientific benefit.

Two concurrent GPU seeds are allowed after the gate is fixed; prior measurement
showed that four concurrent copies reduce sparse-matrix throughput. CPU and
mini-PC work should be limited to independent tests, artifact checks, or small
selector probes that do not require the full CuPy bridge.

## Stop Conditions

- Do not tune against seeds 101 or 102 before promotion.
- Do not proceed from physiology to learning after a failed gate.
- Do not call a one-seed success a capability.
- Do not substitute target-aware host logic for missing neural attribution.
- Record a negative result when the fixed gate fails; preserve useful generic
  simulator improvements in separate commits.
