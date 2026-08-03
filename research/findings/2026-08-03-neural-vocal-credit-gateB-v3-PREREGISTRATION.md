---
type: preregistration
status: live
date: 2026-08-03
mechanism: neural-vocal-action-credit-v3
runner: research/runners/_vocal_action_credit_gate_v3.py
---

# Vocal credit Gate B v3: omission error and local critic normalization

**Filed before any v3 scientific seed was constructed or run.** Implementation
checks may use only smoke seed `0`, which is outside every formal partition.
The first permitted scientific work is calibration on seeds `401` and `409`.

## Why v3 exists

Gate B v1 showed that executed-action collateral creates genuinely local
eligibility, but a shared raw reward burst can reinforce whichever action
happened to precede it. V2 added action-conditioned value populations and
direct value-to-SNc GABA-B inhibition. In its just-completed calibration, both
clean repeats at source `4fddb43e` ended with action-0 preference `1.0` under
contingent reward. Under yoked reward, seed 7 ended at `0.0` and seed 11 at
`1.0`: both were maximally far (`0.5`) from balanced, but in opposite arbitrary
directions. Omitted reward produced no meaningful dopamine dip. The critic
therefore did not prevent a seed-dependent self-reinforcing policy under
noncontingent feedback.

V3 does not weaken the yoked criterion or search another v2 GABA-B gain. The
v2 value-to-SNc weight and GABA-B propagation strength remain unchanged.

## Biological mechanism

V3 adds two local spiking circuits to the existing actor-critic bridge.

1. Each executed motor population drives its own striatal fast-spiking (FS)
   interneurons. Those interneurons inhibit that action's value population.
   This feed-forward normalization is intended to keep critic firing graded as
   the plastic motor-to-value route strengthens, instead of allowing a hot
   critic to become an all-or-none clamp.
2. Each action-value population inhibits a tonic omission gate through slow
   GABA-B. At the generic outcome time, a shared outcome population excites an
   LHb-like omission population. Learned expectation disinhibits that LHb
   population. If reward is absent, LHb excites GABAergic RMTg, which inhibits
   SNc and should produce a dopamine dip. If reward is present, the shared
   reward-US population recruits a GABAergic reward-veto population that
   suppresses LHb, while the reward-US-to-SNc path can still produce a burst.

The design follows three primary-source constraints already documented in the
repository:

- Eshel et al. (2015), *Nature* 527:398: expected reward is subtracted from
  dopamine responses by local VTA GABA circuitry; the operation is
  subtractive rather than divisive.
  https://www.nature.com/articles/nature14855
- Matsumoto and Hikosaka (2007), *Nature* 447:1111: lateral habenula carries
  worse-than-expected and reward-omission signals.
- Hong, Jhou, Smith, Saleem, and Hikosaka (2011), *J Neurosci* 31:11457: the
  negative LHb signal reaches dopamine neurons through GABAergic RMTg.
  https://www.jneurosci.org/content/31/32/11457

This is a functional point-neuron approximation, not a claim that the small
populations reproduce full LHb, RMTg, striosome, or VTA microanatomy.

## Host boundary

The host may present one shared cue, start a generic outcome window, report
whether a sensory reward occurred, observe a neural motor threshold crossing,
and record spikes. The reward contingency is an environmental consequence of
the emitted action; it is never injected into an action channel.

The host may not stimulate a desired channel, assign eligibility, set
dopamine, choose a neural winner by argmax, calculate a prediction error, or
write a synaptic update. Actor and critic eligibility must arise from local
coactivity. The positive and negative teaching signals must arise from the
spiking SNc and LHb-RMTg circuits.

## Seed lock

- Smoke only: `0`.
- Calibration, currently open: `401`, `409`.
- Development, locked: `419`, `421`, `431`, `433`.
- Held out, locked: `439`, `443`.

These sets are mutually disjoint and do not overlap Gate B v1/v2 seeds or the
reserved replay (`228-231`, `326-329`), source (`232-235`, `330-333`), and
visual (`224-227`, `322-325`) partitions. No declared formal v3 seed had been
exercised when this preregistration was written, so no rotation was necessary.

## Fixed protocol

Each calibration seed runs separate, identically initialized brains for:

1. contingent reward;
2. reward-count-matched, trial-shifted yoked reward;
3. executed-action collateral lesion;
4. reward-US-to-SNc dopamine-path lesion;
5. critic-output lesion, cutting value-to-SNc and value-to-omission routes;
6. LHb-RMTg omission-path lesion; and
7. local critic-normalization lesion.

All arms use 20 frozen baseline trials, 40 training trials, two frozen outcome
probes (rewarded and omitted), and 40 frozen evaluation trials. The yoked,
critic-lesion, and omission-lesion arms receive exactly the contingent arm's
reward count with the same fixed schedule rotation. Do not tune between seeds,
drop a failed seed, or inspect development or held-out seeds after a calibration
failure.

## Fixed validity preconditions

A seed is `UNDEFINED`, not a pass or fail, unless every condition satisfies all
of these:

1. All actor, critic, FS, omission-gate, LHb, reward-veto, RMTg, and SNc
   populations share one bridge.
2. Only cue-to-actor and motor-to-value routes are plastic.
3. Outcome and reward afferents are shared and have no action-channel target.
4. Both local FS normalization circuits and the complete expectation-gated
   LHb-RMTg-SNc path are physically present.
5. The v2 value-to-SNc weight and GABA-B propagation strength are unchanged.
6. Every baseline has at least `0.90` clean neural commits.
7. Every yoked arm preserves the contingent reward count.
8. SNc tonic calibration is finite and positive; generic outcome and neural
   reward-veto populations spike in their declared probe windows.
9. Every requested lesion gate is actually cut and plasticity-scope telemetry
   is present.

## Fixed scientific criteria

The numeric thresholds below are frozen as constants in
`research/runners/_vocal_action_credit_gate_v3.py`.

Every criterion must pass on both calibration seeds:

1. Contingent frozen evaluation chooses action 0 on at least `0.90` of clean
   trials, with at least `0.90` of choices made from the learned cue before
   shared arousal.
2. Yoked action-0 preference remains in the unchanged neutral interval
   `[0.25, 0.75]`. Directional dominance toward either action fails.
3. At least `0.90` of rewarded trials have an executed-to-losing actor
   eligibility ratio of at least `10`, and at least `0.90` of training trials
   with critic activity favor the executed action's value population.
4. Collateral and reward-US-to-SNc lesions prevent action-0 acquisition; the
   collateral lesion leaves actor routes unchanged.
5. A frozen expected-omission probe produces LHb and RMTg spikes and lowers
   dopamine by at least `0.001` below its pre-outcome concentration. <!--derived-->
6. A frozen rewarded probe activates reward veto, has fewer LHb and RMTg spikes
   than omission, and raises dopamine by at least `0.001`. <!--derived-->
7. Critic-output and omission-path lesions each reduce the omission dip by at
   least `0.0005`; the omission-path lesion produces zero RMTg spikes. <!--derived-->
8. Intact local FS populations spike during training. Their lesion silences FS
   activity and raises mean critic spikes by at least `1.20x`.
9. No synapse outside the declared actor and critic routes changes.

The calibration verdict is GO only if both fresh seeds pass every validity and
scientific criterion. Any failure is NO-GO and leaves development and held-out
seeds locked.

## Honest scope and successor rule

The fixed connectivity, host-defined trial boundaries, global plasticity-gate
windows, and environmental reward rule remain scaffolds. The new circuit only
tests whether a small spiking actor-critic can distinguish contingent from
yoked consequences with a causally attributable omission dip. It does not
establish natural vocal learning, social understanding, language, or general
agency.

If v3 fails, diagnose which prerequisite failed: critic firing range,
expectation persistence to outcome, reward veto timing, LHb-RMTg transmission,
or actor use of signed dopamine. Do not relax yoked neutrality, choose a
preferred action after seeing results, or retune the v2 GABA-B gain against the
same seeds. A successor must state a new mechanism and use another fresh seed
partition before execution.
