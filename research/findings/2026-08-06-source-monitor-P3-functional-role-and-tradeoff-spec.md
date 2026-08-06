---
type: research-gate
status: active
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency
---

# Source monitoring: whole-brain functional role and the acceptable no-harm tradeoff (P3 spec)

This document resolves the gate ROADMAP blocker #3 sets before any further source-monitoring version runs:
"the acceptable biological tradeoff must be specified from source monitoring's role in the whole brain." It is a
specification, not a measurement. Where it reads a number off a prior run it cites that run's artifact and treats it
as prior evidence, never as tuning data for a future seed.

## 1. What source monitoring must provide to the whole brain

The source pathway is not a second fact database. Its job is to reinstate, from a live cue alone, whether a recalled
item was **seen**, **heard**, or **self-generated**, and to hand that reinstated source activity to ACC/aPFC and the
self-schema so the conversation path can **assert, hedge, or abstain honestly** before speech rendering. Two of our own
findings fix this role:

- The self-schema honesty wire-in reads a source-consistency signal and, when the source echo disagrees or is
  unavailable, floors confidence rather than asserting; it is monotone-downgrade only and can never create content
  (`research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_6seed.json`).
- The plastic-source-memory research gate states the same role: source memory "reinstates source-linked answer activity
  from the live cue, then sends agreement or conflict toward ACC/aPFC and the self-schema," and the self-schema "may
  downgrade speech ... it may never create an answer" (2026-08-03-laneC-plastic-source-memory-research-gate.md).

Three properties follow directly, and they define what "harm" means:

1. **It is a threshold read, not a precise magnitude report.** Downstream honesty asks two things of the source margin
   `M_s = rate(correct source) - max rate(rival source)`: is the correct source dominant (right attribution), and is
   its margin above the decision floor `F` (trust it) or below (hedge/abstain)?
2. **It fails closed.** The wired-in path already floors confidence when the source signal is weak or absent. Over-
   abstention on a genuine memory is the tolerated error; a confident wrong attribution is the intolerable one.
3. **Whole-brain reliability is set by the WEAKEST source, not the strongest.** A single source that dips below `F`
   causes a genuine seen/heard/self memory to be misread or needlessly hedged. Separation an already-strong source
   holds ABOVE `F` buys nothing extra downstream once it exceeds the floor: the honesty gate treats "well above floor"
   and "far above floor" identically.

## 2. The acceptable tradeoff

Property 3 is decisive. Because the downstream decision is limited by the minimum source margin, the correct objective
for any stabilizing mechanism is **max-min**: raise the weakest source above `F` while keeping every source above `F`
and preserving source ordering. A per-source **zero-degradation** rule is stricter than the whole-brain role justifies
and actively forbids the redistributive mechanism biology uses. Lateral biased competition and divisive normalization
work by trimming a strong representation to lift a weak one; demanding zero cost to the strongest bans exactly the
motif that makes the three source outputs comparable and decision-ready. An unbounded average-benefit rule is equally
wrong: it can hide one source falling below `F`.

The acceptable tradeoff is therefore a **bounded-loss, guard-the-floor, max-min** rule. With floor `F = 0.15` (the
inherited functional floor, unchanged), mechanism margin `M_s`, and matched-mechanism-lesion margin `L_s`:

```
loss_s             = max(0, L_s - M_s)     # how much the mechanism weakened source s
spendable_surplus_s = max(0, L_s - F)      # how far source s sat above the floor before the mechanism
```

A mechanism is acceptable on a seed iff:

- **A. Floor held:** every `M_s >= F`. No source may end below the decision floor.
- **B. Only surplus is spendable:** `loss_s <= spendable_surplus_s` for every source. An above-floor source may give up
  only what it holds above `F`; a source at or below `F` may not be weakened at all.
- **C. The minimum strictly improves:** `min_s M_s > min_s L_s`. The mechanism must actually raise the weakest source,
  not merely reshuffle thresholds.
- **D.** No source ordering inversion (the correct source stays dominant for each recall), plus all inherited causal and
  anti-cheat controls.

<!--derived-->
This is not a new invention. The v3 calibration preregistration already stated exactly this bounded-loss rule (criteria
1-3 there), filed before any v3 evidence, with the explicit note that "V2's observed -0.0092 change is neither encoded
nor used." This spec adopts that rule as the standing functional criterion and grounds it in the whole-brain role
above. What was missing was not the criterion but the recognition of which mechanism already meets it.

## 3. Re-adjudication: v2's biased competition already satisfies this criterion

<!--derived-->
Every number in this section is recomputed from the two cited v2 calibration artifacts; it is a re-scoring, not a new
run. The v2 lesion margins L are recovered from the finding's reported competition-margin gains.

The v2 local fast-spiking GABA-A competition circuit was recorded NO-GO only because it failed a per-source
zero-degradation control: seed 217 lost `0.0092` of an already-strong self-generated margin
(`research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed217.json`). Re-scored against the
bounded-loss rule of section 2, using the margins in that finding and its seed-216 sibling
(`research/findings/raw/parallel_gates/source_monitor_coresidency_v2_calibration_seed216.json`):

| seed | M seen/heard/self | L seen/heard/self | min M | min L | max loss vs surplus | verdict under sec 2 |
|---:|---|---|---:|---:|---|:---:|
| 216 | .1683/.2400/.1508 | .1458/.2183/.1508 | .1508 | .1458 | loss 0 | passes A-D |
| 217 | .1733/.1583/.2067 | .1325/.1358/.2158 | .1583 | .1325 | loss .0092 <= surplus .0658 | passes A-D |

<!--derived-->
Both seeds hold the floor, spend only surplus, strictly raise the minimum margin (216: .1458->.1508; 217:
.1325->.1583), and never invert ordering. The `0.0092` reduction on seed 217 is precisely the surplus redistribution
that lifts the weakest source by `0.0258`. Under the criterion the whole-brain role demands, v2's mechanism is a pass,
not a failure. It was rejected by the wrong control.

The two subsequent versions searched for a mechanism with literally zero cost to any source, which the role shows is
unnecessary, and both failed for reasons unrelated to the tradeoff: v3 intrinsic-threshold homeostasis produced zero
margin gain and failed criterion C
(`research/findings/raw/parallel_gates/source_v3_seed232.json`,
`research/findings/raw/parallel_gates/source_v3_seed233.json`); v4 inhibitory STDP was inert, with intact margins
identical to the learning-lesion and rival burden `0.0`, and returned UNDEFINED on a guard bug
(`research/findings/raw/source_monitor_coresidency_v4/calibration.json`). This is the "the wall was smaller than it
felt" pattern: the arc over-searched for a new mechanism when the first one already met the correct requirement.

## 4. Consequence for the next version

The next version does not need new biology. It needs to test whether v2's already-causal, already-lesionable biased-
competition circuit satisfies the bounded-loss max-min criterion across FRESH calibration, development, and held-out
seeds — because the v2 seeds (216, 217) are now observed and cannot be re-scored into a promotion. That test is filed
as the v5 preregistration (2026-08-06-source-monitor-coresidency-v5-calibration-PREREGISTRATION.md): identical v2
mechanism, the section-2 bounded-loss acceptance rule fixed in advance, fresh seed partition, all inherited controls
retained. If a stronger guard band is later wanted, it must be justified from the downstream honesty threshold's noise,
measured, and preregistered — never chosen to make a result pass.

## 5. Scaffolds that remain regardless of outcome

A v5 pass would still leave caller-supplied sparse episode activity, predefined source-afferent identity, hand-wired
competition, externally timed learning windows, and host spike-count evaluation on the scaffold ledger. It would not
claim learned source pathways, natural episodic allocation, language, truthful speech, or a complete self-model. It
establishes only that a co-resident spiking source-monitor can deliver a reliable, comparable seen/heard/self
confidence into the honesty path under the tradeoff the whole brain actually requires.
