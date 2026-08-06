---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v9
runner: research/runners/_laneC_source_monitor_coresidency_gate_v9.py
aggregator: research/runners/aggregate_source_monitor_v9_seeds.py
artifacts:
  - research/findings/raw/source_monitor_v9_generalization/development_verdict.json
  - research/findings/raw/source_monitor_v9_generalization/development_652.json
  - research/findings/raw/source_monitor_v9_generalization/development_653.json
  - research/findings/raw/source_monitor_v9_generalization/development_654.json
  - research/findings/raw/source_monitor_v9_generalization/stepping_history_confound.json
---

# v9 NO-GO: Vogels-Sprekeler inhibitory STDP is causally INERT on the margin (no recall-time rival burden); the aggregator GO is a stepping-history artifact

<!--derived-->
**Verdict: NO-GO at v9.** v9 = v6 UNCHANGED (silent-by-construction settle-to-quiescence recall, fixed local
fast-spiking GABA-A competition, all twenty frozen components + thresholds) PLUS the shipped Vogels-Sprekeler
inhibitory STDP (`cfg.enable_inhibitory_stdp`; Vogels et al. 2011, Science 334:1569) run in an explicit learning-off
recall rehearsal on ONLY the `interneuron -> rival` GABA-A synapses (scoped via the per-synapse plastic mask +
plasticity gain; Hebbian off in the window so the shipped composability validator accepts it; no `sim/` edit; the v6
build is byte-identical). The mechanism is anatomically correct and correctly scoped every seed (432 interneuron->rival
synapses touched, non-competition weights byte-identical, source thresholds unchanged, homeostasis mask stays None).
The frozen aggregator returned GO on dev seeds 652/653/654 — but adversarial verification proves that GO SPURIOUS, so
the earned verdict is NO-GO and held-out (655/656/657) stays SEALED (the verdict artifact records
`frozen_aggregator_verdict: GO`, `verdict: NO-GO`).

## The refutation (this is why it is a NO-GO despite a mechanical GO)

<!--derived-->
The criterion `weakest_source_margin_strictly_improved` = min(intact margins M) > min(competition-lesion margins L).
Control (`research/findings/raw/source_monitor_v9_generalization/stepping_history_confound.json`; per-seed rows in
`research/findings/raw/source_monitor_v9_generalization/development_652.json` .. `_654.json`): replace the iSTDP window
with an identical 5100-step window that changes
ZERO weights (Hebbian OFF, iSTDP OFF, competition ON), then run the runner's exact recall order and the frozen
criterion. It reproduces `STRICT_IMPROVED=True` and `floor_met=True` on ALL three dev seeds — identical to the iSTDP
run (`weight_delta_l1 = 0.0`):

| seed | no-window min M | no-window min L | strict | zero-weight-window min M | min L | strict | floor | iSTDP dev status |
|---:|---:|---:|:---:|---:|---:|:---:|:---:|:---:|
| 652 | .1842 | .1842 | no | .1842 | .1775 | yes | yes | DEV_PASS (spurious) | <!--derived-->
| 653 | .1608 | .1642 | no | .1642 | .1442 | yes | yes | DEV_PASS (spurious) | <!--derived-->
| 654 | .1825 | .1867 | no | .1867 | .1825 | yes | yes | DEV_PASS (spurious) | <!--derived-->

<!--derived-->
A window that changes no weights produces the same verdict as the iSTDP window, so the inhibitory-STDP weight change
is causally INERT on the criterion. The "strict improvement" is a stepping-history artifact: settle-to-quiescence does
not fully reset the Izhikevich sub-threshold state (v, u), so any pre-recall stepping shifts the margins, and because
the intact arm is measured first and the competition-lesion arm after four more intervening recalls, the two arms are
sampled at different history depths and min M drifts above min L. v6 (no window) is the honest baseline: 654 min M ==
min L (tie) → the v6 development NO-GO.

## Why the inhibitory rule is inert (mechanism-level)

<!--derived-->
The recall-time rival burden is 0 on every source, every dev seed (`recall_rival_burden = {seen:0, heard:0, self:0}`):
disjoint episode patterns + silent-by-construction settle mean that during a source's recall the RIVAL memory pools
never fire. So the margin equals the expected source's OWN firing rate — there is no rival activity for
`interneuron -> rival` plasticity to act on. This is exactly the v4 wall (2026-08-03-...-v4-calibration-UNDEFINED:
"rival spike burden was 0.0 ... the plastic inhibitory routes changed physically but did not improve the measured
source attribution"). Being homeostatic (dw = eta*(post_trace - alpha) on an inhibitory pre-spike, alpha the E/I
set-point), and seeing post_trace ~ 0 << alpha, the rule only DEPRESSES the rival inhibition (per-source L1 gain <= 0
every seed), and MOST for the STRONG source's interneurons (which fire during rehearsal) — for the weakest source it
does NOTHING (seed 654 heard L1 gain = 0.0; its own recall is too weak to even drive its interneuron). It can neither
strengthen a weak source's rival suppression nor, since rivals are silent, would that raise the margin if it could.

## This closes the homeostatic-sibling wall AND relocates it

<!--derived-->
v6 (fixed competition) redistributive-win did not generalize; v7 (threshold homeostasis) broke the competition; v8
(synaptic scaling) compressed the excitatory margins. v8's named surpass was "target the CONTRAST via inhibitory
plasticity, not the firing rate." v9 executes that surpass faithfully and shows the contrast pathway is the WRONG place
to intervene for THIS protocol: the disjoint-pattern + silent-recall design removes the between-source overlap, so
there is NO recall-time rival burden for any competition mechanism (fixed OR plastic) to reduce, and the weakest
source's margin is bounded by its OWN sub-floor excitatory recall rate — a quantity no inhibitory rule can raise. The
wall is not the competition rule; it is the protocol (no rival burden) and the weak source's own excitatory gain.

## Next mechanism (no-defer; a new method for the SAME frozen criterion)

<!--derived-->
Two candidates, both targeting the weak source's OWN excitatory recall rather than the (inert) inhibition:
(1) a BCM sliding-threshold / metaplastic selectivity rule on the `episode -> source` recall synapses that RAISES the
low-margin source's own firing gain/selectivity (increase the weak source, do not equalize rates like v8);
(2) introduce genuine episode pattern OVERLAP so a real recall-time rival burden exists that competition + iSTDP can
causally reduce — then re-test v9. Separately, the criterion has an instrument confound worth a gate: the margin drifts
under pre-recall stepping because settle-to-quiescence does not reset the Izhikevich sub-threshold state, and the
intact vs lesion arms are sampled at different history depths; a full state reset (not just quiescence) before each arm
would make the margin history-independent. The frozen criterion and thresholds were NOT loosened here.

## What was NOT done, on purpose

<!--derived-->
No frozen criterion was relaxed; the twenty v6 components + thresholds are byte-for-byte the v6 set (the iSTDP-integrity
checks are preconditions, not new components; the aggregate still requires exactly twenty). Held-out seeds remain
sealed — the mechanical GO was corrected to NO-GO in the verdict artifact after the confound control, so
`_development_is_go()` returns False. The iSTDP operating point (tau 20 ms, target-rate 0.02, eta 0.02, w in [0,6],
5000-step rehearsal) was characterized on calibration seeds 650/651 (both CALIBRATION_PASS, margins ~ v6's, rival
burden 0) and frozen; the dev seeds were run once for the recorded verdict, not tuned. The NO-GO is a verdict on the
METHOD (Vogels-Sprekeler iSTDP on the rival inhibition, under the v6 disjoint-pattern silent-recall protocol), not on
the capability of protecting the weakest source's margin. Given the v6-replay 2-seed-GO lesson, note that even a real
future GO on these 3 dev seeds would need the full held-out seed set before any generalization claim.

## Provenance

All three development seeds, the calibration characterization, and the stepping-history confound control ran locally on
the NumPy backend, deterministic across re-runs. Runner `research/runners/_laneC_source_monitor_coresidency_gate_v9.py`;
aggregator `research/runners/aggregate_source_monitor_v9_seeds.py`; inhibitory STDP scoped via the shipped
per-synapse plastic-mask + plasticity-gain gate with NO `sim/` edit. Artifacts and `.prov.json` sidecars are listed in
the front matter.
