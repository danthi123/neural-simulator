---
type: finding
status: negative
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v5
runner: research/runners/_laneC_source_monitor_coresidency_gate_v5.py
artifacts:
  - research/findings/raw/source_monitor_coresidency_v5/calibration_650.json
  - research/findings/raw/source_monitor_coresidency_v5/calibration_650.json.prov.json
  - research/findings/raw/source_monitor_coresidency_v5/calibration_651.json
  - research/findings/raw/source_monitor_coresidency_v5/calibration_651.json.prov.json
---

# v5 calibration NO-GO: the bounded-loss tradeoff is validated; the residual is an isolated learning-off leak

<!--derived-->
**Verdict: NO-GO at v5 calibration.** Seed 651 is `CALIBRATION_PASS`; seed 650 is `CALIBRATION_FAIL`. Both calibration
seeds must pass, so the gate is a NO-GO. But the failure is NOT the source-margin tradeoff this arc has been stuck on:
seed 650 fails a single inherited anti-cheat control, `learning_off_has_no_source_recall`, by four spikes. The
bounded-loss max-min tradeoff criterion (the P3 deliverable) passes on BOTH seeds.

## What this gate tested

v5 runs the v2 local fast-spiking GABA-A biased-competition circuit UNCHANGED, and scores it against the bounded-loss,
guard-the-floor, max-min acceptance rule specified in the P3 functional-role spec
(2026-08-06-source-monitor-P3-functional-role-and-tradeoff-spec.md) and preregistered on fresh seeds
(2026-08-06-source-monitor-coresidency-v5-calibration-PREREGISTRATION.md). The question: does biased competition meet
the tradeoff the whole brain actually requires, rather than the over-strict per-source zero-degradation control that
recorded v2 NO-GO?

## Result

Margins from the two cited artifacts
(`research/findings/raw/source_monitor_coresidency_v5/calibration_650.json` and
`research/findings/raw/source_monitor_coresidency_v5/calibration_651.json`). `M` is the intact margin, `L` the matched
competition-lesion margin,
`loss = max(0, L-M)`, `surplus = max(0, L-F)`, floor `F = 0.15`.

| seed | M seen/heard/self | L seen/heard/self | min M | min L | bounded loss (all s) | tradeoff crit A-C | status |
|---:|---|---|---:|---:|---|:---:|:---:|
| 650 | .1758/.2725/.2500 | .1600/.2725/.2500 | .1758 | .1600 | 0 / 0 / 0 | PASS | FAIL | <!--derived-->
| 651 | .1817/.2867/.1667 | .0692/.2758/.1667 | .1667 | .0692 | 0 / 0 / 0 | PASS | PASS | <!--derived-->

<!--derived-->
On BOTH seeds the tradeoff criterion is satisfied outright: every source ends above the `0.15` floor, no source loses
any margin (all bounded losses are zero), and the minimum margin strictly improves under competition (650:
.1600->.1758; 651: .0692->.1667). Seed 651 is the clearest possible demonstration of why the mechanism matters and why
zero-degradation was the wrong control: without competition its seen source sits at `0.0692`, far BELOW the floor, and
competition lifts it to `0.1817` while costing no other source anything. The circuit does exactly the redistributive
job the whole-brain role asks of it.

## Why seed 650 fails, and why it is a different defect

<!--derived-->
Seed 650 fails one component only: `learning_off_has_no_source_recall`. With the source-learning gate disabled the
episode-to-source weights stay at exactly zero (`learning_off` L1 norm is `0.0` before and after experience), yet
episode-only recall still produced four `seen` source spikes (rate `0.0033`), 16 seen-aPFC spikes, and 30 ACC spikes.
The preregistered control demands exactly zero, so four spikes fail it. The related anti-cheat controls are clean: the
`unseen` episode produced zero source spikes, and `learning_off_keeps_weights_zero` passed. Seed 651 passed this same
control. So the leak is a small, seed-fragile residual in the inherited v2 substrate: on some initializations a source-
memory neuron receives just enough standing input, through a path other than the learned episode-to-source synapses, to
cross threshold a handful of times during episode-only recall.

This is not the source-margin tradeoff. It is a purity failure of the learning-off control, orthogonal to whether
competition harms strong sources. The arc's blocker has moved.

## Decision

Per the preregistration stop rule, a failure on either calibration seed is a v5 calibration NO-GO. Do not open
development seeds `652, 653, 654` or held-out seeds `655, 656, 657`. Do not relax the `learning_off_has_no_source_recall
== 0` control after seeing this result; it was inherited and preregistered, and weakening it now would be the exact
after-the-fact tuning this arc forbids.

What is banked and does not need re-litigating:

- **The no-harm tradeoff is settled** (ROADMAP blocker #3). The acceptable control is bounded-loss guard-the-floor
  max-min, not per-source zero degradation, and biased competition meets it on both fresh seeds. The two subsequent
  versions (v3 threshold homeostasis, v4 inhibitory STDP) that searched for a zero-cost mechanism were solving a
  problem the whole-brain role does not pose.
- **The residual is isolated and quantified**: a four-spike learning-off leak on 1 of 2 seeds, through a non-learned
  path into source-memory, while the learned weights are provably zero.

## Next mechanism

The successor addresses the learning-off leak directly, on a fresh preregistration and fresh seeds. The candidate cause
is standing drive into source-memory that does not pass through the gated episode-to-source synapses (for example a
residual source-afferent or episode-to-source-memory baseline that survives with learning off). The fix must make
episode-only recall produce source spikes ONLY through the learned pathway, so that with learning off and zero learned
weights the source populations are silent by construction, not merely usually silent. The biased-competition circuit and
the bounded-loss acceptance rule are retained unchanged; only the source-recall gating is tightened.

## Provenance

Both seeds ran on the mini-PC pool (pool40 seed 650, pool41 seed 651) on the NumPy backend from the clean provisioned
archive of commit `4d4268c31b9ba0e4f83f9afb3ab9b01dc0e136b2`, `excluded_worktree_paths=0`. Artifacts and their
`.prov.json` sidecars are listed in the front matter.
