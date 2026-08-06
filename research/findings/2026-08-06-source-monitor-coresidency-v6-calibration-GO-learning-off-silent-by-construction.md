---
type: finding
status: positive
date: 2026-08-06
lane: laneC
mechanism: source-monitor-coresidency-v6
runner: research/runners/_laneC_source_monitor_coresidency_gate_v6.py
artifacts:
  - research/findings/raw/source_monitor_coresidency_v6/calibration_650.json
  - research/findings/raw/source_monitor_coresidency_v6/calibration_650.json.prov.json
  - research/findings/raw/source_monitor_coresidency_v6/calibration_651.json
  - research/findings/raw/source_monitor_coresidency_v6/calibration_651.json.prov.json
---

# v6 calibration GO: the learning-off leak is closed, silent by construction; the bounded-loss win holds

<!--derived-->
**Verdict: GO at v6 calibration.** Both calibration seeds are `CALIBRATION_PASS` (all 20 preregistered components
true). The v5 blocker, an isolated `learning_off_has_no_source_recall` leak of four `seen` spikes on seed 650, is
closed to zero on both seeds, and the bounded-loss guard-the-floor max-min tradeoff (the P3 deliverable) still passes
on both. Development seeds 652, 653, 654 and held-out seeds 655, 656, 657 remain locked.

## Root cause of the v5 leak: residual encoding-phase state, not a synaptic path

<!--derived-->
The v5 successor note guessed the leak came from a standing drive into source-memory that bypassed the learned
synapses. Instrumentation of the seed-650 learning-off net refutes that. The four `seen` spikes fire at recall step 34
with **zero external current and zero source-afferent spikes**, and they reproduce with the episode drive removed
entirely: the leak is pure residual state, not any input path. After the strong source-afferent drive during the
experience phase (weight 80, 5000 pA), the `seen` source-memory neurons are left depolarised (V ~= -40 mV, near
threshold) with a large Izhikevich adaptation variable u (~288 vs ~0 fresh); experience's trailing 80-step rest does
not fully drain it, so a handful of neurons drift across threshold during the immediately following read. One extra
rest window fully drains the state and the leak vanishes.

## The v6 fix: settle to quiescence before the read

v6 changes only the source-recall protocol; the v2 fast-spiking GABA-A biased-competition circuit and the bounded-loss
acceptance rule are unchanged. Before every read window, `recall` settles the substrate at zero input (competition
gated off, as in `_rest`) in `rest_steps` blocks until a full block passes with zero spikes across the source-memory,
aPFC, ACC, and source-afferent populations, capped at 12 blocks. A verified-quiescent start makes the read silent by
construction: with zero learned weights and no afferent drive nothing can move the source populations, on any seed,
not merely on these two. `recall_settle_reaches_quiescence` is a scored component; settle terminated in 160 steps
(2 blocks) on both seeds, well inside the cap.

## Result

Metrics from the two cited artifacts (`research/findings/raw/source_monitor_coresidency_v6/calibration_650.json`,
`research/findings/raw/source_monitor_coresidency_v6/calibration_651.json`). `M` intact margin, `L` matched competition-lesion margin,
`loss = max(0, L-M)`, `surplus = max(0, L-F)`, floor `F = 0.15`.

| seed | M seen/heard/self | L seen/heard/self | min M | min L | learning-off spikes | bounded loss | status |
|---:|---|---|---:|---:|---:|---|:---:|
| 650 | .1767/.2725/.2500 | .1667/.2725/.2500 | .1767 | .1667 | 0 | 0/0/0 | CALIBRATION_PASS | <!--derived-->
| 651 | .1892/.2892/.1667 | .1375/.2850/.1667 | .1667 | .1375 | 0 | 0/0/0 | CALIBRATION_PASS | <!--derived-->

<!--derived-->
On both seeds the learning-off leak is zero (seed 650: 4 -> 0), every source margin holds above the 0.15 floor, no
source loses margin, and the minimum margin strictly improves under competition (650: .1667 -> .1767; 651:
.1375 -> .1667). Seed 651 still shows the redistributive win the whole-brain role asks for: competition lifts the
weakest `seen` source from .1375 to .1892 at zero cost to any other source. The settle is applied to both the intact
and the competition-lesion arms, so the margin gain is attributable to competition alone, not to the protocol change.

## Decision and next mechanism

<!--derived-->
Both calibration seeds pass, so v6 calibration is a GO. The next step is generalization: open development seeds 652,
653, 654 (then held-out 655, 656, 657) against the frozen v6 mechanism and the frozen bounded-loss criteria, to test
whether silent-by-construction recall and the bounded-loss win hold on unseen seeds. The circuit and acceptance rule
stay frozen; only the seed partition advances.

## Scaffolds that remain

Sparse episode activity, physical source-afferent identity, the externally timed learning window, and the pre-read
settle timing remain developmental scaffolds. Population spike counts and winners are host-read for evaluation only.
The competition wiring is specified, not self-organized. No language, confidence scalar, or response policy is
claimed. v6 establishes only that a co-resident spiking source-monitor delivers a reliable, comparable seen/heard/self
confidence into the honesty path under the required tradeoff, and is silent when its learned pathway is empty.

## Verification

<!--derived-->
Verify-go lenses run before landing: control-integrity (the settle is applied to BOTH the competition-on and
competition-off arms, so the margin gain is competition alone; the redistributive win keeps the same sign as v5),
gate-cheat (both the leak control and `recall_settle_reaches_quiescence` are default-scored and invoked; settle
terminated in 160 of a 960-step cap), silent-by-construction (the settle loops to a verified zero-spike window, and
the leak reproduces with the episode drive removed, so it is residual state not an input path), conservatism (settle
raised seed 651 min lesion margin .069 -> .138, making "weakest strictly improved" harder, still passing), and
determinism/seeding (identical minimum margins across re-runs; `cfg.seed` set via the base class).

## Provenance

Both seeds ran locally on the NumPy backend, deterministic across re-runs (identical minimum margins). Runner
`research/runners/_laneC_source_monitor_coresidency_gate_v6.py`; artifacts and `.prov.json` sidecars listed in the
front matter, stamped from git `071b68bca84f7a4d4be04924b9090cabe163e330`.
