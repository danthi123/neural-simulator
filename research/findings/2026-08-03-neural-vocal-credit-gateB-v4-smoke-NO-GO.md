---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-credit-v4
runner: research/runners/_vocal_action_credit_gate_v4.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v4/smoke_numpy.json
  - research/findings/raw/vocal_action_credit_gate_v4/smoke_cupy.json
  - research/findings/raw/vocal_action_credit_gate_v4/adversarial_audit.json
---

# Host-timed action trace makes Gate B v4 unsafe to promote

<!--derived-->
**Verdict: NO-GO at seed-zero smoke; retire v4 before formal execution.** The
dendritic state is real simulator state and is causal on the NumPy smoke, but
the candidate fails three independent requirements: its original selectivity
checks could accept bilateral nonselective activity, Python controls the
duration of the action tag after detecting the winning channel, and the CuPy
physiology exceeds the fixed firing ceiling. Calibration, development, and
held-out seeds remain sealed and unused.

## What survived the audit

The motor-to-value route loads the simulator's graded dendritic conductance,
not a host expected-value array. On NumPy, the intact state was selective
(`83.329/0` before outcome), delay firing was zero, and the generic outcome
read produced `9.375/0 Hz/cell`. Plateau and outcome-read lesions each removed
the response. The fixed channel permutation moved the observed state to the
other neural population, and the frozen expectation route changed no
expectation synapses while actor plasticity remained active.

These observations establish a useful component result: a local action-linked
dendritic state can persist silently and affect a later event. They do not
establish autonomous action credit or a valid Gate B circuit.

## Why v4 is retired

First, `expected_value_channel` was calculated in Python from the observed
winner and configured permutation. The old checks then compared those same
labels. Replacing both intact and permuted neural plateau/output vectors with
equal bilateral values still made every old smoke check pass. The runner now
contains explicit neural-selectivity checks and a regression test, but this
invalidates the original gate as evidence.

Second, Python accumulates channel-specific motor spikes, detects commitment,
stores the winner, advances one step, and closes the expectation route. This
host timing controls how much dendritic state is loaded. The prior statement
that v4 had no Python winner latch was therefore false. The runner and plan now
declare this scaffold explicitly.

Third, the exact CuPy smoke on the RTX 3090 retained a selective plateau
(`52.610/0`) and zero delay firing but produced `45.833/0 Hz/cell` at outcome,
above the locked `20 Hz/cell` ceiling. NumPy produced `9.375/0 Hz/cell`.
Construction hashes showed identical synapses, weights, masks, and initial
membrane state, so this was not different wiring.

<!--derived-->
Instrumentation localized the backend sensitivity to the always-open
`motor -> value-FS` route after commitment. NumPy accumulated `49/0` FS spikes
during the load window and left mean inhibitory conductance `19.57746` on the
selected value pool before outcome. CuPy accumulated `28/40` FS spikes and
left only `0.90843` inhibition on that pool. Late motor activity therefore
drove different FS channels after the host had already closed only the
expectation route. The apparent backend discrepancy exposed a circuit/timing
fault rather than a reason for backend-specific tuning.

## Decision

Do not open v4 calibration seeds `70001/70003`, development seeds
`70009/70019/70039/70051`, or held-out seeds `70061/70067`. Do not tune separate
CPU and GPU constants, retain the host winner latch, or treat a host-derived
channel label as proof of neural selectivity.

The next candidate must use a neural, action-specific event that is available
without Python observing or timing the winner. The existing selector's neural
commit populations are the first route to test. A generic outcome event should
drive matched feed-forward inhibition and excitation so normalization is tied
to the event being read rather than unrelated late motor activity. Before any
formal partition is assigned, a reserved smoke must independently infer the
tagged channel from neural state, reject bilateral activity, pass decisive
lesions, and stay nonzero and below the same firing ceiling on both NumPy and
CuPy.

Artifacts: `research/findings/raw/vocal_action_credit_gate_v4/smoke_numpy.json`,
`smoke_cupy.json`, and `adversarial_audit.json` in the same directory.
