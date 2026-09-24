---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: C · Self/Workspace
mechanism: GNW continuous-swap branch-point snapshot/fork verification instrument
  (research/runners/_gnw_continuous_branchpoint_verify.py); instrument only, no mechanism claim
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-gnw-continuous-swap-branchpoint-snapshot-fork-instrument-PREREGISTERED.md
artifact: research/findings/raw/_gnw_continuous_branchpoint_6seed.json
verdict: GO 6/6 for the INSTRUMENT. On every seed the snapshot captures every live cp_* array, both forks restore
  exactly, the control fork is deterministic, the negative control diverges, the stored snapshot is not aliased, and
  the carry-over check passes. This verifies the FORK only; the parked recency-trace mechanism stays NO-GO and is not
  re-opened here.
---

# GNW continuous-swap branch-point fork instrument: GO 6/6

## Result

Artifact: `research/findings/raw/_gnw_continuous_branchpoint_6seed.json` (runner revision `bdbce4ce3`, `git_dirty: false`,
numpy backend, run locally). The runner's aggregate reads `verdict: GO`, `pooled_go: true`, with seed_go 6/6 and each
check 6/6: snapshot completeness, exact restore at both forks, control-fork determinism, negative-control divergence,
no aliasing of the stored snapshot, and carry-over.

## What it is and is not

It is an instrument: it shows that a branch point of the continuous thought-swap can be snapshotted and forked
exactly, so two continuations can be compared from one identical state. It makes no claim about the recency-trace
mechanism that motivated it; that mechanism stays banked NO-GO on its parked branch.

## Residual (from the adversarial re-review, non-blocking)

The fork check (G3) detects a 1e-6 change to a short-term-depression state variable, a reset of one loop's STD state,
and a 0.01 mV or 5 mV offset on the membrane potential of all 960 neurons. It does NOT detect a membrane offset on a
single neuron of 1e-3 to 5 mV, because the network's dynamics absorb it. A fork error confined to one neuron's
voltage could therefore pass. The runner now checks the resolved backend, so an unset `SIM_BACKEND` on a GPU host
exits as mis-configured instead of reading as a fork failure.

## Honesty

Functional read-outs only; "thought" names the workspace's measured spiking state.
