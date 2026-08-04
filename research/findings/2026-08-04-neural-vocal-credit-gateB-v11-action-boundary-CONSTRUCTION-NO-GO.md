---
type: finding
status: complete
date: 2026-08-04
verdict: CONSTRUCTION_NO_GO
mechanism: neural-vocal-action-boundary-v11
runner: research/runners/_vocal_action_credit_gate_v11_boundary.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_numpy_v2.json
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_numpy_v2.json.prov.json
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_cupy_v2.json
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_cupy_v2.json.prov.json
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_cross_backend_v2.json
  - research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_cross_backend_v2.json.prov.json
---

# V11 action boundary self-activates without a motor action

<!--derived-->
**Verdict: CONSTRUCTION_NO_GO.** The fixed V11 topology failed its locked
construction seed on both NumPy and CuPy. Its corollary-discharge and local
inhibitory populations fired during the no-action catch even though both motor
populations emitted zero spikes. No recurrence point in the preregistered
`[0.25, 0.5, 1.0, 2.0]` ladder qualified on either backend. Formal seed `1`
remains sealed, and no reward, dopamine, eligibility, or policy learning ran.

Seed-waiver: the preregistration assigns seed `991` to construction and requires
retirement on any failed construction criterion; repeating a structurally
invalid candidate across capability seeds would violate that stop rule.

Artifact:
`research/findings/raw/vocal_action_credit_gate_v11/construction_seed991_cross_backend_v2.json`.
The two backend artifacts in the same directory carry the complete step
telemetry and provenance.

## What was tested

V11 added one excitatory action-corollary population downstream of both motor
outputs. That population projected symmetrically through separate local
fast-spiking populations into each proposal, commitment, and motor territory.
Only the corollary population had NMDA enabled. There was no shared long-range
inhibitory population and no host-selected winner route.

Each backend ran the same uninterrupted construction protocol for all four
fixed recurrent weights:

1. `80 ms` unforced warmup.
2. `600 ms` weak-drive catch at `250 pA`.
3. `3000 ms` autonomous recovery.
4. A `600 ms` action window and `3000 ms` recovery.
5. A second `600 ms` action window on the same brain and a final `3000 ms`
   recovery.

The static audit passed for every point: `688` neurons, `33,950` synapses,
`26` regions, symmetric local stopping paths, exact policy-route invariants,
and zero selector-reset current. Initial and final hashes covered every weight
array and were byte-identical in all eight backend-by-weight cells.

## Decisive failure

The catch produced no motor spikes at any point: motor counts were `[0, 0]` in
all eight cells. The boundary nevertheless activated before any possible motor
copy:

- NumPy produced `125-136` corollary spikes and `90-125` spikes in each local
  stopping population during the catch.
- CuPy produced `114-127` corollary spikes and `89-119` spikes in each local
  stopping population during the catch.
- The preceding warmup already contained `626-628` boundary spikes on NumPy
  and `596-604` on CuPy.

This violates the candidate's central causal requirement. A circuit intended
to mark a completed action cannot serve as an action boundary when it enters
the boundary state without an action.

The activity was not permanently unstable. After the fixed `3000 ms` recovery,
all points had zero boundary spikes and residual corollary NMDA conductance was
at most `1.69e-14`. That establishes autonomous recovery, but boundedness does
not repair the missing action contingency.

Action-window behavior also did not identify a shared operating point. Only
NumPy weight `0.25` and CuPy weight `1.0` passed both fixed action windows. All
points still failed the catch, so these observations are diagnostic only and
cannot qualify a weight after the fact.

## Decision

Retire this exact V11 topology. Do not tune the recurrence ladder, extend the
warmup, reinterpret startup activity as an action, consume formal seed `1`, or
open reward and policy learning. The pre-telemetry construction artifacts are
retained as an audit trail; the `_v2` rerun repeats the fixed dynamics while
adding complete initial/final weight hashes.

The next research question is narrower than another weight sweep: how can an
action-corollary population remain quiescent until motor-copy input arrives,
then recruit a symmetric, temporary stopping state and recover autonomously?
Any successor needs a new biology-supported gate. Candidate classes include
locally stabilized excitation/inhibition or motor-triggered disinhibition, but
neither is authorized by this result alone. A Python threshold, forced reset,
or host-timed inhibition would reproduce the shortcut this gate is intended to
remove.
