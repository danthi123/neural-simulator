---
type: finding
status: complete
date: 2026-08-04
verdict: CONSTRUCTION_NO_GO
mechanism: neural-vocal-disinhibitory-action-boundary-v12
runner: research/runners/_vocal_action_credit_gate_v12_disinhibitory.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_numpy.json
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_numpy.json.prov.json
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_cupy.json
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_cupy.json.prov.json
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_cross_backend.json
  - research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_cross_backend.json.prov.json
---

# V12 disinhibitory boundary fails quiet startup and action release

<!--derived-->
**Verdict: CONSTRUCTION_NO_GO.** The fixed V12 topology failed its locked
construction seed on both NumPy and CuPy. All four inhibitory source-on/source-
off twins passed, and the intact circuit became quiet during the no-action catch
and recovery periods. It nevertheless activated the boundary and both motor
channels during the required warmup. During intended actions, guard suppression
also missed the fixed `50%` maximum on both backends; CuPy additionally allowed
the losing motor channel to cross in both action windows. Seed `2` remains
sealed. No reward, dopamine, eligibility, or policy learning ran.

Seed-waiver: seed `997` is the locked construction seed, and any failed
construction criterion retires the topology. Running capability seeds after a
structural no-go would violate the preregistered stop rule.

Instrument: causal attribution used four seed-identical source-on/source-off
inhibitory twin pairs per backend, with exact pre-causal firing-prefix and
weight identity checks, target GABA-A conductance, and post-source target-spike
counts; the intact construction separately measured startup, no-action,
action, and recovery behavior on both CPU and GPU.

Artifact:
`research/findings/raw/vocal_action_credit_gate_v12/construction_seed997_cross_backend.json`.
The backend artifacts in the same directory contain complete per-step firing,
conductance, current, weight, causal-twin, and provenance records.

## What was tested

V12 replaced V11's self-recurrent action-corollary population with a local
disinhibitory circuit. Background practice activity excited a guard population,
which inhibited action corollary. Motor activity excited a disinhibitor and the
corollary; the disinhibitor suppressed the guard. Corollary activity recruited
four local fast-spiking populations that inhibited proposal, commitment, and
motor populations symmetrically.

The new populations received no direct host current. The host did not choose an
action, inspect a threshold to alter neural state, force a reset, or clear state
between phases. The candidate used only AMPA and GABA-A on its new routes, with
no NMDA, GABA-B, recurrence, reward, or plasticity.

Each backend ran the same uninterrupted seed-`997` protocol: `1000` baseline
warmup steps, a `600`-step no-action catch, `3000` recovery steps, two
`600`-step action windows separated and followed by `3000`-step recoveries.
Four additional matched source-on/source-off twin pairs audited guard,
disinhibition, proposal-stop, and commitment/motor-stop inhibition.

The static audit passed: `728` neurons, `36,293` synapses, `28` regions, `53`
declared pathways, exact symmetry and polarity, zero boundary/reset current,
and a closed host boundary. Initial and final complete weight hashes were
identical on both backends.

## Decisive failures

The required warmup was not quiet:

| Backend | Motor spikes | Guard | VIP | Corollary | Four stop populations |
|---|---:|---:|---:|---:|---:|
| NumPy | `95 / 76` | `263` | `89` | `112` | `74 / 77 / 77 / 72` |
| CuPy | `73 / 78` | `375` | `88` | `101` | `72 / 71 / 72 / 74` |

Both motor channels passed the fixed 12-spike action threshold during startup.
The guard also had at least one silent `100`-step warmup bin on each backend.
The later no-action catch was quiet with motor counts `[0, 0]`, and every
recovery was quiet in its required final `1000` steps. The circuit therefore
settled autonomously, but the preregistration explicitly forbids deleting or
reinterpreting startup telemetry.

The intended action windows had the required causal order: motor activity
preceded disinhibitor and corollary activity, which preceded local stopping
activity. The release was too weak under the locked criterion:

| Backend | Action 1 guard ratio | Action 2 guard ratio | Clean actions |
|---|---:|---:|---:|
| NumPy | `58.25%` | `54.29%` | `2 / 2` |
| CuPy | `51.43%` | `54.03%` | `0 / 2` |

All four ratios exceed the preregistered `50%` ceiling. CuPy also admitted a
later loser crossing in both action windows, producing motor totals `72 / 215`
and `136 / 151`. This is a construction failure, not the narrower qualified-
but-too-short outcome, because baseline quiet and motor-triggered release did
not pass.

## What the causal twins establish

Every inhibitory source-on arm produced nonzero target GABA-A conductance,
matched its source-off twin before the causal event, preserved identical
weights, and did not increase target firing. Source-on versus source-off target
spikes were:

| Audit | NumPy | CuPy |
|---|---:|---:|
| Guard to corollary | `92 / 101` | `86 / 191` |
| Disinhibitor to guard | `865 / 937` | `1074 / 1162` |
| Proposal stop | `40 / 55` | `50 / 69` |
| Commitment/motor stop | `257 / 342` | `256 / 344` |

These controls establish the sign and causal engagement of the inhibitory
routes. They do not rescue the intact circuit's failed timing and behavioral
criteria.

## Decision

Retire this exact V12 topology. Do not tune its locked weights, extend or erase
the warmup, change the `50%` guard criterion, rerun seed `997`, consume seed
`2`, or open reward and policy learning.

The next evidence review must address two coupled problems before another
candidate is authorized: biologically grounded suppression of initialization
transients, and a stronger action-contingent release that does not prolong
competition between motor channels. The useful retained evidence is that the
feed-forward GABA-A paths have the intended causal sign and that the network
can return to a quiet state without a host reset. A successor needs a genuinely
different circuit or intrinsic mechanism supported by primary evidence, plus a
new preregistration and fresh construction seed.
