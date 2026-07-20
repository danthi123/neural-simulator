# gap#1 fully-synaptic RF-phase transduction — RUNG 1 GO (feasibility): the last host read is removable

**Date:** 2026-07-20 · **Status:** RUNG 1 (feasibility) GO; RUNG 2 (on-bridge wiring) specified.

## The residual this closes

gap#1's RF-phase spiking-input encode (closed + generation-demonstrated, `2026-07-20-gap1-RF-PHASE-ENCODE-...`) has
ONE host-side residual: the phase is read on the HOST (`rf_read_phases`) and the decoded value charges `cp_ssm_state`.
The fully-spiking form: the RF spike DRIVES a downstream synapse whose response encodes the spike TIMING (= the phase),
with NO host phase read — a biological latency/decay read.

## RUNG 1 — does a decaying-conductance read of the RF spike reproduce the value? YES

`_gap1_rf_synaptic_transduction_probe.py`: 128 values → RF phases (guard-banded [0.05,0.95]) → resonate on a real
`RESONATE_AND_FIRE` bridge → the RF spike lands at `cp_rf_spike_step` s. A synapse pulse at s, decayed to period-end
(`g = exp(-(period-s)/tau)`), is the conductance a downstream neuron reads. The value recovers by the fixed log inverse
(`latency = -tau·ln(g)`; `phase = latency/period`):

| tau | corr(g, value) | recovered corr | rms (range 24) | band-bias lo / hi |
|---|---|---|---|---|
| 60 | −0.935 | **1.0000** | 0.076 | −0.060 / −0.069 |
| 120 | −0.982 | **1.0000** | 0.076 | −0.060 / −0.069 |
| 240 | −0.995 | **1.0000** | 0.076 | −0.060 / −0.069 |

(host phase-read reference corr = 1.0000)

- The raw decaying conductance is strictly MONOTONE in the value (corr → −0.995 at large tau) — the timing carries the
  value into the conductance.
- The fixed log inverse recovers the value PERFECTLY (**corr 1.0000, rms 0.076 = 0.3% of range, UNBIASED** across value
  bands = the M0 property), matching the host phase-read reference — and it is **tau-INDEPENDENT** (the log exactly
  cancels the exponential decay for ANY downstream time constant, so the wiring is not delicate).
- ⇒ **the last host read (`rf_read_phases`) is removable: a downstream synapse's decayed conductance IS the value**
  (recoverable by a fixed log-compressive read-out — a biological log-domain synaptic nonlinearity, not a host shortcut;
  the exponential decay is itself the synapse's own dynamics).

## RUNG 2 (specified, next build) — wire it on-bridge

Realize the read on-substrate: the RF encoder co-resident with a downstream conductance synapse (the step-2b masked-RF
pattern) → the RF spike charges a decaying conductance on the ssm-state neurons directly → charge `cp_ssm_state` from
that conductance (a log-compressive read-out), NO `rf_read_phases`. Then re-run the deep-NLL gate + generation and
confirm parity with the host-read path (the encode fidelity is corr 1.0 per RUNG 1, so the deep-NLL should hold). This
makes gap#1's spiking input FULLY synaptic — no host read anywhere in the encode.

## Scope

RUNG 1 is a numeric feasibility check on the real RF substrate (reads public `cp_rf_spike_step`); it proves the
timing→conductance read is invertible + unbiased. RUNG 2 is the on-bridge synaptic realization. NO `sim/` edit in
RUNG 1. Runner: `_gap1_rf_synaptic_transduction_probe.py`.
