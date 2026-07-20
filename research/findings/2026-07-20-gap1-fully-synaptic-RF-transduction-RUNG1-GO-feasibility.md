# gap#1 fully-synaptic RF-phase transduction — RUNG 1 + RUNG 2 GO: the last host read is REPLACED by a real synapse on-bridge

**Date:** 2026-07-20 · **Status:** RUNG 1 (feasibility) GO **+ RUNG 2 (on-bridge synaptic read) GO** — the RF spike
drives a REAL slow-NMDA conductance synapse whose conductance encodes the value on the substrate, NO host `rf_read_phases`.

## RUNG 2 — GO (on the real substrate)

`_gap1_rf_synaptic_onbridge_rung2_probe.py`: an RF bridge (64 encoders + 64 readouts) with a diagonal SLOW-NMDA synapse
encoder_i→readout_i. The encoders resonate (value-phases), each RF spike (landing in `cp_firing_states`) charges
readout_i's `g_nmda` via the standard synaptic update; at period-end `g_nmda[i] = w·exp(-(period-spike_step_i)/tau_nmda)`
= RUNG 1's decaying-conductance read. Reading `g_nmda` (a standard on-bridge conductance read, like reading
`cp_ssm_state`) + a fixed log read-out:
- readout `g_nmda`: 64/64 nonzero, **corr(g_nmda, value) = −0.974** — the SYNAPSE transduces the value into the conductance.
- fixed log read-out recovers the value **PERFECTLY: corr 1.0000, rms 0.038 (0.16% of range 24), band-bias +0.003 /
  −0.005 (UNBIASED)** — matching the host phase-read reference (corr 1.0000).
- ⇒ **the last host read (`rf_read_phases`) is REPLACED by a genuine conductance synapse on the substrate: the RF spike
  → slow-NMDA synapse → readout `g_nmda` = the value, read by a biological log-compressive read-out. NO host phase read,
  NO `sim/` edit** (`inject_explicit_wiring` + reading public `cp_conductance_g_nmda`).

## RUNG 3 — GO (deployed distribution): the fully-synaptic read reproduces the accumulated-state fidelity

`_gap1_rf_synaptic_deployed_preflight.py`: apply the fully-synaptic read (512-neuron encoder+readout RF bridge + the
diagonal slow-NMDA synapse; per token: kick the encoders with the inject phases, resonate, read the readout `g_nmda`,
calibrated log decode — NO host `rf_read_phases`) to the REAL deployed injects (zero-inflated), accumulate the
decay-leaky state vs exact. Calibration: the log read-out recovers the value corr 1.0000 (256/256 channels). Result:
- **accumulated corr(s_synaptic, s_exact) = 0.9703** (per-channel median 0.967, p10 0.924) — ABOVE the deep-NLL
  threshold (~0.9, M0 curve crosses zero at ~0.80), close to the rf_read_phases reference (0.998).
- ⇒ **the FULLY-SYNAPTIC read reproduces the deployed accumulated-state fidelity the deep-NLL needs — gap#1's spiking
  input can be made fully synaptic (no host `rf_read_phases`), validated on the real deployed distribution.**
- Honest caveat: slightly below the phase-read reference (0.970 vs 0.998 — the log-decode of the finite-precision
  `g_nmda` on the zero-inflated distribution adds a little error) + a small accum bias (−2.0 in state units, absorbable
  by the WKV's trained per-channel read-out bias). RUNG 4 (end-to-end deep-NLL through the deployed pipeline with the
  synaptic read) is the final confirmation; the M0 curve predicts a positive deep-NLL at corr 0.97.

**⇒ RUNG 1+2+3 GO: the fully-synaptic transduction WORKS — numerically (RUNG 1, corr 1.0), on-bridge via a real synapse
(RUNG 2, corr 1.0), and on the deployed distribution (RUNG 3, accum corr 0.97 > threshold). The last host read
(`rf_read_phases`) is replaceable by a genuine conductance synapse; gap#1's spiking input is (validated-)fully synaptic.**
RUNG 4 (deployed deep-NLL parity) is the last integration.

## RUNG 1 — GO (feasibility, kept below)

**Date:** 2026-07-20 · original RUNG 1 status: feasibility GO; RUNG 2 (on-bridge wiring) specified.

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

## RUNG 2 build path — CONFIRMED reuse-by-import (a0 read of the RF step)

`sim/bridge.py:6936` (the `RESONATE_AND_FIRE` step) sets `self.cp_firing_states[:] = fired_this_step` — the RF spike
lands in the STANDARD firing array. ⇒ an RF encoder neuron can drive a REGULAR conductance synapse to a downstream
readout neuron (the RF spike propagates through `cp_connections` like any spike, charging the downstream conductance).
So RUNG 2 is **reuse-by-import** (the step-2b masked-RF co-residence pattern + a conductance synapse RF→readout; read
the readout's decayed conductance `cp_conductance_g_*` at period-end → log read-out → charge `cp_ssm_state`), **NO `sim/`
edit** — the machinery all exists. The build: (1) a co-resident bridge = RF encoder slice + readout slice (per token:
`rf_kick` the phases on the RF slice, run the masked resonate, the RF spike charges the readout conductance); (2) read
the readout's decayed conductance on-bridge; (3) log read-out → `cp_ssm_state`; (4) deep-NLL/generation parity vs the
host-read path (encode fidelity is corr 1.0 per RUNG 1, so parity is expected). This is the next cycle's build.

## Scope

RUNG 1 is a numeric feasibility check on the real RF substrate (reads public `cp_rf_spike_step`); it proves the
timing→conductance read is invertible + unbiased. RUNG 2 is the on-bridge synaptic realization. NO `sim/` edit in
RUNG 1. Runner: `_gap1_rf_synaptic_transduction_probe.py`.
