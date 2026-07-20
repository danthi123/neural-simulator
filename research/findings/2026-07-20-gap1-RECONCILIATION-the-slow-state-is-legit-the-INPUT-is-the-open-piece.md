# gap#1 RECONCILIATION — the slow SSM STATE is a legitimate on-substrate mechanism; the open piece is driving it from SPIKES

Produced by the a-1 RAG check before launching a gap#1 research gate (drift-mode-12 discipline: reconcile today's
summary against the findings). It reconciles a real tension and precisely scopes the next mechanism, so no gate
re-derives the 2026-07-13 reservoir arc.

## The tension

- **Today's M1** (`2026-07-20-gap1-M1-...`): `cp_ssm_state` (a per-neuron slow leaky integrator) holds the WKV
  recurrent state and BEATS the fair interpolated trigram at deep context (+0.126, 6/6). Its finding located the
  residual as "input-delivery fidelity, not the state and not the read-out."
- **Today's 28-agent audit** WITHDREW the "on-bridge" claim: in the M1 runner, `cp_ssm_inject` is host-written and
  NO neurons spike, so total spiking participation is zero.

Read alone these look contradictory ("the state is fine" vs "the state has no spiking").

## The reconciliation — from the 2026-07-13 reservoir arc

`2026-07-13-PAST-RESERVOIR-RUNG4b-i-...` established, on a real bridge, that **the raw Izhikevich membrane leaks too
fast to hold an SSM state** (the input-modulated shunt is swamped by the intrinsic leak; HOLD-RELEASE ~ +0.1 mV).
Its named fix, "the precise next mechanism," was a **SLOW per-neuron leaky integrator with an input-modulated leak**,
via either:
1. a minimal additive `sim/` slow-integrator variable, OR
2. the slow NMDA conductance (tau ~150 ms) with an input-modulated decay.

**`cp_ssm_state` (`enable_selective_ssm_state`) IS route 1** — `s = lam*s + (1-lam)*inject`,
`lam = clip(1 - k_leak*(1+shunt), 0, 1)`. It was BUILT as the answer to that 2026-07-13 boundary. So:

- **The STATE is a legitimate, biology-grounded on-substrate mechanism** — the slow-integrator the reservoir arc
  proved was needed. The audit's "zero spiking" is correct but narrower than it reads: it is about the INPUT path,
  not the state's legitimacy. A slow graded conductance-like integrator IS how biology holds integrator state
  (NMDA/calcium); the SpikeGPT/SSM literature reframe (2026-07-19 gate) established that NO spiking LM holds the
  recurrent state as spikes — all keep it graded and spike only I/O.
- **The genuinely open piece is DRIVING the state from the network's own SPIKES** rather than a host-written scalar.
  Today's M2 tried this via a NEF population that DECODES spikes back to a scalar `v_t` then re-injects — and that
  lossy decode capped at **-0.030** (below baseline), all M2 levers exhausted.

## ⇒ The precisely-scoped next mechanism (2026-07-13 route 2, never built)

**Drive the slow SSM integrator DIRECTLY from a synaptic conductance the network's spikes produce — skip the
scalar-decode bottleneck entirely.** M2 failed because it forced spikes -> scalar -> re-inject (a rate-code
round-trip, the exact lossy step). The alternative: spikes -> `g_syn` (a real synaptic conductance) -> the
integrator reads `g_syn` as its input drive. This is the SAME cross-gap insight established TODAY in gap#4:
**a graded conductance read succeeds (0.92) where a spike-rate read fails (0.000000)** — so read the input as a
CONDUCTANCE, not as a decoded rate. Route 2 (slow NMDA conductance with input-modulated decay) is the biology-
grounded form.

**Why this is well-posed, not a wall:** the state mechanism is validated (cp_ssm_state beats the trigram); the
read-out is validated (2026-07-13 spiking-readout GO); the ONLY unvalidated stage is input, and the specific fix
(conductance-drive, not rate-decode) is named, biology-grounded, and consistent with the day's cross-gap finding.

## Status + next

- **gap#1 honest state:** graded slow-state recurrence beats the trigram at deep context; the state and read-out are
  substrate-legitimate; the open piece is spiking INPUT, and the M2 rate-decode path is a confirmed boundary
  (-0.030). The named next mechanism is conductance-drive (2026-07-13 route 2).
- **This is a NEW mechanism class + past a confirmed boundary ⇒ the research gate fires** before building it. The
  RAG check has already done the gate's reconciliation half; the build gate should rank conductance-drive
  realizations (direct `g_syn` read vs slow-NMDA-with-modulated-decay) cheap-first.
- No re-derivation: this reconciliation PREVENTED launching a gate that would have re-run the 2026-07-13 arc.

---

## SUBSTRATE VERIFICATION (independent of the running gate) — route 2 is CHEAP and ADDITIVE

Verified at source before the gate reports, so its recommendation can be checked against the actual machinery:

- **The audit's point confirmed at source:** `cp_ssm_inject` is host-written in the runner
  (`_emerge_wkv_onbridge_derisk.py:409`: `b.cp_ssm_inject[:] = _cur`). That is the entire "zero spiking input".
- **The conductances the spikes produce ALREADY exist and are ALREADY current at the SSM update point:**
  `cp_conductance_g_e` and `cp_conductance_g_nmda` are bridge arrays (bridge.py:246/1246, saved-state list 2406),
  and the synaptic-conductance update is pipeline step 2 while the SSM-state update
  (`bridge.py:5951-5953: cp_ssm_state = lam*cp_ssm_state + (1-lam)*cp_ssm_inject`) is later in the same step.
- ⇒ **Route 2 needs NO scalar decode and NO structural sim/ edit:** the slow NMDA conductance the network's spikes
  drive through synapses is available to feed `cp_ssm_state`'s input in place of the host-written scalar. The
  cheapest de-risk is: build a small recurrent language net whose token embedding drives real synapses -> the
  spikes those produce set `cp_conductance_g_nmda` -> the SSM integrator reads THAT as its input (not a decoded
  scalar) -> does it still beat the trigram at deep context? This is the M2 test with the lossy decode REMOVED.
- **The pre-flight (today's hardest-won lesson):** BEFORE pre-registering, verify on DEPLOYED inputs that
  `cp_conductance_g_nmda` at the SSM update carries the per-token input with enough fidelity — measure its
  correlation with the intended `v_t` on real spikes, exactly as the M2 failure taught (validate the property on
  the inputs the implementation actually generates, not idealized ones).

This is recorded as substrate fact, NOT a decision to build ahead of the gate — the gate ranks realizations and may
prefer the modulated-decay form or find a burned-arm caveat. But it establishes the route is cheap, additive, and
reuses existing arrays, which bounds the cost of the recommendation.
