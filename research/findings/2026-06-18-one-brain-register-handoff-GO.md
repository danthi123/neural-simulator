# Roadmap phase 2, step 1 — register→register synaptic phase handoff WORKS (one brain, no host round-trip): GO

**Date:** 2026-06-18 (the post-biologization HEADLINE arc: build the REAL "one brain" — the whole conversational
pipeline as one persistent interacting spiking loop, `project_one_brain_integrated_pipeline_and_cleanup`)
**Status:** **GO** (3 seeds × 2 D, 6/6). The genuinely-new primitive the integrated pipeline needs — chaining two
RF ops (`unbind(bind(role, filler), role)`) **register→register on ONE persistent bridge with NO `rf_read_phases`
between them** — recovers the filler perfectly and **== the current host two-call pipeline**, with both anti-cheats
collapsing. ⇒ conversational ops CAN hand off as spikes through synapses with **zero host round-trips**; the real
one-brain pipeline is viable.
**Runner:** `research/runners/_phaseB_onebrain_register_handoff_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_register_handoff.json`

## The problem this solves

Today every conversational op reads its phases OUT to numpy and re-kicks the next op — a flat who/what turn makes
**3 host round-trips**, an embedded clause 6 (`2026-06-18-one-brain-integrated-pipeline-scoping.md`). The megakernel
fuses WITHIN an op; the BETWEEN-op handoffs are still host-mediated. Removing them needs a way to pass one op's
phase output to the next op's input **synaptically**, on the substrate.

## Mechanism

One persistent bridge, 3 registers — filler `[0:D]`, bound `[D:2D]`, unbound `[2D:3D]` — with TWO diagonal complex
synapses installed together: **bind** (filler `k` → bound `D+k`, weight `role[k]`) + **unbind** (bound `D+k` →
unbound `2D+k`, weight `conj(role[k])`). Kick the filler, resonate, read the unbound register. Since
`bound[k] = role[k]·filler[k]` and `unbound[k] = conj(role[k])·bound[k] = |role[k]|²·filler[k] = filler[k]`
(unit-magnitude role), the unbound register recovers the filler — through the substrate, no host hand-off.

## Result — 3 seeds × {D=64, D=128}

| metric | mean | reading |
|---|---|---|
| on-bridge self-recovery (cleans up to the original filler) | **1.000** (6/6 ≥ 0.99) | the handoff recovers the filler |
| == host two-call pipeline (`_bind`→read→`_unbind`) | **1.000** | identical to the current path |
| **permuted-role** anti-cheat (unbind with the *wrong* role) | 0.051 | collapses (role-specific) |
| **lesion** anti-cheat (sever the bind→bound synapse) | 0.077 | collapses (the on-bridge handoff is load-bearing, not kick leakage) |

Both a SINGLE-window (both synapses, one resonate) and a TWO-window (bind settles, then unbind reads it — still no
read-out between) variant reach 1.000; the two-window is the cleaner default. (Anti-cheat note: role *reversal* was
too weak a scramble — it can coincidentally preserve recovery — so the control is unbind with a genuinely different
role; one seed/D still shows a residual 0.31 but the mean control is 0.051, far below the 1.000 signal.)

## Why it matters

This is **step 1 of the real "one brain"**: it proves the substrate can carry an op→op handoff as a phasor in a
register, read by the next op's complex synapse, with no host round-trip. The remaining phase-2 build extends this
to the full who/what turn on one persistent bridge — bind multiple roles → bundle → keep the composite as a stored
register → unbind a cued role → cleanup WTA → familiarity-gate moat → order generator — host doing only text I/O.

## Honest scope + next

- Validated for the single bind→unbind chain. The next de-risk is a **multi-role fact**: bind agent + action,
  bundle into a composite register, then unbind a cued role to query — all register→register, the
  store-and-query-on-one-bridge step. The top risk (scoping doc) is phase coherence across a *longer* chain
  (bundle + a second unbind); the two-window settle pattern is the mitigation, and a phase-latch is the fallback.
- Latency context: 116s for the whole 6-run sweep uncontended (the per-op resonate dominates; the megakernel +
  removing the read-outs are the speed levers as the chain grows).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_register_handoff_derisk --seeds 42,43,44 --dims 64,128
```
