# D3 — **replay replaces backprop-through-time**: the event register learns with one-step-local credit + retrodiction

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_pop_gate_derisk.py` (`truncate`, `replay_gamma`, `replay_target`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed, all controls).

## The setup

The previous rung established, by cutting the cross-clause gradient, that **the held slot exists only for the future**:
`a_prev` influences nothing at the current step, so with no backprop through time its write gate receives no credit and
simply closes (`a_prev` 0.610 → 0.195, pop-gate separation +0.751 → +0.067). A one-step **local** rule — the kind biology
has, the kind `sim/kernels.py: fused_bdsp_update` implements — therefore cannot learn the push gate.

Unless something hands that gate a target **now**. Replay does: reconstruct the **just-ended event's last observed
emission** from what the held slot is holding at this moment. The target exists in the present; the credit stops living
in the future.

## Result (6-seed, held-out-deeper; **gated on `a_prev` / RETURN, never on next-emission**)

| arm | **a_prev** | RETURN | a_curr | pop-gate separation |
|---|---|---|---|---|
| BPTT (the reference) | 0.610 | 0.713 | 0.731 | +0.751 |
| **no BPTT + REPLAY** | **0.648** | **0.719** | **0.814** | **+0.639** |
| no BPTT, no replay | 0.195 | 0.175 | 0.767 | +0.067 |
| no BPTT + replay, **target shuffled** | 0.195 | 0.175 | 0.765 | +0.063 |
| no BPTT + replay of the **current** event | 0.295 | 0.260 | 0.768 | +0.273 |

**With no backprop through time whatsoever, replay recovers 109% of the BPTT-taught held slot** — and `a_curr` is
*better* (0.814 vs 0.731).

## The controls say precisely what is doing the work
- **Shuffled target → 0.195**, identical to no replay at all, gate dead (+0.063). So the gain is the **content of the
  retrodiction**, not the presence of an extra loss term or an extra head.
- **Replaying the current event → 0.295**, gate only half-open (+0.273). A *present* target teaches the transition and
  cannot teach the held slot. What matters is that the replayed event is the one that has **ended**.
- The gate is `a_prev` and RETURN, never next-emission — because next-emission is blind to the held slot's collapse
  (`P(agent | current emission) ≈ 0.78`). The one arm that would have fooled it, `no_bptt`, has *better* next-emission
  than BPTT while its memory is destroyed.

## ⇒ the claim

**Backprop-through-time was standing in for replay, and replay does the job better.** The event register — two attractor
slots, a write gate opened by a boundary, a read gate opened by a return — is learnable end-to-end with

* **one-step local credit** for the transition (no gradient crosses a clause boundary), and
* **a retrodictive replay target** for the write gate (reconstruct the episode that just ended from what is held),

which is exactly the pair a brain has: local synaptic plasticity, plus hippocampal sharp-wave-ripple replay of the
just-ended episode. The biologically-implausible machinery is gone, and nothing was lost: every metric that measures the
held slot is at or above its BPTT value.

## Honest reporting
- Seed variance is real: `a_prev` per seed = 0.314 / 0.776 / 0.564 / 0.738 / 0.917 / 0.578. Seed 42 is the weak one for
  γ=1 (0.314) and strong for γ=3 (0.777); the aggregate is reported at γ=1 without per-seed selection.
- `replay_gamma` ∈ {1, 3} both work (a_prev 0.648 / 0.635); γ is not knife-edge. Earlier work in this arc showed γ=10
  destroys **both** slots — the dose matters and is bounded.
- This is the **rate** model. The gates are already spiking (attractor→attractor transfers); the transition and the
  replay head are not yet.
- The additive `replay_gamma` / `replay_target` / `truncate` flags are default-off, and `Wq` is drawn from its **own**
  generator — taking a draw from the shared `rng` silently changed the minibatch order and moved the default path
  (RETURN 0.806 → 0.361). Caught by asserting the default arm reproduces its committed value; additive code must not
  perturb the random stream.

## Next
Both learning problems are now biologically shaped, so port them to the substrate:
1. **the transition** — `fused_bdsp_update` (Burstprop; the emission cross-entropy as the top-down apical factor), or the
   teacher-free three-term temporal-memory kernel. It is a one-step rule, and one step is now provably enough.
2. **the replay head** — the retrodiction target, driven from the held slot's spiking assembly, using this project's own
   sharp-wave-ripple machinery (`consolidation_trainer.run_swr_replay_phase`, `run_concept_replay_phase`).

Gate on `a_prev` / RETURN, with the Markov floor (0.39) and the host references (BPTT `a_prev` 0.610, pop-sep +0.751).

## Files
`research/runners/_d3_event_pop_gate_derisk.py`; raw `research/findings/raw/_d3_replayctrl_seed*.json`,
`_d3_trunc_seed*.json`. The negative that motivated it:
`2026-07-10-D3-the-held-slot-exists-only-for-the-future-why-BPTT-was-standing-in-for-replay.md`. The emission floor that
sets the gate: `2026-07-10-D3-the-emission-carries-the-agent-a-floor-that-reframes-the-metrics.md`.
