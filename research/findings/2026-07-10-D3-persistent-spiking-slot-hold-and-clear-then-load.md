# D3 → a spiking slot that actually HOLDS: persistent NMDA attractor, and why a gate must CLEAR before it LOADS

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_persistent_slot_derisk.py` (numpy backend, 144-neuron bridge; NO `sim/` edit).
**Verdict:** HOLD = **GO 6/6**. Gate-open overwrite (clear-then-load) = **5/6**, with both clear-controls behaving as the mechanism predicts.

## Why this rung exists — found by reading my own substrate, not by theorizing
Every "spiking slot" in the D3/event arc used `_d3_spiking_attractor_derisk.fswta_drive`. Reading it:
- it **resets** `cp_membrane_potential_v`, `cp_recovery_variable_u`, `cp_firing_states` on **every call**, and
- its pools are built with `internal_density=0.0` — **no recurrent excitation at all**.

So it is a **stateless one-of-K re-discretizer.** It cannot hold anything: in every prior rung the slot's *hold* lived in a Python variable between calls. That was acceptable while the claim was "the winner is chosen by spikes." It is **not** acceptable for the boundary-gated copy, where **the HOLD is the mechanism** (closed gate ⇒ the prior event persists). A stateless WTA would have made "on spikes" a fiction exactly where it mattered.

## The mechanism (Amit & Brunel 1997; Wang 2002)
Each pool recurrently excites **itself** and drives a shared FS pool that inhibits all pools. A brief input selects a winner; when input is removed, the winner's recurrent excitation sustains its own firing while FS keeps the losers silent. That is a slot that holds with **no input** — and it is why biology uses attractors for working memory.

## Three things I got wrong, each caught by instrumenting
1. **AMPA recurrence cannot hold.** Wiring recurrence via `internal_density` gives AMPA (≈5 ms decay). Even at `recur=14` the pool fell from **0.168 (driven) → 0.008 (hold)**. The substrate's own Wang mechanism is `cfg.enable_nmda_recurrent` + a pathway with **`exc_receptor="nmda_slow"`** (τ_decay = 100 ms, AMPA suppressed).
2. **`receptor=` is the *inhibitory* field.** I set `receptor="nmda_slow"` and it silently did nothing — `cp_nmda_recurrent_synapse_mask` stayed `None`, i.e. **zero synapses routed**. The excitatory field is `exc_receptor=` (`sim/regions.py:341`). With it fixed: 2142 synapses routed.
3. **Even with NMDA routed, the bump died — because of the Mg²⁺ block.** Instrumented: the conductance *does* persist (`g_nmda_rec ≈ 12`, still 10.4 after 30 hold steps), but `v` collapses to **−66 mV** and the Mg block shuts the current off. This is Wang's bootstrap: NMDA needs depolarization to conduct, and a strong 1200 pA kick builds Izhikevich adaptation (`u`) whose after-hyperpolarization kills the bump on input removal. Fix = enough recurrent conductance to conduct through the residual block — monotone in `recur`: **3 → hold 0.006; 10 → 0.018; 25 → 0.103** (= the driven rate, one pool, all others silent).

## Result (6-seed)
| | mean |
|---|---|
| **HOLD correct (external input identically ZERO, asserted)** | **1.000** (6/6) |
| hold selectivity | 1.000 |
| pools active during hold | 1.0 |
| **NO-RECURRENCE control (the stateless bridge) hold** | **0.0065** — cannot hold |
| gate OPEN = CLEAR(250 ms) + LOAD | 0.833 (5/6) |
| **CONTROL: no-clear switch** | **0.000** (0/6) |
| **CONTROL: short-clear (60 ms) switch** | 0.167 (1/6) |

## The finding: a gate must CLEAR before it LOADS — and the reset must outlast τ_NMDA
A persistent attractor **resists being overwritten by input alone**: driving a different pool leaves the incumbent winning (**0/6**). Briefly silencing it is not enough either — its recurrent NMDA conductance decays with τ=100 ms, so a short reset leaves enough charge to **re-ignite the old bump** the moment inhibition lifts:

| clear duration | new-pool rate | old-pool rate | switched |
|---|---|---|---|
| 120 ms (1.2τ) | 0.091 | **0.110** | no |
| **250 ms (2.5τ)** | 0.089 | **0.000** | **yes** |
| 400 ms (4τ) | 0.089 | 0.000 | yes |

A further measured subtlety: the clear only works if the inhibition is actually **strong enough to silence** the incumbent. At `fs_to_exc=3` the pool kept firing *through* the clear (rate 0.089) and kept recharging its own NMDA (g stayed 166); at `fs_to_exc=10–30` it fell silent (0.005–0.008) and g drained to ~30. (At 80 it fires again — a post-inhibitory rebound, flagged, not chased.)

⇒ **This is why the PBWM update gate clears and then loads**, and it predicts a real **several-hundred-millisecond cost to event-boundary updating** — a testable consequence, since event-segmentation experiments report exactly such a disruption at boundaries.

## Honest reporting
- **Seed 42 fails the switch gate** on strictness, not direction: the new content wins (0.0892) but the old bump is not fully extinguished (0.0267 > my 0.2× threshold). A longer clear removes it.
- **Seed 43's short-clear control succeeded** (1/6), i.e. on that seed 60 ms sufficed. The τ-requirement is a strong tendency (5/6 fail short-clear, 6/6 fail no-clear), not an absolute per-seed law.
- `recur=25` sits in a working window; too little → the bump dies, and the `fs_to_exc` reversal at 80 shows the upper edge is not explored.

## ⇒ what this unblocks
The **HOLD** in the boundary-gated copy can now be a genuine spiking memory rather than a Python variable — and the **gate OPEN** is not a scalar multiply but a clear-then-load, matching `sim/`'s own `transmission_gate` semantics ("held normally CLOSED, opened on command"). The spiking port of the gated copy is now built on a slot that actually remembers.

## Files
`research/runners/_d3_persistent_slot_derisk.py`; the gated copy `2026-07-10-D3-boundary-gated-copy-the-held-event-is-gated-not-learned.md`.
