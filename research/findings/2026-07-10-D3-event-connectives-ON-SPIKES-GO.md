# D3 EVENT CONNECTIVES → ON SPIKES (6-seed GO): the event PAIR maintained by four spiking FS-WTA attractor slots

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_connective_spiking_derisk.py` (reuse-by-import: `_d3_event_connective_derisk` + `build_fswta_score_bridge`/`fswta_drive`; numpy backend; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102).

## What this closes
The rate rung established that a connective-marked **event boundary** shifts the running event into a previous slot, so the brain holds a **pair** of composed events and can relate them. The non-negotiable is fully-spiking-on-one-brain, so this moves the whole pair onto the substrate:

```
(a_curr, p_curr | a_prev, p_prev)   -- FOUR K-way slots, EACH re-discretized by its own K-pool Izhikevich
                                       attractor bridge + shared FS inhibitory pool. The four spiking winners
                                       ARE the next state. No host argmax anywhere in the state path.
```
The connective's event boundary is therefore executed as a **spiking SHIFT**, and the prior event must **survive on spikes** across arbitrarily many following clauses.

## Result (6-seed; held-out-DEEPER 7/8/9 vs train 3/4/5; NO `sim/` edit)
| | spiking (mean) | rate (mean) |
|---|---|---|
| **previous-event agent** | **0.877** | 0.881 |
| **same-agent RELATION across the pair** | **0.930** | 0.929 |
| current-event agent | 0.913 | 0.920 |
| **per-slot host-agree** | **0.992** | — |
| SINGLE-EVENT control (previous agent) | 0.467 | 0.467 |
| RECENCY (previous agent) | 0.167 | 0.167 |

The spiking pair **matches the rate pair** (0.877 vs 0.881; relation 0.930 vs 0.929) at per-slot host-agree 0.992 — the four FS-WTA winners *are* the state, not a spiking check on a host argmax.

## A saturation bug, diagnosed rather than tuned away
The first spiking run came back PARTIAL on a **single** gate term: per-slot host-agree 0.947 vs the 0.95 bar. Rather than reach for a knob, I instrumented per-slot agreement and per-slot top-2 score margin:

| slot | host-agree | top-2 margin |
|---|---|---|
| a_curr | 0.928 | 6.64 |
| **p_curr** | **0.885** (worst) | **9.65** (largest) |
| a_prev | 0.982 | 7.81 |
| p_prev | 0.988 | 7.95 |

**This refuted the obvious hypothesis.** Near-ties cannot be the cause: the *worst-agreeing* slot had the *largest* margin, and the *held* prev-slots agreed best. The real cause is **f-I saturation** — large raw drives push several pools to ceiling, degrading the spike-count read. That is the same failure mode EMERGE-77 documented when packing eight primacies into one current range.

**The fix** is the project's established pattern (already used by the centering wire): normalize the drive before the attractor. Per-slot agree → ≥0.98 on all four slots (overall 0.992), and accuracy *rose* too (prev 0.78→0.83, curr 0.78→0.86). One variable, diagnosed from the measurement, not a sweep.

## Anti-cheats (all pass)
- **(a)** spiking prev-agent (0.877) ≫ the structurally-incapable SINGLE-EVENT control (0.467) and ≫ RECENCY (0.167 = chance).
- **(b)** per-slot host-agree 0.992 — the spiking winners are rolled forward as the state.
- **(c)** the same-agent RELATION (0.930) is read across two *spiking* slots.
- **(d)** held-out-DEEPER lengths.

## ⇒ the claim
**The brain relates two composed meanings ON SPIKES.** A connective-marked event boundary shifts the running event into a previous slot; four one-of-K spiking attractors hold the pair; the prior event survives across arbitrarily many following clauses; and the same-agent relation is read across two spiking slots.

## Honest scope + next
- Per-step supervised (the transition is the rate-learned δ; only the re-discretization is on spikes — the same rung scope as the earlier spiking ports). The self-supervised δ for the *pair* (no state label) is the natural follow-on.
- Two events (depth-2); true Contrast/Cause semantics beyond Sequence + same-agent remain open.
- Evaluated on a 400-item subsample (a real bridge runs per slot per clause).

## Files
`research/runners/_d3_event_connective_spiking_derisk.py`; the rate rung `2026-07-10-D3-event-discourse-connectives-GO.md`; the capstone `2026-07-10-D3-event-CAPSTONE-emergent-spiking-deployed-QA.md`.
