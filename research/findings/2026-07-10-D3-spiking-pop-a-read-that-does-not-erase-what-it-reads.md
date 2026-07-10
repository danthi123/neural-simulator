# D3 SPIKING POP — the brain resumes a protagonist by **reading a spiking attractor without erasing it**

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_popgate_spiking_agent_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** GO (6-seed, all anti-cheats).

## The substrate asymmetry this rung tests

The held prior event lives in a **persistent slow-NMDA attractor** on a real `SimulationBridge`: with the gate closed it
receives **zero input** and sustains its own firing indefinitely.

* The **push** (a boundary) is a **write**. It must *destroy* what the slot holds — and the earlier rung found it only
  works if the clear **outlasts τ_NMDA**, or the old event re-ignites.
* The **pop** (a return) is a **read**. It must *not* destroy what it holds.

On a point-neuron attractor those are genuinely different problems: a read that drives the pool risks re-igniting a
different assembly or erasing the one it is reading. Here the read drives **nothing** (`assert not zero.any()`), so
non-destructiveness is a **property of the mechanism, not a tuned parameter** — and it is *gated on*, not assumed.

## Result (6-seed, 15 resumption discourses/seed; a real bridge steps per clause)

| "who is doing it now?" immediately after a discourse pop | |
|---|---|
| **spiking (read the attractor)** | **0.689** (0.467 – 0.933) |
| pop-lesion — the **identical** trained model, `r` forced to 0 | 0.111 |
| **stateless slot** (`recur=0`; nothing to read) | 0.111 |
| keep answering the pre-pop agent | 0.056 |
| recency | 0.067 |
| host twin (the slot replaced by an exact host copy) | 0.778 |

| | |
|---|---|
| **held slot survives its own read** | **0.974** (232 checks) |
| deployed gate `r` on **pops** / on **boundaries** | **0.844 / 0.032** |
| BEFORE / ordinary NOW | 0.678 / 0.733 |

## What each control rules out
- **pop-lesion (0.111)** — same weights, same everything, read gate shut. The single-variable contrast: the resumption is
  the gate, not the register's mere existence.
- **stateless slot (0.111)** — with `recur=0` there is no attractor to read, and there is **no host fallback anywhere**: a
  silent slot means nothing is held. (A prior rung measured a stateless control being silently rescued by a Python
  variable and scoring *identically* to the spiking one. A control that cannot fail is not a control.)
- **keep-the-same-agent (0.056)** and **recency (0.067)** — the two shortcuts a listener could take.
- **gate separation (0.844 vs 0.032)** — both pops and boundaries carry a connective; a gate keying on the connective
  would pop *at a boundary*, overwriting the present with a stale past. It keys on the pronoun subject instead.

## Honest reporting
- **The read is non-destructive at 0.974, not 1.000.** The read drives nothing, but it still *advances* the attractor 30
  steps, so a near-tied assembly can flip. The three seeds where this happens (44: 0.950, 101: 0.958, 102: 0.938) are
  exactly the three carrying the gap to the host twin. Named residual; the adaptation wash-out that fixed an analogous
  Izhikevich accumulation elsewhere in this repo is the obvious next lever.
- **Substrate cost: −0.089** (spiking 0.689 vs host twin 0.778). Reported, not hidden.
- The rate model computes `a_curr ← r·a_prev + (1−r)·δ`; a spiking read is a winner over population firing, so the convex
  combination **discretises** to `r > 0.5 ⇒ a_curr = argmax(spikes)`. Stated, and the host twin prices it.
- Seed 101 is the weak seed throughout (resumption 0.467).
- 15 resumption discourses/seed — a real bridge runs per clause, so this is the slow path.

## ⇒ the claim
A brain that was **never told who any agent is** learns a discourse transition from prediction alone. A **boundary opens a
write gate** and the running event is cleared-then-loaded into a self-sustaining spiking attractor. A **return marker
opens a read gate**, and the brain **resumes the protagonist it had set aside — by reading that attractor's spikes,
without erasing them** (0.689, against 0.111 for the identical register with the read gate shut, and 0.111 for a slot with
no attractor to read).

Push on a boundary, pop on a return: **one register, two gates, one attentional stack — on spikes.**

## Next
`a_curr` is still re-chosen on the host each clause. Putting it on **its own** persistent attractor makes the whole event
pair spiking, and turns both gates into attractor→attractor transfers: the push becomes clear-then-load *from* one
attractor *into* the other, and the pop becomes the same in reverse. That is the rung where the register stops being a
host vector with a spiking memory attached, and becomes two competing spiking assemblies.

## Files
`research/runners/_d3_event_popgate_spiking_agent_derisk.py`; the rate mechanism
`2026-07-10-D3-pop-gate-the-discourse-pop-is-a-gated-copy-OUT.md`; the rate deployment
`2026-07-10-D3-pop-gate-deployed-the-brain-resumes-a-protagonist.md`; the persistent slot
`2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`.
