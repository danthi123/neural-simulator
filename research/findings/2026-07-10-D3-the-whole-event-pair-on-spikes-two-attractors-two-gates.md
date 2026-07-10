# D3 — the **whole event pair on spikes**: two persistent attractors, two gates, both directions an attractor→attractor transfer

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_pair_spiking_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** GO (6-seed, all anti-cheats).

## What changed

Until now the register was a **host vector** (`a_curr`) with a **spiking memory** (`a_prev`) bolted on. Both slots are now
persistent slow-NMDA attractors on **one** `SimulationBridge`, and both gates are transfers between attractors:

```
PUSH (boundary):  READ a_curr's spikes  ->  CLEAR a_prev  ->  LOAD a_prev
POP  (return)  :  READ a_prev's spikes  ->  CLEAR a_curr  ->  LOAD a_curr
otherwise      :  the transition proposes  ->  CLEAR a_curr -> LOAD a_curr
```

Every state the register holds is now a self-sustaining spiking assembly, and **both** transition inputs are read out of
spikes.

## Result (6-seed; a real bridge steps per clause, 10 discourses/seed per pool)

| "who is doing it now?" right after a discourse pop | |
|---|---|
| **pair-spiking (both slots are attractors)** | **0.583** (0.300 – 1.000) |
| pop-lesion — the identical model, read gate shut | 0.117 |
| **stateless** (`recur=0` on both slots) | 0.117 |
| keep answering the pre-pop agent | 0.133 |
| recency | 0.050 |
| host twin (both slots replaced by host copies) | 0.750 |

| "who was doing it before?" — restricted to discourses needing the slot to HOLD (≥2 clauses since the push) | |
|---|---|
| **pair-spiking** | **0.617** |
| **stateless** | **0.200** |
| host twin | 0.833 |

| substrate properties | |
|---|---|
| **the POP leaves `a_prev` intact** (it writes `a_curr` while reading `a_prev`) | **0.963** |
| each slot survives its own read | **1.000** |
| gate `r` on deployed pops / boundaries | 0.843 / 0.030 |
| ordinary NOW | 0.750 |

## Two things this rung actually taught

### 1. The clear law is strength × duration, not duration alone
The push rung concluded *"the clear must outlast τ_NMDA, or the incumbent re-ignites"* — calibrated on a slot written
**once per discourse**. `a_curr` is written **every clause**, so it carries a far larger residual NMDA conductance into
each clear, and the validated 250-step / 1500 pA clear **failed**: write-read-back fidelity 10/12, and the failures land
exactly where the incumbent's residual `g_nmda` peaks — the slot reads back the **old** content. Instrumented, not guessed.

Both levers close it, **12/12** each: a **longer** clear (400 steps @ 1500 pA) or a **stronger** one (250 steps @ 4000 pA).
So the governing quantity is the **product of inhibition strength and duration**. Held hyperpolarized hard enough, the
Mg²⁺ block prevents the residual NMDA current from re-igniting the assembly *without* waiting out the time constant. The
stronger clear is taken (cheaper in wall-clock). This sharpens the earlier finding rather than contradicting it.

### 2. Separate inhibition per slot is a wiring requirement, not a preference
The pop **writes `a_curr` while reading `a_prev`**. A shared fast-spiking pool would silence both, so the pop would erase
the very assembly it is reading. Each slot therefore gets its own inhibitory pool, and the `pop_leaves_prev_intact`
counter measures it directly: **0.963**.

## A defect I found in my own control
The first 6-seed run failed its `BEFORE vs stateless` term — and the per-seed data showed **stateless BEFORE at 0.6 on
seed 100**. A slot with `recur=0` cannot hold anything; but if the **last clause is a boundary**, the push has just driven
it, and the 30-step read catches the decaying load trace. The BEFORE pool was not controlling for how long ago the push
happened, so the stateless control was being rescued by the *recency of the write* rather than by memory.

Fixed by requiring **≥2 clauses since the last push**, so the trace has decayed and only a real attractor can still be
holding — the same "the question must require the mechanism" logic that defines every other pool here. Stateless BEFORE
falls to 0.200, and the contrast becomes +0.417. The register's numbers were never the problem; the control's
discriminating power was.

## Honest reporting
- **Substrate cost is now larger: −0.167** on resumption (0.583 vs host twin 0.750), up from −0.089 when only `a_prev` was
  spiking. Expected: *both* transition inputs are now read out of spikes, so read noise enters twice per clause.
- `pop_leaves_prev_intact` is 0.963, not 1.000 (seeds 44 and 101 lose one check each).
- Seeds 101 and 102 are weak on resumption (0.300 each); seed 100 is perfect (1.000).
- 10 discourses per seed per pool — a real bridge runs ~330 steps per clause, so this is the slow path.
- The convex combination `a_curr ← r·a_prev + (1−r)·δ` discretises on a spiking read; the host twin prices that.

## ⇒ the claim
A brain that was **never told who any agent is** learns a discourse transition from prediction alone. Both events it is
tracking live as **self-sustaining spiking assemblies on one bridge**. A **boundary** transfers the running event into the
held slot; a **return marker** transfers it back, **without erasing** the assembly it reads. Resumption 0.583 against
0.117 for the identical register with the read gate shut and 0.117 for slots that cannot hold; "who was doing it before?"
0.617 against a stateless 0.200 once the question genuinely requires memory.

**One brain, two attractors, two gates — an attentional stack made of spikes.**

## Next
- The **substrate cost** (−0.167) is now the headline residual, and it is read noise: each clause reads two attractors.
  A longer read window, or population-vector rather than argmax read-out, is the cheap first lever.
- The **transition itself** (`δ`) is still a host tanh/softmax. Both *gates* are structural and both *memories* are
  spiking; the remaining host computation is the map from (a_curr, a_prev, patient, clause-code) to the next agent. That
  is the next thing to put on the substrate — and it is the one piece that is genuinely *learned*, so it is where a
  spiking learning rule has to earn its place.

## Files
`research/runners/_d3_event_pair_spiking_derisk.py`; the single-slot spiking pop
`2026-07-10-D3-spiking-pop-a-read-that-does-not-erase-what-it-reads.md`; the persistent slot + the original clear law
`2026-07-10-D3-persistent-spiking-slot-hold-and-clear-then-load.md`.
