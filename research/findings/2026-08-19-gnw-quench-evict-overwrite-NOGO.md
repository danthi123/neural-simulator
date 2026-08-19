---
status: live
type: finding
lane: laneC
date: 2026-08-19
---

# GNW quench-evict OVERWRITE — the affect active-clear mechanism (a transient open-loop FS quench) does NOT evict a held GNW coalition: a self-sufficient workspace attractor is INHIBITION-RESISTANT. Strong GABA_A is genuinely delivered (g_i 150->320) yet the incumbent never leaves plateau (0/6); above the E_i reversal it destabilizes UPWARD (n=3). This reproduces the STN-veto / Rung-2c inhibition-resistance wall from an independent direction and localizes the residual to the RECURRENCE.

**Date:** 2026-08-19
**Runner:** `research/runners/_gnw_quench_evict_overwrite_derisk.py` (FORK of `build_swap_bridge` from `_gnw_active_overwrite_derisk.py`, run in its SUPRA-CRITICAL disjoint headline config: overlap=0, uniform recurrence w=34, NO WTA + divisive-norm `norm_pool` + tonic `thal`. Adds ONE additive spiking `quench_fs` FS pool = the affect active-clear mechanism transferred: dense GABA_A onto every workspace-used unit, recruited by an external drive during a transient clear window then RELEASED. Reuse-by-import of the Rung-1 ignition constants, Rung-2 ignition criterion, Rung-2c dense-population, the _p1_2 snapshot/restore, and the active-overwrite stepping/read instruments. **NO `sim/` edit**.)
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** **NO-GO** — a genuine SWAP (incumbent A drops to baseline AND challenger B ignites AND n settles to 1) on **0/6** seeds across the whole quench_w x window x b_overlap x quench_thal_w x w_rec grid tried. The incumbent is NEVER evicted (old-state windowed rate stays at the ~1/3 period-3 plateau on every seed and every operating point). The failure is a mapped substrate property, not a tuning miss.
**Artifacts:** `research/findings/raw/_gnw_quench_evict_6seed.json` (6-seed, w=16) + `research/findings/raw/_gnw_quench_evict_regime_smoke.json` (seed-42 regime map) (+ `.prov.json` sidecars).
**Builds on / cites:** the LEAD — the affect active-clear GO `research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md` + biology `research/biology/affect-active-clear.md` (a transient open-loop quench_fs pulse evicts a saturated BISTABLE affect latch where every STANDING brake failed).
The prior GNW eviction NO-GOs this reproduces/extends: `research/findings/2026-08-18-gnw-stn-stop-veto-NOGO.md` (a standing GPi brake: n_post stuck at 2, then E_i reversal -> n=4), `research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` (inhibition alone cannot evict a frozen attractor), `research/findings/2026-08-18-gnw-bg-thalamus-gate-overwrite-NOGO.md` (gate admits B, cannot evict a thalamus-independent A -> n=2), `research/findings/2026-08-18-gnw-active-overwrite-NOGO.md` (break-in vs lockout catch-22).
The WORKING eviction it points back to: `research/findings/2026-08-18-gnw-rung2d-weakenable-recurrence-GO.md` (STD self-eviction = attack the RECURRENCE, GO 6/6).
Biology: **Compte, Brunel, Goldman-Rakic & Wang 2000**, *Cereb. Cortex* 10:910 (persistent-activity termination by "nonspecific excitatory input recruiting feedback inhibition" — the quench_fs GABA_A pulse is its spiking realization; the transfer was hypothesized from THIS canonical mechanism). **Dehaene & Changeux 2011**, *Neuron* 70:200 (an ignited state must be destabilizable). Corpus-first + external check were run and logged BEFORE building (the research gate FIRED after 3 prior levers on this defect).

## The hypothesis (the LEAD) and why it was well-motivated
The GNW thought-swap = replace the coalition the workspace holds with a new one. A BG-style gatekeeper already lets the NEW thought IN (`bg-thalamus-gate`: admit 6/6); the UNSOLVED half is EVICTING the OLD held coalition. Every prior eviction lever failed on a SUPRA-CRITICAL (self-sufficient) incumbent, and all were of the STANDING-BRAKE / GATE class (STN veto, BG-thalamus gate, active-overwrite WTA/STD, Rung-2b SFA).
This is EXACTLY the pattern the affect ratchet showed: every OUTWARD/STANDING brake failed structurally, and the ONE thing that worked was an ACTIVE, TRANSIENT, OPEN-LOOP CLEAR — a spiking `quench_fs` pool firing strong GABA_A for a window that exceeds the drain threshold, after which the OFF fixed point HOLDS with zero standing force. The de-risked hypothesis: workspace eviction needs the same active-transient-open-loop quench, not a gate/veto.

## What was built (all spiking/synaptic; the anti-cheats ARE the result)
A dedicated `quench_fs` FS pool (30 neurons, appended additively) projects dense GABA_A onto every workspace-used unit (I_TO_E, the same inhibitory mechanism as the working `norm_pool`). During a transient CLEAR window it is recruited by an external drive; it is RELEASED before the identity read (open-loop). Then challenger B is driven in (the IN-gate).
Four arms per seed on the same substrate: **transient** (headline: quench then release then B), **standing** (quench HELD ON through B + read = the standing-brake contrast), **no_quench** (drive B on the held A, no clear = the co-ignition/lockout baseline), and a **lesion** build (quench GABA_A weight 0, timing/drive identical = the load-bearing control).
A `quench_thal_w` limb (quench also inhibits thal = open the whole thalamocortical loop) and a `b_overlap` limb (the challenger volley arrives as the quench wanes) were added and swept as surpass attempts.

## The result — 0/6 swaps; the incumbent is INHIBITION-RESISTANT (per-seed, from the cited 6-seed artifact, w=16)
<!--derived-->

| seed | A rate DURING clear (`old_rate_midclear`) | g_i on A: rest -> mid-clear | old-state rate AFTER (`old_residual_post`) | n_ignited pre->post | quench rate clear / read | swap |
|---|---|---|---|---|---|---|
| 42  | 0.344 | 150 -> 318 | 0.333 | 1 -> 3 | 0.251 / 0.000 | NO |
| 43  | 0.340 | 154 -> 323 | 0.333 | 1 -> 3 | 0.259 / 0.000 | NO |
| 44  | 0.338 | 155 -> 322 | 0.333 | 1 -> 3 | 0.261 / 0.000 | NO |
| 100 | 0.328 | 163 -> 328 | 0.333 | 1 -> 3 | 0.264 / 0.000 | NO |
| 101 | 0.336 | 147 -> 305 | 0.333 | 1 -> 3 | 0.250 / 0.000 | NO |
| 102 | 0.343 | 157 -> 329 | 0.333 | 1 -> 3 | 0.257 / 0.000 | NO |

The two load-bearing reads: (1) **the inhibition IS delivered** — g_i on the incumbent's neurons roughly DOUBLES during the clear (per-seed rest ~150 -> mid-clear ~320, above the reversal knee the STN-veto measured), and the quench pool fires ~0.25 spikes/neuron/step during the clear and **0.000 at the read** (open-loop confirmed **6/6**). (2) **the incumbent does not budge** — its firing rate DURING the clear stays at the ~1/3 period-3 plateau (`old_rate_midclear` ~0.33 on every seed), and it is at plateau again at the read (`old_residual_post` 0.333 **6/6**).
The recurrent excitation (100 units x w=34) dominates the pooled feedback inhibition; the attractor cannot be pushed across its basin.
<!--derived--> A directly-measured control (a live diagnostic, not in the committed artifacts) confirms polarity is correct: with A ignited, driving the quench pool alone raises g_i on the workspace from 73 to 214-281 while A keeps firing at plateau, and `quench2ws` stores the same magnitude/sign as the working `norm2ws` (both mean 16.0, inhibitory via the output-inhibitory index list).

**The regime map (seed 42, `_gnw_quench_evict_regime_smoke.json`, b_overlap=0):** as the quench weight rises, g_i climbs monotonically (171 -> 186 -> 253 -> 318 at w = 2, 4, 8, 16) but the incumbent stays at plateau throughout, and the OUTCOME transitions **lockout (n=1, w<=2) -> co-ignition (n=2, w~4) -> destabilize-UPWARD (n=3, w>=8)**. The n=3 is the STN-veto's documented physics: at g_i > ~200 the RESTING units (the undriven third pattern) are driven below E_i = -75 mV, the GABA_A current reverses to depolarizing, and post-inhibitory REBOUND ignites them. So there is no weight window that evicts A: too weak leaves A untouched (B locks out), enough to matter never crosses A's basin and instead rebounds the workspace upward.

**The contrast arms (all 6/6, confirming the mechanism reads are honest):** the **standing** brake also fails to swap (0/6) and is NOT silent at read (q_read ~0.25 -> it correctly fails the open-loop anti-cheat, reproducing the STANDING-brake class). The **no_quench** and **lesion** arms hold A at plateau with B locked out (n=1, old_res 0.333) — the co-ignition/lockout PARTIAL, and the lesion (GABA_A weight 0, identical timing/drive) shows the quench limb is the only difference. Determinism (build-twice hash) holds 6/6.

## Root cause and why the affect mechanism does NOT transfer
The affect active-clear worked because its target was a single BISTABLE opponent latch driven by SLOW NMDA, with **no standing re-drive** once cleared: a transient shunt pushed the state across the basin ONCE and the OFF fixed point held. Two properties of the GNW workspace break that transfer, and both are inherent, not tunable:
1. **The incumbent is self-sufficient and inhibition-resistant.** A supra-critical recurrent clique (w=34 over 100 units) generates far more recurrent excitation than a feedback-inhibition pool can shunt — g_i doubling to 320 does not stop A firing. Even a 120-neuron quench at w=80 leaves `old_rate_midclear` at plateau (tested). The affect FS pool could dominate a 2-3-pool latch; it cannot dominate a workspace attractor.
2. **The quench is non-selective and symmetric.** It hits A, B, and the resting pattern equally, so it cannot produce the ASYMMETRIC, incumbent-specific weakening a selective swap needs. With divisive normalization two self-sufficient attractors simply co-ignite (n=2); strong enough to matter, the symmetric release rebounds the whole workspace upward (n=3). Opening the thalamocortical loop too (quench -> thal) does nothing, because A does not depend on thal — confirming the BG-gate diagnosis from a second angle.

This is the SAME inhibition-resistance wall the STN-veto (a standing GPi brake) and Rung-2c (frozen-attractor disinhibition) hit; reaching it from the affect-quench direction rules out "the brake was just the wrong shape." **The wall is the self-sufficiency of the RECURRENCE, not the inhibition method.** Soma-level feedback inhibition — standing OR transient, closed- OR open-loop — cannot evict a self-sustaining GNW coalition.

## The residual, precisely quantified — and the next mechanism (already a GO)
Residual old-state activity after the operation: **0.333 (the full period-3 plateau) on 6/6 seeds, at every operating point** — i.e. zero eviction, not a partial one. The IN-gate is confirmed intact (B reaches 0.333 whenever driven; the co-ignition n=2 shows B CAN enter — A just will not leave). The mapping is unambiguous: **eviction must attack the RECURRENCE (make the incumbent's own loop sub-critical), not the soma.**
That is exactly what the working eviction does — `Rung-2d` (STD depression of the recurrent synapses through the incumbent's OWN use) evicts 6/6 as a slow clear-then-reload. The capability is therefore NOT abandoned; this finding banks the affect-quench (soma-inhibition) METHOD with a measured reason and converges the whole eviction arc onto the recurrence-weakening route. A quench pulse could still ACCELERATE a recurrence-weakened collapse, but it is the recurrence change — not the inhibition — that would be load-bearing.

## Honest limits
- **Levers against this defect (research-gate accounting):** this is the 4th distinct eviction mechanism on this workspace (STN veto, active-overwrite WTA/STD, BG-thalamus gate, now affect-quench); the corpus-first + external research was done BEFORE building (logged). All four converge on the same recurrence-self-sufficiency wall.
- **The `additive_substrate` hash anti-cheat is 0/6** (appending the quench pool perturbs the base-slice izh-param draw — the RNG prefix property does not hold on this engine). This does NOT affect the result: the workspace still ignites and behaves as the supra-critical attractor (n_pre=1 6/6), and the stronger DETERMINISM anti-cheat (build-twice identical hash) holds 6/6. Reported, not hidden.
- Pools are hand-wired dense frozen populations (not self-organized); clear/drive timing is host-supplied external drive (world/body-legitimate); this is a de-risk, not wired to production.
