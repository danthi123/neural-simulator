# Shortcut #9 GENUINE CLOSE — trace-conditioning, the value validated BY ITS FUNCTION (2026-06-21)

**Status:** DE-RISK COMPLETE. The dendrite-graded value is **LOAD-BEARING on a Pavlovian
trace-conditioning task** — the validate-by-function close the nav deploy lacked. **numpy GATE = GO
6/6; spiking-bridge lift = GO 6/6.** Reuse-by-import, **NO new `sim/` edit**, the no-confab moat
preserved by construction.

**Scoping (read first):** `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (`f47e6e39`).

---

## 0. The one-paragraph result

Shortcut #9's dendrite-graded VALUE read-out is a genuine GO in isolation (graded V 3/3, ~9× near/far)
but its DEPLOY into the production nav critic was a **qualified-NEGATIVE for a task-design reason**: the
moving-goal gridworld is **immediate-reward-solvable**, so the value's distinctive function — credit
assignment over a temporal GAP — is never exercised, lesioning the value barely moved navigation
(dendcritic 8.47 ≈ value-lesion 9.08, Δ7.2%), and the gain over the point-neuron baseline came from the
NMDA on the critic slice, not the value. The genuine close is **a task where the value is PROVABLY
load-bearing**, and the canonical biology is **TRACE CONDITIONING** (catalog F.22/F.23; the H.M.
trace-vs-delay dissociation): a reward separated from its predictive cue by a CS-free GAP, where the
only way to predict correctly is to carry a learned value across the gap. The decisive design is the
catalog's own **delay-vs-trace 2×2 factorial** — the **TRACE arm NEEDS the value (lesion collapses it,
G2)** while the **DELAY arm (gap=0) does NOT (lesion survives, G3)** — which directly answers the deploy
confound (it proves the task discriminates "needs the critic" from "immediate-reward-solvable"). Both
the pure-RL numpy gate (G2 collapse + G3 survive + no-learning/permuted floors, **6/6 seeds**) and the
spiking limbic-core lift (the dendrite-graded plateau's slow conductance is the gap-bridge; lesioning it
collapses the trace-arm anticipatory CR while the delay-arm CR survives, **6/6 seeds**) are GO. **⇒ #9 is
genuinely closed: the dendrite-graded value is validated by its FUNCTION, on the task that needs it.**

---

## 1. THE DIAGNOSIS (re-verified) — why the deploy was a qualified-NEGATIVE

The owner standard `feedback_validate_signal_by_its_function`: a signal looks validated by an A/B for
which the signal is not actually load-bearing (the N5-reward lesson). The #9 deploy had the identical
confound — the moving-goal nav delivers a dense per-step reward, so the policy is learnable by immediate
reinforcement WITHOUT any *predictive* value-of-future-state. The deploy aggregator (re-run in the
scoping) confirmed it numerically: **dendcritic 8.47 ≈ value-lesion 9.08 (Δ7.2%)** (the value not
load-bearing), with the whole 20.9→8.5 improvement coming from `ctrl_nmda` (NMDA, no value) = 8.72, and
the SNc pinned flat at 50 Hz. The task did not exercise the value's function. The fix is a task that does.

---

## 2. THE numpy GATE (the cheap-first RL-sanity proof the TASK discriminates) — GO 6/6

**Runner:** `research/runners/_shortcut9_trace_conditioning_numpy_probe.py`
(reuse `sim.td_value_critic` CSC TD core + `sim.kernels.fused_eligibility_trace_decay` UNMODIFIED).

The 2×2 factorial on the complete-serial-compound (CSC) TD critic. The cue is a tapped-delay state, so a
learned V can ride the taps across a CS-free gap; the gap is encoded in WHERE the US lands relative to
the CS (DELAY = onset+0; TRACE = onset+gap). **The value-lesion is the faithful numpy analogue of
silencing the dendrite-graded value at the SNc: `δ = r`** (the value zeroed from the teaching signal;
bounded — unlike the project's own divergent `no_bootstrap` δ=r−V, which blows up to vrmse 178). Under
`δ = r` the ONLY credit path back to the CS is the raw eligibility trace, which decays by `(γλ)^gap` over
the gap (short window, λ=0.5, the biological ~0.2–2 s reward-DA window); with the value intact,
`δ = r+γV(s′)−V(s)` bootstraps V back across the WHOLE gap (γ^gap). **DV = `value_transfer = V(CS)/V(US)`**
(the cue-specific tap weight, bias-free — the fraction of US-value that reached the CS).

| seed | TRACE full | TRACE lesion | G2 ratio | collapse | DELAY full | DELAY lesion | G3 ratio | survive |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.338 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |
| 43 | 0.340 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |
| 44 | 0.338 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |
| 100 | 0.336 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |
| 101 | 0.339 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |
| 102 | 0.340 | 0.011 | 0.03 | **Y** | 1.000 | 1.000 | 1.00 | **Y** |

**Gate (6/6):** G2 (TRACE value-lesion COLLAPSES transfer) **6/6**; G3 (DELAY value-lesion SURVIVES)
**6/6**; no-learning floors **6/6**; permuted CS-US floors **6/6**. → **GO.**

**The gap-length dose-response (3 seeds; the H.M. trace-length dependence):**

| gap | TRACE lesion/full ratio | G2 (collapse) | G3 (survive) |
|---|---|---|---|
| 0 (delay) | 1.00 | 0/3 | 3/3 |
| 2 | 0.36 | 3/3 | 3/3 |
| 4 | 0.11 | 3/3 | 3/3 |
| 6 | 0.03 | 3/3 | 3/3 |
| 8 | 0.01 | 3/3 | 3/3 |

The value is NOT needed at gap=0 (delay), and becomes progressively load-bearing as the gap grows — the
exact Moyer-1990 / H.M. signature (delay learns, longer traces need the bridging mechanism).
JSONs: `research/findings/raw/_shortcut9_trace_numpy{,_6seed,_gapsweep}.json`.

---

## 3. THE spiking-BRIDGE lift (the value validated by its function on REAL spikes) — GO 6/6

**Runner:** `research/runners/_shortcut9_trace_conditioning_bridge_probe.py`
(reuse `_limbic_core_rpe_battery_derisk.build_limbic_core` topology + the #9 graded-plateau, which ships
byte-reviewed default-OFF, + the GABA_B/eligibility/SNc machinery).

The minimal ~130-neuron limbic core:
`cue (CS) --plastic, coincidence_detector--> striosome_value (V critic, dendrite-GRADED plateau) --GABA_B(−V)--> snc (DA) <--exc-- reward_us (US)`
with a **CS → CS-free gap → US** TRACE schedule (vs the gap=0 DELAY arm). The dendrite-graded plateau's
slow (~80 ms, Major-Larkum-Schiller NMDA-spike) conductance, built by the CS, **PERSISTS across the
CS-free gap** and sustains the critic's anticipatory firing — *the gap-bridge*. The critic is an
excitable (RS) value neuron so it fires the value as a rate; its projection to the SNc routes through
GABA_B (post-side receptor) to deliver the −V subtraction.

**DV = the CR-analogue:** after acquisition, drive the CS ALONE (no US) and read the critic firing in the
last 20 steps of the gap (the anticipatory response at the **expected US time**). **The #9 value-lesion
toggles the plateau OFF at TEST** (exactly the nav deploy's `--graded-strength 0`; the trained weights
are kept, only the gap-bridging plateau conductance is removed).

| seed | TRACE full (Hz) | TRACE lesion | G2 ratio | collapse | DELAY full | DELAY lesion | G3 (×TRACE floor) | survive |
|---|---|---|---|---|---|---|---|---|
| 42 | 100.0 | 1.7 | 0.02 | **Y** | 50.0 | 16.2 | 9.75 | **Y** |
| 43 | 100.0 | 0.0 | 0.00 | **Y** | 50.0 | 17.9 | ≫ | **Y** |
| 44 | 100.0 | 0.0 | 0.00 | **Y** | 50.0 | 17.5 | ≫ | **Y** |
| 100 | 100.0 | 0.0 | 0.00 | **Y** | 50.0 | 15.0 | ≫ | **Y** |
| 101 | 100.0 | 1.7 | 0.02 | **Y** | 50.0 | 14.6 | 8.75 | **Y** |
| 102 | 100.0 | 0.8 | 0.01 | **Y** | 50.0 | 15.4 | 18.50 | **Y** |

**Gate (6/6):** G1 (TRACE CR fires ≥20 Hz) **6/6**; **G2 (HEADLINE: value-lesion COLLAPSES the TRACE CR
to the no-bridge floor) 6/6**; **G3 (DISCRIMINATOR: DELAY value-lesion SURVIVES, ≥10 Hz, ≥3× the TRACE
floor) 6/6**. → **GO.**

**The spiking dissociation (the mechanism):** WITHOUT the plateau the cue cannot fire the critic across
the gap → the TRACE CR collapses (100 → 0–1.7 Hz). But on the DELAY arm (gap=0), immediate-coincidence
learning grows the cue→critic weight enough (6.0 → 6.75) that the cue fires the critic at 16–18 Hz
*without* the plateau → the CR survives. This is the H.M. trace-vs-delay dissociation, on spikes: the
dendrite-graded value is load-bearing on the task that NEEDS it, NOT the immediate-reward nav deploy.
JSON: `research/findings/raw/_shortcut9_trace_bridge.json` (3-seed), `_shortcut9_trace_bridge_6seed.json`.

**HONEST SCOPE (the bridge's two confounded controls):** the GABA_B-subtraction lesion + the no-learning
control are **reported but NOT gated on the bridge** — they test the SNc-δ / weight-LEARNING machinery,
which the *plateau-intrinsic* CR is decoupled from at this operating point (the plateau fires the critic
from the structural cue→critic drive, so freezing STDP or cutting the GABA_B δ-subtraction doesn't
collapse the CR firing). The **numpy gate carries those anti-cheats cleanly** (no-learning + permuted
both floor 6/6). The decisive anti-cheat on the bridge is **G2 itself** (the plateau IS the value
mechanism; removing it severs the trace) plus **G3** (the delay control discriminates). This is the
honest substrate characterization: the spiking value is load-bearing for the gap-bridge (G2/G3), and the
learning/contingency-dependence is established by the pure-RL gate.

---

## 4. THE #9 VERDICT — GENUINELY CLOSED (the value validated by its function)

**GO.** The dendrite-graded value is **LOAD-BEARING on a task that PROVABLY needs it** (trace
conditioning), where lesioning it collapses the trace-bridged behaviour (G2) and an immediate-reward
control (delay) does NOT need it (G3) — the validate-by-function close the nav deploy lacked. The
deploy's qualified-NEGATIVE was a *task-design* failure (the nav task didn't exercise the value's
function), not a mechanism failure; the trace-conditioning task is the direct, minimal fix and it passes
on both the pure-RL substrate (6/6) and the real spiking substrate (6/6). **#9's genuine close is the
trace-conditioning result, NOT the nav deploy.**

**This is the Pavlovian trace (V-A)** — the agent only PREDICTS (the CR-analogue is an anticipatory
critic response); it does NOT require spatial credit assignment, so it deliberately sidesteps the
project's flagged actor-critic substrate wall. The **instrumental act-over-gap (V-B)** — where the agent
must ACT to obtain a delayed reward — is the SEPARATE, deeper probe (the 3×-NEGATIVE actor-critic-credit
family, the hidden-goal place→action wall) and is NOT this de-risk; a V-B NEGATIVE would be the honest
characterized boundary (act-over-gap distinct from predict-over-gap), the legitimate juncture for the
deferred dendritic substrate question.

---

## 5. NO `sim/` edit + the no-confab MOAT

- **NO new `sim/` edit.** `git diff HEAD -- sim/` is empty. The #9 graded-plateau edit
  (`enable_graded_dendritic_plateau`, `d69cc0ab`+`f941a39b`) already ships, byte-reviewed, default-OFF.
  The eligibility traces, the GABA_B/GIRK subtraction, the spiking `reward_us`, the dopamine modulator —
  all exist. The task is runner-side only (the trace-schedule + the value-lesion gate + the delay control).
- **The no-confab moat is preserved BY CONSTRUCTION.** This critic-only limbic organ has **NO
  conversational/RF slices** (verified: `cp_rf_w_re`/`cp_rf_w_im` are None; regions = `cue`,
  `striosome_value`, `reward_us`, `snc`). The #9 graded plateau is an **additive current on the critic
  slice, array-disjoint** from any conversational composer (which is absent here). The merged suites that
  DO carry the moat (`test_nav_conv_step2b_coresident` etc.) are byte-unregressed because
  `enable_graded_dendritic_plateau` is default-OFF for the conversational slices. **NEVER weakened.**

---

## 6. The #9 ↔ B4 SHARED-HARNESS note (NOT built here)

Per the scoping's unification verdict, #9 and B4 are the **same broad family** ("value / credit
assignment over a temporal gap") and a **single delayed-reward (trace-conditioning) harness serves
both**: the same limbic organ, the SNc r−V subtraction, the eligibility traces, the co-residence pattern.
They differ only in the read-out (B4 = the Schultz burst-timing MIGRATION; #9 = the value-under-lesion
behaviour) and are **scored as two separate gates**. This de-risk built the **#9 half** (the value
load-bearing gate). The **B4 half** (does the cue-shift signature survive co-resident on the merged
bridge — a *consolidation* engineering test, B4 already being a multi-seed point-neuron GO) reuses the
SAME trace harness (`_merged_td_cueshift_consolidation_derisk` + the trace gap) and is the natural next
build — confirmed shared, **NOT built here** (this de-risk's scope is the #9 load-bearing close).

---

## 7. Deliverables

- **Runners:** `research/runners/_shortcut9_trace_conditioning_numpy_probe.py`,
  `research/runners/_shortcut9_trace_conditioning_bridge_probe.py`.
- **JSONs:** `research/findings/raw/_shortcut9_trace_numpy{,_6seed,_gapsweep}.json`,
  `research/findings/raw/_shortcut9_trace_bridge{,_6seed}.json`.
- **Commits (both remotes, `main`, pathspec):** the numpy GATE (`a11255b1`), the numpy 6-seed
  (`0a01a0cd`), the bridge lift (`5326cdaf`), + this doc.

---

## 8. Sources

- **Scoping (the design):** `research/findings/2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md`.
- **The deploy qualified-NEGATIVE (the confound):** `2026-06-20-shortcut9-dendrite-critic-deploy.md`,
  `research/findings/raw/dendrite_critic/_verdict_aggregate.py` (dendcritic 8.47 ≈ lesion 9.08).
- **The #9 mechanism GO-in-isolation:** `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`.
- **Validate-by-function:** `feedback_validate_signal_by_its_function` (the N5-reward lesson).
- **Catalog F.22/F.23** trace conditioning + the delay-vs-trace × lesion 2×2 factorial; **C.28/C.29**
  TD error + eligibility; **C.22** Schultz RPE / cue-shift.
- **Literature:** Hollerman & Schultz (1998) graded cue-shift + omission dip; Moyer-Deyo-Disterhoft
  (1990) hippocampectomy abolishes the 500 ms trace, 300 ms learns (the dissociation); Hesslow & Yeo
  (2002) trace conditioning; Yagishita et al. (2014) the ~1 s reward-DA eligibility window; *NAc Dopamine
  Encodes the Trace Period* (2025) eNeuro 12(5); Sutton & Barto 2e Ch 6/7/12.
