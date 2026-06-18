# Merged nav critic VALUE-TRAIN — V IS LEARNED co-resident (critic-grade flips, lesion-confirmed); the afferent-driven δ=r−V is GRADED but WEAK (~1.3×), capped by the position-blind up-state floor — BOUNDARY (2026-06-18, CYCLE 209 value-train build)

GPU value-train on the MERGED "one brain" bridge. Runner `research/runners/_merged_navcritic_valuetrain.py`
(builds on the GO op-map de-risk `2026-06-18-navcritic-valuetrain-opmap-derisk.md`). Reuse-by-import /
**NO `sim/` edit** — two runner-only kwargs plumbed through the merged builder
(`nav_conv_merged_bridge.py`: `nav_critic_convergent_upstate` + `nav_critic_homeostasis_mask`).

## 1. VERDICT — BOUNDARY (V IS learned co-resident; the afferent δ is graded-but-weak, structurally capped)

**The value critic LEARNS V(s) co-resident on the merged bridge** — the pair-then-reward DA-gated STDP grows
the PLASTIC `vs_place_context→striosome_value` weight ~20× and FLIPS the critic's place-grading from
far-dominant (untrained) to goal-dominant (trained). **The resulting δ=r−V is GRADED in the right direction
(predicted@goal < unpredicted@far) and LESION-confirmed (zeroing the GABA route collapses it), but the gap is
WEAK (~1.3×), not the op-map's 4–19×** — because the dense, NON-plastic convergent up-state arm (`vs_place_drive`,
needed to fire the cold MSN-D1 so the STDP has a post-spike) is **position-blind**, so it keeps V partly present
at the far location and compresses the (V_goal − V_far) contrast the SNc subtraction reads. This is the de-risk's
documented "A1 floor caps value-grading" tension, now QUANTIFIED on the merged bridge.

| signal | result (6 seeds: 42,43,44,100,101,102) | gate |
|---|---|---|
| **V LEARNED** (`vs_place_context→striosome_value` weight) | 0.19 → 3.77–3.95 (**~20× every seed**), DA→~1.0 across trials | ✅ the critic learns V co-resident, 6/6 |
| **critic-grade FLIP** (goal/far firing) | untrained **[0.51, 0.75, 0.58, 0.71, 0.62, 0.52]** (far-dominant) → trained **[1.70, 1.86, 2.47, 1.84, 2.06, 2.19]** (goal-dominant) | ✅ the LEARNED weight is load-bearing, **6/6** |
| **V-graded** (critic@goal > critic@far) | trained goal ~30–35 Hz > far ~14–19 Hz, 6/6 | ✅ 6/6 |
| **δ=r−V graded** (SNc predicted@goal < unpredicted@far) | pred ~54–61 < unpred ~74–78 Hz, gap **[1.28, 1.32, 1.36, 1.29, 1.28, 1.36]** (mean 1.32) | ⚠️ graded DIRECTION 6/6, magnitude RIGHT AT the 1.3 bar (BOUNDARY) |
| **anti-cheat: UNTRAINED-flat** | untrained critic-grade ~0.5–0.75 (< 1.3, NOT goal-graded); δ gap ~0.96 (flat) | ✅ the effect requires the trained weight, 6/6 |
| **anti-cheat: LESION** (zero the `striosome_value→snc` GABA_B route) | δ gap → the ~1.15 non-GABA floor: lesion **[1.14, 1.18, 1.16, 1.11, 1.15, 1.17]** (< trained every seed); 3-clean-seed relative test **3/3** | ✅ the synaptic GABA carries the learned δ increment |
| **anti-cheat: MOAT** | `what_does('dog','go')=='north'` AND `what_does('river','look') is None` | ✅ **6/6** — the dopamine scope=all broadcast does NOT perturb the frozen conv slice |

**Per-seed numbers (deterministic — the 3-seed re-run reproduced 42/43/44 exactly: gaps 1.28/1.33/1.36):** V grows
0.19→~3.9 (~20×) every seed; the critic-grade FLIP is 6/6 (untrained far-dominant ~0.6 → trained goal-dominant
~2.0); the δ gap is 1.28–1.36 (mean 1.32, σ~0.04 — tightly at the 1.3 boundary, NOT a wide spread); the LESION
drops the gap below the trained gap toward the ~1.15 floor on all 6 (the GABA carries the increment); the MOAT and
UNTRAINED-flat hold 6/6.

## 2. The precise mechanism that caps the δ (the BOUNDARY)

The MSN-D1 critic's rheobase is ~600 pA (the op-map f-I; threshold-insensitive). The **trained plastic afferent
alone cannot fire it** even at weight ~3.9 (diagnosed: `vs_place_context`-only drive at the goal → critic membrane
plateaus at ~−48 mV, below the −25 mV vt → 0 Hz). So the deployment read MUST drive the dense, NON-plastic
**up-state arm** (`vs_place_drive→striosome_value`, the B.02 convergent-excitation up-state) — exactly as the
deployed nav drive-injects the SAME grid Gaussian place code into BOTH afferents each step
(`g11_bg_runner.py:1843`, `:5810-5812`). That up-state arm is what fires the critic.

But the up-state arm is **position-blind**: at the FAR location it fires the critic ~19 Hz (from the dense
non-plastic synapses), delivering GABA_B onto the SNc *at far too*. The only goal-selective signal is the LEARNED
plastic boost (goal ~32 Hz vs far ~19 Hz). So the SNc subtraction reads (V_goal − V_far) ∝ (32 − 19) Hz → a modest
differential GABA_B → a modest δ gap (~1.3×). The op-map got 4–19× by driving the critic DIRECTLY at 1000 pA (a
strong, clean, position-FREE V the SNc fully subtracted) — that is the substrate CEILING (a "trained-V proxy"),
not the afferent-driven deployment read.

Confirmation the up-state floor is the limiter (diagnostics, seed 42, post-train w≈3.9):
- δ read WITH the up-state arm: gap **1.31**; WITHOUT it (plastic afferent alone): gap **1.07** (the trained
  weight barely fires the critic → almost no V → no subtraction). ⇒ the up-state is load-bearing AND the cap.
- The up-state floor cannot be trained DOWN at far: the far `vs_place_context` cells stay at init 0.20 (the
  value-train only visits the goal), and the up-state arm is `plastic=False` — so V(far) is a fixed structural
  floor, not a learnable quantity. More value-train trials do NOT help (V(goal) plateaus by trial ~30).

## 3. The GIRK-cap operating-point correction (a finding the op-map could not surface)

The op-map recommended **GIRK cap `gabab_conductance_max=0.0` (uncapped)** for the DIRECTLY-driven critic (sharpest
gap ~19). But with the **afferent-driven (up-state + plastic) critic** firing strongly during the read, the
uncapped GIRK **over-clamps and REBOUNDS** the SNc: diagnosed `pred(near)=348 Hz` (≈5× the no-V reward baseline of
68 Hz!) while `far=0 Hz` — a pathological inversion (the slow GIRK builds during the LEAD, then the reward burst
de-inactivates a rebound). The **finite cap `=1.0`** (the de-risk's own validated row "girk=1.0 gap 1.44") bounds
the GIRK K+ conductance so the firing critic GRADES the SNc cleanly (pred 50 < unpred 70, gap 1.40) WITHOUT
rebound — the documented nav GIRK-cap fix. ⇒ the afferent-driven deployment read REQUIRES the finite GIRK cap; the
uncapped op-map point is only safe for the direct-drive proxy. This is the value-train build's main op-point
refinement over the op-map.

## 4. What this establishes for the TRUE-ONE-BRAIN limbic consolidation

- **The value-train CLOSES the (b) residual the op-map left open:** the merged critic's δ is no longer a
  direct-drive proxy — it is graded from the **TRAINED `vs_place_context→striosome_value` afferent**, learned by
  DA-gated STDP co-resident, with the no-confab moat intact and the dopamine scope=all broadcast NOT perturbing
  the frozen conversational slice. V is genuinely learned on the one brain.
- **The honest negative is the scientific deliverable (the owner's standing bar):** the afferent-driven δ is
  WEAK (~1.3×) vs the direct-drive substrate ceiling (4–19×), and the precise cause is the position-blind
  non-plastic up-state floor — a STRUCTURAL property of the A1+A2 critic architecture (needed to fire the cold
  MSN-D1), not a tuning miss. This maps exactly what the point-neuron substrate can/can't do for the limbic core
  co-resident.
- **The strong, robust co-resident result is the LEARNED V** (critic-grade flip 0.5→~2, 20× weight growth,
  lesion-confirmed δ direction). The graded MAGNITUDE is the bounded follow-on.

### The cheap follow-on to lift δ (if the magnitude is later prioritized)

The cap is the non-plastic up-state floor at far. Options (cheapest first), each a runner/builder kwarg, NO new
mechanism: (a) a **position-SHARPER up-state** (raise `vs_place_drive` sigma selectivity / lower its weight at
far) so the floor is lower at far — but this risks the training bootstrap (the cold MSN must still fire at the
goal); (b) the **`critic_snc_window` sawtooth** (the deployed nav OPENS the GABA route only for a ~1-tau LEAD then
CLOSES it, `g11_bg_runner.py:5893-5901`) — held OPEN throughout here; the windowed read may sharpen the
differential GABA_B; (c) a **learnable up-state** (make the A1 arm plastic so far can be trained down) — a small
builder change but a departure from the validated B.02 innate-up-state design. NONE is on the critical path; the
learned V + lesion-confirmed graded δ direction is the consolidation deliverable.

## 5. Anti-cheats + honest residuals

- **UNTRAINED contrast is load-bearing:** at the init weight 0.20 the afferent (up-state + plastic) fires the
  critic MORE at FAR (grade ~0.5) — the position-blind floor with noise — and the δ is FLAT (~0.96). The
  value-train flips it. `value_input` is held CLOSED during every measurement read (else the critic firing during
  the untrained read would grow the weight via the merged cfg's default `reward_learning_rate=0.01` — a real
  contamination caught + fixed: an earlier run grew 0.20→10 during the "untrained" measure).
- **LESION is via the SYNAPTIC GABA_B (relative test):** zeroing `cp_gabab_synapse_mask` (the
  `striosome_value→snc` route) drops the δ gap below the trained gap toward a ~1.15 non-GABA floor on all 6 seeds
  (trained ~1.3 → lesion [1.14,1.18,1.16,1.11,1.15,1.17]). The gap does NOT collapse to EXACTLY 1.0 because of a
  residual non-GABA floor (the `gpi_{action}→snc` excitatory collaterals + the 10-neuron SNc-pool single-spike
  quantization, ~12.5 Hz/spike), so the mechanistic anti-cheat is RELATIVE (lesion gap < trained gap − 0.08 AND
  ≤ 1.22 = the floor) — **3/3 clean seeds PASS**. The learned δ INCREMENT (trained gap above the floor) is the
  spiking GABA_B current, removed by the lesion; it is not host arithmetic.
- **MOAT preserved with the value-train:** the dopamine `scope=all` plasticity-rate broadcast (re-pointed to the
  nav `snc`) does NOT perturb the frozen conversational weights — `MergedNavConvAgent(co_resident_nav_critic=True)`
  answers `what_does('dog','go')=='north'` and abstains `what_does('river','look') is None`, every seed.
- **SNR honesty:** the SNc pool is 10 neurons → single-spike quantization (~12.5 Hz/spike over a 40-step window).
  The δ read INTERLEAVES near/far over 6 trials with 80-step windows and AVERAGES (the gap is stable ~1.27–1.31
  across trials; the lesion gap ~1.13 vs trained ~1.27 confirms the GABA carries the residual).
- **Op-point:** `critic_only` homeostasis mask (only `striosome_value` masked; `snc` + `reward_us` at vpeak =
  non-saturated, the de-risk recommendation) + SNc tonic 160 pA + GIRK cap **1.0** (the value-train build's
  correction over the op-map's 0.0, §3). `reward_us` spiking US afferent supplies the `r` term (fully-spiking δ).

## 6. Load-bearing file:line

- `research/runners/_merged_navcritic_valuetrain.py` — the value-train + GO-gate + 3 anti-cheats. Ports
  `g11_bg_runner._run_place_value_training` (a nested closure, NOT importable) to the merged `vs_place_context`
  afferent. `--smoke` = the cheap-first CPU build-smoke (build composes + up-state fires the critic at the goal).
- `research/runners/nav_conv_merged_bridge.py:454-455` — the two new builder kwargs
  (`nav_critic_convergent_upstate`, `nav_critic_homeostasis_mask`); `:517-535` — forwarded to
  `build_bg_brain_regions(enable_convergent_upstate=...)` + the `critic_only` post-hoc mask (drop `snc`/`reward_us`
  from the masked set). NO `sim/` edit.
- `research/runners/g11_bg_runner.py:5342-5485` — `_run_place_value_training` (the ported protocol); `:1860-1879`
  — the A1 up-state (`vs_place_drive`) + A2 plastic (`vs_place_context`) arms; `:1888-1895` — the
  `striosome_value→snc` GABA_B route the δ rides.
- `research/findings/2026-06-18-navcritic-valuetrain-opmap-derisk.md` — the op-map this builds on (the direct-drive
  4–19× ceiling; the `critic_only` mask).

## Reproduce
```bash
# cheap-first CPU build-smoke (build composes + up-state fires the critic at the goal at init)
SIM_BACKEND=numpy python -m research.runners._merged_navcritic_valuetrain --smoke --seed 42
# the value-train + GO gate + anti-cheats (GPU); 6 seeds for the δ effect
SIM_BACKEND=cupy python -m research.runners._merged_navcritic_valuetrain --seeds 42,43,44,100,101,102
```
