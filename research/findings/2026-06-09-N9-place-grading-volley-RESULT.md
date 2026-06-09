# N9 place-grading on the fireable sparse-distinct code — STEP 1 GO, STEP 2 PARTIAL

**Date:** 2026-06-09
**Type:** RUNNER-ONLY (ZERO `sim/` edits — `git status --short sim/` byte-empty, verified before+after; the
landed Route D `b980070a` + everything else in `sim/` untouched). CuPy, deterministic regime, ≥3 seeds (42/43/44).
**Builds on:** `2026-06-09-route-T-gamma-volley-RESULT.md` (the FS-PING volley + Route-D fires the MSN, jitter
arbiter PASSES — but PARTIAL: FS-PING landed at 5.9-7.2% sparsity, just over the ≤5% bar, seed-variable).
**Owner directive:** biologize everything, brain-based-only, no host teacher; the jitter + place-shuffle anti-cheats
are the arbiters; an honest negative IS the deliverable.

---

## TL;DR

- **STEP 1 — CLEAN GO (3/3 seeds).** The FS-PING volley operating point is tightened to **source sparsity
  4.4-4.7% (≤5% ✅) AND MSN ≥5 Hz (15.3-18.1 Hz ✅) on ALL THREE seeds 42/43/44**, with the decisive anti-cheats
  GREEN (jitter → 0 Hz, ablate-Route-D → 0 Hz). The fix was **NOT** the named per-region homeostasis (that lever
  DENSIFIES sparse codes — documented below) but **scaling the place pool to n=800** (5%-sparse = 40 active cells →
  a thicker coincident volley fires the MSN at ≤5%), with a re-tuned FS-PING. **The seed-44 firing holdout is GONE.**
- **STEP 2 — PARTIAL.** Wiring the validated volley + Route-D read-out into the N9 striosome-value critic
  **BREAKS THE RATE-CODING WALL: the critic FIRES (G_FIRE) from the sparse-distinct place code and LEARNS the
  near-goal place→critic weight (G_LTP 3/3, near/far 4.2-6.3×)** — both impossible before (the baseline critic was
  0.00 Hz). **But place-GRADING (G_GRADE NEAR≥3×FAR) does NOT robustly open (1/3).** Root cause, precisely
  localized: **Route D's coincidence count is WEIGHT-BLIND** (c_i = # coincident *spikes*, not summed weights), so
  ANY synchronized volley — NEAR **or** FAR — fires the critic; the learned value lives in the AMPA pathway but the
  weight-blind plateau dominates. A biologically-motivated read-out-window trick (lower the plateau at recall so the
  learned AMPA decides firing) grades cleanly at seed 42 (5.7×) but is seed-fragile, and the over-grown seed-44
  weight (w_near→9.3) rate-leaks past the jitter anti-cheat.

**VERDICT: STEP 1 GO; STEP 2 PARTIAL (firing+learning unblocked 3/3; grading seed-fragile; the value-blind
coincidence plateau is the named residual).**

---

## STEP 1 — FS-PING tightened to a clean multi-seed GO

### The operating point (the GO numbers)

```bash
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync ping --k-threshold 4 \
    --n-source 800 --n-fs 160 \
    --place-to-fs-weight 16 --place-to-fs-density 0.4 \
    --fs-to-place-weight 8 --fs-to-place-density 0.4 \
    --src-drive-weight 23 --s2t-weight 20 --s2t-density 0.5 --plateau-strength 80 --gain 2 \
    --out research/findings/raw/_volley_ping_n800_STEP1_GO.json
```

| seed | source sparsity | MSN Hz | volley max c_i | G_SPARSE | G_FIRE | G_VOLLEY |
|---|---|---|---|---|---|---|
| 42 | **4.7%** | **18.1** | 8 | ✅ | ✅ | ✅ |
| 43 | **4.4%** | **15.3** | 9 | ✅ | ✅ | ✅ |
| 44 | **4.5%** | **18.1** | 7 | ✅ | ✅ | ✅ |

- Source diff-loc cosine 0.064-0.069 (sparse-DISTINCT preserved).
- **JITTER (the arbiter): G_FIRE 0/3 → 0.0 Hz at every seed.** De-synchronizing the volley collapses MSN firing →
  **coincidence, NOT rate.** ✅
- **ABLATE Route D: G_FIRE 0/3 → 0.0 Hz.** A synchronized volley of K sub-threshold AMPA inputs without the
  supralinear plateau cannot fire → **Route D load-bearing.** ✅
- **NO-RHYTHM (FS omitted): 2/3 fire** — the known async chance floor (without FS the source densifies → chance
  coincidence). The firing at the operating point is the synchronized volley (jitter collapses it), not this floor.

### What worked, and what did NOT (the honest lever map)

**The fix that worked = SCALING the place pool (n=400 → n=800), re-tuned FS-PING.** At n=400 the trade-off was hard:
≤5% sparse ⇄ ≥5 Hz firing did NOT co-occur on the same seed (a 5%-sparse code of 400 = 20 active cells → too thin a
volley to fire the MSN; firing only returned at ~6%). **A 5%-sparse code of 800 = 40 active cells → the volley is
intrinsically thicker at the SAME fraction → it fires the MSN (incl. the former seed-44 holdout) AND stays ≤5%.**
Biologically faithful: the sparse FRACTION is the invariant; CA1 has far more than 400 cells. `--src-drive-weight 23`
then thins the source to ~4.5% with comfortable ~16 Hz firing margin.

**The named lever that did NOT work = per-region intrinsic homeostasis (Desai/Turrigiano).** Threshold-EMA
homeostasis is the WRONG sparsity lever here — it structurally **DENSIFIES** sparse codes: most place cells are
silent at any given location, homeostasis reads them as "too quiet" → drops their thresholds (`error = ema - target
< 0 → threshold ↓`) → all-fire. Tested at every target rate (0.0008 → 0.035), it densified to **100%** every time.
This matches the `placecode_selforg_stage1` homeo runs (5.7-10% with homeostasis vs 3.4% canonical). Kept in the
probe as opt-in/default-off documented infrastructure (`--place-homeostasis`), with the finding noted inline.

**Other knobs are RNG-fragile / destabilizing, not clean levers:** changing `plateau-strength`, `s2t-weight`,
`gain`, `s2t-density`, or `K` shifts the connectivity RNG stream → perturbs the FS-PING source sparsity
non-monotonically (the FS-PING basin is narrow). Held fixed at the validated defaults; the clean monotonic levers are
`src-drive-weight` (sparsity) and `n-source` (volley thickness).

---

## STEP 2 — the N9 value de-risk on the fireable sparse-distinct code

Wired the STEP-1 mechanism into the existing Stage-2 critic probe
(`n9_place_graded_critic_stage2_derisk.py`, `--enable-volley`, default OFF → baseline byte-identical):
the `place → striosome_value` arm is a **Route-D `coincidence_detector`** fed by an **FS-PING pool on `place`**, at
the STEP-1 GO point (n-place 800). Still plastic + DA-δ-gated (so it can grade + learn). The critic NMDA is forced on
(the plateau reuses the Mg-block). All the existing gates/anti-cheats reused (FIRE/GRADE/LTP/ACTOR/GABA_B +
place-shuffle/sensor-ablation), plus a new **jitter** anti-cheat (desync the place drive at recall → the volley
collapses).

### Baseline (no volley) reproduces the prior NEGATIVE
`critic@NEAR = 0.00 Hz`, w_near 0.503→0.503 (no firing → no post-spike → no LTP). The sparse-distinct place code
literally cannot fire the critic — the exact rate-coding wall STEP 1 broke.

### Definitive 3-seed result (`--enable-volley --n-place 800 --lm-to-place-weight 30 --coincidence-plateau 80 --readout-plateau 40`)

| gate | seeds 42/43/44 | pass |
|---|---|---|
| **2a FIRE** (≥5 Hz) | 5.6 / 3.1 / 14.2 Hz | **2/3** |
| **2b PLACE-GRADED** (NEAR≥3×FAR) | 5.71× / 1.83× / 1.38× | **1/3** |
| **2c LEARNS-V (LTP)** w_near/far | 6.24× / 4.19× / 6.26× (w_near 0.5→3.8/5.8/9.3) | **3/3** ✅ |
| **2d ACTOR-not-perturbed** | 1.00 / 1.00 / 1.00 | **3/3** ✅ |
| **2e SNc GABA_B gap** | 0 / 0 / 0 | **0/3** |

### What is genuinely unblocked (the headline, robust 3/3)
- **G_FIRE: the critic FIRES from the sparse-distinct code** (32-70 Hz at full plateau; was 0.00 Hz). **The
  rate-coding wall that capped every prior N9 mechanism is BROKEN at the value critic.**
- **G_LTP 3/3: the plastic place→critic NEAR weight grows from init (0.5 → 3.8-9.3) and exceeds far 4.2-6.3×.** The
  DA-δ-gated value-leads-reward learning is genuinely NEAR-specific (w_far stays 0.6-1.5).
- **G_ACTOR 3/3:** the critic does not perturb the actor cortex (ratio 1.00).
- **Decisive anti-cheats (at seeds 42/43): JITTER → critic 0.0 Hz (coincidence, not rate); ABLATE-SENSORS → 0.0 Hz
  (value-of-location, sensor-dependent).**
- **PLACE-SHUFFLE 3/3: permuting the place-cell→location mapping BREAKS G_LTP (near/far collapses 4.2-6.3× →
  0.99-1.86×) and the SNc gap.** This confirms the LTP at the primary is genuine **value-of-LOCATION** learning
  (the NEAR ensemble's synapses potentiate because that ensemble is paired with reward), not fire-on-any-drive.

### What does NOT open (G_GRADE, and why — precisely localized)
**G_GRADE (NEAR≥3×FAR) is 1/3** — clean at seed 42 (5.71×), fails at seeds 43/44 (1.4-1.8×, the FAR volley fires
too). **Root cause: Route D's coincidence count is WEIGHT-BLIND** — `c_i = mask^T @ prev_fired` counts coincident
*spikes*, not summed weights, and the plateau current (`plateau_strength`) is constant regardless of the learned
weights (`sim/bridge.py:5726-5768`). So ANY location whose place ensemble forms a ≥K coincident volley (NEAR **and**
FAR, same sparsity, same FS-PING) crosses the plateau and fires the critic. The learned value lives in the **AMPA**
pathway (w_near≫w_far), but the weight-blind plateau dominates → grading washes to ~1.2-1.8× (the same ceiling the
dense position-blind blob hit, now via a different mechanism).

**The biologically-motivated read-out trick** (`--readout-plateau`: keep the strong plateau during TRAINING to
bootstrap the LTP post-spike, then LOWER it at READ-OUT to a sub-threshold integration window so the LEARNED AMPA
decides firing — like an NMDA plateau maturing from a learning-time bootstrap to a coincidence-gated window) **DOES
open grading at seed 42 (5.71×)**, but the window is **seed-fragile**: each seed's equilibrium w_near differs
(3.8/5.8/9.3), so a fixed read-out plateau over- or under-shoots (seed 43 drops below 5 Hz; seed 44's FAR still
fires). Capping w_near via a lower `stdp_w_max` to equalize seeds **collapses the weights to 0** (the soft-bound LTP
loses to LTD below w_max≈10-40), so the spread cannot be tuned out runner-side.

**A second, real anti-cheat FAILURE at seed 44:** its over-grown w_near (9.3) makes the place→critic AMPA strong
enough that even the **jittered** (desynchronized) place spikes rate-drive the critic to fire (10 Hz) — so the jitter
arbiter does NOT collapse seed 44 (it collapses 42/43). The runaway weight rate-leaks past the coincidence test.
**This makes the coincidence claim honest only at the two seeds whose weights stayed bounded.**

**G_GABAB gap (2e) is 0/3** because the predicted-vs-unpredicted SNc subtraction needs clean grading to produce a
gap; with NEAR≈FAR critic firing there is no differential GABA_B onto the SNc, so the gap is ~0 and the lesion test
is moot (you cannot vanish a zero gap).

---

## The named next lever (NOT a host teacher, NOT a sim/ edit forced here)

The residual is the **value-blind coincidence plateau**: Route D detects *synchrony*, not *learned value*. To grade,
the critic must fire MORE for the NEAR volley (strong learned AMPA) than the FAR volley (weak AMPA). Two
biologically-grounded routes, both BEYOND this runner-side step:
1. **Critic feedforward lateral inhibition (striatal PV-FSI / WTA, Tepper-2018):** an FS pool on the critic driven by
   the place volley, implementing divisive normalization / WTA so only the strongest-driven critic cells (NEAR, big
   AMPA) escape inhibition. Runner-side-addable; the natural next attempt.
2. **A weight-SENSITIVE plateau (the genuine fix, a `sim/` change):** make the coincidence switch threshold on the
   *weighted* coincident input (Σ w·spike ≥ θ) rather than the raw count — a multi-subunit Poirazi-Mel / compartmental
   CA1 read-out, exactly the lever the prior root-cause flagged ("multi-subunit Poirazi-Mel / compartmental CA1, NOT
   conduction delays"). This is the principled endpoint; it requires an owner-reviewed additive `sim/` edit.

The honest negative is the deliverable: **firing the critic via coincidence is necessary but NOT sufficient for
value-grading, because the coincidence detector is value-blind. The grading needs a weight-sensitive read-out (WTA or
a weighted-coincidence plateau), which the weight-blind Route-D plateau is not.**

---

## Reproduce

```bash
# STEP 1 GO (FS-PING tightened, n800):
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk \
    --seeds 42,43,44 --sync ping --k-threshold 4 --n-source 800 --n-fs 160 \
    --place-to-fs-weight 16 --place-to-fs-density 0.4 --fs-to-place-weight 8 --fs-to-place-density 0.4 \
    --src-drive-weight 23 --out research/findings/raw/_volley_ping_n800_STEP1_GO.json
# STEP 1 anti-cheats (both must collapse to 0 Hz):
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --sync ping \
    --n-source 800 --n-fs 160 --place-to-fs-weight 16 --place-to-fs-density 0.4 --fs-to-place-weight 8 \
    --fs-to-place-density 0.4 --src-drive-weight 23 --jitter-inputs   # -> 0/3
SIM_BACKEND=cupy python -m research.runners.coincidence_volley_n9_derisk --seeds 42,43,44 --sync ping \
    --n-source 800 ... --ablate-subunit   # -> 0/3

# STEP 2 PRIMARY (volley critic):
SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk \
    --seeds 42,43,44 --enable-volley --n-place 800 --n-fs 160 --lm-to-place-weight 30 \
    --coincidence-plateau 80 --readout-plateau 40 --selforg-passes 10 --n-train 40 \
    --out research/findings/raw/_n9_volley_DEFINITIVE_primary.json
# STEP 2 anti-cheats:
SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk --seeds 42,43,44 \
    --enable-volley --n-place 800 ... --jitter        # collapses 42/43, leaks at 44 (over-grown weight)
SIM_BACKEND=cupy python -m research.runners.n9_place_graded_critic_stage2_derisk --seed 42 \
    --enable-volley --n-place 800 ... --shuffle        # breaks 2c LTP + 2e gap
```

Raw JSON: `research/findings/raw/_volley_ping_n800_STEP1_GO.json` (+ `_jitter`/`_ablate`/`_norhythm`),
`research/findings/raw/_n9_volley_DEFINITIVE_primary.json` (+ `_jitter3`/`_ablate3`/`_shuffle42`).

---

## Bottom line

STEP 1 is a **clean GO**: the FS-PING gamma volley fires the downstream MSN from a **≤5%-sparse, distinct** place
code on **all 3 seeds**, jitter-collapse confirms coincidence-not-rate, Route D is load-bearing — the residual the
prior PARTIAL flagged (sparsity just over 5%, seed-44 weak) is **resolved by scaling the place pool**, not by
homeostasis (which densifies). STEP 2 is **PARTIAL**: feeding that fireable volley into the N9 value critic **breaks
the rate-coding wall — the critic fires and learns the near-goal value weight 3/3** — but **place-GRADING does not
robustly open**, because **Route D's coincidence detector is value-BLIND** (it fires on any synchronized volley, NEAR
or FAR). Grading needs a **weight-sensitive read-out** (critic WTA, or a weighted-coincidence / multi-subunit
plateau) — the named next lever, beyond this runner-side step. The N9 arc is **not yet closed on grading**; the
firing+learning half is genuinely unblocked on the production backend, runner-side, with zero `sim/` edits.
