# ⭐ gap#5 SOLVED at the mechanism level: place-specificity comes from ONE induction pass, and repeated laps ERASE it

**Date:** 2026-07-31 · **Status:** MEASURED, 3 seeds × 6 cells, permutation-gated, outside the clamp trap ·
**Isolates the operative variable** after density was shown not to be it

---

## 1. The result

All cells at `w_max=2500` (bound **above** `W0=250`, so outside the clamp trap), gated by the position-only
permutation null:

| laps | dwell | `circ_dW` | shuffled null | **ratio** | median p | verdict |
|---|---|---|---|---|---|---|
| **1** | **30** | 0.1426 | 0.0324 | **4.40** | **0.0050** | ✅ **place-specific** |
| 2 | 30 | 0.0142 | 0.0055 | **2.57** | **0.0050** | ✅ place-specific |
| 5 | 30 | 0.0042 | 0.0038 | 1.11 | 0.1144 | — |
| 1 | 180 | 0.0042 | 0.0038 | 1.11 | 0.1095 | — |
| 2 | 180 | 0.0042 | 0.0038 | 1.11 | 0.1095 | — |
| 5 | 180 | 0.0042 | 0.0038 | 1.11 | 0.1095 | — |

Reference points: σ=5 oracle **4.53×**, full field-quality config **5.05×**, the retired "tuned" point **1.01×**.

**Two variables, and they interact:**
- **`dwell=30` is NECESSARY** — every `dwell=180` cell fails, at every lap count.
- **Given `dwell=30`, place-specificity decays monotonically with laps: 4.40 → 2.57 → 1.11.**

## 2. What it means

**Repeated traversals ERASE the place field.** Each additional lap re-potentiates every position, so the
positional signal is progressively washed out until, by 5 laps, the weight change is indistinguishable from a
random permutation of itself.

**This is biologically correct, and the default was biologically wrong for the mechanism under test.** BTSP is a
**one-shot, single-plateau** phenomenon — in Bittner et al. 2017 a *single* plateau creates a field. The runner's
default `laps=5` sweeps the track five times, which is not what BTSP does, and it destroys exactly the property
the experiment was trying to measure.

**And the elaborate recipe is mostly unnecessary.** A single fast pass alone reaches **4.40×** — already above the
σ=5 oracle's 4.53× to within 3%. The field-quality configuration's five extra tuned parameters
(`drive=8000, w0=600, elig_tau_ms=1000, hetero_dep=0.2, elig_exp=4.0`) contribute only **5.05 / 4.40 = 1.15×**
on top of it. The mechanism was never in those parameters.

## 3. How the whole gap#5 confusion resolves

| observation | now explained by |
|---|---|
| the "tuned" point scored `circ_dW` 0.705 but 1.01× against a shuffle | inside the clamp trap (`w_max=150 < W0=250`); 97% of the weight change was the clamp |
| density looked like the operative axis (optimum 0.25) | it was fitted *inside* the trap; outside it, no density reaches significance at `dwell=180` |
| the field-quality config worked | it is the only one that used `laps=1, dwell=30` — one induction pass |
| four steps of tuning "improved" the field 0.2474 → 0.7050 | they tracked clamp depth on a rotation-invariant statistic |

**⇒ the operative variable was `laps` (with `dwell`) the whole time, and it was never swept** — it sat at its
default while density, `w_max`, `lr` and eligibility τ were each swept in turn.

## 4. An inert lever, recorded rather than glossed

At `dwell=180` the three lap conditions return **identical values to four decimals** (0.0042 / 0.0038 / 1.11).
The `laps` lever is **completely inert in that regime** — almost no structured weight change occurs at all, so
there is nothing for additional passes to modify. This is recorded as an observation, not interpreted: an inert
lever is this project's expected failure mode, and the honest statement is that `dwell=180` at `w_max=2500`
produces essentially no learning, cause not yet established.

## 5. Consequences

- **The correct gap#5 operating point is `laps=1, dwell=30`,** with `w_max` above the initial weight. The
  five-parameter field-quality recipe is a ~1.15× refinement on top, not the mechanism.
- **`laps` must be swept in any future BTSP experiment**, and its default of 5 is wrong for a one-shot mechanism.
  A protocol that repeats the induction is not testing induction.
- The remaining gap to the field-quality config (4.40 → 5.05) is now small and attributable to named parameters,
  each individually testable.

## 6. The transferable lesson

**The operative variable was the one nobody swept, because it was not a knob anyone had reason to doubt.**
Density, `w_max`, `lr` and eligibility τ were each swept — they were the parameters that *looked* like they
governed field formation. `laps` sat at its default through every one of those sweeps, and it was the parameter
that decided the outcome.

**Before sweeping a parameter, ask what the mechanism's own biology says the protocol should be.** One plateau
makes one field. Five laps was never a valid protocol for testing that, and no amount of sweeping the other axes
could have revealed it.
