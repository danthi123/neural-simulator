# gap#4 RUNG 7 — RESULT: NO-GO, but the test is INVALID. One global `w_max` cannot serve two pathways.

Pre-registered at `38259045`. Verdict NO-GO (P0 `map_ok = 0` on 6/6, P1-P2 unmeasurable, **P3 PASSES** — the
`k_dep = 0` control reproduces 1.213/2.609 exactly on every seed).

## I asserted two causes; measurement refuted both

| my hypothesis | prediction | measured | verdict |
|---|---|---|---|
| stage-1 **saturation** (k=0.02 too large at w_max=300) | mass at `w_max` | **0 / 8320 at w_max, 0 / 8320 at w_min** | **REFUTED** |
| the fixed point **homogenized** the map | spread shrinks | **CV rose 0.150 -> 1.091** (7x) | **REFUTED** |

Two confident diagnoses, two refutations, from probes I wrote to check myself. The second is the more instructive:
I reasoned that a fixed-point rule would compress the weight distribution and erase field structure. It did the
opposite.

## What the data actually shows

`pos -> ca1` weights: **mean 0.6002 -> 55.2791 (92x), range 0.258..0.929 -> 5.000..149.987.**

`btsp_w_max` is a **single global bound**, set to `max(5, 2*l2_w0) = 300` because the layer-2 pathway operates at
weight ~150. But the Milstein fixed point scales with that bound (`pot = k*q*(w_max - w)`), so a `w_max` chosen for
the layer-2 pathway **drags the layer-1 pathway up toward it** — from a natural scale of 0.6 to a terminal spread of
5-150. With layer-1 weights inflated ~92x, every position drives every CA1 cell and no distinct fields form:
`map_ok = 0`, CA1 effectively uninformative, so stage-2 `dw = 0` follows.

**The two pathways' natural scales differ by 250x (150 vs 0.6).**

## This is a KNOWN defect class in this codebase, and I walked into it again

Commit `a5a5e341` recorded exactly this for the depression threshold: *"one global theta genuinely cannot serve
both"*, with a measured layer1/layer2 eligibility ratio of **27.4x**. `w_max` has the identical problem at **250x**,
and I did not check it before pre-registering — despite having written the theta version of this finding myself.

## Status: the mechanism is UNTESTED, not refuted

**I am not recording rung 7 as evidence against weight-dependent BTSP.** The rule never operated in a valid regime:
its fixed point was set by a bound belonging to a different pathway. What the run does establish:

- **P3 passes 6/6** — the control reproduces the baseline exactly, so the harness is sound and any future
  comparison is trustworthy.
- **PF-5 still stands**: the fixed point IS real on deployed traces (starts 0.3 and 2.0 converge to 1.31/1.36,
  final maps r = +0.997, zero floor pinning) — measured in the `w_max = 5` config where the bound matches the
  pathway.

## The cap, and why it does NOT bind here

The pre-registration said: *"one parameterization... if P1 fails I do NOT re-tune k or the thresholds."* **P1 did
not fail — it was never measured**, because P0 failed for a configuration reason upstream of the rule. Fixing a
per-pathway bound is not re-tuning a parameter to chase a result; it is making the rule operate on the pathway it
is meant to act on. I am stating that explicitly rather than relying on it silently, because "the cap doesn't
apply here" is exactly the reasoning a motivated reading would produce.

**Required before any re-run:** per-pathway `w_max` (mirroring the per-synapse `cp_btsp_theta` that already exists
for the same reason), and a pre-flight confirming layer-1 weights stay on their own scale. Then rung 7 gets a fresh
pre-registration and untouched seeds.
