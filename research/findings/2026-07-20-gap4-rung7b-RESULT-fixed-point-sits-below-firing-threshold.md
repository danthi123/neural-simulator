# gap#4 RUNG 7b — RESULT: NO-GO. The rule's fixed point sits BELOW the substrate's firing threshold.

Pre-registered at `093f3f7f` on a repaired instrument (per-pathway `w_max`, PF-6 verified). Seeds 1600-1605.
**P3 PASSES 6/6** (control reproduces 1.213 / 2.609 exactly). **P0 fails 6/6**, so P1/P2 are unmeasurable.

## Measured cause — a genuine property of the rule, not another instrument defect

| | before | after |
|---|---|---|
| layer-1 `pos->ca1` | 0.599 | **1.223** (0 / 8000 at floor) |
| layer-2 `ca1->l2` | 151.239 | **151.239 — unchanged** |
| **CA1 firing** | — | **0.0000 — SILENT** |

**With `k_pot = k_dep` and both sigmoids saturated, the fixed point is `w* = w_max/2`.** For layer 1 that is
**2.5**, while baseline BTSP drives the same weights toward `w_max = 5` — where CA1 fires. **The rule's equilibrium
sits at half the value the baseline reaches, and that is below the firing threshold.** CA1 goes silent, no map
forms, and neighbour-contrast cannot be measured.

**And that explains the -37,000 dw**, which I would otherwise have mis-read as runaway depression. At *zero* overlap
the depression sigmoid is not zero:

- `q_dep(0) = sigmoid(20*(0 - 0.09)) = 0.1637`
- `q_pot(0) = sigmoid(20*(0 - 0.24)) = 0.0082`
- ratio **20x, depression-dominant at silence**

So once CA1 falls silent, the externally-delivered L2 plateau keeps `IS > 0` while the overlap collapses to ~0, and
layer-2 weights are depressed from 150 toward 0. **The large negative dw is a downstream consequence of the silent
map, not an independent failure.**

## The cap binds, and I am honoring it

The pre-registration said: *"One run... If P1 fails I do not touch `k`, the thresholds, or the bounds."*

I can see the adjustment that would likely fix this — raise `k_pot` relative to `k_dep` so `w*` lands above
threshold, since `w* = w_max * k_pot/(k_pot + k_dep)`. **I am not making it.** That is precisely the knob the cap
exists to protect, and "the fixed point just needs to sit a bit higher" is exactly how a mechanism gets fitted to
its task. Recording the value and declining to use it.

## What is established, and what is NOT

**Established:** with published thresholds and no free parameter, weight-dependent bidirectional BTSP produces an
equilibrium **below the firing threshold of this substrate configuration**. That is a real, specific, reproducible
(6/6) interaction between the rule and the substrate.

**NOT established:** anything about the rule's neighbour-contrast properties. P1 has now been unmeasurable in both
rung 7 and 7b, for two different reasons (one my configuration error, one this genuine threshold interaction).
**The mechanism's contrast behaviour remains untested after two attempts.**

**Still standing:** PF-5's fixed point on deployed traces (starts 0.3 and 2.0 -> 1.31/1.36, final maps r = +0.997,
zero floor pinning) — measured in the `w_max = 5` config where the equilibrium *was* reachable.

## The successor experiment, already named and already built

The literature is explicit that this task's evenly-spaced field layout has **no empirical basis** — real CA1
spacing is Poisson with a modal gap of zero (Rich 2014, 0/61 cells deviate). `--poisson-cells` is implemented and
verified to produce varying gaps across seeds. **That is the next experiment**, and notably it changes the TASK
rather than the rule — which is where the evidence has been pointing since the literature reframe.
