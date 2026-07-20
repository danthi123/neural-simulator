# gap#4 PF-5 — the FIRST pre-flight to PASS: the weight-dependent fixed point is real on deployed traces

Seven mechanisms were built; every pre-flight so far has refuted its candidate. This one passes, and it passes on
the property the literature says is the actual mechanism.

## The claim under test

Milstein 2021 (measured): "weak inputs potentiate, and strong inputs depress", producing a **stable target shape**
independent of the starting state — `dVm vs initial Vm: r = -0.91`, **`final Vm vs initial Vm: r = 0.04`**.

## Result — measured on deployed traces, from two very different starting points

| start | initial mean | final mean | at floor |
|---|---|---|---|
| `w0 = 0.3` | 0.3007 | **1.3071** | **0 / 6400** |
| `w0 = 2.0` | 2.0049 | **1.3648** | **0 / 6400** |

**Both converge to ~1.3 from opposite directions.** A 6.7x difference in starting weight collapses to a 4%
difference in the final mean.

| signature | Milstein | measured here |
|---|---|---|
| `corr(initial, dW)` | -0.91 | -0.044 / **-0.301** (right sign, weaker) |
| **`corr(initial, final)`** | **0.04** | **+0.002 / +0.012** |
| **corr(final map from w0=0.3, final map from w0=2.0)** | — | **+0.997** |

**The final map is essentially independent of the initial map.** That is the fixed point, present in the deployed
system rather than in an idealized model.

**And the structural immunities hold as designed:** `at_floor = 0/6400` in both runs. The Miller-MacKay pathology
(51% of weights pinned at `w_min`, surviving positive increments dragging the mean up) cannot arise here because
depression is multiplicative in `(w - w_min)` and vanishes at the floor.

## Unit check passed FIRST — the check the DoG skipped

Before any of the above: the **per-synapse** normalized overlap (n = **8,555,600** deployed samples) populates all
three of Milstein's zones — depress-only 68.9%, mixed 9.6%, potentiate 21.5% — so the **published thresholds are
usable without rescaling**. The DoG died precisely because its condition was derived for a trace pair the
implementation never generates; this one is verified against the traces the implementation actually produces.

*(That check also corrected me: an earlier probe of mine appeared to contradict the research gate's distributional
claim. My probe computed `et.max()*IS.max()` — a per-timestep peak scalar — instead of the per-synapse overlap. The
gate was right; I have corrected that record.)*

## ⚠️ WHAT THIS DOES **NOT** SHOW

**It does not show that adjacent-band contrast improves.** PF-5 validates the mechanism's *own* claimed property —
a start-independent fixed point — and nothing more. Whether that fixed point produces a field with better
neighbour-contrast is a separate question, and it is exactly the question seven previous mechanisms failed.

I am recording this distinction before running the contrast test, because "the first pre-flight to pass" is the
precise moment where a validated sub-property gets quietly reported as a solved problem.

## Next

Pre-register the contrast test (adjacent >= 1.60x, far retained, stage 1 forms, `k_dep = 0` control reproduces the
baseline), on untouched seeds, with the same cap discipline as rungs 4-6. Per the literature, also randomize field
centres — Poisson/uniform rather than evenly spaced — since even spacing has no empirical basis and may generate
its own artifacts.
