# gap#4 — the LAYER-2 pedestal/contrast measured, and the CONSTRAINT any contrast fix must clear

Companion to the rung-3d result. Layer 1's pedestal was already on record (~1.6x contrast on a pedestal BTSP
raised ~5x). Layer 2's was **not measured** — and it is the baseline any pedestal-lowering mechanism must be
scored against, so it is measured here BEFORE the research gate reports.

## Instrument validated FIRST (the session's standing lesson)

The extraction is an ad-hoc probe, so it was validated before its numbers were believed:
- extracts exactly **64 synapses per CA1 cell** into L2 (expected `CA1_PER_CELL * L2_N` = 8*8 = 64) ✓
- CSR orientation confirmed **row = presynaptic**: `pos0 -> ca1_0` yields 80 (expected `POS_N*CA1_PER_CELL`),
  and the reversed lookup yields **0** ✓
- `cfg.btsp_w_max` reads 300.0, as configured ✓

Runners: `_gap4_l2_pedestal_probe.py`, `_gap4_l2_extract_validation.py`.

## ⚠️ My first reading of this probe was wrong, and the error is instructive

The raw output showed `post-stage1 = 5.0000` **identically for all five cells**, and a "pedestal rise" of 0.044x
(i.e. weights DROPPING 23x). I initially read the flat 5.0 as a CLAMP artifact and the 0.044x as contradicting
layer 1's 5x rise. Both readings were wrong:

**1. 5.0 is a FIXED POINT, not a clamp.** With `pot = etilde*(w_max - w)` and `dep = lam_dep*gate*(w - w_min)`,
equilibrium sits at `w_eq = w_max*etilde/(lam_dep + etilde)`. At `w_max=300, lam_dep=0.3`:

| etilde | predicted w_eq |
|---|---|
| 0.004 | 3.95 |
| 0.005 | **4.92** |
| 0.006 | 5.88 |

The observed 5.0 is the analytic equilibrium at etilde ~= 0.005. **The rule drives the weights to ITS OWN
equilibrium regardless of initialization** — they start at ~150 and land on 5.0 during stage 1.

**2. Consequence: `l2_w0` is nearly irrelevant.** The initialized weight is erased by stage 1. This retro-explains
why the earlier `btsp_w_max > l2_w0` soft-bound fix mattered less than expected — `w_max` matters because it sets
the EQUILIBRIUM, not because it clips the initial value.

**3. The 0.044x was measured against the wrong baseline** (the arbitrary init 150). The meaningful reference is the
post-stage-1 equilibrium.

## The corrected layer-2 numbers

Per-CA1-cell mean |w| into L2 after stage 2 (seed 200): `[2.86, 8.98, 11.54, 6.65, 3.24]`, equilibrium 5.0.

| quantity | layer 2 | layer 1 (on record) |
|---|---|---|
| contrast (peak/mean) | **1.734x** | ~1.6x |
| pedestal rise (vs equilibrium) | **1.331x** | ~5x |

**The contrast limitation REPRODUCES at layer 2 at essentially the same magnitude** (1.73x vs 1.6x), on a much
smaller pedestal rise. So low contrast is a property of the RULE, not of one layer's operating point.

## THE CONSTRAINT — the number a fix has to clear

Pairing this with rung 3d's measured response margins:

| | value |
|---|---|
| WEIGHT contrast (measured here) | **1.73x** |
| RESPONSE contrast (measured 6/6 in rung 3d) | **1.09-1.21x** |

**The transfer function COMPRESSES 1.73x of weight contrast into ~1.15x of response contrast.** Contrast is LOST
between synapse and spike, not merely absent from the weights.

⇒ **This sets a quantitative bar on any pedestal-lowering mechanism: restoring the 2x RESPONSE selectivity the gate
asks for needs substantially MORE than 2x weight contrast, because ~1.5x of weight contrast is eaten in transfer.**
A mechanism that merely doubles weight contrast will NOT clear the gate. That is a falsifiable, pre-stated bar the
research gate's recommendation can now be scored against — and it was not knowable before this measurement.

## Status

Measurement only; no mechanism proposed here. The research gate on raising contrast is running and will be scored
against the prediction filed at `29c6f897` (pedestal-lowering via bidirectional BTSP, Milstein 2021) AND against
the compression bar established above.
