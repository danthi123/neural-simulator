# gap#4 RUNG 3d — PRE-REGISTRATION (filed BEFORE the run, on seeds never touched by this arc)

**Filed: 2026-07-20. No rung-3d result exists at the time of writing. This file is committed BEFORE the runner
is launched, precisely so the prediction cannot be adjusted to the outcome.**

## Why this exists

Rung 3c established, with three agreeing controls, that L2 **genuinely learns** a read of the CA1 population
(plasticity-lesion collapses the response 24x; plateau-lesion collapses it identically; moving the plateau moves
the read 7->7 and 11->11). But its peak sits at offset **+0** from the plateau bin, while the pre-registered gate
demanded a **-5 bin** backward-shifted peak. The gate therefore returned NO-GO — and is separately known to be
invalid, since the do-nothing controls PASS it.

I explicitly did NOT re-centre that gate on the data that revealed the offset. The project record already warns
that I have mis-centred this exact metric twice, and a third re-centring with full knowledge of where the peaks
landed would be goalpost-moving. **The legitimate move is to state the prediction in advance and test it on seeds
that have never been run.** That is this file.

## The mechanism being predicted (stated as a mechanism, so it can fail)

At **layer 1** the input is position. A plateau at bin *b* credits inputs whose eligibility is still decaying from
EARLIER bins, so the learned field shifts BACKWARD (rung 1 measured this; the eligibility-tau ablation showed it is
load-bearing).

At **layer 2** the input is the CA1 population — whose fields have **already** shifted. L2 plateaued at bin *b*
therefore sees the CA1 cell whose (already-shifted) field peaks at *b* maximally active **at that instant**, so the
strongest eligibility is the CONCURRENT one.

⇒ **The backward shift happens ONCE, upstream. It does not compound across layers.**

## PRE-REGISTERED PREDICTIONS (all four must hold)

1. **P1 — offset zero:** `l2_peak - plateau_bin == 0 +/- 1` on >= 5 of 6 fresh seeds.
2. **P2 — 1:1 tracking:** moving the plateau to another cell's field moves `l2_peak` by the SAME number of bins
   (|delta_l2 - delta_plateau| <= 1) on >= 5 of 6.
3. **P3 — learning is load-bearing:** freezing L2 plasticity (eta=0) reduces `r_tgt` by >= 5x AND gives dw == 0,
   on 6 of 6.
4. **P4 — the plateau is load-bearing:** removing the L2 plateau does the same, on 6 of 6.

**FALSIFIED if:** P1 fails (a non-zero offset means the shift DOES compound, and my post-hoc story is wrong), or
P3/P4 fail (the read is structural, not learned — which would retract rung 3c's positive mechanism claim).

## Seeds — FRESH, never used in this arc

Rung 1/2/3/3b/3c used 42, 43, 44, 100, 101, 102. **All six are contaminated for this test.**
Rung 3d uses **200, 201, 202, 203, 204, 205** — never run against any gap#4 rung.

## Honest scope, stated in advance

- This tests a MECHANISM (where the learned read lands and what it depends on). It does NOT resurrect the original
  rung-3 gate, which asked a different question and stands as a filed NO-GO.
- Confirming P1-P4 would establish: **one-shot local credit composes across a layer, with the backward window
  applied once at the input layer.** It would NOT establish that a 2-layer stack SOLVES anything a 1-layer cannot
  — that is a separate, harder question and is not claimed here.
- The map-validity assertion (`map_ok`) must pass; any seed where stage 1 fails to form 
  distinct fields is EXCLUDED and reported as excluded, not silently dropped.
