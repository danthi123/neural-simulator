# gap#4 — the "soma-coupling PIPELINE-VALIDATED (`--soma-g 8`)" finding does NOT reproduce; the real coupling threshold is ~100

**Date:** 2026-07-20 · **Status:** CORRECTION (silent-failure discipline: control-first reproduction of a banked
claim, before scaling on it). The 2026-07-19 finding
(`2026-07-19-gap4-soma-coupling-flips-BOUNDARY-to-PIPELINE-VALIDATED-...`) claimed `--soma-g 8.0` flips the microcircuit
BDSP run to PIPELINE-VALIDATED with the apical coupling to bursts ("B rises") and directed hidden credit **415 ≫ 198**.
Board line 59 rests on it ("pipeline-validated, NEXT = scale sweep"). **It does not reproduce.**

## What reproduces vs what doesn't

Ran the finding's EXACT validated smoke (`--task dense --hidden 12 --epochs 3 --microcircuit --graded-credit
--apical-bistable --apical-self-regen 2 --apical-kir-g 3 --differential-readout --soma-g 8.0`, seed 42) and a soma-g
sweep. Reading the coupling code (`bridge.py:6583-6589`): the coupling adds `soma_g · v_apical_scale · depol` pA to the
soma, with **`v_apical_scale = 0.05`** default → at `--soma-g 8` the coupling is only **~6-40 pA**, negligible against
the **700 pA** somatic drive (`_d1_..._derisk.py:539`). So B cannot rise. Confirmed by a soma-g sweep (h12/e3/dense):

| soma-g | B_apical | `apical_couples_to_bursts` | dw_in2hid (bdsp / lesion) | held-out |
|---|---|---|---|---|
| **8** (finding's value) | 0.000 | **False** | 0.484 / 0.045 | 0.731 (floor) |
| 100 | 0.097 | True | 7.6 / 0.045 | 0.731 |
| 400 | 0.303 | True | 54.9 / 0.045 | 0.731 |
| 1500 | 0.496 | True | 73.4 / 0.045 | 0.731 |

- **`--soma-g 8` DOES NOT COUPLE** — B stays 0, `apical_couples_to_bursts=False`. The finding's headline value is wrong
  (its "B rises / 415≫198" is unreproducible; the committed raw json for that config also records
  `B_rises=False, apical_couples_to_bursts=False`, INCONCLUSIVE). The real coupling threshold is **~soma-g 100**.
- **The MECHANISM is SALVAGED at the corrected soma-g:** at soma-g ≥ 100 the apical genuinely couples (B rises), and
  directed credit reaches the hidden layer (dw_in2hid 7.6-73 ≫ lesion 0.045) while the P0 moat holds (lesion dw 0.045).
  So "directed hidden-layer credit + moat" is REAL — just at soma-g ~100-400, not 8.
- **BUT accuracy is still the open question:** held-out stays at the 0.731 floor for ALL soma-g at this smoke scale
  (h12/e3). Directed-credit-reaches-hidden ≠ classification accuracy. Whether accuracy clears the bar at proper scale
  (wider hidden + more epochs) with the CORRECTED soma-g is the actual test — see below (it does NOT clear the bar).

## Corrected scale-up (h48/e24, soma-g 200 — above the ~100 threshold)

| task | oracle | floor | BDSP held | LESION | wrong_sign | couples? / B_apical | dw_in2hid (bdsp / lesion) |
|---|---|---|---|---|---|---|---|
| cleanxor | 0.989 | 0.561 | **0.564** | 0.439 | 0.439 | True / 0.202 | 938.9 / 0.21 |
| dense | 0.764 | 0.731 | 0.731 | 0.731 | 0.728 | True / 0.202 | 840.7 / 0.19 |

**VERDICT: directed credit is REAL but does NOT produce ACCURACY at this scale.** On cleanxor (clean-margin, oracle
0.989): BDSP held-out **0.564 > LESION 0.439** (by 0.125), wrong_sign 0.439 below floor — the credit is sign-informative
+ directional (dw 938 ≫ lesion 0.21). BUT BDSP 0.564 ≈ the linear floor 0.561 — it does NOT climb toward the oracle
0.989 and does NOT clear the 0.75 bar. On dense (poor discriminator, oracle 0.764) everything sits at the 0.731 floor.
⇒ **the finding's "pipeline-validated → just needs the scale sweep" is over-optimistic: at the CORRECTED soma-g, the
h48/e24 scale-up still does not produce accuracy.** Directed-credit-reaches-hidden (the dw signal the finding gated on)
≠ classification accuracy. The apical credit prevents the harmful below-floor learning the lesion falls into (0.439),
but does not build useful hidden features.

**OPEN (the real gap#4-deep-credit-to-ACCURACY question — a METHOD verdict, not a wall):** whether accuracy is
under-tuned (more epochs / forward-drive tuning so the hidden layer forms useful features / wider hidden) or a genuine
boundary of the coarse spiking burst-credit is the NEXT test — an epochs×drive×width sweep on cleanxor at soma-g ~200,
6-seed, vs the 0.75 bar + the (BDSP > lesion, wrong_sign < floor) sign-informative gate. If tuning does not move it, a
research gate for a stronger credit-direction signal (the microcircuit clean-error is already on; next classes: a
learned-feedback / Kolen-Pollack credit direction, or a richer read-out).

## Tuning sweep result (cleanxor, h48, soma-g 200, seed 42) — the accuracy floor is ROBUST across every credit lever

Ran the named epochs×drive×feedback sweep. BDSP held-out (floor 0.561, oracle 0.989):

| lever | BDSP | LESION | wrong_sign | note |
|---|---|---|---|---|
| e24 (base) | 0.564 | 0.439 | 0.439 | at floor |
| e60 | 0.561 | 0.439 | 0.439 | at floor |
| e120 | **0.439** | 0.439 | 0.561 | DEGRADES — collapses to lesion; wrong-sign inverts above |
| apical-hid-gain 500 (vs 190) | 0.564 | 0.439 | — | at floor |
| hidden-bias 220 (vs 520) | 0.561 | 0.439 | — | at floor |
| **KP learned feedback** (vs fixed-random) | **0.564** | 0.439 | — | **no help** |

**VERDICT: the on-bridge spiking BDSP burst-credit does NOT build accuracy above the linear floor on cleanxor, robust
to epochs (24/60/120), credit gain, hidden bias, AND the learned Kolen-Pollack feedback direction.** The credit is
*sign-informative* (BDSP > lesion/wrong-sign at short training; it lifts the net from below-chance anti-learning 0.439
to ~chance 0.564) but does not extract the hidden structure the numpy-backprop oracle finds (0.989). More epochs (120)
DEGRADE it (BDSP → 0.439, wrong-sign inverts above), so it is not under-training. **KP learned feedback not helping
rules out the feedback DIRECTION (fixed-random FA) as the fix within this family.** Whether the residual boundary is
the coarse burst-credit QUALITY or the spiking FORWARD/READOUT (the net reaches ~0.56 while the numpy-backprop oracle
reaches 0.989 on the SAME architecture) is NOT yet distinguished — that is the decisive open control below, and I do
NOT claim "credit boundary" until it is run.

**OPEN (honest scope + the decisive next control):** (1) single-seed characterization — a 2-3 seed confirmation of the
floor firms the boundary. (2) **The credit-vs-forward diagnosis is not yet run:** the on-bridge net reaches only ~0.56
while the oracle (numpy backprop, SAME in→hid→out architecture) reaches 0.989 — so before calling this a CREDIT
boundary, install the oracle's trained weights into the on-bridge spiking net and read its accuracy; if the spiking
forward+differential-readout can't express ~0.99 even with perfect weights, the boundary is the spiking
FORWARD/READOUT, not the credit. (3) THE LAW: raw burst-credit + FA/KP is exhausted → a research gate for a
fundamentally different credit signal (not more of this family).

## Credit-vs-forward diagnostic (the decisive control) — the accuracy floor is a forward OPERATING-POINT issue, not a pure credit boundary

Built a reservoir probe (`_gap4_credit_vs_forward_probe.py`): freeze input→hidden at random (NO BDSP), read each
sample's hidden firing (baseline-subtracted), train a numpy readout on those random-hidden features. Also swept the
operating point (hidden_bias 20-520 × fwd_wmean 6-80). ALL configs (seed 42, cleanxor, oracle 0.989, input-linear
floor 0.510):

| hidden_bias | fwd_wmean | hid-feat-active | reservoir readout | (input-linear 0.510) |
|---|---|---|---|---|
| 520 | 6 | 0.00 | 0.445 | at chance |
| 100 | 6 | 0.00 | 0.445 | at chance |
| 20 | 6 | 0.00 | 0.445 | at chance |
| 100 | 40 | 0.00 | 0.445 | at chance |
| 20 | 40 | 0.00 | 0.445 | at chance |
| 20 | 80 | 0.00 | 0.460 | at chance |

`hid-feat-active=0.00` = the hidden firing is IDENTICAL across inputs at every tested operating point. Direct
verification (build net, drive 3 inputs, read raw firing): the **HIDDEN layer NEVER FIRES** (rate 0.0, 0/48) at hb=20
even with fwd_wmean=80, while the **INPUT layer DOES fire and differ** across samples (rate 0.05, ‖i1−i2‖=2.0). So the
input→hidden synaptic drive is negligible vs the bias/threshold — the hidden is silent (low bias) or bias-saturated
(high bias), never input-selective; the sparse input firing (0.05) does not modulate it.

**⇒ REFRAME (corrects my own premature "bias-saturation" read — the probe bug-check caught it): the ~0.5 accuracy
floor is (at least partly) a forward OPERATING-POINT issue, NOT a pure credit boundary and NOT a
forward-representation limit.** The on-bridge net's forward was never configured for a working computation — the
runner's operating point is tuned for the mechanism SMOKE (does dw move under credit?), which does NOT require an
input-selective hidden layer. No credit rule can build accuracy from a hidden layer that carries zero input signal.

**HONEST BOUND (no overclaim):** I have NOT found an operating point where the hidden IS input-selective — only shown
the tested ones (bias 20-520, drive 6-80, settle 10) carry no signal. Whether an input-selective hidden regime EXISTS
(aggressive drive/bias/settle/input-rate search — the input firing 0.05 is very sparse; longer settle + higher in_hi +
much stronger fwd weights) or is ruled out is the decisive NEXT step, BEFORE re-testing credit. Also single-seed (the
floor itself is 3-seed: BDSP 0.564/0.531/0.489 ≈ lesion, oracle 0.97-0.99). ⇒ the gap#4 deep-credit-to-accuracy
question is not yet answerable — the substrate must first be shown to carry input signal through the hidden layer.

## Method lesson

The banked "pipeline-validated" result was gated on a dw ratio while its OWN coupling diagnostic said DECOUPLED, and its
headline soma-g (8) is below the coupling threshold (~100) — a claim that did not survive a control-first reproduction.
This is the silent-failure class again (a verdict that can pass without its key control): the VERDICT printed
PIPELINE-VALIDATED off `dw_in2hid ≫ lesion` while `apical_couples_to_bursts=False` sat in the same record. **The gate
should require `apical_couples_to_bursts=True` (B rises), not just a dw ratio.** Corrected next: the accuracy sweep runs
at soma-g ~200 (coupled), not 8.

## Artifacts

Runner `_d1_onbridge_learn_to_accuracy_derisk.py`; sweep `_d1_somag_sweep_sg{8,100,400,1500}_s42.json`; reproduction
`_d1_repro_validated_smoke_h12e3_s42.json`; coupling code `sim/bridge.py:6583-6589` (`v_apical_scale=0.05`).
