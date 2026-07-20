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
