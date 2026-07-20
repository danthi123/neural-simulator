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

## Follow-up: BDSP AT the input-selective operating point — under-tuned/decoupled, needs JOINT tuning

Found an input-selective forward operating point (settle=60, fwd_wmean=500, hidden_bias=300, in_hi=2000): the hidden
now fires input-differentially (‖h1−h2‖≈12.8, hid-active 0.99) and a random-hidden reservoir readout reaches **0.67**
(> input-linear floor 0.510; < oracle 0.989 as expected for random features). So the forward IS expressive here.

Ran BDSP at this operating point (soma-g 200, e12, cleanxor, seed 42): **BDSP 0.400 ≈ LESION 0.406 ≈ wrong 0.406**, all
BELOW the reservoir baseline 0.67 AND the linear floor 0.567 — AND **`apical_couples_to_bursts=False`** (B_apical 0.310
but no rise vs rest). So at in_hi=2000's stronger-drive burst regime, soma-g 200 no longer couples (same decoupled
signature as soma-g 8 at the weak drive) and BDSP≈lesion (the decoupled null); the apical gains/lr were not rescaled
for the 3× stronger forward drive. **This is an under-tuned, DECOUPLED run — NOT a credit verdict.** The credit
degrading the reservoir signal (0.67→0.40) is consistent with an un-coupled, mis-scaled update, not "credit is harmful."

**⇒ the gap#4 deep-credit-to-accuracy test requires JOINT tuning — a multi-dimensional search, the next step:**
(1) hold the input-selective forward (settle 60 / fw ~300-500 / hb ~150-300 / in_hi ~2000);
(2) re-find the soma-g that COUPLES at THIS burst regime (sweep soma-g upward until `apical_couples_to_bursts=True`
    with B rising above the now-higher B_rest — likely soma-g ≫ 200);
(3) rescale the credit magnitude (apical_out/hid_gain, bdsp_lr) to the new drive;
(4) THEN read whether BDSP climbs the reservoir baseline (0.67) toward the oracle (0.989), 6-seed, vs the
    (BDSP > reservoir-baseline, wrong_sign < baseline) sign-informative gate. Only a JOINTLY-tuned, COUPLED run answers
    the credit question; every single-lever run so far confounds forward operating point, coupling, and credit magnitude.

## The drive-vs-coupling TENSION (why the joint tuning is hard) — a structural finding

Swept (soma-g × output_bias) at the input-selective forward (fw=500, hb=300, in_hi=2000) measuring coupling only
(cheap, no training). At this forward the output region's baseline burst rate is already HIGH (B_rest ~0.34-0.36,
driven by the now-active hidden→output pathway), so the apical +300pA can barely raise it:

| soma-g | out_bias 520 | out_bias 200 | out_bias 80 |
|---|---|---|---|
| 200 | rise −0.030 (couples=F) | −0.053 (F) | −0.053 (F) |
| 800 | −0.024 (F) | −0.042 (F) | −0.042 (F) |
| 2000 | −0.007 (F) | −0.034 (F) | −0.034 (F) |
| 5000 | **+0.011 (couples=T)** | −0.010 (F) | −0.010 (F) |

**⇒ STRUCTURAL TENSION: the strong forward drive that makes the HIDDEN input-selective is the same drive that
SATURATES the OUTPUT's baseline bursting (B_rest ~0.35), which kills the apical→burst COUPLING the credit needs**
(coupling requires B_rest LOW so the apical can raise it). At the smoke's weak drive (in_hi=750) the output was silent
(B_rest 0.000) so soma-g 200 coupled cleanly; at the input-selective drive it cannot. Only soma-g=5000 barely couples
(+0.011). These two requirements — input-selective hidden (strong drive) and couplable output (low baseline bursting) —
are in direct tension at every tested config.

**LIKELY RESOLUTION + the runner limitation:** the tension is because ONE `--fwd-wmean` sets BOTH pathways. The fix is
INDEPENDENT pathway weights — strong input→hidden (for hidden selectivity) + WEAK hidden→output (so the output does not
over-burst, keeping B_rest low + couplable). The runner's `RegionPathway`s already exist separately (`bridge.py`
`input→hidden`, `hidden→output`), so this is a bounded runner change (`--fwd-wmean-ih` / `--fwd-wmean-ho`), NOT a `sim/`
edit. That is the concrete NEXT experiment before any credit verdict.

**HONEST SCOPE:** single forward config (fw=500/hb=300/in_hi=2000); a different input-selective config (e.g. lower
in_hi + longer settle) might give a lower output B_rest — part of the joint search. The gap#4 KEYSTONE (deep local
credit works + composes across a layer) is separately ESTABLISHED (rung 10, Poisson geometry); THIS sub-thread is the
harder on-bridge learn-to-ACCURACY demonstration, now characterized as blocked by a drive-vs-coupling operating-point
tension — a well-posed engineering problem on a signal-carrying substrate, not a wall.

## Tension RESOLVED (independent pathway weights) → the next layer is a plasticity INSTABILITY

Added independent pathway weights to the runner (`--fwd-wmean-ho`, default None = byte-identical; verified: h12/e3
default path reproduces in2hid 0.484 / held 0.731). Strong input→hidden (fw_ih=500) + WEAK hidden→output (fw_ho=6-20)
RESOLVES the tension: the output B_rest drops to 0.000 (couplable) → the apical **couples cleanly (B rises +0.18 to
+0.50 at soma-g 800-2000)** WHILE the hidden stays input-selective (‖h1−h2‖=13.2, unchanged — selectivity rides
input→hidden, coupling rides the low output baseline). So the operating point where BOTH hold now EXISTS.

BUT the BDSP run there (fw_ih=500, fw_ho=6, soma-g 1000, in_hi=2000, e12) exposes the NEXT layer — a plasticity
INSTABILITY: **couples=True (B_apical 0.504)** but **BDSP 0.439 < LESION 0.575 < reservoir 0.67**, with the smoking gun
**dw_in2hid lesion = 734,339 vs bdsp 5.1** — the input→hidden weights EXPLODED in the lesion arm. The strong drive
(in_hi=2000, fw_ih=500) + unrescaled `bdsp_lr=0.03` causes runaway plasticity (the CLAUDE.md STDP-w_max gotcha in BDSP
form). The run is unstable/mis-tuned, so the BDSP<lesion comparison is CONFOUNDED, not a credit verdict.

**⇒ NEXT (the "rescale credit magnitude" step, now concrete): a lr × w_max STABILITY sweep at the resolved operating
point** — lower `bdsp_lr` (0.03 → 3e-3/3e-4) + tighten `bdsp_w_max` until the lesion dw is bounded (no runaway), THEN
read BDSP-vs-reservoir(0.67), 6-seed, vs (BDSP > reservoir-baseline, wrong_sign < baseline). Only a STABLE, coupled,
input-selective run answers the credit-to-accuracy question. The resolution (independent pathway weights) is a real
mechanistic unlock; the stability sweep is the last confound to clear.

## DEFINITIVE VERDICT (clean substrate) — the BDSP credit does NOT beat a reservoir readout; the value is the readout, not credit-training the hidden

The "instability" was an artifact: lowering `bdsp_lr` (0.003→0.00003) changed NOTHING (BDSP/lesion identical; lesion
dw stuck at 734,339). Reading the weights: `fw_ih=500` init (w_in2hid 1,169,708 ≈ 500/synapse) is CLIPPED to
`bdsp_w_max=200` (435,369 ≈ 189/synapse) — the CLAUDE.md STDP-w_max gotcha, an lr-independent clip artifact, NOT a
runaway. Fixed by using `fw_ih=180 < w_max` (no clip): it STILL gives input-selective hidden (‖h1−h2‖=11.6) AND clean
coupling (B_rest 0.000, rise +0.50) — a fully confound-free operating point.

**THE CLEAN TEST (fw_ih=180, fw_ho=6, in_hi=2000, soma-g 1000, coupled, no clip, seed 42):**

| arm | held-out | (floor 0.510, oracle 0.989) |
|---|---|---|
| **RESERVOIR readout** (trained readout on RANDOM-hidden features, credit-independent) | **0.765** | reservoir computing WORKS — the substrate's random hidden features are useful |
| BDSP credit-trained | 0.553 | ≈ lesion 0.550 ≈ wrong 0.564 |

**VERDICT: at a confound-free operating point, the BDSP graded-burst-credit produces NO accuracy benefit — BDSP 0.553 ≈
lesion 0.550 ≈ wrong 0.564, ALL well BELOW the reservoir baseline 0.765.** The load-bearing comparison is clean (BDSP vs
the credit-INDEPENDENT reservoir): **the BDSP-credit-trained net UNDERPERFORMS a simple trained readout on the same
random hidden features.** So the deep BDSP credit does not learn accuracy-relevant hidden features on this substrate —
the value is in the trainable READOUT over a fixed random hidden layer, not in credit-training the hidden. **This
directly echoes the project's own R3 reservoir reframe** (a fixed random recurrent/hidden scaffold + a trained readout
BEATS training the scaffold; ROADMAP §9.1).

**⇒ CLEAN NEGATIVE (finally on a valid substrate) — a verdict on the METHOD, not the capability, per THE LAW.** The raw
+ graded burst-credit + FA/KP family does not beat reservoir computing on this task. gap#4's KEYSTONE (deep local credit
works + composes across a layer, rung 10 Poisson geometry) stands SEPARATELY — this is the harder learn-a-classification-
task-to-ACCURACY sub-thread. **NEXT:** (1) firm it multi-seed (single-seed here; but the whole arc is now confound-free);
(2) a fresh research gate for a stronger credit signal that beats a reservoir readout — OR accept that on this substrate
the honest on-bridge capability IS the reservoir readout (0.765), with the credit's role being the read-out training,
consistent with R3. (Note: the lesion dw 64,403 at fw_ih=180 shows the lesion arm's weights still drift, so BDSP-vs-lesion
is mildly confounded; the BDSP-vs-RESERVOIR comparison, which is credit-independent, is the clean load-bearing one.)

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
