# Dendrite de-risk A (Stage 0) — GRADED dendritic-plateau READ-OUT: GO (the dendrite's ONE genuine unlock) (2026-06-20)

**Item:** the corrected cheap-first dendrite de-risk (the dendrite scoping `2acebf6b`, controller-verified). The dendrite
is the genuine unlocker for ONLY the GRADED READ-OUT of a distributed code (Mikulasch-Priesemann) — NOT credit-assignment/
survival (those were NEGATIVE 2026-06-19), NOT nav orienting (point-neuron loop-stability). Its cleanest instance is the
nav value-critic **δ = r − V**. This de-risk asks: does a GRADED (non-saturating, smooth NMDA/sigmoidal) dendritic-plateau
read-out produce a graded value signal where the two POINT-NEURON read-outs **provably can't** (burndown-9)?

**Verdict: GO — unanimous 6/6 seeds, all six anti-cheats green. NO `sim/` edit (Stage 0 reuse-by-import).** The graded
dendritic-plateau read-out reaches the host-Gaussian δ band (δ=1.33 ≥ host ~1.30) at faithful grid-32, multi-seed, where
BOTH point-neuron read-outs fail (LINEAR flat δ~1.0, all-or-none PLATEAU over-clamp). The graded-ness is load-bearing (the
all-or-none lesion loses the graded middle). **⇒ the dendrite's ONE genuine unlock is confirmed; this GREENLIGHTS Stage 1**
(the guarded protected `sim/` edit making a graded dendritic plateau a first-class bridge read-out).

## What burndown-9 proved (the floor this builds on)

Both POINT-NEURON read-outs fail at faithful grid-32 (`2026-06-20-burndown-9-critic-graded-readout.md`):
- **LINEAR** (point-neuron MSN-D1, linear synapse) = SUB-RHEOBASE → critic 0 Hz → no subtraction → near==far burst (~100 Hz
  each) → **flat δ 1.00**. A point neuron cannot express the value at all.
- **PLATEAU** (all-or-none coincidence) = OVER-CLAMPS → critic ~176–219 Hz → GABA_B annihilates BOTH bursts → **δ 0.00**.
  A point neuron with a regenerative all-or-none switch over-subtracts.

So a point neuron cannot express the graded MIDDLE: linear is 0, all-or-none saturates. The dendrite's claim is that a GRADED
(smooth) dendritic-plateau read-out CAN.

## Two substrate facts confirmed directly here (the WHY the dendrite is needed)

Both probed on the deterministic-nav-faithful bridge (the exact burndown-9 regime):
- **(i) The MSN-D1 striosome will NOT fire gradedly.** A DIRECT depolarizing current onto it (0 → 500 pA) produces **0 Hz at
  EVERY magnitude** (deep rest + high rheobase + inward-rectifier down-state). The graded value CANNOT live in the MSN somatic
  spike rate — the point-neuron wall, directly. (This is why the LINEAR arm reads 0 Hz at any weight, and why the only thing
  that fires the MSN is the saturating all-or-none plateau.)
- **(ii) A graded value SUBTRACTED at the SNc DOES grade the reward burst** (100 → 75 → 50 → 25 → 0 Hz as the subtraction
  rises 0 → 400 pA). So if a graded analog value can be produced AND made location-selective, the δ = far/near gap opens in
  the graded middle. The point neuron can't produce it; the dendrite can — its plateau read-out is a graded analog quantity.

## The dendritic arm (Stage 0, NO `sim/` edit — reuse `sim/dendritic_neuron` + `sim/dendritic_plasticity`)

A `DendriticLayer` (Larkum BAC two-compartment) reads the SAME grid-32 place population code that drives the point-neuron
critic. Its basal weights `W_basal` LEARN location-selectively via the LOCAL Urbanczik-Senn rule (`urbanczik_senn_update`,
apical-gated by the SNc-derived reward delta — biologically LOCAL, NO backprop, NO weight transport) so V(near) rises and
V(far) stays low. The GRADED dendritic-plateau read-out is the SMOOTH (non-saturating across the active range) sigmoid of the
plateau drive: **V = sigmoid((v_basal − θ)/slope)** — the graded analog read-out the point neuron provably cannot produce.
V is delivered as a GRADED inhibitory subtraction at the SNc (the dendritic analogue of the striosome→SNc GABA_B subtraction
— the SAME SNc, the SAME subtract-at-SNc mechanism probe (ii) confirmed grades the burst; only the value's SOURCE differs:
a graded dendritic plateau, not the un-fireable point-neuron MSN spike rate). δ = far_burst/near_burst is read EXACTLY as
burndown-9.

Runner: `research/runners/_dendrite_deriskA_graded_plateau_readout.py` (reuse-by-import of the validated
`snc_stageb_critic_probe_navfaithful` machinery + the burndown-9 point-neuron arms verbatim).

## δ TABLE — faithful grid-32, deterministic regime, lead 150 ms, 6 seeds (δ = far_burst[unpredicted] / near_burst[predicted]; host-Gaussian ref ~1.3)

| seed | DENDRITIC δ | V_dend near / mid / far | graded-3 | apical-lesion δ (loses-middle) | LINEAR δ | PLATEAU δ (above-floor) |
|---|---|---|---|---|---|---|
| 42  | **1.33** | 0.662 / 0.368 / 0.181 | Y | 2.00 (all-or-none, loses-middle Y) | 1.00 | over-clamp (no) |
| 43  | **1.33** | 0.645 / 0.428 / 0.205 | Y | 2.00 (loses-middle Y) | 1.00 | over-clamp (no) |
| 44  | **1.33** | 0.661 / 0.529 / 0.189 | Y | 2.00 (loses-middle Y) | 1.00 | over-clamp (no) |
| 100 | **1.33** | 0.661 / 0.380 / 0.176 | Y | 2.00 (loses-middle Y) | 1.00 | over-clamp (no) |
| 101 | **1.33** | 0.742 / 0.388 / 0.296 | Y | 2.00 (loses-middle Y) | 1.00 | over-clamp (no) |
| 102 | **1.33** | 0.744 / 0.421 / 0.267 | Y | 2.00 (loses-middle Y) | 1.00 | over-clamp (no) |
| **median** | **1.33** | — | 6/6 | — | **1.00** | 6/6 over-clamp |

Reference: host-Gaussian nav-deployment value-train δ ~1.3 (CYCLE-219/212). The dendritic δ lands exactly in the host band
(≥1.30, ≤ the 1.69 host-ceiling), at every seed, where the point-neuron read-outs do not.

(Note on the PLATEAU column: at seed 43 the raw far/near ratio reads a huge floor-division artifact — far ≈1.7 Hz noise vs
near 0 Hz — but `above_floor=False` at every seed, i.e. BOTH bursts are annihilated below the 10 Hz SNc floor. The over-clamp
signature is `not above_floor`, not the raw ratio value; the de-risk's `plateau_fails` criterion uses `not above_floor OR
δ ≤ 0.15` so the over-clamp is correctly counted 6/6.)

## ANTI-CHEAT collapse table (the burndown-9 + #6-lesson battery — ALL, 6 seeds)

| anti-cheat | result | reading |
|---|---|---|
| **(a)** TWO POINT-NEURON CONTROLS fail (the two-sided validity gate) | LINEAR flat 6/6 ; PLATEAU over-clamp 6/6 | the harness is correctly calibrated; the substrate genuinely can't |
| **(b)** APICAL/plateau LESION (graded → all-or-none) LOSES the graded middle | 6/6 lose-middle vs GRADED expresses the near>mid>far 3-continuum 6/6 | the GRADED-ness is LOAD-BEARING (a binary read-out can't carry the continuum: V(mid) snaps 0.37–0.53 → 0.000) |
| **(c)** GABA_B-equiv SUBTRACTION lesion collapses the headline δ | 6/6 collapse to δ≤1.15 | the gap IS the subtraction's doing, not host arithmetic |
| **(d)** REGIME FIDELITY (grid-32 deterministic; OU/cond-noise/homeostasis OFF) | asserted per seed (`_assert_deterministic_regime`) | replicates deployment — NOT a permissive smoke (the #6 lesson) |
| **(e)** HOST-CEILING (δ ≤ host×1.30 = 1.69) | 6/6 below | no goal/reward smuggling beyond what the place code carries |
| **(f)** LOCATION-SELECTIVITY of the LEARNED value (V_dend near>far + grew) | 6/6 | the value is LEARNED + place-specific (not hand-set, not place-blind) |

## The graded-necessity test (anti-cheat b — the genuine Mikulasch-Priesemann discriminator)

A clean binary near/far pair is NOT a sufficient test of "graded-ness": even an all-or-none single value unit opens a gap when
near is clearly above and far clearly below a threshold (observed: the apical-lesioned all-or-none arm gives δ=2.00, a gap, not
a collapse — because near→1, far→0 is one clean subtraction level). The graded-ness becomes strictly load-bearing only for a
graded value GRADIENT. So the de-risk adds a MID location (an intermediate value, ~0.4) and tests the 3-level continuum:
- **GRADED read-out:** V(near) > V(mid) > V(far) (e.g. 0.66 > 0.37 > 0.18) — the continuum expressed (6/6).
- **ALL-OR-NONE lesion:** V(mid) snaps to its binary level (→ 0.000) — the middle is LOST, the continuum broken (6/6).

This is the genuine claim: a graded analog read-out of a distributed code expresses a CONTINUUM a binary/saturating point-neuron
read-out cannot (the value continuum is read at the dendritic VALUE V — where the Mikulasch-Priesemann claim lives — because the
n_snc=30 SNc population quantizes the downstream burst to ~25 Hz steps, too coarse to display 3 levels).

## Honest scope, caveats, and what is/isn't claimed

1. **What is GO:** a graded dendritic-plateau read-out of the distributed place code produces a graded, location-selective,
   learned value V that (a) subtracts at the SNc to a host-band δ=1.33, (b) expresses a 3-level continuum, where the point
   neuron's somatic spike rate provably cannot (it is 0 or saturated). This is the **graded analog read-out** the dendrite
   uniquely affords.
2. **The graded read-out is a dendritic-compartment quantity, not a somatic spike rate.** Probe (i) shows the MSN won't fire
   gradedly at ANY injected current; the graded value lives in the dendritic plateau (the `soma_rate`/sigmoid of `v_basal`),
   delivered as a graded analog subtraction. This IS the Mikulasch-Priesemann distinction (the analog/graded computation is
   dendritic; a point-neuron substrate fundamentally cannot do it from somatic spiking).
3. **The learning is brain-based-shaped + local:** Urbanczik-Senn somato-dendritic mismatch, apical-gated by the SNc-derived
   reward delta projected through the neuron's OWN fixed-random apical feedback — NO backprop, NO weight transport.
4. **Seed-stability fix (honest engineering note):** a single-fibre apical projection (`n_teacher=1`) varied 10× seed-to-seed
   (seed 43's `B_apical` was 0.063 vs seed 42's 0.675 → the apical plasticity gate vanished → seed 43 stalled). The fix is an
   8-fibre apical tuft (`n_teacher=8`) so the fixed-random projection magnitude is seed-stable — biologically faithful (the
   apical tuft integrates a population of feedback axons), not a tuning hack. With it, all 6 seeds learn the graded value.
5. **NOT claimed:** that the dendrite is the unlocker for credit-assignment/survival (NEGATIVE 2026-06-19) or nav orienting
   (point-neuron loop-stability). This de-risk is scoped to the GRADED READ-OUT — its cleanest instance — and that is what
   greenlights.
6. **The Stage-0/Stage-1 boundary held:** NO `sim/` edit. The graded read-out is composed entirely from existing modules
   (`sim/dendritic_neuron.DendriticLayer`, `sim/dendritic_plasticity.urbanczik_senn_update`) + the navfaithful δ harness.
   No `sim/` edit was needed or made.

## Stage 1 (the greenlit next step — NOT this task)

The GO greenlights **Stage 1**: a guarded, default-off, byte-reviewed protected `sim/` edit that makes a graded dendritic
plateau a first-class bridge read-out — a graded (smooth, non-saturating) regenerative plateau current on a dedicated critic
compartment, so the value V is produced on-substrate by the spiking-bridge dendrite (replacing the Stage-0 numpy `DendriticLayer`
held alongside the bridge), and the δ = r − V flows entirely through bridge state. The Stage-0 numpy arm is the teaching scaffold
+ the validation ceiling Stage 1 must match.

## Reproduce

```bash
# Faithful 6-seed (CPU numpy is fine — tiny ~290-neuron bridge): the δ table + anti-cheats + verdict
SIM_BACKEND=numpy python -m research.runners._dendrite_deriskA_graded_plateau_readout \
    --seeds 42,43,44,100,101,102 --n-train 40 --lead-ms 150 \
    --out research/findings/raw/_dendrite_deriskA_graded_plateau.json
# CPU smoke (single seed, reduced training):
SIM_BACKEND=numpy python -m research.runners._dendrite_deriskA_graded_plateau_readout --seed 42 --n-train 20
```

Raw: `research/findings/raw/_dendrite_deriskA_graded_plateau.json`. Runner:
`research/runners/_dendrite_deriskA_graded_plateau_readout.py`.
