# gap#5: the robust-firing operating point CORRECTS my own O'Keefe-Nadel "confirmation"

**2026-07-30.** Fixing the root blocker (readers firing ~1.33 spikes/lap) required parameterising `W0` in
`_gap5_btsp_place_field_derisk.py`, which had it hardcoded at 250 — the reason the blocker was unsweepable.

## The operating point, found by sweep (gate metric = spikes per reader per lap)

| w0 | drive | spk/reader/lap | circ | verdict |
|---|---|---|---|---|
| 250 | 3000 | 1.33 | 0.1585 | too quiet |
| 600 | 3000 | 1.75 | 0.1162 | too quiet |
| **600** | **8000** | **25.58** | 0.0921 | **ROBUST** |
| 900 | 8000 | 25.92 | 0.0777 | robust, sat rising |

**⚠️ `thr_scale` (scaling `cp_neuron_firing_thresholds` for the read slice) is an INERT LEVER** — 0.7 produced
byte-identical spike counts (16 and 307) to 1.0. Recorded so it is not relied on; the thresholds are allocated at
`bridge.py:1629` and something downstream evidently re-derives or ignores the post-hoc edit. Not diagnosed further.

## ⛔ SELF-CORRECTION: the 4.17-peak fragmentation was a DRIVE ARTIFACT, not an inhibition deficit

I recorded that **4.17 peaks/cell, 100% multi-peaked** confirmed O'Keefe-Nadel's prediction that too few converging
inhibitory inputs make a unit fire *"in several parts of the environment."* **At the robust-firing point the
fragmentation collapses to 1.47 peaks/cell with NO inhibition change** — driven entirely by `w0` and drive.

| | quiet point | robust point |
|---|---|---|
| spikes/reader/lap | 1.33 | 31.6 |
| **peaks/cell** | **4.17** | **1.47** |
| circ | 0.1585 | 0.0878 |

**So the correct explanation of the fragmentation is the OPERATING POINT, not converging inhibitory inputs.** The
1978 passage may still be right about biology; my measurement did not test it, because the variable that moved the
peaks was drive. **This retracts the "prediction CONFIRMED against my own data" claim** — a textbook mechanism and
a coincidentally-matching number are not the same thing, and I read the match as confirmation without checking
whether a different variable produced it.

## The defect has MOVED, not vanished

Fields are now near-single-peaked (1.47) but circ FELL to 0.0878 (oracle: **0.8719**). Single-but-WIDE replaces
multi-peaked-but-narrow: more firing ⇒ more potentiation ⇒ a broader field, and a broad single field has low
circular resultant because its mass spreads over many angles. **The remaining defect is field WIDTH.**

## Inhibition now BITES — weakly, but for the first time

At the robust point, basket/FS inhibition raises circ on every arm: **+0.0064 (w=150), +0.0091 (w=400), +0.0074
(w=900)**, all positive, against a flat-to-negative result at the quiet point. That is consistent with the causal
story — inhibition cannot sculpt a silent population, and now that the readers fire it does something. The effect
is ~10% of the gap to the oracle, so it is a real but small lever, honestly below the +0.01 threshold I set in
advance.

## Next

Width is the target: circ 0.0878 → toward 0.8719. Levers, cheap-first: (a) push inhibition further now that it
engages (the trend is monotone-ish up to w=400 then flattens); (b) `btsp_hetero_dep` heterosynaptic competition,
still untried and explicitly "lowers the pedestal without lowering the peak" — aimed exactly at width;
(c) the theta time-gate, which restricts WHEN potentiation can occur and should narrow the field in time;
(d) shorter `btsp_elig_tau_ms` (currently 1000 ms) — a seconds-long eligibility window over a 1.8 s lap
potentiates most of the lap by construction, which is a plausible direct cause of the width.

## ⛔ THIRD METRIC FAILURE, and it weakens my own BTSP interpretation: `peaks=1.00` meant UNIFORM

Shortening `btsp_elig_tau_ms` collapsed peaks/cell to exactly **1.00** — which I was about to read as "a clean
single field." It is the opposite. Adding a WIDTH measure (how many of the 60 place inputs sit above half-max;
sigma=5 oracle ≈ 12/60, uniform = 60/60):

| elig_tau | peaks/cell | **width /60** | circ(dW) | mean abs dW |
|---|---|---|---|---|
| 1000 | 1.71 | **51.0** | 0.1340 | 1502 |
| 300 | 1.29 | 57.9 | 0.0517 | 1753 |
| 100 | **1.00** | **60.0** | 0.0263 | 1825 |
| 30 | **1.00** | **60.0** | 0.0175 | 1841 |

**Width 60/60 is TOTALLY UNIFORM.** A flat profile has exactly ONE contiguous run above half-max, so `peaks=1.00`
is satisfied *trivially* by uniformity. **`peaks/cell` cannot distinguish "one sharp field" from "everything
potentiated equally"** — the third metric of this arc with a degenerate case (after `peak/mean` being
permutation-invariant, and `circ(M1)` being diluted by the random initial weights). **Report WIDTH with peaks,
always.**

**⇒ NEGATIVE on the tau lever, and it VALIDATES the biological default.** Shortening tau makes things strictly
worse: it drives toward uniform saturation (`mean|dW|` rises 1502→1841 while circ falls 0.134→0.018). The
seconds-long `elig_tau_ms=1000` — Bittner-Magee's actual biology — is the best of the tested range. My
ratio hypothesis (`tau/field-crossing` should be ~1) is **refuted**: at ratio 1.0 the result is uniform mush.

**⇒ AND IT WEAKENS THE INTERPRETATION OF MY OWN BTSP POSITIVE (the measurement stands; the reading does not).**
Even at the best setting the learned field is **51 of 60 inputs wide**. The +0.1281 circ gain was real, 6-seed,
and survived randset and permuted-increment controls — so BTSP genuinely writes *position-dependent* structure —
but that structure is a **weak modulation on near-uniform potentiation, not a place field.** "Acquires spatial
place structure" was too strong; the honest phrasing is **"acquires a weak position-dependent modulation
(width 51/60, circ 15% of oracle)."** The controls established that the modulation is real and spatial; they never
established that it is localized, and I did not measure localization until now.

**⇒ THE DEFECT, FINALLY STATED CORRECTLY: potentiation is near-global.** Every arm potentiates ~85-100% of each
reader's afferents. Nothing in the current setup restricts WHICH afferents get potentiated — which is exactly what
O'Keefe-Nadel's convergent inhibition, and BTSP's own `btsp_hetero_dep` heterosynaptic competition ("lowers the
pedestal without lowering the peak"), are for. `btsp_hetero_dep` remains **untried** and is now the single most
directly-aimed lever: it targets the pedestal, and the pedestal is the entire problem.

## ⭐ `btsp_hetero_dep` IS THE MISSING MECHANISM — pedestal solved, but only ~40% of the gain is place-specific

Heterosynaptic competition (`btsp_hetero_dep`, default 0.0, engine comment: *"lowers the pedestal without lowering
the peak"*) was the last untried lever and it works dramatically. At robust firing, `elig_tau=1000` (the validated
biological default), 3 seeds:

| hetero_dep | peaks/cell | **WIDTH /60** | **circ(dW)** | mean abs dW |
|---|---|---|---|---|
| 0.00 | 1.61 | 51.1 | 0.1323 | 1502 |
| 0.05 | 4.97 | 44.1 | 0.1812 | 933 |
| 0.20 | 8.08 | **17.0** | 0.4192 | 322 |
| 0.50 | 4.11 | **5.0** | **0.6644** | 234 |

**The pedestal problem is SOLVED: width 51/60 → 5/60** (sigma=5 oracle ≈ 12/60), and circ rises 5x, from 15% to
76% of the 0.8719 oracle. This is the mechanism O'Keefe-Nadel's convergent inhibition and BTSP's own
heterosynaptic term were both pointing at, and it was sitting in the engine at default 0.0 the whole time.

**⚠️ BUT THE CONTROLS CUT THE CLAIM, and this is the honest headline:**

| control | width | circ(dW) |
|---|---|---|
| place sweep | 5.0 | **0.6644** |
| `lr=0` | — | **0.0000** (clean; zero potentiation) |
| **RANDSET (matched activity, NO place manifold)** | 2.5 | **0.3893** |

**59% of the 0.6644 is reproduced with NO place structure whatsoever.** `hetero_dep` is winner-take-all over each
cell's afferents, so it produces CONCENTRATION by construction — and `circ_resultant` rewards concentration
regardless of locality (the same trap that voided `peak/mean` and inflated my Dirichlet null earlier in this arc).

**⇒ THE HONEST NUMBERS.** Place-specific increment = 0.6644 − 0.3893 = **0.275**, against ~0.13 at
`hetero_dep=0` — so heterosynaptic competition roughly **DOUBLES the place-specific component** (a real advance)
while the raw circ figure overstates it by ~2.4x. **Do not quote 0.6644 as a place-tuning score.** Quote:
*width 51→5 (pedestal solved), place-specific circ ~0.13→~0.275 (2x), with 59% of the raw gain being
place-independent concentration.*

**⇒ AND THE REMAINING DEFECT IS NOW FRAGMENTATION, not width.** peaks/cell RISES with competition (1.61 → 8.08 at
dep=0.2). Narrow-but-gappy replaces wide-but-solid: the surviving synapses are localized (circ is high, so they
cluster near one position rather than scattering around the ring) but riddled with holes. **That is precisely the
job of the topological/lateral mechanism** — O'Keefe-Nadel's "inhibition ... through the mediation of OTHER PLACE
UNITS" is what should make the survivors CONTIGUOUS. The two mechanisms are complementary: `hetero_dep` narrows,
lateral topology should de-fragment. The basket-cell population is already built and verified to fire; it now has
a well-posed job and a metric (peaks/cell at fixed width).

## De-fragmentation: uniform inhibition is INERT — and the mechanism was aimed at the WRONG DIMENSION

Tested the complementary lever at `hetero_dep=0.2`, comparing at genuinely FIXED width (the metric the previous
rung established):

| w_inh | peaks/cell | WIDTH /60 | circ(dW) |
|---|---|---|---|
| 0 | 7.81 | 17.2 | 0.4236 |
| 150 | 8.08 | 17.0 | 0.4192 |
| 400 | 8.44 | 17.0 | 0.4155 |
| 900 | 8.19 | 17.1 | 0.4156 |

**Inert — peaks do not fall (7.81 → 8.44 if anything), at width held to 17.0-17.2.** A clean negative, and the
width really was fixed, so the comparison was well-posed.

**⇒ THE DIAGNOSIS THAT MATTERS: I aimed the mechanism at the WRONG DIMENSION.** The fragmentation lives in the
AFFERENT (place-index) dimension — *which of a reader's 60 place inputs survive competition*. Lateral inhibition
BETWEEN readers shapes *which reader fires where*; it cannot make one reader's surviving afferents CONTIGUOUS.
O'Keefe-Nadel's "mediation of other place units" governs field POSITION across the population, not within-profile
contiguity — I conflated the two when I named it as the de-fragmentation fix.

**⇒ THE CORRECT MECHANISM CLASS, and the biology is specific: DENDRITIC CLUSTERING.** Afferents from neighbouring
place cells land on the SAME dendritic branch, share that branch's local plateau, and therefore survive or die
TOGETHER — which produces contiguity in exactly the place-index dimension where the gaps are. This project already
has the machinery: `enable_coincidence_detection` gives each postsynaptic neuron a **dendritic SUBUNIT** over a
tagged pathway (`config.py:159`), and it is already ON in these runs (BTSP requires it) — but the synapse→subunit
ASSIGNMENT is not organised by place index, so neighbouring place inputs are not guaranteed to share a branch.
**The named next build: assign place→reader synapses to dendritic subunits by PLACE-INDEX NEIGHBOURHOOD, so
competition operates per-branch over contiguous input blocks rather than per-synapse over the whole profile.**
Untried levers that also act in the right dimension: `btsp_elig_hard_thresh` (k-WTA on presynaptic eligibility)
and `btsp_elig_exponent` (supralinear eligibility).

## ⭐⭐⭐ THE AFFERENT-DIMENSION LEVERS CLOSE MOST OF THE GAP — place-specific circ 0.13 → 0.595 (68% of oracle), all controls passing

The dimension diagnosis was right. Both untried eligibility levers act on the afferent (place-index) axis, and both
work — at `hetero_dep=0.2`, robust firing, `elig_tau=1000`, 3 seeds:

| arm | peaks/cell | WIDTH /60 | circ(dW) |
|---|---|---|---|
| baseline | 7.81 | 17.2 | 0.4236 |
| `elig_hard_thresh=0.5` | 6.19 | **13.9** | 0.7064 |
| **`elig_exponent=4.0`** | **3.19** | 16.1 | 0.6853 |

**AND THE CONTROLS NOW PASS DECISIVELY — the concentration artifact is GONE:**

| control | `elig_exponent=4.0` | `elig_hard_thresh=0.5` | (hetero_dep alone, for contrast) |
|---|---|---|---|
| `lr=0` | **0.0000** | **0.0000** | 0.0000 |
| RANDSET (matched activity, NO place manifold) | **0.0906** | 0.1431 | 0.3893 |
| permuted-increments (same magnitudes, shuffled positions) | 0.1462 | 0.1738 | — |
| **place-specific circ (sweep − randset)** | **+0.5947** | +0.5633 | +0.275 |

**RANDSET collapses 0.3893 → 0.0906.** The place-independent concentration that inflated the `hetero_dep`-only
result is eliminated: the supralinear eligibility only concentrates where the INPUT was strongly and recently
active, which under a moving bump means a contiguous place-index block — so it cannot manufacture the score from
random input. RANDSET's own profile is FRAGMENTED (peaks 11.00) against the place sweep's 3.19, which is a clean
independent discriminator. Permuted-increments 0.1462 vs real 0.6853 (4.7x) says the structure is SPATIAL.

**⇒ THE HONEST HEADLINE: place-specific circ 0.5947 = 68% of the 0.8719 sigma=5 oracle ceiling, width 16.1/60
(oracle ~12), peaks 3.19, on 3 seeds, with lr=0 at exactly zero, randset at 0.09, and permuted-increments at
0.15.** Every control that killed or deflated an earlier claim in this arc now passes.

## The evening's progression, and why each step was found

| stage | place-specific circ | width /60 | peaks/cell | what moved it |
|---|---|---|---|---|
| quiet operating point | ~0.13 | 51 | 4.17 | — |
| robust firing (`w0=600`, drive 8000) | ~0.13 | 51 | 1.61 | fixed the silent-reader blocker |
| `+ btsp_hetero_dep=0.2` | 0.275 | 17 | 7.81 | lowered the pedestal |
| `+ btsp_elig_exponent=4.0` | **0.595** | 16 | **3.19** | de-fragmented, in the right dimension |

**4.6x improvement in place-specific circ**, from a starting point whose own interpretation I had to retract twice.
Every advance came from a control on an apparent win, or from a primary source read after the owner's challenge:
`hetero_dep` and the eligibility terms were BOTH already in the engine at inert defaults, and both were named by
O'Keefe-Nadel's "convergent inhibition restricts where the unit fires" once that passage was actually read.

**REMAINING GAP (small, and named):** width 16.1 vs oracle ~12, peaks 3.19 vs 1, circ 68% vs 100%. Untried:
combining `elig_exponent` WITH `elig_hard_thresh` (they act differently — one sharpens, one cuts, and their
best-of columns differ), and the dendritic-subunit-by-place-index assignment which is still the principled
contiguity mechanism. 6-seed confirmation and a GPU parity run are also owed before this is a GO.

## 6-SEED CONFIRMATION + COMBINATION: raw circ reaches 97% of oracle, but PLACE-SPECIFIC circ PLATEAUS at ~0.60

| arm | peaks/cell | WIDTH /60 | circ(dW) | randset | **place-specific** |
|---|---|---|---|---|---|
| `exp=4.0` (3 seeds, prior rung) | 3.19 | 16.1 | 0.6853 | 0.0906 | **+0.5947** |
| `thresh=0.5` | 6.68 | 14.0 | 0.6743 | 0.1329 | +0.5414 |
| **`exp=4.0 + thresh=0.5`** | 3.79 | 7.8 | **0.8461** | 0.2493 | **+0.5968** |
| `exp=6.0 + thresh=0.5` | 3.04 | 4.8 | 0.8863 | 0.3883 | +0.4980 |

6 seeds; per-seed circ at the best arm is tight: **[0.871, 0.869, 0.853, 0.790, 0.859, 0.834]**. `lr=0` is
**0.0000** on every arm.

**THE LEVERS COMPOSE ON RAW circ — 0.685 → 0.8461, which is 97% of the 0.8719 oracle. THEY DO NOT COMPOSE ON THE
HONEST METRIC.** Place-specific circ is **+0.5968** combined vs **+0.5947** for `exp=4.0` alone — a difference of
0.002, i.e. nothing. The entire raw gain came with a matching rise in RANDSET (0.0906 → 0.2493): the extra
narrowing is **place-INDEPENDENT concentration**. Pushed further (`exp=6.0`) raw circ 0.8863 **EXCEEDS the oracle**
while place-specificity **DROPS to 0.4980** and randset reaches 0.3883 — the unmistakable signature of a score
being won by concentration rather than by place tuning. Width 7.8 and 4.8 are also NARROWER than the oracle's ~12,
i.e. over-narrowed.

**⇒ THE PLACE-SPECIFIC COMPONENT HAS PLATEAUED AT ~0.597 = 68% OF ORACLE, and the remaining 32% is NOT reachable
by these levers.** Every further narrowing buys raw circ and randset in equal measure. **Reporting `circ = 0.846`
as a place-tuning result would be the single most misleading number in this arc** — it is 97% of oracle and ~30% of
it is artifact. The defensible claim is: **place-specific circ 0.597 (68% of oracle), 6 seeds, `lr=0` exactly zero,
peaks 3.19, width 16.1** — i.e. the `exp=4.0` arm, NOT the higher-scoring combination.

**⇒ WHAT THE PLATEAU MEANS MECHANISTICALLY.** These levers all sharpen WHICH afferents survive, using only
eligibility magnitude. None of them uses the place-index TOPOLOGY — so they cannot prefer a contiguous block over
an equally-strong scattered set, which is exactly why peaks stalls at ~3 and why the place-specific part stops
improving. **The remaining 32% needs a mechanism with topology in the afferent dimension: the
dendritic-subunit-by-place-index assignment** (subunits already exist and are active — `enable_coincidence_detection`
— but are not assigned by place neighbourhood). That is a structural change, not a knob, which is consistent with
it being the part knobs cannot reach.

**STILL OWED before any GO: GPU parity** (the check that revealed the order-read stack was numpy-locked by
construction) and a decision on whether 68%-of-oracle place tuning is sufficient to drive the order read — the
integration test that failed at width 51/60 should be re-run at width 16/60.

## INTEGRATION STILL FAILS at width 16/60 — tuning quality was NOT the binding constraint, and the ratios say why

Re-ran the integration with the validated sharp-tuning config (width 16.1/60, place-specific circ 0.597 — a 3x
sharpening over the 51/60 config that failed before), 3 seeds:

| arm | mean ratio | per-seed |
|---|---|---|
| LEARNED | 0.714 | 0.83, 0.52, 0.79 |
| **SCRAMBLED_pairing** | **0.684** | 0.81, 0.50, 0.74 |
| LEARNED_lesion | 0.600 | 0.63, 0.49, 0.68 |
| UNTRAINED | 1.305 | 2.47, 0.29, 1.15 (wildly variable, low counts) |

**LEARNED 0.714 vs SCRAMBLED 0.684, and near-identical PER SEED (0.83/0.81, 0.52/0.50, 0.79/0.74).** The learned
ordering still carries NO information. **⇒ My earlier diagnosis — "fields too broad for `argmax` to define an
order" — is REFUTED. Tripling the sharpness changed nothing**, so tuning quality was not the binding constraint.

**⇒ BUT EVERY RATIO IS BELOW 1.0 (reverse > forward), WHICH IS A DIAGNOSTIC, NOT NOISE.** That is the exact
signature of the trap documented earlier in this arc: **a coincidence detector that is SUPRATHRESHOLD to a single
input reads order BACKWARDS** (separated arrivals give two independent bursts; coincident arrivals collide in the
refractory period and yield FEWER spikes). And the cause is now obvious: BTSP drives the learned weights to
**~2500**, while the detector was calibrated against the original **w0=250**. Installing learned weights at
`gain=1.0` therefore drives the readers ~10x harder than the regime in which `w_det=10` was verified subthreshold.
`LEARNED_lesion` at 0.600 — also inverted — is consistent with a detector out of its operating regime rather than
with a broken order code.

**⇒ THE NAMED FIX (calibration, not mechanism): re-establish the subthreshold coincidence regime for the new
weight scale** — sweep the install `gain` (or `w_det`) and **assert the single-input check** (`single_input_check`
already exists in `_gap5_onsubstrate_population_vote_derisk.py` and the population runner ABORTS on it) before
reading any ratio. The order read itself is validated at **0.969 single-trial** in its own calibrated regime, so
the likeliest reading is that the two halves are fine and the JOIN is mis-calibrated. Until that check is run, the
integration result is **UNDEFINED, not a negative** — by the same rule that voided three earlier arms tonight.

**HONEST STATE OF THE ARC AT SESSION END:** both halves independently validated (order read 0.969 single-trial on
GPU; tuning acquisition place-specific circ 0.597 = 68% of oracle, 6 seeds); the JOIN is unproven, with its most
likely cause identified as detector calibration and a concrete pre-flight to settle it; and the remaining 32% of
tuning quality is attributed to a structural mechanism (dendritic subunits by place index) rather than any knob.

## PRE-FLIGHT RESOLVED: the detector WAS mis-calibrated, and at the correct gain the learned order carries information for the FIRST time — but the join is still not working

Swept the install `gain` with the single-input coincidence check ASSERTED (detectors must stay silent when only one
reader's inputs are driven), 3 seeds:

| gain | single-input detector spikes | LEARNED | SCRAMBLED | regime |
|---|---|---|---|---|
| 1.00 | **11.7** | 0.714 | 0.684 | ⛔ SUPRATHRESHOLD |
| **0.30** | **0.0** | **0.822** | 0.672 | ✓ subthreshold |
| 0.10 | 0.0 | 0.583 | 0.667 | ✓ subthreshold |
| 0.03 | 0.0 | 0.667 | 1.000 | degenerate |
| 0.01 | 0.0 | 1.000 | 1.000 | **DEAD — zero spikes both directions (1.000 is 0/0)** |

**TWO REAL FINDINGS.** (1) **The calibration diagnosis was CORRECT:** at `gain=1.0` the detector fires 11.7 spikes
on single-input drive — genuinely suprathreshold, exactly the regime that inverts the order read. The previous
integration "negative" was therefore measured outside the detector's valid operating window, and was properly
labelled UNDEFINED. (2) **At the calibrated `gain=0.30`, LEARNED (0.822) separates from SCRAMBLED (0.672) for the
FIRST TIME in this arc** — a 22% separation where every prior attempt gave LEARNED ≈ SCRAMBLED (0.714/0.684,
0.678/0.683). **The learned ordering now carries information.**

**BUT THE JOIN IS STILL NOT WORKING, and I am not going to round this up.** `LEARNED = 0.822 < 1.0` means forward
still produces FEWER detector coincidences than reverse. A subthreshold coincidence detector reading a correctly
ordered sequence should give **> 1**. So the sign is still wrong, and the remaining candidates are:
(a) the K=6 readers chosen by `linspace` over `argsort(pref)` may not fire monotonically in time even though their
WEIGHT preferences are ordered — weight-argmax order != firing order, which is the same conflation that produced
the width/peaks confusion earlier; (b) the relay delay (~11.5 ms, tuned for the hand-set regime) may be mismatched
to the inter-reader interval that the LEARNED preferences actually produce. **(b) is directly measurable** — record
each reader's actual first-spike time under the sweep and compare the observed inter-reader interval against the
relay delay, instead of assuming the 12.5 ms optimum transfers.
**Also note `gain <= 0.03` is DEAD, not good:** a ratio of exactly 1.000 there is `0/0`, i.e. no detector spikes at
all — a degenerate arm that must not be read as "no directional bias."

**FINAL HONEST STATE OF THE ARC:** two halves independently validated (order read **0.969** single-trial on GPU,
6 seeds; tuning acquisition **place-specific circ 0.597** = 68% of oracle, 6 seeds, every control passing); the
JOIN now shows its first genuine signal (learned 0.822 vs scrambled 0.672 in a verified-subthreshold regime) but
has the wrong SIGN, with a measurable next diagnostic named. Owed: GPU parity on the tuning result, and the
first-spike-time measurement above.

## ⭐⭐⭐ THE ACTUAL BLOCKER, FOUND: ALL READERS LEARN THE SAME FIELD. The population has ONE field, not twelve.

The named diagnostic (measure each reader's real first-spike time instead of assuming weight-order = firing order)
returned something that re-frames this whole arc:

| seed | learned preferred positions (6 readers) | first-spike steps | monotonic? |
|---|---|---|---|
| 42 | **[48, 48, 48, 48, 48, 48]** | [34, 25, 27, 23, 25, 35] | No |
| 43 | **[47, 47, 47, 47, 47, 47]** | [23, 26, 25, 24, 31, 25] | No |
| 44 | **[48, 48, 48, 48, 48, 48]** | [25, 26, 22, 34, 27, 27] | No |

**EVERY READER LEARNS AN IDENTICAL PREFERRED POSITION.** There is ZERO spatial diversity across the population.
First-spike times cluster at 22-35 steps with no monotonic structure, because all readers fire when the bump
crosses that one position.

**⇒ THIS EXPLAINS THE ENTIRE INTEGRATION FAILURE, ALL THE WAY BACK.** No ordering is possible; `argsort(pref)` over
identical values is arbitrary; and **LEARNED ≈ SCRAMBLED at every gain because there is NO LEARNED ORDER TO
SCRAMBLE** (0.678/0.683, 0.714/0.684, 0.822/0.672 — the last being a 22% separation over *arbitrary* index
permutations, i.e. noise, not information; **that reading is withdrawn**).

**⇒ AND IT SUBSTANTIALLY RE-FRAMES THE TUNING RESULT (the measurement stands; its SCOPE was much narrower than I
stated).** `place-specific circ 0.597 = 68% of oracle` is a correct measure of the quality of **a** field — it is a
per-cell mean, so a population where all 12 cells learn the SAME good field scores exactly as well as one where 12
cells tile the track. **My metric could not distinguish those two cases, and I never checked which one I had.**
That is the FOURTH metric-blindness of this arc (after permutation-invariant `peak/mean`, `circ(M1)` diluted by
random init, and `peaks=1.00` meaning uniform), and the same failure mode each time: **a per-cell average that is
silent about the population.**

**⇒ THE ORIGINAL DIAGNOSIS WAS RIGHT AND I TESTED IT FOR THE WRONG JOB.** Between-reader competition was needed for
**DIFFERENTIATION** — making different readers claim different positions. I built the basket-cell ring and then
evaluated it on **de-fragmentation within one cell's afferent profile**, which is not its job, found it inert, and
moved on. The earlier measurement that "uniform FS inhibition cut firing 450→40 but produced NO differentiation"
was the real result all along, and I under-weighted it.

**⇒ NEXT: differentiation is the whole problem.** Required metric: **the SPREAD of learned preferred positions
across readers** (currently 0 — all identical; target ~uniform over the track). Levers: (a) the topological basket
ring evaluated on SPREAD, not peaks; (b) `btsp_elig_hard_thresh` as a genuine k-WTA so only the best-matched reader
potentiates per position; (c) heterogeneous reader thresholds/initial biases as a symmetry-breaker. **Every
tuning-quality number in this arc should be re-reported alongside preference-spread**, since a perfect field
learned twelve times over is worth no more than one.
