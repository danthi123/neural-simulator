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
