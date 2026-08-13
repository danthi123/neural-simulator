---
type: finding
status: boundary
date: 2026-08-13
mechanism: graded-dendritic-plateau-magnitude-preserving-readout
lanes: [D-pragmatics, A-affect]
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_magnitude_preserving_plateau_readout_derisk.py
artifacts:
  - research/findings/raw/_magnitude_preserving/w4_6seed.json
  - research/findings/raw/_magnitude_preserving/affect_6seed.json
  - research/findings/raw/_magnitude_preserving/w4_smoke.json
  - research/findings/raw/_magnitude_preserving/affect_smoke.json
builds_on:
  - research/findings/2026-08-13-w4-detector-k-recalibration-BOUNDARY.md
  - research/findings/2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md
  - research/findings/2026-06-20-dendrite-derisk-A-graded-plateau-readout.md
---

# The GO'd graded dendritic-plateau read-out is VERIFIED magnitude-preserving (6/6) but closes NEITHER 2026-08-13 boundary — because the two boundaries only LOOKED alike: W4's residual is the METRIC/OBJECTIVE, affect's is the WEIGHT SOURCE. Neither is read-out-limited (6-seed A/B, both).

<!--derived-->

**One-line verdict:** two lanes hit what looked like the SAME wall on 2026-08-13 — the substrate reads a
SIGN / present-absent robustly but graded MAGNITUDE weakly — and the named common fix was the GO'd graded
dendritic-plateau read-out (`enable_graded_dendritic_plateau`, de-risk A GO 2026-06-20; Mikulasch & Priesemann
analog dendritic read-out), whose output GRADES with the coincident input instead of SATURATING. Built + tested on
6 seeds with a verified-clean instrument. **The read-out IS magnitude-preserving** (Part A: 6/6 solo-silent,
monotonic, near-proportional; r(0.27)/r(1)=0.24 vs the all-or-none's ~1.0 saturation) — but it closes NEITHER
task boundary. Part A: graded still LOSES to onehot on the valid M1 (move **−0.082, 0/6**), even MORE than the
all-or-none read (−0.035, which reproduces the W4 finding EXACTLY). Part B: it does NOT lift the affect salience
(composed 0.081→0.096, +0.015, 3/6) and the ridge already reaches the 0.27 target with the POINT-SOMA read
(0.327). **The single "magnitude-preserving read-out" hypothesis is FALSIFIED as the common fix** — the two
boundaries have DIFFERENT causes now precisely localized: W4 is a METRIC-AGGREGATION / objective mis-specification
(not the detector read), affect is a WEIGHT-SOURCE saturation (not the read-out). NOT a GO; an honest boundary
that BANKS the read-out method (it works as a read, verified) and relocates each residual to its true, DIFFERENT,
already-named next mechanism. The refuted deep-credit / two-compartment / BDSP rule is NOT re-proposed.

## The mechanism, and why it is NOT a `sim/` edit

<!--derived-->

The graded dendritic plateau (`fused_graded_dendritic_plateau`, bridge block "2.3a-ter") is the SMOOTH,
non-saturating sibling of the all-or-none coincidence switch: it passes the WEIGHTED coincident drive
`c_w = Sum_j w_eff_j*x_j` through a GENTLE CENTERED logistic `V = sigmoid(slope*(c_w-center)) - floor`, scaled
to a regenerative Mg2+-self-limiting plateau current — so a fractional-mass coincidence yields a PROPORTIONALLY
smaller plateau than a full-mass one (V(near) > V(mid) > V(far)), the magnitude the all-or-none switch destroys.
It is enabled entirely by CONFIG (the exact pattern in `_dendrite_stage1_onbridge_graded_plateau.py`), NO `sim/`
edit:

```
cfg.enable_coincidence_detection = True   # builds the coincidence_detector routing mask (needed by BOTH forms)
cfg.coincidence_plateau_strength = 0.0    # the ALL-OR-NONE current OFF (g_inc==0) -> pure graded read
cfg.enable_graded_dendritic_plateau = True + graded_plateau_center/slope/strength
```

It is a READ-OUT NONLINEARITY, NOT a learning rule (the deep-credit / two-compartment / BDSP family is
tested-NEGATIVE for hidden credit on spikes — `2026-07-22-gap4-real-issue-NOT-dendrites` — and is NOT proposed
here). Additive NEW runner, reuse-by-import of BOTH boundary runners; plasticity off (fixed operating point).

## Part A (W4) — the read-out IS magnitude-preserving, but M1 does NOT move

<!--derived-->

Per-seed the graded plateau's center/slope are calibrated on the ignition CURVE (controlled fractional/solo
drives, content-independent — a detector PROPERTY, exactly like the W4 recal calibration), by the
magnitude-preservation objective: solo-silent, monotonic, minimal proportionality error. The instrument is CLEAN
6/6: mean max_solo=0.004 (silent), all monotonic, mean proportionality error 0.048, mean **sat_ratio
r(0.27)/r(1)=0.24** (the all-or-none read saturates to ~1.0). Then the exact W4 onehot-vs-graded A/B is re-run
through this read.

| M1 (intent-averaged magnitude-fidelity), 6-seed | onehot | graded | move | seeds graded>onehot | scramble |
|---|---|---|---|---|---|
| **GRADED plateau read** | **0.888** | 0.807 | **−0.082** | **0/6** | 0.192 (loses → M1 valid) |
| **ALL-OR-NONE read (control)** | 0.653 | 0.618 | −0.035 | 0/6 | — |

The all-or-none control reproduces the W4 finding's default move (−0.035) EXACTLY. The graded read makes graded
LOSE by MORE (−0.082), not less. Per-intent (graded read) shows WHY — the analytic RSA landscape is mostly
one-hot (only the single cell S[all|some]=0.25 is graded), so the graded belief's off-diagonal mass HURTS on the
CLEAN intents while barely helping on the one implicature intent:

| per-intent M1 (graded read), 6-seed | intent=none | intent=SBNA | intent=all (the implicature) |
|---|---|---|---|
| onehot | **0.931** | 0.888 | 0.846 |
| graded | 0.708 | 0.878 | 0.835 |

Graded loses BIG on intent=none (0.708 vs 0.931) and does not even WIN on intent=all (0.835 vs 0.846, a tie). The
implicature cell M2 S[all|some] reads graded=0.316 vs onehot=0.076 (the read DOES pick up the graded mass, and it
moves toward the analytic 0.20 from the all-or-none recal's 0.360), BUT M2 remains CHEATABLE (scramble 0.662 >
graded 0.316), so M2 is not a valid surpass metric — the pre-registered VALID metric is M1, and it does not move.

**Diagnosis (airtight).** The graded read IMPROVES absolute fidelity for BOTH beliefs (onehot 0.653→0.888, graded
0.618→0.807) — the detector's magnitude-blindness IS surpassed at the read level. But the gap WIDENS, because the
W4 residual was NOT the detector: it is the intent-averaged M1 / the graded belief's cross-intent spread. The
analytic RSA landscape carries graded structure in ONE cell; a faithful magnitude-preserving read of the graded
belief reproduces its spurious off-diagonal mass on the two one-hot intents, which the objective averages in and
penalizes. Even on the implicature intent, the neural landscape's diagonal (full-mass) cell does not dominate the
off-diagonal enough after row-normalization (M2 overshoots 0.20). The residual is RELOCATED entirely onto the
OBJECTIVE / aggregation — the W4 finding's own named residual part (b), now ISOLATED as the sole remaining term.

## Part B (affect) — the read-out is NOT the bottleneck; the WEIGHT SOURCE is (the 4-cell design proves it)

<!--derived-->

The same graded plateau is applied to the affect opponent `code_in->appr_vplus/appr_vminus` FF read, and the C-A2
salience `|differential|~valence-strength` correlation is re-measured on a 4-cell design (weights × read-out) that
ISOLATES the read-out from the weight source. The graded center/slope are calibrated per seed to MAXIMIZE the
graded salience (a GENEROUS test — the best shot).

| C-A2 salience_r (6-seed) | point-soma read | graded plateau read |
|---|---|---|
| **composed** self-organized weights | 0.081 (reproduces the ~0.10 boundary) | 0.096 (+0.015, 3/6) |
| **ridge**-to-Warriner weights | **0.327** (already ≥ the 0.27 target) | −0.048 (graded HURTS it) |

The ridge (magnitude-supervised) weights already reach 0.327 with the POINT-SOMA read — so the point-soma read is
magnitude-preserving ENOUGH; if the read-out were the bottleneck, the ridge would be limited too. The graded read
does NOT lift the composed opponent (+0.015 is noise, 3/6) and DESTROYS the ridge (−0.048). The affect boundary is
the WEIGHT SOURCE: the Rescorla-Wagner asymptote `s_c = (n_pos−n_neg)/(n_pos+n_neg)` SATURATES (encodes sign
robustly, magnitude weakly), so the composed weight carries weak magnitude no read can recover. This is exactly
the affect boundary's own diagnosis, now CONFIRMED by the read-out being ruled out.

## The unified conclusion — one wall, two DIFFERENT causes; a single read-out fix was the wrong hypothesis

<!--derived-->

Both boundaries present as "reads magnitude weakly", and the graded dendritic-plateau read-out was a principled,
GO'd, biologically-grounded candidate to fix BOTH. It does not — and the 6-seed A/B says precisely WHY, and the
answers DIFFER:

- **W4 (D-pragmatics)** is a METRIC / OBJECTIVE mis-specification. The magnitude-preserving read surpasses the
  detector's magnitude-blindness (verified) yet M1 does not move, because the intent-averaged fidelity dilutes the
  single implicature-carrying cell and penalizes the graded belief's off-diagonal mass on the two clean intents.
  **Next mechanism (already named):** an implicature-localized / RSA-informativeness-weighted pragmatic-alignment
  objective (Frank & Goodman, 2012 informativeness), NOT a read-out.
- **Affect (A)** is a WEIGHT-SOURCE saturation. The point-soma read already suffices (the ridge proves 0.327), so
  the read-out is not the lever. **Next mechanism (already named):** a GRADED reinforcement-STRENGTH third factor
  — a dopamine-ramp amplitude that scales with reinforcement intensity (Bayer & Glimcher, 2005) — so `s_c` becomes
  a graded associative strength instead of the saturating RW ratio, NOT a read-out.

The value banked: the graded dendritic-plateau read-out is verified to WORK as a magnitude-preserving read (a
clean instrument, 6/6), and is thereby RULED OUT as the fix for both residuals — which relocates each to its true,
distinct, already-named next lever. Neither residual is a credit-assignment problem; the refuted deep-credit / BDSP
rule is NOT re-proposed.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **Magnitude-preserving, VERIFIED (Part A):** the graded read's response curve grades MONOTONICALLY and
  near-proportionally with the coincident input (mean prop-error 0.048; r(0.27)/r(1)=0.24), 6/6, printed per seed
  — vs the all-or-none read which saturates (r(0.27)/r(1)≈1.0). The read genuinely preserves magnitude; the null
  result is therefore about the TASK metric, not a broken read.
- **VALID metric only (Part A):** M1's SCRAMBLE control (graded mass on WRONG intents) LOSES to onehot on the
  graded read (0.192 << 0.888) — so graded≤onehot is a REAL negative, not an instrument failure. M2 is NOT
  reported as a win (its scramble control fails: 0.662 > 0.316).
- **The all-or-none control reproduces the negative (Part A):** move −0.035 (the W4 finding's exact default),
  confirming the graded read specifically CHANGES the read while the wall persists.
- **The point-soma control reproduces the boundary (Part B):** composed point-soma 0.081 (≈ the affect finding's
  0.10), so the graded arm's failure to lift is measured against a faithful reproduction.
- **Read-out isolation (Part B):** the ridge+point-soma cell (0.327 ≥ 0.27) proves the read-out is not the
  bottleneck — the load-bearing control that assigns the boundary to the weights.
- **6 seeds 42/43/44/100/101/102** (smoke first; the 6-seed is authoritative — a single-seed smoke was consistent
  here but the project bar is 6).

## Honest scope

<!--derived-->

FUNCTIONAL pragmatics + affect correlates. Both tests change ONLY the read-out (the graded dendritic plateau vs
the all-or-none switch / the point-soma spike rate), beliefs + weights byte-comparable to the boundary runners;
plasticity off (fixed operating point). Per-seed center/slope calibrated on the ignition CURVE / the opponent's
own drive range — a detector PROPERTY (Part A content-independent; Part B calibrated to MAXIMIZE the graded
salience, a generous test). numpy-CPU real spiking Izhikevich bridges; NO `sim/` edit
(`enable_graded_dendritic_plateau` via config); additive NEW runner
(`research/runners/_magnitude_preserving_plateau_readout_derisk.py`), reuse-by-import of the W4 A/B + the Leg-1
detector + the affect-deepen circuit + the composed self-organized opponent. Warriner is EVAL-only ground-truth.
NOT a claim of phenomenal access to another mind / phenomenal experience; a self-report would be a functional
read-out.

## Sources

- Mikulasch & Priesemann — dendritic ANALOG / graded read-out (the graded plateau is the point neuron's
  magnitude-blindness surpass; the mechanism GO'd in `2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`).
- Larkum (2013), Trends in Neurosciences 36(3):141 — "A cellular mechanism for cortical associations" (the
  plateau's tunable coinciding-input threshold; the NMDA-spike coincidence framing).
- Frank & Goodman (2012), Science 336(6084):998 — "Predicting Pragmatic Reasoning in Language Games" (the RSA
  objective is a graded MAGNITUDE / informativeness, not an argmax — the W4 next lever is an informativeness
  weighting of the objective).
- Bayer & Glimcher (2005), Neuron — dopamine neurons encode a quantitative (graded-magnitude) reward-prediction
  error (the affect next lever: a graded reinforcement-strength third factor).
- Rescorla & Wagner (1972) — the associative-strength ASYMPTOTE (why the composed `s_c` encodes sign > strength;
  the affect weight-source saturation).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part A \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_magnitude_preserving/w4_6seed.json
SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part B \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_magnitude_preserving/affect_6seed.json
```
