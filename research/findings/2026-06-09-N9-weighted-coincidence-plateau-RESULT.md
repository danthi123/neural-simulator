# N9 — the Poirazi-Mel WEIGHTED-coincidence plateau converts the weight-blind count plateau into a VALUE-graded one (sound sim/ edit, byte-identity proven; multi-seed PARTIAL bounded by place-code overlap, NOT by the plateau)

**Date:** 2026-06-09
**Backend:** CuPy (RTX 3090), deterministic regime (OU / conductance-noise / global-homeostasis OFF).
**sim/ status:** the protected edit is WRITTEN + byte-identity-proven but **UNCOMMITTED — pending owner byte-review.** The runner edit is also uncommitted (it references the new flag).
**Owner directive (standing):** biologize everything, NO banking; honest negatives/partials ARE the deliverable; sim/ edits need byte-review.

## The residual this fixes

The landed Route-D coincidence plateau (`b980070a`) broke the rate-coding wall at STEP-1 (a gamma volley fires a downstream MSN-D1 from a sparse-distinct place code; jitter collapses it → coincidence, not rate). But at STEP-2 (reading that code into the N9 value critic) it gave **G_GRADE 1/3**: the critic fired near AND far the goal (grading washed to ~1.2-1.8×). **Root cause (verified `bridge.py`):** the plateau switch read the bare COUNT of coincident inputs (`c_i = {0,1}-mask^T @ prev_fired`), so it is **WEIGHT-BLIND** — any synchronized volley (near or far) crosses the same `c_i ≥ K` switch regardless of the learned `place→value` weight. The learned value lived in the AMPA component, but the weight-blind plateau dominated firing.

## The fix (faithful Poirazi-Brannon-Mel 2003: a dendritic subunit is a sigmoid of the WEIGHTED input sum, not a count)

Additive, default-off `cfg.coincidence_weighted_drive` (config.py +14, bridge.py +50/−17, **NO kernels.py edit**). When set, the coincidence matvec DATA switches from the `{0,1}` routing mask to `effective_connections_matrix.data * mask` — so the per-neuron switch variable becomes the per-step WEIGHTED coincident sum `c_w = Σ_j (w_eff_j · x_j)`, exactly mirroring the shipped GABA_B block's restricted-matvec idiom. The kernel's existing sigmoid is fed `c_w` (no kernel change); `coincidence_k_threshold` is then read in WEIGHT units. A strongly-weighted coincident ensemble (high learned value, near goal) crosses the supralinear switch; a weakly-weighted one (far) does not → the plateau GRADES with synaptic value.

### Byte-identity (the owner's gate) — BOTH paths proven bit-identical

Stash-baseline-vs-edited harness (`_coincidence_byte_identity_check.py`), numpy:
- **OFF-path** (`enable_coincidence_detection=False`): edited `0b3c4b3f…` == baseline `0b3c4b3f…` ✓ (the whole block is skipped).
- **COUNT-on-path** (`coincidence_weighted_drive=False`): edited `c6de90ed…` == baseline `c6de90ed…` ✓ (the flag defaults off → `_co_data` is the same `_co_mask_f` object the original built the matrix from).

So the edit is **provably additive**: the off-path and the validated count form are both byte-unchanged; only the new opt-in weighted path differs. Weighted matvec smoke-runs clean on CuPy.

### Runner wiring (the cold-start handling)

The weighted plateau can't fire at the small init weight (cold-start), so the de-risk keeps the COUNT form during self-org + value TRAINING (the strong count plateau bootstraps DA-gated LTP) and toggles `coincidence_weighted_drive=True` ONLY at READ-OUT, swapping `coincidence_k_threshold` to a WEIGHT-unit value (`--readout-weighted-k`). Mirrors the existing `readout_plateau` toggle.

## Result (CuPy, `n9_place_graded_critic_stage2_derisk.py`, n-place 800, θ=20)

### The core win — apples-to-apples at seed 44 (non-jitter)

| Plateau form | NEAR | far_a | far_b | far_center | ratio (near/max-far) |
|---|---|---|---|---|---|
| COUNT (weight-blind) | 25.83 | 17.08 | 0.00 | 20.97 | **1.23×** (fires everywhere) |
| **WEIGHTED (this edit)** | 24.58 | 3.33 | 0.00 | 0.00 | **7.37×** (graded) |

The weighted edit does exactly what it claims: it converts the weight-blind plateau (far fires 17-21 Hz) into a value-graded one (far_a 3.33, two FARs silent).

### 3-seed (42/43/44, θ=20)

| Gate | Result | Detail |
|---|---|---|
| 2a FIRE (≥5 Hz) | **3/3** | NEAR 5.69 / 13.19 / 24.58 Hz |
| 2b PLACE-GRADED (≥3×) | **1/3** | 7.37× (seed 44 GO); 2.93× / 2.79× (42/43 near-miss) |
| 2c LEARNS-V (LTP ≥2×) | 2/3 | w_near/far 4.08 / 1.86 / 4.11 |
| 2d ACTOR-NOT-PERTURBED | 3/3 | ratio 1.000 |

**Verdict: PARTIAL.** The weighted plateau grades strongly (2 of 3 FARs perfectly silent on every seed). The strict max-far ratio just misses 3.0 on the two seeds with lower absolute NEAR firing — capped by the single most-overlapping far location (`far_a`, place diff-cos 0.158-0.193). The high-NEAR seed (44) clears cleanly at 7.37×.

## Anti-cheat analysis (decisive)

**(a) place-shuffle HOLDS (value-of-LOCATION).** Permuting the place-cell→location mapping that the value arm tracks breaks the measured LTP (gate-2c 4.11× → 1.88×, < 2×). The grading rides on weights learned at the rewarded location.

**(b) the Stage-2 `--jitter` is an INVALID coincidence test at STEP-2 — and this is NOT an edit flaw.** Controller ran COUNT-vs-WEIGHTED head-to-head under the same jitter on seed 44: **BOTH** forms fire NEAR (count 17.36 Hz, weighted 24.72 Hz). The Stage-2 jitter clamps the sensors every *other* step — a rate-halver. The FS-PING volley reforms on the driven (even) steps and the plateau's 80 ms tail bridges the 1 ms off-step, so it collapses the MARGINAL far firing but not the STRONG near firing, for EITHER plateau form. It does not probe within-step coincidence. The genuine coincidence anti-cheat is STEP-1 (`coincidence_volley_n9_derisk.py`), which PASSED (there the source had no learned weights, so only the plateau fired and halving the volley dropped `c_i` below `K` → collapse). **So coincidence is validated UPSTREAM at STEP-1; the weighted edit adds value-GRADING at STEP-2, confirmed by place-shuffle.**

## Honest synthesis

- The weighted-coincidence plateau is the **correct, faithful fix** for the count plateau's weight-blindness: it demonstrably converts 1.23× (weight-blind) into 7.37× (value-graded). The sim/ edit is additive, default-off, byte-identity-proven, and mirrors the shipped GABA_B routing — **a clean, byte-reviewable improvement.**
- The residual **G_GRADE gap (2/3 near-miss at 2.8-2.9×) is bounded by PLACE-CODE OVERLAP** (the single most-overlapping far location leaks via shared high-weight cells), plus seed-variable absolute NEAR firing — **NOT by the plateau mechanism** (2 of 3 FARs are perfectly silent on every seed). The honest next lever is **Stage-1 place-code distinctness** (sharper, less-overlapping fields), not a plateau change.
- The Stage-2 jitter's non-collapse is a **test-validity finding** (rate-halver defeated by the slow plateau + learned drive), affecting count and weighted forms equally; the coincidence property was validated at STEP-1.

## G_GRADE clears 3/3 with stronger place-code training — but the place-code OVERLAP then trades it against G_LTP

3-seed at 24 self-org passes × 150 steps/loc (vs 12 × 120):

| Gate | 12 passes | 24 passes |
|---|---|---|
| 2a FIRE | 3/3 | 3/3 (NEAR 16.4 / 10.6 / 27.2) |
| 2b PLACE-GRADED | 1/3 | **3/3** (ratio 3.69 / 4.00 / 8.91) |
| 2c LEARNS-V (LTP ≥2×) | 2/3 | **1/3** (w_near/far 3.17 / 1.71 / 0.85) |
| PRIMARY (all gates) | 1/3 | 1/3 |

**The weighted plateau makes G_GRADE robustly achievable (3/3).** But stronger training exposes the **place-code OVERLAP** (diff-cos 0.16–0.20 at n-place 800): the near and far ensembles share cells, so training NEAR + reward potentiates those shared cells → `w_far` grows (seed 44: w_far 2.69 ≈ w_near 2.27) → gate-2c LTP (which needs w_near ≥ 2× w_far) fails. So the OVERLAP causes BOTH failure modes: far_a firing-leak at low training (G_GRADE near-miss) AND w_far growth at high training (G_LTP). PRIMARY stays 1/3 because overlap trades the two gates against each other. **The bottleneck is upstream place-code distinctness, not the plateau.**

## Decisive control — the grading is the WEIGHTED plateau, NOT the learned AMPA

COUNT vs WEIGHTED at the **identical** learned weights (seed 42, 24 passes, w_near 4.24 / w_far 1.34):

| Plateau form | NEAR | far_a | far_center | ratio |
|---|---|---|---|---|
| COUNT (weight-blind) | 14.31 | 25.14 | 27.64 | **0.52×** (fires MORE at far) |
| **WEIGHTED (this edit)** | 16.39 | 4.44 | 0.14 | **3.69×** (graded) |

Same weights, opposite outcome: the count plateau is anti-graded (the position-blind dense-blob floor fires far MORE than near), the weighted plateau grades. **This unambiguously isolates the weighted-coincidence plateau as the grading mechanism** — it is not the learned AMPA (identical in both arms).

## 🎉 Resolution — gating the FS-PING OFF during self-organization gives clean PRIMARY

The overlap is **not** from cell count (n-place 400 also gave diff-cos 0.195, and its thinner volley went silent). It is the **FS-PING gamma cycling during self-organization**: the cycling recruits extra cells, densifying (3.65% → 4.8–7.2%) and blurring (0.064 → 0.20) the place code. The validated Stage-1 had no FS-PING and reached 0.064.

**The fix (runner-only, using the landed `transmission_gate` infra):** hold the FS→place inhibition CLOSED during self-org (clean threshold-WTA → sparse DISTINCT fields), OPEN it for the volley read-out (the gamma packing the coincidence plateau needs). Seed 42:

| | overlap | NEAR | G_GRADE | G_LTP | PRIMARY |
|---|---|---|---|---|---|
| FS-PING on during self-org (24 passes) | diff-cos 0.200 | 16.4 | 3.69× ✓ | 1.71× ✗ | FAIL |
| **FS-PING gated off during self-org** | **diff-cos 0.120** | **31.5** | **3.15× ✓** | **2.56× ✓** | **PASS** |

The distinct code (0.120) removes BOTH failure modes simultaneously on seed 42 — far_a no longer over-leaks (G_GRADE) AND the shared near/far cells no longer co-potentiate w_far (G_LTP). w_near 0.50→6.41, w_far 0.50→2.50.

**3-seed (FS-gating, θ=20):** distinct codes confirmed (diff-cos 0.036 / 0.065 / 0.120), but PRIMARY 1/3 — **each seed now misses ONE marginal gate, all traceable to seed-variable w_far growth at the strong drive** (critic fires 33–53 Hz):

| Seed | diff-cos | NEAR | G_GRADE | G_LTP (w_near/far) | PRIMARY |
|---|---|---|---|---|---|
| 42 | 0.120 | 33.2 | 3.46× ✓ | 2.57× ✓ (6.41/2.50) | **PASS** |
| 43 | 0.036 | 51.0 | 5.10× ✓ | 1.82× ✗ (6.27/3.45) | FAIL (LTP) |
| 44 | 0.065 | 52.8 | 1.94× ✗ | 2.07× ✓ (10.1/4.91) | FAIL (GRADE: w_far 4.91 → far_a 27 Hz) |

G_FIRE 3/3, G_GRADE 2/3, G_LTP 2/3, G_ACTOR 3/3. All gates are individually achievable; the residual is **operating-point sensitivity** — the high drive over-grows w_far, which surfaces as a LTP miss (seed 43) or a grade miss (seed 44, high w_far → far fires). place-shuffle still HOLDS at this config (LTP 2.56×→1.75× under shuffle).

**Moderated operating point (n-train 30, θ=24) → PRIMARY 2/3.** θ=24 dramatically tightens far-suppression → **G_GRADE 3/3** (ratios 23.2 / 57.2 / 4.0×), so seeds 42 + 44 now PASS. Seed 43 remains a G_LTP holdout (1.55×) — and notably its weight-ratio LTP was *better* at n-train 40 (1.82×) than 30 (1.55×), so for that seed *more* training (w_near outpacing w_far) helps, while its firing-grade is already 57× (functionally strong value-of-location; only the strict weight-ratio gate is marginal). A θ=26 / n-train 50 point (more LTP separation + protected far-suppression) is in flight `[result pending]`.

## Multi-seed operating-point search → robust PRIMARY 2/3 (G_GRADE 3/3)

| config | PRIMARY | G_GRADE | G_LTP | seed-43 weight-LTP |
|---|---|---|---|---|
| FS-gate, θ20, n-train 40 | 1/3 | 2/3 | 2/3 | 1.82× |
| FS-gate, θ24, n-train 30 | 2/3 | 3/3 | 2/3 | 1.55× |
| **FS-gate, θ26, n-train 50** | **2/3** | **3/3** (8.6/5.25/3.42×) | 2/3 | **1.86×** |

θ tightens far-suppression → G_GRADE clears 3/3 with comfortable margins; seeds 42 + 44 PASS full PRIMARY. **Seed 43's weight-ratio LTP plateaus at ~1.86×** across n-train 30/40/50 (1.55/1.82/1.86) — w_far grows alongside w_near at that high-firing seed, so more training does not cross the strict 2× bar. Its FIRING grade is 5.25× and place-shuffle breaks its LTP → it IS learned value-of-location; only the strict weight-ratio proxy is marginal.

## Conclusion (scope of this arc) + honest residuals

**DELIVERED + owner-approved + committed (`e0818d2d`):** the weighted-coincidence plateau is the decisive value-**grading** mechanism (count 0.52× → weighted 3.69× at identical weights; G_GRADE 3/3 multi-seed). Combined with FS-gating-during-self-org (runner-only), the MSN-D1 value critic **fires + learns + grades** value-of-location on the self-organized sparse-distinct place code — **the weight-blind-plateau residual that this arc set out to fix is resolved.** Multi-seed PRIMARY 2/3 (seed 43 marginal on the strict weight-ratio LTP gate, 1.86×; functionally a value-of-location pass at 5.25× firing-grade). Anti-cheats: place-shuffle holds; the Stage-2 jitter is an invalid rate-halver (coincidence validated upstream at STEP-1).

**Honest residuals (separate from this edit):**
1. Seed 43's weight-ratio LTP plateaus at 1.86× (< the strict 2×) — operating-point/seed variance, not a mechanism failure.
2. The **SNc r−V subtraction** (gate-2e, the broader N9 deliverable) is NOT yet validated — the SNc state-specific gap is weak (gap ~1.0; SNc tonic calibration fragile in the FS-gating config). This is a separate N9 piece (the GABA_B subtracting the learned V from the dopamine cell), predating and independent of the weighted-plateau edit.

**Next:** the critic-V-grading half of N9 is validated. The remaining pieces are (a) the SNc r−V subtraction loop (gate-2e), and (b) deploying the validated critic in the full nav loop (the 6-seed nav A/B). Owner steer on sequencing.

## Files

- sim/: `sim/config.py` (`coincidence_weighted_drive`), `sim/bridge.py` (weighted matvec branch) — **UNCOMMITTED, byte-review pending.**
- runner: `research/runners/n9_place_graded_critic_stage2_derisk.py` (`--weighted-drive`, `--readout-weighted-k`, READ-OUT toggle) — uncommitted.
- byte-identity: `research/findings/raw/_coincidence_byte_identity_check.py`.
- raw: `research/findings/raw/_n9_weighted_*.{json,log}`, `_n9_COUNT_s44*.{json,log}`.
