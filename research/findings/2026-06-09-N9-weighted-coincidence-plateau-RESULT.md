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

## Recommendation

The weighted-coincidence sim/ edit is **sound, additive, byte-identity-proven, and the decisive cause of value-grading** (count 0.52× → weighted 3.69× at identical weights; G_GRADE 3/3 achievable). Commit it on owner byte-review. The residual full-PRIMARY gap is **upstream place-code distinctness** (n-place 800 gives diff-cos 0.16–0.20; the validated Stage-1 reached 0.064 at n-place 400) — reducing the near/far ensemble overlap is the honest next lever (it removes BOTH the far-leak and the w_far-growth failure modes simultaneously), a Stage-1 perception-quality sub-arc, NOT a plateau change. N9 value-of-location is substantially advanced: the critic fires + learns + GRADES on the sparse-distinct place code; the residual is cleanly localized to place-code overlap.

## Files

- sim/: `sim/config.py` (`coincidence_weighted_drive`), `sim/bridge.py` (weighted matvec branch) — **UNCOMMITTED, byte-review pending.**
- runner: `research/runners/n9_place_graded_critic_stage2_derisk.py` (`--weighted-drive`, `--readout-weighted-k`, READ-OUT toggle) — uncommitted.
- byte-identity: `research/findings/raw/_coincidence_byte_identity_check.py`.
- raw: `research/findings/raw/_n9_weighted_*.{json,log}`, `_n9_COUNT_s44*.{json,log}`.
