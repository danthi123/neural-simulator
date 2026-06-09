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

## Stronger place-code TRAINING clears the near-miss (seed 42: 2.93× → 3.69× PASS)

Re-running seed 42 with more self-organization (24 passes × 150 steps/loc vs 12 × 120) flips it to **G_GRADE PASS (3.69×)**. Notably the place diff-cos got slightly *worse* (0.200 vs 0.158) — so it is **NOT** reduced overlap. It is **stronger absolute NEAR firing**: more place-code exposure → the value arm sees a more robust ensemble → w_near grows 0.50→4.24 (was 2.68) → NEAR 16.39 Hz (was 5.69) → it pulls away from far_a (4.44) despite the overlap. So the near-miss seeds are limited by **place-code drive STRENGTH** (→ NEAR firing), a legitimate training-quality / experience lever (more experience → stronger place fields), not by the plateau. Multi-seed validation at 24 passes is in flight `[3-seed result pending]`.

## Recommendation

Commit the byte-reviewed weighted-coincidence edit (sound, additive, byte-identity-proven, a real grading improvement: count 1.23× → weighted 7.37×) on owner approval. The residual G_GRADE near-miss on the lower-NEAR-firing seeds is a Stage-1 place-code-strength matter (more self-org → stronger NEAR firing clears it, as seed 42 shows), NOT a plateau issue. The N9 value-grading is substantially advanced: the value critic now fires + learns + GRADES on the sparse-distinct place code, with the residual cleanly localized to upstream place-code drive strength.

## Files

- sim/: `sim/config.py` (`coincidence_weighted_drive`), `sim/bridge.py` (weighted matvec branch) — **UNCOMMITTED, byte-review pending.**
- runner: `research/runners/n9_place_graded_critic_stage2_derisk.py` (`--weighted-drive`, `--readout-weighted-k`, READ-OUT toggle) — uncommitted.
- byte-identity: `research/findings/raw/_coincidence_byte_identity_check.py`.
- raw: `research/findings/raw/_n9_weighted_*.{json,log}`, `_n9_COUNT_s44*.{json,log}`.
