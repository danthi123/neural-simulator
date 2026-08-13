---
type: finding
status: qualified
claim_check: synthesis
verdict: >
  GO (6-seed, 6/6). COMPOSING the mouth's two substrate graded reads END-TO-END and BIOLOGIZING the base-rate prior
  head_b. The output projection Wo_sp@state (signed graded-conductance read, corr 0.984) — gated by the host r_h —
  becomes the read-out head_w@h's INPUT feature (also a signed graded-conductance read), so the state->logits chain is
  ONE substrate signed-graded pipeline (every matmul a cp_conductance_g_e/g_i read, 0 host matmul on the margin); AND
  head_b is moved off host arithmetic (parity-close's `margin + s*head_b`) onto a TONIC BIAS-INPUT POPULATION (matched
  bias_e/bias_i firing tonically onto the word-pools, per-pool weights ~ head_b, same driving-force ratio the feature
  read uses) so the base-rate prior is a genuine synaptic conductance the pools carry. HEADLINE (6-seed, seed-42
  calibrated, 5 UNSEEN): composed_biaspop recov_argmax 0.9495 mean / 0.9309 min, argmax_agree 0.769, 6/6 GO —
  NEAR-parity with the isolated reads (readout_hostb 0.978 == parity-close; readout_biaspop 0.9757), a small MEASURED
  composition penalty (recov -0.026, argmax_agree -0.07 vs the isolated read: the projection stage's 0.984
  reconstruction amplified at near-ties), NOT a collapse. head_b as a SPIKING SYNAPSE ~ host arithmetic (readout_biaspop
  0.9757 vs readout_hostb 0.978, -0.0023; both 6/6 GO). The tonic bias population is LOAD-BEARING on the clean same-
  position arm A/B: composed_biaspop 0.9495 vs head_b-off composed_nobias 0.9084 = +0.042, positive on all 6 seeds.
  Anti-cheats 6/6: scramble->chance, zero-STATE-input->chance (cache-immune, drives the whole composed chain),
  zero-feature->chance, 0 host draws, silent 0.0; the read-out inhibitory shadow (NEGATIVE weights) LOAD-BEARING 6/6.
  NOT "fully spiking" / NOT production-wired: the WKV recurrent STATE + r_h + LN are still host, the projection/read-out/
  head_b weights host-designed; a single seed-42-calibrated scalar (proj_out_scale) maps the substrate margin's
  arbitrary conductance units to the read-out's feature scale. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1
date: 2026-08-13
mechanism: >
  Two chained substrate signed graded-conductance reads + head_b as a spiking tonic-bias synapse. STAGE 1 (projection,
  its own bridge): the WKV state [ap,an] drives matched carriers; Wo_pos/Wo_neg (Dale-split Wo_sp) wire them onto D
  hidden pools; hpre_sub = df_e*g_e + df_i*g_i off cp_conductance reconstructs Wo_sp@state (corr 0.984). Host glue: h =
  r_h * (proj_out_scale * hpre_sub), dual-nonneg [h+,h-]. STAGE 2 (read-out bridge): [h+,h-] drives hid/hidinh
  (hid_pop=4); Wp/Wn (head_w Dale-split) wire them onto V word-pools; margin = df_e*g_e + df_i*g_i. STAGE 3 (head_b,
  same read-out bridge): bias_e (head_b>0, EXCITATORY onto g_e) + bias_i (head_b<0, INHIBITORY onto g_i, same ratio)
  fire tonically onto the pools, so the margin PICKS UP head_b as a per-pool synaptic conductance (0 host arithmetic on
  the margin). Winner = argmax over the substrate net-current margin. bias_scale (0.14) + proj_out_scale (0.30)
  calibrated ONCE on seed 42 (wide plateaus). Runner-only, default-off, NO sim/ edit.
artifacts:
  - research/runners/_wkv_mouth_endtoend_substrate_read_derisk.py
  - research/findings/raw/_wkv_endtoend_substrate_read_6seed.json
---

# gap#1 / A1 — composing the mouth's two substrate graded reads END-TO-END + head_b as a spiking synapse (GO, 6-seed 6/6)

## What was isolated, and what this composes

Two mouth matmuls became substrate signed graded-conductance reads, EACH validated with the other's input taken
host-side:
- **output projection** `h_pre = Wo_sp @ state` (`2026-08-13-fluid-mouth-upstream-output-projection-GO`, corr 0.984
  6/6) — reconstructs the hidden feature FROM the WKV state, but its downstream `head_w@h` read was applied HOST-side
  ("the end-to-end chain was not run");
- **read-out** `logits = head_w @ h` (`2026-08-13-fluid-mouth-graded-conductance-read-GO`, recov_argmax 0.921; closed
  to 0.978 by `2026-08-13-fluid-mouth-read-parity-close-GO`) — reads the winner word-pool from the net-current margin,
  but takes the hidden feature `h = r_h*(Wo_sp@state)` HOST-side and injects `head_b` in HOST ARITHMETIC (`margin +
  s*head_b`; parity-close's residual #2 named the fix: "wire it as a tonic bias-input population").

This lane does the two named compositions (wiring, NO new mechanism): (A) CHAIN the reads — the projection's substrate
graded margin (gated by the host r_h) becomes the read-out's input feature, so the state->logits path is ONE substrate
signed-graded pipeline, every matmul a `cp_conductance_g_e/g_i` read; (B) BIOLOGIZE head_b — a TONIC BIAS-INPUT
POPULATION replaces the host arithmetic, so the base rate is a genuine synaptic conductance the pools carry.

## The mechanism (three substrate stages, one host glue)

STAGE 1 — projection (its own bridge): the nonneg WKV state `[ap,an]` drives matched carriers; `Wo_pos`/`Wo_neg`
(Dale-split `Wo_sp`) wire them onto D hidden pools; `hpre_sub = df_e*g_e + df_i*g_i` off `cp_conductance` reconstructs
`Wo_sp@state` (corr 0.984). HOST GLUE — the r_h receptance gate (a named upstream residual; shunting is its next rung)
and a SINGLE seed-42-calibrated scalar `proj_out_scale=0.30` that maps the projection margin's arbitrary conductance
units (RMS ~3.3x the host feature) to the read-out's validated feature scale: `h = r_h*(proj_out_scale*hpre_sub)`,
dual-nonneg `[h+,h-]`. STAGE 2 — read-out (its own bridge): `[h+,h-]` drives `hid`/`hidinh` (hid_pop=4, the
parity-close population density); `Wp`/`Wn` (head_w Dale-split) wire them onto V word-pools; `margin = df_e*g_e +
df_i*g_i`. STAGE 3 — head_b (same read-out bridge): `bias_e` (head_b>0, EXCITATORY onto `g_e`) and `bias_i` (head_b<0,
INHIBITORY onto `g_i`, with the SAME driving-force `ratio` the feature read uses) fire tonically (constant drive) onto
the pools, so the margin PICKS UP `head_b` as a per-pool synaptic conductance — 0 host arithmetic on the margin. Winner
= argmax over the substrate net-current margin.

## RESULT — 6-seed A/B (42/43/44/100/101/102; V=1000; P=4; hid_pop=4; n_eval=200; GPU cupy; 1418s)

<!--derived: research/findings/raw/_wkv_endtoend_substrate_read_6seed.json summary (mean over 6 seeds)-->

| arm | feature | head_b | recov_argmax (mean / min) | argmax_agree | signed LB | GO |
|---|---|---|---|---|---|---|
| readout_hostb   | HOST h        | HOST arithmetic | 0.9780 / 0.9663 | 0.852 | 6/6 | 6/6 |
| readout_biaspop | HOST h        | TONIC BIAS POP  | 0.9757 / 0.9670 | 0.840 | 6/6 | 6/6 |
| composed_nobias | SUBSTRATE proj| off             | 0.9084 / 0.8684 | 0.684 | 6/6 | 1/6 (diagnostic) |
| **composed_biaspop** | **SUBSTRATE proj** | **TONIC BIAS POP** | **0.9495 / 0.9309** | **0.769** | **6/6** | **6/6** |

Per-seed composed_biaspop recov_argmax: 42=0.965, 43=0.933, 44=0.957, 100=0.959, 101=0.952, 102=0.931 (6/6 >= 0.93).

**Three findings, each measured, not assumed:**

1. **head_b as a SPIKING SYNAPSE ~ host arithmetic.** With the SAME host feature, moving head_b from the parity-close
   host `margin + s*head_b` onto a tonic bias-input population (readout_hostb 0.9780 -> readout_biaspop 0.9757, a
   -0.0023 mean recov, argmax_agree 0.852 -> 0.840) is a near-null change: the base-rate prior injected as a real
   per-pool synaptic conductance reproduces the host arithmetic. The readout_hostb arm REPRODUCES the parity-close
   deliverable exactly (0.978 mean, matching that finding's 0.9775). head_b is now a spiking synapse, 6/6 GO.

2. **The composed end-to-end chain HOLDS NEAR-PARITY.** Feeding the read-out from the SUBSTRATE projection instead of
   the host feature (composed_biaspop 0.9495 / min 0.9309, 6/6 GO) sits a MEASURED -0.026 mean recov below the isolated
   readout_biaspop (0.9757); argmax_agree 0.769 vs 0.840 (-0.07). This is the honest COST of stacking two substrate
   reads: the projection stage reconstructs `Wo_sp@state` at corr 0.984, and the ~1.6% error is amplified at the peaked,
   near-tied next-word distribution — exactly what the output-projection finding measured as its downstream
   argmax_agree ~0.77 through a HOST read (the substrate read-out here matches). The chain does NOT collapse under
   composition; it holds at recov ~0.95 with the state->logits path fully on the substrate.

3. **The tonic bias population is LOAD-BEARING.** On the clean same-position arm A/B, composed_biaspop (0.9495) beats the
   head_b-off composed_nobias (0.9084) by +0.042 mean, POSITIVE on ALL 6 seeds (range +0.028 to +0.063) — the base-rate
   prior lift is real and generalises, matching parity-close's host-arithmetic base-rate lift (+0.040). (The within-run
   `recov_biassilence` control — drop the bias-pop DRIVE to 0 in place — is directionally consistent on average
   (mean 0.9245 < intact 0.9495) but NOISY per-seed (0.83-0.99), because it is measured on a position SUBSET against the
   full-set argmax ceiling; the arm A/B is the clean instrument and is what the load-bearing claim rests on.)

The read-out inhibitory shadow (NEGATIVE `head_w` weights) stays LOAD-BEARING 6/6 on every arm (signed argmax-agree >
positive-only on identical conductances; the base-rate term is added to BOTH margins, so this isolates the SIGN).

## Anti-cheats (6-seed, composed_biaspop)

- **Scramble -> chance:** post-hoc pool->word relabel collapses argmax-agreement to 0 on every seed.
- **Zero-STATE-input (cache-immune):** silencing the projection INPUT (`state=0`) — the head of the WHOLE composed
  chain — drops argmax-agreement to ~0 (0.0-0.06); the state drives the end-to-end read.
- **Zero-feature -> chance** (silence the read-out INPUT): argmax-agreement to ~0.
- **Provenance:** winner from `cp_conductance_g_e/g_i`, head_b via a spiking synapse (0 host arithmetic on the margin),
  `host_rng_draws_on_read_path = 0` on every seed; pools spike ~2.0/read (as the validated graded read does — the
  margin is the conductance drive, not the pool spike count); silent 0.0.

## External grounding

According to PubMed: **Mulder, Wagenmakers, Ratcliff, Boekel & Forstmann (2012)**, J Neurosci 32(7):2335-43
([DOI](https://doi.org/10.1523/JNEUROSCI.4156-11.2012)) — prior probability biases perceptual choice "primarily due to
a change in the starting point of the accumulation process," a common frontoparietal substrate. This grounds head_b as
a per-pool TONIC BASELINE / starting-point offset (a resting excitability the pools carry), NOT a change of the
evidence weights — exactly a tonic bias-input population onto the pools. The hidden population density (hid_pop=4) is
grounded in Zohary, Shadlen & Newsome (1994) via the parity-close finding.

## Honest residuals (why this is an end-to-end read-fidelity GO, not "the mouth works" / not "fully spiking")

1. **A measured COMPOSITION penalty (~0.026 recov / 0.07 argmax_agree), not a collapse.** composed_biaspop (0.9495)
   sits below the isolated readout arms (0.976-0.978) because the projection stage reconstructs `Wo_sp@state` at corr
   0.984, and that error is amplified at near-ties. The last few % is a higher-fidelity projection code (the projection
   finding's own next rung) — the composition holds near-parity, it does not close the gap to 1.0.
2. **head_b as a spiking synapse ~ host arithmetic, a small residual.** A fixed tonic bias is a CONSTANT per-pool
   baseline (biologically the base rate IS a constant resting excitability), where the host `s = hb_k*std_over_pools
   (margin)` renormalises per-position; the constant tonic loses that per-position gain. readout_biaspop (0.9757) trails
   readout_hostb (0.978) by only -0.0023 — the faithful-biology cost is small and named.
3. **A seed-42-calibrated unit scalar (proj_out_scale) is host glue.** The substrate projection margin is in arbitrary
   conductance units (RMS ~3.3x the host feature); one fixed scalar (like the reads' `ratio`) maps it to the read-out's
   feature scale. It is a unit-balance calibration, NOT a per-channel gain — but it IS a host multiply between the two
   stages, alongside the r_h gate and the LN inside the WKV state.
4. **NOT "fully spiking" / NOT production-wired.** The WKV recurrent STATE (Wv + leaky integrator + BPTT decay), the
   r_h gate and LN are host; the projection/read-out/head_b weights are host-designed (from the trained checkpoint).
   This lane composes the two substrate MATMUL reads end-to-end and moves head_b onto a spiking synapse; it is a
   runner-level de-risk (default-off), not the mouth's completion.

## Files
- Runner: `research/runners/_wkv_mouth_endtoend_substrate_read_derisk.py`
- Raw: `research/findings/raw/_wkv_endtoend_substrate_read_6seed.json`
- Builds on: `2026-08-13-fluid-mouth-upstream-output-projection-GO.md` (the projection read this chains),
  `2026-08-13-fluid-mouth-graded-conductance-read-GO.md` + `2026-08-13-fluid-mouth-read-parity-close-GO.md` (the
  read-out + host-arithmetic head_b this composes with and biologizes).
