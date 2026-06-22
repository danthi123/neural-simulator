# Tier-2 #6 Route-B deployment smoke — shared spiking dopamine MODULATES the composer's encoding gain end-to-end on the one brain; behaviorally LATENT at the deployed D=128 read model (2026-06-22)

**Scope:** Tier-2 (TRUE ONE BRAIN) #6 — let the shared limbic dopamine REACH the conversational composer (the
owner's "one self" closure). Route B = DA-gated ENCODING strength (`encoding_gain_fn`). Deployment smoke on the MERGED
nav+conv bridge with the REAL spiking SNc. `research/runners/_tier2_routeB_deployment_smoke.py`. Single-seed (42),
GPU. **NO `sim/` edit** (the composer hooks pre-exist). Subagent-built, controller trust-but-verified the JSON +
the wiring against source.

## The positive core — the "one self" mechanism WORKS
The shared spiking dopamine modulates the composer's encoding gain end-to-end on ONE brain:
- The DA source is the REAL co-resident spiking SNc (`limbic_snc`; `get_concentration("dopamine")` = the signed-RPE
  `from_region_firing_signed`), driven to two operating points via the Route-A `_settle_snc` recipe: tonic I=80 pA →
  **DA=0.539** (0 Hz); salient I=600 pA → **DA=0.843** (358.5 Hz spiking).
- `encoding_gain_fn = clip(1 + 2.0·(DA − 0.5), 0.5, 3.0)` → **g=1.078** (low DA) vs **g=1.686** (high DA), applied
  hi>lo on the DEPLOYED `MergedNavConvAgent(co_resident_limbic=True).composer` (the `MergedRFComposer`, substrate
  store on).
⇒ the limbic core reaches the cortical composer in spikes — the deepest one-brain integration (a shared spiking
neuromodulator gating a cortical read-out), driven by the real SNc, not a hand-set scalar.

## The boundary — behaviorally LATENT
The DA-driven gain spread does NOT flip recall at the deployed read model:
- main (noise=260): hi_recall=1, lo_recall=1, **diff=0** (both recall; the gain does not separate them).
- σ-knee sweep (noise 260→2500): **NO** moat-safe level where HI(g=1.69) survives + LO(g=1.08) fails
  (`sigma_knee_moat_safe_diff_levels=[]`). The ONLY differential (noise=900, diff=−1) is **BACKWARDS** — the
  lower-gain LO fact survives because it is intrinsically more noise-robust at D=128.
- Diagnosis: the deployed composer is **D=128** (the numpy de-risk's positive was D=64; D=128's matched filter
  averages over 2× the components → far more noise-robust), so the per-fact CONTENT-robustness asymmetry of the
  specific FACT_HI/FACT_LO pair dominates the achievable DA-gain spread. Confirmed not a spread artifact (forcing
  g_hi=2.03 vs g_lo=1.12 → same pattern).

## Anti-cheats — all clean
- **MOAT (HARD):** the unstored cue abstains (→`None`) at BOTH DA levels AND the lesion — **0 false-accepts**
  (`moat_intact_all=True`).
- **DA-LESION:** both `hear`s at baseline DA → `lesion_diff=0` (no differential).
- **REGRESSION:** `encoding_gain_fn=None` byte-identical to the deployed default (`regression_default_identical=True`);
  the deployed regression suite `test_nav_conv_merged_agent` (8) + `test_nav_conv_step2b_coresident` (7) = **15/15
  verbatim** (the no-confab moat untouched).

## Verdict — LATENT (the pre-registered outcome)
The gain is correct + load-bearingly applied by the real spiking DA, but behaviorally latent at the deployed D=128
read model. A CHARACTERIZED boundary (the σ-knee sweep maps it) per the BRAIN-BASED-ONLY "honest boundary IS the
deliverable" standard — NOT a mechanism failure, NOT a NEGATIVE.

## Next (the decisive Route-B follow-up + the alternative one-self routes)
1. **CONTENT-MATCHED averaged test (the decisive isolation):** the LATENT here is confounded by the SPECIFIC
   FACT_HI/FACT_LO pair's intrinsic-robustness asymmetry. Average over N fact-pairs with HI/LO DA RANDOMLY assigned
   per pair → does HI-DA recall better ON AVERAGE (the gain effect, content averaged out) at a moat-safe knee? If
   yes → Route-B GO (the gain IS behaviorally load-bearing; LATENT was a content confound). If no → the
   encoding-strength gain genuinely does not help on the deployed read model (the boundary stands).
2. **The emergent-feature routes (the alternative closures if encoding-strength stays latent):** DA-gated
   RECONSOLIDATION (high DA labilizes/restabilizes a cued fact — a biologically-richer DA→memory effect) OR online
   salience-gated RECALL (DA gates retrieval, not encoding). These act where the read model is more sensitive than
   a uniform encoding scalar.
