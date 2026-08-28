---
type: finding
status: partial
date: 2026-08-27
lane: e-language-mouth-read-snr
mechanism: a DECORRELATION/whitening read to surpass a hypothesised SHARED spiking-population read-fidelity wall behind BOTH the mouth (fluent speech) and learn-through-use (recall) -- candidate A, remove the correlated common-mode so a structured/coherent target survives the read. Tested on both frontiers.
verdict: The hypothesised shared wall is REFUTED and has TWO different causes. MOUTH leg is VOID -- the "structure-selective substrate read wall" is a STALE-WEIGHTS MEASUREMENT ARTIFACT (cached COO transmission not invalidated on weight edits); on a correct read (fresh build per weight) the structured head_w decodes as well as a random direction, so read fidelity for structured codes rises from near-zero to full, with NO decorrelation. LTU leg does NOT share the artifact (its read builds fresh per read) and decorrelation is a decisive NO-GO there (subtractive and divisive both leave recall unchanged or worse). Decorrelation was neither needed (mouth) nor sufficient (LTU).
external:
  - Ruda, Zylberberg, Field 2019 "Ignoring correlated activity causes a failure of retinal population codes" Nat Commun 10:4605 https://www.nature.com/articles/s41467-019-12439-4 -- assuming independence among a correlated local population can decode worse than a single cell; grounds the decorrelation-read candidate.
  - Pitkow & Meister 2012 "Decorrelation and efficient coding by retinal ganglion cells" Nat Neurosci 15:628 https://www.nature.com/articles/nn.3064 -- the retina decorrelates (whitens) its output; the biological precedent for a decorrelating stage before the read.
artifacts:
  - research/findings/raw/_wkv_freshbuild_verify/fb_6seed.json
  - research/findings/raw/_wkv_freshbuild_verify/fb_s42_smoke.json
  - research/findings/raw/gap5_ecker_adex/decorr_read_ltu_6seed_sub.json
  - research/findings/raw/gap5_ecker_adex/decorr_read_ltu_div.json
  - research/findings/raw/gap5_ecker_adex/decorr_read_ltu_scan101.json
runner: research/runners/_wkv_mouth_read_freshbuild_structure_verify_derisk.py
---

# The shared mouth+LTU read-fidelity wall is REFUTED: the MOUTH leg was a stale-weights measurement artifact (read is faithful), the LTU leg is separate and decorrelation is NO-GO on it

Artifact: `research/findings/raw/_wkv_freshbuild_verify/fb_6seed.json` (6-seed mouth fresh-build-per-probe verify, cupy) + `research/findings/raw/_wkv_freshbuild_verify/fb_s42_smoke.json` (seed-42 detail) + `research/findings/raw/gap5_ecker_adex/decorr_read_ltu_6seed_sub.json` (6-seed LTU decorrelation, subtractive) + `research/findings/raw/gap5_ecker_adex/decorr_read_ltu_div.json` (LTU divisive) + `research/findings/raw/gap5_ecker_adex/decorr_read_ltu_scan101.json` (LTU lambda scan).

## The premise, and how it broke

The arc set out to build a DECORRELATION/whitening read (retina/LGN decorrelate their output; ignoring correlations fails population codes -- Ruda et al. 2019; Pitkow & Meister 2012) to surpass a hypothesised SHARED read-fidelity wall behind both the mouth and learn-through-use. The diagnosis it inherited: correlated drive onto shared postsynaptic conductance pools cancels a structured target while a random one survives -- so random reads corr ~0.31 and structured head_w reads ~0.03. <!--derived--> (both quoted from the prior finding's `char_cupy_6seed.json`)

Before building on that diagnosis, we checked the instrument. It was wrong.

## MOUTH: the "structure-selective substrate read wall" is a STALE-WEIGHTS measurement artifact (VOID)

The prior `2026-08-27-mouth-readsnr-structure-characterization-cupy-SUBSTRATE-WALL` ⛔ reported head_w corr mean -0.0103 vs random 0.3146, "0/6 recodable, 6/6 all-wall". But its three random ANCHORS read 0.956, -0.003, -0.008 on seed 42 (same pattern every seed): only the FIRST-measured probe read high; the 2nd, 3rd and head_w (all later) read ~0. <!--derived--> (anchor triples quoted from that finding's `char_cupy_6seed.json`)

A controlled order-vs-direction probe on the real cupy substrate settled it, and the runner reproduces it as its ARTIFACT/TRUTH split. On the shared-build (ARTIFACT) read, a random probe measured first reads high while head_w measured second reads ~0; on a FRESH build per probe (TRUTH), head_w reads as well as random. Seed-42 detail (`fb_s42_smoke.json`): TRUTH fresh-build headw 0.9541, sparse-structured 0.9479, eigen-top-structured 0.9685, random 0.9557/0.9544/0.955; ARTIFACT shared-build head_w 2nd read 0.0091 while the same random 1st read 0.9557 and the fresh random 3rd read -0.008.

Root cause (traced in `sim/bridge.py`): `BatchedSubstrateReadout` reuses ONE built bridge across many `set_weights()` calls, but synaptic transmission reads `_get_cached_coo()`, a cached COO matrix invalidated ONLY on a STRUCTURAL change (`_invalidate_coo_cache`, on synapse formation/elimination) -- NEVER on a weight edit. `set_weights` writes `cp_connections.data` correctly, but every read after the first transmits the FIRST-loaded weight matrix. So a random probe measured first read faithfully; head_w measured later transmitted the random probe -> corr ~0. The softmax diagnostic's probes A/B/C were the SAME direction rescaled (all matched the stale weights, all high) and its order-control reused that direction, so it never caught the bug.

The correct instrument is one weight per built substrate (a fresh build, whose COO cache is built from the loaded weights: `build_store` runs no steps; the first replay step builds the cache). On that read the STRUCTURED head_w decodes as faithfully as random. 6-seed means (`fb_6seed.json`): headw 0.9569, sparse-structured 0.9545, eigen-top-structured 0.9724, random 0.9473 -- structured == random; 6/6 seeds structured-faithful and 6/6 reproduce the artifact (head_w stale 2nd-read mean -0.0107, matching the prior finding). So the structured-read corr rises from ~0 (the artifact) to 0.9569 (fresh build), matching random -- decorrelation is not needed.

So the mouth read has NO structure-selective deficit. Read fidelity for structured codes rises from near-zero to full -- not by decorrelation, but by measuring correctly. The "SUBSTRATE-WALL" verdict is VOID.

## LTU: a different read, decorrelation is a decisive NO-GO

The learn-through-use graded-recall read does NOT share the mouth's bug: it calls `build_store` fresh for every read then `_load_weights` before any step (`inject_explicit_wiring` invalidates the cache; the first replay step rebuilds the COO from the loaded weights), so each read faithfully transmits its own weights.

We still tested the decorrelation candidate on it directly (a cross-assembly common-mode removal on the graded read -- lateral inhibition / divisive normalisation of the shared sharp-wave common-mode; byte-identical at lambda=0, verified in `decorr_read_ltu_6seed_sub.json`). It does not lift the weak-cue recall floor. Decisive 6-seed SUBTRACTIVE: NO-GO 0/6 (`decorr_read_ltu_6seed_sub.json`) -- every seed's weak-cue depth_frac FELL after consolidation instead of rising, e.g. seed-101 0.6307692->0.5153846, seed-100 0.5628205->0.4628205, seed-42 0.4423077->0.4782051 (the only riser, below the +0.05 bar). It is invariant across lambda (seed-101 weak-cue depth gain identical at lambda 0.25/0.5/1.0/2.0, `decorr_read_ltu_scan101.json`) because the depth metric is onset-ORDER based and subtracting a shared common-mode is order-preserving. DIVISIVE decorrelation WORSENS recall (`decorr_read_ltu_div.json`: seed-42 weak depth 0.6026->0.5897, seed-101 0.7423->0.6269, and tau drops below 1.0 -- it manufactures order errors).

## What this settles

The hypothesised SHARED read-fidelity wall behind both frontiers is REFUTED: the two legs have DIFFERENT causes. The mouth "wall" was an instrument artifact (a cache not invalidated on weight edits) -- exactly the project's deepest lesson (the instrument is part of the emulation; a mechanism you cannot measure correctly you will tune in the wrong direction for weeks). The LTU residual is a separate, genuine per-seed substrate effect that decorrelation does not touch. Decorrelation was neither needed on the mouth nor sufficient on the LTU.

Banked positives: the mouth substrate read is faithful for structured codes (the eprop mouth's read is not the wall it was thought to be); a correct measurement instrument (fresh-build-per-probe). Open follow-on (FAILURE_LOG 2026-08-27): the mouth eprop TRAINING loop also reuses one build across per-step `set_weights`, so the same stale COO likely starves training's read of its own weight updates -- a strong candidate root cause of the |W|->cap runaway, and the real next mechanism for fluent speech.
