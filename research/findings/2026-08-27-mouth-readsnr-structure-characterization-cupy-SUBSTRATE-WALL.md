---
type: finding
status: negative
date: 2026-08-27
verdict: 6-seed cupy SUBSTRATE-WALL for the mouth read — NO structured/coherent target direction is decodable by the graded-conductance read (0/6 recodable, 6/6 all-wall), and steering the target toward the substrate's read-eigenbasis does NOT help; the earlier CPU eigen-alignment "signal" was a numpy-backend artifact. The mouth read wall is a genuine substrate limit, not fixable by a better objective (softmax + dendritic both NO-GO) nor by target-recoding.
mechanism: mouth read-SNR — is the structure-selective read collapse RECODABLE (steer toward a readable code) or a SUBSTRATE WALL? Tested on the REAL cupy backend across 4 direction families.
lane: e-language-mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_structure_characterization/char_cupy_6seed.json
runner: research/runners/_wkv_mouth_readout_structure_characterization_derisk.py
---

# Mouth read-SNR: the structure-selective read collapse is a SUBSTRATE WALL on cupy (0/6 recodable), not fixable by target-recoding

Artifact: `research/findings/raw/_wkv_structure_characterization/char_cupy_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=cupy`, the production backend).

## The question (do not re-derive)

The softmax-confidence + dendritic-objective NO-GOs established the mouth's `||W||`->cap runaway is a STRUCTURE-SELECTIVE read collapse (a structured target weight direction reads corr ~0 while random directions of equal norm read high), and that no OBJECTIVE fixes it. The open fork: is it RECODABLE (steer the readout target toward a substrate-readable code -> the mouth learns a readable representation) or a genuine SUBSTRATE WALL? A prior CPU characterization was VOID (a numpy-backend seed confound built a wall-free net); this is the decisive re-run on the REAL cupy substrate.

## Result — SUBSTRATE WALL, 6/6, not recodable

Across 4 direction families at a fixed probe norm (37.5), 6-seed means on cupy:

- random directions read `corr 0.3146`; `head_w` (the structured target) reads `-0.0103`.
- the BEST structured direction across ALL families reads `max_structured_corr_mean 0.0253` (max over seeds 0.0299) — essentially zero.
- `n_seeds_recodable: 0`, `n_seeds_all_wall: 6`.
- EIGEN-ALIGNMENT (the CPU run's hinted lever) FAILS on cupy: `eigen_topk_corr_by_k` is ~0 at every k (0.0036 down to -0.0123); `eigen_bottomk` ~0; `sparsity` ~0; `interp` ~0. Steering the target onto the substrate's own top read-eigenvectors does NOT make it readable.

So the earlier CPU "top-PC 0.94 vs bottom-PC 0.37" eigen-alignment signal <!--derived--> (quoted from `2026-08-27-mouth-readsnr-structure-characterization-BACKEND-SEED-CONFOUND.md`) was a NUMPY-BACKEND ARTIFACT (the numpy net is wall-free; see that backend-seed-confound finding + the ENGINE_REFERENCE cross-backend seed trap). On the production cupy substrate, EVERY structured/coherent direction is unreadable; only random/high-entropy directions decode.

## What this settles

The mouth read wall is a genuine SUBSTRATE-LEVEL read limit: the graded-conductance margin decodes random/high-entropy weight directions but NOT structured/coherent ones — and it is NOT recodable by steering the target toward the substrate's read-eigenbasis. Combined with the softmax + dendritic NO-GOs (no OBJECTIVE helps), BOTH escape routes tried so far (better objective, target-recoding) are closed.

## NOT a stopping point — the surpassing mechanism (NO-DEFER)

A wall is a verdict on a METHOD, never on the capability. The read must be made to decode CORRELATED/structured population codes. Candidate mechanisms, un-tried:

1. **Decorrelation / whitening at the read.** The softmax-NO-GO diagnosis: correlated drive onto SHARED postsynaptic pools does not summate like independent drive — so a coherent target's contribution cancels in the shared-conductance margin. Decorrelate/whiten the hidden population BEFORE the read (as retina/LGN decorrelate), so a structured target projects onto an effectively-independent basis. This is the most direct surpass.
2. **A different read primitive** designed for correlated codes: spike-timing / latency code, or a population-vector read, rather than the summed graded-conductance margin.
3. **A high-entropy target-code architecture** — exploit that RANDOM directions read (0.31): map words to random/high-entropy projections rather than a structured `head_w`, then decode semantics downstream.

## SHARED WALL with learn-through-use (#182)

This is very likely the SAME read-fidelity limit that caps learn-through-use recall (the reverse-edge-depression NO-GO relocated its residual to "the substrate's read-side noise floor"). One shared spiking-population READ-FIDELITY wall may sit behind BOTH the mouth (fluency) AND learn-through-use (grow) — so a single surpassing read mechanism (decorrelation/whitening read) could unblock both. The next arc is a JOINT read-mechanism investigation, researched properly (population-decoding + decorrelation literature) rather than a reflexive lever. Related: Schuessler et al. 2023 (aligned/oblique readout regimes, recorded).
