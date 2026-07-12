# The spiking learn-W_in demo's task was the WRONG instrument (Johnson-Lindenstrauss): the R3 learn-W_in benefit is next-token PREDICTION + distributional GENERALIZATION, not cue classification — and on the correct instrument, learning W_in gives a strong 6-seed rate headroom (held-out 0.900 vs fixed 0.322)

**Date:** 2026-07-12
**Status:** ✅ REFRAME + ceiling-first GO (rate) — the correct instrument for the spiking learn-W_in demo is found and its headroom is confirmed 6-seed, structural (noise-free ⇒ transfers to spikes). Reuse-by-import; NO `sim/` edit; NO runner edit.
**Supersedes:** `2026-07-12-rate-headroom-ceiling-first-...K30-in-n80-...md` (the K-cue "headroom regime" it found was noise-dependent and did NOT transfer to spikes — this finding explains why and finds the instrument that does).
**Frontier:** the R3-reframe biological long-range path — *fixed spiking reservoir + e-prop-learned INPUT projection W_in (committed `enable_bdsp`) + local read-out* — the SPIKING realization. Mechanism already verified on spikes (BDSP moves W_in, `dw_rec≡0`, no weight transport); this finds the task on which its FUNCTIONAL win is demonstrable.

## The problem: two "headroom" instruments both failed, for the same reason

1. **Noise-headroom on the K-cue task didn't transfer to spikes.** A matched-difficulty rate map found headroom at K=30 cues in an n=80 reservoir + state-noise (learn 0.63 vs fixed 0.25) — but running the SPIKING arms there gave **fixed 0.933, learn 0.933** (no headroom). The rate noise proxy did not match the spiking reservoir's regime.
2. **Overlapping-code headroom at noise=0 doesn't exist at all.** A deterministic overlapping-sparse-code check (cues share dims, reservoir n < code-dims m = a lossy random compression) gave **fixed 1.000 at every (m, s, K, n)** — zero headroom.

**The unifying cause = Johnson-Lindenstrauss.** A fixed *random* projection preserves the distinctness of distinct inputs (w.h.p.), so **classification of distinct cues never needs a learned W_in** — a random one already separates them, and ridge-decodes them at ceiling. Noise creates *apparent* headroom (it collides the reads) but that's a noise artifact, not the R3 mechanism, and it doesn't survive the spiking regime. And a never-seen one-hot cue's W_in column is *untrainable* (input-synapse eligibility is nonzero only when that token is present), so cue-classification cannot even express generalization. **The K-cue delayed-decode task is the wrong instrument for the R3 learn-W_in benefit.**

## The reframe: what R3 actually measured

R3's massive headroom (learn +1.257 vs fixed −1.657 deep, on real LM) is a **next-token PREDICTION + distributional GENERALIZATION** phenomenon, not classification: the same tokens recur across many contexts, and learning the embedding places distributionally-similar tokens together so the *fixed* reservoir's dynamics produce predictive states and GENERALIZE to rare tokens. Reproducing it needs an instrument with all four ingredients the cue task lacks: **(i) prediction** (not lookup), **(ii) shared token structure** (so generalization to rare tokens is possible at all), **(iii) a class-irrelevant confound** (so learning must *suppress* something — the actual work), and **(iv) a bottleneck** (so a fixed random projection can't preserve everything).

## The correct instrument + the confirmed headroom (`_reslm_generalize_rate_check.py`)

Class-structured next-token task: G classes × `syn` synonyms; class-mates SHARE `sf` code dims (the class feature) and each token has `idn` unique identity dims that are **class-irrelevant** (the confound); a near-deterministic class→class Markov transition (0.85); the read-out predicts the NEXT class from the current token. One rare synonym per class is **held out** of training; its class must be predicted from its shared class dims (seen via class-mates) despite novel identity dims. Metric = **held-out next-class accuracy**, learn-W_in vs fixed-W_in (input-synapse e-prop, broadcast random feedback, no weight transport).

**6-seed (42/43/44/100/101/102), noise=0, deterministic:**

| G | syn | sf | idn | n | held-out **learn** | held-out **fixed** | margin | train | chance |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 5 | **2** | **30** | 60 | **0.900** | **0.322** | **+0.578** | 0.89 | 0.167 |
| 6 | 5 | 2 | 30 | 50 | 0.900 | 0.344 | +0.556 | 0.89 | 0.167 |
| 6 | 5 | 3 | 30 | 60 | 0.900 | 0.361 | +0.539 | 0.89 | 0.167 |
| 6 | 5 | 3 | 20 | 60 | 0.900 | 0.517 | +0.383 | 0.89 | 0.167 |
| 6 | 5 | 3 | 10 | 60 | 0.900 | 0.778 | +0.122 | 0.89 | 0.167 |

**12/12 confound configs show headroom**, and the margin scales cleanly with the confound ratio `idn/sf`. At the strongest (sf=2, idn=30, n=60), the learned W_in holds the Markov ceiling (0.900) while the fixed random W_in **collapses to 0.322** (near chance) — the 30 class-irrelevant identity dims swamp its held-out generalization; the learned W_in suppresses them. **This is the R3 mechanism, isolated, noise-free, and 6-seed robust** — so it should transfer to the spiking reservoir (the collision is structural, not a noise artifact).

## Next concrete action (building now)

Build the SPIKING version of THIS task on a `SimulationBridge` reservoir (reuse `WinLearnReservoir`'s bridge + committed-`enable_bdsp` machinery; drive the multi-hot class+identity code into the input sub-pops; read next-class at the predict step; held-out synonyms). Gate: does the on-bridge BDSP `learn_win` arm hold high held-out generalization while `fixed_win` collapses, with the anti-cheats (input-lesion→chance; label-scramble→chance; `dw_rec≡0`, no weight transport)? If yes → the first genuine FUNCTIONAL win for the on-substrate learn-W_in generalization mechanism (the R3 arc's "real remaining build"); then 6-seed + adversarially verify before committing it as a surpass.

## Files
- `research/runners/_reslm_generalize_rate_check.py` — the correct instrument + confirmed 6-seed headroom.
- `research/runners/_reslm_overlap_rate_check.py` — the overlap NEGATIVE (JL: no deterministic headroom).
- `research/runners/_reslm_rate_headroom_sweep.py` — the noise-headroom map (did not transfer to spikes).
- `raw/_reslm_generalize_confound.json` — the strong-confound 6-seed sweep.
- Builds on `2026-07-11-R3-REFRAME-...md`, `2026-07-12-spiking-realization-scoping-...md`.
