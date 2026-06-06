# A deep-grounding arc — Földiák local-rule decorrelation de-risk → GO (seed-fragile) — 2026-06-05

Owner steered (after cheat D resolved): "tackle A's deep grounding." First cheap-first de-risk.

## Scope (what "A deep grounding" reduces to)
A (concept codes random/given) was already LARGELY grounded by the 2026-06-04 work
(`2026-06-04-v-multimodal-grounding-decorrelation-unifies.md`): a correlated V1(vision)+word(language) codebook
collapses composition to **0% raw** and reaches **100% (78/78, 2 seeds) once DECORRELATED** — the decorrelation being
the ventral-hierarchy stand-in. But that decorrelation was numpy **ZCA**. The cheat-A research's named residual is to
replace the numpy ZCA with an **on-bridge biological decorrelation by LOCAL rules** — Földiák 1990 (Hebbian
feed-forward + anti-Hebbian lateral inhibition + adaptive thresholds = sparse decorrelated codes). So the deep-grounding
arc = make the ventral→IT decorrelation a real on-bridge spiking layer, not a numpy matrix op.

## Cheap-first de-risk: does the Földiák ALGORITHM decorrelate ≈ ZCA?
`research/findings/raw/_A_foldiak_decorrelation.py`: a correlated codebook (16 concepts in 4 modality-like blocks,
within-block strongly correlated — the V1-block/word-block structure) → RAW vs ZCA vs Földiák cross-concept coherence.

| seed | RAW (mean/max) | ZCA (mean/max) | FÖLDIÁK (mean/max) |
|---|---|---|---|
| 42 | 0.418 / 0.915 | 0.067 / 0.070 | **0.000 / 0.000** |
| 43 | 0.420 / 0.931 | 0.067 / 0.071 | 0.048 / **1.000** |
| 44 | 0.443 / 0.936 | 0.067 / 0.071 | **0.000 / 0.000** |

**Verdict: GO (seed-fragile).** The corrected Földiák (binary threshold outputs, anti-Hebbian lateral toward target
p², adaptive thresholds keeping each output active ~p — the first naive version lacked the threshold + target and
COLLAPSED to coherence 1.0) decorrelates the codebook to mean coherence ~0.00–0.05 — comparable to / better than ZCA
(the binary sparse codes are near-orthogonal, beating ZCA's residual 0.067). **CAVEAT (the cheat-A research's exact
prediction): seed-FRAGILE** — seed 43 has a max-1.0 collision (two concepts mapped to the same sparse code). The
LOCAL rule approximates ZCA but isn't equal; at scale it will occasionally collide.

## What this establishes + the next steps
- The on-bridge-realizable LOCAL decorrelation (Földiák) IS viable for the ventral→IT decorrelation — the deep
  grounding need not rely on a numpy ZCA. The arc's foundational mechanism is confirmed.
- NEXT: (1) the FUNCTIONAL gate — does a Földiák-decorrelated grounded codebook COMPOSE at parity (≈ the 2026-06-04
  ZCA's 100%), and handle the collisions (a collision makes two concepts indistinguishable → hurts composition);
  (2) reduce the seed-fragility (more output neurons / sparser p / overlap-rejection); (3) the SPIKING on-bridge
  realization (a pool with plastic Hebbian feed-forward + plastic anti-Hebbian FS lateral inhibition — the project's
  FS neurons already do lateral inhibition; make it plastic/anti-Hebbian + adaptive thresholds).
- HONEST BOUNDARY (unchanged): abstract-concept grounding from raw sensation is the embodied-cognition limit
  (the word-encoder block stands in for it); the 320-scale on-bridge decorrelation is seed-fragile (this de-risk
  shows the fragility concretely).

## Artifact
`research/findings/raw/_A_foldiak_decorrelation.py` (correlated codebook + RAW/ZCA/Földiák coherence, multi-seed).
Numpy reference (the algorithm); the spiking on-bridge layer is the next build. NO sim/ edits.
