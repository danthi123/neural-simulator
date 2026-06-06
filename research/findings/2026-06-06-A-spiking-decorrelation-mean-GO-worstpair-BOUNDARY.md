# A deep-grounding: spiking on-bridge decorrelation — MEAN decorrelation realized (GO), worst-pair is an E/I-substrate BOUNDARY — 2026-06-06

Owner steered "tackle A's deep grounding." The arc: replace the numpy ZCA decorrelation (the 2026-06-04
ventral-hierarchy stand-in that lifted a grounded V1+word codebook from 0%→100% composition) with a **real
on-bridge spiking decorrelation by local rules** (Földiák 1990). This is the honest result of realizing it.

## The three de-risk steps
1. **Cheap-first (numpy Földiák algorithm), GO** (`_A_foldiak_decorrelation.py`, `2026-06-05-A-deep-grounding-foldiak-decorrelation-GO.md`):
   the corrected Földiák (binary threshold + anti-Hebbian lateral toward p² + adaptive thresholds) decorrelates a
   correlated codebook to mean coherence ~0.00-0.05 (vs raw 0.42, ZCA 0.067). CAVEAT it already flagged: seed-fragile
   (a seed-43 max-1.0 collision — the local rule approximates, not equals, ZCA).
2. **On-bridge SPIKING competitive layer, STABLE PARTIAL** (`_A_spiking_decorrelation.py`, GPU): an IT pool with
   plastic Hebbian feed-forward (input features → IT) + FS WTA lateral inhibition + homeostasis (adaptive thresholds
   keeping each IT neuron active ~p). After fixing a COLD-START (first run DEGENERATE — drive ~94 pA + feed-forward
   weight 0.3 both below rheobase → silent IT pool; fixed via peak-normalized 600 pA drive + feed-forward weight 8.0):

   | seed | RAW coh (mean/max) | IT-code coh (mean/max) | mean active IT | silent |
   |---|---|---|---|---|
   | 42 | 0.418 / 0.915 | **0.218** / 0.808 | 26.9 | 0/16 |
   | 43 | 0.420 / 0.931 | **0.255** / 0.912 | 34.1 | 0/16 |
   | 44 | 0.443 / 0.936 | **0.175** / 0.780 | 19.5 | 0/16 |

   STABLE, real firing (0 silent), **mean coherence 0.42 → ~0.22** — a genuine on-bridge decorrelation. But max-coh
   0.78-0.91: the worst within-block concept pair still CLUSTERS (the competitive WTA groups similar inputs).
3. **+ anti-Hebbian FS lateral (plastic IT→FS so co-active pairs strengthen shared inhibition), UNSTABLE:**

   | seed | IT-code coh (mean/max) | mean active IT | silent | verdict |
   |---|---|---|---|---|
   | 42 | **0.150** / 0.908 | 17.5 | 0/16 | better mean |
   | 43 | 0.228 / 0.915 | 35.9 | 0/16 | ~unchanged |
   | 44 | 0.087 / 0.912 | **6.9** | **1/16** | over-suppressed toward silence |

   Seed-variable: helps 42, neutral on 43, over-suppresses 44 (the global anti-Hebbian drives the pool toward silence
   before the pair resolves). **Max-coh stays ~0.91 across ALL seeds + BOTH configs** — the worst pair never resolves.

## The honest finding
**The MEAN/global decorrelation IS realized on the spiking substrate** — the project's own Hebbian feed-forward +
FS WTA + homeostasis drop the correlated codebook's mean cross-concept coherence 0.42 → ~0.22 stably (real firing,
multi-seed). The ventral hierarchy's *global* efficient-coding pressure is a genuine on-bridge spiking mechanism.

**The all-pairs ZCA-level decorrelation (worst-pair → 0.067) is a BOUNDARY.** Földiák's pair-specific anti-Hebbian
lateral (a W_ik per output pair) does not map cleanly onto a single-FS-pool E/I spiking substrate: FS-mediated lateral
inhibition is NON-SPECIFIC (a shared FS pool inhibits all IT globally), so it can apply global decorrelation pressure
(the mean drops) but CANNOT selectively split the single worst within-block pair (the max persists at ~0.9), and
forcing it (stronger anti-Hebbian) over-suppresses the whole pool toward silence (seed 44) before the pair resolves.

**The principled next lever (future, deeper build):** real cortical decorrelation uses DIVERSE interneuron types
(PV / SST / VIP) with specific microcircuit connectivity — that diversity is exactly what Földiák's pairwise W_ik
abstracts. A single FS pool captures the global effect; the pairwise specificity needs the interneuron diversity
(multiple specialized FS sub-pools). This is a multi-week architectural build, named here as the path, not forced.

## Deep-grounding arc status (honest)
- **Grounding INTERFACE:** works (Phase 3 — V1-Gabor / word-encoder codes → composer at parity, `grounded_codes`).
- **On-bridge DECORRELATION (mean/global):** ✅ realized in spikes (this finding) — the project's components, multi-seed.
- **On-bridge DECORRELATION (all-pairs / worst-pair):** ⚠️ BOUNDARY — single-FS E/I can't realize Földiák's pairwise
  specificity; interneuron diversity is the principled path. numpy ZCA stays the reference for full all-pairs.
- **Abstract-concept grounding from raw sensation:** embodied-cognition boundary (the word encoder stands in).
- **Deep SEMANTIC grounding (real object-image dataset → V1→IT):** the dataset/embodied boundary (multi-month).

Per the top-level goal (artificial life with a proper brain analogue; honest negatives under strict biology ARE the
deliverable): the on-bridge spiking decorrelation realizes the ventral hierarchy's global decorrelation genuinely, and
names the pairwise-specificity limit + the interneuron-diversity path as a concrete, biology-translatable boundary —
not a papered-over success.

## Artifacts
`research/findings/raw/_A_spiking_decorrelation.py` (the spiking competitive + anti-Hebbian layer, GPU, multi-seed,
cold-start probe). NO sim/ edits — reuses the brain-region framework + Hebbian + FS + homeostasis.
Follow-on (not run, heavy): the functional gate (`unified_agent_multimodal_grounded.py` with ZCA→spiking codes) would
quantify whether the mean-0.22 codes compose (expected: most concepts compose, the worst-pair confusable).
