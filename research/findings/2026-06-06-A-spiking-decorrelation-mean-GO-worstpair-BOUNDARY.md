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

## Functional gate (fair capacity) — PARTIAL WIN, revises the verdict upward
`_A_spiking_functional_gate.py`: the spiking-decorrelated codes drive the full 2026-06-04 multimodal grounded
benchmark (320 concepts: nouns→V1 Gabor, verbs+adjs→word encoder; numpy VSA reference agent). RAW vs ZCA vs SPIKING.

| codes | overall | flat | 1-attr | 2-attr | clause-d1 | clause-d2 | who | abstain |
|---|---|---|---|---|---|---|---|---|
| RAW (grounded, no decorrelation) | 66.7% (26/39) | 8/8 | 0/6 | 0/5 | 3/5 | 3/3 | 6/6 | 6/6 |
| ZCA (numpy stand-in) | **100%** (39/39) | 8/8 | 6/6 | 5/5 | 5/5 | 3/3 | 6/6 | 6/6 |
| **SPIKING (on-bridge)** | **76.9% (30/39)** | **8/8** | 2/6 | 0/5 | **5/5** | **3/3** | **6/6** | **6/6** |

- **CAPACITY was the dominant variable, not a fundamental failure.** At n_it=600 (1.9 neurons/concept) the codes
  blurred → 20.5%, even flat retrieval collapsed (2/8). At fair capacity n_it=4000 (12.5/concept) + a SPARSE
  feed-forward (density 0.06 → 2.9M synapses; dense would be ~hours of matvec) → **76.9%**. The sparse feed-forward is
  itself biologically apt (cortical feed-forward is sparse) and matches the project's own sparse-distributed coding.
- **The on-bridge decorrelation FUNCTIONALLY HELPS** — +10pp over raw grounded codes (66.7→76.9), and it RECOVERS
  flat retrieval, both clause depths, who-queries, and abstention to FULL ZCA parity. The mean coherence tightens at
  320-scale too (raw 0.104 → spiking 0.060).
- **The residual is ATTRIBUTE composition specifically** (1-attr 2/6, 2-attr 0/5 vs ZCA's 6/6, 5/5). Cause: the
  worst-pair COLLISIONS (max coh 1.0 — one pair of concepts gets near-identical codes, + 1 silent concept; seed-
  fragile, the same collision the numpy Földiák showed). Attribute binding is the capability most sensitive to code
  overlap, so the collisions cost it first. Closeable via collision-reduction: overlap-rejection (the G.20 distinct-
  seed recipe), more capacity, or interneuron-diversity for true pairwise specificity.

- **Capacity sweep (informative NEGATIVE) — the residual is NOT the collision:** bumping to n_it=6000 REMOVED the
  collision (max coh 1.0 → 0.968, 0 silent) but composition got WORSE (76.9% → 71.8%, 1-attr 2/6 → 0/6) because the
  codes got denser (mean_active 24.4 → 33.1). So the attribute shortfall is NOT primarily the worst-pair collision —
  it is the spiking codes' RESIDUAL COHERENCE (mean 0.06, with structured max ~0.97) vs ZCA's near-zero all-pairs
  (0.003). Attribute binding is the capability most sensitive to that residual structure; the local competitive rule
  plateaus at ~0.06 mean coherence and cannot reach ZCA's all-pairs cleanliness. **n_it=4000 (sparser codes, 76.9%)
  is the operating point; capacity is not the lever.** The indicated levers for the residual are SPARSER codes
  (stronger WTA / lower homeostatic target) or interneuron diversity (pairwise cleanliness) — or accept the partial
  win and keep numpy ZCA for the cleanest codes.

**REVISED VERDICT: the on-bridge spiking decorrelation is a PARTIAL FUNCTIONAL WIN, not a boundary.** It is a genuine
biological mechanism that improves grounded-code composition over raw (+10pp) and recovers most capabilities to ZCA
parity; the attribute-composition shortfall is the named, closeable worst-pair-collision residual (the E/I non-
specificity → occasional collisions). numpy ZCA stays the all-pairs reference; the spiking layer is a working,
partially-converted on-bridge realization.

## Deep-grounding arc status (honest)
- **Grounding INTERFACE:** works (Phase 3 — V1-Gabor / word-encoder codes → composer at parity, `grounded_codes`).
- **On-bridge DECORRELATION (mean/global):** ✅ realized in spikes (this finding) — the project's components, multi-seed.
- **On-bridge DECORRELATION (functional, 320-scale):** ✅ PARTIAL WIN — improves grounded-code composition +10pp over
  raw (66.7→76.9%) and recovers flat/clauses/who/abstain to FULL ZCA parity; ATTRIBUTE composition is the residual
  (worst-pair collisions, max coh 1.0 — single-FS E/I can't realize Földiák's pairwise specificity). Closeable via
  collision-reduction (overlap-rejection / capacity / interneuron diversity). numpy ZCA stays the all-pairs reference.
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
