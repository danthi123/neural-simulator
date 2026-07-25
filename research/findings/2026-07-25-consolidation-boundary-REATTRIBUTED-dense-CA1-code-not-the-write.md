# Consolidation A1 selectivity boundary RE-ATTRIBUTED (adversarial-verify + direct weight probe): it is NOT a WRITE failure — it is a DENSE, OVERLAPPING CA1 code (no ca1→slot rule can localize on it); the lever is upstream CA1 PATTERN-SEPARATION (Rank 2), not the write (2026-07-25)

**Supersedes the attribution in `2026-07-25-consolidation-opsweep-INTERIM-write-nonselective-plateau-saturates.md`.** That
interim (honestly self-corrected on mean-vs-median) attributed the non-selectivity to "the co-activation write" via the
LINEAR `g_e` proxy. A 9-agent adversarial-verify + research + design workflow (`wf_d539cd2c-31a`, 1.5M tokens, 0 errors)
+ a direct weight probe overturned that attribution. The interim's TWO solid points stand; its attribution was confounded.

## What HOLDS (adversarially confirmed)
- **The bistable plateau SATURATES** — `g_coincidence` own/other ≈ 1.000 across all 58 seed-42 op-configs; the all-or-none
  two-compartment plateau ignites fully at any supra-threshold slot and erases the input margin. Mechanism-independent, solid.
- **The written `ca1→slot` weights are FLAT** — confirmed by DIRECTLY reading the CSR (`cp_connections`, rows=pre/cols=post):
  `ca1_engram_i → slot_j` own/other ≈ 0.96–1.03, own-is-max at chance. Not a conductance proxy.

## What was WRONG / confounded (the correction)
- **`g_e` is NOT a valid proxy for write selectivity.** The linear-arm `g_e` own/other of 1.43–1.94 (my interim "some
  separation") is a **1.4–1.9× NOISE FLOOR from near-uniform weights** (w≈0.0525, barely moved off the 0.05 init) × random
  connectivity — decoupled from weight structure. My g_e median-1.025 read was measuring partly noise; the mean-2.49
  "outliers" were near-zero-denominator artifacts. Do not use `g_e`/`g_coincidence` to attribute (in)selectivity to the write.
- **The attractor-OFF-write refutation was a near-VACUOUS experiment** (decision sound, evidence empty): with
  `skip_nmda_additions=True` + `comp_no_pool_slot=True` there is NO cross-slot excitatory path, so "recurrent spread makes
  all slots fire" was *structurally impossible*, not empirically ruled out. Don't cite it as write evidence.

## The DECISIVE re-attribution — the direct WEIGHT + engram-overlap probe (STEP A)
`research/runners/_consol_direct_weight_probe.py` (promoted from the workflow verifier + extended: 6 seeds, engram-threshold
sweep, rate-weighted, seed-hash). It reads `ca1_engram_i → slot_j` weights directly AND measures CA1 engram overlap.
**Seed 42 (dw=0.045, the biggest-write OP):**
- **CA1 engram overlap Jaccard = 0.58 at thresh_frac=0 (any spike), 0.0 at thresh_frac=0.5** — ~80–91 of the ~88 CA1
  neurons fire for EACH of the 3 facts. **This RESOLVES the record's contradiction** (0.55–0.67 "dense" vs 0.00–0.11
  "near-disjoint" were the SAME substrate at different engram thresholds — any-spike inflates overlap; sparse is disjoint).
- **DISTINCTIVE-cell own/other ≈ [0.967, 1.005, 0.982], rate-weighted ≈ [1.02, 1.002, 1.002], own-is-max 1/3 (chance).**
- **ROUTE = DENSE-CODE → CA1 PATTERN-SEPARATION (Rank 2).** The presynaptic signal is NOT selectable: ~90% of CA1 is
  co-active per fact, so NO `ca1→slot` write rule (heterosynaptic or otherwise) can localize `ca1_i→slot_i` — the
  distinctive set is tiny (a few cells) and the shared ~90% pedestal is fact-agnostic. **6-SEED CONFIRMATION
  (42/43/44/100/101/102): 6/6 DENSE-CODE, 0/6 selectable-write — UNANIMOUS.** Jaccard@any-spike **0.557 ± 0.023**,
  @>50%-fire **0.000 ± 0.000** (the high-firing distinctive cells ARE disjoint — that is what the sparse-definition
  "0.00–0.11" measured); rate-weighted own/other **1.014 ± 0.006** (chance), own-is-max 1–2/3. **6/6 unique
  `cp_neuron_firing_thresholds` hashes ⇒ the substrate is genuinely seeded** (the seed-never-controlled-substrate trap is
  NOT present here). The re-attribution is robust, not a seed-42 artifact.

**No sparse-readout shortcut (checked):** at `thresh_frac=0.5` the engrams are EMPTY (0 distinctive cells, all 6 seeds) —
i.e. CA1 responds to each tag with ~90% of cells firing WEAKLY (a few spikes each over 40 steps), a **dense, low-rate,
overlapping** code with no sparse high-rate distinctive core. So a sparse readout that focuses on high-firing cells cannot
rescue it (there is no such subset) — CA1 must be made to respond SPARSELY + DISTINCTLY (pattern-separation) upstream.

⇒ **The A1 consolidation boundary is an UPSTREAM CA1 pattern-separation problem, not a write or readout problem.** This is
the project's own documented dense-CA-code failure mode (CA3-no-feedback-inhibition, 43% active vs sparse-wants-<5%;
[[feedback_read_own_substrate_before_theorizing]]) — read from the substrate this time, not re-derived.

## The next mechanism (per THE LAW — the method verdict names the surpassing ingredient), ranked
1. **Rank 2 becomes Rank 1 — CA1 PATTERN-SEPARATION** (the actual first lever): sparsify CA1 to <5% active via a
   feedforward-inhibitory kWTA pool, OR source the consolidation slot-drive from the ALREADY-VALIDATED DG pattern-separated
   locus (the trisynaptic loop's DG gives cos 0.218 from 0.800, P1-validated). Biology: Marr 1971 / O'Reilly-McClelland CLS.
   GO-gate: engram Jaccard < 0.2 (or active <5%) AND the write contrast then rises. Anti-cheats: lesion the FFI → density
   returns + contrast collapses; permuted-tag. Buildable NOW (kWTA/FFI is point-neuron standard; DG source exists).
2. **Then the heterosynaptic-competition WRITE** (`fused_btsp_hetero_update`/`fused_htm_winner_inactive_depression` shipped;
   extend the `comp_btsp` branch with `btsp_hetero_dep` + a calibrated `btsp_hetero_theta`) — potentiate own + thresholded-
   depress non-own. It sharpens the distinctive cells a separated CA1 supplies; on a dense code it has nothing to sharpen.
3. **Then a GRADED (non-saturating) readout + supralinear (n=2) divisive normalization** (`enable_graded_dendritic_plateau`
   + `enable_input_divisive_norm` shipped) to preserve + peak the margin the all-or-none plateau erases (necessary-not-
   sufficient — it manufactures no selectivity).
4. **Then, if needed, a discrete-well line/bump attractor with assembly-SELECTIVE surround inhibition** (Kim-Kim 2025;
   Ecker standing-bump) replacing the uniform-global WTA. Highest build/tuning risk; the last rung.

**The true closure is the STACK**, in this order: CA1 separation → heterosynaptic write → graded readout/surround-WTA → the
shipped per-cell bistable HOLD (the one genuinely-dendritic ingredient, gap#5-validated). All spiking-realizable NOW on the
existing bridge; NONE needs a new sim mechanism. The STEP-A measurement is what prevents building an amplifier onto a flat
write / dense code (the trap that produced this whole boundary).

## Recommended first de-risk (GPU-when-free / Tuesday)
Build the **Rank-2 CA1-separation de-risk**: add a feedforward-inhibitory kWTA pool to CA1 (or route slot-drive from DG),
re-run `_consol_direct_weight_probe` → confirm Jaccard < 0.2 / active <5% AND own/other rises + own-is-max. Then chain the
heterosynaptic-write arm. Full ranked spec + GO-gates + anti-cheats in the workflow synthesis (task `wf_d539cd2c-31a`).

## Provenance
Workflow `wf_d539cd2c-31a` (4 adversarial-verify + 4 deep-research + 1 design; one verify agent stubbed, flagged +
disregarded; verdict rests on the 3 that ran code + the direct probe). Probe `_consol_direct_weight_probe.py` (seed-42 GO,
6-seed running). Reuse-by-import, NO sim/ edit. Part of the [downtime 3-lane compute](2026-07-25-consolidation-opsweep-downtime-MANIFEST.md).
