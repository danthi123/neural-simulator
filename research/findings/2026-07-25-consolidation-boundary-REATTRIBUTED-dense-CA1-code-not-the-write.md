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

## Rank-2 FIRST DE-RISK RESULT (2026-07-25, seed-42 indicator): naive CA1 FFI-kWTA is INERT → pivot to the DG-source option
Built the CA1 FFI-kWTA augmentation (`build_substrate` `ca1_ffi_kwta`, additive/default-off: a `ca1_ffi` inhibitory basket
driven by CA1, inhibiting CA1, mirroring the shipped `comp_attr_inh` WTA pool) + swept the inhibition strength. **FFI ∈
{6, 30, 60} ALL give engram_sizes ~85 + Jaccard ~0.57 — FLAT, no gradient → the FFI is inert on the density.** Root cause
(read from the code): the engram tag is committed at a **FIXED `top_k = max(8, n_per_pool//4)` ≈ 85** (`_encode_facts`,
`per_regime_monitor_runner.py:318`) with a strong `lang_input`+teacher drive, and the probe re-drives that committed tag
DIRECTLY at 1500 pA — so inhibition can change neither the fixed tag size nor (as measured) the encode overlap. ⇒ **the
naive FFI-on-CA1 is the wrong lever; the Rank-2 build should PIVOT to the DG-source option** — source the consolidation
engram/slot-drive from **DG**, the ALREADY-VALIDATED pattern-separated locus (`dg`+`dg_pv_basket` in the trisynaptic loop;
P1: DG cos 0.218 from input 0.800), OR drastically shrink the commit `top_k` AND sparsify the encode together. This is the
precise, buildable next step (a real negative that refines the Rank-2 direction — the FFI infra + probe sweep are shipped).

## Rank-2 SECOND de-risk (DG-source) + a REFINEMENT of the "dense code" picture (2026-07-25, seed-42)
`_consol_dg_overlap_probe.py` measured the per-tag engram overlap for EVERY hippocampal region (ec/dg/ca3/ca1):
- **@any-spike: ALL regions are dense + overlapping** — ec active-frac ~0.77 / J 0.593, **dg ~0.72 / J 0.582**, ca3
  ~0.72 / J 0.556, ca1 ~0.73 / J 0.580. **DG is NOT more separated than CA1** → the naive DG-source doesn't obviously help.
- **@>25%-fire: ALL regions become SPARSE + DISTINCT** — active-frac ~0.2, **ca1 Jaccard drops to 0.084** (ec 0.132, dg
  0.117, ca3 0.105). **A sparse distinct CORE exists** in the strongly-firing cells of every region.
⇒ **REFINEMENT of the re-attribution:** the "dense CA1 code" is substantially a **tag-RE-STIMULATION artifact** — the
probe/replay drives the committed tag DIRECTLY at 1500 pA, flooding the committed cells so ~90% fire weakly; the natural
pattern-separation (P1 DG cos 0.218) is masked by this flood. The write is diluted by the weak-firing majority; the
distinct core is there but not what the STDP write latches onto. ⇒ **the Rank-2 lever refines to: (a) a GENTLER / more
NATURAL replay drive** (drive CA1 via the trisynaptic feedforward at a physiological level, NOT a 1500 pA tag flood, so
only the distinct strong-firing core fires during the write), **+ (b) a RATE-GATED HETEROSYNAPTIC write** (Rank 1;
potentiate from the strong distinct cells, thresholded-depress the weak overlapping majority — `fused_btsp_hetero_update`
shipped). Not "CA1 vs DG separation" (both dense under the flood). This is the precise, buildable next mechanism; the
FFI-inert + DG-also-dense results narrow it to the write-drive + rate-gated-competition pair. `_consol_dg_overlap_probe.py`.

## Rank-2 THIRD de-risk (gentler replay drive) + the FINAL characterization (2026-07-25, seed-42)
Swept the tag/replay drive `tag_drive ∈ {300, 500, 800, 1500}`: engram STILL ~85, Jaccard ~0.57, own/other ~1.0 at ALL
drives. **Decisive:** `stimulate_tag` drives EVERY committed tag cell above threshold, so the engram COUNT = the fixed
`top_k ≈ 85` (drive changes firing RATE, not cell-count). ⇒ **the density is fundamentally the FIXED `top_k` tag-commit +
the re-stimulation flood, not CA1-vs-DG and not the drive level.**

**⇒ FINAL characterization + the precise multi-part next mechanism** (3 de-risks converge — FFI-inert · DG-also-dense ·
drive-independent): the A1 selectivity boundary is that the consolidation engram is a **dense fixed-`top_k`≈85 tag** whose
re-stimulation floods ~90% of CA1 weakly; a **sparse distinct core exists** (>25%-fire, ~20 cells, Jaccard 0.08) but the
STDP write is diluted by the flood. The buildable fix is a **STACK, not a single lever**:
1. **Sparse distinct tag-COMMIT** — commit the engram from the strongly-firing core (`top_k`≈20 / a higher `threshold_hz`
   in `_encode_facts`:`per_regime_monitor_runner.py:318`, currently `top_k=max(8, n_per_pool//4)`≈85), so the tag IS the
   distinct core.
2. **A gentle, non-flooding replay** — reinstate via the trisynaptic feedforward at a physiological level (not a 1500 pA
   direct tag flood), so only the distinct core drives the write.
3. **A rate-gated HETEROSYNAPTIC write** (Rank 1, `fused_btsp_hetero_update` shipped) — potentiate from the strong distinct
   cells, thresholded-depress the weak overlapping majority.
Each is buildable now (no `sim/` edit for 1–2; 3 reuses shipped kernels). The measurement gate at each stage: the direct
`ca1_engram_i→slot_j` weight own/other (`_consol_direct_weight_probe.py`) must rise above ~2.5 with own-is-max, 6-seed.

**DECISIVE capstone (6-seed, free re-read):** even on the genuinely-DISTINCT core (>25%-fire cells, ~18/fact, Jaccard
0.08), the `ca1→slot` weights are FLAT — own/other **1.001**, own-is-max **7/18 ≈ chance**. So the distinct code exists but
the write FLOODS it (the replay re-stimulates the whole tag, so STDP potentiates every firing cell equally). ⇒ **the cheap
rate-gated-READOUT shortcut is RULED OUT** (no readout recovers selectivity from a flat write) — the WRITE + its drive must
change. Stack elements 1 (sparse commit) + 2 (gentle replay) + 3 (rate-gated heterosynaptic write) are all load-bearing,
not optional. This is the decisive, precise handoff for the build.

## THE 3-PART STACK — BUILT + TESTED (2026-07-25, seed-42); marginal, does NOT reach the GO-gate → a stronger mechanism is needed
The owner greenlit building the scoped stack. All three elements were built (additive/default-off, NO `sim/` edit beyond
reusing shipped kernels) + tested on the direct-weight gate:
- **Element 1 — sparse tag-COMMIT** (`commit_top_k` thread-through, `unified_per_regime_monitor_runner._encode_facts`):
  INSUFFICIENT ALONE. At `top_k`=10/20/40 the engram STILL fires ~85 cells — **CA1 recurrence re-densifies** the sparse
  tag once `stimulate_tag` drives it.
- **Element 2 — gentle FEEDFORWARD reinstatement** (drive only the CA3 tag portion → `ca3→ca1` completion,
  `_consol_ff_reinstate_probe.py`): INSUFFICIENT. **CA1 responds DENSELY to ANY input path** (direct flood OR CA3-driven,
  active-frac ~0.7, Jaccard ~0.57 either way). CA1's dense weak halo is inherent on this substrate.
- **Element 3 — rate-gated HETEROSYNAPTIC write** (`comp_btsp` + `btsp_hetero_dep` + the auto-thresholded `btsp_hetero_theta`,
  the shipped `fused_btsp_hetero_update`): the BEST but still MARGINAL. `theta`=0 doesn't rate-gate (the halo is coincident,
  not depressed); at `theta`=0.3, `dep`=0.6 the write own/other rises to **~1.09, own-is-max 2/3** (higher theta
  over-depresses → flat). Combined 1+3 (`top_k`=20 + `dep`=0.8 + `theta`=0.2–0.3): still **own/other ≤1.09, own-is-max ≤2/3**.
- **VERDICT (per THE LAW — a METHOD verdict, not a capability abandonment):** the cheap reuse-only stack gives only a
  marginal own-slot bias (~1.09), FAR below the GO-gate (own/other ≥2.5, own-is-max 3/3, 6-seed). The **core-vs-halo Etilde
  gap is fundamentally too small** — the dense weak halo fires enough to be coincident + carry moderate eligibility, so
  neither sparsification (1,2) nor rate-thresholded depression (3) separates it to the gate. This is the recurring
  point-neuron divisive-normalization / kWTA limit (CA1 cannot sparsify its halo; no reuse-only write-rule localizes on it).
- **NEXT METHOD (the boundary stays OPEN):** a genuinely stronger core-vs-halo separator — (i) a **dendritic** write where
  the strong-core apical plateau gates the write per-branch (the deep dendritic line/bump the fork named, months-scale),
  or (ii) a different consolidation architecture that does not route through a dense CA1 tag (e.g. a DG→cortex direct
  pattern-separated write, since DG's *committed* sparse code — not its flooded re-stim — is separated). The stack
  infrastructure (commit_top_k · FF-reinstatement · heterosynaptic write) is shipped + reusable for the next method.

## Family-A SUPRALINEAR eligibility (first sim edit, additive/default-off) — PARTIAL, doesn't close the gate (2026-07-25)
Root of the marginal element-3: the BTSP presynaptic eligibility is a LINEAR low-pass of firing (`elig = τ·elig +
(1−τ)·fired`, `bridge.py:8043`), so the strong-core-vs-weak-halo eligibility gap a threshold must cut is small. Built a
SUPRALINEAR eligibility exponent (`cfg.btsp_elig_exponent`, `bridge.py:8055` `etilde**p`; additive, default 1.0 →
byte-identical; Ca²⁺/CaMKII cooperativity) to widen it. Result (seed-42, dep=0.8): exp=2/theta=0.15 raises own/other to
**1.18–1.21 for 2/3 facts** (fact 1 regresses to 0.79); exp=3 or theta=0.25 flat. ⇒ a real but PARTIAL + inconsistent
lever — still far below the 2.5 gate. **Confirms the write-side levers (rate-threshold + supralinear eligibility) cannot
cleanly separate the core from the halo on point neurons** — the halo fires enough to overlap the core's eligibility even
supralinearly. A focused research gate (workflow `wf_2e5d85a3-9ce`) is assessing whether ANY reuse/small-edit lever
reaches the gate or the dendritic per-branch build is genuinely required. Infra shipped: `btsp_elig_exponent` +
`_consol_direct_weight_probe --btsp-elig-exp`.

## Research-gate mechanisms A + D TESTED (2026-07-25 workflow `wf_2e5d85a3-9ce`) — both NO-GO (point-neuron sparsification wall)
The focused research gate (4 families) named 2 cheap-first levers; both tested NO-GO:
- **Family A — normalize-then-supralinear eligibility** (the workflow caught my first exponent as BROKEN: raw
  `magnitude**p` vanishes the write; fixed to normalize-by-peak-then-`**p`, `bridge.py:8055`). Fixed result (seed 42,
  exp=2): own/other **~1.1, own-is-max 2/3, dw 0.117** (not collapsed) — a real lever but STILL marginal (halo-tail
  leakage holds it below the 2.5 gate, exactly the workflow's caveat that the mean-field over-predicts).
- **Family D — DG-direct write** (the RECOMMENDED primary; structural argument = DG has NO recurrence, verified, so a
  sparse DG can't re-densify). **NO-GO on the Step-1 density de-risk:** under NATURAL perforant drive
  (`language_input→ec→dg`, `_consol_dg_natural_probe.py`), DG is DENSE (active-frac 0.70–0.77, Jaccard 0.56–0.63) at every
  drive/sparsity tried (drive 100–400, input-sparsity 0.03–0.1), and the **`dg_pv_basket` FFI lesion barely changes it
  (0.72→0.77)** — the fixed FFI does NOT sparsify DG. So DG is dense in THIS substrate too; the sparse-DG premise is
  empirically false here. (The P1 DG-separation numbers were a different DG config/drive.)
⇒ **Both cheap primaries hit the SAME point-neuron FFI/kWTA sparsification wall** (CA1 dense · DG dense · fixed FFI
inert-ish). Testing Family B next (activity-SCALED self-tuning inhibition — divisive-norm / homeostatic adaptive
threshold, target ~2%), the design's decisive-either-way fallback; if it too can't sparsify, the wall is confirmed and
the route is the Family-C dendritic numpy oracle. `_consol_dg_natural_probe.py`, `btsp_elig_exponent` (fixed).

## Recommended first de-risk (GPU-when-free / Tuesday)
Build the **Rank-2 CA1-separation de-risk**: add a feedforward-inhibitory kWTA pool to CA1 (or route slot-drive from DG),
re-run `_consol_direct_weight_probe` → confirm Jaccard < 0.2 / active <5% AND own/other rises + own-is-max. Then chain the
heterosynaptic-write arm. Full ranked spec + GO-gates + anti-cheats in the workflow synthesis (task `wf_d539cd2c-31a`).

## Provenance
Workflow `wf_d539cd2c-31a` (4 adversarial-verify + 4 deep-research + 1 design; one verify agent stubbed, flagged +
disregarded; verdict rests on the 3 that ran code + the direct probe). Probe `_consol_direct_weight_probe.py` (seed-42 GO,
6-seed running). Reuse-by-import, NO sim/ edit. Part of the [downtime 3-lane compute](2026-07-25-consolidation-opsweep-downtime-MANIFEST.md).
