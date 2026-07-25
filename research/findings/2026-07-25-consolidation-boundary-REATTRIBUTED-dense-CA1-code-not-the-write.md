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

## Family B TESTED — NO-GO (divisive-norm is GAIN control, not kWTA) → the point-neuron sparsification wall is CONFIRMED
Family B (activity-scaled self-tuning inhibition) on DG: added `divnorm_regions` to `build_substrate` (per-region
`input_divisive_norm`, additive/default-off). **VERIFIED ENGAGED** (`cp_input_divisive_mask` sum=200=n_dg, cfg flag on)
— yet DG stays DENSE (active 0.74–0.78, Jaccard 0.59–0.62), IDENTICAL to no-divnorm. Root: **divisive normalization
scales input MAGNITUDE, it does not reduce the COUNT of active cells** — it is gain control, not a kWTA, so it cannot
sparsify. (Homeostasis, also on by default at target 2%, is too slow for a 40-step reinstatement.)

⇒ **THE VERDICT (comprehensive, definitive): the consolidation A1 selective-write boundary is blocked by the point-neuron
SPARSIFICATION wall.** Two research workflows + ~12 probes + 2 additive sim edits exhausted every cheap-to-moderate method:
no hippocampal region (CA1, DG) can be sparsified to the sparse regime (fixed FFI barely works · divisive-norm is gain not
kWTA · homeostasis too slow · sparse commit re-densifies via recurrence), and no write-rule localizes on the dense code
(rate-threshold, supralinear eligibility, heterosynaptic all top out ~1.1 vs the 2.5 gate). This is the recurring,
well-documented point-neuron divisive-normalization/kWTA limit. **Per THE LAW the boundary stays OPEN**; the one remaining
named method is the **multi-branch dendritic per-branch write (Family C, months-scale)** — gated FIRST by a cheap numpy
oracle: model K per-branch plateaus + plateau-gated write on the REAL measured CA1 codes, sweep branch-assignment
(oracle-clustered vs random). GO(oracle ≥2.5, random ≈1.0) ⇒ the months build is warranted; KILL(oracle <2.5) ⇒ per-branch
gating cannot work on this dense-halo code even with perfect clustering ⇒ Family C is dead and the capability needs a
different substrate entirely. That oracle is the decisive next gate.

## Family-C numpy ORACLE (the gate for the months build) — result REDIRECTS away from the dendritic build (2026-07-25)
Built `_consol_multibranch_oracle.py`: on the REAL measured CA1 codes, model K per-branch plateaus + a plateau-gated
rate-proportional ca1->slot write, sweep branch-assignment.
- **Per-branch gating is NOT the key:** K=1 (NO branches) already gives own/other **8.19**; oracle-clustered 7.8–34,
  RANDOM 3.5 — all >> the 2.5 gate. So the months multi-branch + C2-clustering build is **NOT warranted** (branches
  don't move it; the localization is the rate-proportional write on the fact-specific core).
- **But the oracle over-predicts the real substrate 8×:** the oracle uses the clean FIRE-UNDER-TAG pattern (fact_i's
  core fires more under fact_i) + a LINEAR rate-proportional write. The real co-activation-replay write on the SAME cells
  is FLAT (~1.0). Tested the obvious reconciliations: **fewer replay cycles (3/8) does NOT help** (flat — so it's not
  w_max saturation over cycles); the rate-gated + supralinear write is marginal (~1.1).
- ⇒ **THE ROOT (final):** the real co-activation REPLAY produces a DENSE/FLOODED CA1 firing DURING the write that is NOT
  the clean fact-specific fire-under-tag pattern the oracle assumes — so the write's eligibility is flat -> flat write.
  The mechanism (a rate-proportional write) WOULD localize (oracle 8.19) IF the replay produced a clean fact-specific
  CA1 firing; the point-neuron replay cannot (every reinstatement floods CA1 — gentler drive, FF-synaptic, divisive-norm,
  fewer cycles all fail to sparsify/clean it). **The boundary is the point-neuron REPLAY-FLOODING / sparsification wall.**
- **Last cheap lever TESTED — NO-GO:** pool-co-drive-off replay (`pool_drive=0`, tag+slot only) is still FLAT (~1.0,
  plain and rate-gated) → the concept-pool co-drive is NOT what floods CA1.

## FINAL VERDICT (exhaustive) — the point-neuron write/replay wall; the capability needs a different substrate
Across 2 research workflows + ~16 probes + 4 additive sim edits, EVERY cheap-to-moderate method is NO-GO or marginal:
sparsification (fixed FFI · sparse commit · FF-synaptic reinstatement · divisive-norm · homeostasis — CA1 and DG both stay
dense ~0.7), write-rule (rate-threshold · supralinear-normalized eligibility · heterosynaptic depression — all ~1.1 vs the
2.5 gate), replay (gentler drive · fewer cycles · pool-off — all flat), and the multi-branch dendritic idea (the numpy
oracle shows K=1 already 8.19, so branches aren't the key — but the oracle over-predicts the real substrate 8× because it
assumes a clean rate-proportional write on the fire-under-tag pattern, which the real STDP/BTSP write on the flooded
co-activation replay does not achieve). ⇒ **The consolidation A1 selective-write capability is genuinely blocked on the
point-neuron substrate:** the enabling mechanism (a rate-proportional write on a sparse fact-specific hippocampal code)
WOULD localize (oracle 8.19), but point neurons cannot (a) sparsify any hippocampal region to a sparse fact-specific code
(the divisive-norm/kWTA limit) nor (b) realize the idealized rate-proportional write via spike-timing STDP on the flooded
replay. **Per THE LAW the boundary stays OPEN**, but the honest next method is a genuinely DIFFERENT substrate/mechanism (a
dendritic substrate that produces + reads sparse per-branch fact-specific codes, or a rate-based write on a
developmentally-sparsified code) — all substantial, none a cheap knob. This is the deep, well-characterized boundary; the
cheap-to-moderate search is comprehensively exhausted. Shipped reusable infra: `_consol_direct_weight_probe.py` (the write
gate) · `_consol_multibranch_oracle.py` (the in-principle gate) · `btsp_elig_exponent` · `divnorm_regions` · `commit_top_k`.

## ⚠️ Option-3 diagnostic OVERTURNS the "needs a different substrate" verdict — the fact-specific signal EXISTS (2026-07-25)
`_consol_replay_firing_probe.py` measured the CA1 firing DURING the co-activation replay vs the fire-under-tag, over the
distinctive cores (seed 42):
- **fire-under-tag own/other = [4.67, 3.05, 5.79] (mean ~4.5)** — the clean fact-specific code the oracle assumed IS real;
  point neurons DO produce a fact-specific CA1 code (this REFUTES the "can't sparsify to a fact-specific code" claim above).
- **REPLAY firing own/other = [1.98, 0.51, 2.12] (mean ~1.54)** — the co-activation replay DILUTES the 4.5 signal to ~1.5.
- The WRITE then flattens 1.54 → ~1.0 (STDP timing loses even the residual).
⇒ **Two SEPARABLE, potentially-cheap losses, NOT a substrate wall:** (1) the replay dilutes the fire-under-tag signal
(4.5→1.5) — a cleaner reinstatement should preserve it; (2) the write flattens the residual (1.5→1.0) — a RATE-based
Hebbian write (vs spike-timing STDP) should preserve it. If a cleaner replay preserves ~4.5 AND a rate write preserves it,
own/other → ~4.5 >> the 2.5 gate. **This is the live path; the FINAL VERDICT above is SUSPENDED pending these two tests.**

### Both Option-3 fixes TESTED — NO-GO (the write can't preserve the signal); FINAL VERDICT RE-INSTATED, refined
- **Fix 1 (cleaner replay) — can't recover 4.5:** the replay firing own/other stays ~1.1–1.6 across no-pools / no-attractor
  / settle-between-facts / gentler-drive (gentle is WORSE, 0.73). The co-activation dynamics (slot drive + open ca1→pool→slot
  pathways) dilute the isolated 4.5 to ~1.5 irreducibly.
- **Fix 2 (rate-based write) — flat:** a pure rate-Hebbian write (`no_stdp` + `hebbian_rate_window` + boosted lr, additive
  default-off) gives own/other ~1.0 (lr 0.005 and 0.02) — it does NOT preserve even the 1.5 replay-firing fact-specificity.
- ⇒ **REFINED final verdict:** the point-neuron substrate DOES produce a fact-specific code in ISOLATION (fire-under-tag
  4.5), but the brain-based CONSOLIDATION WRITE cannot preserve it — the co-activation replay dilutes it to ~1.5 (needs the
  slot post-drive + open reinstatement pathways, which flood CA1) and NO write rule tried (STDP, BTSP-hetero, rate-Hebbian,
  supralinear) preserves even that. The idealized plateau-gated rate write WOULD localize (oracle 8.19) but the real substrate
  can't realize it (plateau saturates; rate write flattens). **Per THE LAW the capability stays OPEN**; Option-3's cheap-space
  is now EXHAUSTED, so the next method is **Option 2 — a genuinely different write MECHANISM** (a dendritic write that
  reinstates ISOLATED + writes rate-proportionally with a non-saturating per-branch plateau, so the isolated 4.5 signal
  survives the write) — the months-scale build the fork flagged. Infra shipped: `+--pure-hebbian`/`no_stdp` · replay-firing
  diagnostic · the oracle. This is the honest, exhaustively-characterized boundary.

### Option-2 FIRST-MOVE (decoupled-plateau write) → DECISIVE re-attribution: the write is NOT the lever, CA1 code-overlap is
Built `_consol_decoupled_plateau_probe.py` (NO sim/ edit): reinstate each fact ISOLATED (tag only → the clean 4.5 CA1
pattern) + drive slot_i's apical plateau DIRECTLY as a pure BTSP teaching signal (clamp `cp_v_apical[slot_i]` high, hold
the OTHER slots down = exclusive), so no somatic slot drive floods CA1. The BTSP write becomes clean + exclusive (dw≈7).
**Result (seed 42): own/other = 0.99 — flat.** Direct measurement of WHY closes the question:
- The write's selectivity is a **bilinear form of the CA1 rate code with itself**: own/other = `Σ_k fire_i[k]·w[k→slot_i] / mean_j Σ_k fire_i[k]·w[k→slot_j]`, and w[k→slot_j] ∝ eligibility ∝ fire_j[k], so own/other ≈ `Σ fire_i² / Σ fire_i·fire_j` = the code's **self/cross overlap**. No write mechanism can exceed it.
- **Dense-code overlap CEILING = 1.54** (< the 2.5 gate). The measured write (0.99) sits at/below it. ⇒ **a better WRITE cannot pass** — Option-2-as-a-write-mechanism is fundamentally insufficient. Supralinear eligibility (exp 4/8), thresholded heterosynaptic depression, FFI-kWTA, attractor-off — ALL stay flat, because they can't change the code's overlap.
- **SPARSE (>25%-of-max) overlap CEILING = 5.56** (≫ gate), n_active [1,10,15], Jaccard 0.064 (near-disjoint). ⇒ **the fix is UPSTREAM CA1 pattern-separation** — a working kWTA/DG sparse code giving each fact a robust ~10-15-cell disjoint ensemble lifts the ceiling above the gate, and then ANY write passes.
- **Reconciles the record:** the earlier "fire-under-tag 4.5" was measured over fragile sparse DISTINCTIVE CORES (which don't even exist for fact 0, size 0); the robust FULL-code ceiling is 1.54. The 4.5 was a sparse-core artifact, not the write's achievable operating point.

**⇒ RE-ATTRIBUTION, sharpened + PROVEN:** the consolidation A1 selective-write boundary is NOT a write-mechanism problem —
it is the **CA1 code density** (no feedback inhibition → ~43% active, the known structural fact / point-neuron sparsity
limit family). The write is already at its code-bounded ceiling. **The lever is CA1 pattern-separation (sparse coding),
upstream of the write** — proven achievable-in-principle (sparse ceiling 5.56) but the point-neuron sparsification itself
is the known-hard part (my FFI-kWTA attempt made Jaccard WORSE). This fires the RESEARCH GATE (new mechanism class in the
documented DG/CA3-separation + Mikulasch-Priesemann whitening/point-neuron-sparsity family). Next: research-gate scope of
CA1 sparsification mechanisms + an idle-GPU kWTA sweep (can any config get CA1 to <5% active with a high ceiling?).

### Sparsification is the lever — but the CHEAP fixes are ALL falsified (the any-spike code the write sees is dense)
Ran the empirical sparsification battery + a read-only research-gate scope (`2026-07-25-ca1-sparsification-research-gate-scope.md`).
The research correctly reframed it as a **kWTA sparsification** problem (point-neuron-standard), NOT decorrelation/dendrites
(a category error I'd made), and recommended a cheap fix (sparse `commit_top_k=15` + isolated SWR reinstatement + rate write).
**Directly tested → the cheap fix is FALSIFIED, and so is every cheap sparsifier:**
- **feedback-inhibition kWTA** (ca1→ca1_ffi→ca1, swept inh 10-50 × drive 3-6 × n 30-60, `_consol_ca1_sparsify_sweep.py`): CA1 stays **91-96% active**, ceiling stuck 1.30-1.39. Feedback inhibition is too slow/divisive vs the drive.
- **sparse commit `top_k=15`**: the any-spike CA1 engram is still **91 cells** (Jaccard 0.62), ceiling 1.44 — `stimulate_tag` re-densifies via the drive. The sparse **>25%-core** is 12-13 cells (ceiling 4.68) — matching the research's "8-14 cells" — but the WRITE sees the any-spike code, not the core.
- **gentle reinstatement drive** (400/600/900 pA) + **pool-drive OFF** + **pure-Hebbian rate write**: all flat (own/other ~1.0, ceiling ~1.45). Lowering the drive does NOT sparsify the graded rate code.
- **per-region map** (`_consol_dg_overlap_probe`): the density is SYSTEMIC — ec 80% / dg 76% / ca3 78% / ca1 73% active under the tag flood. Even NATURAL perforant drive (`_consol_dg_natural_probe`) leaves DG **75% active** (Jaccard 0.61) — the existing `dg_pv_basket` FFI does not sparsify.
- **RECONCILED with the research:** we agree the sparse >25% BINARY core exists + is near-disjoint (ceiling 4.68-5.56); the research's error was assuming the write would USE it. The write's eligibility is a low-pass of ALL firing → it copies the DENSE any-spike code (ceiling 1.45). **The genuine residual = make the any-spike CA1 code itself <5% active + fact-specific**, which no cheap point-neuron inhibition tried achieves.
- **The real remaining path = the research's ESCALATION LADDER** (untested): fast spike-frequency adaptation (M-current / Izh-`d` so weak cells adapt out over the burst), structured/fast lateral inhibition (Diehl-Cook true k-WTA, not the slow divisive feedback loop), threshold-raising, or routing the write through a genuinely sparse CA3 assembly. These are substantial (need substrate levers), squarely in the known-hard point-neuron-sparsity family. Next: an exhaustive multi-mechanism workflow testing whether ANY sparsifies the any-spike code to lift the ceiling >2.5 (6-seed, anti-cheated: sparsity must be EARNED by inhibition/adaptation, the sparse code must stay fact-SPECIFIC not just sparse).

### DECISIVE FINAL characterization (cheap space EXHAUSTED, ~10 methods) — the surpass is the dendritic substrate, precisely
Added a biology-motivated sparse hippocampal PHENOTYPE lever (`hippo_izh_type` — give DG/CA3/CA1 the down-state-stable,
high-threshold, strongly-adapting **IZH2007_STRIATAL_MSN** phenotype, vs the too-excitable default HIPPO_PYRAMIDAL; DG
granule cells are biologically ~2% active). It sharpens the >25% core (ceiling **8.0**, n_active 16-23) — but the
any-spike/fire-count code the write reads STILL has ceiling ~1.45, and the write stays flat (0.99) even combined with
`elig_exp=8` + thresholded heterosynaptic depression. **The precise, airtight boundary:**
- The point-neuron substrate DOES produce a beautifully separable **sparse >25%-spike-count BINARY core** (ceiling 8.0, near-disjoint). The separable structure is REAL and present.
- But it is **not the OPERATIVE set**: both the consolidation WRITE (graded BTSP eligibility = a low-pass of firing) AND the RECALL (the full CA1 pattern activating ca1→slot) read the **dense fire-count code** (self/cross overlap ~1.45), in which the sparse core is drowned by the weakly-firing halo. Concentrating the write (`elig_exp`) doesn't help because the RECALL still activates the dense halo.
- **Making the sparse core operative requires EITHER (a) a genuinely sparse CODE (only the core fires) — FALSIFIED across ~10 point-neuron methods** (feedback-FFI, sparse-commit, gentle/strong drive, pool-on/off, natural perforant drive, MSN phenotype, and combinations; the fire-count overlap 1.45 is ROBUST to all) **— OR (b) a DENDRITIC nonlinear per-cell spike-count-threshold read** that gates both write and recall on SUSTAINED firing (a dendritic branch thresholding on its inputs' spike-count-over-window). (b) is exactly the deferred **dendritic-substrate** work (D2 Phase 0-2 built) — and NOTE this CORRECTS the research-gate scope's "not dendrites" claim: it is not dendrites-for-DECORRELATION (the code is already separable), it is **dendrites-for-the-nonlinear-READ** of an existing sparse core.
- **⇒ VERDICT: the consolidation A1 selective-write boundary is a GENUINE, precisely-located point-neuron limit — the sparse fact-specific structure exists but a point-neuron graded read cannot make it operative; the surpass is a dendritic spike-count-threshold read (the substantial substrate arc the fork flagged), NOT a cheaper write/inhibition tweak.** This is one of the most thoroughly-characterized negatives in the project (~20 probes, ~10 falsified methods, the residual pinned to a single measurable quantity: fire-count overlap 1.45 vs binary-core ceiling 8.0). Per THE LAW the capability stays OPEN; the next METHOD is the dendritic nonlinear read. Infra shipped (all reuse-by-import / additive default-off, NO protected-behavior change): `hippo_izh_type` phenotype override, decoupled-plateau probe (+exclusive teaching, +code-overlap ceiling, +sparse ceiling, +commit_top_k, +tunable reinstate drive), CA1-sparsify sweep, per-region + DG-natural diagnostics, the research-gate scope.

### Write-side hard-threshold eligibility de-risk → NO-GO (the eligibility is CROSS-FACT compressed; the read must be two-sided) — `640ae2d2`
Built the cheapest rung of the dendritic surpass: a HARD k-WTA gate on the BTSP presynaptic eligibility (`sim/config.py`
`btsp_elig_hard_thresh` + `sim/bridge.py:8072-8081`, ADDITIVE / default-off / **verified byte-identical when off** — the
guard `if thresh>0` skips the block; the 2 pre-existing `test_onbridge_btsp` fails are unchanged, confirmed they don't
touch the param). Sweep (seed 42, `commit_top_k=15`, MSN phenotype dg/ca3/ca1): own/other stays FLAT ~1.0 at every
thresh (0.25→0.90). **NO-GO vs the ~2.5 analytic prediction — and the failure mechanism is newly pinned:** the BTSP
eligibility is a τ=1000ms low-pass that INTEGRATES ACROSS THE WHOLE ~40-cycle MULTI-FACT write, so it is **cross-fact
compressed** — 100% of synapses survive at thresh 0.25 (the eligibility min is ≥25% of its max), so a magnitude threshold
cannot isolate a per-FACT core; and when it does gate hard (0.8-0.9), it keeps the network's SHARED strongest-firing
cells (not each fact's DISTINCT core) and dw collapses without adding fact-specificity. **⇒ the surpass requires a
PER-FACT-WINDOWED spike-count read (reset between facts, not a cross-fact-integrated eligibility), applied TWO-SIDED
(write eligibility AND recall `ca1→slot` activation).** The recall side is naturally per-fact (each cue is one fact); the
write side needs per-fact windowing. This is the full dendritic spike-count-threshold READ — the substantial D2 substrate
arc, now scoped to its precise mechanism. The cheap space (write-side-only, magnitude-threshold) is DEFINITIVELY
exhausted (~11 methods across ~20 probes + this de-risk).

## Recommended first de-risk (GPU-when-free / Tuesday)

## Recommended first de-risk (GPU-when-free / Tuesday)
Build the **Rank-2 CA1-separation de-risk**: add a feedforward-inhibitory kWTA pool to CA1 (or route slot-drive from DG),
re-run `_consol_direct_weight_probe` → confirm Jaccard < 0.2 / active <5% AND own/other rises + own-is-max. Then chain the
heterosynaptic-write arm. Full ranked spec + GO-gates + anti-cheats in the workflow synthesis (task `wf_d539cd2c-31a`).

## Provenance
Workflow `wf_d539cd2c-31a` (4 adversarial-verify + 4 deep-research + 1 design; one verify agent stubbed, flagged +
disregarded; verdict rests on the 3 that ran code + the direct probe). Probe `_consol_direct_weight_probe.py` (seed-42 GO,
6-seed running). Reuse-by-import, NO sim/ edit. Part of the [downtime 3-lane compute](2026-07-25-consolidation-opsweep-downtime-MANIFEST.md).
