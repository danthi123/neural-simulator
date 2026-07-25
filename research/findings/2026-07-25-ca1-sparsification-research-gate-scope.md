# CA1 sparsification research-gate scope — the consolidation write's presynaptic-code density is SURPASSABLE on the point-neuron substrate, and cheaply. The "point-neuron sparsification wall / needs a different substrate" verdict is PROTOCOL-CONFOUNDED and invokes the WRONG limit (whitening ≠ sparsification). The numpy oracle already proved dendrites are NOT the key (K=1 → own/other 8.19)

**Date:** 2026-07-25
**Type:** READ-ONLY deep-research gate (no build, no `sim/` edit). Local corpus (RAG + direct source reads) → substrate reads → external literature.
**Fires because:** the consolidation ca1→slot BTSP write hit a confirmed boundary and the exhaustive
`2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md` closed with a soft "needs a genuinely
DIFFERENT substrate" verdict. Per the SURPASS-sharpening directive, that verdict — a DISGUISED boundary — mandates this
round: isolate the genuine residual, reframe via real biology, rank cheap-first surpasses, verdict.

---

## EXECUTIVE SUMMARY (the 4 gate moves, condensed)

1. **The residual is TINY and precisely located.** The write's selectivity is a bilinear self/cross overlap of the CA1
   rate code (`own/other ≈ Σfire_i² / Σfire_i·fire_j`; the re-attribution proved this). On the DENSE 85-cell tag the
   ceiling is **1.54** (< the 2.5 gate); on the SPARSE core (~10–18 cells, Jaccard 0.064) the ceiling is **5.56** (≫ gate);
   and the project's OWN numpy oracle (`_consol_multibranch_oracle.py`) shows a rate-proportional write on the sparse code
   gives **own/other 8.19 with K=1 (NO dendritic branches)**. ⇒ the residual is NOT the write and NOT dendrites — it is
   **producing/preserving a sparse (~5–15-cell, disjoint) fact-specific PRESYNAPTIC code during the write window.**
2. **The record is CONFOUNDED by the consolidation PROTOCOL, not the substrate.** The "dense 85-cell CA1 tag" is a
   `per_regime_monitor_runner._encode_facts` COMMIT artifact — `top_k = max(8, n_per_pool//4) ≈ 85` (`:318`) — re-stimulated
   by a **direct 1500 pA `stimulate_tag` current flood** that bypasses the pattern-separation circuit and co-activation-
   replays all facts together. On the *other* harness (`nmda_compositional_consolidation.py`) the SAME substrate produces
   **sparse, near-disjoint CA1 engrams (8–14 cells, Jaccard 0.000/0.111/0.083)** (`2026-07-25-consolidation-coactivation-...`).
   Same neurons, different protocol → sparse vs dense. The wall was measured on the dense-protocol harness.
3. **The invoked limit is the WRONG one.** The re-attribution files this under the Mikulasch-Priesemann point-neuron
   **whitening/decorrelation** limit. But this is a **sparsification (k-WTA)** problem, a DIFFERENT computation:
   decorrelation removes the common-mode structure of a dense correlated code (analog/dendritic-hard); sparsification
   *selects the strongly-driven few* (somatic feedforward/lateral inhibition — the standard point-neuron SNN mechanism).
   The sparse core is ALREADY near-disjoint (Jaccard 0.064) once selected — no whitening is needed, only k-WTA selection.
   Point-neuron DG models sparsify to ~1–5% active by exactly this route (external lit below), and this project has
   validated it THREE times (D.12 DG separation; EMERGE-41 rank-order k-WTA; riii sparse CA3 assemblies).
4. **VERDICT: SURPASSABLE, cheap-first, NO dendritic rewrite.** Drive the write from the SPARSE locus (a sparse-committed
   engram / the riii sparse CA3 assembly / DG at its D.12 sparse op-point), reinstated via the FEEDFORWARD trisynaptic
   path ONE fact per SWR ripple (biological replay), with a rate-proportional write. Every ingredient is shipped. The
   genuinely-hard sub-residual (a robust *drive-invariant* <5% k-WTA at dt=1.0) is a tuning risk for a single 40-step
   window, not a wall. The dendritic per-branch write is oracle-KILLED for THIS problem (K=1 already 8.19).

---

## (MOVE a) ISOLATE + QUANTIFY THE GENUINE RESIDUAL

### a.1 — What the write is bounded by (proven, holds)
From `2026-07-25-consolidation-boundary-REATTRIBUTED-...` (the decoupled-plateau probe, the decisive experiment): the
`ca1→slot` write own/other is a **bilinear form of the CA1 rate code with itself** — `own/other ≈ Σ_k fire_i[k]² /
mean_j Σ_k fire_i[k]·fire_j[k]` = the code's **self/cross overlap**. No write mechanism (supralinear eligibility exp 4/8,
thresholded heterosynaptic depression, FFI-kWTA, attractor-off, rate-Hebbian) can exceed the code's overlap. Measured:
- **Dense full-code overlap ceiling = 1.54** (< 2.5 gate). The measured writes sit at/below it (0.99–1.18).
- **Sparse (>25%-of-max-fire) overlap ceiling = 5.56** (≫ gate); n_active [1, 10, 15]; Jaccard **0.064** (near-disjoint).

⇒ the residual is **entirely the presynaptic code's density**. This is the write's whole story — quantified, not vague.

### a.2 — The oracle already isolated it AWAY from dendrites
`_consol_multibranch_oracle.py` (the gate the arc built for the months-scale dendritic build): on the REAL measured CA1
codes, model K per-branch plateaus + a plateau-gated rate-proportional write, sweep branch-assignment. Result:
- **K=1 (NO branches) already gives own/other 8.19**; oracle-clustered 7.8–34; RANDOM branch-assignment 3.5 — ALL ≫ 2.5.
- ⇒ **the months multi-branch + clustering build is NOT warranted** (branches don't move it). The localizer is the
  **rate-proportional write on the fact-specific SPARSE core**, which needs no dendrites. The oracle's own conclusion.

The oracle over-predicts the *real* substrate 8× ONLY because the real co-activation replay produces a **flooded** CA1
firing (not the clean fire-under-tag pattern the oracle assumes). I.e. the 8× gap is exactly (and only) the
**sparse-code-during-replay** residual — the same residual as a.1, from the other direction.

### a.3 — Why the substrate reads "dense" (the honest 3-part decomposition, read from the code)
1. **CA1 has NO dedicated inhibitory/k-WTA pool.** `text_minimal_isolation.py:721–728`: `ca1` is `exc_fraction=0.85,
   internal_density=0.05, inh_weight_mean=0.8` — its only inhibition is 15% internal interneurons at density 0.05 /
   weight 0.8 (weak). There is NO `ca1_pv_basket` feedforward/feedback pool (only `dg_pv_basket` exists, `:700–707`).
   Same structural root cause as the documented CA3-no-feedback-inhibition (43% active) fact
   [[feedback_read_own_substrate_before_theorizing]]. So CA1's *natural* response to strong drive is dense — as designed.
2. **The dense "85-cell tag" is a COMMIT + FLOOD protocol artifact, not the substrate.** `per_regime_monitor_runner._encode_facts`
   commits the engram at a FIXED `top_k = max(8, n_per_pool//4) ≈ 85` (`:318`), then the probe/replay `stimulate_tag`
   drives ALL 85 committed cells DIRECTLY at 1500 pA (bypassing perforant→DG→CA3 pattern-separation). The re-attribution's
   own sweeps confirm density = the fixed top_k (tag_drive ∈ {300…1500} all give ~85 cells; drive changes RATE not COUNT).
   On the *other* harness the same substrate gives sparse 8–14-cell disjoint engrams (a.4). ⇒ the density is the protocol.
3. **A fixed-weight FFI pool sparsifies only ~2×, and only in a narrow drive band (not drive-invariant).** riii CA3:
   `ca3_pv_basket` global feedback dropped sparsity 0.43→0.15–0.21 (2–3×, saturating). EMERGE-41: the FS pool dropped the
   fired fraction 0.57→0.28 (~2×). Neither reaches <0.05, and the DG probe found the fixed `dg_pv_basket` FFI "barely
   changes" DG at the flood op-point (0.72→0.77). This IS a genuine point-neuron/coarse-dt residual — but it is a
   *drive-invariance/tuning* residual (below), not "point neurons can't sparsify."

### a.4 — The reconciling fact the wall verdict MISSED
The `nmda_compositional_consolidation.py` harness (build_biological_brain_regions, same substrate) produces
**sparse near-disjoint CA1 engrams: Jaccard 0.000 / 0.111 / 0.083, 8/14/12 active cells per fact**
(`2026-07-25-consolidation-coactivation-...`, the "DECISIVE diagnostic"). That is EXACTLY the sparse disjoint code the
write needs (ceiling 5.56). It exists NOW, on this point-neuron substrate. In that harness the write's problem was NOT
the presynaptic code (it was the *downstream one-of-N attractor WTA* collapsing to a single winner — a SEPARATE problem;
see the scope note at the end). So: **the sparse fact-specific presynaptic code is not a wall — it is already produced by
one of the two harnesses.** The task's premise ("route CA1 from a sparse code") is achievable, not aspirational.

**Genuine residual, one sentence:** *deliver a sparse (~5–15-cell, disjoint) fact-specific presynaptic code to the write
during the replay window* — surpassable because (i) it already exists in one harness, (ii) the substrate has three
validated sparsifiers, and (iii) the write on it is oracle-proven (8.19) without dendrites.

---

## (MOVE b) REFRAME — how real biology makes sparse hippocampal codes (and why point neurons CAN)

**The key category correction: SPARSIFICATION (k-WTA) ≠ DECORRELATION (whitening).**
- *Whitening/decorrelation* (Mikulasch-Priesemann point-neuron limit): remove the common-mode / off-diagonal correlation
  structure of a dense correlated code. This IS analog/dendritic-hard (the project's conversational-cortex finding). The
  re-attribution invoked this — but it is the WRONG limit here.
- *Sparsification (k-WTA)*: SELECT the strongly-driven few cells and silence the rest. This is a **somatic
  feedforward/lateral-inhibition** computation — the standard point-neuron SNN mechanism. Once the sparse set is selected
  it is *already* near-disjoint (the measured sparse core has Jaccard 0.064) — no whitening is required. The consolidation
  write needs sparsification, not whitening.

**How biology sparsifies the hippocampal code (Kandel 6e Ch 54; catalog D.12/D.05):**
1. **DG "expansion recoding" → ~1–5% active** (Marr 1971). Mechanism: (a) intrinsically high granule-cell threshold /
   low excitability, (b) STRONG **feedforward** inhibition — the perforant path drives PV basket cells that inhibit
   granule cells on the SAME volley, so inhibition SCALES with the drive (drive-tracking), (c) sparse strong perforant
   synapses, (d) lateral inhibition (basket/HIPP cells) enforcing E-I balance.
2. **CA3 → ~1–5% active** via mossy-fiber "detonators" (few, powerful, decorrelated) + PV-basket feedback inhibition.
3. **CA1 is a RELAY / comparator, NOT a pattern-separator** (`2026-07-09-riii-swr-generative-replay-...`, read directly:
   "CA1 is a relay/comparator... a fixed feedforward Schaffer cannot produce assembly-specific CA1 codes; the
   assembly-specific CA3→CA1→cortex mapping is what SWR-replay STDP LEARNS"). ⇒ **do not fight CA1 to make it sparse; the
   sparse code lives UPSTREAM (DG/CA3) and CA1 either inherits it or the consolidation write reads from the sparse locus.**
4. **First-spike / rank-order selection (Thorpe latency).** The di-synaptic FFI arrives ~1–2 ms after the direct EPSP, so
   the strongest-driven principal cells fire FIRST, in a narrow window, before inhibition clamps the rest → the winners
   are the strongest-driven. This is *drive-invariant in WHICH cells win* and is the project's OWN validated k-WTA
   (EMERGE-41). It is not "more inhibition"; it is the timing window.
5. **SWR replay reinstates ONE memory per ripple** (~20× compressed, coupled to cortical slow-osc + spindles). Consolidation
   is per-fact, per-ripple, in ISOLATION — NOT the all-facts-co-activation replay the arc used. The isolated fire-under-tag
   own/other of 4.5 (record) confirms an isolated single-fact write CAN be fact-specific; the co-activation dilutes it.

**External literature (confirms point-neuron feasibility — this is the STANDARD mechanism):**
- **PV+ interneurons establish a powerful lateral-inhibition k-WTA microcircuit in DG** (Espinoza et al., *Nat Commun*
  2018): "large DG neural population + powerful circuit inhibition mediates a winner-take-all coding scheme → extremely
  sparse activation of DG granule cells." Point-neuron models reproduce this (DG lamellar cluster = GCs + one basket cell
  E-I loop → WTA sparse activation; Kim & Lim 2021, *Cogn Neurodyn*; PMC9666645 disynaptic hilar pattern separation).
- **Diehl & Cook 2015** (the canonical SNN k-WTA): structured lateral inhibition — each excitatory cell drives ONE
  inhibitory cell one-to-one, which inhibits ALL excitatory cells EXCEPT its driver → a true k-WTA that SHAPES the winner
  set. This is DIFFERENT from a single global FS pool (which EMERGE-41 found *inert for selection*) and is a concrete
  cheap lever the project has not tried for hippocampal sparsification.
- Kopsick et al. 2024 (full-scale spiking CA3): assemblies held <1% sparse via inhibition + a 20 ms gamma-window encoding
  drive; symmetric STDP then binds them. Point neurons, sparse, on the same rule family this project has.

**Reframe:** the arc kept asking "can a WRITE localize on the (dense, flooded) CA1 code?" (No — bounded at 1.54.) Biology
never writes from a flooded relay. It (1) makes DG/CA3 sparse by feedforward/lateral inhibition + high threshold +
rank-order timing, and (2) reinstates ONE memory per ripple through that sparse feedforward path. Then any co-activity
write localizes (oracle 8.19). The fix is to supply the sparse feedforward replay, not to keep tuning the write on a flood.

---

## (MOVE c) RANKED CHEAP-FIRST SURPASS MECHANISMS

Shared GO-gate (the write gate, existing): `_consol_direct_weight_probe.py` — `ca1_engram_i→slot_j` own/other **≥ 2.5**,
own-is-max **3/3**, **6-seed**. Shared anti-cheats: (i) **sparsity must be EARNED** — presynaptic engram Jaccard < 0.2 /
active < 15% AND the reinstated core must still FIRE/complete (not silenced to 0 — a silent code trivially "separates");
(ii) **lesion** the sparsifier → density returns AND own/other → ~1.0 (load-bearing); (iii) **permuted-tag** → the write
follows the permutation (fact-specific, not a wiring artifact); (iv) **control-outperforms-real** guard.

### Rank 1 — SPARSE-LOCUS + ISOLATED (one-fact-per-ripple) REINSTATEMENT + RATE-PROPORTIONAL WRITE (cheapest; the decisive gate)
- **What:** stop writing from the flooded dense CA1 tag. Commit the engram from the strong core (`commit_top_k ≈ 15`,
  already threaded through `encode_facts_with_reinstatement`, `:440–465`), reinstate EACH fact ISOLATED — only its own tag,
  NO other slots driven, NO pool co-drive — per replay window (biological one-memory-per-ripple), and use a rate-proportional
  write. The record's decoupled-plateau probe reinstated "tag only" but over the DENSE 85-cell committed tag → 0.99; the
  **untested combination is SPARSE-commit + isolated reinstatement + rate write.**
- **Why it's the bet:** it directly supplies the sparse fact-specific code the oracle needs (8.19). It reuses the one
  harness (`nmda_compositional`) that ALREADY yields sparse 8–14-cell disjoint CA1 engrams (a.4). It removes the two
  confounds the wall was measured under (the 85-cell top_k flood + all-facts co-activation).
- **Reusable machinery:** `_encode_facts`/`commit_top_k` (shipped), `_consol_direct_weight_probe.py` (the gate),
  `fused_btsp_hetero_update` + `--pure-hebbian`/`no_stdp` + `hebbian_rate_window` (all shipped), the isolated-reinstatement
  path from `_consol_decoupled_plateau_probe.py` (adapt to drive only the sparse-committed tag).
- **The precise decisive check:** does isolated (one-fact, tag-only, no-pool-co-drive) reinstatement of a `top_k≈15`
  engram + a rate write reach own/other > 2.5? If YES → the boundary is closed by protocol, no new mechanism. **Honest
  risk (the record's claim to test, not assume): "sparse commit re-densifies to ~85 via CA1 recurrence."** But CA1
  recurrence is weak (`internal_density=0.05, w=0.3`); the likelier densifier was the pool→CA1 co-drive (removed by
  tag-only isolated reinstatement) — so the re-densification claim was measured under co-drive and is the FIRST thing to
  re-measure with pool-drive OFF.

### Rank 2 — A PROPER FFI k-WTA POOL AT THE D.12 SPARSE OP-POINT, driving the write locus via the feedforward path (not the 1500 pA flood)
- **What:** the arc's FFI-kWTA was inert because it inhibited a *directly-current-injected* tag (nothing for FFI to gate).
  Instead, reinstate through `ec→dg(+dg_pv_basket)→ca3` at the D.12 **sparse operating point**, and read the write from the
  sparse DG/CA3 code. The 2026-05-31 DG gate reached DG **0.018–0.044 active** (between-concept cos 0.169–0.296) — but note
  it used `dg=800, dg_pv_basket=240` (30%) with **tuned drive_scale/ffi_scale**; the 2026-07-25 probe used the default
  `n_dg=200, n_dg_pv_basket=60` (`:213–214`, same 30% ratio) at the *flood* drive and missed the sparse band. The lever is
  the (drive, ffi-weight) tuning into the band, plus scaling the pool, NOT a substrate change.
- **Anti-cheat that catches the fixed-FFI failure mode:** a **drive-invariance sweep** — 2–4× drive must keep active < 5%
  (fixed-weight FFI *fails* this: the 2026-05-31 dose-response degraded 0.16→0.54→0.81 as drive rose). If a fixed FFI can't
  hold the band, escalate to feedforward inhibition driven by the SAME afferent (drive-tracking) or Rank 3.
- **Reusable:** `dg_pv_basket` FFI wiring (`:1100–1109`), the D.12 validator (`validate_trisynaptic_loop.py`), the
  `_consol_dg_natural_probe.py` DG-density probe (re-run at the 2026-05-31 op-point + a larger pool).

### Rank 3 — STRUCTURED (Diehl-Cook) LATERAL INHIBITION = a TRUE k-WTA that shapes the winner set
- **What:** replace the single global FS pool (which EMERGE-41 proved is INERT for the winner SELECTION — it only
  sparsifies the loser pool 0.57→0.28) with a structured lateral-inhibition circuit: one-to-one E→I + I→(all-but-driver)E
  (Diehl-Cook 2015). This makes the inhibition *shape* which ~k cells survive, giving a genuine drive-robust sparse count.
- **Why ranked here:** it is the mechanism EMERGE-41 explicitly named as required ("a local/structured inhibition vs one
  global pool would be required for the FS to *shape* the winner set") and it is point-neuron-standard. Slightly more
  wiring than Rank 1–2, so third.
- **Reusable:** the EMERGE-41 FS-WTA harness (`_emerge41_fs_wta_kwinners_derisk.py`) + the RegionPathway primitives;
  additive/default-off, NO `sim/` edit.

### Rank 4 — FAST spike-frequency adaptation (M-current / AdEx `b`) on the write locus — NOT the slow homeostatic threshold
- **What:** the arc dismissed "homeostasis" as too slow (τ≈5 s) — the WRONG mechanism. Fast SFA (M-current tens of ms;
  AdEx adaptation current `b`) makes each firing cell raise its OWN threshold within the 40-step window → the population
  self-limits to the earliest/strongest firers (a temporal k-WTA). Genuinely different, fast.
- **Reusable:** `fused_hh_m_current_update` (HH M-current, shipped), AdEx `b` presets; enable on the write-locus region.
- **Caveat:** the coactivation arc tried runner-side SFA on the SLOT neurons for the *attractor* one-of-N problem and it
  didn't help THAT — but that was SFA on the readout attractor, not SFA on the PRESYNAPTIC write locus for sparsification.
  Different target; worth a cheap check.

### Rank 5 — Assembly-selective inhibition (Kim-Kim 2025): plastic E→I + heterosynaptic "spare-own-engram" I→E
- **What:** make `X→pv_basket` plastic (co-active E-I potentiate → feature-tuned interneurons) + reshape I→E to spare own /
  inhibit competitors. The paper: global inhibition ~30–60% retrieval; selective ~90%.
- **Why LAST:** highest build/tuning cost, and the prior CA3 two-assembly arc found the SOMATIC version NEGATIVE and the
  apical version geometry-sensitive (1/6) — BUT that was the harder *recall-time two-assembly discrimination* problem. For
  the WRITE (one fact per window, no recall-time competition) Ranks 1–4 should suffice; keep this as the escalation only if
  they don't.

### NOT recommended — the dendritic per-branch write (the months-scale build the fork named)
**Oracle-KILLED for this problem:** K=1 already gives 8.19; branches don't move it (a.2). The residual is sparsity, not
per-branch gating. The dendritic substrate is warranted for the *downstream one-of-N attractor* (a different problem, below),
NOT for the write's presynaptic sparsity.

---

## (MOVE, reuse) EXISTING PROJECT MACHINERY (the surpass is assembly, not invention)

| Need | Shipped machinery | Status |
|---|---|---|
| Sparse-committed engram | `encode_facts_with_reinstatement(commit_top_k=)` (`nmda_compositional…:440`) | shipped |
| The write gate | `_consol_direct_weight_probe.py` (own/other, 6-seed, engram-Jaccard) | shipped |
| Rate / heterosynaptic write | `fused_btsp_hetero_update`, `--pure-hebbian`/`no_stdp`, `hebbian_rate_window`, `btsp_elig_exponent` | shipped |
| DG FFI k-WTA + sparse op-point | `dg_pv_basket` (`text_minimal_isolation.py:1100–1109`); `validate_trisynaptic_loop.py`; D.12 gate (2026-05-31, DG→0.018–0.044) | shipped/validated |
| Sparse CA3 assembly | riii `ca3_pv_basket` + mossy-detonator → 18–40-cell assemblies, **Jaccard <0.05** (`_riii_ca3_synchronous_assembly_derisk.py`, `_gap5_r4_emergent_btsp_store.py`) | validated |
| Rank-order (first-spike) k-WTA | EMERGE-41 (`_emerge41_fs_wta_kwinners_derisk.py`) — drive→spike-time selection | validated |
| Isolated reinstatement | `_consol_decoupled_plateau_probe.py` (adapt: drive only the sparse-committed tag) | shipped |
| Sparse CA1 engram (existence proof) | `nmda_compositional` harness → 8–14-cell disjoint CA1 engrams | measured |
| DG/CA density probes | `_consol_dg_natural_probe.py`, `_consol_dg_overlap_probe.py`, `_consol_replay_firing_probe.py` | shipped |

NO `sim/` edit is needed for Ranks 1–3 (additive runner wiring + existing kernels). Rank 4 uses shipped HH/AdEx presets.

---

## (MOVE d) RECOMMENDED CHEAP-FIRST DE-RISK TO RUN NEXT

**Run Rank 1 — the sparse-locus + isolated + rate-write gate — it is one 6-seed run and is decisive either way.**

Concretely, on `nmda_compositional_consolidation.py` (the harness that already yields sparse 8–14-cell disjoint CA1 engrams):
1. `encode_facts_with_reinstatement(commit_top_k=15)` (sparse tag = the strong core).
2. Reinstate EACH fact ISOLATED per window: drive ONLY that fact's tag; NO other slots; **pool co-drive OFF** (this is the
   biological one-memory-per-ripple SWR, and removes the pool→CA1 densification path the "re-densifies to 85" claim rode).
3. Write with a rate-proportional rule (`--pure-hebbian`/`hebbian_rate_window`, or `fused_btsp_hetero_update` with the
   auto-`btsp_hetero_theta`).
4. Read `_consol_direct_weight_probe.py`: own/other, own-is-max, engram-Jaccard, active-fraction.

**GO** = own/other **≥ 2.5**, own-is-max **3/3**, presynaptic Jaccard **< 0.2** / active **< 15%**, 6-seed, with the four
anti-cheats (earned-sparsity + reinstated-core-still-fires; sparsifier-lesion → dense + own/other→1.0; permuted-tag →
follows; control-outperforms guard). **First sub-measurement (cheapest, do it first):** with pool-drive OFF + `top_k=15`,
does `stimulate_tag` fire ~15 cells or re-densify to ~85? That single read confirms or refutes the "re-densifies" claim
that the wall verdict rests on. If it stays ~15 with Jaccard < 0.2 → the sparse code is delivered and the write gate should
clear (oracle 8.19 predicts it).

If Rank 1 fails the drive-invariance/re-densification check, escalate to Rank 2 (D.12 sparse op-point FFI, larger pool) →
Rank 3 (Diehl-Cook structured lateral inhibition). Build the sparse-CA3-assembly variant (riii, Jaccard <0.05) in parallel
as the belt-and-suspenders sparse locus (CA3, not the CA1 relay).

---

## (MOVE d) VERDICT

**SURPASSABLE, cheap-first, and the "needs a genuinely different substrate" verdict is PROTOCOL-CONFOUNDED + mis-attributed.**

- **The sparse fact-specific presynaptic code EXISTS on THIS point-neuron substrate** — measured, not hoped: the
  `nmda_compositional` harness gives 8–14-cell disjoint CA1 engrams (Jaccard <0.11); the write finding's own >25%-fire
  core is Jaccard 0.064 (ceiling 5.56); riii built sparse CA3 assemblies (Jaccard <0.05); D.12 drove DG to 0.018–0.044.
  The project has validated sparse hippocampal coding on point neurons THREE times, and the external literature makes DG
  k-WTA sparse coding a STANDARD point-neuron result.
- **The write on that sparse code is already oracle-proven at own/other 8.19 with K=1 — dendrites are NOT the key** (the
  arc's own oracle). The residual is sparsity delivery, not the write mechanism and not a dendritic substrate.
- **The wall was measured under a confound:** a fixed dense `top_k≈85` tag commit + a direct 1500 pA re-stimulation flood
  + all-facts co-activation replay — which bypasses pattern-separation and floods the CA1 relay. The exhaustive
  NO-GO sweep (FFI-inert, DG-dense, divisive-norm, homeostasis, sparse-commit) is honest FOR THAT PROTOCOL but does not
  license the substrate-level "needs a different substrate" conclusion, because the untested biological protocol
  (sparse-locus + isolated one-per-ripple reinstatement + rate write) is the one biology actually uses and the one that
  matches the oracle's assumption.
- **The invoked limit is the wrong one.** This is a k-WTA **sparsification** problem (somatic feedforward/lateral
  inhibition — point-neuron-standard), NOT the Mikulasch-Priesemann **whitening/decorrelation** limit (analog/dendritic).
  The sparse core is already disjoint once selected; no whitening is required.
- **The genuinely-hard sub-residual, precisely stated + bounded:** a robust, *drive-invariant* <5% k-WTA on point neurons
  at dt=1.0 is a knife-edge with a fixed-weight FFI (holds sparsity only in a drive band; feedback inhibition has a 1-step
  delay). This is real — but the consolidation write only needs the sparse code to hold for a single calibrated ~40-step
  replay window per fact, which the harnesses already achieve. It is a tuning risk mitigated by rank-order (first-spike)
  selection and structured (Diehl-Cook) inhibition, not a wall. **No dendritic-substrate rewrite is warranted for the
  write.**

**Scope caveat (do not conflate two problems):** there are TWO selectivity problems in the consolidation record. (1) The
WRITE's presynaptic sparsity — THIS gate — is surpassable and cheap (above). (2) The DOWNSTREAM one-of-N *attractor/WTA*
selectivity (the `2026-07-25-coactivation-...` finding: even with disjoint 8–14-cell CA1 engrams, the slot WTA collapses
to a single dominant winner, seed-variable ~chance) is a SEPARATE, genuinely-harder problem that DID name the dendritic
line/bump attractor (P0.3 saturation family). That dendritic build, if pursued, is for the attractor READOUT, NOT for the
write's presynaptic code. Solving the write (this gate) is a prerequisite either way and should be done first — it is
cheap, and it de-confounds the attractor problem (a clean sparse selective write is the input the attractor needs).

---

## Files & citations

- **Sim (file:line):** CA1 region no-FFI-pool `research/runners/text_minimal_isolation.py:721–728`; DG + `dg_pv_basket`
  `:693–707`; DG FFI pathways `:1100–1109`; mossy `dg→ca3` `:1111–1117`; `ec→ca1` + `ca3→ca1` Schaffer `:1119–1137`;
  hippo sizes `n_dg=200 / n_dg_pv_basket=60 / n_ca3=100 / n_ca1=120` `:213–216`. Consolidation harness
  `research/runners/nmda_compositional_consolidation.py` (`build_substrate` + `ca1_ffi_kwta` `:195–203`;
  `encode_facts_with_reinstatement`/`commit_top_k` `:440–465`); dense commit `top_k=max(8,n_per_pool//4)`
  `unified_per_regime_monitor_runner._encode_facts:318`. Probes: `_consol_direct_weight_probe.py`,
  `_consol_multibranch_oracle.py`, `_consol_dg_natural_probe.py`, `_consol_decoupled_plateau_probe.py`,
  `_consol_replay_firing_probe.py`.
- **Project findings:** `2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md` (ceilings
  1.54/5.56, oracle 8.19, the exhaustive NO-GO sweep + "different substrate" verdict this gate surpasses);
  `2026-07-25-consolidation-coactivation-potentiation-fix-CONFIRMED-...md` (the sparse 8–14-cell disjoint CA1 engrams +
  the one-of-N attractor collapse); `2026-05-31-DG-separation-gate-PASS-...md` (DG→0.018–0.044, sparsity-dependent, the
  op-point the flood missed); `2026-07-09-riii-ca3-feedback-inhibition-sparsifies-but-nonselective.md` +
  `2026-07-09-riii-sparse-synchronous-ca3-ensemble-research-gate.md` (ca3_pv_basket sparsifies 0.43→0.21; global inhibition
  non-selective; mossy detonator + gamma-window); `2026-07-09-riii-swr-generative-replay-...md` (CA1 = relay not
  separator; the ca1_fb_inhib knife-edge; one-memory-per-ripple); `2026-07-02-emerge41-fs-wta-kwinners-GO.md` (rank-order
  first-spike k-WTA; single global FS pool INERT for selection, only sparsifies loser pool 0.57→0.28);
  `2026-07-18-gap5-specificity-research-gate-assembly-selective-inhibition.md` +
  `2026-07-21-gap5-2assembly-selective-inhibition-family-NEGATIVE-...md` (Kim-Kim assembly-selective inhibition;
  somatic-fails/apical-partial for the harder recall-time two-assembly problem).
- **Biology / external:** Kandel 6e Ch 54 pp 1357–1361 (DG expansion recoding, mossy detonators, CA3 recurrent LTP);
  Marr 1971; catalog D.12 (DG sparse ~2–5% + feedforward inhibition), D.05 (CA3 runaway excitation). Espinoza et al.
  *Nat Commun* 2018 ([nature.com/articles/s41467-018-06899-3](https://www.nature.com/articles/s41467-018-06899-3), PV+
  lateral-inhibition k-WTA microcircuit in DG → extremely sparse activation); Kim & Lim 2021/2022 *Cogn Neurodyn*
  ([PMC9120338](https://pmc.ncbi.nlm.nih.gov/articles/PMC9120338/), [PMC9666645](https://pmc.ncbi.nlm.nih.gov/articles/PMC9666645/),
  point-neuron DG WTA sparse coding); Diehl & Cook 2015 *Front Comput Neurosci* (structured one-to-one E→I / all-but-one
  I→E k-WTA, [frontiersin PMC7970006 comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC7970006/)); Kopsick et al. 2024
  (PMC10996657, full-scale spiking CA3, <1% sparse + 20 ms gamma-window encoding).

## Provenance
Read-only research gate, 2026-07-25. Local RAG + direct source/substrate reads + 2 external searches. NO `sim/` edit, no
build, no bars moved. The surpass is an ASSEMBLY of shipped machinery; the one decisive next run (Rank 1) is a 6-seed gate.
