> # 📍 READ FIRST — CURRENT STATE (rewritten 2026-07-26)
>
> **This doc is an APPEND-LOG containing SIX reversals. Sections below are in the order they were written, so many are
> SUPERSEDED by later ones. This header is the only place that reflects the FINAL state — trust it over any section.**
>
> **1. ✅ THE CONSOLIDATION CAPABILITY WORKS — 6-seed, 2×2 factorial, both ingredients necessary and jointly
> sufficient.** Hippo-lesioned recall of a cortically-stored fact: **18/18 (3/3 on 6/6 seeds)**, chance 6/18.
> The two required ingredients:
> - **Miller-MacKay subtractive normalization** on the BTSP increment (`btsp_mean_subtract`) — fixes the WRITE.
>   Store own/other **12.51–46.61**, own-is-max **3/3 on 6/6**, permuted-target control **≤0.154** (lesion: ~1.0).
> - **A READ-ONLY READ** (`--freeze-read`) — the recall was overwriting the store *while reading it* (drift
>   **+1.28–1.41** live vs **+0.000000** frozen).
>
> Each ingredient ALONE gives chance (7/18, 7/18, 8/18). **Anti-cheat — scramble-teach is CAUSAL, not just a null:**
> teach a DERANGED pool→slot mapping and recall follows the DERANGEMENT — 1/18 vs the true mapping (BELOW chance)
> but **17/18 vs the mapping actually taught**. Below-chance is the signature of a real association read correctly,
> and it rules out recency / write-order / leftover-state by construction.
>
> **2. ⚠️ SCOPE OF (1) — three limits, none of them optional reading.**
> (a) **`--freeze-read` is a HOST intervention**, not yet neural. The biology is **SPEAR / Hasselmo ACh
> encoding-vs-retrieval**, already designed in-project (`2026-05-19-shared-rhythm-SPEAR-…`,
> `2026-05-22-acetylcholine-staged-…`) with a native target available (`plasticity_gate`,
> `scope="gate:<name>"`, `sim/neuromodulators.py`). **Tracked shortcut under BRAIN-BASED-ONLY, not closed.**
> (b) **NOT the full A1 gate.** The probe cues concept pools DIRECTLY by teacher current, because word→pool
> binding is UNBUILT. This is consolidation IN ISOLATION.
> (c) The metric is a 3-way slot argmax over a *shared* rate vector, so the 3 facts within a seed are COUPLED.
> **RESOLVED — and it cuts the OTHER way.** The confound that couples the trials is a per-slot excitability bias,
> and a fixed slot bias **CAPS a seed at 1/3 correct** (if one slot always wins, at most one fact can be right) —
> it cannot manufacture 3/3. The lesion arm sits at exactly **1.17/3**, that bias's signature. Aggregating to the
> genuinely independent unit (the seed): arm **6/6 perfect** vs scramble-control **0/6**, **Fisher exact
> p = 0.00108** (the minimum attainable at 6-vs-6). Coupling therefore makes 3/3 HARDER, not easier.
>
> **3. ⛔ RETRACTED — "A1's blocker is UNDER-TRAINING (200 vs 800 events → 87.5%)". THIS WAS WRONG.** Word→pool
> binding has **never worked above chance on any configuration reproducible today** (0–6.2% across A1 @200ev/@800ev,
> the reference harness @200ev/@800ev, and BOTH the current and 2026-05-22 checkouts). **No regression exists.** The
> recorded **87.5% is RETIRED as an unreproducible baseline** — it depended on cached `.simstate.h5` Phase-1
> substrates that are **DELETED**. ⇒ **word→pool binding is UNBUILT, not broken; establishing it is NEW CONSTRUCTION.**
> Do not use "it used to work, so something broke" — that framing cost this arc four withdrawn conclusions.
>
> **4. ✅ The 2026-07-25 "boundary" was NEVER REAL** — an artifact of `comp_apical_R=50.0`, a **333×** miscalibration
> of a pA→mV constant (engine default `0.15`) that parked `v_apical` at ~2×10⁵ mV and, via
> `apical_g_couple_to_soma=5.0`, drove every soma. The "dense 93%-active CA1 code" was runaway current, not a code.
> **VOID:** the dense-code re-attribution, the overlap ceilings, the 6-seed two-sided NO-GO, the M0 GO, and the
> "surpass = dendritic per-branch write" conclusion. On a valid substrate the `ca1→slot` write LOCALIZES (6-seed GO,
> own-is-max 18/18 fact-seeds, own/other 4.06 vs permuted 0.43); the suppressor was **soft-bound SATURATION** of
> BTSP's rank-1 outer product. Real CA1 code: sparse, near-disjoint (Jaccard 0.03–0.10), fact-specific.
>
> **5. ⛔ ALSO RETRACTED (2026-07-26) — "the cortical store does not drive the slots" was VOID.** The lesion never
> held: zeroing `cp_connections.data` survives ~1 instant and regrows (0 → 0.05 in 5 steps) because the read ran with
> plasticity live. Caught by the NEXT instrument contradicting it (the drive budget reported the "deleted" synapses at
> **90.85–95.04%** of all charge into slots). **The store is NOT out-weighted** — the architectural fork raised on that
> basis is withdrawn (per-synapse weight ≠ drive share: 64.5k store vs 15.5k recurrent synapses).
>
> **6. Other settled facts worth not re-deriving.** STDP is **INERT** on `concept_to_comp_attr` (`--no-stdp` is
> byte-identical). **Every RATE lever is inert BY CONSTRUCTION** — the store settles at a soft-bound FIXED POINT, so a
> learning rate changes how fast `w_max` is reached, never where it is (this explains 5 orders of BTSP lr and 100×
> Hebbian lr all reading "invariant"). Stability and selectivity are **the same knob**: raising the Hebbian bound to 50,
> removing Hebbian, or adding synaptic scaling all go UNPHYSIOLOGICAL (`v_apical` +198 / −284 mV, pool leak 21–46% vs
> <1%). `hebbian_max_weight` **defaults to 1.0** and inverts the rule against typical design weights — the same trap
> now seen on FOUR rules (STDP, BDSP, BTSP, Hebbian).
>
> **7. The through-line: every "boundary" in this arc was a CONFIGURATION or an INSTRUMENT, measured and reported as a
> property of the BIOLOGY.** Six retractions; **five were caught by an instrument built to answer a DIFFERENT
> question.** The habit that actually paid: **put the assertion in the DATA, never in a comment** (`read_weight_drift`
> is what exposed the void lesion). Rules earned here are encoded in `.claude/skills/verify-go/SKILL.md`.
>
> **▶ NEXT:** (a) wire ACh to the plasticity gate to retire the host freeze; (b) build word→pool binding (new
> construction); (c) port the winning protocol into `nmda_compositional_consolidation.py` main() and run the
> end-to-end A1 gate with its four anti-cheats at 6 seeds.


# 🔴 CRITICAL (2026-07-25): `comp_apical_R = 50.0` is a **333× miscalibration** of a pA→mV units constant — the consolidation arc's dendritic substrate has been running at `v_apical ≈ ±10⁵–10⁶ mV`

**Status: VERIFIED INDEPENDENTLY by the controller (not taken from the subagent report).** Found by the M1′ build's
localiser diagnostics (`13fd52ac`), then re-measured directly.

## The defect
| | value | source |
|---|---|---|
| engine default | `apical_R = 0.15` | `sim/config.py:267` — comment: *"plateau-current -> apical-voltage scale (**pA -> mV**)"* |
| consolidation runner | `comp_apical_R = 50.0` | `research/runners/nmda_compositional_consolidation.py:351` — comment: *"thin-high-R apical"* |
| ratio | **333×** | |

`apical_R` is a **calibrated unit conversion**, not a free style knob. The apical ODE (`sim/bridge.py:7184-7186`) is
`dv = -(v_apical - E_rest) + R·I_coincidence + g_c·(v_soma - v_apical)`, whose fixed point is
`v_apical ≈ E_rest + R·I_coincidence`. At R=50 with the arc's coincidence drive this parks the compartment five orders
of magnitude outside any physiological range.

**Controller's direct measurement** (seed 42, the arc's standard `comp_dendritic` config, teaching clamp at −25 mV):
```
step  0: v_apical global max=2.008e+05  mean=-1.527e+06  slot0 mean=1.684e+05
step 29: v_apical global max=2.093e+05  mean=-6.126e+05  slot0 mean=1.812e+05
VERDICT: v_apical max = 2.0932e+05 mV -> PATHOLOGICAL
```
A membrane potential belongs in ≈ −90…+50 mV. Note the **mean is hugely NEGATIVE** (−6.1e5) while the max is +2.1e5 —
so this is not confined to the clamped slots; apical compartments across the network are in a nonphysical state.
This is **not** an integration instability and **not** a `sim/` bug: the engine integrates the ODE it was given. It is a
configuration error in the runner.

## Why this single line explains the ENTIRE multi-hour boundary
The arc's write applies an "exclusive" apical teaching clamp: target slot to −25 mV, all other slots held to ≈ −70 mV.
Against a compartment sitting at ~2×10⁵ mV, **a ±45 mV clamp is numerically irrelevant.** Measured consequences
(M1′ localiser, 15 configs):
- the instructive signal is **not** exclusive: `IS` diagonal/off-diagonal = **3.45–3.48 at all 15 configs** — every
  non-target slot receives ~**29%** of the target's write drive;
- since the BTSP eligibility `Ẽ` is *per-presynaptic-cell*, `w[k→j] = Σᵢ Ẽᵢ[k]·IS[i,j]`, so the three slots receive
  **~97% the same weight vector**;
- ⇒ **own/other is pinned at ~1.0 for ANY write-side gate**, which is precisely what ~15 write variants, the two-sided
  read, and M1′ itself all measured (M1′: 1.00 ± 0.10, 0/45 fact-passes, indistinguishable from its own θ=0 lesion arm).
The chain was localised: firing→count→gate→eligibility is **intact** (`corr(elig, fire)` 0.83–0.91); eligibility→**weight**
is **broken** (`corr ≈ 0`). The break is the non-exclusive instructive signal, caused by the miscalibration.

## Scope of invalidation — read this before citing any number from this arc
Every consolidation result measured with `comp_dendritic=True` (which is the whole 2026-07-25 arc: the direct-weight
probe, the decoupled-plateau probe, the two-sided generalization probe, M0, M1′) ran at `apical_R=50`, i.e. on this
pathological operating point. Specifically **suspect, pending re-measurement at a physiological R**:
- the dense-CA1 code-overlap ceilings (1.45 dense / 8.0 sparse-core) and the "sparse separable core" they rest on;
- the 6-seed two-sided NO-GO;
- **the M0 GO** (single-burst + 100 ms recovery ⇒ peaked, mass-balanced, 3/3-fact code, ceilings 2.9–4.0).
What is NOT invalidated: the *methodological* results — the winner-slot artifact, the mean-vs-per-fact artifact, the
mass artifact, and the hardened verify-go triad; and the M1′ **edit** itself (byte-identical-when-off verified by md5
on the deterministic numpy backend, 4 new CI tests) which is sound and is the instrument that found this.

## The measured lever, and the tension that must be resolved next
Sweeping `--comp-apical-R` (runner-only, no `sim/` edit):
| R | `IS` diag/off (exclusivity) | `v_apical` | eligibility→weight link | CA1 core sizes |
|---|---|---|---|---|
| 50 (arc default) | 3.45 | ~2×10⁵ mV | broken (`corr≈0`) | ~10–30 (the arc's "sparse core") |
| 5 | 4.2 | physiological | **repaired** (`corr` 0.39–0.68) | **[0, 0, 0]** |
| 1 | **53** | +54 / −48 mV | — | **[0, 0, 0]** |
⇒ a physiological R **repairs the write** (exclusivity 3.5 → 53) but the **CA1 sustained core VANISHES**. In other words
**the separable sparse code this entire surpass plan rests on may itself be an artifact of the runaway-apical regime** —
the pathological apical, coupling into somas via `apical_g_couple_to_soma`, was plausibly driving the network activity
that produced it. This is now the load-bearing open question.

## Next de-risk (free, runner-only)
**Joint retune: physiological `apical_R` (≈0.15–1, i.e. the engine's calibrated scale) + a raised coincidence/tag drive**
so the CA1 sustained code re-emerges *without* the nonphysical compartment. **GO gate: `core_sizes ≥ 10` per fact AND
`IS` off-diagonal ≤ 5% (exclusivity ≥ 20:1) AND `v_apical` within −90…+50 mV** — then re-run the M0 oracle and the M1′
θ sweep (both already built and in place). If the sparse core does **not** survive a physiological operating point, then
the honest verdict is that this arc's separable structure was an artifact, and the surpass must be re-derived from a
substrate that is physically valid — a far better place to be than continuing to optimise a write inside a broken regime.

**Process lesson:** the arc spent many hours characterising a "boundary" that was a 333× units error in a config line,
and no metric could have revealed it — only reading the substrate's own state (`v_apical`) in physical units did.
**When a mechanism refuses to work across many well-controlled variants, measure the substrate's state variables in
their physical units and check them against physiological range, BEFORE concluding the capability is bounded.**

---

# ⛔ THE BOUNDARY ITSELF IS **VOID** — the "dense CA1 code" was the artifact. The real code is SPARSE and FACT-SPECIFIC.

Apples-to-apples, identical measurement (`_fire_under_tag`, seed 42, drive 1500, same encode), three configs:

| config | CA1 active (of 120) | Jaccard | **cosine specificity** | `v_apical` range (mV) |
|---|---|---|---|---|
| **1) BASE — no `comp_dendritic`** | `[8, 11, 10]` = **8%** | **0.058** | **[8.22, 15.0, 5.31]** | — |
| **2) the ARC's config** (R=50, gc_read=5) | `[116, 111, 109]` = **93%** | **0.877** | **[1.34, 1.36, 1.36]** | **−2.2e6 … +6.8e3** |
| **3) PHYSIOLOGICAL** (R=0.15, gc_read=0.5) | `[13, 15, 30]` = **16%** | **0.079** | **[6.55, 6.44, 3.29]** | **−5.3 … −3.0** |

**The premise of the entire 2026-07-25 consolidation arc — "the CA1 code is DENSE and OVERLAPPING, therefore no
`ca1→slot` write can localize (ceiling 1.45)" — IS FALSE.** That dense code existed *only* in the miscalibrated
operating point, where a diverging apical compartment (spanning −2.2×10⁶ … +6.8×10³ mV) drove **every** soma through
`apical_g_couple_to_soma=5.0`, producing 93%-active, Jaccard-0.88, specificity-1.35 network-wide activity that was never
a hippocampal code at all. The **real** CA1 tag response is **sparse (8–16% active), near-disjoint (Jaccard 0.06–0.08),
and strongly fact-specific (cosine specificity 3.3–15.0 vs the artifact's 1.35)** — i.e. exactly the pattern-separated
sparse code the hippocampus is supposed to produce, and exactly the regime in which a selective write should succeed.

**This retroactively explains the whole arc.** Every failure — ~15 write-rule variants, the sparsification battery, the
two-sided read, the eligibility thresholds, M1′ — was an attempt to write onto a 93%-dense artifact. The measured
"ceilings" (1.45 dense / 8.0 sparse-core), the winner-slot bias, the mass artifacts, and the M0 "GO" were all properties
of that artifact, not of the substrate. **The boundary was never real.**

**VOID (not merely suspect):** the dense-code re-attribution, the code-overlap ceilings, the 6-seed two-sided NO-GO,
the M0 GO, and the "surpass = dendritic per-branch write" conclusion that followed from them. **STILL VALID:** the
methodological results (winner-slot / mean-vs-per-fact / mass artifacts and the hardened verify-go triad), the M1′ edit
itself (byte-identical-when-off, 4 CI tests) and its localiser, and this correction.

**▶ NEXT: re-run the actual consolidation write + the END-TO-END capability test at the physiological operating point**
(`comp_apical_R=0.15`, `comp_gc_read=0.5`, verified `v_apical` ∈ −90…+50 mV and CA1 sparse+specific before trusting any
number). Prediction: on a sparse, near-disjoint, fact-specific code the `ca1→slot` write localizes without any dendritic
gate. A1 may simply close. **Nothing from this arc should be cited until re-measured in a physically valid regime.**

## Reconciliation of the probe-vs-standalone discrepancy, and the state of the re-tune

The physiological probe run initially reported **0 CA1 spikes / `dw`=0**, contradicting a standalone measure of the same
R/gc pair (13–30 active). **Cause found:** `_consol_twosided_generalize_probe.run_seed` DEFAULTS to
`hippo_izh_type="IZH2007_STRIATAL_MSN"` — the sparse phenotype introduced EARLIER IN THIS ARC specifically to fight the
(artifactual) dense code. MSN is down-state-stable with `vt=−25 mV`; at the pathological apical it was driven, but on a
physically valid substrate it cannot be driven at all. **The phenotype was a compensation FOR the artifact.** With the
default hippocampal pyramidal phenotype (`--hippo-izh-type ""`) CA1 fires again and the clamp stays exclusive
(target −12.0 mV vs non-target −66.8 mV; `plateau_v_hold=−50` ⇒ non-target `IS` is exactly 0).

**Confirmed on the valid substrate (seed 42/43):** codes are **near-disjoint** — cosine specificity `[73.8, 2.77, 2.79]`
and cases of literally zero overlap — versus the artifact's uniform 1.35.

**The re-tune is OPEN, and it is a genuine tuning problem, not a boundary.** Every constant in this arc was fitted to
the artifact's ~100× inflated activity and must be re-derived:
- **tag drive is SATURATED** — 3000 vs 6000 pA gives 101 vs 102 spikes (identical); CA1 drive is not the lever.
- **`core_thr_frac`** (≥25% of a 40-step window = 10 spikes) was calibrated for the artifact; on real activity it yields
  1–3-cell cores. Lowering it to 0.1 helps but interacts with the next item.
- **`btsp_lr=3e-6`** was fitted to the artifact's firing; on real activity it gives `dw ≈ −0.002` (no write). Raising it
  to 1e-2 produces a real write (`dw=987`) **but corrupts the ENCODE phase** — BTSP is active during
  `encode_facts_with_reinstatement`, so a high lr alters the codes *before* they are measured (`core_sizes=[3,7,112]`).
  ⇒ **the write-phase and encode-phase learning rates must be separated** in the re-tune.
- own/other remains ~1.0 at every point tried so far, **with the permuted-core and random-CA1 controls also ~1.0** — i.e.
  currently an honest null, not an artifact.

**⇒ HONEST STATE: the boundary is VOID (established); whether the write localizes on a valid substrate is UNKNOWN and
requires a structured re-derivation of the consolidation operating point** — physiological `apical_R`/`gc_read`, default
pyramidal phenotype, an activity-matched `core_thr_frac`, separated encode/write learning rates, and only then the
own/other measurement with the full mass triad at 6 seeds. Ad-hoc sweeping from the artifact's constants is not the way
in; the operating point should be derived from the substrate's own measured activity statistics.

## The CALIBRATED operating point (derived from measured activity, not swept from the artifact's constants)

New probe `research/runners/_consol_operating_point_calibration.py`: checks `v_apical` against physiological range,
measures the CA1 per-cell spike-count distribution under an isolated tag, DERIVES the core threshold from that
distribution (target core band), and reports code quality with the mass triad. Result (R=0.15, gc_read=0.5, default
pyramidal phenotype):

| | seed 42 | seed 43 |
|---|---|---|
| `v_apical` range | **[−5.32, −3.41] mV — PHYSIOLOGICAL ✓** | **[−5.33, −3.44] mV ✓** |
| CA1 total spikes / window | `[159, 191, 307]` | `[162, 222, 61]` |
| per-cell percentiles (50/90/95/99) | `0 / 0–12 / 12–14 / 18–20` | `0 / 0–8 / 1–15 / 17–20` |
| cells active (>0 of 120) | `[13, 15, 30]` (11–25%) | `[12, 19, 9]` |
| **derived core threshold** | **9 spikes** → cores `[9, 12, 20]` | 1 → cores `[10, 16, 3]` |
| **core Jaccard** | **0.101** | **0.032** |
| **core cosine specificity** | **[6.71, 7.75, 3.59]** | **[25.3, 8.95, 13.86]** |
| rate cosine specificity | `[6.55, 6.44, 3.29]` | `[21.2, 7.44, 10.18]` |

**The median cell fires 0 spikes while the 95–99th percentile fires 12–20** — a textbook sparse hippocampal code, with
**near-disjoint cores (Jaccard 0.03–0.10)** and **fact-specificity 3.6–25.3**. Contrast the artifact regime: 93% active,
Jaccard 0.877, specificity 1.35. **This is the separability a selective `ca1→slot` write is supposed to exploit, and it
was present in the substrate the whole time — hidden under the miscalibration.**

**Constants now FIXED for the write experiment** (supersede every value used in the arc): `comp_apical_R=0.15`,
`comp_gc_read=0.5`, default hippocampal pyramidal phenotype (NOT MSN), core threshold ≈ 9 spikes / 40-step window
(derive per seed), tag drive 1500 (saturated above this — not a lever).
**One honest caveat carried forward:** per-fact drive is uneven (seed 43 fact 2: 61 spikes, 3-cell core vs facts 0/1 at
10–16 cells), so a per-fact write result must be read against per-fact core size, never pooled.
**The one piece still to build before the write experiment:** separate the **encode-phase** `btsp_lr` from the
**write-phase** one — BTSP runs during `encode_facts_with_reinstatement`, so a write-scale lr corrupts the codes before
they are measured (observed: `core_sizes=[3,7,112]` at lr=1e-2).

## FIRST FULLY-CONTROLLED WRITE MEASUREMENT ON THE VALID SUBSTRATE — an HONEST NULL (all confounds removed)

Two further bugs fixed first, both of the arc's recurring class:
1. **encode/write lr separation** — BTSP runs during `encode_facts_with_reinstatement`; a write-scale lr corrupts the
   codes before measurement. Encode now runs with the write rule quiescent (`--encode-btsp-lr 0`).
2. **the MEASUREMENT was PLASTIC** — restoring the write lr right after encode left BTSP learning *during* the
   `fire_under_tag` read, so `core_sizes` varied with the WRITE lr (`[2,1,2]` → `[22,120,120]` → `[56,117,114]`) even
   though cores are defined pre-write. The lr is now restored only immediately before the write; cores are confirmed
   **invariant to write lr**. *A measurement must never be plastic.*
3. **config aligned to the calibration** — the write probe defaulted to `--commit-top-k 15` while the calibration used
   ~85; matching them reproduces the calibrated cores.

**Result (seed 42, calibrated constants: R=0.15, gc_read=0.5, pyramidal, core_thr_frac 0.225 ⇒ 9 spikes, k=85,
cycles 10, settle 200, encode lr 0):**
| lr | `dw` | core sizes | **own/other** | permuted-core | random-CA1 | per-slot weight |
|---|---|---|---|---|---|---|
| 0.01 | 1017 | `[9, 20, 22]` | **`[0.895, 0.942, 1.005]`** | `[1.00, 1.02, 1.04]` | `[1.04, 1.02, 0.94]` | `[336.8, 337.9, 332.1]` |
| 0.1 | 1695 | `[9, 20, 22]` | **`[0.915, 0.998, 1.008]`** | `[0.94, 1.05, 1.07]` | `[1.02, 1.03, 0.95]` | `[546.6, 573.3, 556.2]` |
Cores match the calibrated reference; `v_apical` physiological; **per-slot weights balanced within 2%** (vs the arc
artifact's `[24, 80, 24]` — the winner-slot bias is GONE); controls collapse to ~1.0. **own/other is FLAT, 0/3 facts.**

**This is an honest null on a valid substrate** — categorically different from the arc's artifact-driven null, because
every confound that produced the earlier numbers has been removed and verified absent.

**Leading hypothesis for the null (untested): SOFT-BOUND SATURATION compresses the rank-1 write.** BTSP is
`dw[k→j] = η·Ẽ[k]·IS[j]·(w_max − w)`, a rank-1 outer product of a per-PREsynaptic eligibility and a per-POSTsynaptic
instructive signal — which, with an exclusive IS, *should* write fact i's pattern into slot i alone. But over
10 cycles × 30 steps the `(w_max − w)` soft bound drives every eligible synapse toward the same ceiling, erasing the
graded pattern (per-slot means all ≈336 of w_max=2000). Note the suspicious discontinuity: lr 1e-3 → `dw ≈ −0.10`
(no write at all, and NEGATIVE) while lr 1e-2 → `dw ≈ 945` — a 10× lr change giving a ~10⁴× `dw` change, which is not a
smooth response and suggests a threshold/depression interaction that must itself be explained.
**▶ NEXT: find the UNSATURATED graded regime** (sweep lr across the 1e-3…1e-2 gap and/or cut cycles, requiring
`dw ≪ w_max` AND a non-degenerate write), explain the negative-`dw` low-lr branch, then re-measure own/other with the
mass triad at 6 seeds. If the write remains flat in a genuinely graded, unsaturated regime on this substrate, THAT is
the real boundary — and it will be the first one in this arc measured on physically valid dynamics.

## ✅ THE WRITE LOCALIZES — saturation was the suppressor, and the null was a saturated operating point

The honest null above was itself an operating-point artifact — the third and last one. **BTSP's soft bound was
saturating the write.** The rule is `dw[k→j] = η·Ẽ[k]·IS[j]·(w_max − w)`: a **rank-1 outer product** of a
per-PREsynaptic eligibility and a per-POSTsynaptic instructive signal. With an exclusive `IS` this *should* write fact
i's pattern into slot i alone — but if `η` is large enough that every eligible synapse runs into `(w_max − w)`, the
graded pattern is crushed to a common ceiling and all slots end up with the same vector. Sweeping η down through the
saturation knee recovers it, monotonically:

| lr | `dw` (w_max = 2000) | own/other (seed 42) | own-is-max |
|---|---|---|---|
| 0.01 | 917 — **saturated** | `[0.95, 0.86, 1.04]` | 1/3 |
| 0.005 | 0.63 | `[1.26, 1.20, 1.03]` | 2/3 |
| 0.002 | 0.30 | `[1.83, 2.03, 1.64]` | 3/3 |
| 0.001 | 0.19 | `[2.02, 4.56, 2.51]` | 3/3 |
| **0.0005** | **0.18 — graded** | **`[2.33, 4.55, 3.23]`** | **3/3** |

**Seeds 42 / 44 at the converged point (lr 0.0005–0.001), with the full mass triad:**
| seed | own/other | own-is-max | n_pass (≥2.5) | **permuted-core** | random-CA1 | per-slot mass |
|---|---|---|---|---|---|---|
| 42 | `[2.33, 4.55, 3.23]` | **3/3** | 2/3 | `[0.16, 0.70, 0.75]` | `[0.89, 0.95, 1.17]` | 0.69/0.76/0.75 |
| 44 | `[4.24, 7.92, 4.61]` | **3/3** | **3/3** | `[0.08, 0.36, 0.79]` | `[1.34, 0.31, 1.74]` | 0.66/0.72/0.69 |
- **The permuted-core control collapses far BELOW 1.0 in every case** (0.08–0.79) while the true cores reach 2.3–7.9 ⇒
  the selectivity is EARNED by each fact's own core, not by slot mass (which is balanced within ~15%).
- **Seed 44 has CONVERGED** — lr 0.001 and 0.0005 give near-identical values (`[2.98, 7.92, 4.61]` vs
  `[4.24, 7.92, 4.61]`), so this is a structural fixed point, not ratio-noise from a vanishing `dw`.

**⇒ The `ca1→slot` consolidation write DOES localize on a physically valid substrate.** The "boundary" was, end to end,
three stacked operating-point errors: a 333× units miscalibration (→ a 93%-dense pseudo-code), a phenotype patch adopted
to fight that artifact (MSN, undrivable at physiological voltages), and a saturating learning rate fitted to the
artifact's ~100× inflated activity. None of them were properties of the substrate or of the biology.

**Still open / not claimed:** 6-seed completion is running (43/100/101/102); seed 42 clears the 2.5 gate on 2/3 facts,
seed 44 on 3/3, so the formal multi-seed GO is not yet declared. And this remains the **PROXY** metric — A1 closes only
on the end-to-end hippo-lesioned recall test with its four anti-cheats (see `GAP_CLOSURE_MISSION.md`).

## ✅✅ 6-SEED GO (proxy metric) — the `ca1→slot` consolidation write localizes, fully controlled

Operating point: `comp_apical_R=0.15`, `comp_gc_read=0.5`, default hippocampal pyramidal phenotype, `core_thr_frac=0.225`
(≈9 spikes, derived from measured activity), `commit_top_k=85`, blocked schedule, single burst + 200-step (100 ms)
recovery, `--encode-btsp-lr 0` (write rule quiescent during encode AND during the measurement), `btsp_lr=0.0005`
(unsaturated: `dw` 0.12–0.18 vs `w_max` 2000).

| seed | own/other | own-is-max | n_pass (≥2.5) | **permuted-core** | random-CA1 | per-slot mass | core sizes |
|---|---|---|---|---|---|---|---|
| 42 | `[2.33, 4.55, 3.23]` | **3/3** | 2/3 | `[0.16, 0.70, 0.75]` | `[0.89, 0.95, 1.17]` | 0.69/0.76/0.75 | `[9,20,22]` |
| 43 | `[2.55, 5.74, 3.85]` | **3/3** | **3/3** | `[0.18, 0.49, 0.51]` | `[0.91, 0.26, 2.68]` | 0.67/0.69/0.71 | `[10,13,20]` |
| 44 | `[4.24, 7.92, 4.61]` | **3/3** | **3/3** | `[0.08, 0.36, 0.79]` | `[1.34, 0.31, 1.74]` | 0.66/0.72/0.69 | `[7,13,6]` |
| 100 | `[4.14, 2.74, 4.05]` | **3/3** | **3/3** | `[0.34, 0.70, 0.07]` | `[1.04, 1.28, 0.73]` | 0.66/0.71/0.71 | `[6,13,13]` |
| 101 | `[4.30, 4.80, 3.60]` | **3/3** | **3/3** | `[0.29, 0.45, 0.58]` | `[1.46, 0.75, 0.88]` | 0.67/0.70/0.69 | `[12,17,13]` |
| 102 | `[2.48, 4.34, 3.69]` | **3/3** | 2/3 | `[0.27, 0.46, 0.48]` | `[0.51, 0.64, 2.51]` | 0.66/0.70/0.70 | `[8,22,16]` |

**Verdict: GO.** `own-is-max` on **18/18 fact-seeds**; **16/18** clear the 2.5 gate (the two misses, 2.33 and 2.48, are
marginal); **≥2/3 facts pass on 6/6 seeds** (4 seeds at 3/3). Mean true own/other **4.06** vs mean permuted **0.43** —
a ~9.4× separation.

**Verify-go triad, all satisfied:** (a) the **permuted-core control collapses far below 1.0 in every one of the 18
fact-seeds** (0.07–0.79) while the true cores reach 2.3–7.9 ⇒ selectivity is earned by each fact's own core, not slot
mass; (b) **per-slot masses are balanced within ~15%** (0.66–0.76) — the winner-slot bias that faked earlier "positives"
is absent, and the selectivity runs *against* the residual mass gradient; (c) reported **per fact, never as a mean**,
with a degenerate-gate guard — all core sizes 6–22 cells, none degenerate. Additionally seed 44 is **converged** across
two learning rates (0.001 vs 0.0005 ⇒ `[2.98,7.92,4.61]` vs `[4.24,7.92,4.61]`), so this is a structural fixed point,
not ratio-noise from a vanishing `dw`.

**⚠️ SCOPE — this is the PROXY, not the capability.** A1 closes only on the end-to-end test in
`nmda_compositional_consolidation.py` main(): cue a noun with the **hippocampus lesioned**, the bound adjective pool is
selectively active, `--min-recall 2`/3 with `--antichance 1` across its four anti-cheats (no-replay · nmda-lesion ·
hippo-lesion-before-consolidation · no-confab on the withheld "cat" fact). That runner still uses the original
`coactivation_replay` and the arc's (invalid) constants, so the next step is to wire the calibrated operating point and
this write protocol into it and run the 4-control GO at 6 seeds.

## ⚠️ SCOPE CORRECTION — the 6-seed GO is on a DIFFERENT pathway than the A1 capability test

I had written that the 6-seed proxy GO would close A1 once run through the end-to-end test. **Tracing the code, that
mapping is wrong and is corrected here:**
- **The A1 capability test** (`nmda_compositional_consolidation.py` main → `measure_recall`) cues a NOUN and reads the
  **adjective pools**; its load-bearing cortical store is **`cross_pool_concept`** (noun→adj), consolidated by
  `run_concept_replay_phase` with that gate open. The route is `tag → CA1 → concept pools → cross_pool_concept`.
- **The 6-seed GO** is on **`ca1→comp_attr`** — the *dedicated compositional-attractor SLOT* route (the later "Option 1"
  addition), which is **additive and DEFAULT-OFF in that test** (`comp_attractor_slots=0` ⇒ byte-identical).
These are different pathways. The slot-write GO is a real result about what the substrate can do, but it is **not** the
concept-pool binding the A1 gate measures.

**A second consequence, in the other direction:** the main A1 runner never sets `comp_dendritic` (default `False`, and
no CLI flag for it), so **the original A1 capability test never used the two-compartment apical and was never affected
by the 333× miscalibration.** The miscalibration invalidated *this arc's probes*, not the A1 test. The VOID scope above
stands as written for the arc's probe results, and should not be read as touching the pre-arc A1 status.

**⇒ Two distinct, separately-valid threads, and neither should be cited as the other:**
1. **Slot route (this arc's GO):** a selective `ca1→slot` write is achievable at a physically valid operating point in
   the unsaturated regime — 6-seed, fully controlled. Open: give it a capability read-out (recall keyed on slots).
2. **Concept-pool route (the original A1 gate):** unaffected by the miscalibration; its current end-to-end status is
   being re-measured from the runner's own defaults to establish ground truth before anything is wired.

## A1 END-TO-END BASELINE: the capability test currently fails its OWN POSITIVE CONTROL — the blocker is UPSTREAM of consolidation

Ran the untouched A1 capability test at its own defaults (`--seeds 42 --diagnostic`, 15.9 min) to establish ground truth
before wiring anything:
```
direct-binding sanity: 1/16 = 6.2%
[full        ] recalled 0/3  selective 1/3  lifted 0/3 | no-confab top=0.0010 confab=False | xpool_w=0.000
[no_replay   ] recalled 0/3 ... xpool_w=0.000
[nmda_lesion ] recalled 0/3 ... xpool_w=0.000
[hippo_before] recalled 0/3 ... xpool_w=0.000
--> SEED 42 NO (grounded 0/3 | a=True b=True c=True noconfab=True)
```
**The NO verdict is UNINFORMATIVE about the capability**, per the negative-verification rule (`verify-go`, rule 5 — *if
the harness cannot demonstrate the effect where it MUST exist, the harness is what you measured*):
- the **positive control fails**: direct-binding sanity **1/16 = 6.2%**;
- **`xpool_w = 0.000` in EVERY arm including `full`** — the cortical `cross_pool_concept` store is never written at all,
  so there is nothing for replay to consolidate and nothing for recall to read;
- consequently all four anti-cheat controls "pass" (`a=b=c=noconfab=True`) **trivially** — every arm is zero, so the
  controls discriminate nothing. A gate that passes because everything is zero is not a gate.

**NOT a regression from this arc — verified.** `git diff <pre-arc>..HEAD -- research/runners/nmda_compositional_consolidation.py`
shows **zero removed or changed lines** (every arc edit was purely additive/default-off), so the default code path is
byte-unchanged. This is a **pre-existing** state of the A1 harness.

**⇒ Reframing of the A1 work: the blocker is UPSTREAM of consolidation.** Phase-1 word→pool direct binding is not
working (6.2%), so the compositional-consolidation question — which presupposes a bound noun/adjective representation to
consolidate — **cannot yet be asked** on this harness. Improving the consolidation mechanism cannot move a test whose
input is empty. **▶ NEXT for the concept-pool thread: restore the direct-binding sanity first** (diagnose Phase-1
`train_word_to_pool` / the pool-drive operating point — note this harness has its own constants, and per the lesson
above they should be checked in physical units before anything is concluded), and only then re-measure the four-control
capability gate. The slot-route 6-seed GO above is unaffected and remains the arc's standing positive result.

## A1 upstream blocker LOCALIZED: Phase-1 training changes no weights at all

Direct instrument test (rather than more reading): build the A1 substrate at the runner's own defaults, measure the
`language_input→pool` gate weights, run 200 `train_word_to_pool` events for `apple`→`noun_pool_APPLE`, re-measure.
```
enable_stdp=True  enable_hebbian=False        <- the main runner's DEFAULT (--enable-hebbian is store_true)
  BEFORE language_input_to_noun_pool: 3.01526   AFTER: 3.01509   (d = -0.00017)   <- the TRAINED pathway
  BEFORE language_input_to_adjective_pool: 3.01305  AFTER: 3.01286 (d = -0.00019) <- untrained control
  BEFORE language_input_to_motor: 3.01236          AFTER: 3.01220 (d = -0.00016)  <- untrained control
```
**The trained pathway moves by the same amount as the untrained controls** — a uniform ~-0.0002 drift (decay), not
learning. **Phase-1 word→pool training is a no-op**, which fully explains direct-binding sanity 1/16 = 6.2% (chance),
`xpool_w=0.000`, and the trivially-passing anti-cheats. The consolidation question was never reachable on this harness.

**Leading hypothesis under test: `enable_hebbian=False` is the A1 runner's default**, while this arc's probe `BASE`
config sets `enable_hebbian=True`. If the teacher-current word→pool protocol is Hebbian-dependent (STDP alone being
insufficient for the co-driven, order-free pairing this protocol produces — cf. the project's own finding that STDP is
the WRONG rule for symmetric co-occurrence, 656k events / 0 weight change at Δt≈0), then the A1 test has been running
with its learning rule effectively disabled. Test in flight: identical 200-event protocol at
`enable_hebbian=True` vs `False`, trained gate vs untrained control.

**Wider implication worth checking beyond A1:** a `store_true` flag means the DEFAULT is off, and an absent flag in a
recorded command reads as "off" — the same class as the documented `.cmd.json` gotcha (an absent flag means *default*,
not *disabled*). Any A1-family result recorded without `--enable-hebbian` may have been produced with the learning rule
off; that should be checked before any of them is cited.

### A1 upstream blocker — full localization (gates OPEN, neurons FIRING, rule cannot bind)

Ruled out, each by direct measurement rather than inspection:
| candidate | measurement | verdict |
|---|---|---|
| drive not reaching | one instrumented training event: `language_input` **881 spikes**, TARGET `noun_pool_APPLE` **4230 spikes** (200 cells), non-target pool **23** | **REFUTED** — the protocol produces exactly the right pattern (input active, target driven, non-target quiet) |
| plasticity gate closed | `_plasticity_gate_to_synapses` → `language_input_to_noun_pool` gain **mean 1.0000 / max 1.0000** (492k synapses); `cross_pool_concept` and `ca1_to_concept_pool` likewise **1.0** | **REFUTED** — gates fully open |
| `enable_hebbian=False` is the bug | 200 events, trained gate vs untrained control: `hebbian=True` → **−2.31 trained, −2.31 control**; `hebbian=False` → −0.00018 vs −0.00016 | **REFUTED as a fix** — Hebbian ON collapses everything UNIFORMLY (global decay/scaling dominates); OFF changes nothing. Neither yields *differential* change |

**⇒ The rule cannot bind this protocol.** The teacher protocol CO-DRIVES input and target simultaneously, which is a
**symmetric, order-free pairing** — and this project has already measured that **STDP is the wrong rule for exactly
that**: *"656k events / 0 weight change at Δt≈0, because symmetric co-occurrence has no pre→post order"* (the on-bridge
Hebbian co-occurrence finding, CLAUDE.md). STDP is the only rule active at the A1 runner's defaults. The rule that
*should* apply — rate-Hebbian co-occurrence — is off by default, and when switched on at this operating point its
decay/scaling term swamps potentiation (uniform −2.31 on trained and control alike).

**⇒ A1's upstream blocker is a RULE-SELECTION + OPERATING-POINT problem, not a substrate limit** — structurally the
same shape as the saturation defect that was suppressing the slot write (a learning rule at an operating point where its
own decay/bound dominates the signal). **▶ NEXT: find the Hebbian operating point at which co-activation produces
SELECTIVE potentiation** (sweep `hebbian_learning_rate` against the decay/scaling terms, requiring trained-gate Δ ≫
untrained-control Δ — the control comparison is the whole test), then re-run direct-binding sanity, and only then the
4-control A1 gate. Do NOT tune this against the 16-word sanity score directly; tune it against the trained-vs-control
weight delta, which is cheap, immediate, and cannot be faked by chance.

### ROOT CAUSE of the uniform Hebbian collapse: `hebbian_max_weight=1.0` vs design weights of 3.015 — the STDP `w_max` gotcha, recurring on the Hebbian rule

`sim/config.py:535` sets **`hebbian_max_weight: float = 1.0`**, while the `language_input→pool` pathway is built at
**mean weight 3.015**. CLAUDE.md already documents this exact failure mode for the *other* rule:
> **STDP bounds gotcha:** `stdp_w_max=2.0` default. The rule is **soft-bound** so when `weight_mean > stdp_w_max`,
> every "LTP" event is strongly negative and weights collapse to `w_max` within ms.

**The same trap exists on the Hebbian rule and nothing warns about it.** With weights at 3.015 above a bound of 1.0,
every Hebbian update is strongly negative — which is precisely the measured **uniform −2.31 collapse on the trained
pathway AND the untrained control** (3.015 → 0.704). Enabling the correct rule made things *worse than doing nothing*,
and did so in a way that looks like "Hebbian doesn't help here" rather than "the bound is misconfigured."

Also noted for the fix: **`hebbian_rate_window` (default `False`)** is the rate-based co-activity-trace variant, whose
own config comment states the alternative `hebbian_symmetric` rule "only potentiates on EXACT same-step co-spikes, which
are rare" — i.e. the rate-window rule is the one designed for exactly the symmetric, co-driven pairing this teacher
protocol produces. At the A1 defaults it is off.

**Pattern (not an incident) — a `*_max_weight` / `*_w_max` bound BELOW the design weights silently inverts its own
learning rule.** It has now bitten this project on STDP (documented), on BDSP (`bdsp_w_max` clamp applied even at
`lr=0`, documented in CLAUDE.md), on BTSP (the saturation that suppressed the slot write, this session), and now on
Hebbian. **Standing check: before enabling ANY plasticity rule, compare its bound against the actual pathway weight —
`_mean_gate_weight(bridge, gate)` vs `cfg.<rule>_max_weight` — and verify the trained pathway moves DIFFERENTLY from an
untrained control.** A rule whose bound sits below the weights does not merely fail to learn; it actively destroys the
weights, uniformly, which reads as a substrate limitation.

### ⚠️ RE-DIAGNOSIS: the A1 sanity failure is UNDER-TRAINING at the default, not a rule/config defect

Before concluding the Hebbian/homeostasis chase, I checked whether this harness's Phase-1 has EVER worked. It has —
`research/findings/2026-05-21-catastrophic-forgetting-multi-seed-...md` records:
> *"Pre-silence direct binding (16-word task) at **800ev** saturated training: seed 42 = **15/16**, seed 43 = 13/16,
> seed 44 = 14/16 (multi-seed aggregate 42/48 = **87.5%**)"* … *"the substrate has near-saturated direct binding at 800ev"*

**The A1 runner defaults to `--train-events 200`** (per target: 16 targets × 200 = the observed 3200 events), i.e. **4×
below the validated 800ev budget**, and yields 1/16 = 6.2% = chance. **⇒ direct binding is achievable on this substrate
family; the A1 default is a FAST setting that does not train enough to bind.** The earlier chase (Hebbian bound,
rate-window, homeostasis) was diagnosing a non-bug: gates were open, neurons fired correctly, and the rule was fine —
there simply was not enough training. *(The `hebbian_max_weight=1.0` vs 3.015 inversion recorded above is still a REAL
and separate defect — enabling that rule actively destroys weights — but it is not the cause of the sanity failure.)*

**Corroborating detail from the 2-word probe:** the two TRAINED pools were the QUIETEST (APPLE 0.035, RIVER 0.015) while
UNTRAINED pools dominated (DOG 0.215, CAT 0.145). Under-training plus homeostatic suppression of the heavily-driven
target reproduces exactly that inversion — the teacher drive (4230 spikes) triggers rate regulation without yet having
built compensating input weight.

**HONEST CAVEAT:** the 800ev/87.5% figure comes from the v14-only / Unified substrate runs of 2026-05-21, a related but
not necessarily identical harness/config to this A1 runner. It establishes achievability on the substrate family, not
that 800ev alone fixes THIS runner. **▶ TEST IN FLIGHT: the A1 harness's own Phase-1 at `--train-events 800`** (≈4×
14.4 min ≈ 1 h/seed), scored on its own `direct_binding_sanity`. If it clears, the A1 capability gate becomes
measurable for the first time in this arc and the four-control run can proceed; if it does not, the difference between
this runner and the 2026-05-21 harness is the next thing to isolate.

### ⚠️ SCOPING CHECK BEFORE BUILDING THE SLOT CAPABILITY READ-OUT: `ca1→slot` CANNOT deliver hippo-independent recall

The recorded next step for the slot thread was "give the 6-seed-GO slot write a capability read-out". Checked whether
that is buildable BEFORE building it — it is not, as specified:
- the A1-style capability test **lesions the hippocampus** and then cues the fact;
- **CA1 *is* hippocampus**, so lesioning it removes the `ca1→comp_attr` pathway the 6-seed GO trained;
- ⇒ **a selective `ca1→slot` write cannot, by construction, support hippo-independent recall.** It is the *hippocampal
  reinstatement* half of consolidation, not the lasting cortical store.

**What the slot route WOULD need:** the substrate already wires a cortex-resident path —
`RegionPathway(from_region=<noun/adj pool>, to_region=comp_attr_s, density=0.15, weight_mean=1.5, plastic=True,
plasticity_gate="concept_to_comp_attr")` (runner :293-298). **For a hippo-lesioned slot recall the LOAD-BEARING store is
`concept_to_comp_attr` (cortex→cortex), not `ca1→comp_attr`.** The `ca1→slot` selectivity matters only insofar as it
drives the correct slot DURING replay so that the cortical `concept→slot` weights are written selectively — exactly the
role `ca1→concept_pool` plays for `cross_pool_concept` on the original A1 route.

**⇒ CORRECTED next step for the slot thread:** measure and, if needed, make **`concept_to_comp_attr` selective** under
the calibrated operating point (same instrument: per-fact own/other with the mass triad — permuted-core control,
magnitude-free twin, per-fact passes), THEN a hippo-lesioned slot recall becomes a meaningful capability read-out. The
`ca1→slot` 6-seed GO stands as a validated *component* (the replay-time reinstatement), not as the capability.

**Process note:** this is the pre-flight the session's own lessons demand — *verify the pathway can deliver the
capability before building the read-out*. Building it as originally specified would have produced a guaranteed null
(everything lesioned away) that would have looked like yet another "boundary".

### ⛔ UNDER-TRAINING HYPOTHESIS REFUTED — 800ev is WORSE (0/16). The rule/config diagnosis is REINSTATED.

Ran the A1 harness's own Phase-1 at `--train-events 800` (12,800 events, 57.8 min):
```
Phase-1 done (12800 events, 57.8 min)
direct-binding sanity: 0/16 = 0.0%          <- 200ev gave 1/16 = 6.2%; 4x MORE training is WORSE
[full] recalled 0/3 ... xpool_w=0.000       (all four arms identical to the 200ev run)
--> SEED 42 NO
```
**⇒ my "the A1 failure is UNDER-TRAINING, the Hebbian chase was a non-bug" re-diagnosis was WRONG and is retracted.**
More training does not help; it hurts. The earlier rule/config line was correct.

**The coherent mechanism, now supported by every measurement taken:**
1. `enable_hebbian` is **False** at the A1 defaults ⇒ **STDP is the only active rule**.
2. The teacher protocol **co-drives** input and target simultaneously — a symmetric, Δt≈0 pairing — and this project has
   already measured that **STDP cannot learn exactly that** (*656k events / 0 weight change at Δt≈0*). Hence the
   trained pathway moving identically to untrained controls (−0.00017 vs −0.00019).
3. Meanwhile the teacher drive is strong (target pool 4230 spikes/event) and triggers **homeostatic rate regulation**,
   which SUPPRESSES the heavily-driven target. The 2-word probe showed exactly this inversion: TRAINED pools were the
   quietest (APPLE 0.035, RIVER 0.015) while UNTRAINED pools dominated (DOG 0.215, CAT 0.145).
4. ⇒ **more training buys no binding but accumulates more suppression** — precisely the observed 200ev 1/16 → 800ev 0/16.

**The 87.5%-at-800ev record therefore comes from a materially different harness/config** (the 2026-05-21 v14-only /
Unified substrate runs). The caveat recorded when I cited it was correct and is now load-bearing: **isolating what that
harness did differently is the concrete next step**, and is cheaper than further knob-search here.

**LIVE LEAD (from the Hebbian work that the retracted re-diagnosis wrongly dismissed):** with the bound corrected
(`hebbian_max_weight` 1.0 → 8.0, above the 3.015 design weights) a **selective signal exists and scales with the
learning rate** — trained-minus-control **+0.0056 at lr 5e-4 → +0.0516 at lr 5e-3** — but is swamped by a uniform
−0.775 decay (`hebbian_weight_decay` over ~30k steps, affecting trained and control alike). **▶ NEXT: (a) diff this
runner's Phase-1 config against the 2026-05-21 harness that scored 87.5%; (b) tune Hebbian lr vs decay (and check
whether homeostasis must be quiesced during teacher-driven training) until trained-minus-control ≫ 0, scoring on the
weight delta, not the 16-word accuracy.**

**Process note — two reversals on one sub-question.** The under-training hypothesis was well-motivated (a real recorded
87.5% at 4× the default) and was refuted in one run. That is the system working: it was recorded as a hypothesis with an
explicit caveat about harness identity, tested directly, and retracted on the evidence rather than defended.

### 🎯 CONFIG DIFF vs the 87.5% harness: the A1 runner has **global NMDA OFF by default**

Followed the "diff against the known-good config" step rather than more knob-search. The 87.5% direct-binding result
came from `research/findings/raw/unified_per_regime/…` ⇒ the **`unified_per_regime_monitor_runner`**, which the A1
runner already imports its Phase-1 recipe/kwargs from (so the TRAINING protocol is shared). Comparing the two substrate
configs:

| setting | unified (87.5%) | A1 runner | note |
|---|---|---|---|
| `enable_hebbian_learning` | **False** | False (default) | **identical — so STDP ALONE achieves 87.5%** |
| `enable_short_term_plasticity` | False | False | identical |
| `enable_structural_plasticity` | False | False | identical |
| `fast_spike_reset` | True | True | identical |
| `enable_per_type_stp` | False | False | identical |
| **`enable_nmda`** | **`True`** | **`bool(args.enable_global_nmda)` ⇒ DEFAULT FALSE** | **THE DIFFERENCE** |

**Two consequences, both important:**
1. **My "STDP cannot learn this symmetric co-driven pairing" mechanism is REFUTED** — the harness that scores 87.5% runs
   with Hebbian OFF, i.e. on STDP alone. The Δt≈0/656k-events finding does not transfer to this protocol. That line of
   reasoning is withdrawn.
2. **The live candidate is `enable_nmda`.** Global NMDA supplies the slow, voltage-dependent current that sustains
   post-synaptic depolarisation across the teacher window; without it the association plausibly cannot form at all —
   which is exactly consistent with **more training not helping** (200ev 1/16 → 800ev 0/16), since repetitions of a
   non-forming association accumulate only the homeostatic suppression of the driven target.

**TEST IN FLIGHT:** the A1 runner at its own defaults **plus `--enable-global-nmda`** (200ev, ~16 min), scored on its own
`direct_binding_sanity`. This is a single-variable change against the failing baseline.

**Process note:** this is the third mechanism proposed for the A1 failure (rule-can't-bind → under-training →
config-delta). The first two were each refuted by a direct test within one run of being proposed. The value of the
"check the record for a known-good configuration" rule is precisely that it replaces mechanism-guessing with a
**difference** that can be tested one variable at a time.

### `enable_nmda` REFUTED too (1/16, unchanged) — 0-for-3 on mechanisms; switching from guessing to a REFERENCE test

A1 at its own defaults **plus `--enable-global-nmda`** (single-variable change): `direct-binding sanity: 1/16 = 6.2%` —
**identical to the baseline**, `xpool_w=0.000` in all arms, seed 42 NO. **The NMDA delta is not the cause.**

**Three mechanisms proposed for the A1 harness failure, three refuted, each within one run of being proposed:**
| # | hypothesis | refuted by |
|---|---|---|
| 1 | the rule can't bind a symmetric co-driven pairing (STDP, Δt≈0) | the 87.5% harness runs **Hebbian OFF** — i.e. STDP alone — so STDP demonstrably CAN |
| 2 | under-training at the default 200ev | 800ev is **WORSE** (0/16 vs 1/16) |
| 3 | `enable_nmda` default-False (the one config delta found) | **1/16, unchanged** |

**Being 0-for-3 is the signal to stop proposing mechanisms.** Each hypothesis was cheap and honestly tested, but the
pattern says the difference is not where I keep guessing. **Switched to the decisive REFERENCE test:** build the
substrate with the reference runner's own `_build_bridge_with_phase1_recipe`, then run the **identical** training
protocol and the **identical** `direct_binding_sanity`. Single variable = the substrate builder.
- **If the reference PASSES (~87.5%)** ⇒ the delta lives in the A1 runner's `build_substrate` augmentations, and can be
  bisected directly rather than guessed.
- **If the reference ALSO FAILS** ⇒ a **shared-code regression since 2026-05-21**, which would be a far more important
  finding than the A1 blocker itself — it would put every result depending on that Phase-1 recipe in question.
Either outcome is decisive and neither requires another guess. Test in flight.

## 🔴🔴 THE REFERENCE HARNESS ALSO FAILS (0/16) — this is a SHARED-CODE REGRESSION, not an A1 problem

Decisive single-variable test: the **reference runner's own** substrate builder
(`unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe`, the harness whose recorded score is **87.5%**),
with the **identical** training protocol and the **identical** `direct_binding_sanity`:
```
REFERENCE substrate + identical protocol + identical sanity: 0/16 = 0.0%
```
**Word→pool direct binding is broken in the SHARED code path**, not in the A1 runner. This immediately explains why all
three A1-specific hypotheses missed: the fault was never in A1. A1's failing capability gate is a **symptom**.

**Why this matters more than the A1 blocker.** Phase-1 word→pool binding is the foundation the concept-pool line of work
stands on — every result that presupposes "the brain has learned word↔meaning links" depends on it. If it has been
silently broken for some time, results produced in that window need re-checking. Given this session has already found a
**333× units miscalibration** and a **rule-inverting weight bound** sitting unnoticed in this codebase, a silent shared
regression is entirely plausible rather than exotic.

**HONEST CAVEATS — what is NOT yet established:**
- The reference test was run at **200 events/word**; the recorded 87.5% was at **800**. Not strictly like-for-like.
  *(Mitigating: the A1 runner at 800ev also gave 0/16, so it fails at both budgets — but the REFERENCE builder has only
  been run at 200 here.)*
- "Regression" presumes it once passed **on this code**. The 87.5% came from a commit ~2 months old; that has not yet
  been re-run. **It is equally possible the recipe always needed something this reproduction omits.**
- Therefore the correct label right now is **"the reference harness does not bind today"**, NOT "we broke it on date X".

**▶ NEXT (systematic, no more guessing): run the SAME probe at the 2026-05-21-era commit in a git worktree.**
- passes there ⇒ a genuine regression; `git bisect` the exact commit that broke it (mechanical from that point).
- fails there too ⇒ the historical 87.5% depended on something this reproduction omits, and the reproduction is what
  needs fixing — which would retire the "regression" framing entirely.
Either way the next step is a measurement, not a hypothesis. **Nothing about the shared Phase-1 path should be cited
until this resolves.**

### Blast-radius check: the 6-seed slot-write GO is NOT affected by the shared word→pool breakage

Checked rather than assumed, because a broad "the shared path is broken" finding invites over-reading:
- **Neither slot-write probe ever runs Phase-1 word→pool training** — `grep -c "train_word_to_pool"` = **0** in both
  `_consol_twosided_generalize_probe.py` and `_consol_direct_weight_probe.py`.
- `encode_facts_with_reinstatement` drives the concept pools by **teacher current** ("the pools fire from lang+teacher"),
  i.e. it does not depend on *learned* `language_input→pool` weights to make pools fire.
- The GO metric is measured on `ca1→slot` weights after **tag-driven** replay — a path that never traverses the broken
  binding.
⇒ **the 6-seed slot-write GO stands independently.** The shared breakage bounds what can be claimed about the
**concept-pool / A1 capability** line, not about the slot-write component result.

**Current scoreboard for this arc, with each item's dependency stated:**
| result | status | depends on the broken path? |
|---|---|---|
| the "dense CA1 code" boundary | **VOID** (a 333× units miscalibration) | — |
| `ca1→slot` selective write, 6-seed | **GO, stands** | **no** (teacher-driven, tag-driven replay) |
| A1 end-to-end capability gate | **unmeasurable today** | **yes** — blocked by it |
| word→pool binding in the shared path | **does not bind today; cause unresolved** | is the path |

### ⚠️ "SHARED-CODE REGRESSION" RETRACTED — the 2026-05-22 code fails identically (0/16 at 200ev)

Ran the same probe against **commit `5b532756` (2026-05-22)** in an isolated git worktree — i.e. code from the era of
the recorded 87.5%:
```
OLD CODE (2026-05-22), 200ev: 0/16 = 0.0%      # current code, same probe: also 0/16
```
**Identical failure on old and current code ⇒ nothing regressed.** The "shared-code regression" framing is **withdrawn**.
The caveat recorded when that framing was proposed — *"'regression' presumes it once passed ON THIS CODE, which has not
been re-run; the correct label today is 'does not bind', not 'we broke it'"* — is exactly what stopped this becoming a
false alarm about a swath of past results. **Recording the caveat did the work.**

**What the evidence now actually says**, assembled:
| harness | events/word | result |
|---|---|---|
| reference (recorded, 2026-05-21) | **800** | **87.5%** |
| reference, CURRENT code (this session) | 200 | **0/16** |
| reference, OLD 2026-05-22 code (this session) | 200 | **0/16** |
| A1 runner | 200 | 1/16 |
| A1 runner | 800 | 0/16 |
⇒ **200 events/word is simply insufficient for the reference harness** — on any code version. My reproduction was
under-trained, and I tested the under-training hypothesis on the WRONG harness (the A1 runner, which turns out to carry
an additional problem of its own, since it fails at 800ev too).

**⇒ Partial vindication of the retracted under-training hypothesis, with the correction that matters:** under-training
was real *for the reference reproduction*; it is NOT the whole story for the A1 runner, which fails at 800ev where the
reference reportedly succeeds. **▶ DECISIVE TEST IN FLIGHT: reference harness, CURRENT code, 800 events/word.**
- ≈87.5% ⇒ the shared path is **healthy**; the A1 runner's own delta is the entire remaining problem, and it can be
  bisected against a known-good reference instead of guessed at.
- still ~0 ⇒ the recorded 87.5% depended on something neither harness reproduces today, and THAT becomes the question.

### Static elimination round (done while the reference@800ev test runs) — candidate A1 deltas RULED OUT

Comparing the A1 substrate builder against the reference builder parameter-by-parameter. **A first range-limited diff
was misleading** — it made several settings look A1-only that the reference also sets. Corrected comparison:
| parameter | reference | A1 | verdict |
|---|---|---|---|
| `concept_pool_exc_weight_mean` | `0.3` (:471) | `0.3` | **identical** |
| `concept_pool_internal_density` | `0.05` (:470) | `0.05` | **identical** |
| `concept_pool_inh_weight_mean` | `0.8` (:472) | `0.8` | **identical** |
| `enable_hippocampus_consolidation` | `True` (:505) | `True` | identical |
| `enable_dlpfc_verb` | `True` (:507) | `True` | identical |
⇒ the "WEAK concept dynamics" choice A1 documents is **the reference's setting too** — not a delta.

Also ruled out: **plasticity gates during Phase-1.** A1's `train_phase1` closes all `_CONCEPT_GATES` to 0.0 (:401-402),
which looked like a strong candidate — but `train_word_to_pool` **reopens the target kind's gates per word**
("Open ONLY the target kind's gates during training"), so the closure is compensated by design. Consistent with the
direct measurement earlier: gate gain was **1.0** on all 492k `language_input→noun_pool` synapses.

**Remaining A1-only additions** (all additive wiring, none obviously load-bearing for word→pool binding):
`enable_cross_pool_concept_pathways` (the cortical noun→adj store) · the `comp_attr` slots (default-off) · dlpfc sizing.

**Status: no static explanation found for A1's 800ev failure.** Recording the eliminations because they are the useful
product — the next step is narrowed, not widened. **The reference@800ev measurement remains the arbiter**, and further
static guessing before it lands would repeat the 0-for-4 pattern.

### ⚠️ MY RETRACTION WAS ITSELF INVALID — it rested on a comparison at a FLOORED operating point

**Reference harness, CURRENT code, 800 events/word: `2/16 = 12.5%`** — against the recorded **87.5%**, and barely above
chance (1/16 = 6.25%). So the reference does **not** reproduce its recorded result today even at the correct budget.

**This invalidates the retraction two sections above.** I retracted "shared-code regression" because old (2026-05-22)
code and current code both gave 0/16 — **but that comparison was run at 200 events, where BOTH arms are floored at
zero.** A comparison between two arms that are both at the floor has **no discriminating power**: it cannot detect a
difference that exists. The conclusion "nothing regressed" did not follow from it. This is precisely the
control-integrity failure the project's own checklist names — *verify the arms genuinely differ before believing a null*
— committed while writing up a section about being careful.

**Corrected evidence table** (⚠ = the void comparison):
| harness | code | events | result |
|---|---|---|---|
| reference | recorded (May) | 800 | **87.5%** |
| reference | **current** | **800** | **12.5%** |
| reference | current | 200 | 0/16 ⚠ floored |
| reference | old (2026-05-22) | 200 | 0/16 ⚠ floored |
| A1 | current | 200 | 1/16 |
| A1 | current | 800 | 0/16 |

**⇒ the regression question is REOPENED and undecided.** The only comparison that can settle it is **OLD code at 800
events** — the one budget where the reference is known to have produced a strong result. **Test in flight.**
- old@800 ≈ 87.5% ⇒ **genuine regression** between 2026-05-22 and now; `git bisect` becomes mechanical, and every result
  resting on this Phase-1 path in that window needs re-checking.
- old@800 ≈ 12.5% ⇒ no regression; the recorded 87.5% depended on an ingredient neither harness reproduces today
  (a cached Phase-1 state, a calibration step, or a different `tiny_synth`/ckpt path), and finding it is the question.

**Lesson (new, and earned twice today): never conclude "no difference" from arms that are both at the floor or both at
the ceiling.** Saturation destroyed the slot-write signal earlier; a floor destroyed this comparison. Same defect class,
opposite end of the range.

### The fast bisect instrument is FLOORED too — and it retroactively undermines my "Phase-1 is a no-op" claim

Anticipating a possible bisect (58 min/run × ~8 steps = 10+ h), I tested whether the cheap 1-word × 200-event
weight-delta probe discriminates old from current code. **Applying the floor/ceiling rule added an hour ago — check the
instrument responds before trusting it:**
```
OLD CODE  1-word probe: trainedΔ=-0.00019  controlΔ=-0.00017  SELECTIVE=-0.00002
CURRENT   1-word probe: trainedΔ=-0.00017  controlΔ=-0.00019  SELECTIVE=+0.00002
⇒ does NOT discriminate — both arms at noise level
```
**Consequence I have to own:** this is the very probe I used earlier to conclude **"Phase-1 word→pool training is a
no-op — the trained pathway moves identically to untrained controls."** That measurement is **floored on both arms**, so
it never supported the claim. **The "Phase-1 is a no-op" conclusion is WITHDRAWN** — not because it is disproven, but
because the evidence offered for it had no discriminating power. (The reference historically reaching 87.5% at 16 words
× 800 events implies learning *does* occur at scale; a 1-word/200-event probe is simply far below the scale where the
effect lives.)

**This is the THIRD floored-comparison error in this session** — the saturating learning rate, the 200ev regression A/B,
and now this probe (which I built as an instrument for the very rule that caught it). What actually stands is only the
16-word sanity at adequate scale: **current code 12.5% @800ev vs a recorded 87.5%**.

**Consequence for the bisect plan:** there is no cheap instrument yet. Options, in order of preference:
1. a **graded margin** metric (target-pool rate minus best non-target rate) instead of 16-way argmax accuracy — a
   continuous score is far more sensitive than a binary win/lose and may discriminate at much lower training cost;
2. a reduced word set (4 words × 800ev ≈ 14 min/step ⇒ ~2 h for an 8-step bisect) — feasible but not cheap;
3. full 16-word × 800ev (58 min/step) — last resort.
Whichever is chosen, **it must first be shown to discriminate old vs current** before being used to bisect. That check
is now the precondition, not an afterthought.

## ⛔ ALL MY REFERENCE-HARNESS MEASUREMENTS ARE INVALID — my reproduction omitted `apply_concept_topographic_bias`

**old code @800ev = 3/16 (18.8%)** vs current 12.5% vs recorded 87.5%. A one-word difference on a 16-item binary
measure is noise ⇒ **regression definitively RULED OUT**. But reading the runner's own Phase-1 generator
(`_phase1_train_if_needed`, :596-640) shows why *both* of my numbers are meaningless:

> **Step 3 — `cpd.apply_concept_topographic_bias(...)`**: a Pulvermüller-style cortical somatotopy
> (`topographic_factor` 1.5 on-target / `off_target_factor` 0.7) applied to the substrate **BEFORE any training**.

**My hand-rolled reproduction skipped it entirely** (and also built `word_to_idx` via `_all_words_word_to_idx()` rather
than the runner's explicit `{w: i for i, w in enumerate(DIRECTION+NOUN+VERB+ADJECTIVE)}` ordering). Training a uniform,
un-pre-structured substrate is simply a different experiment. ⇒ **every "the reference harness fails" number I produced
today (current 12.5%, old 18.8%, both 0/16 at 200ev) is VOID**, and the **"shared-code path is broken" conclusion built
on them is WITHDRAWN** — for the fourth time on this thread, because my instrument was wrong rather than because the
claim was disproven.

**Crucially, this does NOT rescue A1: the A1 runner DOES apply the bias** (`nmda_compositional_consolidation.py:405`,
inside its own `train_phase1`) — so **A1's failure is real and remains unexplained** (1/16 @200ev, 0/16 @800ev *with*
the bias correctly applied).

**Corrected status of this whole sub-thread:**
| claim | status |
|---|---|
| shared code path is broken | **WITHDRAWN** — measured with an invalid reproduction |
| a regression between May and now | **RULED OUT** — old and current agree (18.8% vs 12.5%, noise) *but both reproductions were invalid, so even this is weak* |
| Phase-1 training is a no-op | **WITHDRAWN** earlier (floored probe) |
| **A1 fails its own binding sanity WITH the bias applied** | **STANDS — the one solid fact** |

**▶ TEST IN FLIGHT: run the runner's OWN `_phase1_train_if_needed` (bias included, its code path, not my
reconstruction) and score it with the same sanity check.** ≈87.5% ⇒ the reference is healthy, my reproduction was the
whole problem, and A1's delta-vs-reference becomes the sole remaining question — now bisectable against a genuinely
known-good baseline. Materially below ⇒ the recorded figure needs an ingredient even the runner's own path no longer
supplies (e.g. the deleted cached `.simstate.h5` states, which are **gone** — the cache dirs exist but are empty).

**LESSON: when reproducing a recorded result, CALL THE ORIGINAL CODE PATH — do not re-implement it.** Four withdrawals
on this thread trace to a reconstruction that silently differed from the thing it claimed to reproduce.

### Runner's OWN Phase-1 path, at its OWN default: 0/16 — and the default is **200 events**, not 800

Called `_phase1_train_if_needed(42, …, tiny_synth=False)` — the runner's real code path (topographic bias, its word
ordering, its interleaved schedule, its gating) — then scored with the same sanity check:
```
RUNNER-OWN Phase-1 (topographic bias INCLUDED): 0/16 = 0.0%     (~5 min, not ~58)
```
The short runtime gave it away: `_phase1_recipe(False)["n_train_events"] = **200**` — the runner's own default is 200
events/word, and **the recorded 87.5% was produced at 800**, from a separately-named cache
(`phase1_800ev_post_interference_50per/`). So this run is a *valid* measurement of the original code path, but taken at
the budget where every measurement today has been floored.

**Net: at 200 events, EVERYTHING fails — my reproduction, the runner's own path, old code, and A1 alike.** The single
un-floored data point remains the recorded 87.5% @800ev, and nothing today has yet reproduced it *on the original code
path* at that budget.

**▶ TEST IN FLIGHT — the clean one, finally:** the runner's OWN `_phase1_train_if_needed` with a **surgical override of
one documented parameter** (`n_train_events` 200 → 800) and nothing else re-implemented. This is the reproduction that
should have been run first: same bias, same ordering, same schedule, same gating, only the budget varied.
- ≈87.5% ⇒ the reference is healthy at its documented budget; everything I measured today was a floored or invalid
  proxy; **A1's failure *with* the bias at 800ev becomes the sole real anomaly**, bisectable against a known-good baseline.
- materially below ⇒ the recorded figure needs an ingredient the current code no longer supplies — and the cached
  `.simstate.h5` states that could have settled it are **deleted** (cache dirs exist, all empty).

### The clean reproduction: runner's OWN path @800ev on current code = **1/16 (6.2%)** — chance

Surgical one-parameter override (`n_train_events` 200→800), everything else the runner's own code (topographic bias,
word ordering, interleaved schedule, gating, cache write/load):
```
n_train_events now: 800
RUNNER-OWN path @800ev: 1/16 = 6.2%     [record: 87.5%]
```
**⇒ the recorded 87.5% is NOT reproducible on current code by the original code path at the documented budget.** This is
the first properly-instrumented statement in this whole sub-thread — no hand-rolled loop, no floored budget, no
1-word proxy. (Note it ran in ~13 min, not the ~58 my hand-rolled loop took for the same nominal budget — a further
sign the reconstruction differed from the original in more than the bias.)

**⇒ the regression question can now be asked PROPERLY for the first time.** Every earlier old-vs-current comparison used
the invalid reconstruction and/or a floored budget. Re-running the identical, valid instrument on the 2026-05-22
checkout — same entry point (`_phase1_train_if_needed` exists there, verified), same override, same scoring:
- old ≈ 87.5% ⇒ **genuine regression**, and now cheaply bisectable (~13 min/step, so an 8-step bisect is ~2 h).
- old ≈ 6% ⇒ **no regression**; the recorded figure depended on an ingredient neither checkout supplies today — most
  plausibly the **cached `.simstate.h5` Phase-1 states, which are DELETED** (the `unified_per_regime/phase1_*` cache
  dirs all exist but are empty), in which case the 87.5% is **unreproducible in principle** and should be retired as a
  citable baseline rather than chased further.

## ✅ THREAD RESOLVED: no regression — the recorded 87.5% is UNREPRODUCIBLE and is hereby RETIRED as a baseline

Valid instrument (runner's OWN `_phase1_train_if_needed`, surgical `n_train_events` 200→800, nothing re-implemented),
run on both checkouts at the un-floored budget:
| checkout | result |
|---|---|
| current code, runner's own path @800ev | **1/16 = 6.2%** |
| 2026-05-22 code, **identical** instrument @800ev | **1/16 = 6.2%** |
| recorded (2026-05-21 findings) | **87.5%** |

**⇒ NOT A REGRESSION — old and current agree exactly.** (Also refuted along the way: the old checkout took ~29 min vs
current ~13 for the same nominal budget, so the two versions genuinely do different amounts of work per event — but
that extra work produced **no accuracy difference**, so it is not the explanation either.)

**⇒ The recorded 87.5% depended on something neither checkout supplies today, and the leading candidate is
unrecoverable: the cached `.simstate.h5` Phase-1 substrates.** The `research/findings/raw/unified_per_regime/phase1_*`
cache directories all still exist and are all **EMPTY** — the states were lost (consistent with the documented drive
loss). A headline number whose substrate lived only in a deleted cache **cannot be reproduced or audited**.

**⇒ ACTION: the 87.5% direct-binding figure is RETIRED as a citable baseline.** It should not be used to justify "this
used to work, so something broke" — that framing sent this session down a multi-hour path that produced four withdrawn
conclusions. If word→pool binding is needed, it must be **established and validated fresh**, not restored.

**⇒ CONSEQUENT REFRAMING OF A1 (important):** the plan recorded earlier — *"restore the direct-binding sanity first"* —
was predicated on binding having worked at some point. **On every configuration reproducible today it has never
worked** (0–6.2%, i.e. chance, across: A1 @200ev and @800ev, the runner's own path @200ev and @800ev, and both
checkouts). So A1's concept-pool route is not a repair job; **it is unbuilt**. That is a materially different and much
larger piece of work than "restore", and the roadmap should say so.

**What survives from this entire sub-thread — the complete honest list:**
1. A1 fails its binding sanity with everything applied correctly (the one solid fact throughout).
2. There is no regression; nothing broke.
3. The 87.5% baseline is unreproducible and retired.
4. Word→pool binding is **unbuilt**, not broken — a scoping correction with real planning consequences.
5. Five verification rules, each earned by a specific failure here, now encoded in `verify-go`.

## Cortical store (`concept_to_comp_attr`) — controlled NO, with a specific mechanism

New probe `_consol_cortical_store_probe.py`: measures the CORTEX-resident store (the half that survives a hippo lesion,
unlike the validated `ca1→slot` write) at the calibrated operating point, cueing pools DIRECTLY by teacher current so it
does not depend on the unbuilt word→pool binding. **Explicitly NOT the full A1 gate**, and the probe prints that on
every run.

**First run was my own config error, caught instantly by the probe's raw magnitudes:** `dw=0.0` and **per-slot mass
`[0,0,0]`** — not a failed write but an absent pathway. `BASE` sets `comp_no_pool_slot=True`, which DROPS the pool→slot
pathways. That flag is right for the `ca1→slot` measurement (the all-pools→all-slots broadcast is a write-selectivity
killer there) but wrong here, since those pathways ARE the cortical store. **The two measurements need opposite settings
of one flag** — reasoning now written into the probe. *(Two minutes to find, versus hours for the earlier config
problems — the difference is entirely that this probe reports raw magnitudes by default.)*

**Corrected run (seed 42), substrate verified physiological (`v_apical` −5.28…−3.31):**
| measure | result |
|---|---|
| store `concept→slot` own/other | `[0.945, 0.978, 1.091]` — **FLAT** |
| permuted-target control | `[0.95, 1.095, 0.931]` — flat, consistent with no signal |
| per-slot mass | `[1.249, 1.280, 1.390]` — balanced (no winner-slot artifact) |
| hippo-lesioned recall | **"2/3" but NOISE** — rates `[[0.008,0,0], [0.192,0.117,0], [0.058,0.067,0.108]]` |

**The "2/3" must not be reported as a partial success.** Fact 0 "wins" on a rate of **0.008** purely because the other
two are exactly 0; fact 1 is **wrong** (peaks on slot 0, target slot 1); only fact 2 is a real win (0.108). Store weights
(~1.25–1.39) barely moved from their `weight_mean=1.5` initialisation ⇒ **co-activation replay is not writing a
selective cortical store at all.**

**Mechanism (specific, and consistent with what already worked):** `coactivation_replay` drives the target slot
**somatically**, but the BTSP write requires an **apical plateau** as its instructive signal. Somatic drive does not
supply one, so there is no teaching signal on the cortical synapses — exactly the problem the **decoupled apical
teaching clamp** solved for `ca1→slot` (which is why that one reached a 6-seed GO). **▶ NEXT: apply the same validated
decoupled-plateau teaching signal to the slot during co-activation replay, so `pool→slot` receives a real BTSP write**,
then re-run this probe with the mass triad at 6 seeds. This reuses a mechanism already validated on this substrate
rather than introducing a new one.

## Teaching clamp on the cortical store: the write HAPPENS but is perfectly UNIFORM — a STRUCTURAL (connectivity) limit

Applied the exact mechanism that earned the `ca1→slot` 6-seed GO (drive fact i's pools for presynaptic eligibility;
clamp slot i's apical HIGH and all other slots LOW; 200-step recovery gap) to the cortical `pool→slot` store:
| measure | without clamp | **with teaching clamp** |
|---|---|---|
| per-slot mass (init 1.5) | `[1.249, 1.280, 1.390]` | **`[1.7772, 1.7735, 1.7756]`** |
| store own/other | `[0.945, 0.978, 1.091]` | **`[1.000, 0.997, 1.000]`** |
| permuted control | `[0.95, 1.10, 0.93]` | `[1.002, 1.002, 1.004]` |
| hippo-lesioned recall | "2/3" (noise) | **0/3**, all slots firing ~0.5 |

**The clamp works — a write clearly occurred (mass 1.5 → 1.78) — but it potentiated EVERY slot equally**, to three
decimal places. An exclusive instructive signal would have written slot i preferentially; instead all three slots
received identical mass.

**Leading explanation, and the codebase already documented it.** The `pool→slot` wiring is an **ALL-pools→ALL-slots
broadcast**, and the runner's own comment at that pathway says exactly why that is fatal:
> *"this is an ALL-pools->ALL-slots BROADCAST -> ...drives every slot non-selectively (**the write-selectivity killer**)"*
Driving ANY pool therefore delivers coincidence drive to EVERY slot, so the per-step recomputation of `v_apical` from
`I_coincidence` overrides the exclusive clamp within the step. This is precisely why the identical clamp DOES work for
`ca1→slot` (a route without that broadcast) and fails here. It also explains why `comp_no_pool_slot=True` exists in
`BASE` at all — the broadcast had to be removed to make the `ca1→slot` measurement meaningful.

**⇒ This looks STRUCTURAL, not an operating point.** Unlike every earlier "boundary" in this arc (all of which were
configuration artifacts), no write rule can isolate a target when the connectivity delivers the drive to every target.
**HONEST STATUS — mechanism is CONSISTENT WITH the evidence, not yet PROVEN.** The uniform mass is strong evidence of a
non-exclusive instructive signal, but the causal claim needs one direct measurement: **log per-slot `v_apical` during
the clamped write and confirm all slots sit above `v_hold`** (the twosided probe already does exactly this for the
`ca1→slot` case and can be reused). That check must be run before this is cited as a structural limit — this session
has repeatedly shown that a plausible mechanism inferred from a downstream number is how the earlier false boundaries
were born.

**If confirmed, the fix is architectural, not a knob:** the cortical store needs sparse//competitive `pool→slot`
connectivity (or lateral inhibition between slots) so a fact's pools can drive one slot rather than all — which is a
design question about how a cortical "slot" should be addressed, and deserves the research gate rather than a tuning pass.

## Cortical store, resolved to a SUGGESTIVE-but-unproven signal — and TWO of my own mechanisms refuted by measurement

Chased the flat cortical store through three explanations. **The first two were refuted by direct measurement, which is
the entire point of running them:**

**1. "ALL-pools→ALL-slots broadcast defeats the clamp" — REFUTED.** The codebase's own comment endorses this idea
(*"the write-selectivity killer"*), it explained the uniform write, and it explained why the same clamp works for
`ca1→slot`. Measuring `v_apical` *inside* the step loop (after the engine recomputes it from `I_coincidence`) killed it:
| window | slot 0 | slot 1 | slot 2 |
|---|---|---|---|
| fact 0 | **−9.33** | −66.07 | −66.13 |
| fact 1 | −66.20 | **−9.64** | −66.21 |
| fact 2 | −66.23 | −66.19 | **−9.81** |
`v_hold = −50` ⇒ **the instructive signal is perfectly exclusive.** Had I cited "structural connectivity limit" without
this check it would have been a fourth false boundary — and the worst kind, since it invites redesigning the
architecture rather than checking a parameter.

**2. "Eligibility bleeds across facts" — REFUTED.** `btsp_elig_tau_ms` defaults to 1000 ms against a 15 ms fact window,
the same cross-fact bleed already fixed for `ca1→slot`. Setting `elig_tau=30` still gave a flat raw read
(`[1.003, 0.99, 1.0]`) — though it did flip per-slot mass from **1.78** (τ=1000, above the 1.5 init) to **1.19** (below
it), showing the raw metric was tracking net decay-vs-potentiation across the whole population rather than the write.

**3. The METRIC was diluting the signal — SUPPORTED.** `W[i,j]` averaged ALL pool→slot synapses at density 0.15, so a
selective change on the few coincident synapses vanished among untouched ones. Adding the **firing-weighted read** (the
same correction that made `ca1→slot` visible):
```
(A)  raw mean         own/other=[1.003, 0.990, 1.000]   own_is_max=[F, F, T]
(A2) firing-weighted  own/other=[1.031, 1.057, 1.037]   own_is_max=[T, T, T]   <- correct slot wins for EVERY fact
```

**HONEST STATUS: SUGGESTIVE, NOT ESTABLISHED.** own-is-max 3/3 is the right *direction*, but the margins are 3–6% —
precisely the size a mass artifact produces — and **the permuted-target control was only computed for the RAW read, not
the firing-weighted one.** Per this session's own rule, a ratio without its permuted control is not evidence. The write
is also plainly swamped by the pathway's **1.5 initialisation**, against which a small learned component cannot show.

**▶ NEXT (concrete): (a) add the permuted-target control to the firing-weighted read — mandatory before any claim;
(b) lower the `pool→slot` initial `weight_mean` (currently 1.5) and/or raise the write so the learned component
dominates its own baseline — the direct analogue of the unsaturated-regime fix that unlocked `ca1→slot`; (c) then the
mass triad at 6 seeds.** The mechanism is not in doubt (exclusive instructive signal verified, correct direction 3/3);
what is unproven is whether the magnitude can be made real.

## Cortical store — FINAL STATE: a small but CONTROL-SURVIVING signal; the "swamped by init" fix REFUTED

**(a) The mandatory permuted control was run on the firing-weighted read, and it COLLAPSES:**
```
true      own/other = [1.031, 1.057, 1.037]   own_is_max = [T, T, T]
permuted  own/other = [1.002, 0.975, 0.986]   <- collapses
```
mean true **1.042** vs mean permuted **0.988**. ⇒ **the 3/3 direction is EARNED, not a mass artifact** — the effect is
real. But it is only **~5%**, nowhere near the 2.5 gate (`ca1→slot`, by contrast, reached 2.3–7.9).

**(b) "The learned component is swamped by the 1.5 initialisation" — REFUTED.** Added `comp_pool_slot_weight` (additive,
default 1.5 ⇒ byte-identical) and ran init **1.5 vs 0.2**:
```
init=1.5  per-slot mass [1.1925, 1.1894, 1.1906]   own/other [1.031, 1.057, 1.037]
init=0.2  per-slot mass [1.1925, 1.1894, 1.1906]   own/other [1.031, 1.057, 1.037]
```
**Identical.** (Recall rates differ in the 3rd decimal — 1.425 vs 1.433 — so the substrates genuinely DID differ; this
is not a knob that failed to reach the builder, which was checked: builder `:302`, probe `:56`/`:214`.) ⇒ **the
pool→slot weights converge to the same fixed point (~1.19) regardless of where they start**, so the learned component
cannot be exposed by lowering the baseline. Something pins this pathway — synaptic scaling, the global clip, or
homeostasis — and identifying it is the next question.

**Honest close of this thread — four hypotheses, three refuted by measurement, one supported:**
| # | hypothesis | verdict |
|---|---|---|
| 1 | ALL-pools→ALL-slots broadcast defeats the clamp | **REFUTED** — `v_apical` exclusive (−9 vs −66 mV, v_hold −50) |
| 2 | eligibility bleeds across facts (τ=1000 ms) | **REFUTED** — τ=30 still flat on the raw read |
| 3 | learned component swamped by the 1.5 init | **REFUTED** — weights converge to ~1.19 from any init |
| 4 | the raw mean DILUTES a real selective change | **SUPPORTED** — firing-weighted read reveals it, control collapses |

**⇒ STATUS: the cortical store IS selectively written — verified exclusive instructive signal, correct slot 3/3,
permuted control collapses — but only by ~5%, which is far too weak to drive hippo-lesioned recall (measured 1/3, at
slot rates that do not separate).** The blocker is no longer "is there a signal" but "what pins these weights to a
fixed point, and can the selective component be made to dominate it". **▶ NEXT: identify the pinning force (synaptic
scaling / global clip / homeostasis on `concept_to_comp_attr`) — measure the weight distribution's evolution during the
write rather than only its mean — then re-test. Multi-seed only after the magnitude is real; a 5% effect is not worth
6 seeds yet.**

## The cortical store's REAL blocker: a bound/saturation TENSION — and a probe-validity gap of my own

Continued the chain. **Hypothesis 4 CONFIRMED then neutralised, hypothesis 5 produced an INVALID substrate:**

**(4) `hebbian_max_weight` pins the weights — CONFIRMED, but fixing it just moves the pin.**
`hebbian_max_weight` defaults to **1.0** while pool→slot sits at ~1.2–1.5, i.e. **above the bound**, so every Hebbian
"potentiation" is negative (the trap documented in CLAUDE.md this morning — *5th instance today, 2nd on this
mechanism*). Raising it 1.0 → 8.0 moved per-slot mass **1.19 → 8.28**, confirming it was the pinning force. **But
selectivity did NOT improve** (own/other 1.02, own-is-max 2/3): the weights are now simply pinned at the NEW bound.
**Hebbian drives to whatever ceiling it has, and a saturated weight cannot carry graded selectivity** — the same
saturation that held `ca1→slot` flat until its write was moved into the unsaturated regime.

**(5) Disable Hebbian so BTSP's graded write acts alone — INVALID SUBSTRATE, arm VOID.**
```
v_apical during write: [500.17, 80.68, 81.23] / [75.65, 474.23, 75.14] / [83.08, 81.85, 501.28]
per-slot mass: ~31,000,000
```
Without Hebbian nothing bounds the pathway: weights run away to ~3×10⁷, driving `I_coincidence` and hence `v_apical` to
**500 mV** — far outside −90…+50, the same class of invalid dynamics as the original 333× miscalibration. **This arm's
numbers are void** and are not interpreted. (Note the clamp genuinely IS defeated here — all slots above `v_hold` — but
only as a consequence of the substrate having broken, not as evidence for the earlier broadcast hypothesis, which
remains refuted under valid dynamics.)

**⇒ THE REAL BLOCKER IS A TENSION, not a single defect:** the pathway needs a bound to avoid runaway, but any bound
Hebbian can reach **saturates** the weights and destroys the graded component BTSP writes. The viable window is
`bound high enough to avoid inversion` ∧ `low enough to prevent runaway` ∧ `BTSP write unsaturated within it` — a joint
(bound, lr, cycles) tuning problem, and a genuinely narrower target than anything tried so far.

**INSTRUMENT GAP FOUND IN MY OWN PROBE:** its `v_apical_physiological` check runs **after encode, before the write**, so
it reported `True` while the write drove the substrate to 500 mV. **A validity check that does not cover the phase under
study is not a check.** Fix: assert physiological range *during* the write (the per-slot `v_apical` logging added for
the broadcast test already collects exactly this — it simply is not gated on). This is the same lesson as the floored
probes: the guard must live in the instrument and cover the phase being measured.

**▶ NEXT:** joint (hebbian bound, btsp lr, cycles) sweep constrained to arms whose `v_apical` stays physiological
THROUGHOUT the write, scored on the firing-weighted read with its permuted control. Reject any arm whose substrate
leaves range — do not interpret it, as this one was nearly interpreted.

## CORTICAL STORE — CHARACTERIZED: a real but ~5% signal, INVARIANT across every operating-point lever

**The selectivity is REAL:** firing-weighted own-is-max **3/3**, and the permuted-target control **collapses every time**
(true ~1.04 vs permuted ~0.98). It is not a mass artifact. **But it is ~5%, and it does not move:**
| lever | range swept | per-slot mass | own/other |
|---|---|---|---|
| `hebbian_max_weight` | 1.0 → 2.5 → 4.0 → 8.0 | pins at the bound each time (1.19/2.67/4.22/8.28) | 1.03 / 1.05 / 1.04 / 1.02 |
| `hebbian_learning_rate` | 5e-4 → 5e-5 → 5e-6 (100×) | 2.671 / 2.677 / 2.681 | 1.05 / 1.05 / 1.05 |
| `btsp_learning_rate` | 5e-4 → 5e-8 → 5e-9 (**5 orders**) | 2.67 / 1.81 / 1.43 | 1.05 / 1.03 / **1.00 (signal gone)** |
| `pool→slot` init | 1.5 → 0.2 (7.5×) | identical | identical |
Every arm verified physiological throughout the write by the new gate.

**Mechanism, and my "unconditional clip" guess REFUTED by reading the code** (`sim/bridge.py:838`):
`delta = hebbian_learning_rate * coact * (hebbian_max_weight - w)` — a **SOFT bound driven by COACTIVITY**, not a clip.
That is the whole story: during fact i's write window the target slot fires and its pools fire, so **Hebbian potentiates
every coactive pool→slot pair broadly**, driving them all toward the bound, while **BTSP potentiates selectively** via
the exclusive apical plateau. **The broad rule sets the weight; the selective rule contributes a ~5% perturbation on
top.** Lowering Hebbian's rate doesn't help (it still converges over the ~2000 steps of write+recovery); removing it
causes runaway (mass 3×10⁷, `v_apical` 500 mV, substrate invalid).

**⇒ THE BLOCKER, precisely stated:** a **broad coactivity rule and a selective plateau rule compete on the same
synapses, and the broad one wins.** This is not an operating point that needs tuning — five levers across up to five
orders of magnitude leave the ratio unchanged. **▶ NEXT: bound the pathway with a mechanism that is NOT
coactivity-driven** (synaptic scaling, a true hard clip, or per-pathway suppression of Hebbian on `concept_to_comp_attr`
while leaving it elsewhere), so BTSP's plateau-gated write determines the PATTERN and the bounding mechanism only
controls the SCALE. That is a substrate change, and it is the first thing on this pathway that survived every cheap test.

**Hypothesis ledger for the cortical store — 6 proposed, 5 refuted by measurement, 1 supported:**
broadcast defeats clamp ✗ · eligibility bleed ✗ · swamped by init ✗ · `hebbian_max_weight` pins it ✓(but not the
magnitude lever) · Hebbian *rate* is the lever ✗ · unconditional clip ✗ · **diluting metric ✓ (the firing-weighted read
is what made the real signal visible at all)**.

## FINAL: the ~5% is a CEILING on what BTSP can express here — every bounding mechanism tried saturates the pathway

**Per-pathway gating cannot separate the two rules** — verified in code: **both** Hebbian (`bridge.py` ~7700) and BTSP
(~8031) multiply by `cp_plasticity_rate_gain`, so closing the gate on `concept_to_comp_attr` silences the selective
rule along with the broad one. That eliminated the cheapest option before spending a run on it.

**Synaptic scaling as the NON-coactivity bound — REFUTED, and it makes things worse:**
```
scaling 0.001 : per-slot mass 4.9999   own/other [0.992, 0.989, 0.995]  own_is_max 0/3
scaling 0.01  : per-slot mass 4.9919   own/other [0.991, 0.987, 0.997]  own_is_max 0/3
```
(both substrates physiological). The mass pins at **exactly 5.0 regardless of scaling rate** — a hard ceiling, so the
rate is not even the operative variable — and **the selectivity disappears completely.**

**The unifying explanation, and it is now well-supported:** the BTSP write is strong enough to drive this pathway into
**whatever ceiling exists**, and selectivity requires **headroom below the ceiling**. Hebbian's *soft* bound left a
little (weights settled at 2.67 against a 2.5 bound), which is exactly why the ~5% signal survived there; a hard ceiling
with no headroom (5.0) destroys the gradation and the signal with it. Driving the write weaker to stay below a ceiling
was already swept — 5 orders of `btsp_lr` — and the signal is ~5% throughout the usable band and gone below it.

**⇒ CHARACTERIZED LIMIT: ~5% is the ceiling of what BTSP's plateau-gated write can express on `concept_to_comp_attr`,
across every bounding mechanism (Hebbian soft bound at 4 values · synaptic scaling at 2 rates · none) and every rate
(Hebbian 100× · BTSP 5 orders) tried, with all valid arms verified physiological.** The signal is REAL (own-is-max 3/3,
permuted control collapses) but far too small to drive recall.

**⇒ NEXT IS STRUCTURAL, not a knob.** The `pool→slot` wiring is all-pools-to-all-slots at density 0.15, so *every* fact's
pools contact *every* slot and all of them compete for the same bounded weight budget. The candidate is **sparse or
competitive connectivity** (each pool contacting few slots, or lateral inhibition between slots so a winner takes the
budget) — i.e. changing WHICH synapses exist rather than how strongly they learn. That is a design question about how a
cortical "slot" should be addressable and **belongs in the research gate**, not in another sweep.

**Cortical-store hypothesis ledger: 7 proposed, 6 refuted by measurement, 1 supported.**

## ⛔⛔ RETRACTION: the cortical-store "characterized ceiling" is NOT established — I cited code that does not execute

A read-only research gate (`2026-07-26-cortical-slot-addressability-research-gate.md`) found errors in the
characterization above. **I verified all four against the source myself before accepting them:**

| my claim | reality (verified) |
|---|---|
| "Hebbian is a soft **COACTIVITY**-driven bound (`bridge.py:838`)" | **`:838` IS DEAD CODE** — it sits inside `_apply_branchless_hebbian`, *"opt-in via cfg.enable_branchless_plasticity"*, and `enable_branchless_plasticity: bool = False` (`config.py:196`). The probe never enables it. **The mechanism I reported was never running.** The rule that DOES run (`bridge.py:7710-7717`) is causal **spike-coincidence**: `pre_fired & post_fired`, `delta = lr·(w_max−w)` — gated on postsynaptic SPIKE, not coactivity. |
| "synaptic scaling pins mass at a hard 5.0 ceiling ⇒ scaling refuted" | **CONFOUNDED.** `bridge.py:8704`: `_hw_max = cfg.hebbian_max_weight if cfg.enable_hebbian_learning else **5.0**` — a hard-coded literal reached *precisely because* that arm ran `--no-hebbian`. **Synaptic scaling was never actually tested**; I measured a fallback constant. |
| the 7-hypothesis ledger | **STDP is missing from it entirely** — `enable_stdp: bool = True` (`config.py:598`) and `concept_to_comp_attr` is `plastic=True`, so STDP was writing this pathway the whole time and was never considered. |
| "~5% is a CHARACTERIZED CEILING ⇒ next is STRUCTURAL" | **NOT ESTABLISHED.** The unifying account ("BTSP saturates whatever ceiling exists; selectivity needs headroom") rests on the confounded scaling arm. **Verdict REOPENED.** |

**What still stands:** the ~5% signal itself is real (firing-weighted own-is-max 3/3, permuted control collapses every
time). What is retracted is the *explanation* and the *ceiling* claim.

**NEW FAILURE MODE, and it is the sharpest of the session: CITING A LINE IS NOT VERIFYING IT EXECUTES.** I grepped for
`hebbian_max_weight`, found `:838`, read it, and reported it as *the* mechanism — with a file:line citation that made it
look verified. I never checked whether that branch was reachable. A `grep` hit proves code EXISTS; it proves nothing
about whether it RUNS. **Before citing any code as the mechanism: find its guarding flag and check that flag's DEFAULT
and the actual config in use.** (Related and also missed: an ENABLED-BY-DEFAULT rule — STDP — was silently writing the
pathway under study and never entered the analysis. Enumerate every rule with `plastic=True` on the pathway, do not
assume the one you are tuning is the only one acting.)

**▶ NEXT — Rank 0, FREE, one seed, zero edits:** report per-slot **somatic firing** during each write window and during
recall. `_consol_cortical_store_probe.py:115/137` **already accumulates this into a full-network array and discards it**.
It separates the two live diagnoses at no cost, and answers a question never asked in this entire arc: *do non-target
slots even spike during a write window?* Only after that do the ranked mechanisms (per-slot FS cross-inhibition ·
Miller-MacKay **subtractive** normalisation, which is config-only via `btsp_mean_subtract` · HTM winner-inactive
depression) come into play.

## RANK-0 + the clean BTSP-alone test: selection is EXCELLENT, and the bound is what CREATES it

**RANK-0 (free — the data was already collected and discarded): somatic selection WORKS, ~5:1.**
| write window | per-slot spikes | target : best other |
|---|---|---|
| fact 0 | `[1313, 235, 280]` | **4.7×** |
| fact 1 | `[459, 2232, 397]` | **4.9×** |
| fact 2 | `[252, 216, 1480]` | **5.9×** |
The question no hypothesis in this arc had asked. **The correct slot is strongly selected in SPIKES**, the apical
plateau is exclusive — and yet the weights come out **1.05:1**. A ~5:1 signal arriving as 1.05:1 is compression.

**The `btsp_lr` sweep was CONFOUNDED and could not have shown this.** STDP (`enable_stdp` defaults **True**, pathway
`plastic=True`) and Hebbian were BOTH writing throughout, so lowering BTSP's rate never removed the saturating drive.
The "5 orders of magnitude" sweep never tested a graded write. (Third distinct confound on this pathway.)

**Clean test — STDP off, Hebbian rate 1e-7, bound raised to 2000 so it neither writes nor inverts:**
- **`btsp_lr=5e-4` ⇒ INVALID SUBSTRATE** (`v_apical` −20521 mV). **The write-phase validity gate CAUGHT it** and printed
  *"this arm's metrics are VOID, do not interpret"*. Its numbers looked entirely plausible (own/other 1.0, mass 1992) —
  **without the gate added two hours ago I would have interpreted them.** The instrument paid for itself.
- **`btsp_lr=5e-6` ⇒ valid, own/other `[1.059, 1.055, 1.063]`, own-is-max 3/3, permuted collapses (~0.98)** — still ~6%.
  **But the somatic selection COLLAPSED:** slot spikes `[9734, 9371, 9500]` / `[10496, 10313, 10078]` — essentially
  EQUAL, versus 5:1 in the bounded configuration.

**⇒ THE REAL STRUCTURE OF THE PROBLEM, and it inverts the earlier reading.** Raising the bound to 2000 let the weights
grow (per-slot mass 1.5 → **77**), which drove every slot to fire massively (~10,000 spikes each) and **destroyed the
selection**. So:
- **bound PRESENT (2.5):** slots fire **5:1 selectively** — but the weights saturate AT the bound ⇒ 1.05:1 weights.
- **bound ABSENT/high (2000):** weights grow, **all slots fire equally** ⇒ selection lost entirely.
**The bound is not merely a constraint — it is what CREATES the somatic selection** (it keeps non-target slots below
firing threshold). The earlier bounded configuration was the *good* regime for selection; its only defect is that the
weights saturate at the bound.

**▶ NEXT (the configuration never yet tried): keep the bound at 2.5 — preserving the 5:1 selection — while making the
write GRADED beneath it:** `hebbian_max_weight=2.5` (selection) + `enable_stdp=False` + `hebbian_learning_rate≈1e-7`
(present but not writing) + a low `btsp_lr`, so BTSP is the only writer and lands below 2.5 rather than at it. Verify
per-slot spikes still show ~5:1 (selection intact) AND weights stay below the bound (graded), then read own/other.

## Matched ceilings DO produce a graded write — but selectivity tracks SPIKE selectivity, and the write is compressive

**The mismatched-ceiling diagnosis was CORRECT.** BTSP's soft bound is `(btsp_w_max − w)`; at `w_max=2000` against an
effective ceiling of ~2.5 (the Hebbian clip, `bridge.py:8704-8710`) that term stays ≈2000, so BTSP drives at full
strength into the clip and every synapse truncates at the same value **regardless of rate** — which is why 5 orders of
`btsp_lr` never produced gradation. Setting `btsp_w_max = 2.5` to match:
```
btsp_lr 5e-4 : per-slot mass 2.284   (BELOW the 2.5 bound — GRADED, first time this session)
btsp_lr 5e-5 : per-slot mass 1.592   (well below — graded)
```
**⇒ gradation achieved. But own/other FELL to ~1.015** (from ~1.05), because the lower weights also weakened the
somatic selection (**1.35:1**, vs 5:1 in the bounded configuration).

**The governing relationship, across every valid arm: weight selectivity tracks SPIKE selectivity — and does so with a
large compression factor.**
| somatic selection (spikes) | resulting own/other |
|---|---|
| 5:1 | 1.05 |
| 1.35:1 | 1.015 |
Roughly `own/other − 1 ≈ (spike_ratio − 1) / 80`. **Even a 5:1 firing difference yields only ~5% weight difference.**
That compression — not the ceiling, not the rate, not the bound — is the actual blocker.

**Leading explanation for the compression, and it is FREE to test.** BTSP's instructive signal is exclusive *within a
window* (verified: target −9 mV vs others −66 mV). So `w[pool_i → slot_j≠i]` cannot grow during fact i's own window. It
must therefore grow during **other facts' windows** — i.e. **pool_i cells are not silent while fact j is being written**,
so they carry eligibility exactly when slot_j has its plateau. The per-slot spike readout already shows non-target slots
firing 235–500 spikes in other windows; the pools are almost certainly doing the same. **▶ NEXT (free, same pattern as
Rank-0): report per-POOL firing during each write window from the `pool_fire` array already collected.** If pool_i fires
substantially during fact j's window, the leak is identified and the fix is about isolating pools between windows (or
gating eligibility on the driven pool), not about any bound or rate.

**Also learned: STDP was CONTRIBUTING to the somatic selection, not merely adding noise** — disabling it dropped
selection from 5:1 to ~1.1:1. It should not be silenced casually in future arms.

## LEAK REFUTED — pools are 99%+ isolated. Everything upstream is selective; the compression must happen OFF-window.

Free readout (same pattern as Rank-0, from the array already collected):
| write window | pool spikes `[fact0, fact1, fact2]` | cross-window leak |
|---|---|---|
| 0 | `[43668, 297, 353]` | **0.8%** |
| 1 | `[263, 43840, 334]` | **0.8%** |
| 2 | `[221, 202, 44438]` | **0.5%** |
**⇒ the "pool_i fires during fact j's window" leak hypothesis is REFUTED.** Pools are essentially isolated.

**The full upstream picture is now measured, and it is EXCELLENT everywhere:**
- apical plateau **exclusive** (target −9 mV vs non-targets −66 mV, `v_hold` −50) — verified
- slot somatic selection **~5:1** — verified
- pool isolation **>99%** — verified
- resulting weights: **1.05:1**

With that much upstream selectivity, `w[pool_i → slot_{j≠i}]` should barely grow — yet all pool→slot weights end up
near-identical (mass 2.67 from a 1.5 init, i.e. an average gain of ~1.17 spread almost uniformly). **A selective drive
producing a uniform weight change means the weight change is not coming from the driven windows.**

**NEW LEADING CANDIDATE (fits every measurement, not yet tested): the UNDRIVEN RECOVERY GAPS dominate the write.** The
protocol is 10 cycles × 3 facts × (30 driven steps + **200-step recovery gap**) ⇒ **~900 driven steps vs ~6000 undriven
steps**. Plasticity is live throughout, and during the gaps nothing is driven — no exclusive plateau, no selective
firing — so spontaneous/OU-driven activity potentiates broadly and **uniformly**, and it has **6.7× more steps in which
to do so** than the selective write has. That would swamp a perfectly selective write with non-selective potentiation
and produce exactly the observed pattern.

**▶ NEXT (cheap, decisive, two arms): freeze plasticity during the recovery gaps** (or shorten them) and re-measure.
The recovery gap was introduced for a real reason — it was load-bearing for the `ca1→slot` result — so the fix is to
keep the gap but suppress LEARNING during it (`set_plasticity_gate(...,0)` around the gap loop), not to remove it.
Verify per-slot spikes still show ~5:1 and weights stay graded, then read own/other with the permuted control.

**Session note:** this is the 8th hypothesis on this pathway; 7 have been refuted by direct measurement and the surviving
one (the diluting raw-mean metric) is a measurement artifact rather than a mechanism. Every refutation came from a
measurement chosen in advance, and three of them — the exclusive plateau, the 5:1 selection, and this pool isolation —
came free from data the probe was already collecting and discarding.

## Gap-freezing: hypothesis PARTIALLY confirmed, but the result is a winner-slot artifact — NOT a win

Froze plasticity during the undriven recovery gaps (keeping the gap itself, which is load-bearing for `ca1→slot`), so
only the selective driven windows write. Baseline run side-by-side, single variable:
| | per-slot mass | own/other | own_is_max | permuted |
|---|---|---|---|---|
| baseline | `[1.193, 1.189, 1.191]` (balanced) | `[1.031, 1.057, 1.037]` | **3/3** | `[1.00, 0.98, 0.99]` |
| **gap-frozen** | **`[0.999, 1.000, 3.093]` (3× IMBALANCE)** | `[0.917, 0.954, **7.248**]` | **1/3** | `[0.88, 1.22, 0.24]` |

**The 7.248 is NOT a result.** Slot 2 carries 3× the mass of slots 0 and 1, which is the winner-slot signature the probe
prints a warning for on that exact line — a fact reading the heavy slot scores high regardless of whether its write was
selective. And `own_is_max` **fell to 1/3** (baseline 3/3), which is the honest headline: gap-freezing made per-fact
selectivity *worse*, not better.

**What IS confirmed:** freezing the gaps changed the weights substantially (balanced ~1.19 → `[1.0, 1.0, 3.09]`), so
**the undriven gaps were indeed supplying most of the potentiation** — the arithmetic (≈6000 undriven vs ≈900 driven
steps) was right about *where the weight comes from*. But removing that contribution does not reveal a clean selective
write underneath; it reveals a driven-window write that is **strongly asymmetric across facts** (one slot takes almost
everything while the other two decay below their 1.5 init).

**⇒ HONEST STATE AT SESSION END.** Every upstream stage is verified excellent — exclusive plateau (−9 vs −66 mV), ~5:1
slot selection, >99% pool isolation — and the weights still do not carry it. Two mutually-constraining facts now bound
the problem: **(a)** with gap plasticity ON, the weight is dominated by non-selective potentiation and comes out uniform
(~5% selectivity); **(b)** with it OFF, the driven-window write alone is asymmetric and collapses to one slot. The
question is no longer "is the drive selective" (it is, measured at three stages) but **"why does a selective drive
produce either a uniform or a one-slot weight outcome, and never a per-fact-matched one"**.

**▶ NEXT (do these before any new mechanism):** (1) **multi-seed the gap-frozen arm** — a single seed cannot distinguish
"one slot always wins" from "a different slot wins each seed" (schedule/ordering asymmetry), and that distinction picks
the next move; (2) check whether the winning slot tracks **write order** (the schedule is shuffled per cycle) — if it is
the last-written fact, the asymmetry is a recency/ordering effect, not a substrate property; (3) only then consider the
research gate's ranked mechanisms (per-slot FS cross-inhibition · Miller-MacKay subtractive normalisation via
`btsp_mean_subtract`, which is config-only · HTM winner-inactive depression).

**Cortical-store ledger: 9 hypotheses, 8 refuted or narrowed by direct measurement.** Three decisive facts came FREE
from data the probe was already collecting and discarding (exclusive plateau · 5:1 selection · pool isolation).

## ✅ DIAGNOSIS COMPLETE: the write is WINNER-TAKE-ALL with a GLOBAL winner — the shared inhibitory pool is the cause

Multi-seeded the gap-frozen arm (the diagnostic flagged as necessary before any new mechanism):
| seed | per-slot mass | winner |
|---|---|---|
| 42 | `[0.999, 1.000, **3.093**]` | slot 2 |
| 43 | `[1.000, **3.310**, 0.999]` | slot **1** |
| 44 | `[1.000, 0.999, **3.184**]` | slot 2 |
**Exactly ONE slot takes ~3.1–3.3 while the other two sit at ~1.0, and WHICH slot varies with the seed.** That is
symmetry-breaking / winner-take-all, not a fixed property of any particular slot — and it rules out the ordering
explanation as the whole story (the schedule is reshuffled per cycle, yet the winner is stable *within* a seed).

**⇒ THE MECHANISM, and it converges with the research gate's independent Rank-1.** `comp_attr_inh`
(`nmda_compositional_consolidation.py:278-289`) is a **single SHARED inhibitory pool** that every slot drives and that
inhibits every slot — **global symmetric inhibition**. Global inhibition resolves competition **once, globally**, so one
slot wins outright and suppresses the rest for the whole run. What the task needs is competition resolved
**per write window**, so the winner can differ per fact. This is exactly the structure the read-only research gate
independently identified as the residual, and this project has three prior results showing shared/global inhibition is
selection-inert (EMERGE-41 FS-on vs FS-lesion winner overlap 1.00; riii CA3 sparsity changed but ratio unchanged 1.16×).

**This closes the diagnostic chain.** Every stage is now measured, and each is excellent until the last:
| stage | measured | verdict |
|---|---|---|
| apical plateau (instructive signal) | −9 mV target vs −66 mV others | **exclusive ✓** |
| pool isolation across windows | >99% (0.5–0.8% leak) | **isolated ✓** |
| slot somatic selection | ~5:1 | **selective ✓** |
| where the weight comes from | ~6000 undriven vs ~900 driven steps | gaps dominate (confirmed) |
| **the write itself** | **one global WTA winner, seed-dependent** | **✗ THE BLOCKER** |

**▶ NEXT BUILD (now well-founded, not a guess): replace the shared `comp_attr_inh` with per-slot FS pools + CROSS-
inhibition** (`comp_attr_FS_s → comp_attr_{t≠s}`, no self-inhibition) at the multi-seed-GO operating point already in
this repo (`biased_competition_buffer.py:114-115,164-176` — gentle `20/5`, `sel_recurrent_weight=0.35`, α<1). Lesion arm
= the shipped global pool, so the claim under test is **topology**, not weight. Gate on: per-slot mass BALANCED (the
artifact check that exposed this), own-is-max ≥2/3, permuted control collapsing, substrate physiological throughout the
write, at 6 seeds.

**Final cortical-store ledger: 10 hypotheses, 9 refuted or narrowed by direct measurement, 1 standing (this one) — and it
is corroborated independently by a read-only research gate that reached it from the code rather than from the data.**

## ⛔ "DIAGNOSIS COMPLETE" RETRACTED — the prescribed fix refuted the diagnosis that prescribed it

Built the per-slot FS + cross-inhibition topology (additive, default-off; shipped global pool = lesion arm; operating
point borrowed from the in-repo multi-seed-GO `biased_competition_buffer.py`, density 1.0 / drive 20 / inhibit 5, no
self-inhibition). **It changes nothing:**
| condition | global pool (lesion) | per-slot FS cross-inhibition |
|---|---|---|
| gaps live | mass `[1.193, 1.189, 1.191]`, own/other mean **1.042** | mass `[1.151, 1.150, 1.152]`, mean **1.048** |
| gaps frozen, bound 1.0 | mass `[0.999, 1.000, **3.093**]` | mass `[0.999, 1.000, **2.951**]` |
| gaps frozen, bound 2.5 | mass `[2.499, 2.499, **4.928**]` | mass `[2.499, 2.499, **4.703**]` |
**⇒ the global shared inhibitory pool is NOT the cause of the winner-take-all.** My "DIAGNOSIS COMPLETE" verdict — which
the read-only research gate independently corroborated from the code — **is RETRACTED.**

**And the follow-up explanation is refuted too.** I then noticed the gap-frozen runs had used `hebbian_max_weight`'s
**default 1.0** against a 1.5 init (the inversion trap, 7th instance today) — slots pinned at exactly `0.9994` = the
bound. But re-running at bound **2.5** (init 1.5 BELOW it, no inversion) **preserves the single winner**: two slots pin
at the bound, one escapes above it. So the inversion is not the cause either.

**HONEST STATE: the single-winner phenomenon is ROBUST and its cause is UNKNOWN.** It survives both inhibitory
topologies and both bound regimes. What is established: two slots saturate at whatever the Hebbian bound is, while ONE
slot receives enough BTSP write to escape above it — and it is not simply the most-active slot (fact 1's window drives
slot 1 to 2232 spikes vs fact 2's 1480, yet slot 1 pins at the bound and slot 2 escapes).

**Cortical-store ledger: 12 hypotheses, 11 refuted by direct measurement.** The two most confident ones — "global
inhibition causes it" (corroborated independently) and "the bound inversion causes it" — were both wrong, and both were
killed by tests I chose to run rather than by later discovery.

**▶ NEXT (for a fresh session, not 03:20): find why ONE slot escapes the bound while others pin at it.** Instrument the
per-slot BTSP `dw` per window directly (not the post-hoc weights) — the probe already has the write loop; log
`Σ dw[pool→slot_j]` per window and compare against per-slot spikes. That distinguishes "one slot gets more write" from
"one slot resists the pull-down", which is the fork the current data cannot separate. **Do NOT prescribe another
mechanism before that measurement** — this thread's last two mechanisms were both refuted by the builds meant to
implement them.

## 🔬 THE MEASUREMENT THAT SEPARATES THE FORK — the write is NEAR-SYMMETRIC; the store is a ~3% RESIDUAL

Instrumented the per-window `dw` directly (snapshot the pool_i→slot_j block means after every write window;
masks precomputed once, `_consol_cortical_store_probe.py`). Seed 42, `--teaching-clamp --elig-tau 30 --freeze-gap
--hebbian-max-w 2.5` — i.e. the exact condition where the single winner appears.

| slot | potentiation | depression | **net** |
|---|---|---|---|
| 0 | +65.98 | −68.01 | −2.04 |
| 1 | +69.62 | −71.59 | −1.98 |
| 2 | +71.35 | −66.07 | **+5.28** |

**Neither branch of the fork I posed is right.** The winner does not "receive more write" (pos spread 65.98→71.35,
8%) and does not "resist the pull-down" (neg spread −66.07→−71.59, 8%). **Both flows are ~70 units and nearly cancel;
the entire store is the ~3% residual.** That is a signal-to-noise result, not a mechanism result — and it explains
why the winner is SEED-DEPENDENT (slot 2 on seeds 42/44, slot 1 on 43): a residual that small is set by noise.

**The BTSP write itself is FINE and correctly signed.** Decomposing the DRIVEN pool's own contribution:
`diag = +196.84` (pool_i → its own slot_i) vs `off = −6.01` (pool_i → other slots). Strongly selective, right sign.
**It is swamped in the per-slot totals by depression arriving from the NON-DRIVEN pools.** So the defect is not in
the instructive signal, the plateau exclusivity, the pool isolation, or the slot selection (all previously measured
good) — it is that ~70 units of non-selective depression ride on top of a ~197-unit selective potentiation.

**Instrument verified, not assumed:** `gap`-phase dw is **exactly 0.0** on every slot, confirming `--freeze-gap`
genuinely freezes the pathway (rule 2 — the "this is inert" claim is now an assertion in the data, not a comment).

**⇒ NEXT (a measurement, not a prescribed mechanism): WHICH rule supplies the ~70 units of depression?** STDP was
ON throughout this arc and Hebbian is live; ablation `{both, −STDP, −Hebbian, −both}` launched. Only once the
depression source is named does a corrective mechanism (e.g. the still-untried Miller-MacKay `btsp_mean_subtract`)
have a defined target.

## ✅ THE DEPRESSION SOURCE IS NAMED — Hebbian's soft bound; and STDP is INERT on this pathway

Ablation `{both-on, −STDP, −Hebbian}` at seed 42, `--teaching-clamp --elig-tau 30 --freeze-gap --hebbian-max-w 2.5`:

| arm | diag (selective) | off | per-slot mass | verdict |
|---|---|---|---|---|
| both-on (baseline) | +196.838 | −6.009 | `[2.4985, 2.4990, 4.9278]` | ✓ physiological |
| **−STDP** | **+196.840** | **−6.009** | **`[2.4985, 2.4990, 4.9279]`** | ✓ **BYTE-IDENTICAL to baseline** |
| −Hebbian | +2813.1 | +483.0 | `[29.2, 22.6, 60.8]` | ⛔ **VOID** — `v_apical` −284 mV, runaway |

**(1) STDP is INERT on `concept_to_comp_attr`.** Disabling it changes the result in the 6th significant figure. STDP
was live throughout this entire arc and was repeatedly flagged as a confound (it is what invalidated an earlier
`btsp_lr` sweep); on THIS pathway it contributes nothing. That removes a suspected variable permanently.

**(2) Hebbian cannot be removed** — the substrate goes unphysiological (the validity gate I built caught it and
printed VOID rather than letting me read the numbers, which is the gate working as intended).

**⇒ THE MECHANISM, now measured rather than inferred:** at `hebbian_max_w=2.5` the weights sit at **2.4985 / 2.4990 —
exactly at the bound**. Hebbian's soft bound is `dw ∝ (w_max − w)`, so it drives every coactive pool→slot pair to the
ceiling and then oscillates about it; **the ±70-unit flows ARE that oscillation**, and BTSP's selective +196.8 is
absorbed because Hebbian pulls it back to the bound within the same window. Only the ~3% residual escapes — hence a
seed-dependent winner.

This CONFIRMS, with a direct per-window measurement, the mechanism the board had recorded from code-reading alone
("a broad coactivity rule and a selective plateau rule compete on the same synapses and the broad one wins"). The
difference is that it is now a measurement with an ablation behind it, not an inference — and it adds the new fact
that STDP plays no part.

**▶ NEXT (in flight): Hebbian-lr scan DOWNWARD** (default / 1e-3 / 1e-4 / 1e-5, STDP off since proven inert). If
Hebbian's role is to BOUND rather than to WRITE, there should be a rate low enough to keep the substrate physiological
while letting BTSP set the pattern. If no such rate exists — if selectivity and stability cannot coexist on this
pathway — then the bound must come from a NON-coactivity mechanism (synaptic scaling · a true hard clip · the
still-untried Miller-MacKay `btsp_mean_subtract`, **verified live + reachable**: `config.py:396`, `bridge.py:8153`
inside `_run_one_simulation_step`, guarded by an `elif` whose preceding branch is the default-off Milstein path).

## 🔑 WHY EVERY LEVER WAS INERT — the store is a FIXED POINT, so RATE levers cannot move it

`--hebbian-lr 0.001` (1000× down) vs default, same condition: `diag` **196.88 vs 196.84**, net
**`[−2.088, −2.031, +5.297]` vs `[−2.038, −1.977, +5.278]`** — identical to 4 significant figures.

**This retroactively explains the arc's entire "invariant across every lever" history.** Hebbian's soft bound is
`dw ∝ (w_max − w)`: the weights settle at a FIXED POINT (`w → w_max`). A learning RATE changes how *fast* that fixed
point is reached, **never where it is** — so *every* rate sweep ever run on this pathway (Hebbian lr 100×, BTSP lr
across FIVE orders) was inert **by construction**, not by coincidence. Measured confirmation: the weights sit at
**2.4985 / 2.4990** against `hebbian_max_w = 2.5`.

**And the bound cannot simply be moved, because stability and selectivity are the SAME knob:**

| condition | substrate | pool leak | somatic selection |
|---|---|---|---|
| `hebbian_max_w=2.5` (stable) | ✓ physiological | **0.4–0.8%** | target dominates 3/3 |
| `hebbian_max_w=50` | ⛔ `v_apical` **+197.9 mV** | **20.9–46.2%** | FAILS on fact 0 |
| `--no-hebbian` | ⛔ `v_apical` **−284 mV** | — | runaway |
| `hebbian_max_w=50 --syn-scaling 0.001` | ⛔ still unphysiological | 21.3–45.9% | FAILS on fact 1 |

⇒ **the low Hebbian bound is load-bearing for STABILITY, and that same low bound is what pins every synapse at the
ceiling and destroys selectivity.** Synaptic scaling at 1e-3 does not decouple them. So the remaining lever must be
STRUCTURAL and must act WITHOUT moving the stabilising bound — which is exactly Miller-MacKay subtractive
normalization on the BTSP increment (`btsp_mean_subtract`, enforcing `Σ_j dw_ij = 0` per postsynaptic cell so no
common-mode pedestal can form). Verified live and reachable before use (`config.py:396`; `bridge.py:8153` inside
`_run_one_simulation_step`, its `elif` guarded by the default-off Milstein branch). Now wired into the probe
(`--mean-subtract`) and running at the STABLE operating point.

**⚠️ INSTRUMENT DEFECT FOUND (3 variants today, all the same root — an empty wrapper read as data):** parallel GPU
arms died on VRAM contention exiting **0** with empty output; then a serial loop's `timeout 900` killed runs
mid-flight and buffered `grep` output was discarded. In both cases the surviving/absent arms looked like clean nulls.
**An arm that produces NO output is a FAILED RUN to reproduce, never a null result.** Now: serial, longer timeout,
`grep --line-buffered`, explicit rc. Encoded in `.claude/skills/verify-go/SKILL.md`.

## 🎯 MILLER-MACKAY SUBTRACTIVE NORMALIZATION — the first mechanism in this thread to SURVIVE the permuted control

`--mean-subtract 1.0` at the STABLE operating point (`--teaching-clamp --elig-tau 30 --freeze-gap --no-stdp
--hebbian-max-w 2.5`), i.e. the stabilising Hebbian bound left in place and only BTSP's increment normalized:

| seed | own/other | own-is-max | **permuted control** | per-slot mass (heavy slot) | substrate |
|---|---|---|---|---|---|
| 42 | `[19.20, 27.09, 46.61]` | **3/3** | **`[0.089, 0.077, 0.044]`** | `[0.886, 0.890, 2.125]` (slot 2) | ✓ |
| 43 | `[16.66, 30.81, 11.51]` | **3/3** | **`[0.123, 0.059, 0.148]`** | `[0.931, 2.205, 0.910]` (slot **1**) | ✓ |
| 44 | `[15.47, 16.83, 32.79]` | **3/3** | **`[0.154, 0.084, 0.060]`** | `[0.921, 0.922, 2.128]` (slot 2) | ✓ |

Baseline for comparison (same config, mechanism off): own/other `[0.98, 1.03, 3.93]`, own-is-max **1/3**.

**Why this is not the artifact that killed the earlier "3.67 lead":**
1. **The permuted-target control COLLAPSES** — 0.044–0.154 in all 9 fact-seeds (the earlier lead's permuted control
   sat at 3.38–3.77 and never collapsed). This is the control that refuted the previous lead; here it passes.
2. **The heavy slot MOVES with seed (2 / 1 / 2) but ALL THREE facts pass regardless.** The earlier artifact's
   smoking gun was that the "passing" fact moved WITH the heavy slot; that signature is ABSENT here.
3. **A mass artifact cannot produce this shape.** A heavy slot j inflates `W[i,j]` for every fact i, so it raises
   own/other for fact j and DEPRESSES it for i≠j. Here the two LIGHT slots read 11.5–30.8 — far above 1.
4. The off-diagonal write collapses to ~0 (`diag 112.19` vs `off 0.78`), which is the mechanism's stated action
   (`Σ_j dw_ij = 0` per postsynaptic cell ⇒ no common-mode pedestal).

**⚠️ THIS IS NOT A GO, AND THE CAPABILITY IS NOT CLOSED.** Seeds 43 and 44 return **`VERDICT: NO`** — the runner's
verdict combines the weight read (A) with the **hippo-lesioned RECALL (B)**, and (B) does not follow on 2/3 seeds.
So: **the STORE is now strongly and genuinely selective (the half that was blocked all session); whether RECALL
follows is UNRESOLVED.** A selective weight matrix that does not produce selective recall is exactly the
proxy-vs-capability gap this arc has been burned by before — the weight read is a PROXY, `(B)` is the capability.

**▶ IN FLIGHT: the full 6-seed gate** (`_consol_meansub_gate.sh`, 42/43/44/100/101/102) with the **mechanism-OFF
LESION arm at the identical operating point**, so the claim tested is the MECHANISM and not the op-point, capturing
`(B)` explicitly (my earlier greps silently dropped it — the same class of instrument defect logged above, caught
here by the runner's own VERDICT disagreeing with my reading of (A)).

## ✅✅ 6-SEED GO ON THE CORTICAL STORE WRITE — and a decisive NEGATIVE that RE-LOCATES the blocker

Full gate, `_consol_meansub_gate.sh`, 6 seeds × {mechanism, mechanism-OFF lesion} at the IDENTICAL operating point
(`--teaching-clamp --elig-tau 30 --freeze-gap --no-stdp --hebbian-max-w 2.5`), so the claim tested is the MECHANISM:

| arm | own-is-max 3/3 | min own/other | max permuted (all 18 fact-seeds) | recall (B) |
|---|---|---|---|---|
| **`--mean-subtract 1.0`** | **6/6 seeds** | **11.51 – 22.95** | **0.154** | **7/18** |
| **LESION (off)** | **0/6 seeds** | 0.93 – 1.01 | 1.156 | **8/18** |

**(A) THE STORE WRITE IS A 6-SEED GO.** No overlap between arms on any metric; every fact-seed clears the 2.5 gate
by 4–9×; the permuted-target control collapses in all 18 fact-seeds (max 0.154) while the lesion's sits at ~1.0–1.16.
The heavy slot moves with seed while all three facts pass regardless — the winner-slot signature that refuted this
thread's earlier lead is ABSENT. Substrate physiological in every arm. **Miller-MacKay subtractive normalization
(`btsp_mean_subtract`) solves the write problem that blocked this thread all session.**

**(B) AND IT DOES NOT HELP RECALL AT ALL — 7/18 vs lesion 8/18, with CHANCE AT 6/18.** Both arms sit at chance.
Making the store 20–46× selective moved hippo-lesioned recall by nothing (slightly down, within noise).

**⇒ THE DECISIVE STRUCTURAL RESULT: the store and the recall are DECOUPLED. The recall read is not reading the
store.** This re-locates the blocker with evidence rather than inference — for the whole session the two were one
undifferentiated failure, and the write's selectivity was the assumed cause. It is now excluded: a demonstrably,
controlled-ly selective store yields chance recall.

**Why this was nearly mis-reported (both caught, both now encoded in `verify-go`):** (1) my grep omitted the
permuted control, so the first `own_is_max 3/3` looked like a win while per-slot mass was still unbalanced — the
exact signature that killed this thread's earlier "3.67 lead"; the control happened to pass, but that was not known
when the claim was held. (2) The recall failure was visible only because the runner's `VERDICT: NO` **contradicted my
reading of (A)** — I had been grepping past that line. Trusting my own read of the selectivity numbers would have
recorded a GO on a capability that sits at chance.

**▶ IN FLIGHT — the read side.** The `(B)` rates decline MONOTONICALLY across successive cues in **BOTH** arms
(arm seed 101: 2.05 → 1.36 → 0.56; lesion seed 43: 2.9 → 1.5 → 0.42), a global gradient far larger than the
between-slot differences the read must resolve — so `argmax` can be decided by WHEN a fact was cued rather than by
what is stored, independent of the write. The recall loop cues all facts BACK-TO-BACK with no recovery
(`_consol_cortical_store_probe.py:283-298`); the WRITE phase had the identical defect and its fix — an inter-fact
recovery gap — is on record as load-bearing. `--read-gap` (additive, default 0 = byte-identical) now scanning
{0, 200, 1000} × seeds 42/43/44.

## ⛔ READ-GAP REFUTED — adaptation was REAL, the gap FIXED it, and recall did NOT move

`--read-gap {0, 200, 1000}` × seeds 42/43/44 on the GO arm: **4/9 → 3/9 → 3/9.** No improvement.

**The lever verifiably worked** (so this is a refutation of the HYPOTHESIS, not a failed manipulation): the monotonic
decline across cues is gone — seed 42 fact 1 `1.817 → 4.400`, fact 2 `0.517 → 1.508` — and fact 0's rates are
**byte-identical across all three gaps** (`[1.542, 1.408, 1.467]`), exactly as required since the gap follows each
cue. ⇒ **adaptation was real, was relieved, and was NOT the cause of the recall failure.** (Hypothesis 13 of 13;
12 refuted by direct measurement.)

**What the corrected reads now expose is a DIFFERENT and much sharper defect: weight selectivity is not being
converted into RATE selectivity.** Within any single cue the three slots differ by only **~5–15%**, in a largely
FIXED ordering, e.g. seed 43 @ gap 200: fact 0 `[2.308, 2.608, 2.933]` · fact 1 `[3.058, 3.017, 2.792]` · fact 2
`[1.408, 1.392, 0.858]` — slot 0 leads in 2 of 3 cues regardless of which fact is being recalled. **A store that is
20–46× selective in WEIGHTS yields 5–15% differences in FIRING RATE, swamped by a fixed per-slot bias.** That is a
transduction failure at the read, and it is fully consistent with the earlier decoupling result (a maximally
selective store gave chance recall).

**▶ NEXT — the measurement that separates the two remaining candidates (do NOT prescribe a mechanism first; this
thread's last three prescribed mechanisms were all refuted, twice by the very build that implemented them):**
1. **Fixed per-slot bias** — measure each slot's firing rate under a NULL cue (no pool drive) and under a
   uniform-drive cue. If the same ordering appears with no fact cued, the read is dominated by intrinsic
   excitability and the fix is a read-side normalization, not a write-side one.
2. **Drive-fraction** — measure what fraction of each slot's total input current arrives via `concept_to_comp_attr`
   versus the WTA/attractor recurrence. If the store is a small minority of the drive, a 20–46× weight ratio
   CANNOT move the rate regardless of how selective it is, and the fix is the drive balance.
These are distinguishable in ONE instrumented run and both are free of new mechanism.

## 🎯🎯 BLOCKER PRECISELY LOCATED (lesion-grade): the STORE DOES NOT DRIVE THE SLOTS DURING RECALL

One instrumented run, seeds 42/43/44, separating the two surviving candidates. **C2 CONFIRMED, C1 REFUTED.**

**(C2) Zero the `concept_to_comp_attr` synapses outright and repeat the IDENTICAL recall — nothing changes:**

| seed | recall, store INTACT | recall, store **ZEROED** | Δ |
|---|---|---|---|
| 42 | `[1.54,1.41,1.47]` `[4.40,2.93,2.73]` `[1.51,1.57,1.73]` | `[1.46,1.35,1.49]` `[4.28,2.78,2.53]` `[1.75,1.62,1.83]` | ~5% |
| 43 | `[2.31,2.61,2.93]` `[3.06,3.02,2.79]` `[1.41,1.39,0.86]` | `[2.21,2.55,2.95]` `[3.12,3.13,2.74]` `[1.13,1.27,0.68]` | ~4% |
| 44 | `[2.03,1.77,2.04]` `[2.43,1.99,2.39]` `[1.94,1.70,2.19]` | `[1.89,1.79,2.39]` `[2.45,2.03,2.33]` `[1.93,1.86,2.18]` | ~5% |

**Deleting the ENTIRE store moves the slot firing rates by ~5%.** ⇒ the store is a negligible minority of each
slot's drive; the slots fire from the WTA/attractor recurrence and background, not from the cued pools.

**(C1) REFUTED as the explanation:** null-cue (nothing driven) per-slot rates are **0.125–0.358** against driven
rates of **1.4–4.4** — intrinsic bias is ~10% of the signal, not dominant, and its ordering does not track the cued
ordering.

**⇒ THIS CLOSES THE DECOUPLING QUESTION WITH A LESION-GRADE MEASUREMENT: a 20–46× selective store cannot move a
rate it does not drive.** Recall sitting at chance is not a write problem, not an adaptation problem, and not an
intrinsic-bias problem — **the read is not connected to the store in any functionally meaningful way.**

**⚠️ HONEST CONSEQUENCE FOR THIS SESSION'S OWN RESULT.** The `btsp_mean_subtract` 6-seed GO is real and correctly
controlled **as a statement about the WRITE** (own-is-max 3/3 on 6/6, permuted ≤0.154, lesion 0/6) — and it was
written into a pathway the recall does not read. It is a validated mechanism on a pathway whose functional role is
now in question. That is exactly why the capability gate (B), not the weight proxy (A), is the deliverable: without
(B) this would have been recorded as consolidation working.

**▶ NEXT (measurement first, still no prescribed mechanism — 13 hypotheses, 12 refuted, and the last three
prescribed mechanisms were all refuted, two by their own builds):** quantify the drive budget at a slot during
recall — what fraction of its input current arrives via `concept_to_comp_attr` vs the WTA (`comp_wta_weight=5.0`),
the self-regeneration (`comp_self_regen=0.15`) and background. `comp_pool_slot_weight` is **1.5** against a WTA
weight of **5.0**, so the store may simply be out-weighted by design. If so the question becomes an ARCHITECTURAL
one — should a cortical store drive its slot directly, or gate/bias an attractor that is driven elsewhere — and
that is a design fork worth surfacing rather than silently tuning a weight ratio.

## ⛔⛔ RETRACTION — (C2) "the store does not drive the slots" IS VOID. The lesion never held.

**The zeroing did not persist.** Direct verification (`cp_connections.data` zeroed, then 5 simulation steps):
`after in-place write -> first 1000 all zero? True` · `after 5 steps -> still zero? False, max|w| 0.05`.
**The recall read runs with plasticity LIVE**, so the "deleted" store regrew *during the very read that was supposed
to measure its absence*. Over 3 cues × (60 read + 200 gap) = 780 steps it regrew substantially.

**Caught by my own next measurement contradicting it:** the (D) drive budget runs AFTER (C2) zeroed those synapses
and reports the store at **90.85 / 95.04 / 90.85%** of all charge into slot neurons. Had the lesion held, (D) would
have read ~0%. A "lesion-grade" claim was refuted by the instrument I built to answer the *next* question.

**⇒ RETRACTED:** "deleting the entire store moves slot rates by ~5%", "the store is a negligible minority of each
slot's drive", "the read is not functionally connected to the store", and the conclusion that the decoupling was
explained. The board entry asserting these is corrected in the same cycle.

**⇒ ALSO REFUTED (the opposite way): the store is NOT out-weighted.** It carries **~91–95% of the charge into slot
neurons** against `slot_recurrent` ~3.4–6.9% and everything-else ~1.5–2.2%. `comp_pool_slot_weight=1.5` vs
`comp_wta_weight=5.0` was misleading — per-synapse weight is not drive share (64.5k store synapses vs 15.5k
recurrent). **So the architectural fork I was about to surface is withdrawn: there is nothing to rebalance.**

**⇒ THE REAL FINDING, and it undercuts every (B) in this session: THE RECALL READ IS NOT READ-ONLY.** Plasticity is
live throughout `(B)`, and the read drives pool *i* at 1400 pA for 60 steps — so Hebbian potentiates pool *i* → **all**
slots *while the answer is being read*, writing a fresh NON-selective pattern over the stored one. That is a
mechanism that would produce chance recall from a perfectly selective store, and it has been under every recall
measurement in this arc, including the 7/18-vs-8/18 decoupling result.

**▶ IMMEDIATE TEST (the read with plasticity FROZEN):** gate `concept_to_comp_attr` to 0 for the duration of the
recall, then re-run (B) on the GO arm. If recall lifts off chance, the blocker is *the read overwriting the store*
and the decoupling result must be re-measured. If it does not, the decoupling stands but for a reason still unknown.

**PROCESS: this is silent-failure #4 today and the most consequential** — an in-place GPU array write that the
engine undoes on the next step, reported as a lesion. It joins the class: **a manipulation must be VERIFIED to have
held, at the time the measurement is taken, not merely issued.** Encoded in `verify-go`.

## 🎉🎉 CONSOLIDATION WORKS — 6-seed, 2×2 factorial, both ingredients necessary and jointly sufficient

The retraction above ("the store doesn't drive the slots") led directly to the real blocker: **the recall read was
not read-only.** With that fixed, the capability lands.

**THE 2×2** (seeds 42/43/44/100/101/102; recall = hippo-lesioned, pools cued directly; **chance = 6/18**):

| | live read | **frozen read** |
|---|---|---|
| **LESION** (no `mean_subtract`) | 8/18 | **7/18** |
| **`--mean-subtract 1.0`** | 7/18 | **🎉 18/18 (3/3 on 6/6 seeds)** |

**Each ingredient ALONE gives chance; TOGETHER they give perfect recall.** Store selectivity tracks it exactly:
arm own/other **12.51–46.61** with own-is-max **3/3 on 6/6**; lesion **~1.0** with own-is-max **1/3**.
Per-fact rates are sharply selective, e.g. seed 42 `[0.60,0.23,0.35]` `[0.03,0.98,0.15]` `[0.17,0.03,1.60]`
(target dominating 3–60×).

**Anti-cheats satisfied:** permuted-target control on the store collapses in all 18 fact-seeds (≤0.154) vs lesion
~1.0–1.16 · the mechanism-OFF lesion runs at the IDENTICAL operating point so the claim is the MECHANISM · the
freeze is **asserted in the data** (`read_weight_drift +0.000000` frozen vs **+1.28 to +1.41** live — the very
assertion that exposed the earlier void lesion) · substrate physiological throughout · 6 seeds.

**WHY IT WAS BROKEN, and it was the MEASUREMENT as much as the mechanism:** recall drives pool *i* at 1400 pA for
60 steps with plasticity LIVE, so Hebbian potentiated pool *i* → **all** slots *while the answer was being read*,
overwriting the stored pattern with a fresh non-selective one (drift +1.3, comparable to the stored weights
themselves). Every (B) number in this session sat on that confound — including the "store and recall are DECOUPLED"
result I recorded as decisive.

**⚠️ HONEST SCOPE — three limits, stated plainly:**
1. **The freeze is a HOST intervention** (`_try_pgate` around the read), NOT yet neural. It is not ad-hoc — it is
   **SPEAR (Separate Phases of Encoding And Retrieval) / Hasselmo's ACh encoding-vs-retrieval mode switch**, which
   this project has ALREADY DESIGNED (`docs/plans/2026-05-19-shared-rhythm-SPEAR-conversational-architecture-design.md`,
   `2026-05-22-acetylcholine-staged-recurrence-consolidation-variant-design.md`; memory
   `feedback_conversational_path_resolution` names SPEAR as the path). The engine already exposes the exact target
   (`plasticity_gate`, `scope="gate:<name>"` — `sim/neuromodulators.py:44,70`), so the biologization is wiring an
   ACh modulator to gate `concept_to_comp_attr`, with theta-gamma supplying the timing. **Until then this is a
   tracked shortcut, per BRAIN-BASED ONLY.**
2. **NOT the full A1 gate.** This probe cues pools DIRECTLY by teacher current because word→pool binding is UNBUILT.
   It tests CONSOLIDATION in isolation. The A1 capability gate is the end-to-end test in
   `nmda_compositional_consolidation.py` main() with its four anti-cheats.
3. **Recall lacks its own scramble control.** The permuted control validates the STORE; a permuted pool→fact cue
   mapping at recall should be added before this is called closed.

**▶ NEXT:** (a) recall-side scramble control; (b) wire the freeze to an ACh neuromodulator (native target exists) to
retire the host shortcut; (c) port the winning protocol into the main runner and run the 4-anti-cheat end-to-end A1
gate at 6 seeds.

**LEDGER: 14 hypotheses, 12 refuted, 2 confirmed — and the two confirmed ones are jointly the capability.** Of five
retractions today, four were caught by an instrument built to answer a DIFFERENT question. **The single highest-value
habit this session: build the assertion into the data (`read_weight_drift`), never the comment.**

## ✅ THE OWED ANTI-CHEAT LANDS — scramble-teach: recall follows the TAUGHT association, whatever it is

The board recorded "recall still needs its own scramble control" as outstanding. Run, 6 seeds, arm vs control.

**Design note (why this form):** the obvious control — permuting the SCORING — is **true by construction** here.
With N=3 a derangement makes every trial wrong automatically, so it would "pass" while proving nothing (the same
shape as the winner-slot artifact that already cost this thread a retraction). The informative control perturbs the
**TEACHING**: drive fact *i*'s pools during the write but raise the apical plateau on slot *(i+1) mod 3*, then cue
normally at recall. `--scramble-teach`, fixed deterministic derangement (seed-independent, so it can never
accidentally coincide with the true mapping).

| | scored vs TRUE mapping | scored vs TAUGHT mapping |
|---|---|---|
| **TRUE-TEACH (arm)** | **18/18** | 18/18 (same mapping) |
| **SCRAMBLE-TEACH (control)** | **1/18** (BELOW chance 6/18) | **17/18** |

Per-seed under scramble, the recall reports the DERANGED target it was taught, e.g. seed 42:
fact 0 `[0.325, **0.600**, 0.192]` → slot 1 (taught 1) · fact 1 `[0.283, 0.008, **1.125**]` → slot 2 (taught 2) ·
fact 2 `[**1.642**, 0.225, 0.017]` → slot 0 (taught 0). The lone miss (seed 100 fact 0) is a near-tie,
0.533 vs 0.525. `read_weight_drift +0.000000` in **all 12 runs** — the freeze held in every arm.

**⇒ This is CAUSAL, not merely a null.** Recall does not just fail when the teaching is scrambled — it **follows the
scramble**. Falling BELOW chance is the signature of a genuine learned association being read out correctly:
a deranged teaching makes the true-scored answer systematically wrong, which random guessing cannot produce.

**It also pre-empts two alternative explanations by construction:** if recall were reporting RECENCY, WRITE-ORDER,
or residual state left by the write, then changing only WHICH SLOT WAS TAUGHT (the cue, the order, the drive, the
timing all identical) could not flip the answers to track the new mapping. Both the positional-artifact hypothesis
(which this arc has been burned by before — the retracted "cumulative degradation across the schedule") and the
leftover-state hypothesis are inconsistent with 17/18 taught-target accuracy under derangement.

**Remaining honest scope is UNCHANGED** (this control does not touch it): the freeze is still a HOST intervention
with a named biologization (SPEAR/ACh); the probe still cues pools DIRECTLY because word→pool binding is UNBUILT,
so this is consolidation in isolation and NOT the full A1 gate.

## 📊 THE STATISTICS, DONE HONESTLY — the trial-coupling worry resolves AGAINST the null

I flagged in the result's own scope note that the 3 facts within a seed are scored by argmax over a *shared*
3-vector, so they are NOT independent and a naive binomial on 18/18 would overstate significance. Worked through:

| arm | vs TRUE mapping | vs TAUGHT mapping |
|---|---|---|
| mean-subtract + frozen read | **18/18** | 18/18 |
| scramble-teach control | **1/18** (below chance 6/18) | **17/18** |
| lesion (no mechanism) + frozen | 7/18 (per-seed `[1,1,1,2,1,1]`, mean **1.17/3**) | — |

**1. The coupling confound CANNOT produce the result.** The mechanism that couples the trials is a per-slot
excitability bias. But a fixed slot bias **caps a seed at 1/3 correct** — if one slot systematically wins, at most
one of the three facts can be scored correct. **3/3 is unreachable under it.** The lesion arm's per-seed
`[1,1,1,2,1,1]` (mean 1.17/3) is precisely that signature. ⇒ coupling makes the observed result **HARDER**, not
easier; the worry is real but points the other way.

**2. Seed-level exact test (avoids the coupling entirely).** Treating each seed as the one independent unit and
"perfect seed" as 3/3: arm **6/6** vs scramble-control **0/6** → **Fisher exact p = 0.00108**, which is the smallest
p attainable with 6-vs-6 — i.e. the design is saturated, not marginal.

**3. For reference only** (NOT the claim, since it assumes independence): under independent unbiased argmax,
P(one seed = 3/3) = (1/3)³ = 0.037 and P(all six) = 2.6×10⁻⁹.

**4. The control does not merely fail — it INVERTS**, which is the strongest available evidence. Scoring the
scramble arm against the mapping it was actually taught gives **17/18 (94%)**. Landing *below* chance against the
true mapping is not something noise produces; it requires a real association being read out correctly.

**⇒ The trial-coupling caveat in the header is RESOLVED and rewritten.** The honest headline number is the
seed-level one: **6/6 vs 0/6, p = 0.00108.**

## 📈 CAPACITY — the mechanism holds at N=4 (the vocabulary ceiling): 24/24, and selectivity IMPROVES

The 18/18 result's own scope note conceded that a 3-way argmax is a soft discrimination. Re-ran at **N=4**, the
maximum the current 4-noun × 4-adjective vocabulary supports (`--n-facts 4`; chance drops 1/3 → **1/4**, i.e. 6/24).

| | N=3 | **N=4** |
|---|---|---|
| recall (hippo-lesioned) | 18/18 | **24/24 — 4/4 on 6/6 seeds** |
| chance | 6/18 | **6/24** |
| store own-is-max | 3/3 on 6/6 | **4/4 on 6/6** |
| firing-weighted own/other | 12.51–46.61 | **11.00–57.84** |
| `read_weight_drift` | +0.000000 | **+0.000000** (all 6) |
| substrate | physiological | physiological (all 6) |

**No interference from a fuller store — selectivity went UP, not down.** A capacity limit would show as cross-fact
bleed in the pool→slot weights degrading own/other as facts are added; the opposite happened. So N=4 is a
*vocabulary* ceiling, not a *mechanism* ceiling — testing beyond it requires building new concept pools, which is a
separate construction, and the mechanism gives no sign of being the binding constraint.

**Latent trap caught while building this:** `BASE` freezes `comp_attractor_slots` at IMPORT time from the 3-fact
list, so varying the fact count without overriding it would silently build **3 slots for 4 facts** — a fact with no
slot to write to, which would have read as a clean "capacity boundary". Slot count now follows `N`. The default
path is VERIFIED unchanged (`comp_attractor_slots` resolves to 3 either way, checked by import, not asserted).
This is the same shape as the `hebbian_max_weight` inversion and the void lesion: **a configuration artifact
presenting as a property of the biology** — the arc's dominant failure mode, now caught pre-emptively rather than
after a wrong conclusion was recorded.

**⚠️ Note for the record:** `--n-facts 4` consumes `FACTS_ALL[3]` = the WITHHELD fact, so an N=4 run **cannot also
serve as a no-confab test** (that control lives in the main runner and needs an unconsolidated concept).

**▶ N=4 scramble-teach control in flight** (4-cycle derangement) — the arm alone is not the claim.
