> # 📍 READ FIRST — CURRENT STATE (this doc is an append-log containing TWO major reversals; the top sections are the ORIGINAL discovery, not the final picture)
>
> **1. The 2026-07-25 consolidation "boundary" was NEVER REAL.** It was an artifact of `comp_apical_R=50.0` — a **333×
> miscalibration** of a pA→mV units constant (engine default `0.15`) that parked `v_apical` at ~2×10⁵ mV and, through
> `apical_g_couple_to_soma=5.0`, drove every soma. The "dense 93%-active CA1 code" it produced was runaway current, not a
> hippocampal code. **VOID:** the dense-code re-attribution, the code-overlap ceilings, the 6-seed two-sided NO-GO, the
> M0 GO, and the "surpass = dendritic per-branch write" conclusion.
> **2. On a physically valid substrate the write LOCALIZES — 6-seed GO** (own-is-max **18/18 fact-seeds**, mean own/other
> **4.06 vs permuted 0.43**). The suppressor was **soft-bound SATURATION** of BTSP's rank-1 outer product; sweeping the
> learning rate down through the knee recovers selectivity monotonically. Real CA1 code: sparse (median cell 0 spikes),
> near-disjoint (Jaccard 0.03–0.10), fact-specific (3.6–25.3).
> **3. SCOPE:** that GO is on the **`ca1→comp_attr` SLOT** route — **not** the `cross_pool_concept` route the A1
> capability test measures. The A1 runner never sets `comp_dendritic`, so **the original A1 test was never affected by
> the miscalibration**; the VOID scope covers THIS ARC'S PROBES ONLY.
> **4. A1's own blocker is UNDER-TRAINING, not a defect** — `--train-events 200` (the default) vs a recorded **800ev →
> 87.5%** direct binding. The Hebbian/homeostasis debugging chased a non-bug. *(Separate real defect found en route:
> `hebbian_max_weight=1.0` sits below the 3.015 design weights and INVERTS the rule — the same trap already documented
> for STDP and BDSP, now seen on BTSP and Hebbian too.)*
> **5. A1's capability test is a SEPARATE, still-open thread** — it fails its own binding sanity even with everything
> applied correctly. **Four conclusions on that thread were proposed and WITHDRAWN today** (rule-can't-bind ·
> under-training · a shared-code regression · "Phase-1 is a no-op"), every one traceable to measuring a PROXY of the
> thing instead of the thing: a hand-rolled training loop instead of the runner's own, a 1-word probe instead of the
> 16-word sanity, a 200-event budget instead of the documented 800. **The one solid fact is that A1 fails at 800
> events with the topographic bias correctly applied.** Read that thread's sections in order — several supersede
> earlier ones.
> **6. The through-line:** every "boundary" in this arc was a CONFIGURATION or an INSTRUMENT measured and reported as a
> property of the BIOLOGY. Verification rules earned here are encoded in `.claude/skills/verify-go/SKILL.md`
> (mass-artifact triad · verifying a NEGATIVE · floor/ceiling comparisons are VOID · reproduce by CALLING the original
> code path · check the record for a known-good config first).

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
