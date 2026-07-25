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
