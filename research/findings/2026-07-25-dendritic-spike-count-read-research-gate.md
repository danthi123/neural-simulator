# Research gate — the DENDRITIC SUSTAINED-SPIKE-COUNT READ: the missing primitive is a per-source windowed COUNT with an ABSOLUTE gate applied BEFORE the synaptic sum; but a free re-read of the existing raw artifacts shows the write-window CA1 code is NOT fact-specific, so a PREREQUISITE gate must fire first (2026-07-25)

**Read-only research gate** (no code edits) for the confirmed consolidation A1 boundary
(`2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md`, ~25 probes, ~15 write variants, the
"live lead" retracted as a winner-slot artifact `2106b143` / re-corrected `b053ea05`). The named surpass was: *a dendritic
per-cell/per-branch sustained-firing-count-threshold read that gates BOTH the write eligibility AND the recall
`ca1→slot` activation, so the sparse separable core becomes the operative set.* This doc scopes it.

**Headline (three parts, in order of what changes the build):**
1. **The missing primitive is precisely identified and is SMALL** — a per-presynaptic-source, box-car-**windowed spike
   COUNT** (with an explicit reset) passed through an **ABSOLUTE** (not network-normalized) supralinear/threshold gate,
   inserted **BEFORE** the synaptic summation, on **both** reads. All three qualifiers are load-bearing and all three are
   absent today. The bridge already contains every ingredient except this one array + gate.
2. **The multi-branch build is NOT warranted** — the project's own numpy oracle already showed K=1 (no branches) gives
   own/other 8.19 (`_consol_multibranch_oracle.py`). The dendritic requirement is the per-source *nonlinear read*, not
   per-branch clustering. This collapses the "months-scale D2 arc" to a **bounded additive `sim/` edit** (~25 lines,
   1 array, 4 config fields, default-off ⇒ byte-identical).
3. **⚠️ But a PREREQUISITE gate must fire first, and it is FREE.** A re-read of the already-committed raw artifacts
   (`research/findings/raw/consol_opsweep_gpu/twosided_*.json`) shows the realized write tracks the **during-write**
   CA1 cross-fact specificity (~1.03–1.08), **not** the isolated fire-under-tag specificity (~1.8) and not its sparse
   ceiling (3.5–7.9). **The write windows destroy the fact-specificity the isolated tag has.** Gating a non-fact-specific
   signal harder cannot manufacture selectivity — that is the exact trap ("build the amplifier onto a flat signal") that
   produced this whole boundary. §3 gives the decisive, zero-`sim/`-edit measurement that must pass before the build.

---

## 1. The precise missing primitive — what `comp_dendritic` lacks

### 1.1 What already exists on the bridge (all shipped, guarded, default-off, byte-identical when off)

| Piece | Location | What it does |
|---|---|---|
| Coincidence subunit routing mask | `sim/bridge.py:2824-2837` (`cp_coincidence_synapse_mask`), `sim/regions.py` `coincidence_detector=True` | per-synapse routing of a pathway into a dendritic subunit |
| Coincident drive (all-or-none) | `sim/bridge.py:7115-7138` — `c_drive = _co_matT @ cp_prev_firing_states` | per-post **spatial** count (or weighted sum) of routed inputs firing this step |
| All-or-none plateau kernel | `sim/kernels.py:309-343` `fused_coincidence_plateau` | sigmoid switch `σ(gain·(c−k_thresh))` → dual-exp (τ 80 ms) → Mg²⁺-gated current; `self_regen` bistable sustain |
| Graded plateau kernel | `sim/kernels.py:346-397` `fused_graded_dendritic_plateau`, drive at `bridge.py:7231-7253` (`c_weighted`) | gentle logistic on the **weighted** coincident drive (non-saturating analog read-out) |
| Two-compartment apical | `sim/bridge.py:7139-7194` (`cp_v_apical`, `apical_R/tau/g_couple/g_couple_to_soma`), KIR down-state `7177-7183` | separate apical voltage, bistable band, attenuated apical→soma read |
| BTSP write | `sim/bridge.py:8023-8110`, kernels `552/580/613/644` | `dw = η·Etilde_pre·IS_post·(w_max−w)`; `IS_post = max(v_apical − v_hold, 0)` (`bridge.py:8053`) |
| BTSP pre-eligibility | `sim/bridge.py:8042-8045` | per-neuron **exponential low-pass** of `fired_this_step`, τ = `btsp_elig_tau_ms` |
| Per-source supralinear gate (write only) | `sim/bridge.py:8060-8066` `btsp_elig_exponent` | normalize-by-`etilde.max()`-then-`**p` |
| Per-source hard threshold (write only) | `sim/bridge.py:8067-8081` `btsp_elig_hard_thresh` | zero eligibility below `thresh · etilde.max()` |
| **True windowed spike COUNT with reset** | `sim/bridge.py:3541-3568` (`start_engram_recording` → `rec["spike_counts"] += fired_f32`, `_tick_engram_recordings`) | **exists, but is host-side engram bookkeeping only — never read by any plasticity or drive path** |

### 1.2 The gap, stated exactly

Both operative reads apply their nonlinearity **after** summing over synapses and are **linear in each presynaptic
source's spike count over the window**:

- **RECALL:** `c_drive = Σ_j w_ij · x_j(t)` with `x_j = cp_prev_firing_states` — a **binary, this-step** indicator
  (`bridge.py:7135`; graded sibling `c_weighted`, `bridge.py:7253`). Summed over a W-step window this is
  `Σ_j w_ij · count_j(W)` — **linear per source**. A source that fires once contributes exactly as much per spike as one
  that fires thirty times. The sigmoid (`kernels.py:325`) acts on the *sum*, per step.
- **WRITE:** `Etilde_j` is an exponential low-pass (`bridge.py:8043-8045`) — a **recency-weighted rate**, not a windowed
  count with a reset.

The consequence is the boundary doc's own decisive algebra: because `w[k→slot_j] ∝ g_write(fire_j[k])` and the recall
reads `Σ_k g_recall(fire_i[k])·w[k→slot_j]`,

> **own/other ≈ Σ_k g(f_i[k])·g(f_i[k]) / mean_j Σ_k g(f_i[k])·g(f_j[k]) = the self/cross overlap of the code AS SEEN
> THROUGH g.**

Measured on this substrate: **g = identity (fire-count) → 1.30–1.54**; **g = 1[count > 0.25·W] (the binary sustained
core) → 4.8–8.0** (`sparse_core_ceiling`, and `_consol_decoupled_plateau_probe`'s sparse ceiling 5.56). No write rule can
exceed the ceiling set by the *worse* of the two sides — which is exactly why ~15 write-side-only variants all topped out
at ~1.0–1.2.

**⇒ THE MISSING PRIMITIVE (one sentence):**

> A **per-presynaptic-source, box-car-WINDOWED SPIKE COUNT with an explicit reset**, passed through an **ABSOLUTE
> (non-network-normalized) supralinear/threshold gate**, applied **BEFORE the synaptic summation**, on **BOTH** the
> recall drive vector and the write eligibility.

Three separable sub-gaps, all currently absent — and note the write side already has *approximations* of two of them,
which is why the write-side-only de-risks looked like they should have worked and did not:

| # | Sub-gap | Recall side (`c_drive`) | Write side (`Etilde`) |
|---|---|---|---|
| **(i) ORDER** — per-source nonlinearity **before** the sum | **ABSENT.** `c_drive` is a *spatial* coincidence count: it structurally cannot distinguish "90 halo cells firing once" from "20 core cells firing 10× each" when the halo is 4.5× more numerous. | present (`btsp_elig_exponent` / `_hard_thresh` are per-source) |
| **(ii) WINDOW SHAPE + RESET** — box-car over the cue window, reset at onset | **ABSENT** (this-step binary) | **ABSENT.** Exponential low-pass only. τ=1000 ms integrates **across facts** (documented cross-fact compression: "100% of synapses survive at thresh 0.25"); τ=30 ms is recency-dominated. The separable structure is defined by an **absolute box-car** criterion (>25% of the 40-step window ⇒ **≥10 spikes / 40 steps**) — a shape the exponential trace never has. |
| **(iii) THRESHOLD REFERENCE** — absolute, not network-relative | **ABSENT** | **ABSENT.** `etilde / etilde.max()` takes `.max()` over **every synapse in the bridge** (`bridge.py:8064, 8074`). Biologically a spine thresholds its *own* accumulated Ca²⁺ against a fixed molecular set-point; there is no "network max" in a spine. This is also a **brain-based-only** concern: a global max is a non-local host-style op. |

**This also explains the retracted two-sided attempt without contradiction.** `_consol_twosided_generalize_probe.py`
gated the recall by weighting core cells by their **raw fire count** (`w_pre[ci] = fire[i][core[i]]`, line ~168) — i.e.
`g` = identity restricted to the core's *support*, not a binary/Hill gate; and the write gate was
`btsp_elig_hard_thresh` on the network-relative exponential low-pass, which the finding itself measured as **not
cutting**. So the "two-sided read" has been *attempted*, but the actual `g` was ≈ identity on **both** sides. The genuine
residual is untested — with the important caveat in §3.

### 1.3 What is NOT the missing primitive (scope reduction — saves the months-scale build)

- **Per-branch clustering / multi-branch dendrites.** `_consol_multibranch_oracle.py` already measured **K=1 (no
  branches) → own/other 8.19**; oracle-clustered 7.8–34; random-assignment 3.5. Branch assignment does not move it.
  The mechanism carrying the localization is the *rate-proportional read on the fact-specific core*, not compartments.
- **Decorrelation / pattern separation.** Already falsified as the lever across ~10 methods (fixed FFI, sparse commit,
  gentle/FF reinstatement, divisive-norm, homeostasis, MSN phenotype). The separable structure **already exists** in the
  sustained-count code (near-disjoint, Jaccard 0.06–0.12). This is dendrites-for-the-nonlinear-READ, not
  dendrites-for-decorrelation — the correction the boundary doc already recorded.
- **Bistability / plateau HOLD.** Shipped and gap#5-validated (`coincidence_plateau_self_regen` + `apical_kir_g`).

---

## 2. The biology of sustained-count dendritic thresholding (sources READ, not skimmed)

### 2.1 Kandel 6e Ch 13, pp. 296–298 — read directly (`~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt` @ offset 1661595)

Three quoted mechanisms, all directly on point:

1. **The NMDA spike is a locally regenerative, purely local event.** "Moderate synaptic stimuli are able to activate a
   sufficient number of AMPA receptors to produce an intermediate level of depolarization that is able to lead to
   expulsion of Mg²⁺ from a fraction of NMDA receptors… produce a further depolarization that leads to even greater
   unblocking of Mg²⁺… In some instances, this leads to a local regenerative depolarization, referred to as an **NMDA
   spike**. Such NMDA spikes are purely local events — they cannot propagate actively in the absence of synaptic
   stimulation because they require glutamate release." *(This is what `fused_coincidence_plateau` already models.)*
2. **The spine is a PER-SYNAPSE accumulator with a diffusion barrier — the biological locus of a per-source count.**
   "their thin necks provide a barrier to diffusion of various signaling molecules from the spine head to the dendritic
   shaft. As a result, a relatively small Ca²⁺ current through the NMDA receptors can lead to a relatively large increase
   in [Ca²⁺] that is **localized to the head of the individual spine that is synaptically activated**… Because the thin
   spine neck restricts… the rise in Ca²⁺ and, thus, long-term plasticity, **to the spine that receives the synaptic
   input**, spines also ensure that activity-dependent changes in synaptic function… are **restricted to the synapses
   that are activated**." Figure 13-18 plots the spine Ca²⁺ transient on a **100 ms** time base.
   **⇒ This is the primitive verbatim: each synapse independently integrates ITS OWN presynaptic activity over ~100 ms,
   and the accumulation does NOT pool across synapses.** A spine receiving repeated (sustained) presynaptic firing builds
   a large local [Ca²⁺]; a spine receiving one spike does not — *regardless of how many other spines fired once.* That is
   exactly what our spatial-coincidence `c_drive` cannot express.
3. **The plateau potential is the long-lasting, count/pairing-gated form.** "when a distal stimulus is paired with a
   proximal stimulus, the backpropagating spike summates with the distal EPSP to trigger a **long-lasting type of
   dendritic spike called a plateau potential**, which depends on activation of voltage-gated Ca²⁺ channels and NMDA
   receptors. When the plateau potential arrives at the cell body, it can trigger a brief burst."
   Also: "Ca²⁺ accumulation thus provides, **at an individual synapse, a biochemical detector**… which is thought to be a
   key requirement of memory storage (Chapter 54)." — *at an individual synapse*, i.e. per-source, not per-cell.

### 2.2 Polsky, Mel & Schiller, *J Neurosci* 29:11891 (2009), "Encoding and Decoding Bursts by NMDA Spikes in Basal Dendrites of L5 Pyramidal Neurons" ([PMC3850222](https://pmc.ncbi.nlm.nih.gov/articles/PMC3850222/)) — the decisive quantitative source

This paper *is* the sustained-count primitive, measured:

- **Frequency/count dependence:** high-frequency inputs (ISI 10–20 ms) trigger the NMDA spike **at the second pulse and
  at lower stimulus intensity**; low-frequency inputs (ISI > 200 ms) require **1.6× higher intensity** (p = 0.005).
- **Integration window:** ≈ **72.2 ± 7.7 ms** for *in vivo*-like patterns. (Our `coincidence_tau_decay_ms = 80.0`,
  `config.py:256`, is already at this value — the window constant is right; what is missing is *what* is integrated.)
- **Mechanism = a PER-SYNAPSE memory, not a voltage effect:** the facilitation is carried by **residual glutamate
  prebound to NMDARs from previous activations**, which persists "for several hundred milliseconds." Controls:
  cancelling residual voltage in simulation changed required conductance by **< 3%**, whereas removing prebound NMDARs
  required **60% higher** conductance. Paired-pulse recovery τ = **1143 ± 449 ms**.
- **The exact discrimination we need, measured:** *"High-frequency bursts from **few** afferents proved more effective
  than single spikes from **many** afferents — only **5–10 coincident synapses firing at > 50 Hz** could initiate spikes,
  while **20+ randomly distributed synapses** required lower frequencies."*
  **⇒ A real basal dendrite prefers 5–10 SUSTAINED sources over 20+ TRANSIENT ones. Our `c_drive` has the opposite
  preference by construction (it is a headcount).** This is precisely the dense-halo-vs-sparse-core failure.

### 2.3 The threshold is ABSOLUTE and ultrasensitive — CaMKII (Hill ≈ 8)

Bradshaw, Kubota, Meyer & Schulman, *PNAS* 100:10512 (2003) and the *Nat Rev Neurosci* "Ultrasensitive switch"
commentary: formation of autonomous (autophosphorylated) CaMKII has a **Hill coefficient ≈ 8 with respect to Ca²⁺** —
"indicating a **virtual threshold**"; with PP1 in the reaction the dependence is steeper still. CaMKII "might remain
dephosphorylated in response to small increases in Ca²⁺ (such as those that elicit LTD), but strongly autophosphorylate
in response to larger elevations (such as those that elicit LTP)," and autophosphorylation sustains activity long after
Ca²⁺ falls.
**⇒ The per-source gate should be a steep Hill function of the ACCUMULATED count against a FIXED set-point
(`c^n/(c^n+θ^n)`, n ≈ 4–8), not a comparison against a network-wide maximum.** This directly indicts sub-gap (iii).
*(Honest caveat: the strict bistable-switch reading of CaMKII is contested — recent spatial models argue for prolonged
-but-temporary autophosphorylation / kinetic proofreading rather than true bistability. The ultrasensitivity (steep,
absolute threshold) is what we rely on and is not contested; we do not need the bistability claim.)*

### 2.4 Where the standard modelling literature sits (why this is a genuine gap, not an oversight)

Major, Larkum & Schiller's canonical framing — and the mainstream plateau models built on it, e.g. Kastellakis-style
sequence models ([Frontiers in Cognition 2023, 10.3389/fcogn.2023.1044216](https://www.frontiersin.org/journals/cognition/articles/10.3389/fcogn.2023.1044216/full),
read: *"a volley of 4–20 or even up to 50 spikes within a window of 1 ms to 4 ms is needed"*) — model the plateau as a
**fast spatial coincidence detector whose OUTPUT persists for hundreds of ms**. That is exactly what our sim implements,
faithfully. The sustained-count read is the **complementary, less-modelled** property (Polsky/Schiller's prebound-
glutamate frequency facilitation + Kandel's per-spine Ca²⁺ compartmentalisation). So we are not fixing a bug — we are
adding a second, independently-attested dendritic nonlinearity that the coincidence model omits. Cluster **G.02** of the
reference catalog (`feature-catalog.md`, "Active dendrites — local computation, dendritic spikes", Kandel 6e Ch 13
p 293–298) lists exactly this family as **"Sim status: missing"**.

---

## 3. ⚠️ PREREQUISITE GATE — a free re-read of the committed raw artifacts says the write-window code is NOT fact-specific

This is the most decision-relevant thing in this doc, and it costs nothing: `_consol_twosided_generalize_probe.py`
already computes and saves a **cross-fact core-firing matrix** in two conditions — `xfire_under_tag[K][W]` (mean firing
of fact-K's core under fact-W's isolated tag) and `xfire_during_write[K][W]` (the same cores during fact-W's actual write
reinstatement windows). Those matrices are in the committed JSONs and were never analysed.

Computed from `research/findings/raw/consol_opsweep_gpu/twosided_*.json`, seed 42
(specificity = diagonal ÷ mean off-diagonal, per core):

| run | core-specificity **UNDER TAG** | core-specificity **DURING WRITE** | realized core-gated own/other | sparse ceiling (under tag) |
|---|---|---|---|---|
| `reset_blocked_settle30` (full isolation) | [1.93, 1.90, 1.59] **mean 1.81** | [1.61, 0.75, 0.90] **mean 1.08** | **[1.045, 1.024, 1.006]** | [7.41, 7.71, 3.55] |
| `blocked` | [1.79, 1.93, 2.05] mean 1.92 | [0.79, 1.10, 1.15] mean 1.01 | [0.43, 0.44, 3.70]* | [7.28, 8.86, 6.18] |
| `twosided` (interleaved) | [1.62, 1.78, 2.09] mean 1.83 | [1.07, 0.89, 1.13] mean 1.03 | [0.46, 3.25, 0.49]* | [3.48, 5.00, 7.90] |

\* the non-flat entries are the retracted winner-slot artifact (`slot_mean_weight` = [24.4, 24.1, **80.4**] and
[24.2, **80.5**, 23.8]); the top row is the properly-isolated run where `slot_mean_weight` = [5.88, 5.80, 5.80], i.e.
equalized, and the permuted-core control collapses to [0.94, 0.95, 1.04]. **That top row is the trustworthy one.**

**The attribution is clean and, as far as I can find, new:**

> **The realized write own/other (mean 1.03) tracks the DURING-WRITE cross-fact specificity (1.08) almost exactly — NOT
> the under-tag specificity (1.81), and nowhere near the under-tag sparse ceiling (3.5–7.9).**

Two further reads of the same matrices sharpen it:
- In the fully-isolated run, `xfire_during_write` **column** 0 is (core₀ 8.47, core₁ **10.8**, core₂ 7.76) — the diagonal
  is **not the column maximum**; the same holds for facts 1 and 2. Under the isolated tag the diagonal *is* clearly
  dominant (col 0 = 13.4, 5.85, 7.59).
- Column magnitudes decay monotonically across the blocked schedule (≈ 7.8–10.8 → 5.8–6.6 → 4.7–6.7). **The dominant
  variance in the write-window firing is a global order/adaptation drift, not fact identity** — plausibly the strongly-
  adapting `IZH2007_STRIATAL_MSN` hippocampal phenotype over 3×30 back-to-back burst steps.

**⇒ Consequence for this research gate (stated honestly):** the sustained-count read's premise is that the write's
eligibility is ∝ the fact-specific fire-under-tag pattern. **On the current write protocol that premise is falsified.**
A harder per-source gate applied to a signal whose fact-specificity is ~1.0 cannot manufacture selectivity — it is the
same "build the amplifier onto a flat signal" trap that produced this boundary, and the r-iii "no specific structure to
amplify" failure the earlier design doc already foregrounded.

**This does NOT kill the mechanism** — the sustained-count gate is still the right *read*, and the gated ceiling on the
isolated code is 4.8–8.0. It re-orders the build: **measure whether a sustained-count gate recovers fact-specificity in
the DURING-WRITE window before spending a `sim/` edit on it.** That is mechanism **M0** below, and it needs no `sim/`
edit at all.

---

## 4. Ranked cheap-first buildable mechanisms

Ordered by information-per-minute. Every one keeps the existing `comp_dendritic` machinery; none needs multi-branch
dendrites (§1.3).

### M0 — **[FREE, no `sim/` edit, decisive]** Gated-ceiling oracle on the DURING-WRITE window counts

**What.** `_consol_twosided_generalize_probe.py` already captures per-cell `window_fire[W]` (the true box-car spike count
per CA1 cell over fact-W's write windows, `instrumented_write` line ~63). Add ~6 lines of *probe-side numpy* (research
file only) to report, alongside the existing `sparse_core_ceiling`:
- `write_core_sizes` / `write_core_jaccard` — the >θ-count core computed **on `window_fire`**, not on `fire_under_tag`;
- `gated_ceiling_during_write[i]` = `Σ_k g(Wf_i[k])² / mean_j Σ_k g(Wf_i[k])·g(Wf_j[k])` for `g` = binary
  `1[c ≥ θ]` and `g` = Hill `c⁸/(c⁸+θ⁸)`, swept over absolute θ;
- the same with `g` = identity (the current 1.3–1.5 baseline) as the in-run reference.

**Why it is decisive.** This is the *exact analytic ceiling* the two-sided read can achieve, evaluated on the signal the
write actually sees. It costs one probe re-run (or, if `window_fire` is re-dumped, pure host numpy).
**GO:** `gated_ceiling_during_write ≥ 2.5` at some absolute θ ⇒ the sustained-count read is warranted → build **M1**.
**KILL:** `< 2.5` at every θ ⇒ no per-source gate on either side can pass, and the lever is the **write-window drive
protocol** (M0b), not dendrites. Given §3's numbers, **KILL is the more likely outcome and that is exactly why this must
run first.**

**Reused machinery.** The whole probe (build, encode, isolated write, cores, permuted/random controls, per-slot weights).
**Anti-cheats.** Report `g`=identity in the same run (must reproduce ~1.3–1.5, proving the harness is unchanged); report
the permuted-core ceiling (must be ~1.0); report `write_core_sizes` (must not be 0/1-cell degenerate — the fact-0
degeneracy that inflated the retracted lead); 6 seeds.

### M0b — **[FREE, no `sim/` edit]** Restore the isolated tag's specificity inside the write window

**What.** Make the write-window condition identical to the `_fire_under_tag` measurement condition: full recovery between
bursts (long settle / adaptation reset), one fact per bridge-quiet epoch, identical drive and window length; then re-read
`xfire_during_write`. Optionally drop the strongly-adapting `hippo_izh_type=IZH2007_STRIATAL_MSN` for the write phase.
**GO:** `xfire_during_write` specificity → ≈ `xfire_under_tag` (≥ 1.6 mean, diagonal = column max 3/3).
**Why it matters.** M1's ceiling is bounded by whichever code the write sees. If M0b restores specificity, M0's gated
ceiling should jump — run M0 *after* M0b if M0 alone kills.
**Anti-cheat.** The under-tag matrix must be unchanged by the protocol change (it is measured on a quiet bridge) — if it
moves, the two conditions are not comparable and the comparison is void.

### M1 — **[RECOMMENDED BUILD, bounded additive `sim/` edit]** Per-source sustained-count gate on the coincidence/graded drive vector

**The minimal `sim/` edit** (additive, default-off, byte-identical when off — mirrors `btsp_elig_hard_thresh` exactly):

1. `sim/config.py` — 4 fields:
   `sustained_count_window_steps: int = 0` (0 ⇒ **off**, the byte-identical default), `sustained_count_theta: float`
   (an **ABSOLUTE** spike count, e.g. 10 = ">25% of a 40-step window"), `sustained_count_hill_n: float = 8.0`,
   `sustained_count_gate_write: bool = False`.
2. `sim/bridge.py` — one array `cp_sustained_count` (n,), allocated next to `cp_neuron_activity_ema` (`bridge.py:1595`),
   ticked in the step exactly like `_tick_engram_recordings` (`bridge.py:3557-3568`, the existing box-car idiom):
   `cp_sustained_count += fired_f32`, with a **reset** every `sustained_count_window_steps` steps (or via a public
   `reset_sustained_count()` a runner calls at cue onset — the biologically-honest form, since a real cue onset resets
   nothing; the decaying-box-car variant below is the more faithful one).
3. **RECALL gate** (`bridge.py:7115-7138` and its graded sibling `7231-7253`): when the feature is on, replace the matvec
   vector with the gated one —
   `s = c^n / (c^n + θ^n)` where `c = cp_sustained_count` (Hill, ABSOLUTE θ — §2.3);
   `c_drive = _co_matT @ (cp_prev_firing_states.astype(f32) * s)`.
   Multiplying by `s` (rather than replacing) keeps the plateau spike-driven (an NMDA spike still needs glutamate
   release — Kandel §2.1) while making a source's contribution supralinear in its sustained count. **Guard:**
   `if sustained_count_window_steps <= 0: <unchanged expression>` ⇒ byte-identical.
4. **WRITE gate** (`bridge.py:8055-8081`): when `sustained_count_gate_write`, replace the network-relative
   `etilde/etilde.max()` normalisation with the same **absolute** Hill gate on `cp_sustained_count[coo_bt.row]`. This
   fixes sub-gaps (ii) and (iii) on the write side and makes `g_write ≡ g_recall` — which is what the ceiling algebra
   requires.

**Size.** ~25 lines, 1 array, 4 config fields, 2 guarded call sites. **NOT a months-scale arc.**
**A more faithful variant (M1′, +3 lines):** make the counter a **leaky** box-car, `c ← c·exp(−dt/τ_c) + fired`, with
`τ_c ≈ 70–100 ms` — this *is* Polsky/Schiller's prebound-glutamate accumulator (§2.2: 72.2 ± 7.7 ms window, several-
hundred-ms glutamate residence) and removes the artificial hard reset. Prefer M1′ if M0 passes; it is strictly more
biological and no more code.
**Ideal variant (M2, see below):** per-**synapse** rather than per-source.

**Cheapest de-risk.** On `_consol_decoupled_plateau_probe.py` / `_consol_twosided_generalize_probe.py` unchanged except
for the new flags — the harness already produces every metric needed (`core_gated_own_over_other`,
`permuted_core_own_over_other`, `random_ca1_own_over_other`, `slot_mean_weight`, `sparse_core_ceiling`, `dw`,
`core_sizes`, `thr_hash`).

**GO-gate (6 seeds 42/43/44/100/101/102):** `core_gated_own_over_other ≥ 2.5` **AND** `own_is_max` on **3/3 facts**, on
**≥ 5/6** seeds.

**Anti-cheat controls (ALL mandatory — this is the arc that already produced one retracted lead):**
| Control | Requirement | Rationale |
|---|---|---|
| **PERMUTED-CORE** (`verify-go` lens 7) | read slot_i weighted by fact-(i+1)'s core → **must collapse to ~1.0** | the retracted lead's permuted control read **3.37** — unearned winner-slot bias |
| **RANDOM-SOURCE** | read with a random equal-size CA1 set → **must collapse to ~1.0** | ditto (the lead's read 3.28) |
| **PER-SLOT RAW MAGNITUDE** | print `slot_mean_weight` per slot; **max/min ≤ 1.3** | `[24, 80, 24]` was the smoking gun; the isolated run's `[5.88, 5.80, 5.80]` is the acceptable shape |
| **LESION (the load-bearing test)** | `sustained_count_window_steps = 0`, everything else identical → own/other **must fall back to ~1.0–1.5** | proves the gate, not the schedule, carries the result |
| **GATE-ENGAGED** | report the surviving fraction (`n(s>0.5)/n`) per region; must be **sparse but non-empty** (~10–25%) | the write-side hard-threshold NO-GO was partly "100% of synapses survive" — a gate that does not cut is a void arm |
| **DEGENERATE-CORE guard** | `core_sizes ≥ 5` for every fact | the retracted lead's fact-0 core was **1 cell** |
| **CEILING REFERENCE** | report `sparse_core_ceiling` **and** the new during-write gated ceiling in the same run | own/other must be ≤ ceiling; if it exceeds it, the metric is broken |
| **SEEDING** | print `thr_hash` (`cp_neuron_firing_thresholds` md5); **6/6 unique** | `actual_seed_used` seeds nothing (CLAUDE.md) — the probe already does this |
| **BYTE-IDENTICAL-WHEN-OFF** | an assertion, not a comment: a default-config run before/after the edit must be bit-equal | the standing `sim/`-edit discipline |

### M2 — **[larger, more faithful]** Per-SYNAPSE prebound-glutamate accumulator

**What.** Move the counter from per-neuron (`n`,) to per-synapse (`nnz`,): `cp_syn_glut_resid[e] ← c·exp(−dt/τ) + pre_fired`,
gated Hill-wise, used as a multiplier inside the routed matvec data. This is the literal Kandel/Polsky biology (§2.1–2.2:
Ca²⁺ and prebound glutamate are **per-spine**, not per-cell), and it is the only version that discriminates a source that
is sustained *onto this dendrite* from one that is sustained generally.
**Cost.** An `nnz`-sized float32 array (the bridge already carries several: `cp_eligibility_trace`,
`cp_coincidence_synapse_mask`) + one fused kernel, ~40 lines.
**Why ranked below M1.** For the consolidation task each CA1 cell has one synapse per slot, so per-source ≡ per-synapse
*for this measurement* — M1 buys the same result for less code. M2 is the right generalisation once the mechanism is
proven, and is the honest "faithful" target.
**Reuse note (a genuinely cheap M2 approximation, worth a single run):** the existing **STP facilitation** variable `u`
*is* a per-synapse trace of recent presynaptic activity (`fused_stp_decay_recovery`, `config.py:545-553`,
per-type `stp_U_per_type` / `stp_tau_f_per_type` at `config.py:844-849`), and `coincidence_weighted_drive=True`
(`config.py:447`) already feeds `effective_connections_matrix.data` — **which carries STP** — into the coincident drive
(`bridge.py:7121`). So a strongly-facilitating, weakly-depressing STP on `ca1→comp_attr` gives a *linear* per-source
sustained multiplier **with zero `sim/` edit**. It will not reach the gate on its own (linear, and `x` depression fights
it), but it is a 1-config-line probe of whether *any* per-source weighting moves the number — worth running alongside M0.

### M3 — **[deprioritised]** Multi-branch / per-branch clustered gating

Explicitly **not warranted**: the oracle already measured K=1 → 8.19 (§1.3). Revisit only if M1/M2 pass and a *second*
capability (e.g. multi-attribute binding) demands compartments.

---

## 5. Recommended first build + de-risk

**Run M0 first (free, decisive, ~1 GPU-hour or less).** Extend `_consol_twosided_generalize_probe.py` (research file
only) to report the **gated ceiling on `window_fire`** — binary and Hill `g`, absolute-θ sweep — alongside the existing
identity baseline, permuted-core control, and per-slot magnitudes, at 6 seeds.

- **M0 GO (gated during-write ceiling ≥ 2.5):** build **M1′** (leaky-box-car per-source Hill gate, τ_c ≈ 80 ms, absolute
  θ) as the bounded additive `sim/` edit; de-risk on the same probe against the full anti-cheat table; 6 seeds.
- **M0 KILL:** do **not** spend the `sim/` edit. Run **M0b** (restore the write window's fact-specificity), then re-run
  M0. If M0b also fails to lift the during-write specificity above ~1.6, then the load-bearing residual is **not** the
  dendritic read at all — it is that *this consolidation write protocol destroys the fact-specific CA1 code it is
  supposed to store*, and the next method is a different reinstatement/write architecture (per THE LAW, a verdict on the
  method, not the capability).

Run the cheap M2-approximation (facilitating STP + `coincidence_weighted_drive`) **in parallel** with M0 — it is one
config line and shares the harness.

**Ordering rationale.** M0 costs a probe edit and no `sim/` edit; it either warrants or forecloses the entire build. The
whole reason this boundary took ~25 probes is that amplifiers were repeatedly built onto signals that had already been
measured flat. M0 is the measurement that prevents the 26th.

---

## 6. Verdict

**On the mechanism:** the surpass is a **bounded, additive `sim/` edit reusing `comp_dendritic`** — ~25 lines, one
per-neuron array, four default-off config fields, two guarded call sites (`bridge.py:7115-7138`/`7231-7253` for recall,
`8055-8081` for write). It is **NOT** the months-scale multi-branch dendritic arc: the project's own numpy oracle already
showed branch structure is not the operative variable (K=1 → 8.19), and the bridge already ships every other ingredient
(two-compartment apical, KIR bistability, plateau kernels, BTSP, and even a box-car spike counter at `bridge.py:3552-3568`
that is simply not wired to any drive or plasticity path). The missing primitive is small, precisely located, and
biologically well-attested (Kandel Ch 13 per-spine Ca²⁺ compartmentalisation; Polsky/Schiller 2009 prebound-glutamate
frequency facilitation with a 72 ms window, "5–10 sustained afferents beat 20+ transient ones"; CaMKII Hill ≈ 8 giving an
absolute, not network-relative, threshold).

**On whether to build it now — NO, not yet, and this is the load-bearing conclusion.** A free re-read of the committed
raw artifacts (§3) shows the realized write own/other (1.03) tracks the **during-write** cross-fact specificity (1.08),
not the isolated fire-under-tag specificity (1.81) nor its sparse ceiling (3.5–7.9): **the write windows destroy the
fact-specificity the gate would amplify.** Building M1 before M0 would repeat, for the 26th time in this arc, the exact
error that created the boundary. **Run M0 (free, decisive); build M1′ only on its GO.**

**Honest scope of this doc.** §1–§2 (the primitive and its biology) are solid and source-grounded. §3 is a new,
quantitative attribution derived from already-committed artifacts — it has **not** been re-run or independently verified,
and it rests on a mean-over-core statistic rather than the per-cell inner product the write actually realizes (which is
why M0 computes the inner-product ceiling directly rather than trusting the mean). §4's rankings are pre-registered
predictions, not results. Nothing here is a GO.

## Provenance
Read-only research gate, 2026-07-25. Sources read in depth: Kandel 6e Ch 13 pp. 296–298 (local corpus, full text);
`feature-catalog.md` G.02; Polsky/Mel/Schiller *J Neurosci* 2009 (PMC3850222); Bradshaw et al. *PNAS* 2003 + *Nat Rev
Neurosci* "Ultrasensitive switch"; Kastellakis-style plateau-sequence model (Front. Cognition 2023) as the contrasting
mainstream framing. Substrate read directly: `sim/bridge.py`, `sim/kernels.py`, `sim/config.py`,
`research/runners/_consol_decoupled_plateau_probe.py`, `_consol_twosided_generalize_probe.py`. §3 computed from
`research/findings/raw/consol_opsweep_gpu/twosided_{reset_blocked_settle30,blocked,}_seed42.json`. Anti-cheat table per
`.claude/skills/verify-go/SKILL.md` lens 7 (permuted-target / winner-slot artifact). NO code edits.
