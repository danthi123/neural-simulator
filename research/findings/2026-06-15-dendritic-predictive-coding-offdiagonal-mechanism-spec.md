# Dendritic predictive coding for the OFF-DIAGONAL: a precise, buildable mechanism spec (the faithful Mikulasch-Priesemann / interneuron-whitening answer)

**Status:** READ-ONLY deep-research scout (the standing "deep research FIRST before a months-scale commit" move). NO `sim/` edits, NO build, NO GPU. Single deliverable = this doc. **Date:** 2026-06-15. **Role:** read-only research subagent. Every load-bearing mechanism claim was read from a primary source (Mikulasch-Priesemann PNAS 2021 / arXiv 2010.12395 full text; the Lipshutz-Golkar-Chklovskii normative-framework arXiv 2302.10051 full text; the Duong-Lipshutz-Chklovskii-Simoncelli adaptive-whitening ICML 2023 arXiv 2301.11955 full text; Lipshutz-Simoncelli NeurIPS 2024 abstract). Every load-bearing *project* fact was re-verified against the repo (file/line cited). This is a scoping/build-spec doc, NOT a brain-based result and NOT a commitment to build.

**The owner's standing approval (verbatim scope):** build a functional learned cortex that recovers REAL text-corpus category structure WITHOUT curated concepts — which requires the OFF-DIAGONAL (cross-neuron) decorrelation that no local online mechanism has yet reached on the project's substrate. This doc produces the PRECISE, BUILDABLE specification of the faithful dendritic-predictive-coding / Mikulasch-Priesemann mechanism so a cheap-first numpy de-risk can be built correctly.

**The two-line answer (the rest is detail):** the somatic similarity-matching network collapses because its **feedforward rule is Oja** (`ΔW ∝ y·xᵀ − y²·W`) — a *purely correlational* rule that, on a correlated corpus, drives correlated units to learn the same weights (Mikulasch's named "vicious cycle"), collapsing W onto 1–4 PCs. The escape is to replace the Oja feedforward rule with an **error-gated** rule in which a **per-input dendritic compartment first cancels the predictable common-mode input via recurrent inhibition, and the feedforward weight then learns only on the residual prediction error** (`ΔW ∝ y·(x − x̂)ᵀ`, the residual delivered per-compartment). Because the common mode is subtracted *before* plasticity, correlated units stop reinforcing each other → no W-collapse; because the decorrelation is realized by a **bounded interneuron pool whose size = the rank** and a **shaped (non-identity) target**, it does not over-whiten the noise subspace. The single cleanest buildable realization is the **Duong-Lipshutz fixed-random-frame + plastic-gains whitening** (`Cₓₓ^{1/2} = I + W·diag(g)·Wᵀ`, gain rule `Δgᵢ ∝ z̄ᵢ² − 1`), where K interneurons impose exactly K marginal-variance constraints so the rank is controlled by construction and over-whitening is structurally impossible.

---

## 1. DIAGNOSIS recap (the off-diagonal low-rank-convergence problem)

On a concept × context-hub co-occurrence matrix from real TinyStories text (64 concepts, 8 categories, ~500 hubs), recovering category structure (Pearson of cosine codes vs the a-priori same-category block `S_true`) decomposes into a **DIAGONAL** half (per-feature normalization — solved locally by subtractive adaptation or a per-hub divisive gain; caps ~+0.22–0.31 on real) and an **OFF-DIAGONAL** half (cross-neuron low-rank decorrelation — the wall: reachable *offline* by rank-8 ZCA at +0.437–0.49 ≈ host PPMI+SVD +0.442, but NO local online mechanism converges to it). Eight prior multi-seed probes precisely characterized *why the obvious online rules fail*, and this spec builds on that characterization (verified, not re-derived):

- **A LEARNED feedforward W (Oja) + anti-Hebbian lateral M** (the Pehlevan-Chklovskii somatic similarity-matching network = the project's `learn_simmatch` / `graded_lateral`): the W **collapses onto the top 1–4 principal components** (eff-rank 3–4 at k=64, eff-rank 1.1 at a k=8 bottleneck) → plateaus at +0.35. The joint projection-plus-decorrelation does not converge.
- **A FIXED random W + learned lateral M**: **over-whitens** (eff-rank 44, +0.32) — keeps all dimensions, never denoises to rank-8.
- Only the offline SVD selects exactly the top-8 informative directions AND whitens them (+0.49).

So the core unsolved problem is **an ONLINE, LOCAL mechanism that converges to the low-rank (≈8) off-diagonal whitening — selecting the right informative subspace AND decorrelating it — without the W-collapse and without over-whitening.** The off-diagonal deep-research (`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md`) already established that this is a *mathematical* fact (a diagonal `D` cannot rotate off-diagonal correlations away; every biological off-diagonal decorrelator is a cross-neuron recurrent interaction) and that **D2-as-designed (per-hub divisive gain `x/(σ+g)`) is provably diagonal-only** (measured +0.216 = 49% of host). The cheap *point-neuron* lateral route (Option O1) was then **falsified** (CYCLE 87: forcing the somatic lateral low-rank collapses it to rank-1, +0.198). **This doc goes deeper on the one remaining biological route the prior arc flagged but did not specify to a buildable level: the dendritic-predictive-coding family (Option O2 + the interneuron-whitening O3 that is its rate cousin).**

---

## 2. THE BUILDABLE MECHANISM

I lay out the mechanism in three layers: (2.1) the *faithful* dendritic model (Mikulasch-Priesemann + the normative framework) with exact equations; (2.2) the precise mechanistic reason it escapes BOTH failure modes — and an honest statement of the residual risk; (2.3) the **single cleanest buildable form** for the de-risk (Duong-Lipshutz fixed-frame + plastic-gains), which is the one I recommend coding first because its rank control and over-whitening immunity are *structural*, not tuned.

### 2.1 The faithful mechanism — exact equations (read from the primary sources)

**(a) Mikulasch-Priesemann PNAS 2021 (arXiv 2010.12395) — "local dendritic balance" (the faithful spiking decorrelator).** A network of neurons j with feedforward weights `F` (excitatory) and recurrent weights `W` (inhibitory). The decisive move is to split each neuron into **per-input dendritic compartments** indexed by i, each receiving ONE feedforward input plus the recurrent inhibition that targets *that compartment*:

```
(point-neuron, the FAILURE baseline)   u_j(t) = Σ_i F_ji x_i(t)  +  Σ_k W_jk z_k(t)        [their Eq. 3]
(dendritic, the ESCAPE)                u_j(t) = Σ_i u_ji(t),   u_ji(t) = F_ji x_i(t) + Σ_k W_jki z_k(t)   [Eq. 7]
spike:   p_spike(u_j) = σ((u_j − T_j)/Δu)                                                  [Eq. 2]
```

Here `z_k(t)` is neuron k's spike trace; `W_jki` is the inhibitory weight from neuron k onto **compartment i of neuron j** (the index `jki` is the cross-neuron, per-input lateral — this is the off-diagonal carrier). At equilibrium the recurrent inhibition cancels the *predicted* part of each compartment's input, so the compartment potential becomes the **local prediction error**:

```
u_ji  ∝  F_ji (x_i − x̂_i)         where x̂_i = the population's prediction of input i
```

The two **local** learning rules:

```
recurrent (inhibitory, anti-Hebbian, "balance"):   ΔW_jki ∝ − z_k · u_ji      [Eq. 8]
feedforward (excitatory, error-gated):             ΔF_ji  ∝ (1/F_ji) · z_j · u_ji   [Eq. 9, "learning by errors"]
```

`ΔW_jki ∝ −z_k·u_ji` is anti-Hebbian: when neuron k fires and compartment i is still depolarized, strengthen the inhibition until the compartment is balanced (`u_ji → 0` for the *predictable* component). `ΔF_ji ∝ (1/F_ji)·z_j·u_ji` is the **escape** — because `u_ji ∝ F_ji(x_i − x̂_i)`, the feedforward weight learns on `z_j · (x_i − x̂_i)`, i.e. **on the residual error, NOT on the raw input** (contrast the Hebbian/Oja rule `ΔF_ji ∝ z_j·x_i`, their Eq. 6). The paper's own load-bearing sentences:

> "When the activity of neurons is correlated, Hebbian-like learning adapts the feedforward weights of these neurons into a similar direction. This even further strengthens the correlations between neurons — a vicious cycle."
> "In contrast … learning by errors selectively weakens connections to overrepresented inputs and thereby helps to reduce the correlations between coding neurons."

That "vicious cycle" is *exactly* the project's W-collapse (correlated → same W → more correlated → rank 3–4). The result is a **decorrelated, non-redundant (efficient) code**: "Learning recurrent weights decorrelates neural responses; learning feedforward weights makes neural responses more specific." It is robust even under inhibitory transmission delays where pairwise Hebbian collapses ("all neurons learn the same feedforward weights"). **This is a published SPIKING realization** (their network is stochastic-LIF), which matters for the eventual bridge step.

**(b) The Lipshutz-Golkar-Chklovskii normative framework (arXiv 2302.10051) — the SAME escape, derived top-down, with the rank made explicit.** This paper derives multi-compartment-neuron + interneuron circuits from a normative similarity-matching objective via a minimax Lagrangian. Its update rules (verbatim):

```
feedforward:   W ← W + 2η (ζ_t ξ_tᵀ − W B_t)
lateral:       M ← M + (η/τ) (ζ_t ζ_tᵀ − M)
output settle: z_t ← z_t + γ (W ξ_t − M z_t)     ⇒  z_t = M⁻¹ W ξ_t   (full recurrent equilibrium)
```

The lateral rule `M ← M + (η/τ)(zzᵀ − M)` is **structurally identical** to the project's `learn_simmatch` lateral (`dM = yyᵀ − M`, `_phaseB`/`fair_test`) — so **the lateral is NOT the differentiator.** The differentiator is the **feedforward rule rewritten as a dendritic prediction error**:

```
W_y ← W_y + 2η ( a_t − [M − I_k] z_t ) y_tᵀ        (a_t = distal/feedforward current; the bracket = recurrent prediction)
```

> "The difference between the distal currents a_t and the recurrent lateral feedback −[M−I]z_t [is interpreted] as the **calcium plateau potential that drives non-Hebbian plasticity in the proximal synapses**."

i.e. the synaptic change is gated by **(input − recurrent prediction)**, NOT by pre×post — the dendritic compartment computes the prediction error and a Larkum-style Ca²⁺ plateau gates the proximal plasticity. **Rank control is explicit:** "the rank of the output is set by the dimension k of the eigenvector subspace being extracted … the number of interneurons mediating lateral connections directly controls the dimensionality of the learned representation." So k (≈8) is a designed quantity, not an emergent collapse.

### 2.2 WHY it escapes BOTH failure modes (concrete, honest)

The prompt asks whether the key is (a) input-specific degrees of freedom, (b) the dendritic nonlinearity, (c) a two-error-stream / timescale separation, or (d) something else. The honest, source-grounded answer: **(a) + the error-gating it enables are the load-bearing pieces; (c) helps convergence; (b) is NOT required for THIS problem.**

- **W-collapse — escaped by the error-gated feedforward rule (the per-input degrees of freedom that make it local).** The somatic SM's Oja rule `ΔW ∝ yxᵀ − y²W` learns on the *raw* (correlated) input → the vicious cycle → rank 3–4. The dendritic/normative rule learns on `y·(x − x̂)ᵀ` — the **residual after the recurrent inhibition has cancelled the predictable common mode in each compartment**. Once the common mode is removed *before* the feedforward update, correlated units no longer pull each other's weights together, so W does not collapse. The per-input compartment is what makes `(x − x̂)` available *locally* (the soma sums everything to one error and cannot separate which input to balance — the literal Mikulasch-Priesemann point-neuron limit). **This is the deep reason O1 (the falsified low-rank somatic lateral) failed and this does not: O1 kept the Oja-style single-W projection and merely bottlenecked the *output* dimension k, which forced the *same* collapse onto fewer axes (rank-1 at k=8). The dendritic mechanism does NOT bottleneck the output; it changes WHAT the feedforward weight learns on.**

- **Over-whitening — escaped by (i) a bounded interneuron pool whose size = the rank, and (ii) a shaped target.** The Pehlevan-Chklovskii primary source states plainly that pure whitening "decorrelates the outputs and equalizes their variances" and "amplifies noise in low-variance directions" — the project's full-ZCA collapse (−0.012) and fixed-W over-whitening (rank-44). The fix is **never to whiten the full covariance**: keep only ~k informative directions (rank control above), and target a *shaped* second-order statistic rather than the identity. Lipshutz-Simoncelli 2024 (NeurIPS) makes this controllable via an **optimal-transport objective** that "dynamically adjust[s] both the synaptic connections between neurons as well as the interneuron activation functions" to reach an *arbitrary* target covariance — so the noise subspace can be left alone (partial whitening) instead of equalized.

- **(c) Timescale separation HELPS convergence but is not the escape.** The normative settle `z = M⁻¹Wξ` is the FULL recurrent equilibrium; the project's bridge `graded_lateral` does only a *single* lateral step (the documented non-convergence, CYCLE 74–78) and `learn_simmatch` does a finite `settle_steps`. A fast-lateral / slow-feedforward separation (Duong NeurIPS 2023; the project's CYCLE-81 timescale probe) is the regime in which the lateral keeps up with the feedforward — necessary for the *online* version to track the equilibrium, but it does not by itself change the rank (CYCLE 81 confirmed timescale alone plateaus at rank 3–4 *on the somatic Oja network*, because the missing piece there was the error-gating, not the timescale).

- **(b) The dendritic NONLINEARITY (NMDA branch products) is NOT required for this problem.** A branch-product dendrite computes `xᵢ·xⱼ` for *co-located* synapses only — a sparse, pre-clustered nonlinearity, not the full off-diagonal `Σ` (verified against the eLife-2024 / Ujfalussy-Lengyel feature-binding literature in the prior doc). The decorrelation here is **linear** (it is whitening of a Gaussian-ish second-order statistic); the "dendrite" needed is the *compartmental separation of inputs* (so the per-input balance is local) plus a Ca²⁺-plateau *gate* on plasticity — NOT a supralinear product nonlinearity. This is good news for buildability: the de-risk does not need NMDA plateaus, only per-input balance + error-gated plasticity.

**Honest residual risk (the load-bearing uncertainty).** The literature **clearly establishes the mechanism in theory and in rate/spiking simulations on standard datasets** (natural images, bars tasks, Gaussian sources) — it does decorrelate, it is local, it does not collapse, and the rank is controllable. What the literature does **NOT** establish is that it reaches host on *this specific moderate-SNR text corpus*, where even the offline optimum caps at +0.44 and the off-diagonal residual (+0.31→+0.44) is small. The risk is therefore **not** "does the mechanism escape the collapse" (it provably does, by construction) but "**is the +0.13 off-diagonal residual on the real corpus large/clean enough that a learned online realization recovers most of it before noise dominates**." That is precisely what the cheap de-risk (§3) measures, and the BOUNDARY/NEGATIVE outcomes are pre-registered for exactly this uncertainty.

### 2.3 The single cleanest buildable form (RECOMMENDED for the de-risk first) — Duong-Lipshutz fixed-frame + plastic-gains whitening

Among the three faithful realizations, **the Duong-Lipshutz-Chklovskii-Simoncelli adaptive-whitening circuit (ICML 2023, arXiv 2301.11955) is the cleanest to code and the most structurally immune to both failure modes**, because it moves ALL the learning into bounded interneuron **gains** with a **FIXED** frame — so there is no feedforward W to collapse, and the rank is the interneuron count by construction. Exact model (verbatim):

```
objective:        min_y ⟨‖x − y‖²⟩  s.t.  ⟨(wᵢᵀ y)²⟩ = 1  for i=1..K      [their Eq. 5]
                  (K marginal-variance constraints along a FIXED frame {w_i}; K ≥ N(N+1)/2 ⇒ FULL whitening,
                   K < that ⇒ controlled PARTIAL/low-rank whitening — the over-whitening escape)
primary output (closed-form equilibrium):   ȳ = [ I_N + W·diag(g)·Wᵀ ]⁻¹ x        [Eq. 8]
interneuron input:                          z̄ = Wᵀ ȳ
GAIN update (multiplicative, the ONLY learned quantity):   g ← g + η ( z̄^{⊙2} − 1 )      [Eq. 10]
non-negative variant (ill-conditioned data):               g ← ⌊ g + η ( z̄^{⊙2} − 1 ) ⌋   (half-wave rectified)  [Eq. 14]
factorization it converges to:              I_N + W·diag(g)·Wᵀ  →  C_xx^{1/2}
```

- **W is FIXED (random frame) → cannot collapse.** The only plastic quantity is the gain vector `g`. The W-collapse is structurally impossible because there is no feedforward weight that adapts toward the principal subspace.
- **Rank = K interneurons → over-whitening is structurally controlled.** K marginal constraints whiten exactly a K-dimensional subspace; choosing K ≈ 8 (or a small overcomplete multiple) targets the informative subspace and leaves the noise subspace untouched. "When K < K_N, global whitening is not guaranteed" — that is the FEATURE here, not a bug: it is the low-rank target.
- **`Δgᵢ ∝ z̄ᵢ² − 1` is purely local** (each interneuron's gain depends only on the variance of its own input vs 1) and **non-Hebbian** (variance-based, not pre×post). It maps DIRECTLY onto the project's existing divisive-gain machinery (`cp_dendritic_source_activity` / the D1 per-hub gain) — the gain is the project's known operation, just driven by `z̄ᵢ²−1` (an interneuron's own output variance) instead of a hub's mean drive.
- **Rate-based** (no spiking realization in the paper) — so it is a *rule* to port, exactly like O3. For the numpy de-risk that is irrelevant (numpy is rate); for the eventual bridge it is the same spiking-realization risk the whole arc carries, but Mikulasch's PNAS spiking network (§2.1a) is the spiking fallback for the *faithful* form.

**Why this first, then the faithful form if needed:** the de-risk's job is to answer "is the off-diagonal residual recoverable online at all on the real corpus." The fixed-frame+gains circuit answers that with the *fewest moving parts and the strongest structural guarantees* (no collapse, controlled rank) — if even *it* cannot beat +0.35 toward +0.49 on the real corpus, the residual is intrinsically marginal for any local mechanism (ship the flat cortex). If it DOES, the faithful Mikulasch dendritic-balance form (§2.1a, which additionally *learns* the frame online and has a spiking realization) becomes the on-substrate build target with a confirmed signal. **The Mikulasch mechanism is the more-faithful, plastic-frame, spiking version of the same escape; the Duong gains-circuit is the cleanest de-risk instrument.** Both are in the spec below as arms.

---

## 3. THE CHEAP-FIRST NUMPY DE-RISK SPEC (afternoon-scale, NO `sim/` edits)

**The single open question:** can an **online, local, error-gated / fixed-frame-gain** decorrelator reach the off-diagonal ceiling on the REAL corpus — beating BOTH the diagonal-only ~+0.22–0.31 AND the somatic similarity-matching ~+0.35 — toward the offline rank-8 ZCA +0.49 — *in numpy*, with eff-rank recovering toward ~8 (not collapsing to 1–4, not exploding to 44)? A GO localizes the remaining risk to the bridge spiking realization only; a NEGATIVE is a clean, citable result that the off-diagonal on this corpus is intrinsically marginal for any local mechanism → ship the flat 2,048-concept curated cortex.

This reuses the existing harness verbatim (`build_real_corpus`, `ppmi_matrix`, `learn_simmatch`, `_cos_sim`, `_pearson_vs_Strue`, `heldout_generalization`, `effective_rank`, `perhub_residual`, the `zca` from `_phaseB_offdiagonal_derisk`); the ONLY new code is the two off-diagonal arms (~60–80 lines).

### 3.1 The arms (all on the SAME PPMI-encoded real corpus; 3 seeds 42/43/44; the contrast IS the result)

| arm | what it is | role / pre-registered expectation |
|---|---|---|
| **HOST** | PPMI + truncated-SVD cosine (`ppmi_svd_sim`+`score`) | the data-carries-it ceiling **+0.442** (gate: ≥ +0.40) |
| **ZCA_rank8** (offline target) | low-rank ZCA, keep top-8 directions (`zca(Xc, rank=8)`) | the achievable off-diagonal ceiling **≈ +0.49** (proves the off-diagonal is real + reachable offline) |
| **SM_somatic** (must-fall-short control) | `learn_simmatch` at k=64 (Oja W + anti-Hebbian M) | the somatic single-lateral network — **must plateau ~+0.35**, eff-rank 3–4 (the W-collapse) |
| **DIAG_gain** (must-fall-short control) | per-hub divisive gain `perhub_residual(C, g, σ)` | the diagonal ceiling — **must stay ~+0.22–0.31** (D2-as-designed) |
| **GAINS_whiten** (MECHANISM A — recommended) | fixed random frame W (N×K, K≈8–16) + plastic gains; eq. below | the test arm — **beat +0.35 toward ZCA_rank8**, eff-rank ≈ K |
| **DEND_balance** (MECHANISM B — faithful) | error-gated feedforward (`ΔW ∝ y·(x−x̂)ᵀ`) + anti-Hebbian lateral, full settle; eq. below | the faithful arm — same gate; tests whether the *plastic-frame* form also reaches it |

**Exact update equations, directly codeable in numpy.** `Xn` = PPMI rows L2-normalized per concept (the `learn_simmatch` convention); N = n_hub (or a PCA-pre-reduced n_hub for speed); concepts presented one at a time over epochs.

**MECHANISM A — fixed-frame + plastic-gains (Duong-Lipshutz; the recommended first arm):**
```python
# fixed random frame: K interneurons (K controls the rank ~ 8..16; overcomplete small multiple is fine)
W = rng.standard_normal((H, K)) / np.sqrt(H)        # FIXED, never updated  (H hubs x K interneurons)
g = np.zeros(K)                                       # plastic gains (the ONLY learned quantity)
for ep in range(epochs):
    for i in shuffled(concepts):
        x  = Xn[i]                                    # (H,)
        # closed-form whitened output (eq. 8): y = (I + W diag(g) Wᵀ)^{-1} x   -- H x H solve, H small after PCA
        A  = np.eye(H) + (W * g) @ W.T                # (H,H);  (W*g) broadcasts g over columns
        y  = np.linalg.solve(A, x)                    # the whitened concept code in hub space
        z  = W.T @ y                                  # (K,) interneuron inputs
        g  = np.maximum(g + eta_g * (z**2 - 1.0), 0.0)# eq.14 non-negative gain update (LOCAL, per-interneuron)
# read-out codes = the whitened y per concept; structure = Pearson(cos(Y), S_true)
```
*Notes:* solving an H×H system per sample is the only cost; with H=500 it is fine for 64 concepts × ~200 epochs (minutes). For speed, optionally PCA-reduce the PPMI hub dimension to ~64 first (whiten in the reduced space) — that is standard and does not change the mechanism (report it). The rank is **K** (number of interneurons); sweep K ∈ {8, 12, 16, 24}. The over-whitening guard is automatic (K ≪ N(N+1)/2).

**MECHANISM B — error-gated dendritic balance (Mikulasch / normative; the faithful arm):** keep a plastic frame `W_ff (k×H)` and lateral `M (k×k)`, but replace `learn_simmatch`'s **Oja** feedforward with the **error-gated** feedforward (the single decisive change), and use the FULL recurrent settle:
```python
W_ff = rng.standard_normal((k, H)) * 0.1
M    = np.zeros((k, k))
for ep in range(epochs):
    for i in shuffled(concepts):
        x   = Xn[i]                                   # (H,)
        a   = W_ff @ x                                # distal/feedforward current (k,)
        # FULL recurrent equilibrium  z = M^{-1} W_ff x  (normative eq.) -- solve, do NOT one-step:
        z   = np.linalg.solve(np.eye(k) + M, a)       # settled output (k,);  (I+M) z = a  is the stable settle
        # --- the ESCAPE: feedforward learns on the RESIDUAL (x - x_hat), x_hat = back-projected prediction ---
        x_hat = W_ff.T @ z                            # (H,) the population's reconstruction/prediction of x
        W_ff += eta_w * np.outer(z, (x - x_hat))      # ERROR-GATED (normative: (a-[M-I]z)yᵀ form); NOT Oja yxᵀ
        dM   = np.outer(z, z) - M                     # anti-Hebbian lateral, fixed-point (same as learn_simmatch)
        np.fill_diagonal(dM, 0.0)
        M   += eta_m * dM
# read-out codes = settled z per concept; structure = Pearson(cos(Z), S_true)
```
*The one load-bearing line vs `learn_simmatch`:* the feedforward update is `np.outer(z, (x − x_hat))` (residual/error-gated) instead of `np.outer(y, x) − (y**2)[:,None]*W_ff` (Oja). This IS the Mikulasch "learning by errors" escape, in rate form. Sweep k ∈ {8, 16, 32, 64}; the prediction is that the *error-gated* k=64 does NOT collapse to rank 3–4 the way the Oja k=64 does (eff-rank should hold near k, structure should exceed +0.35). The `(I+M)z = a` settle (a true solve, not a single step) is the second decisive difference from the bridge's `graded_lateral`.

### 3.2 Gates (multi-seed 42/43/44; reuse the project metrics exactly)
- **host_carries** — HOST ≥ +0.40 AND ZCA_rank8 ≥ +0.45 (the data + the off-diagonal ceiling are real).
- **somatic_falls_short** — SM_somatic ≤ +0.38, eff-rank(SM_somatic) ≤ 5 (reproduces the W-collapse; the contrast baseline).
- **diagonal_falls_short** — DIAG_gain ≤ +0.32 (reproduces the diagonal plateau; the second contrast baseline).
- **mechanism_beats_collapse (the single most important gate)** — `GAINS_whiten` (or `DEND_balance`) **peak** ≥ +0.40 AND ≥ SM_somatic + 0.06 toward ZCA_rank8, **AND eff-rank ∈ [6, 16]** (recovers ~8; NOT collapsed to 1–4, NOT exploded to >30). *The eff-rank band is co-equal with the Pearson: a +0.40 with eff-rank 3 would be the collapse sneaking through; a +0.40 with eff-rank 44 would be over-whitening luck. The mechanism is only validated if it hits the right rank.*
- **generalizes** — `heldout_generalization` above chance for the mechanism arm, at the diagonal-plateau level for the controls.

### 3.3 The full anti-cheat battery (mandatory; mirrors the deep-research standard)
- **point-neuron / somatic-lateral MUST fall short** — SM_somatic plateaus ≤ +0.38 AND DIAG_gain ≤ +0.32 WHILE the mechanism exceeds +0.40. *The contrast IS the result*; if the somatic lateral does NOT fall short, the toy doesn't reproduce the established collapse → re-tune before trusting any GO.
- **host-carries** — PPMI+SVD = +0.44 AND offline rank-8 ZCA = +0.49 on the same `C` (else corpus issue).
- **permuted-similarity collapses** — shuffle which concepts are same-category → `S_perm` → mechanism Pearson AND generalization → ~0 (structure not an artifact).
- **lesion** — freeze the gains to a constant (Mechanism A) / set the feedforward back to Oja, or freeze M to identity (Mechanism B) → the off-diagonal lift collapses back to the diagonal/somatic plateau → proves the lift rides the *error-gated lateral*, not a code property.
- **over-whitening guard (the project-specific one)** — report eff-rank for every arm; the FULL-rank ZCA must show the collapse (−0.012), the mechanism must hold eff-rank ≈ 8 (this is folded into `mechanism_beats_collapse` but report it explicitly per-arm).
- **input-lesion** — run the mechanism on raw (non-PPMI) input → must trail the PPMI version (the diagonal front-end is load-bearing; the off-diagonal stacks ON it).
- **S_true a-priori** — assert `S_true` is the constructed taxonomy block, never corpus-derived (the harness already enforces this).
- **multi-seed 42/43/44** — all gates must hold on all three (the standing rule; 6-seed only if a variable effect needs it — here the gates are pass/fail per seed).

### 3.4 Outcomes (three-state, pre-registered)
- **GO** — `GAINS_whiten` or `DEND_balance` beats +0.40 toward +0.49 on real, multi-seed, eff-rank ≈ 8, controls clean. ⇒ the off-diagonal IS reachable by an online local error-gated / fixed-frame-gain circuit; **the remaining risk is ONLY the bridge spiking realization** (the known `graded_lateral`-convergence piece, now with a confirmed numpy target + the Mikulasch spiking blueprint) → that becomes the sharp, bounded build target, and the months-scale dendritic cortex is GREENLIT with a measured signal.
- **BOUNDARY** — beats the diagonal + somatic controls but falls short of host (e.g. +0.40, not +0.46). ⇒ the right family but an online-convergence gap; the faithful plastic-frame Mikulasch form (Mechanism B, with the spiking realization) is warranted WITH A SHARP TARGET (the measured shortfall), and a fuller-settle / better-conditioned / longer-training arm is the next cheap step before the bridge.
- **NEGATIVE** — even the structurally-collapse-immune fixed-frame-gain circuit plateaus at ~+0.35 in numpy. ⇒ a clean, citable result that the off-diagonal on this moderate-SNR corpus is intrinsically marginal for *any* local mechanism (the +0.13 residual simply isn't cleanly separable online) → **ship the flat 2,048-concept curated cortex** as the conversational product; reserve the full dendritic build only for the artificial-life goal, eyes open it may also plateau on real experience.

### 3.5 Cost
Afternoon-scale, CPU/numpy. The off-diagonal ZCA instrument + `learn_simmatch` + the diagonal control + all metrics/anti-cheats already exist; the only new code is the two off-diagonal arms (~60–80 lines reusing the harness). NO GPU, NO `sim/` edits. Mechanism A's H×H solve per sample is the only nontrivial cost; PCA-pre-reduction to ~64 keeps it to minutes.

---

## 4. RANKED FALLBACKS (if the top mechanism's de-risk is NEGATIVE/BOUNDARY)

1. **Mechanism B with online frame learning + longer settle + fast-lateral/slow-feedforward timescale (Duong NeurIPS 2023 two-timescale + Mikulasch spiking).** If Mechanism A (fixed frame) plateaus because the *fixed* random frame doesn't align with the informative subspace, the faithful plastic-frame form that *learns* the frame on the error-gated rule may select the right 8 directions. Cheap-ish (still numpy, +a frame-learning loop). This is also the on-substrate target if Mechanism A is a GO — so it is built either way; here it is the fallback-from-BOUNDARY.
2. **Shaped (non-identity / optimal-transport) target — Lipshutz-Simoncelli 2024.** If the failure is over-whitening of the noise subspace (eff-rank explodes, structure drops), replace the identity whitening target with a *shaped* target covariance (whiten only the top-8 eigendirections to unit variance, leave the rest at their input variance). The NeurIPS-2024 OT objective makes this a drop-in change to the gain/weight rules. Reaches a *partial* whitening that should match the offline rank-8 ZCA exactly. Cheap (numpy; modify the target in Mechanism A).
3. **PCA-pre-reduce then whiten (the offline-front-end concession).** If no fully-online arm reaches host, an honest *hybrid* — an offline/slow PCA front-end (the "select the top-8 subspace" step, which the project already accepts as the slowly-learned cortical basis) followed by an online local gain-whitening of that subspace — would reach host while keeping the *decorrelation* online and local. This concedes that subspace *selection* may be a slow/offline operation (biologically: a developmentally-set basis) while *decorrelation* is the online local piece. Weakest from the "fully learned from experience" purity standpoint, but it is the pragmatic path that would still ship a real-corpus cortex; flag it explicitly as a partial concession, not a clean win.

---

## 5. REUSABLE project machinery + honest risk assessment

**Reusable for the numpy de-risk (file-cited):**
- **The harness, verbatim:** `research/runners/learned_graded_cortex_fair_test.py` — `build_real_corpus(seed, n_hub)` (real TinyStories concept×hub counts), `ppmi_matrix` (the diagonal PPMI front-end), `learn_simmatch` (**the somatic must-fall-short control**), `pca_lowrank_sim` (offline reference). `research/runners/dendritic_d1_learn_graded_structure_derisk.py` — `_cos_sim`, `_pearson_vs_Strue`, `heldout_generalization`, `effective_rank`, `perhub_residual` (**the diagonal must-fall-short control**), `learn_perhub_gains`. `research/runners/_phaseB_offdiagonal_derisk.py` — `zca(Xc, rank=k)` (**the offline rank-8 ZCA target +0.49** + the full-ZCA over-whitening reference −0.012). `research/runners/option_c_paradigmatic_host_precheck.py` — `ppmi_svd_sim`, `score` (the host instrument + the a-priori `S_true`). **All metrics + anti-cheats already written and validated — the new code is only the two off-diagonal arms.**

**Reusable for the eventual on-substrate (bridge) build (if the de-risk is GO):**
- **`sim/dendritic_neuron.py`** — a two-compartment Larkum/Guerguiev-Lillicrap-Richards neuron (basal + apical + soma; fixed-random apical feedback; BAC integration). The apical-as-prediction / basal-as-feedforward split is exactly the normative-framework compartment structure; the Ca²⁺-plateau-gates-plasticity reading maps onto `effective_threshold`. **The compartment scaffold exists** (the prior "Dendritic-fair" arc VOIDed on a DIFFERENT question — credit assignment / MNIST — per `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-triangulation.md`; the VOID was an optimizer bug in `dendritic_mlp.py`, NOT a flaw in the compartment model — so the neuron module itself is sound to reuse).
- **`sim/dendritic_plasticity.py`** — `urbanczik_senn_update`: the local somato-dendritic mismatch rule `Δw ∝ apical-gated·(soma − φ(v_basal))·pre`. The `mismatch = soma − φ(v_basal)` self-prediction-error branch is structurally the **error-gated feedforward** rule of Mechanism B (it learns on a *mismatch*, not pre×post). Directly reusable as the on-substrate feedforward rule.
- **`graded_lateral` on the bridge** (`sim/bridge.py:309–319, 1740–1809`) — the on-bridge anti-Hebbian lateral `M` (`ΔM ∝ ⟨aaᵀ⟩ − I − λM`). This is the lateral half of both mechanisms, already wired (guarded default-OFF, byte-identical when off). Its known gap (one-step, not the full `(I+M)z=a` settle) is EXACTLY what the de-risk's full-solve tests — so a GO tells you precisely what to fix on the bridge (the settle + the feedforward error-gating).
- **The shipped diagonal front-end** (`cfg.enable_input_mean_adapt` subtractive adaptation; the D1 per-hub divisive gain `cp_dendritic_source_activity`) — the off-diagonal mechanism **stacks ON TOP of** this (PPMI/diagonal first, then the off-diagonal lateral). For Mechanism A specifically, the existing divisive-gain machinery is the *closest* on-substrate primitive to the interneuron-gain `Δgᵢ ∝ z̄ᵢ²−1` (same divisive operation, different driver).
- **Catalog G.02 (active dendrites)** — confirms compartments are MISSING ("one of the largest abstractions … ~10× compute/neuron"); this build IS the G.02 addition, scoped to the *minimal* compartment (per-input balance + error-gated plasticity), NOT the full NMDA-plateau dendrite (§2.2b — not needed for this problem).

**Honest risk assessment.** This is the entry to a months-scale, owner-approved build, and the de-risk gates it. Three honest risk layers, in order:
1. **The mechanism is real and the escape is provable.** Unlike O1 (falsified) and D2-diagonal (mathematically capped), the dendritic/interneuron-whitening family DOES escape both the W-collapse (error-gating breaks the Oja vicious cycle) and over-whitening (bounded rank + shaped target) — by construction, corroborated by primary sources and standard-dataset simulations. This is the strongest-founded option the arc has reached for the off-diagonal. **Confidence: high on the mechanism, in general.**
2. **The corpus-specific risk is the real unknown.** Whether the small (+0.13) off-diagonal residual on *this* moderate-SNR text corpus is recoverable online is genuinely open — the offline optimum itself is only +0.44, and the project has repeatedly found this corpus's structure to be weak/overlapping (CYCLE 80–85: k-means ceiling +0.217, not block-separable). The de-risk measures exactly this; a NEGATIVE here is a *data*-marginality result, not a mechanism failure, and is itself the deliverable (ship the flat cortex). **Confidence: deliberately uncertain — that is why it is a de-risk, not a claim.**
3. **The bridge spiking-convergence risk is deferred but real.** Even a numpy GO leaves the on-substrate realization (the project's recurring `graded_lateral`-doesn't-converge-in-the-streaming-window problem) as the hard engineering piece — but a GO converts it from "unknown if reachable at all" to "reachable offline + in numpy, now realize it in spikes," with Mikulasch's published spiking network as the blueprint. **Confidence: this is the months-scale work the owner is approving; the de-risk's job is to ensure it is built on a confirmed signal, not a hope.**

**One-line recommendation:** run the §3 numpy de-risk (Mechanism A fixed-frame+gains first, Mechanism B faithful-error-gated as the second arm) on `build_real_corpus` before any bridge work — it costs an afternoon, has the strongest structural guarantees of any option the arc has produced (no collapse, controlled rank, both by construction), and resolves the whole fork: GO ⇒ greenlight the dendritic cortex on a measured signal (risk = bridge spiking only); BOUNDARY ⇒ the faithful plastic-frame form with a sharp target; NEGATIVE ⇒ a clean citable "the real corpus's off-diagonal is intrinsically marginal for any local mechanism" → ship the flat 2,048-concept curated cortex.

---

## Trust-but-verify flags (load-bearing claims I am NOT 100% certain of)
1. **The error-gated feedforward rule (Mechanism B's `outer(z, x−x̂)`) escapes the rank-3–4 collapse at k=64 on the real corpus.** I argue it should (it learns on the residual, breaking the Oja vicious cycle — Mikulasch's own claim) but the *online rate* realization may under-converge vs the spiking/closed-form theory. This is precisely what the de-risk's `mechanism_beats_collapse` + eff-rank-band gate tests; pre-registered as possibly BOUNDARY. **Medium-high confidence on the mechanism, deliberately untested on this corpus.**
2. **Mechanism A's fixed random frame W aligns well enough with the informative subspace.** A FIXED frame avoids collapse but a *random* frame may not span the top-8 informative directions as efficiently as a learned one → could under-reach host even though it cannot collapse. Mitigation: sweep K (overcomplete) + the Mechanism-B plastic-frame arm + the PCA-pre-reduce option. **Medium confidence — this is the most likely reason A could BOUNDARY rather than GO.**
3. **The Duong/normative equations as transcribed.** Read from the ICML-2023 ar5iv HTML and the 2302.10051 ar5iv HTML (the PDF streams did not parse); the eq. numbers (Duong Eq. 5/8/10/14; normative `W←W+2η(ζξᵀ−WB)`, `M←M+(η/τ)(ζζᵀ−M)`, `z=M⁻¹Wξ`) are quoted from those renders. High confidence on the *structure* (corroborated across two independent papers + Mikulasch); a reader building it should re-confirm the exact constants/normalizations against the published PDFs (the de-risk is self-checking — if a transcription error makes the rule wrong, the arm will simply fail the gate, not silently pass).
4. **The "dendritic nonlinearity is NOT required" claim (§2.2b).** High confidence — the decorrelation is linear (second-order whitening), and the prior doc verified the branch-product literature; the compartment's role here is per-input *balance* + a plasticity *gate*, not a supralinear product. But if the real corpus's off-diagonal has genuinely nonlinear (higher-order) structure the linear whitening misses, that would show as a BOUNDARY that none of the linear arms close — in which case the NMDA-plateau product dendrite (a *different*, harder build) re-enters as a fallback-4.

## Sources

### Current literature (consulted this pass; primary-source-verified)
- **Mikulasch, Rudelt, Priesemann**, *Local dendritic balance enables learning of efficient representations in networks of spiking neurons*, **PNAS** 2021 (arXiv 2010.12395; PMC8685685) — the faithful spiking dendritic-balance decorrelator. Eq. 3 (point-neuron baseline), Eq. 7 (dendritic compartments `u_ji = F_ji x_i + Σ_k W_jki z_k`), Eq. 8 (anti-Hebbian recurrent `ΔW_jki ∝ −z_k u_ji`), Eq. 9 (**error-gated feedforward `ΔF_ji ∝ (1/F_ji) z_j u_ji`, "learning by errors"**), the "vicious cycle" of Hebbian-on-correlated-input. Read from PMC full text.
- **Lipshutz, Golkar, Chklovskii** et al., *Normative framework for deriving neural networks with multi-compartmental neurons and non-Hebbian plasticity*, arXiv 2302.10051 — the top-down derivation: feedforward `W←W+2η(ζξᵀ−WB)`, lateral `M←M+(η/τ)(ζζᵀ−M)`, full settle `z=M⁻¹Wξ`, the **feedforward-as-dendritic-prediction-error** `W_y←W_y+2η(a−[M−I]z)yᵀ` gated by a **Ca²⁺ plateau**, rank = k subspace dimension = interneuron count. Read from ar5iv HTML full text.
- **Duong, Lipshutz, Chklovskii, Simoncelli**, *Adaptive whitening in neural populations with gain-modulating interneurons*, **ICML** 2023 (arXiv 2301.11955) — **the cleanest buildable form**: fixed frame W + plastic gains, `ȳ=[I+W diag(g)Wᵀ]⁻¹x` (Eq. 8), gain rule `g←g+η(z̄^{⊙2}−1)` (Eq. 10) / non-neg (Eq. 14), `I+W diag(g)Wᵀ→C_xx^{1/2}`, K interneurons = K marginal constraints = rank control (K≥N(N+1)/2 ⇒ full; K< ⇒ partial). Rate-based. Read from ar5iv HTML full text.
- **Duong, Nguyen, Lipshutz, Chklovskii, Simoncelli**, *Adaptive whitening with fast gain modulation and slow synaptic plasticity*, **NeurIPS** 2023 — the two-timescale version (fast gains + slow weights); the fast-lateral/slow-feedforward regime (maps to the project's CYCLE-81 timescale separation). (PDF did not parse to text; cited from the ICML-2023 lineage + abstract.)
- **Lipshutz & Simoncelli**, *Shaping the distribution of neural responses with interneurons in a recurrent circuit model*, **NeurIPS** 2024 (arXiv 2405.17745) — the **over-whitening fix**: an *optimal-transport* objective that adjusts synaptic connections AND interneuron activation functions to reach an *arbitrary* (non-identity / partial / low-rank) target second-order statistic. (Abstract verified; the partial-whitening control is the load-bearing claim — fallback-2.)
- **Mikulasch, Rudelt, Spitzner, Priesemann**, *Where is the error? Hierarchical predictive coding through dendritic error computation*, **Trends in Neurosciences** 2023 — the conceptual frame: dendritic compartments compute local prediction errors, soma integrates and spikes; "an efficient implementation of hPC with spiking neurons … errors computed in the dendritic membrane potentials are integrated at the soma." (Cell full text 403-blocked; verified via search abstract + the PNAS-2021 / arXiv-2205.05303 lineage.)
- **Pehlevan & Chklovskii**, *Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening*, arXiv 1511.09468 — the **over-whitening result** ("amplifies noise in low-variance directions"), already primary-source-verified in the prior off-diagonal deep-research doc.

### Project record (re-verified this pass, file/line cited)
- The decomposition + numbers + W-collapse/over-whitening characterization: `research/findings/2026-06-15-natural-learning-over-time-CONVERGES-on-locality-wall.md` (CYCLE 81 eff-rank 3–4 plateau; CYCLE 85 diagonal +0.216; CYCLE 87 O1 rank-1 collapse), `research/findings/2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md` (D2-diagonal-only proof; O1/O2/O3 scoping; the over-whitening constraint).
- The harness (reuse verbatim): `research/runners/learned_graded_cortex_fair_test.py` (`build_real_corpus`, `ppmi_matrix`, `learn_simmatch`, `pca_lowrank_sim`), `research/runners/dendritic_d1_learn_graded_structure_derisk.py` (`_cos_sim`, `_pearson_vs_Strue`, `heldout_generalization`, `effective_rank`, `perhub_residual`, `learn_perhub_gains`), `research/runners/_phaseB_offdiagonal_derisk.py` (`zca` rank-k = the +0.49 target + the −0.012 full-ZCA reference), `research/runners/_phaseB_lowrank_lateral_derisk.py` (the falsified O1 — the must-not-repeat pattern: bottlenecking the *output* k forces the collapse), `research/runners/option_c_paradigmatic_host_precheck.py` (`ppmi_svd_sim`, `score`, `S_true`).
- The on-substrate machinery (reuse if GO): `sim/dendritic_neuron.py` (two-compartment Larkum; sound — the prior VOID was the `sim/dendritic_mlp.py` optimizer on a DIFFERENT question per `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-triangulation.md`), `sim/dendritic_plasticity.py` (`urbanczik_senn_update` mismatch rule = the error-gated feedforward primitive), `sim/predictive_coding.py` (Rao-Ballard prediction-error scaffold), `sim/bridge.py:309–319, 1740–1809` (`graded_lateral` = the on-bridge anti-Hebbian lateral, one-step — the piece the de-risk's full-solve diagnoses).
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` G.02 (active dendrites — MISSING, ~10× compute/neuron), A.12 / E.05 / olfactory entries (decorrelation = lateral inhibition throughout the catalog's framing).
- Corpus verified present: `data/corpus/tinystories.txt`.
