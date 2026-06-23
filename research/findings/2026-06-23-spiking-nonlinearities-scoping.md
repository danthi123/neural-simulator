# Read-only scoping: making the consolidated generator's PARAMETER-FREE nonlinearities (softmax / GELU / LayerNorm) SPIKING on the bridge — completing fully-spiking C1 (2026-06-23)

**Mode:** read-only deep-research scoping (no edits, no experiments, no GPU, no pytest). The controller will
trust-but-verify the load-bearing claims and present.

**The gate that fired:** this is new-mechanism-class work on a fresh part of the sim (CLAUDE.md research gate (d)
"about to design a mechanism class not previously built"), and softmax is the same FAMILY as the documented
rate-code / point-neuron / divisive-normalization wall (gate (b)). So: deep-research FIRST, the fix ranked as one
option, never the default.

---

## 0. Verified context (quoted file:line)

- **C1 ACHIEVED — the full Gen-F generator's WEIGHTS run on the bridge and GENERATE.** `research/findings/raw/_genseq_loopstep3_full_genf_generate.json`:
  - `learned_matvec_params_total: 3408384` (4 blocks × [attn Q/K/V/O + MLP W1/W2] + output head), **ALL on the
    conductance-free RF complex-synapse path**, `rf_exact_max_err_over_all_matvecs: 1.87e-06` (float32, EXACT).
  - GENERATION: `greedy_token_match_rf_vs_offbridge: 1.0`; held-out `ppl_ratio_rf_over_offbridge: 1.0000`;
    `LESION(scrambled-RF-weights) logit_fid=-0.0183 << real` (the matvecs carry the computation).
  - `scope_note`: "WEIGHTS-on-RF + nonlinearities-as-faithful-reads. The fully-SPIKING nonlinearities (spiking
    softmax / spiking LayerNorm / spiking GELU) are a SEPARATE follow-on." ⇒ **exactly this doc's target.**
- **The bridge ALREADY carries two of the three needed circuits** (this is the decisive finding — they were built
  for the learned-cortex read-out and the navigation accumulator, and are byte-identical no-ops when off):
  - **DIVISIVE normalization** (Carandini-Heeger) — `sim/bridge.py:6190-6199`: for flagged neurons,
    `total_input_current_pA = ... / (sigma + gain * MEAN_over_pool(input))`. Config `enable_input_divisive_norm`,
    `input_divisive_sigma`, `input_divisive_gain` (`sim/config.py:488-492`). A reusable builder already exists:
    `build_divnorm_score_bridge` (`research/runners/_phaseC_S5_divnorm_derisk.py:115`).
  - **SUBTRACTIVE per-feature mean** (spike-frequency adaptation / point-neuron predictive coding) —
    `sim/bridge.py:6238-6250`: `adapted = raw - gain*m; m <- (1-alpha)m + alpha*raw`, masked. Config
    `enable_input_mean_adapt`, `input_mean_adapt_alpha/gain` (`sim/config.py:469-477`).
  - Together these are **exactly LayerNorm's two arms** (centre then scale-by-spread) and **softmax's denominator**
    (the divisive arm). Both run inside the same `_run_one_simulation_step`, BEFORE the spike threshold.
- **The graded transfer function** `a_cont = clip((v - rest)/scale, 0, 1)` — `sim/bridge.py:6144` — a saturating
  sub-threshold readout of the membrane (the retina's horizontal-cell analog). A ready-made monotone nonlinearity
  on the *same* substrate (for GELU).
- **The spiking WTA / NEF cleanup** (the project's softmax/argmax-like competition) —
  `research/runners/rf_phasor_composer.py:316` `_spiking_cleanup` (input-normalize → Izhikevich bank → integrate
  firing → argmax-over-FIRING), mirrored in `one_brain_composer.py:398` `_spiking_select`. Validated == numpy
  argmax multi-seed (`2026-06-05-composer-cleanup-NEF-GO.md`). This is a *hard* max (top-1), NOT a soft
  distribution — relevant to softmax below.
- **The read-out-norm biologization, already de-risked at ~96% of host** — `2026-06-16-biologization-sweep-conversational-pipeline.md`
  ("read-out normalization → per-hub spike-freq adaptation + per-concept feedforward inhibition, POST-f-I →
  de-risked GO — 96% of host (6 seeds) with realistic rate-coded-pool noise; both ops load-bearing") +
  `research/runners/_phaseB_biologize_readout_norm_derisk.py:48` `neural_norm` (the two subtractive arms + their
  1/√pool noise). **This IS a LayerNorm-without-affine, already validated.**

---

## 1. SOFTMAX — `softmax(QKᵀ/√d)`  (content-dependent attention weights)

**The op.** Per query row: `a_i = exp(s_i) / Σ_j exp(s_j)` over the causal key set, then `out = a @ V`. Two
ingredients: an **exponential** (amplification / contrast) and a **sum-normalization** (the divisive denominator).
The math identity the field now agrees on: *softmax = exp + a vector norm; "any normalization works as well as
softmax, the important parts are an exponential plus a vector norm"* (Hu et al., the recurrent-RNN-view of
softmax). Heeger (1992) DIVISIVE NORMALIZATION is **mathematically the same competitive renormalization** as
softmax's denominator (Towards-Data-Science synthesis; ORGaNICs explicitly "replaces the softmax in an attention
head with divisive normalization").

**Spiking realizations, ranked cheapest-first on THIS substrate:**

1. **(reuse) divisive normalization for the denominator** — the shipped `enable_input_divisive_norm` block
   (`bridge.py:6190`) computes `x / (σ + g·mean(x))` over a flagged pool. With the pool = one query's key-logit
   set, this IS softmax's sum-normalization arm (the linear-attention / `softmax≈first-order-DN` regime). **No
   `sim/` edit** — it exists. Approximate (it normalizes by the *mean*, i.e. linear-attention, not by `Σexp`).
2. **the EXPONENTIAL is the genuine residual.** The amplification (winner-sharpening) is the part DN-by-mean does
   NOT supply. Options, none yet on-substrate as a *soft* (graded, multi-key) op:
   - **f-I exponentiation** — place the key-logit pool's operating point on the *expansive* (supra-rheobase,
     accelerating) part of an Izhikevich/AdEx f-I curve so firing-rate ≈ `exp`-ish in `s`, THEN divisively
     normalize the rates. Biologically grounded (cortical f-I is expansive near threshold). Approximate; needs a
     small de-risk that the f-I shape × the divisive arm reproduces the trained softmax temperature.
   - **ORGaNICs recurrent DN** (Heeger/Rubin; arxiv 2409.18946, unconditionally stable) — a recurrent circuit
     whose fixed point IS divisive normalization with a tunable exponent; the closest *exact-soft* biological
     softmax. Heavier (a recurrent settle per attention row) and would be a new mechanism class.
3. **(the literature's actual answer — but it is RE-ARCHITECTURE, not faithful)** — Spikformer / Spike-Driven
   Self-Attention / A²OS²A all **ELIMINATE softmax**: make Q binary {0,1} and K non-negative (ReLU/spike-count) so
   `QKᵀ ≥ 0` and no softmax is needed for non-negativity (arxiv 2503.00226, 2508.07710). This is the standard
   spiking-transformer trick — *but it requires RETRAINING the attention with that constraint*; it does NOT
   realize the **already-trained Gen-F softmax weights** the C1 result depends on. It is a path to a *different*
   (re-trained) spiking generator, not a faithful spiking realization of the consolidated one.

**Verdict for softmax:** the **denominator is free** (reuse the shipped divisive-norm circuit, approximate =
linear-attention); the **exponential is the genuine residual** and is the prime point-neuron-limit suspect — a
graded, content-dependent, multi-key amplification is what a rate code does poorly (this is the same family as the
whitening / Mikulasch-Priesemann wall). The honest expectation: **softmax is APPROXIMATE on this substrate** (f-I
exp + divisive norm, or recurrent ORGaNICs DN), with a fidelity cost to be measured; an EXACT spiking softmax of
the *trained* weights is the likely boundary. The hard-WTA `_spiking_cleanup` is NOT a substitute (it collapses
the soft distribution to top-1, which would change the value-mix and the generation).

---

## 2. GELU — `x·Φ(x)` (the MLP activation)

**The op.** A smooth, signed, *unbounded*, monotone-ish scalar transfer function (≈ `x` for large +x, ≈ 0 for
large −x, a smooth dip near 0). Per-feature, NO cross-feature mixing, 0 learned params.

**Spiking realizations, cheapest-first:**

1. **graded transfer-function read (CHEAPEST)** — drive the MLP-hidden pool, read a *graded* membrane nonlinearity
   instead of the host `gelu_exact`. The shipped `a_cont = clip((v−rest)/scale,0,1)` (`bridge.py:6144`) is already
   one saturating nonlinearity on the substrate; GELU is monotone like it on `x>0` but is signed+unbounded, so a
   single clip is not GELU-shaped — it needs a calibrated read (an affine of `a_cont`, or a two-pool signed
   read). Approximate; cheapest because the graded path exists.
2. **custom spiking neuron fitted to GELU (the literature's faithful answer)** — "Precise spiking neurons" jointly
   regulate threshold / reset / membrane to fit *any* activation (Springer, *Applied Intelligence* 2025); MBE
   neurons with basis components approximate GELU/Tanh in spiking transformers (arxiv 2508.07710; note: they
   constrain GELU's input domain because LayerNorm precedes it — relevant: our LN read already bounds the input).
   Faithful-approximate; a few-component fit. This is a *new* neuron-fit mechanism but a well-trodden one.
3. **population code** — a small bank with staggered thresholds whose summed firing traces the GELU curve (the
   standard expand-and-read). More neurons, smoother fidelity.

**Verdict for GELU:** **spiking-realizable, faithful-approximate, LOW risk.** It is a fixed scalar monotone
function — exactly what a calibrated spiking transfer function / fitted neuron does well. Cheapest = the graded
`a_cont`-style read, calibrated; the fitted-neuron route is the higher-fidelity fallback. The input is already
LN-bounded, which (per the literature) is what makes the fit accurate. Expect a small, characterized fidelity cost
(not a boundary).

---

## 3. LayerNorm — `(x − μ)/√(var+ε) · w + b`

**The op.** Per token (across features): subtract the feature-mean μ, divide by the feature-std, then a *learned*
per-feature affine `w,b` (Gen-F's `ln1/ln2/lnf` — the affine is the only learned part, and it is NOT a matvec, so
it already "rides on the read" in C1).

**Spiking realization — REUSE, DIRECTLY:**
- **subtract-mean (μ)** → the shipped **subtractive per-feature mean** circuit (`enable_input_mean_adapt`,
  `bridge.py:6238`) = `x − m`, OR per-concept FEEDFORWARD INHIBITION (a global interneuron subtracting the pool
  mean). This is **already validated** as the centring arm of the read-out norm.
- **divide-by-std** → the shipped **divisive normalization** circuit (`enable_input_divisive_norm`,
  `bridge.py:6190`) = divide by `(σ + g·mean)`. (Subtlety: the bridge's divisor uses the mean of the *(already
  mean-subtracted, rectified)* drive, which is an L1/mean-absolute spread, not the exact RMS √var. Approximate but
  the right monotone divisive contrast — the read-out-norm de-risk shows this recovers the structure.)
- **affine `w,b`** → rides on the read (a fixed per-feature scale+shift), exactly as in C1 today. Zero new
  mechanism.

**Direct evidence it works:** the read-out-norm biologization de-risk (`_phaseB_biologize_readout_norm_derisk.py`)
realized precisely "subtract per-hub mean (adaptation) + subtract per-concept mean (feedforward inhibition)" — a
double-centring LayerNorm-without-the-divide — at **96% of host with rate-coded-pool noise, both arms
load-bearing** (`2026-06-16-biologization-sweep...`). LayerNorm here = that + the divisive arm + the affine on the
read.

**Verdict for LayerNorm:** **spiking-realizable by REUSE, faithful-approximate, LOWEST risk** — both arms are
shipped `sim/` circuits, the affine already rides on the read, and the double-centring half is independently
validated at 96%. The only approximation is mean-absolute-spread vs exact RMS in the divisor; characterize it,
likely small.

---

## 4. The cheapest FIRST de-risk

**Do LayerNorm first** (highest reuse, lowest risk, and it *bounds the GELU input* the others depend on). It is a
single co-resident block on the existing pieces, no `sim/` edit anticipated.

- **Probe:** take ONE Gen-F block's real LN1 input `x` (the exact float vector C1 already produces). Compute the
  host `LayerNorm(x)` (teacher). Realize the spiking LN: route `x` (as drive) through a small Izhikevich/graded
  pool flagged `input_mean_adapt=True` (centre) + `input_divisive_norm=True` (scale), read the
  (divisively-normalized, mean-subtracted) pool output, apply the learned affine `w,b` on the read. Feed that into
  the EXISTING exact-RF MLP path and measure the block-output fidelity vs the all-host-read C1 teacher (the SAME
  spearman/cosine basis as the full-block de-risk).
- **GO/NO-GO bar:** block-output **spearman/cosine ≥ 0.90** vs the host-LN C1 baseline (matching the prompt's
  ~0.9 bar and the project's 0.80 generation bar with headroom). A *generation* check (greedy distinct-trigram +
  PPL-ratio ≤ ~1.2 on a short window) is the confirmer.
- **Anti-cheats (mandatory):**
  1. **Specificity** — each token's spiking-LN output maps to ITS block output (matched ≫ mismatched margin), not
     a constant (the C1 specificity_margin basis, ~0.8).
  2. **Load-bearing lesion** — drop EITHER LN arm (centre or scale) → fidelity must drop (both load-bearing, as in
     the read-out-norm de-risk). Scramble the affine → drop. (Guards against "the downstream exact-RF MLP
     manufactures the output regardless of LN.")
  3. **No-norm control** — feed raw `x` (no centre/scale) → far below, confirming the norm is doing work.
  4. **Pool-noise honesty** — the means are rate-coded pools (1/√pool noise), as in `neural_norm`; report fidelity
     WITH that noise, not noise-free.

If LN passes, do **GELU second** (graded-read calibrated, same single-block fidelity basis), then **softmax last**
(the risky one — start with divisive-norm-by-mean as the denominator + f-I exp, measure the temperature
mismatch).

---

## 5. Honest verdict — are all three spiking-realizable, or is one a genuine boundary?

| Op | Spiking realization | Exact / approx | Reusable machinery | Risk |
|---|---|---|---|---|
| **LayerNorm** | subtractive mean-adapt (centre) + divisive-norm (scale) + affine-on-read | **approx** (mean-abs spread vs RMS; affine exact) | `enable_input_mean_adapt` + `enable_input_divisive_norm` (BOTH shipped, `bridge.py:6238`/`6190`); validated 96% (read-out-norm de-risk) | **LOW** |
| **GELU** | graded `a_cont`-style read calibrated, OR a GELU-fitted spiking neuron (Precise/MBE) | **approx** (fixed scalar monotone fit; LN bounds input) | `a_cont` graded path (`bridge.py:6144`); literature neuron-fit | **LOW** |
| **softmax** | divisive-norm denominator (linear-attention regime) + expansive-f-I exponential, OR recurrent ORGaNICs DN | **approx** (exact-trained-softmax is the suspect); hard-WTA is NOT a substitute | divisive-norm circuit (shipped) + `_spiking_cleanup` (WTA only, not soft) | **HIGH — the genuine-boundary candidate** |

**Bottom line.** **LayerNorm and GELU are spiking-realizable now, faithful-approximate, low-risk, with most of the
machinery already shipped in `sim/`** — LayerNorm by direct reuse of the two shipped normalization arms (96%
already shown for the double-centring half), GELU by a calibrated graded read or a fitted neuron. **Softmax is the
genuine boundary candidate**: its *denominator* is free (reuse divisive normalization, in the linear-attention
approximation), but its *content-dependent exponential amplification* is exactly the rate-code/point-neuron-limit
family — a graded, multi-key, soft amplification a point-neuron rate code does poorly. The literature's working
spiking softmaxes (Spikformer/SDSA/A²OS²A) **eliminate softmax by re-architecting + retraining** the attention,
which would NOT preserve the consolidated Gen-F weights; the only *faithful* paths (expansive-f-I + divisive norm,
or recurrent ORGaNICs DN) are approximate and untested here. So the honest framing for the build: **C1-fully-spiking
is reachable for 2 of 3 ops cheaply; softmax should be scoped as "approximate, fidelity-cost-to-be-measured, with
an honest-negative as a real deliverable if the rate-code exponential wall holds"** — and per the standing SURPASS
rule, that negative is itself only accepted after a dedicated round isolating the *exact* residual (the exponential
temperature) and measuring how far the divisive-norm + f-I approximation actually falls from the trained softmax.

---

## Sources (literature)

- Spiking Transformer: Accurate Addition-Only Spiking Self-Attention (A²OS²A) — arxiv.org/abs/2503.00226 (eliminates
  softmax via binary-Q + ReLU-K; no softmax-removal accuracy analysis).
- Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformer — arxiv.org/abs/2508.07710
  (decomposes softmax/GELU/LayerNorm into basis functions + custom neurons; GELU input domain constrained because
  LayerNorm precedes it).
- Precise spiking neurons for fitting any activation function in ANN-to-SNN Conversion — Springer, *Applied
  Intelligence* 2025 (joint threshold/reset/membrane fit of arbitrary activations → GELU).
- Unconditional stability of a recurrent neural circuit implementing divisive normalization (ORGaNICs) —
  arxiv.org/abs/2409.18946 (DN replaces softmax in an attention head; recurrent, unconditionally stable).
- "Any normalization works as well as softmax; the important parts are an exponential plus a vector norm" —
  recurrent-RNN-view of softmax / linear-attention = first-order softmax (arxiv 2507.23632).
- Heeger (1992) divisive normalization ≡ softmax competitive renormalization (synthesis: Towards Data Science,
  "We Didn't Invent Attention — We Just Rediscovered It").

## Project sources (verified file:line above)
- `research/findings/raw/_genseq_loopstep3_full_genf_generate.json` (C1 generate, scope_note).
- `research/findings/2026-06-22-genseq-loopstep3-fullblock-rf-integration-GO.md` (full-block, the faithful-reads table).
- `sim/bridge.py:6190` (divisive norm), `:6238` (subtractive mean-adapt), `:6144` (a_cont graded read), `:6762` (RESONATE_AND_FIRE).
- `sim/config.py:469-492` (the three config flags).
- `research/runners/_phaseC_S5_divnorm_derisk.py:115` (`build_divnorm_score_bridge`).
- `research/runners/_phaseB_biologize_readout_norm_derisk.py:48` (`neural_norm` — validated 96% double-centring).
- `research/runners/rf_phasor_composer.py:316` (`_spiking_cleanup` — NEF WTA, hard-max only).
- `2026-06-16-biologization-sweep-conversational-pipeline.md` (read-out norm 96% of host).
