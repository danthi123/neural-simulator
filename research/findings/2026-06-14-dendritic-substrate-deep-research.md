# Dendritic substrate — deep research + reference-catalog review (the opening move before any months-scale build)

**Status:** READ-ONLY deep-research + catalog/literature/code review (the project's standing "deep research FIRST at a
new direction / multiply-confirmed roadblock" move, CLAUDE.md). No `sim/` code, no build, no GPU. Single deliverable =
this doc. **Date:** 2026-06-14. **Author role:** read-only research subagent. Every load-bearing project fact below was
re-verified against the project's own record (file/line citations); the surprising ones were read in full, not trusted
from a prior summary. **This is a scoping/decision doc, NOT a brain-based result and NOT a commitment to build.**

---

## 0. Terms (defined once — no undefined acronyms)

- **point neuron** — a neuron with ONE membrane state (`v,u` for Izhikevich; `v,w` for AdEx; `V,m,h,n[…]` for
  Hodgkin-Huxley): all synaptic input is summed at a single somatic compartment before the spike decision. The project's
  substrate. Cannot do per-input-source sub-threshold computation.
- **dendritic compartment** — a separate electrical sub-region of a neuron with its OWN local membrane potential that
  integrates a SUBSET of the input and computes a local sub-threshold quantity BEFORE it reaches the soma. The
  capability point neurons lack.
- **decorrelation / whitening** — removing the shared "common mode" so codes become statistically independent enough to
  bind/invert. Whitening is the stronger form (decorrelate AND equalize variance → identity covariance). The "common
  mode" is the large shared component all correlated concept codes carry; subtracting it exposes the small
  concept-distinguishing residual.
- **Mikulasch-Priesemann limit** — the peer-reviewed result (Mikulasch, Rudelt, Priesemann, *PNAS* 2021) that a point
  neuron with a single global inhibitory pool cannot whiten correlated inputs; lateral inhibition must be computed
  LOCALLY in dendritic compartments. The project's documented decorrelation/whitening boundary.
- **FHRR (Fourier Holographic Reduced Representation)** — the production binding algebra (concept = vector of phases;
  bind = element-wise complex product; unbind = multiply by complex conjugate; "exact-inverse" because unbind is the
  algebraic inverse of bind). Realized on the bridge's resonate-and-fire neurons + complex synapses.
- **denoise64 codes** — the composer's REAL concept codes: captured + denoised firing of concept pools on a trained
  bridge. Correlated (between-code cosine ≈ 0.79–0.81), lossy, grounded in the brain's own activity.
- **reproducibility gate** — same input fed twice under realistic spiking noise must produce the same code: same-input
  cosine ≥ 0.9 at σ=0.1 (10% of input norm). The load-bearing gate that has killed multiple prior attempts.
- **Urbanczik-Senn** — the canonical learnable two-compartment (soma + dendrite) neuron: the dendrite predicts the
  somatic firing and a LOCAL rule minimizes the mismatch (Urbanczik & Senn, *Neuron* 2014). Committed in the project as
  `sim/dendritic_plasticity.py`.
- **CLS / dual architecture** — Complementary Learning Systems (McClelland-McNaughton-O'Reilly 1995): two linked
  representations — similar/correlated codes in "cortex" (for generalization) + decorrelated codes in a
  "hippocampal/cerebellar" expansion (for binding) — coupled by encode/decode.

---

## (a) Headline verdict

**The dendritic path is warranted, but NOT as a whitening front-end for binding — that specific premise was already
falsified this arc, and a dendrite cannot rescue it.** The cheapest de-risk the prior research doc recommended
(`2026-06-11-option-B-dendritic-substrate-research.md` §4) HAS RUN
(`2026-06-11-option-B-whitening-derisk-NEGATIVE.md`): on the brain's REAL correlated `denoise64` codes, decorrelation +
reproducibility(≥0.9 @ σ=0.1) + composition **do not co-occur even for a god's-eye ideal whitening** (concept-whiten
ceiling reproducibility 0.609 ≪ 0.9), because the codes are **sub-reproducible at σ=0.1 BEFORE any whitening** (raw
0.161) and whitening *amplifies* the low-variance noise. A dendrite computes whitening more faithfully; it cannot
manufacture an input signal-to-noise ratio (SNR) the codes do not contain, and it makes the over-whitened-direction
problem worse, not better — so "whiten denoise64 to bind" is a confirmed dead end the dendrite does not change. The
project correctly pivoted: the **dual/CLS architecture** was de-risked **GO end-to-end on the real substrate**
(`2026-06-11-dual-CLS-cortex-channel-derisk-GO.md`) with ONE unbuilt piece — *a learned spiking-cortical embedding that
PRODUCES graded-but-reproducible cortex codes on neurons* (today a synthetic codebook stands in for it) — and a
2,048-concept conversational cortex was DELIVERED on the decorrelate-first build
(`2026-06-14-phase1-production-32bridge-2048-concept-cortex-DELIVERED.md`). **Where the dendrite is genuinely the
principled escape is the OTHER, still-open question: can the substrate LEARN graded semantic structure from experience
so that similar concepts get similar-but-reproducible codes** — exactly the capability Option C just failed to learn on
the point-neuron substrate (`2026-06-14` Stage-B GPU run: brain `Pearson(S_learned, S_true) = −0.008`, dead zero, vs a
host distributional-semantics control proving the structure IS in the data at +0.532; the **fifth** mechanistically-
distinct point-neuron NEGATIVE). **The single cheapest de-risk:** an afternoon-scale numpy toy asking whether a minimal
**2-compartment / dendritic-error unit** can LEARN a known a-priori category block-structure `S_true` from a
common-mode-injected sparse-code stream where the **point-neuron control gives ≈0** (reproducing the Option-C −0.008 and
the four prior NEGATIVEs) — re-run, faithfully, the project's actual failing case with the one capability the substrate
lacks added. Do NOT commit the months-scale build until that toy returns GO; if it returns NEGATIVE it is itself a
major, citable result (a clean map of what even a dendritic point-of-departure cannot learn from experience).

---

## (b) Diagnosis — why the point-neuron limit is real and dendritic computation is the principled escape

### (b.1) What a point neuron does each step (grounded in the real per-step update)

In `sim/bridge.py::_run_one_simulation_step` (line 5456) the engine assembles ONE scalar drive per neuron,
`total_input_current_pA`, by summing every current source — AMPA/GABA synaptic current, NMDA, the dendritic-COINCIDENCE
plateau (5805–5849), GABA_B, neuromodulator drive, OU noise — then at line ~5956 dispatches on `cfg.neuron_model_type ∈
{IZHIKEVICH, HODGKIN_HUXLEY, ADEX}` and updates a single somatic `(v,u)`/`(V,w)`/`(V,m,h,n)` BEFORE the spike decision.
**Excitation and inhibition from all sources are averaged together at one point.** This is exactly the structure
Mikulasch-Priesemann prove cannot whiten.

### (b.2) Why two compartments break the limit (the crisp WHY)

To whiten, the lateral inhibition between two units must match THEIR specific shared correlation — an **input-specific**
quantity (a different inhibitory weight per correlated pair). The mechanism, read from the source (Mikulasch-Priesemann
2021, PMC8685685; arXiv 2010.12395): each neuron has dendritic compartments, one per input dimension, each holding a
local potential `u_ji = F_ji·x_i + Σ_k W_jki·z_k` where the recurrent-inhibition term `Σ_k W_jki z_k` is delivered
**locally to that compartment**. The paper's key identity is that this local dendritic potential is **proportional to the
coding error**, `u_ji = F_ji·(x_i − x̂_i)` — the part of the input NOT already predicted by the rest of the
population. The recurrent inhibition **locally cancels the predictable (common-mode) part of the excitation in each
compartment**, and crucially **this cancellation happens at the analog membrane-voltage level, BEFORE the soma spikes**
("decorrelation occurs primarily at the analog membrane voltage level, not spike output"). The soma then sums the
already-balanced compartments and fires from the residual.

A point neuron CANNOT do this because all inhibition arrives at the one somatic compartment and is **averaged together
with excitation, destroying the input-specific structure before any plasticity rule can read it** — a single global
inhibitory pool can only apply uniform suppression. Hence point-neuron + global inhibition reaches *mean* decorrelation
at best, never the *all-pairs / worst-pair* whitening that binding needs. **The second compartment is the analog
averaging stage** (over inputs AND over time) that raises SNR before the spike; that is precisely why Mikulasch-
Priesemann report dendritic balance "prevents the sudden collapse" point neurons suffer at transmission delays > 0.3 ms
and is robust across 50 network realizations. The 2024–2025 literature states this identity directly: the tight
excitation-inhibition balance during dendritic error computation **"effectively decorrelates neural responses to inputs
and thereby ensures an efficient neural code"** (Mikulasch et al., *Dendritic predictive coding*, arXiv 2205.05303;
*Where is the error?*, *Trends in Neurosciences* 2022) — the catalog's G.02 "active dendrites" entry, flagged MISSING.

### (b.3) Why the five point-neuron NEGATIVES all converge here

| NEGATIVE (committed) | what it tried | why it failed |
|---|---|---|
| vanilla Hopfield | post-hoc clean correlated codes via Hebbian attractor | common-mode eigenvalue ~18× signal → all cues collapse to one attractor |
| Storkey (local covariance) | a LOCAL covariance-corrected weight rule | locality wall: a local pairwise rule cannot reach `C⁻¹`; only a non-local matrix inverse removes the common mode |
| spiking dentate-gyrus + rate-kWTA | decorrelate by sparse spiking expansion + competitive read | input-driven signal below the spiking noise floor → reproducibility ≈ separation at every k |
| fixed random expansion + threshold (Marr-Albus) | deterministic granule expansion + top-k | common mode survives the linear expansion (between-cos stays ~0.55); threshold margin ~3,800× below realistic noise → repro 0.03–0.06 |
| **Option C — learn paradigmatic similarity from REAL text (2026-06-14)** | spiking-Hebbian recurrent learn over context-inclusive TinyStories co-occurrence + divisive-norm read-out | brain `Pearson(S_learned,S_true) = −0.008` (dead zero; generalization 1.1× chance) **while the host PPMI+SVD control = +0.532** → the structure IS in the data; the point-neuron substrate learned **none** of it |

The common thread: **all five operate on the SPIKE side (or a single-shot threshold on a summed-at-soma activation)**,
where the small concept-distinguishing residual sits under a large correlated common mode at low SNR. The dendritic
mechanism removes the common mode in the **analog sub-threshold domain** where the membrane averages over compartments +
time, raising SNR before the spike — the analog averaging stage all five spike-side attempts lacked. Option C is the
sharpest: a host distributional-semantics measure recovers the paradigmatic category taxonomy (+0.532, all 8 categories,
nearest-neighbour-same-category 0.859), so the failure is **not** the data and **not** the measure — it is the
point-neuron Hebbian learn, exactly the Mikulasch-Priesemann prediction.

### (b.4) The honest counter-weight: the dendrite is NOT a universal solvent here

Two project results bound the optimism and MUST be carried to the owner:
1. **Whitening denoise64 to bind is dead even for an ideal whitening** (`2026-06-11-option-B-whitening-derisk-NEGATIVE`):
   the codes are sub-reproducible at σ=0.1 *raw* (0.161), reaching 0.9 only at σ≈0.012; whitening amplifies the
   low-variance directions (ideal ZCA repro 0.254 < raw at low noise). A dendrite computes the SAME whitening more
   faithfully → it would hit the **identical** wall (the NEGATIVE is in the codes-at-this-noise-level, not the
   mechanism's fidelity). **⇒ the dendrite is the wrong tool for the binding-input problem; that problem is solved by
   generated decorrelated codes + the dual architecture, already shipped.**
2. **The dual/CLS architecture already routes graded similarity on-substrate GO** (`2026-06-11-dual-CLS-cortex-channel-
   derisk-GO`, 3/3 seeds): graded cortex-channel generalization 1.000 (4× chance), orthogonal + permuted controls
   collapse — but with a **SYNTHETIC graded codebook** standing in for "the learned spiking-cortical embedding that
   PRODUCES graded codes on neurons," explicitly flagged as the one unbuilt months-scale piece.

**⇒ the precise, narrowed question the dendrite IS the principled escape for: can a dendritic substrate LEARN, from
experience, a graded-but-reproducible cortical embedding (similar concepts → similar codes) that a point neuron
provably cannot (Option C −0.008)?** That is the dual architecture's last unbuilt piece AND the Option-C capability,
and it is squarely the Mikulasch-Priesemann/dendritic-predictive-coding wheelhouse.

---

## (c) Ranked, biologically-grounded options

### Option D1 (RECOMMENDED FIRST — the cheap-first de-risk target) — a minimal 2-compartment "dendritic-error" learner, OFF-bridge numpy, learning graded structure from a common-mode stream

- **Mechanism.** A reduced 2-compartment unit (soma + one dendrite). The dendrite integrates the feedforward drive AND
  local recurrent inhibition and computes the coding-error residual `u = F·(x − x̂)` in the analog (graded) domain
  BEFORE the soma; inhibition learns the local-balance rule `ΔW_jk ∝ −z_k·u_j` (Mikulasch-Priesemann) so it cancels the
  predictable common mode; the soma fires from the residual; the readout learns the graded similarity over the residual
  codes. This is the *function-faithful reduced* form — one dendrite, not N — sufficient to test the load-bearing claim.
- **Biological grounding.** Mikulasch-Priesemann 2021 (dendritic balance whitens); Urbanczik-Senn 2014 (dendrite
  predicts soma, local rule); the 2024 PLOS Comp Biol V1 model where tight E/I balance yields sparse coding + balance +
  **decorrelation** from a biologically-grounded spiking net; the Nov-2025 V1 two-compartment predictive-coding preprint
  (positive/negative prediction errors via soma–dendrite comparison, decorrelation from tight E/I balance).
- **Fit to this sim.** HIGH as a *de-risk* (pure numpy, reuses the project's exact failing pipeline + the committed
  `urbanczik_senn_update` rule); it is the smallest thing that confirms-or-falsifies the whole dendritic premise before
  any protected edit. **This is option (e) below, fully specified.**

### Option D2 (the faithful on-substrate target IF D1 is GO) — a true two-compartment `NeuronModel` in the bridge step-loop

- **Mechanism.** Add `NeuronModel.TWO_COMPARTMENT` (or extend Izhikevich with a second `v_dend` state): each neuron gets
  a dendritic compartment computing the analog coding-error residual + local lateral inhibition; the soma fires from the
  residual; basal plasticity is apical-gated (Bono-Clopath 2017 / Urbanczik-Senn). The step-loop slot is exactly where
  the **dendritic-COINCIDENCE plateau** already sits (`bridge.py:5805–5849`): a per-postsynaptic-neuron NONLINEAR
  restricted-matvec sub-threshold term, fully guarded (byte-identical when off), computed in the current-accumulation
  phase BEFORE the soma integrates. A dendritic compartment is the same SHAPE — a guarded, per-neuron, restricted-matvec
  sub-threshold term added before the dispatch at line ~5956.
- **Biological grounding.** The literature-faithful Mikulasch-Priesemann / Larkum BAC / Urbanczik-Senn mechanism; the
  catalog's repeatedly-flagged missing capability (G.02 active dendrites; B.17 dendritic linearization; D.* CA1 distal
  apical; F.* cerebellar granule/Golgi max-pool; I.* multi-compartment channels). The PNAS-2025 "Spiking world model with
  multicompartment neurons" is a recent existence proof that multicompartment SNNs match GRU-class performance — i.e.
  the substrate class is viable at scale.
- **Fit to this sim.** HIGHEST fidelity, HIGHEST cost. A new `NeuronModel` is a protected `sim/` edit on the hottest
  code path; the byte-identity-when-off discipline (the coincidence-plateau precedent) is the mitigation. The prior
  design doc (`2026-05-05-dendritic-learning-design.md`) scopes a *reduced* 2-compartment neuron at **1.5–2 months**;
  the literature-faithful N-compartment form is heavier still.

### Option D3 (alternative escape, lower fidelity, cheaper) — gain-modulating-interneuron adaptive whitening as a separate analog pre-spike population

- **Mechanism.** Duong-Lipshutz-Heeger-Chklovskii-Simoncelli (ICML 2023; ICLR 2025) map adaptive whitening onto a
  recurrent net with **fixed synaptic weights + gain-modulating interneurons** (the marginal-variance route), a
  different biological substrate than plastic lateral inhibition. Realize it as a separate analog population that
  pre-whitens before the spiking binder.
- **Biological grounding.** Gain modulation is well-documented; but there is **no demonstrated SPIKING realization**
  (confirmed by the project's prior doc and this search), so it is rate/graded only — portable as a *rule*, not a
  turnkey spiking module.
- **Fit to this sim.** MODERATE; it inherits the §(b.4.1) wall on denoise64 (it is still whitening), so it is only
  relevant inside the dual architecture's decorrelated-expansion side or for the graded-cortex-learning question, not as
  a denoise64 binding front-end. Ranked below D1/D2 because it does not address the *learn-graded-structure* question
  that is actually open.

### Ranking summary

| | Computes/learns by | Substrate touch | Fidelity (function/morphology) | Cost | What it actually tests |
|---|---|---|---|---|---|
| **D1 (do FIRST)** | reduced 2-compartment dendritic-error learner, numpy | NONE (off-bridge probe) | MODERATE-HIGH / reduced | **afternoon (numpy)** | the load-bearing claim, cheaply falsifiable |
| **D2 (faithful target if D1 GO)** | true `NeuronModel` soma+dendrite, learned U-S rule | new `NeuronModel`, hot path (guarded) | HIGHEST / HIGHEST | **months (1.5–2+)** | the proper brain analogue |
| **D3 (alt, lower fidelity)** | fixed-weight + gain-modulating interneurons | new analog population | MODERATE / low | weeks (rate) | adaptive whitening; no spiking realization yet |

---

## (d) Reusable project machinery (concrete, file-cited)

- **The committed dendritic stack (the biggest reuse — do NOT reinvent).** `sim/dendritic_neuron.py` (58 lines —
  spiking two-compartment BAC pyramidal: basal forward + apical FIXED-RANDOM feedback + Larkum threshold lowering, no
  weight transport), `sim/dendritic_plasticity.py` (41 lines — the LOCAL **Urbanczik-Senn** `urbanczik_senn_update`
  somato-dendritic mismatch rule, apical-gated), `sim/dendritic_mlp.py` (195 lines — Guerguiev-Lillicrap-Richards 2017
  feedback alignment, GPU-capable via `sim.backend`). All on `main`, all pure numpy, **off-bridge, and currently
  SUPERVISED** (the apical signal is a teaching signal). **The repurposing for D1/D2:** swap the supervised teaching
  signal for the UNSUPERVISED local-balance objective; wire into the step-loop (D2 only). Calling this "already done"
  would over-claim — it is the morphology + local-rule scaffold, not the unsupervised graded-learning front end.
- **`sim/predictive_coding.py` (125 lines)** — a Rao-Ballard top-down predictor (`PredictiveCoder`), off-substrate,
  local-only; the prediction-error idiom the dendritic-error learner needs.
- **The dendritic-COINCIDENCE plateau (the step-loop integration precedent for D2).** `bridge.py:5805–5849` +
  `sim/kernels.py:252` (`fused_coincidence_plateau`): the EXACT pattern for a per-neuron NONLINEAR restricted-matvec
  sub-threshold term, guarded to byte-identity when `cfg.enable_coincidence_detection` is off (the block is unreached,
  the kernel never invoked, `total_input_current_pA` byte-identical), computed BEFORE the soma integrates. Its docstring
  even cites Poirazi-Brannon-Mel two-layer subunit + Major-Larkum-Schiller NMDA spike — a dendritic-compartment term
  slots in the same guarded way. This is the protected-edit template.
- **The rich multi-current HH machinery (assess honestly).** `sim/kernels.py` has fused NaP (persistent Na⁺), CaT
  (T-type Ca²⁺), Ih (h-current), M-current, and the core HH dynamics; `sim/enums.py` has L5-pyramidal HH presets with
  per-gate Q10; `sim/config.py` has the HH params. **But every one of these is SINGLE-COMPARTMENT** — they add active
  *currents* to one somatic membrane, NOT a second electrical compartment with its own potential. The catalog confirms
  it bluntly (G.01: "dendrites are collapsed to a point neuron"; G.02 active dendrites MISSING; "single compartment can
  only sum inputs"). **So there is NO reusable dendritic compartment in the bridge today** — the smallest credible
  addition is a second per-neuron state array `v_dend` + a guarded sub-threshold term (D2), reusing the CaT/NaP/NMDA
  kernel idioms for the dendritic nonlinearity but in a NEW compartment. The HH currents are the right *ingredients* for
  the dendritic nonlinearity; they are not the compartment.
- **The Option-C / whitening de-risk harnesses (reuse verbatim for D1).** `research/runners/option_B_whitening_derisk_
  probe.py` (the four-gate harness: decorrelation, reproducibility@σ, composition, generalization; the analog-settle +
  M-ratio + lesion + native-convention controls), `research/runners/option_c_real_cooccurrence_derisk.py` /
  `option_c_paradigmatic_host_precheck.py` (the host PPMI+SVD pre-gate + the a-priori `S_true` taxonomy + the
  spiking-Hebbian learn + divnorm readout + permuted-co-occurrence anti-cheat), `cortex_fixed_expansion_decorrelation_
  probe.py` (the `between_cos` / `repro@σ` / `margin/noise` instrumentation). These ARE the D1 metric + anti-cheats,
  already written and validated.
- **The dual/CLS substrate the learned embedding must plug into (all validated).** `research/runners/dual_cls_cortex_
  channel_derisk_probe.py` + `dual_cls_strong_encode_derisk_probe.py` (strong-DG encode, repro 1.000 + decorr ≈0 at k=40;
  spiking Hopfield recall identity 1.000; cortical reinstatement); `concept_pool_sparse_distributed.py`
  (`generate_sparse_patterns`, the decorrelated codes); `_D_sparse_heteroassoc.py` (the permuted-control-clean
  attractor); the V=320 vocab-ceiling acceptance matrix; the merged one-bridge builder. The dendritic embedding's job is
  to *supply the graded cortex code* this validated plumbing already routes.
- **`sim/backend.py`** — any new kernel (the D2 compartment update) MUST route through `get_backend()`/`fuse()` so it
  works on CuPy + NumPy (the committed dendritic stack already does this).
- **The prior design docs.** `2026-05-05-dendritic-learning-design.md` (reduced 2-compartment scope, `target_compartment`
  routing on `RegionPathway`, the 1.5–2 month estimate); `2026-06-02-dendritic-predictive-coding-cheap-first-probe-
  design.md` (the local-rule predictive-coding probe design); `2026-06-11-option-B-dendritic-substrate-research.md` (the
  prior opening-move research this doc updates).

---

## (e) THE recommended cheap-first de-risk (fully concrete, runnable by a later engineer)

**The single open scientific question the dendrite is the principled escape for:** *can a minimal dendritic
nonlinearity (one extra compartment) LEARN a known a-priori category structure from a correlated, common-mode-injected
code stream where the point-neuron substrate provably gives ≈0 (the Option-C −0.008 and the four prior NEGATIVEs)?*
Run the project's ACTUAL failing case with the one capability the substrate lacks added, on numpy, before any protected
edit.

### (e.1) The toy (smallest thing that falsifies the dendritic premise)

**Inputs — the faithful synthetic analogue of the failing case (so the result is interpretable AND the a-priori
control is exact):**
- `C` categories × `M` concepts/category (start C=8, M=8 = 64, matching the Option-C taxonomy). Each concept = a
  **K-of-N sparse binary code** (start N=512, K≈26, the project's sparse-distributed scale) drawn so that within a
  category the codes share a category-specific sub-pattern (graded similarity) and an **injected COMMON MODE** (a shared
  dense component added to every concept, mean-magnitude tuned so raw between-cos ≈ 0.8, matching denoise64). `S_true` =
  the a-priori `C×C` (or concept-level) block-diagonal category-similarity matrix — **constructed, never data-derived**
  (this is the crux: the structure is known by construction, not read from the stream).
- A **co-occurrence STREAM** drawn from the codes (the "experience"): present concepts with category-structured
  co-occurrence (same-category concepts share context more often), exactly the build-toy-cooccurrence mechanism the
  Option-C runner uses — so the structure is present in the stream but the learner must extract it.

**The unit model (D1):** a reduced 2-compartment learner over the codes —
- *dendrite:* `u_j = F_j·x − Σ_k W_jk·z_k` (feedforward drive minus local recurrent inhibition), the analog coding-error
  residual; settle the recurrent term over a few sub-steps (the analog membrane averaging — reuse the stable leaky
  settle `dr/dt = W_ff·x − r − M·r̂` from `_A_whitening_computation_derisk.py`, which already carries a noiseless-solver
  control). *(Reuse `urbanczik_senn_update` for the soma-prediction variant if testing the U-S form.)*
- *soma:* fires from `u_j` (graded rate read for the metric; spike read for the on-substrate confirmation).
- *learning:* the LOCAL Mikulasch-Priesemann inhibition rule `ΔW_jk ∝ −z_k·u_j` (inhibition learns to cancel the
  predictable common mode) + a regularizer/decay `−λW` (the rank-deficiency guard the project already learned it needs);
  the readout learns the graded similarity over the residual codes.

**The point-neuron NEGATIVE control (MUST reproduce ≈0):** the IDENTICAL pipeline with the dendritic compartment
COLLAPSED to the soma (sum all input at one point, single global inhibition) — i.e. the project's existing spiking-
Hebbian learn. It MUST give `Pearson(S_learned, S_true) ≈ 0` (reproducing Option-C −0.008 and the four NEGATIVEs); if it
does not, the toy is mis-calibrated (the common mode is too weak) and must be re-tuned until the point neuron fails,
BEFORE trusting any dendritic GO.

### (e.2) The gates (reuse the project's exact metrics; D1 GO requires the contrast)

1. **STRUCTURE RECOVERY** — `Pearson(S_learned, S_true)`: dendritic learner **well above 0** (target ≥ +0.3, ideally
   approaching the host ceiling +0.5) WHILE the point-neuron control ≈ 0. *The contrast IS the result.*
2. **GENERALIZATION (held-out)** — train relations on SOME similar-concept pairs, test inference on a HELD-OUT
   similar pair never trained in that relation; require **above chance** for the dendritic learner and **at chance** for
   the point-neuron control (reuse the Option-C generalization gate + the leakage assert).
3. **REPRODUCIBILITY** — same code fed twice at σ=0.1 → residual-code same-input cosine ≥ 0.9. *Honest caveat:* if the
   common mode is injected at denoise64 levels this gate may be the binding constraint exactly as in the Option-B
   NEGATIVE; report it explicitly. The dendritic claim is that the analog residual is MORE reproducible than the
   point-neuron read at the same noise — that delta is the informative quantity even if neither hits 0.9.
4. **COMPOSITION (do NOT gate on coherence alone — the 06-06 lesson)** — bind+unbind over the residual codes; confirm
   recovery, because a noise-collapsed output passes (1)–(3) trivially but will NOT compose.

### (e.3) Outcomes (three-state, pre-registered)
- **GO** — dendritic learner clears structure + generalization (above chance) WHILE the point-neuron control is at ≈0,
  with the analog-not-host proof + lesion + reproducibility-delta, multi-seed (42/43/44). ⇒ the dendrite IS the escape
  for the learn-graded-structure question → present the D2 on-substrate build plan + cost. **This is the likely-good
  outcome given Mikulasch-Priesemann's own correlated-input robustness claims.**
- **BOUNDARY** — dendritic learner beats the point neuron but only partially (e.g. recovers some categories, like the
  host's animals/family/actions, not all). ⇒ partial escape; informs whether a fuller N-compartment form (heavier) is
  warranted, or whether the dual-architecture-with-curated-similarity (already shipped) is the pragmatic answer.
- **NEGATIVE** — even the dendritic learner gives ≈0 (or fails reproducibility identically). ⇒ a clean, citable result
  that the *learn-graded-structure-from-experience* gap is deeper than one extra compartment (points to the full
  multi-compartment form, or to "the agent's structured experience must be curated" — Option B, which is shipped). Saves
  the months-scale build.

### (e.4) Cost
**Afternoon-scale, CPU/numpy** (the four prior whitening/expansion/Option-C precheck probes each ran in minutes-to-an-
hour on CPU; this reuses their harnesses + the committed dendritic rule). NO GPU, NO `sim/` edits for D1. The
on-substrate confirmation (if D1 GO) is a small GPU step on the existing dual/CLS bridge; the full D2 build is the
months-scale protected edit, deliberately gated behind D1.

---

## (f) Anti-cheat controls (so a "success" cannot be a host shortcut)

- **The decorrelation/learning must be done BY the dendritic mechanism, not a host op.** The residual must be the FIXED
  POINT of the simulated leaky settle integrated over sub-steps (the membrane averaging), NOT a one-shot host
  `C^{−1/2}x` / `numpy.linalg` whitening; show it CONVERGES over steps (a host ZCA has no settle). The lateral matrix
  must be LEARNED from zero by the local rule and BOUNDED (the M-ratio guard from `_A_whitening_learn_lateral_derisk.py`:
  `‖M_learned‖/‖M_analytic‖` near 1, NOT 9000× — the guard that caught the prior blow-up false-positive).
- **`S_true` must be A-PRIORI, never corpus/stream-derived.** Assert `s_true_independent` (the Option-C runner's exact
  check — it held every seed there). The category block-structure is constructed; the stream only *exhibits* it.
- **The point-neuron control MUST FAIL on the IDENTICAL pipeline** (≈0). This is the headline anti-cheat: a dendritic GO
  only counts against a point-neuron NEGATIVE on the same codes/stream/metric/seeds. (Re-tune the common-mode magnitude
  until the point neuron fails before trusting any dendritic number.)
- **PERMUTED-SIMILARITY control must collapse.** Shuffle which concepts are "similar" (break code-similarity ↔ category
  correspondence) → structure recovery AND held-out generalization collapse to chance. Otherwise the "generalization" is
  a code-overlap artifact unrelated to meaning.
- **PERMUTED-CO-OCCURRENCE control must collapse** (the Option-C headline anti-cheat): re-learn on a scrambled stream →
  `Pearson(S,S_true)` collapses, generalization → chance. (It did collapse robustly in the Option-C Stage-B run even at
  −0.008 — keep it.)
- **LESION the compartment → the effect collapses.** Zero the dendritic term / freeze the settle → structure recovery
  returns to ≈0 and reproducibility collapses → proves the effect RIDES the simulated dendritic dynamics, not a leftover
  code property (the fixed-expansion + Option-B NEGATIVEs both used exactly this lesion; reuse it).
- **Native-convention unit check** — codes read in native binary/mean-removed form, between-cos ≈ the expected ~0.8 (NOT
  ≈1.0 from accidental median-bipolarization — the exact artifact that manufactures a false common mode and would
  produce a false NEGATIVE).
- **Composition-not-coherence gate** — catches a noise-collapsed output that passes decorrelation + reproducibility but
  cannot bind (nearly shipped THREE false positives in the 06-06 arc; composition is the gate that caught it).
- **At the on-substrate confirmation (if any):** the shuffled-fact permuted control at zero false hits + the abstention
  floor (the no-confab moat) carried from the V=320 vocab-ceiling acceptance.

---

## (g) Honest cost + intermediate value

**Cheap de-risk (e):** afternoon-scale, CPU/numpy, no `sim/` edits, reuses four existing harnesses + the committed
dendritic rule. **Full step-3(B) dendritic-substrate build (D2):** the prior design doc estimates **1.5–2 months** for a
*reduced* 2-compartment neuron (new `NeuronModel`/kernel + apical plasticity gate + `target_compartment` routing +
validation); the literature-faithful Mikulasch-Priesemann N-compartment form (one compartment per input dim) is heavier
still — and the project's own dendritic credit-assignment arc terminated in an honest VOID
(`2026-05-18-dendritic-fairscale-SOUND-instrument-VOID`: the machinery was sound but the *discriminating* science could
not be built at feasible scale on a *different* question), so the estimate is a floor, not a ceiling, and the build
should be escalated only after (e) returns GO. It is a protected `sim/` edit on the hottest code path → byte-level diff
review required (the owner's standing rule), with the coincidence-plateau byte-identity-when-off discipline as the
template.

**Intermediate value of the cheap de-risk regardless of go/no-go:**
1. It converts the *current* state (Option-C −0.008 dead-zero + four prior NEGATIVEs) from "the substrate can't, probably
   because it's a point neuron" into a **directly-tested** claim: it isolates whether ONE extra compartment closes the
   learn-graded-structure gap, on the project's exact failing case, with the point-neuron control reproducing the
   failure — the cleanest possible attribution.
2. **A GO** justifies the months-scale build with a sharp target (learn the graded cortex embedding for the dual/CLS
   architecture's one unbuilt piece) and a validated downstream (encode → bind → recall identity → cortical
   reinstatement → generalization is all GO, waiting for the learned embedding).
3. **A NEGATIVE** is itself a major, citable scientific deliverable under the project's "honest negatives are the
   deliverable" standard: a clean map of what even a dendritic point-of-departure cannot learn from experience on this
   substrate, pinning the boundary one level deeper than Mikulasch-Priesemann's whitening claim and confirming that
   "the agent's structured experience must be curated" (Option B — already shipped at 2,048 concepts) is the pragmatic
   answer for the conversational goal, with the dendritic build reserved for the *artificial-life / biology-translatable*
   goal where learning structure from raw experience is the point.
4. Either way it **resolves the owner's go/no-go on a months-scale, protected, hot-path rewrite for an afternoon of CPU
   time** — the entire purpose of the standing "deep-research + cheap-first de-risk before committing build resources"
   practice.

**The single most important framing correction for the controller:** the prompt's premise ("is the dendritic substrate
the path past the whitening wall?") is answered NO for the *binding-input* version of the question (whitening denoise64
to bind was falsified even for an ideal whitening; the dendrite cannot manufacture the missing input SNR; that problem
is already solved by generated decorrelated codes + the shipped dual/CLS architecture + the 2,048-concept delivered
cortex), and YES, *conditionally and cheaply-testable*, for the *learn-graded-structure-from-experience* version (the
Option-C −0.008 failure is exactly the point-neuron limit, and the dendrite is its named escape). The recommended
de-risk targets the version that is actually open. **No banking** — reported exactly as found, including the part that
reframes the prompt's question.

---

## Sources

### Current literature (consulted this pass; 2019–2026 prioritized)
- Mikulasch, Rudelt, Priesemann, *Local dendritic balance enables learning of efficient representations in networks of
  spiking neurons* (**PNAS** 2021; PMC8685685; arXiv 2010.12395) + code **Priesemann-Group/dendritic_balance** —
  inhibition learns to LOCALLY balance excitation in dendritic compartments; whitening at the analog membrane-voltage
  level; robust to correlated inputs + transmission delays where Hebbian fails.
- Mikulasch, Rudelt, Wibral, Priesemann, *Dendritic predictive coding: a theory of cortical computation with spiking
  neurons* (arXiv 2205.05303, 2022) + *Where is the error? Hierarchical predictive coding through dendritic error
  computation* (**Trends in Neurosciences** 2022) — "the tight excitation-inhibition balance during error computation
  effectively decorrelates neural responses and ensures an efficient code"; two compartments suffice to compare
  target vs prediction.
- *Modelling Predictive Coding in V1: Layer 2/3 circuits for prediction-error computation through compartmentalized
  spiking neurons* (**bioRxiv** 2025.11.01.686040, Nov 2025) — two-compartment pyramidal cells encode bidirectional
  prediction errors via soma–dendrite comparison; decorrelation from tight E/I balance.
- Tang/Sun et al., *Emergence of Sparse Coding, Balance and Decorrelation from a Biologically-Grounded Spiking Neural
  Network Model of Learning in V1* (**bioRxiv** 2024.12.05.627100 → **PLOS Computational Biology** 2025) — sparse coding,
  E/I balance, and decorrelation emerge together from biologically-grounded spiking learning.
- Koren, Blanco Malerba, Schwalger, Panzeri, *Efficient coding in biophysically realistic excitatory-inhibitory spiking
  networks* (**bioRxiv** 2024.04.24.590955, 2025) — tight instantaneous E/I balance for efficient coding; feature-
  specific E/I recurrence.
- Urbanczik & Senn, *Learning by the dendritic prediction of somatic spiking* (**Neuron** 2014) — the canonical
  learnable two-compartment neuron + local rule (committed as `sim/dendritic_plasticity.py`; NEST has a reference impl).
- Sacramento, Costa, Bengio, Senn, *Dendritic cortical microcircuits approximate the backpropagation algorithm*
  (**NeurIPS** 2018; arXiv 1810.11393) — apical/basal dendritic-error microcircuit (rate-based).
- Guerguiev, Lillicrap, Richards, *Towards deep learning with segregated dendrites* (2017) — segregated apical/basal +
  fixed-random feedback (committed as `sim/dendritic_neuron.py`, `sim/dendritic_mlp.py`).
- Payeur, Guerguiev, Zenke, Richards, Naud, *Burst-dependent synaptic plasticity can coordinate learning in hierarchical
  circuits* (**Nature Neuroscience** 2021; bioRxiv 2020.03.30.015511) — apical-dendrite-controlled bursting + short-term
  plasticity multiplex feedforward/feedback so feedback steers plasticity locally without disrupting bottom-up signaling.
- Major, Larkum, Schiller, *Active properties of neocortical pyramidal neuron dendrites* + *The decade of the dendritic
  NMDA spike* — NMDA-spike supralinear branch integration → the two-layer-subunit (Poirazi-Brannon-Mel) nonlinearity
  (the project's `fused_coincidence_plateau` already cites these).
- Duong, Lipshutz, Heeger, Chklovskii, Simoncelli, *Adaptive whitening in neural populations with gain-modulating
  interneurons* (**ICML** 2023; arXiv 2301.11955) + *…with fast gain modulation and slow synaptic plasticity* (**ICLR**
  2025) — the gain-modulation whitening route; rate/graded, **no demonstrated spiking realization** (D3 caveat).
- *Spiking world model with multicompartment neurons for model-based reinforcement learning* (**PNAS** 2025; arXiv
  2503.00713) — multicompartment SNNs match GRU-class world models; existence proof that the substrate class scales.
- CLS theory: McClelland, McNaughton, O'Reilly 1995; Kumaran, Hassabis, McClelland 2016.

### Project record (internal, all re-verified this pass)
- **The prior opening-move research this doc updates:** `docs/plans/2026-06-11-option-B-dendritic-substrate-research.md`.
- **The de-risk that already ran (the load-bearing reframe):** `research/findings/2026-06-11-option-B-whitening-derisk-
  NEGATIVE.md` (ideal whitening fails reproducibility on denoise64; failure is upstream of any whitening → DUAL
  architecture, not the dendritic whitening front-end).
- **The dual/CLS GO + its one unbuilt piece:** `research/findings/2026-06-11-dual-CLS-cortex-channel-derisk-GO.md`
  (+ `-strong-encode-derisk-BOUNDARY.md`, `-architecture-proof-GO.md`).
- **The delivered cortex (the shipped Option-A/B answer):** `research/findings/2026-06-14-phase1-production-32bridge-
  2048-concept-cortex-DELIVERED.md`; the build plan + fork: `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-
  bind.md`.
- **The FIFTH NEGATIVE (Option C, the dendrite's actual open target):** `research/findings/2026-06-14-option-c-
  paradigmatic-host-precheck-VIABLE.md` (host +0.532) + `research/findings/raw/_option_c_stageB_fair_multiseed.log`
  (brain `Pearson(S_learned,S_true) = −0.008`, seed 42; multi-seed in flight) + `2026-06-13-option-c-real-cooccurrence-
  derisk-INCONCLUSIVE.md`.
- **The four prior point-neuron NEGATIVEs:** `2026-06-11-cortex-storkey-ca3-cleanup-NEGATIVE.md`, `-dg-ratekwta-cleanup-
  NEGATIVE.md`, `-fixed-expansion-decorrelation-NEGATIVE.md`, `-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md`;
  the positive control `2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`.
- **The split-record optimistic prior:** `2026-06-06-whitening-computation-spikes-CAN-compute-it.md` (spikes COMPUTE
  whitening with handed-in lateral inhibition), `2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md`,
  `2026-06-06-realobject-grounding-and-whitening-synthesis.md`.
- **The committed dendritic stack + its honest terminus:** `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`,
  `sim/dendritic_mlp.py`, `sim/predictive_coding.py`; `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-
  triangulation.md`; the prior designs `docs/plans/2026-05-05-dendritic-learning-design.md`, `2026-06-02-dendritic-
  predictive-coding-cheap-first-probe-design.md`.
- **The step-loop precedents + dispatch:** `sim/bridge.py:5805–5849` (dendritic-COINCIDENCE plateau, guarded),
  `sim/kernels.py:252` (`fused_coincidence_plateau`), `sim/bridge.py` neuron-model dispatch (~5956; IZHIKEVICH /
  HODGKIN_HUXLEY / ADEX — all single-compartment); the multi-current HH kernels `sim/kernels.py` (NaP/CaT/Ih/M, all
  single-compartment); `sim/enums.py` (L5-pyramidal HH presets, per-gate Q10).
- **Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):** G.01 (spatial summation = point
  neuron), **G.02 (active dendrites — MISSING, "one of the largest abstractions in the simulator," ~10× compute/neuron)**,
  B.17 (dendritic linearization, needs multi-compartment MSN), D.* (CA1 distal apical, needs multi-compartment), F.*
  (cerebellar granule expansion + Golgi max-pool needs lower-vs-upper dendrites), I.* (multi-compartment channels). Kandel
  6e Ch 13 pp 290–298 (passive + active dendrites).
