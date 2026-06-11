# Step 3 (the true cortex) — the CORE learned-binder research: opening-move findings after the cleanup arc closed

**Status:** READ-ONLY deep-research + catalog/literature review (the project's standing "deep research FIRST at a
new direction" opening move). No `sim/` code, no GPU, no build. The single deliverable is this doc + one commit.
**Date:** 2026-06-11.
**Author role:** read-only deep-research subagent. Every load-bearing project fact below was re-verified from the
project's own record (file + line/number citations); two were re-measured directly from the cached data rather than
trusted from a prior doc. Catalog (`sim-catalog/references/feature-catalog.md`), Kandel-derived textbook notes, and
2019–2025 literature were reviewed before any recommendation.
**Context:** the "post-hoc clean the FIXED correlated codes" cleanup sub-arc CLOSED with three mechanistically-distinct
NEGATIVES (vanilla Hopfield common-mode collapse; Storkey locality wall; spiking DG→CA3 sub-reproducibility, incl. the
rate-accumulated-k-WTA variant). Per systematic-debugging discipline (≥3 distinct fixes failed → question the
architecture), the frame is wrong. This doc verifies-or-refutes the PIVOT thesis and ranks the real step-3 options.

---

## 0. Terms (defined once — owner standing requirement; no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop. The "brain."
- **composer** — the conversational module that holds facts and answers questions. Production default is
  `RFPhasorComposer` (`research/runners/rf_phasor_composer.py`); a legacy rate variant is `CoreSimComposer`
  (`research/runners/core_sim_composition.py`). Both reuse-by-import (no `sim/` edit).
- **role-filler binding** — combining a *role* (agent / action / patient / polarity / attribute) with a *filler*
  (a concept word) into one composite vector. A fact "dog go north" is the **bundle** (vector sum) of three bound
  role-filler pairs.
- **bind / unbind** — *bind* makes the composite; *unbind* recovers a filler given a role ("who is the agent?").
- **VSA (Vector Symbolic Architecture)** — symbols as high-dimensional vectors, bound by an algebraic operation with
  a defined inverse (so unbind is exact up to noise).
- **FHRR (Fourier Holographic Reduced Representation)** — the production VSA scheme. Each concept is a vector of
  phases (a complex unit vector per dimension); bind = element-wise complex product, unbind = multiply by the
  complex conjugate. Realised on the bridge's **resonate-and-fire** neurons + **complex synapses**
  (`NeuronModel.RESONATE_AND_FIRE`, `rf_kick`, `rf_set_complex_weights`, `rf_read_phases`).
- **cleanup** — after unbind, the recovered vector is noisy; cleanup snaps it to the nearest stored concept code.
  Today this is a nearest-neighbour `argmax` over the codebook (with an opt-in spiking NEF/WTA variant).
- **the no-confab(ulation) moat / abstention** — the agent returns "I don't know" (`None`) when no stored fact
  matches the query, instead of inventing an answer. The hardest validated acceptance bar (100% = 20/20 unstored
  cues abstain, multi-seed, V=320).
- **familiarity / novelty signal** — a scalar/small population that reports "have I seen this before?" independent
  of *what* it is (perirhinal recognition memory).
- **NEF (Neural Engineering Framework)** — Eliasmith-Anderson's method for realising a vector function as a spiking
  population (encoders, decoders, a placed firing threshold). The project's spiking cleanup is an NEF circuit.
- **TPAM (Threshold-Phasor Associative Memory)** — Frady-Sommer complex-Hopfield: the phasor analogue of the
  NEF/argmax cleanup; vocabulary in `W = S Sᵀ*/D`, magnitude-gated phase-preserving transfer.
- **VTB (Vector-derived Transformation Binding)** — Gosmann-Eliasmith 2019: a *learned/derived* linear-transform
  binding operation for spiking VSAs, more dimension-efficient than circular convolution.
- **MAP (Multiply-Add-Permute)** — a VSA family that binds by element-wise multiply (the ±1 Hadamard scheme the
  rate composer uses is a MAP variant).
- **DG (dentate gyrus)** — the hippocampal stage that performs **pattern separation** (sparse expansion recoding).
- **CA3** — the hippocampal stage that performs **pattern completion** (a recurrent autoassociator / Hopfield net).
- **mixed selectivity** — single neurons tuned to nonlinear *combinations* of inputs (role AND filler), which lets a
  downstream linear readout separate combinations a pure rate code cannot (Rigotti-Fusi 2013).
- **systematicity (Fodor-Pylyshyn)** — if a system understands "dog chases cat," it should automatically understand
  "cat chases dog" without separate training. A symbolic algebra has this for free; learned networks notoriously do
  not. The central risk of step 3 (§2, §6).
- **denoise64 codes** — the composer's REAL concept codes: captured + denoised firing of 16 concept pools on a
  trained bridge, cached at `research/findings/raw/activity_level_integration_cache/denoise64_seed{42,43,44}.npz`.
  Correlated, lossy, grounded in the brain's own activity.
- **sparse-distributed (G.20) codes** — each concept = a scattered K-of-N random pattern (K=100 in an N=2000 pool).
  Generated by `concept_pool_sparse_distributed.generate_sparse_patterns`. Near-orthogonal by construction.
- **BPTT spiking cortex** — the project's surrogate-gradient backprop-through-time spiking network
  (`sim/bptt_snn.py`, `sim/bptt_snn_gpu.py`, `sim/surrogate_grad.py`, `sim/char_tokenizer.py`,
  `research/runners/cortex_pretraining.py`). All confirmed on `main` (§5).

---

## 1. VERIFY THE LOAD-BEARING FACTS

Each fact the pivot rests on, checked against the project's own record. **Verdict tags:** HOLDS / HOLDS-WITH-NUANCE
/ REFUTED. The two most load-bearing (the code cosines) were re-measured directly from the cached data.

### 1.1 Two code schemes + their actual between-code cosines

**(a) The RF/FHRR composer's `denoise64` codes — claimed cos ≈ 0.81 (correlated).** **HOLDS (with a convention
nuance worth stating).** Source: `core_sim_composition.load_concepts` (line 77) loads
`activity_level_integration_cache/denoise64_seed{N}.npz` — `obs__<word>` arrays averaged over samples, optionally
projected by a Gaussian (cosine-preserving), then mean-centered + unit-normalized. I re-measured the cosines
directly from the three cached seeds:

| readout convention | seed 42 | seed 43 | seed 44 | what cites it |
|---|---|---|---|---|
| pure activity, no centering, no projection | **0.811** (max 0.863) | — | — | the cleanup probes' "raw cos ≈ 0.81" |
| project to D=800 then center (the `load_concepts` path) | **0.820** | 0.805 | 0.799 | `2026-06-04-stage1.5-captured-code-correlation-derisk.md` ("~0.80") |
| project to D=2048 then center | 0.807 | 0.795 | 0.806 | the production operating point |
| center only at full 3200-dim (no projection) | 0.699 | 0.656 | 0.667 | the de-risk PARTIAL doc's "≈ 0.70" |

So the codes the composer faces are correlated at **~0.66–0.82 depending on the exact transform**; the headline
"≈ 0.81" is the bare-activity cosine and the production `load_concepts` path sits at **~0.80**. The de-risk PARTIAL
doc's "≈ 0.70" is the same codes after a *further* complex random phase projection (`phase = angle(W_c @ code)`,
which it documents as the FHRR-faithful map that preserves the cross-code correlation), and the Storkey doc's "0.61"
is the bipolarized-at-median version. **All consistent; the codes are genuinely, strongly correlated.** HOLDS.

**(b) The G.20 sparse-distributed codes — claimed between-cos ≈ 0.045 at 320 concepts (decorrelated).** **HOLDS.**
Source: `concept_pool_sparse_distributed.generate_sparse_patterns(n_concepts, n_pool, pattern_size, seed)`
(line 137) — each concept = `pattern_size` random indices drawn without replacement from `n_pool`. I re-measured the
substrate-pattern overlap-cosine at the 320-tier parameters (K=100, N=2000, the 64-per-bridge × 5 bridges = 320):

| bridge seed | between-cos mean | between-cos max | mean overlap |
|---|---|---|---|
| 42 | 0.0504 | 0.150 | 5.0 / 100 |
| 43 | 0.0500 | 0.130 | 5.0 / 100 |
| 44/45/46 | 0.0495–0.0504 | 0.130 | ~5.0 / 100 |

The **captured flat codes** (post-training projected activity, the actual fillers the composition uses) are reported
at **between-cos mean 0.045, max 0.604** at full 320 in `2026-06-02-full-320-flat-distinct-composition-RESOLVES-
multiseed.md` (job bh4o2reg3, seeds 42–46) — consistent with the ~0.05 substrate value. HOLDS.

> **The decisive structural contrast, made precise.** The decorrelation in scheme (b) is **NOT learned by a plasticity
> rule** — it is a *structural* property of sparse random expansion: K/N = 100/2000 = 5% sparsity, so two random
> codes overlap by ≈ K²/N ≈ 5 of 100 active bits → cos ≈ 0.05. This is exactly the catalog's **F.12 codon /
> D.12 DG pattern-separation** mechanism (expansion recoding: pattern overlap scales as (W/L)^R; <5% sparsity). This
> is the single most important fact for the pivot: **the brain decorrelates UP FRONT by sparse expansion recoding,
> not by a post-hoc cleanup rule on dense correlated codes.** The cleanup arc's three NEGATIVES were all attempts at
> the latter; the pivot is to the former.

### 1.2 The ZCA result — decorrelated codes → attractor recovers argmax parity 1.000

**HOLDS, and I extended it with a NEW, stronger result on the project's REAL decorrelated codes.** From
`2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md` (TEST 2): on the brain's raw correlated codes the learned
Hopfield attractor sits at chance (0.045); with a host ZCA whitening step (cos → 0.00) it recovers to argmax parity
**1.000**. That is the documented fact. It HOLDS — but it used a **host ZCA linear transform**, which is a shortcut
(the brain is not computing it), and the three subsequent NEGATIVES proved no *local/spiking* rule reproduces ZCA on
the correlated codes (the locality wall).

**NEW (I re-measured, host-numpy preview of the recommended de-risk — see §6):** does a vanilla/mean-field attractor
recover argmax parity on the project's REAL **decorrelated sparse** codes (cos ≈ 0.05) with **NO host ZCA**? Using
the *correct* sparse-code readout (binary {0,1} with population-mean removal — see the critical caveat below):

| codes (V=16) | matched-filter (argmax) | mean-field Hopfield attractor | seeds |
|---|---|---|---|
| **decorrelated sparse (cos 0.05)** | 16/16 | **16/16** | 42, 43, 44 (all) |
| correlated denoise64 (cos 0.70) | 16/16 | 16 / 4 / 12 (seed-variable, collapses) | 42, 43, 44 |

**On the decorrelated codes a distributed attractor recovers cleanly, 16/16 on all three seeds, with no host ZCA.**
On the correlated codes the same attractor collapses (seed-variable, fails on seed 43). **This is direct numerical
support for the pivot:** the problem was never "the attractor cleanup is impossible on-substrate"; it was "the
attractor cleanup is impossible on *correlated* codes." Move to decorrelated codes and the wall dissolves.

> **⚠️ SURPRISING METHODOLOGICAL CAVEAT — the convention the failed probes used would FALSELY sink the de-risk.**
> When I first ran the preview with the cleanup probes' own convention (**bipolarize each code at its median, ±1**),
> the sparse codes scored 1/16 = **chance** for BOTH argmax and Hopfield. The reason: a sparse 100-of-2000 code
> median-thresholded becomes ≈ −1 everywhere (only ~100 entries are +1), so all sparse codes collapse onto a huge
> shared "−1" common mode and become nearly identical (cos → ~1). **Median-bipolarization manufactures a common mode
> on sparse codes.** The decorrelation only appears under the correct readout (binary {0,1} or mean-removed). The
> de-risk MUST NOT reuse the median-bipolarize convention from `cortex_storkey_ca3_cleanup_probe.py` /
> `cortex_learned_cleanup_derisk.py` on sparse codes — it would produce a false NEGATIVE. This caveat is itself a
> deliverable: it explains part of why "clean the dense codes" kept failing (the bipolar convention is appropriate
> for dense ±-balanced codes, hostile to sparse ones), and it pins the exact readout the de-risk must use.

### 1.3 The NEF/TPAM localist cleanup — validated brain-based argmax replacement, == numpy at D=2048 multi-seed

**HOLDS, and it is genuinely brain-based — with one important characterization of what it presupposes.** From
`2026-06-05-phase1-tpam-cleanup-derisk-GO.md` + `2026-06-05-composer-cleanup-NEF-GO.md` (and verified in
`rf_phasor_composer.py:202–245` + `core_sim_composition.NEF_CLEANUP_OP` line 58): the Stewart-Tang-Eliasmith 2011
NEF thresholded cleanup (the Spaun cleanup) reaches **seed-robust numpy parity** (worst-case 0.978, mean 0.993,
seeds 42/43/44) and, wired into the composer (`enable_spiking_cleanup=True`), answers the V=320 D=2048 capability
matrix **27/27 IDENTICALLY to numpy across all three seeds**, no regression, no `sim/` edits.

The mechanism (genuinely on-substrate spiking): input-normalizing FS pool → a **matched-filter bank of `n_per`=12
neurons per concept whose encoders ARE the stored codes** → a placed negative bias so off-target concepts (cos ≈ 0)
emit ZERO spikes and the true concept fires → sum each concept's neurons' firing → argmax-over-firing. The only numpy
left is the membrane/firing *readout* (a readout of spiking output, not a computation of the match).

**What it presupposes (the load-bearing nuance):** it is a **per-concept LOCALIST readout built from a KNOWN
codebook** — one population per vocabulary entry, encoder = that entry's code. This is the cortically-plausible
"grandmother-cell-ish" recognition readout (Spaun's cleanup memory; Crawford-Gingerich-Eliasmith human-scale). It is
NOT the failed *distributed* attractor — and that is exactly why it works on correlated codes where the distributed
Hopfield collapsed: a localist matched filter is a **linear filter immune to the common mode**. The contrast is the
whole story of the cleanup arc:

| cleanup kind | on correlated codes | on decorrelated codes | status |
|---|---|---|---|
| **localist matched filter (NEF/TPAM/argmax)** | **works (immune to common mode)** | works | the production default; brain-based; GO |
| distributed Hebbian/Storkey Hopfield | collapses (common-mode / locality wall) | **works (§1.2, 16/16)** | 3× NEGATIVE on correlated; viable on decorrelated |

So "the NEF cleanup is brain-based" HOLDS; the precise statement is **"a localist codebook-indexed readout is
brain-based and already works"** — which means the cleanup is NOT the open step-3 problem (it is solved, by a
localist circuit). The open problem is the *binding* and the *codes* (§2).

### 1.4 The no-confab familiarity gate — learned Bogacz-Brown anti-Hebbian signal PASSED in the cheap-first

**HOLDS.** From `2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md` (TEST 3): a learned anti-Hebbian familiarity
pool (Bogacz-Brown = the projector onto the stored span; novelty energy N(x) = ‖x‖² − xᵀWx) separates known cues
(novelty mean 0.000) from never-imprinted cues (mean 0.991) with margin **+0.982** (seed 43: +0.985), 100%/100% at
the midpoint threshold, on the brain's **correlated** codes (anti-Hebbian's headline high-capacity-on-correlated-
inputs property). Lesion anti-cheat fires: zero the learned weights → margin → 0. It is a learned, lesionable
spiking-realizable signal.

**Why this matters for step 3:** the current no-confab moat is **host bookkeeping**, not a neuron. I verified this
directly in `rf_phasor_composer.py:305–310`: `query_agent` iterates `self.kb`, unbinds action+patient, compares with
`==`, and returns `None` only if the host `for`-loop finds no match. The cleanup `argmax` ALWAYS returns *some* word
(`_cleanup`, line 252), so abstention rides entirely on the host `if matches` logic — exactly the design's "I-3" point.
A learned binder cannot keep this for free (a learned readout always emits *some* filler), so the familiarity gate is
the load-bearing replacement, and its cheap-first PASS de-risks it. HOLDS.

### 1.5 The vocab-ceiling spec the cortex must match

**HOLDS exactly as the design states.** From `2026-06-10-vocab-ceiling-multiseed-GO.md` +
`docs/plans/2026-06-10-conversational-vocab-ceiling-characterization-design.md` + the harness
`research/runners/vocab_ceiling_probe.py`: the full `BrainConversationalAgent` capability matrix (comprehension +
who/what Q&A, abstention, negation/yes-no, embedded clause, one-attribute, two-attribute, generation, dialogue)
holds at **V=320 across 6 seeds (42–47)**, on the **merged nav+conv one-bridge substrate**. The no-confab moat is
**100% (20/20) in every cell**; the old K=5 two-attribute boundary **resolves everywhere**; the shuffled-fact control
has **zero false hits everywhere**. The **only** degradation is the embedded clause: a pure code-dimension floor —
5/6 seeds at D=128, all 6 at **D≥256**. That degradation map is the spec a learned binder inherits: it must match the
matrix while learning lossy/robust codes that do NOT demand the algebra's clean-code precision. HOLDS.

### 1.6 Summary of the load-bearing-fact audit

| fact | verdict | the real number / nuance |
|---|---|---|
| denoise64 correlated (~0.81) | **HOLDS** | 0.81 bare / 0.80 production path / 0.70 post-phase-proj / 0.61 bipolar — all correlated |
| sparse-distributed decorrelated (~0.045) | **HOLDS** | 0.050 substrate (K=100,N=2000); 0.045 captured-flat at 320; **structural sparsity, not learned** |
| ZCA → attractor parity 1.000 | **HOLDS + extended** | NEW: vanilla attractor = 16/16 on the REAL sparse codes, NO ZCA (seeds 42/43/44) |
| NEF/TPAM cleanup brain-based, == numpy | **HOLDS-WITH-NUANCE** | genuinely brain-based BUT it is a **localist codebook readout**; that is why it survives correlation and the distributed attractor did not |
| familiarity gate PASSED cheap-first | **HOLDS** | +0.982 margin, lesionable; replaces a moat that is currently host `if matches` bookkeeping |
| vocab-ceiling spec to V=320 | **HOLDS** | moat 100%, two-attr resolved; only clause needs D≥256 |

**Nothing the pivot rests on is REFUTED.** The most valuable corrections are (i) the cosine is convention-dependent
(0.66–0.82, all correlated — the "0.81" is the right order); (ii) the decorrelation in the sparse codes is
*structural sparse expansion*, not learned — so "learn the codes" more precisely means "**generate the codes by a
brain-based sparse-expansion front end**" (DG/F.12), not necessarily "train a plasticity rule to decorrelate dense
codes" (which is the locality wall); and (iii) the surprising readout caveat (§1.2) that would falsely sink the
de-risk if the wrong convention is used.

---

## 2. DIAGNOSIS — the cortex-core problem statement

**What the learned binder must achieve.** Replace the composer's exact-inverse FHRR algebra (idealization I-1: a
fixed, hand-specified, perfectly-invertible transform; I-2: it demands decorrelated full-precision codes; I-3: its
cleanup is `argmax` over a known codebook and its abstention is a host `if`) with a **learned spiking binder over
learned/decorrelated codes** that:

- **(a) matches the §1.5 vocab-ceiling spec on the MERGED one-bridge substrate** (nav + parser + dlPFC + composer
  are already consolidated — `research/runners/nav_conv_merged_bridge.py`, roadmap step 2 DONE). The full capability
  matrix at V=320, moat 100%, two-attribute resolved, clause at D≥256.
- **(b) does NOT demand the algebra's clean-code precision** — it learns lossy/robust readouts that tolerate
  correlated or grounded codes (the genuine "learned cortex reads whatever messy code arrives" property).
- **(c) dissolves the cross-code rate-vs-phasor wall** — navigation perception is a *rate* code; the composer is a
  *phasor* code. The deeper perception→memory interaction (binding a *perceived* grounded ensemble into memory) needs
  them commensurable; a learned binder that reads grounded rate codes is the bridge.
- **(d) keeps the no-confab abstention moat** as a computed spiking familiarity signal (§1.4), not host bookkeeping.

**The honest core risk: SYSTEMATICITY (Fodor-Pylyshyn).** A learned binder generalizes to never-seen role-filler
combinations only if it is systematic. The algebra has this for free (the operation is identical for any operands);
a learned readout notoriously tends to memorize trained combinations and fail on novel ones. This is the single
place step 3 can go NEGATIVE, and that negative is itself the scientific deliverable (it maps exactly where a learned
spiking cortex stops being systematic). **The ranking below is organized around how much systematicity each option
puts at risk.**

**Crucial reframe vs the prior design's framing of the problem.** The cleanup arc closing changes the diagnosis. The
prior step-3 design (`docs/plans/2026-06-10-step3-true-cortex-design.md`) treated "the learned attractor cleanup on
the brain's correlated codes" as the cheap-first de-risk. **Three NEGATIVES proved that is the wrong target.** The
corrected diagnosis: the cleanup is **already solved by a localist readout** (§1.3, the NEF/TPAM is brain-based and
works on correlated codes precisely because it is localist and common-mode-immune). The residual step-3 problem is
therefore NOT "make a distributed attractor clean up correlated codes" — it is **"get decorrelated-enough codes by a
brain-based front end (so the algebra/binder is well-conditioned), and decide whether the BINDING itself needs to be
learned (Option B/C) or whether learned CODES + the existing algebra+localist-cleanup suffice (Option A)."**

---

## 3. CATALOG + LITERATURE + EXISTING-SIM REVIEW (ground it; not designed from papers alone)

### 3.1 Project catalog (`sim-catalog/references/feature-catalog.md`, ~323 entries, clusters A–Q)

The catalog entries that bear directly on learned cortical binding + decorrelation. (No `glossary.md` exists in the
worktree; textbook notes under `references/textbooks/` were consulted for D.* and F.*.)

- **F.12 Codon representation — sparse expansion recoding via granule layer** (catalog line 1613). Marr/Albus: each
  granule cell fires only when ≥R of its 4–5 mossy-fibre claws are active; pattern overlap X scales as (W/L)^R, so
  codons separate similar inputs *geometrically*, <5% active. "The codon code is what makes a single perceptron-
  classifier viable — without expansion recoding the raw input space is too low-dimensional to be linearly
  separable." **Sim status: MISSING** — the catalog explicitly flags the missing generator
  `build_mf_gc_codon_layer(n_mf, n_gc, claws=4, target_codon_size=R)`. This is the canonical **mixed-selectivity /
  decorrelation-by-expansion** substrate and is *exactly* what a learned-code front end (and Option C's expansion
  layer) needs. **The project's G.20 sparse codes ARE a codon code in disguise** (K-of-N random expansion = a fixed
  codon layer); F.12 is its biologically-named, tunable form.
- **D.12 Pattern separation — DG sparsifies overlapping inputs** (line 1223). "Marr expansion recoding — divergence
  onto a larger sparse population orthogonalizes similar inputs." This is the brain's UP-FRONT decorrelation — the
  pivot's mechanism. The three cleanup NEGATIVES tried to *spike-implement* DG separation as a cleanup front end and
  hit the sub-reproducibility wall (the project's stock DG fires ~15 spikes/600 neurons → noise-dominated). The
  catalog's D.12 is the *function*; the project's sparse-random codes already realise the function structurally
  (cos 0.05) without needing the unreliable spiking-DG read.
- **D.13 Pattern completion — CA3 recurrents reconstruct full pattern from partial cue** (line 1235). "Trade-off with
  separation: too much completion → confused episodes; too little → no generalization." **This is the exact
  separation-vs-reproducibility tension the rate-k-WTA NEGATIVE found** (`2026-06-11-cortex-dg-ratekwta-cleanup-
  NEGATIVE.md`: repro ≈ sep at every k). The catalog names it as a fundamental trade-off, not a tuning miss — which
  is why the spiking-DG path is correctly BENCHED. The O&N supplemental note (line 1246) adds: CA3 completion is
  *sequential* (theta-paced), a different sim target from a Hopfield point attractor.
- **A.12 Sparse, decorrelated cortico-striatal convergence — Kincaid wiring rule** (line 224). "A decorrelation rule
  — cortical drive to neighbouring MSNs is statistically independent... the 1–2 contacts per axon + non-overlapping
  neighbour rule is the substrate for ensemble decorrelation." A *second* place the catalog says biology decorrelates
  by **sparse wiring** (structural), not by a learned post-hoc rule — reinforcing the pivot.
- **D.02 Relational binding / "memory space" (Eichenbaum-Cohen)** (line 1098) + **D.21 Cognitive-map theory**
  (line 1041, "episodic binding"). The hippocampal relational-binding framework — items-in-context, flexible
  (transitive) inference. The conversational binder is functionally a relational-memory store; D.02 is the
  biological frame. **Sim status: MISSING** (no relational-binding primitive). The composer is the project's de-facto
  relational store; step 3 is its learned-cortical version.
- **G.03 Object-based attention & feature binding** (line 2603) + **N.19 Gamma binding-by-synchrony — ING vs PING**
  (line 1028). The two non-VSA biological theories of binding: feature binding via attention, and binding-by-
  synchrony (gamma phase). Relevant as *alternative* binding mechanisms — but neither is exact-invertible nor scales
  to the recursive-clause / two-attribute capability the spec requires; the project's VSA route is the pragmatic one.
- **D.14 Engram cells (Tonegawa)** (line 1248). Activity-tagged ensembles stored + reactivated as a unit. SHIPPED on
  the bridge (the engram-tag API). Reusable as a **codebook-free** store (store "the pattern that fired") — the
  mechanism that lets a *perceived* grounded ensemble be stored without a phasor code (the §2(c) cross-code bridge).

**Catalog synthesis:** the catalog is emphatic and consistent — **biology decorrelates by sparse expansion recoding
UP FRONT (F.12 codon, D.12 DG, A.12 cortico-striatal sparse wiring), and cleans up by a localist/attractor completion
stage (D.13 CA3) whose separation/completion trade-off is fundamental.** This is precisely the pivot. The cleanup arc
failed because it inverted the order (dense correlated codes first, decorrelate-as-cleanup second); the brain does
sparse-expansion first.

### 3.2 Existing biology-grounded sims (check what's already implemented — CLAUDE.md "check existing sims FIRST")

- **Spaun / Semantic Pointer Architecture (Eliasmith, Nengo).** The direct ancestor of the project's composer. Its
  cleanup is the **Stewart-Tang-Eliasmith auto-associative memory built from a learned vocabulary**, which
  "significantly outperforms linear associators, direct function approximators and MLPs in accuracy AND scalability"
  — i.e. a **localist codebook-indexed cleanup**, which is exactly the NEF/TPAM the project already adopted (§1.3).
  **Adopt-not-reinvent: the project's cleanup IS the Spaun cleanup; it is solved.**
- **Crawford-Gingerich-Eliasmith 2016, "Biologically Plausible, Human-scale Knowledge Representation"** (Cognitive
  Science). Learned, robust VSA-on-spikes at *human-scale* vocabulary (100k+ terms); the load-bearing piece is the
  associative cleanup memory. Evidence the VSA-on-spikes route scales far past V=320 — the project's spec is modest
  by this standard. **The project's `enable_substrate_store` (Crawford-Eliasmith per-fact weight store) already
  imports this.**
- **Gosmann-Eliasmith 2019, VTB** (Neural Computation; verified via web search). A **learned/derived linear-transform
  binding** for spiking VSAs, **on-par-to-better than circular convolution and more dimension-efficient** (its
  headline). Directly = **Option B**. Relevant because the spec's one dimension-floor (embedded clause needs D≥256)
  is the kind of thing a dimension-efficient bind could lift.
- **Mikulasch-Priesemann 2021, "Local dendritic balance enables learning of efficient representations in spiking
  neurons"** (PubMed 34876505; verified via web search). Decorrelation in spiking nets requires **voltage-dependent /
  dendritic plasticity with recurrent decorrelation, NOT pairwise Hebbian** ("pairwise Hebbian works only under
  unrealistic requirements"). This is the *mechanistic explanation of the project's locality wall* (the Storkey
  NEGATIVE): a local pairwise rule provably cannot decorrelate correlated codes. It is already in CLAUDE.md as the
  "whitening is analog/pre-spike in biology" reframe. **Takeaway: do NOT try to fix the dense-correlated-code
  cleanup with a better local rule — that path is closed by theory; decorrelate up front (sparse expansion) or learn
  with a richer (dendritic / gradient) rule.**
- **Learned-binding systematicity literature** (web search): Training NNs to encode symbols (VARS, Royal Society
  2020) and **Tensor Product Representations** (TPR) *can* answer novel role-filler queries ("who is the lover?" for
  a filler never seen in that role) — but in the bespoke-training / meta-learning regime (Lake-Baroni Nature 2023),
  which is contested ("Fodor and Pylyshyn's Legacy — Still No Human-like Systematic Compositionality," arXiv
  2506.01820, 2025). **Net: a learned binder CAN be systematic, but only with deliberate structure/meta-training; a
  vanilla learned spiking readout should not be assumed systematic — it must be measured (the §6 anti-cheat).**

**Existing-sim synthesis:** the project is already standing on the right shoulders (Spaun cleanup, Crawford-Eliasmith
store, the FHRR phasor substrate). The two genuinely-new pieces the literature offers are **VTB (a learned,
dimension-efficient binding = Option B)** and the **Mikulasch-Priesemann verdict that local decorrelation needs
dendritic/voltage plasticity** (which both explains the locality wall and points at the sparse-expansion alternative).

---

## 4. RANKED, BIOLOGICALLY-GROUNDED OPTIONS

Refining the step-3 design's A/B/C (`docs/plans/2026-06-10-step3-true-cortex-design.md` §2). **The cleanup-arc
closure sharpens the ranking decisively toward Option A**, because the cleanup is now known to be solved (localist),
and the only thing Option A needs that the dense codes lacked — decorrelated codes — the project ALREADY HAS
(the G.20 sparse codes, cos 0.05, §1.1b) and they ALREADY compose at full 320 (§1.5 / the flat-distinct result).

### Option A (RECOMMEND FIRST) — keep the FHRR bind/unbind ops; supply DECORRELATED codes by a brain-based sparse-expansion front end; cleanup via the localist NEF readout (already validated)

- **Mechanism:** the bind/unbind *operation* stays the FHRR phasor product (already on-substrate spiking, validated
  to 320). The "learned cortex" content is the **code front end**: concept codes are the project's **sparse-
  distributed (codon/DG) codes** (cos 0.05) — decorrelated *by construction* (structural sparse expansion = F.12/
  D.12), not by a fragile post-hoc rule. Cleanup is the **localist NEF/TPAM** readout (§1.3), which is brain-based and
  already == numpy at 320. The no-confab moat becomes the **learned familiarity gate** (§1.4).
- **What it reuses:** the RF phasor substrate (`NeuronModel.RESONATE_AND_FIRE` + complex synapses), the validated
  `concept_pool_sparse_distributed` code generation, the validated flat-distinct 320 composition
  (`2026-06-02-full-320-flat-distinct...`), the NEF cleanup (`enable_spiking_cleanup`), the substrate fact store
  (`enable_substrate_store`), the familiarity gate (de-risked), and the vocab_ceiling_probe harness verbatim.
- **Systematicity exposure: LOW.** The binding is the exact-inverse FHRR operation — identical for every operand → it
  inherits the algebra's free systematicity (the PARTIAL de-risk already showed held-out novel combos at 1.000 under
  Option A). The §1.5 flat-distinct ANY-BANK result (any concept in any role, 0.992 mean 6-seed) is direct evidence
  the sparse-coded FHRR composes systematically over never-co-occurring fillers.
- **Cleanup + moat:** cleanup solved (localist NEF). Moat = familiarity gate (computed spiking abstention, lesionable)
  — the one genuinely-new build, shared by all options.
- **Cost: LOW** (everything is validated machinery; the new work is wiring the sparse codes into the composer's code
  slot + building the familiarity gate to the 20/20 bar). **Plausibly hits the full V=320 spec on the first try.**
- **Honest scope:** this is a *hybrid* — the binding OPERATION is still the fixed exact-inverse algebra (I-1
  unremoved); only the CODES (now decorrelated-by-construction, grounded in the substrate's own sparse activity) and
  the cleanup/abstention are "learned/brain-based." An owner who wants the *binding itself* learned will see A as
  partial (correctly). **But it is the honest, lowest-risk step that removes the load-bearing clean-code demand (I-2)
  and the host-`argmax`/host-`if` (I-3), and it dissolves the cross-code wall's cleanup half** — and per §1, the
  decorrelated codes it relies on are real and already compose at 320.

### Option B — a LEARNED transformation-binding (VTB, Gosmann-Eliasmith 2019)

- **Mechanism:** replace the fixed phasor product with a **learned/derived linear-transform** bind (per role), unbind
  = the derived inverse transform. Still needs Option A's decorrelated codes + cleanup.
- **What it reuses:** an NEF/plastic `RegionPathway` (the transform = a weight matrix between two ensembles), or the
  BPTT cortex trained to implement the transform; Option A's code front end + cleanup + moat.
- **Systematicity exposure: LOW.** VTB stays transform-shaped (approximately invertible) → it inherits the algebra's
  systematicity gift, so it does NOT actually test whether a *genuinely-learned readout* generalizes (Option C does).
  It is "learned but still algebra-shaped."
- **Cleanup + moat:** same as A.
- **Cost: MODERATE.** Its headline dimension-efficiency could lift the embedded-clause D-floor (the spec's one
  degradation). A real step beyond A (the binding is now learned, not hand-specified) at moderate cost.

### Option C — a FULLY LEARNED mixed-selectivity binder + learned readout (the BPTT cortex)

- **Mechanism:** a high-dimensional **mixed-selectivity expansion** (neurons tuned to nonlinear role×filler
  conjunctions; F.12 codon / Rigotti-Fusi) → a **learned readout** (surrogate-grad BPTT, or a local three-factor
  rule) that maps "(role cue, composite) → filler." No algebra assumed; the bind is whatever the expansion+readout
  learns (the conjunctive population state IS the bound representation — the most cortex-like: there is no separate
  bind operator).
- **What it reuses:** `sim/bptt_snn_gpu.py` as the learned readout; the F.12 expansion (`build_mf_gc_codon_layer`,
  the catalog-flagged missing generator — a few lines of fixed sparse connectivity, not a `sim/` edit); the code
  front end + cleanup + moat.
- **Systematicity exposure: HIGH — this is the core risk.** A learned readout will tend to memorize trained
  combinations and fail on novel ones. This is where the arc can go NEGATIVE, and that negative is the deliverable
  (it maps where a learned spiking cortex stops being systematic). Per the literature (§3.2), systematic learned
  binding needs deliberate structure/meta-training (Lake-Baroni, contested); a vanilla BPTT readout should not be
  assumed systematic.
- **Cleanup + moat:** same as A (cleanup is localist; moat is the familiarity gate). The learned part is the binding.
- **Cost: HIGH** (training a cortex to V=320 across roles is the heaviest GPU item). Recommend C **only after** A/B,
  and **gate it on the toy-scale systematicity probe FIRST** (§6) so the likely negative is found in minutes.

### Ranking summary

| | Learns | Bind/Unbind | Removes | Cost | Systematicity risk | cleanup | moat |
|---|---|---|---|---|---|---|---|
| **A (first)** | codes (sparse front end) | fixed phasor + localist cleanup | I-2, I-3 | **low** (all validated machinery) | **low** | localist NEF (solved) | familiarity gate |
| **B** | the binding transform (VTB) | learned transform + localist cleanup | partial I-1, I-2, I-3 | moderate | low (transform-shaped) | localist NEF | familiarity gate |
| **C** | everything (mixed-sel + readout) | learned joint code + readout | I-1, I-2, I-3 | high | **high — the core risk** | localist NEF | familiarity gate |

**Recommendation: A first** (it removes the load-bearing clean-code demand on existing, validated machinery — the
decorrelated codes are real and already compose at 320 — and dissolves the cleanup half of the cross-code wall),
**then B** (learned dimension-efficient binding, may lift the clause floor), with **C as the explicitly-gated
systematicity experiment** whose honest negative maps the learned-cortex limit. Every option shares the familiarity
gate + the localist cleanup, so build those once (they are the reusable core).

---

## 5. REUSABLE PROJECT MACHINERY (concrete, with file paths — all verified present)

- **The RF phasor substrate** — `NeuronModel.RESONATE_AND_FIRE` (confirmed in `sim/enums.py`) + complex synapses
  `rf_kick`, `rf_set_complex_weights`, `rf_resonate_steps`, `rf_read_phases` on `sim/bridge.py`; the masked-RF-ops
  edit (`rf_kick(neuron_mask=)`, owner-approved, default-off byte-identical) lets the binder co-reside on a shared
  bridge. The spiking bind/unbind for Options A/B.
- **The BPTT spiking cortex** — `sim/bptt_snn.py`, `sim/bptt_snn_gpu.py` (CuPy, validated == numpy), `sim/
  surrogate_grad.py`, `sim/char_tokenizer.py`, `research/runners/cortex_pretraining.py`. **Verified all on `main`**
  (`git ls-tree main`); tests `tests/test_bptt_snn.py`, `tests/test_surrogate_grad.py`, `tests/test_char_tokenizer.py`
  present (their "path-f-hybrid only" headers are stale). The learned readout for Option C.
- **The sparse-distributed / concept-pool code generation** — `research/runners/concept_pool_sparse_distributed.py`
  (`generate_sparse_patterns`, `build_sparse_pool_bridge`, `apply_sparse_topographic_prior`); the curated 320 vocab
  `research/runners/g20_vocab_spec_320.py`; the validated 320 flat-distinct composition codes (5 bridges, seeds
  42–46). The decorrelated-code front end for Option A.
- **The validated learned sparse recurrent heteroassociative memory** — `research/runners/_D_sparse_heteroassoc.py`
  (`2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md`): a **genuinely-learned, anti-cheat-clean (permuted
  control passes), multi-seed** cue→associate completion on sparse codes, ON the bridge. This is the strongest
  evidence a *distributed* spiking attractor WORKS on decorrelated sparse codes (the §1.2 result, but on the real
  bridge with learned weights) — directly reusable as Option A's optional distributed-completion cleanup and as the
  perception→memory association substrate.
- **The NEF/TPAM localist cleanup** — `research/runners/core_sim_composition.py` (`NEF_CLEANUP_OP`, line 58,
  `enable_spiking_cleanup`), `research/runners/rf_phasor_composer.py` (`_spiking_cleanup`, line 202),
  `research/runners/resonate_fire_fhrr.py::ResonateFireTPAM` (numpy ref). The cleanup — solved.
- **The substrate fact store** — `rf_phasor_composer._store_substrate` (`enable_substrate_store`, line 276), the
  Crawford-Eliasmith per-fact complex-weight store. Removes the numpy-held-fact shortcut.
- **The engram-tag API** — `bridge.start_engram_recording` / `commit_engram_tag` / `stimulate_tag` (the D.14 store);
  the codebook-free store for perceived grounded ensembles (the cross-code bridge).
- **The familiarity gate prototype** — the anti-Hebbian novelty pool in
  `research/runners/cortex_learned_cleanup_derisk.py` (TEST 3, de-risked +0.982 margin). The no-confab replacement.
- **The capability spec + probe harness** — `research/runners/vocab_ceiling_probe.py` (full matrix, pass/fail, with
  abstention floor + shuffled-fact control per cell). Step 3 must pass THIS verbatim.
- **The merged one-bridge builder** — `research/runners/nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge`,
  `MergedNavConvAgent`, `MergedRFComposer`). The substrate the learned binder must run on.
- **The composer interface** — `BrainConversationalAgent` / `BridgeParser`
  (`research/runners/brain_conversational_agent.py`) delegate storage/retrieval to a composer object
  (`store`, `query_agent`, `query_patient`, `ask_yes_no`, `render_fact`, `elaborate`). A learned binder implementing
  this interface is a drop-in — step 3 is "write a new composer class," not "rewire the agent."

---

## 6. THE SINGLE CHEAPEST-FIRST DE-RISK TO RUN NEXT (CPU-first) + its anti-cheats

**Recommended probe: the attractor-cleanup POSITIVE CONTROL on the project's REAL decorrelated sparse-distributed
codes (cos ≈ 0.05), on the bridge, with NO host ZCA** — plus a minimal learned-binding-systematicity feasibility test
in the same run (both are cheap). This directly confirms the pivot's load-bearing claim: **decorrelated codes dissolve
the locality wall**, so Option A's cleanup is brain-based-viable, where the dense-correlated-code cleanup was 3×
NEGATIVE.

### Probe spec

**Codes / harness to reuse:**
- Decorrelated codes: `concept_pool_sparse_distributed.generate_sparse_patterns(V, n_pool=2000, pattern_size=100,
  seed)` at V=16 (to compare head-to-head with the denoise64 NEGATIVE) and V=64. These are the REAL codes the 320
  composition uses.
- Correlated contrast (the documented NEGATIVE baseline): the `denoise64` cache via
  `core_sim_composition.load_concepts`. Run BOTH so the probe shows the decorrelated codes pass where the correlated
  ones fail — a within-probe positive/negative control.
- The on-bridge distributed attractor: reuse `research/runners/_D_sparse_heteroassoc.py`'s sparse recurrent
  heteroassociative memory (already validated, learned, anti-cheat-clean) as the cleanup attractor — OR the simplest
  CPU mean-field Hopfield for the first pass (the §1.2 preview already shows 16/16 on sparse codes, numpy). Run on
  `SIM_BACKEND=numpy`, CPU, minutes.

**⚠️ The load-bearing methodological constraint (from §1.2, the surprising caveat):** read the sparse codes in their
**native binary {0,1} / mean-removed** form. **Do NOT bipolarize-at-median** (the convention in
`cortex_storkey_ca3_cleanup_probe.py` / `cortex_learned_cleanup_derisk.py`) — on sparse codes that manufactures a
common mode and produces a FALSE NEGATIVE (I observed exactly this: 1/16 = chance under median-bipolarize vs 16/16
under the correct readout). The probe must assert this convention explicitly and include a unit check that the
between-code cosine of the codes-as-read is ≈ 0.05 (not ≈ 1.0).

**Gates to proceed (Option A GO):**
1. **The distributed attractor recovers ≥ argmax parity on the decorrelated codes** (target: ≈ 1.000 at V=16/64,
   multi-seed 42/43/44), with NO host ZCA — confirming the pivot.
2. **The same attractor collapses on the correlated denoise64 codes** (the documented NEGATIVE reproduced) —
   confirming the difference is the codes, not the mechanism.
3. **The full 320 sparse composition still passes the vocab-ceiling matrix** (this is already validated —
   `2026-06-02-full-320-flat-distinct...` + `2026-06-10-vocab-ceiling-multiseed-GO.md` — so it is a re-confirmation,
   not new work; cite it rather than re-run unless integrating the localist cleanup changes it).

**The minimal learned-binding-systematicity feasibility test (same run, cheap — the real Fodor-Pylyshyn control):**
on a tiny role-filler grid (agents × actions × patients, e.g. 4×4×4) with the decorrelated sparse codes, **train/
imprint binding on a SUBSET of role-filler combinations, HOLD OUT a never-trained combination whose parts were each
trained, then store + unbind the held-out triple.** For Option A (fixed FHRR op) this should pass trivially at 1.000
(the op is operand-independent) — if it does NOT, that is a bug, not a science result. The value is that this is the
**ready, validated probe to run on Option C BEFORE any GPU train**: if a learned (BPTT) readout fails the held-out
combination at toy scale, it fails at full scale → report the systematicity NEGATIVE and stop the C build.

### Anti-cheats (mandatory)

1. **SYSTEMATICITY = held-out-novel-combo** (the deepest control): train binding on a subset of role-filler combos,
   test unbind on **never-seen** combos. Score fraction-correct vs chance vs the algebra (100% by construction). This
   is the control that exposes Fodor-Pylyshyn and the one Option C must pass to be more than a lookup table.
2. **The no-confab MOAT = abstention floor**: unstored cues must return "I don't know." With the familiarity gate,
   present known cues (must read familiar → answer) and a disjoint set of never-stored cues (must read novel →
   abstain); require a clean threshold (max-known < min-unknown). The §1.4 de-risk already shows +0.982 margin on
   correlated codes; on decorrelated codes it should be at least as clean.
3. **Shuffled-fact permuted control**: re-query who/what with a RANDOM permutation of (agent, action)→patient
   pairings; correct answers must collapse to chance (catches a system that echoes the most-recent/frequent filler).
   The `_D_sparse_heteroassoc.py` permuted-encoding anti-cheat (completion follows the encoding, not a fixed
   structure) is the template.
4. **Lesion**: zero the attractor's recurrent weights (or the familiarity pool's weights) → cleanup/abstention
   collapses → confirms the behaviour rides the learned weights, not a host path or leftover structure.
5. **Provenance / brain-based audit**: bind/unbind/cleanup/familiarity must be neuron firing + synaptic current on
   the bridge; host legitimate only for presenting the sentence (environment) and reading the final answer. (The
   localist cleanup's only numpy is the firing readout, accepted; the moat's host `if matches` is exactly what the
   familiarity gate replaces.)

**Why this probe is the right cheapest-first move:** it is CPU-minutes, it reuses entirely-validated machinery
(`generate_sparse_patterns` + `_D_sparse_heteroassoc.py`), it produces a clean within-probe positive/negative control
(decorrelated passes, correlated fails) that *directly confirms or refutes the pivot*, and it carries the
systematicity + moat anti-cheats that gate the entire step-3 arc. If gate 1 passes (and §1.2 already strongly predicts
it will), **Option A is GO** → wire the sparse codes + localist cleanup + familiarity gate into a new composer class
implementing the `BrainConversationalAgent` interface, and run `vocab_ceiling_probe` verbatim at V=320 multi-seed on
GPU. If gate 1 somehow fails on the bridge (it passes in numpy), that localises a substrate-vs-numpy gap to chase
before any heavy build.

---

## 7. Honest framing / what could still go NEGATIVE

- **Option A is a hybrid, not "the learned cortex" in full.** It keeps the exact-inverse FHRR operation (I-1). It is
  the honest *minimum* that removes the clean-code demand (I-2) and the host cleanup/abstention (I-3) on validated
  machinery. The genuinely-learned *binding* is Option B/C, and Option C is where the real systematicity question
  lives. Calling A "the true cortex" would over-claim; calling it "the learned-code + brain-based-cleanup + computed-
  abstention front end that removes the load-bearing idealizations cheaply" is honest.
- **The decorrelation is structural, not a learned plasticity rule.** The sparse codes are decorrelated by sparse
  random expansion (F.12 codon), not by a learning rule that *discovers* decorrelation. That is biologically
  legitimate (DG/cortico-striatal sparse wiring IS structural), but if the owner specifically wants the *codes
  learned end-to-end* (a plasticity rule that produces decorrelated codes from grounded input), that is a deeper
  build — and the Mikulasch-Priesemann verdict says it needs dendritic/voltage plasticity, not pairwise Hebbian
  (a known, larger arc). The honest position: the sparse codes are GIVEN by encoding (cheating-audit per the
  flat-distinct doc); the composition on top is genuine and robust at 320.
- **Option C's systematicity is the live risk** and the literature is blunt that vanilla learned binders are not
  reliably systematic. The disciplined move (the §6 toy probe FIRST) finds that negative in minutes if it exists.
- **The cross-code (rate↔phasor) perception→memory wall is only half-addressed by A.** A's localist cleanup +
  engram-tag store make a *perceived grounded ensemble* storable, but binding a rate-coded percept into the phasor
  composer end-to-end is the genuinely-open functional-integration payoff (step 3 §8 step 6), deferred to after A
  passes the spec.

---

## VERDICT

The pivot **HOLDS** on every load-bearing fact (with the cosine-convention nuance and the structural-vs-learned
decorrelation nuance stated honestly), and I added direct numerical support: **a distributed attractor recovers
argmax parity 16/16 on the project's REAL decorrelated sparse codes (cos 0.05) with NO host ZCA, while collapsing on
the correlated denoise64 codes** — the cleanup arc's three NEGATIVES were a wrong-codes problem, not an impossible-
mechanism problem. The cleanup is already solved by a localist NEF readout (Spaun's cleanup; brain-based; == numpy at
320). The recommended path is **Option A** — keep the validated FHRR ops, supply decorrelated codes by the brain-based
sparse-expansion front end the project already has, cleanup via the localist NEF, and make abstention a computed
spiking familiarity signal — with **B** (learned VTB binding) and **C** (fully-learned, gated on the toy
systematicity probe) staged after. The single cheapest-first de-risk is the **attractor positive-control on the real
sparse codes (CPU, minutes, reusing `generate_sparse_patterns` + `_D_sparse_heteroassoc.py`)** with the
held-out-novel-combo + abstention-floor + shuffled-fact + lesion anti-cheats — and it MUST read the sparse codes in
their native binary/mean-removed form, never median-bipolarized (the surprising caveat that would otherwise produce a
false NEGATIVE).

**No banking** — reported exactly as found.

## Sources (literature consulted beyond the in-repo catalog/code)

- Gosmann & Eliasmith, *Vector-Derived Transformation Binding* (Neural Computation, 2019;
  https://compneuro.uwaterloo.ca/files/publications/gosmann.2019b.pdf) — learned, dimension-efficient binding for
  spiking VSAs (Option B).
- Crawford, Gingerich, Eliasmith, *Biologically Plausible, Human-scale Knowledge Representation* (Cognitive Science,
  2016; https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12261) — learned VSA-on-spikes at human scale; the
  load-bearing piece is the associative cleanup.
- Eliasmith et al., *Spaun / Semantic Pointer Architecture* (the Stewart-Tang-Eliasmith auto-associative cleanup;
  https://en.wikipedia.org/wiki/Spaun_(Semantic_Pointer_Architecture_Unified_Network),
  https://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html) — the localist cleanup the project
  already adopted.
- Mikulasch, Leugering, Priesemann, *Local dendritic balance enables learning of efficient representations in
  networks of spiking neurons* (PNAS/PubMed 34876505, 2021) — decorrelation needs dendritic/voltage plasticity, not
  pairwise Hebbian; the mechanistic explanation of the locality wall.
- Webb et al. / VARS, *Training neural networks to encode symbols enables combinatorial generalization* (Phil. Trans.
  R. Soc. B, 2020; https://arxiv.org/pdf/1903.12354) + Tensor Product Representations — learned binders CAN do
  novel role-filler queries, in the bespoke-training regime.
- *Fodor and Pylyshyn's Legacy — Still No Human-like Systematic Compositionality in Neural Networks* (arXiv
  2506.01820, 2025) — the live systematicity challenge (§2, §6 core risk).
- Lake & Baroni, *Human-like systematic generalization through a meta-learning neural network* (Nature, 2023;
  https://www.nature.com/articles/s41586-023-06668-3) — the contested counter-claim that learned nets can be
  systematic with the right meta-training.
