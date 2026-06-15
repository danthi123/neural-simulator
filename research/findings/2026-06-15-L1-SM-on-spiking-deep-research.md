# L1 similarity-matching on the spiking bridge — deep-research scope (CYCLE 76 roadblock)

**Date:** 2026-06-15. **Type:** READ-ONLY deep-research + reference-catalog review at a roadblock (per the
standing directive). NO `sim/` edits, NO GPU jobs. **Deliverable:** diagnosis → ranked options → verdict +
cheap-first de-risk + anti-cheat → reusable machinery.

**The roadblock (verbatim from AUTONOMOUS_STATE CYCLE 76):** the whitening front-end is built + validated on
the bridge — per-feature mean-centering (`enable_input_mean_adapt`, shipped) + a signed projection via E/I
balance (`--enable-ei`, escapes the excitatory-only collapse, +0.155 on a FROZEN random projection toward
host +0.44). But when the hub→cortex projection is THAWED to LEARN (`--learn-projection`), the bridge's plain
Hebbian STDP does NOT realize the L1 similarity-matching learning. Three fixes, each monotonically better,
none beating even the random projection: naive STDP (+0.060, E/I unbalanced→collapse) → E/I-plastic (+0.092,
eff-rank collapses to **1.5** = Hebbian rank-1 collapse) → tight STDP soft-bound (+0.131, eff-rank 4.8) — STILL
below random +0.155, far below the numpy L1 ceiling +0.48.

---

## 1. DIAGNOSIS

### 1a. Why plain feedforward STDP fails: it optimizes the wrong objective; the SM rule's missing piece is a recurrent anti-Hebbian lateral

The L1 learned cortex is **online similarity-matching** (Pehlevan–Chklovskii): minimize
‖XᵀX − YᵀY‖²_F over the codes Y — make output similarities match input similarities, with k<H so it is a
low-rank (denoising) similarity-preserving embedding. The canonical result
([Pehlevan, Sengupta & Chklovskii 2018, *Neural Computation* 30:84, "Why do similarity matching objectives
lead to Hebbian/anti-Hebbian networks?"](https://direct.mit.edu/neco/article/30/1/84/8348/);
[Pehlevan & Chklovskii 2015, arXiv:1511.09468, "Optimization theory of Hebbian/anti-Hebbian networks for PCA
and whitening"](https://arxiv.org/abs/1511.09468)) is structural and load-bearing here:

> the SM/whitening objective decomposes, via a min-max (saddle) formulation, into a **feedforward Hebbian W**
> (excitation, `ΔW ∝ y xᵀ`) **AND a recurrent anti-Hebbian lateral M** (inhibition, `ΔM ∝ y yᵀ − …`). The
> output is the *settled* state of the recurrent dynamics **y = (I + M)⁻¹ W x** (in practice
> `y ← y + η(Wx − My)` iterated). The anti-Hebbian M **decorrelates** the outputs — it is the term that turns
> a pile of feedforward Hebbian units (which all chase the top principal component) into a *distributed*
> subspace/whitening code.

Plain feedforward STDP is **only the Hebbian W half** with no M. Two consequences, both observed on the bridge:

- **Rank-1 / Hebbian collapse.** With no decorrelating lateral, every output neuron's weight vector is pulled
  toward the **same** dominant direction (the top eigenvector of the input covariance). The bridge's
  E/I-plastic thaw measured **eff-rank 1.5** — textbook rank-1 collapse. Tightening the STDP soft-bound
  (the CLAUDE.md soft-bound gotcha, here used to saturate LTP and *spread* the learning) raised eff-rank to
  4.8 but cannot install the *coordinated* decorrelation M provides — so it plateaus below random.
- **It is not the SM objective at all.** Potentiate-co-active STDP greedily maximizes output for co-active
  inputs; SM minimizes a *similarity-reconstruction* error. They share the Hebbian outer-product but differ in
  the normalization/competition that the `−My` lateral (or the Oja `−y²W` self-normalization) supplies.

This is the canonical **Földiák 1990** result too
([*Biol. Cybern.*, "Forming sparse representations by local anti-Hebbian learning"](https://link.springer.com/article/10.1007/BF02331346)):
a layer of Hebbian units connected by **modifiable anti-Hebbian feedback** reduces statistical dependency
between the code elements (decorrelates) while preserving information. The decorrelating lateral is the piece
plain feedforward Hebbian/STDP structurally lacks.

### 1b. Resolving the CYCLE-58 contradiction ("bounded-Hebbian, NO lateral, reached +0.545 in numpy")

**This is the single most important clarification, and it is NOT a contradiction — the numpy "no-lateral" run
was never truly no-lateral / no-normalization, and it ran on a regime the bridge does not (yet) reproduce.**
Reading the actual de-risk code:

1. **`_l1_oja_validated.py` ("Oja, no explicit M matrix") is NOT lateral-free.** `oja_subspace`
   (`research/runners/_l1_centered_online_pca_probe.py:41`) is **Oja's symmetric SUBSPACE rule**
   `dW = lr·(y xᵀ − y yᵀ W)`. The runner's own docstring calls it "local Hebbian feedforward **+ symmetric
   lateral decay**." The `− y yᵀ W` term **is** the decorrelation — it is the implicit/algebraic equivalent of
   the explicit recurrent M (Oja's subspace rule provably converges to the top-k *principal subspace*, which
   is exactly what the SM lateral computes). So "no lateral" was a naming artifact: the decorrelation is folded
   into the weight update instead of a separate matrix. The bridge's plain STDP has **neither** the explicit M
   **nor** the Oja `−y²W` self-normalizing term — which is precisely why it rank-1 collapses where Oja does not.

2. **The "bounded-Hebbian (no lateral) +0.545" claim is the Phase-A end-to-end *spiking* arm**
   (`_l1_phaseA_end_to_end_spiking.py:146`), where `bounded_Hebbian = saturating=True` →
   `W_ff += lr·outer(y_spk, x_in); clip(W_ff, −5, 5)` — genuinely no lateral, no Oja term, just a hard clip.
   It reached the ceiling, but **only because of three enabling conditions the bridge thaw lacks**:
   - the input is **CENTERED PPMI, unit-normalized per row** (`Xc = center_cols(ppmi); x/‖x‖`) — common mode
     removed AND every input vector renormalized to unit length;
   - **the input is re-normalized to its own norm every presentation** (`x_in / ‖x_in‖`) — an *output-side*
     magnitude control that the bridge does not do;
   - **the readout is window-integrated** (`acc / read_window`), and the output is non-negative (rectified)
     with **subtractive-inhibition centering of the input** as the only "competition."
   The runner *flags this exact possibility*: "the spiking build can likely DROP the lateral … use
   **subtractive-inhibition centering + a homeostatically-bounded Hebbian feedforward** instead — a SIMPLER
   build." That is a hypothesis that the *centering + normalization* can substitute for the lateral **when the
   input is fully whitened and the codes are kept bounded** — NOT a claim that raw potentiate-co-active STDP
   realizes SM.

3. **Why the bridge needs the lateral when that numpy run didn't.** The bridge thaw differs from the numpy
   "no-lateral" arm in the two missing magnitude controls: (i) **no per-presentation output/code normalization**
   (Oja's `−y²W` or the explicit `‖x‖` renorm) — so STDP grows unboundedly into the top component (rank-1
   collapse), where the tight soft-bound is a crude, *uncoordinated* substitute; (ii) the **E/I signed spiking
   projection is lossy** — the numpy centering reaches +0.31 but the bridge E/I realization only +0.155, so the
   centered signal arriving at the cortex is already half-degraded, leaving less for any learner. Net: the
   numpy run got away without an explicit lateral because it had **strong input whitening + per-step output
   normalization**; the bridge has *partial* input whitening and **no** output normalization, so it needs the
   decorrelating lateral to do the job the normalization did in numpy.

**⇒ The honest resolution:** there is no real contradiction. Every numpy run that "worked without a lateral"
either *had* the implicit lateral (Oja `−y²W`) or substituted **per-presentation output normalization +
fully-whitened input** for it. The bridge's plain STDP has neither the lateral nor the normalization, so on the
bridge the decorrelation must be supplied **explicitly** — by a recurrent anti-Hebbian lateral on the cortex.
This is exactly what the SM theory predicts and what `graded_lateral` already implements.

---

## 2. RANKED OPTIONS (cheapest-first) to realize the SM competition on the bridge

### (a) **Reuse `graded_lateral` as the recurrent anti-Hebbian lateral on the CORTEX region.** ★ RECOMMENDED LEAD.

- **Biology + citation.** `graded_lateral` (`sim/bridge.py:1776–1884`, `sim/regions.py:176–188`,
  `sim/config.py:381–398`) is **already the Pehlevan–Chklovskii recurrent anti-Hebbian lateral**, built for
  the retina/LGN decorrelation stage:
  - it learns a per-region dense plastic **M (K×K)** with `ΔM = lr·(⟨aaᵀ⟩ − I) − λM` — the **identical**
    anti-Hebbian co-activity update the SM derivation produces (`y yᵀ` target, `−λM` decay → bounded fixed
    point), symmetrized, diagonal held at 0 (Pehlevan 2015 / Földiák 1990);
  - it adds the recurrent inhibition **`−(M @ a)·gain` to the input current BEFORE the spike threshold** — the
    feedforward-inhibition realization of the settled output `y = Wx − My`;
  - **critically, it acts on SUB-THRESHOLD ANALOG activity `a = clip((v−rest)/scale, 0, 1)`, NOT spikes**
    (`bridge.py:1827–1841`). This is the decisive property: the whole arc's wall is that **rate-coded
    point-neuron output cannot do analog decorrelation** (Mikulasch–Priesemann). `graded_lateral` does the
    decorrelation in the **analog membrane domain**, sidestepping that wall — it is the one bridge primitive
    that can.
- **Bridge realization sketch (reuse, near-zero new code).** Flag the **cortex** region (not just the LGN hub)
  with `graded_lateral=True`; set `cfg.enable_graded_lateral=True`. The lateral then decorrelates the cortex's
  analog drive before its spike threshold, *while* `--learn-projection` STDP learns the feedforward
  hub→cortex weights on that decorrelated drive — exactly the SM split (feedforward Hebbian W + recurrent
  anti-Hebbian M). **One caveat to tune, not re-engineer:** the hard-coded **target is `I` (full ZCA
  whitening)**, and the off-diagonal de-risk (`_phaseB_offdiagonal_derisk.py`) showed **full ZCA collapses
  (−0.012)** while **low-rank whitening reaches host +0.44**. The `−λM` decay already exists *for this exact
  reason* (config comment: "the rate-model regularizer that settles a GENTLE, bounded fixed point ≈ C^−1/3
  instead of over-whitening"). So the de-risk is to find the **partial-whitening regime** via the existing
  knobs `graded_lateral_lr`, `graded_lateral_lambda` (raise λ → gentler/lower-rank decorrelation),
  `graded_lateral_gain_pA`, `graded_lateral_act_scale` — i.e. decorrelate **without** going to full identity.
- **Edit class.** **Config/runner only** to start (flag the cortex region + set the global flag + sweep the 4
  existing knobs). A *possible* small guarded `sim/` edit later if the partial-whitening regime needs a
  configurable **target** (e.g. `M`'s target = `βI` with β<1, or a low-rank-projected target) — but try the
  λ-knob first; it may already give the bounded sub-whitening fixed point.
- **Risk.** Two real risks: (i) `graded_lateral` is scoped to **ONE contiguous region** (the first flagged,
  `bridge.py:1794–1810`) — fine for a single cortex region, but it asserts contiguous indices and warns if >1
  region is flagged; the ON/OFF cortex is **two** regions (`cortex_on`, `cortex_off`), so the build must either
  use a **single** cortex region or extend the lateral to span both (a small guarded edit). (ii) The full-ZCA
  collapse is real — if no λ setting gives a partial-whitening sweet spot, this degrades to option (c)/(d).
  But the numpy off-diagonal de-risk *proves a low-rank whitening regime exists at +0.44*, so the target exists.

### (b) **FS-interneuron WTA (Diehl–Cook): sparse competition, NOT decorrelation — INSUFFICIENT alone.**

- **Biology + citation.** [Diehl & Cook 2015, *Front. Comput. Neurosci.* 9:99](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full):
  unsupervised STDP on MNIST via **lateral-inhibition WTA + adaptive per-neuron thresholds (homeostasis)**.
  Lateral inhibition makes the strongest-responding output fire first and suppress the others → each neuron
  specializes to a distinct input class.
- **What it does on the substrate.** A shared inhibitory (FS) population implements **global gain / k-WTA
  sparsity + homeostatic threshold balancing** — it makes the code *sparse and competitive*, so different
  inputs recruit different neurons. The project already has this pattern (`exc_fraction=0.0` inhibitory pools;
  `enable_motor_fs`; the `cm` region in `spiking_sm_cortex.py:147–156`).
- **Why it is NOT the SM rule.** The project already measured this (CYCLE-66 / 2026-06-06): "a shared-FS
  SPIKING lateral does **GLOBAL gain, not pairwise whitening**" (`bridge.py:1768–1770`). Diehl–Cook WTA
  **sparsifies** (reduces *which* neurons fire) but does **not decorrelate the residual output dimensions** —
  it has no learned pairwise `M_ij`. SM needs the **pairwise** anti-Hebbian decorrelation (the off-diagonal of
  M), which a single shared inhibitory pool (rank-1 inhibition) cannot supply. CYCLE-62/63 confirmed exactly
  this: WTA + adaptive-θ homeostasis recovered structure −0.07.
- **Edit class / role.** Config/runner. **Verdict: keep as a complementary sparsity/stability layer
  (homeostasis to prevent dead/saturated units), but it cannot replace the decorrelating lateral.**

### (c) **A learned anti-Hebbian INHIBITORY lateral via real inhibitory synapses + inhibitory STDP (NEW).**

- **Biology + citation.** The most literally-biological SM realization: a population of **inhibitory
  interneurons** reciprocally connected to the cortex pyramidals, with the inhibitory synapses learning an
  **anti-Hebbian (inhibitory-STDP) rule** (Vogels-2011-style E/I-balancing iSTDP generalized to a
  decorrelating target; Pehlevan 2015's M *is* an inhibitory weight matrix). This is the SPIKING SM network of
  [Pehlevan 2019, arXiv:1902.01429, "A Spiking Neural Network with Local Learning Rules Derived From
  Nonnegative Similarity Matching"](https://arxiv.org/abs/1902.01429) — integrate-and-fire units, local rules,
  the canonical published spiking realization.
- **Why it is heavier than (a).** The bridge has **no learnable inhibitory-STDP / anti-Hebbian rule on ordinary
  synapses** (confirmed: grep for `cp_synapse_sign` / inhibitory plasticity finds nothing; `graded_lateral` is
  the *only* anti-Hebbian decorrelation primitive). Building (c) means a new guarded `sim/` plasticity rule on
  a real spiking inhibitory population — and it re-incurs the **rate-code decorrelation wall** that (a) avoids
  by using analog activity (a spiking inhibitory lateral does the decorrelation in spikes, which the whole arc
  shows is lossy). So (c) is more biologically literal but **strictly harder and riskier** than (a).
- **Edit class.** A **guarded `sim/` edit** (new inhibitory-STDP rule + reciprocal cortex↔interneuron wiring).
  Owner-gated (byte-level diff review). **Verdict: the principled "fully-spiking SM" target, but defer behind
  (a)** — (a) is the same M, learned more cheaply in the analog domain that actually works on this substrate.

### (d) **The full Pehlevan–Chklovskii recurrent-dynamics SM (explicit settle loop + both rules).**

- **Biology + citation.** The complete algorithm: per presentation, **iterate the recurrent dynamics to
  convergence** (`y ← y + η(Wx − My)`, many steps), then update both W (Hebbian Oja) and M (anti-Hebbian),
  exactly as `learn_simmatch`/`simmatch` do in numpy (`learned_graded_cortex_fair_test.py:162`,
  `_l1_nonneg_simmatch_check.py:37`). This is the gold-standard SM and the validated numpy GO (+0.515 nonneg).
- **Why it is the heaviest.** It needs a genuine **multi-step recurrent settle per input** on the bridge (the
  bridge runs one membrane step per `_step_with_time`; a full settle is many sub-steps with the lateral in the
  loop) plus both plastic rules. `graded_lateral`'s **one-step** `−(M@a)` feedforward-inhibition is the
  *single-iteration approximation* of this settle (Pehlevan 2018 shows the one-step/feedforward-inhibition form
  is a valid approximation). So (d) is (a) taken to full convergence.
- **Edit class.** Larger guarded `sim/` edit (recurrent settle loop). **Verdict: the fallback if the one-step
  `graded_lateral` approximation (a) proves too weak — escalate the settle depth, not the mechanism.**

---

## 3. VERDICT + the cheap-first de-risk the controller should run next

**VERDICT.** The missing piece is unambiguous and the literature is decisive: **the L1 SM rule requires a
recurrent anti-Hebbian lateral M that decorrelates the cortex outputs; plain feedforward STDP (even E/I-plastic
+ tight-bound) is structurally only the Hebbian-W half and rank-1 collapses without it.** The project's
**`graded_lateral` machinery IS that lateral, already on-substrate, already with the correct
`ΔM ∝ ⟨aaᵀ⟩ − I − λM` rule, and — uniquely — it operates on sub-threshold ANALOG activity, which is the only
way a point-neuron substrate can decorrelate (sidestepping the Mikulasch–Priesemann rate-code wall that broke
every spike-based attempt).** The recommended Phase-3 build is **option (a): flag the cortex region with
`graded_lateral`, run it alongside the `--learn-projection` STDP, and tune λ/gain to the partial-whitening
(low-rank) regime** the off-diagonal de-risk proved reaches host +0.44 (full ZCA over-whitens).

**The cheap-first de-risk (numpy, no `sim/` edits, build once — exactly the controller's next move):** add ONE
arm to the existing L1 numpy battery that mimics the bridge's *actual* one-step `graded_lateral` mechanism, and
confirm it beats the random projection toward host:

> On the **centered + E/I-projected** input (the front-end the bridge already has), run a learner that is
> **feedforward STDP/Oja-Hebbian W + a ONE-STEP anti-Hebbian lateral** `y = relu(Wx − M·â)` with `â` the
> *analog* (pre-rectification) drive and `M` learned `ΔM = lr(⟨ââᵀ⟩ − βI) − λM` (β as a partial-whitening
> knob ∈ {1, 0.5, 0.25}; sweep λ). Compare to: (i) the **frozen random projection** (+0.155 bridge / +0.31
> numpy — must BEAT it), (ii) **full SM** `learn_simmatch` (the +0.48 ceiling), (iii) **feedforward-only**
> (the rank-1 collapse, must trail). The one-step lateral approximates `graded_lateral` exactly, so a GO here
> de-risks the bridge build; a NEGATIVE says the one-step approximation is too weak → escalate to the full
> settle (option d) BEFORE the `sim/` work.

This is ~1 new function bolted onto `_l1_centered_online_pca_probe.py` / `learned_graded_cortex_fair_test.py`
(reuse `center_cols`, `ppmi_matrix`, the ON/OFF + E/I projection, the metrics). CPU/numpy, minutes.

**Anti-cheat controls (reuse the battery already in the L1 runners):**
- **beats-random** — the lateral learner must beat the frozen random projection (the whole point; the bare
  minimum the 3 STDP fixes failed). This is THE gate.
- **learning-load-bearing** — the *learned* M must beat a **fixed random M** (and a zero M) on the same input,
  so the win is the *learned decorrelation*, not just adding inhibition. (Mirrors `_l1_oja_validated.py`'s
  random-projection honesty control — note that arm flagged the structure is partly in the PPMI *input*; the
  load-bearing gate must isolate the M's contribution.)
- **permuted-label** — `Pearson(cos, S_perm) ≈ 0` (structure not an artifact) — already in every L1 runner.
- **eff-rank** — the learned code must have **eff-rank ≫ 1.5** (the collapse signature); rising eff-rank toward
  the SM ceiling is the mechanistic tell the lateral is doing its job.
- **generalization (Fodor–Pylyshyn / held-out)** above chance — already in `heldout_generalization`.
- **no host shortcut** — the centering, projection, and decorrelation must all be on-substrate (the bridge
  primitive computes M from the membrane `a`; no host `XᵀX` or host whitening in the neural path). Per the
  BRAIN-BASED-ONLY standard, an honest NEGATIVE (the one-step analog lateral underperforms host) IS the
  deliverable.

---

## 4. What existing machinery is reusable (and the headline answer: `graded_lateral` IS the lateral)

| Need | Reusable bridge machinery | Status |
|---|---|---|
| **Recurrent anti-Hebbian decorrelating lateral M** (the SM missing piece) | **`graded_lateral`** (`bridge.py:1776–1884`, `regions.py:188`, `config.py:381–398`): plastic K×K M, `ΔM ∝ ⟨aaᵀ⟩−I−λM`, `−(M@a)·gain` pre-threshold, **on analog membrane `a`** | **IS THE ANSWER — reuse directly; tune λ/gain for partial whitening.** |
| Per-feature mean-centering (axis-0 / DC half of whitening) | `enable_input_mean_adapt` + `BrainRegion.input_mean_adapt` (shipped, byte-clean, CYCLE 71) | reuse as-is (already in the Phase-3 bridge) |
| Signed projection (carry the centered signal past Dale's law) | E/I balance: `--enable-ei` inhibitory hub copies (`_phaseB_input_mean_bridge.py:125–149`) | reuse as-is |
| Learn the feedforward W on the decorrelated drive | `--learn-projection` STDP thaw + tight `--stdp-w-max` | reuse; the lateral is what was missing alongside it |
| Sparse competition / dead-unit prevention | FS inhibitory pools (`exc_fraction=0.0`), `enable_homeostasis`, adaptive-θ (Diehl–Cook) | complementary stability layer — does NOT decorrelate; do not rely on it for SM |
| Full settle / fully-spiking SM (fallbacks d / c) | none yet (one-step `graded_lateral` is the approximation; a recurrent settle loop or inhibitory-STDP is a NEW guarded edit) | defer behind (a) |

**Two structural notes on `graded_lateral` to flag before the build:** (1) it is **scoped to ONE contiguous
region** — the ON/OFF cortex is two regions, so use a single cortex region OR extend the lateral to span both
(small guarded edit); (2) the **target is hard-coded `I`** (full ZCA) and full ZCA collapses on real data — the
build must reach the **partial/low-rank** regime via λ (and possibly a configurable `βI` target). The
off-diagonal de-risk proves that regime exists at host +0.44, so this is a tuning problem, not a wall.

---

### Citations (load-bearing)

- Pehlevan, Sengupta, Chklovskii 2018, *Neural Comput.* 30:84 — SM ⇒ Hebbian-W + anti-Hebbian-M; the settled
  output `y=(I+M)⁻¹Wx`. https://direct.mit.edu/neco/article/30/1/84/8348/
- Pehlevan & Chklovskii 2015, arXiv:1511.09468 — Hebbian/anti-Hebbian networks for PCA **and whitening**; the
  `−λM` bounded fixed point (the `graded_lateral` rule's source). https://arxiv.org/abs/1511.09468
- Pehlevan 2019, arXiv:1902.01429 — published **spiking** (integrate-and-fire) SM with local rules (option c
  target). https://arxiv.org/abs/1902.01429
- Földiák 1990, *Biol. Cybern.* — local **anti-Hebbian feedback** decorrelates a code while preserving info
  (the canonical decorrelating lateral). https://link.springer.com/article/10.1007/BF02331346
- Diehl & Cook 2015, *Front. Comput. Neurosci.* 9:99 — unsupervised STDP via **WTA lateral inhibition + adaptive
  θ** = sparse competition, NOT pairwise decorrelation. https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full
- Catalog (`sim-catalog/references/feature-catalog.md`): **A.12** sparse decorrelated cortico-striatal
  ("decorrelation rule"); **E.05** lateral inhibition / center-surround ("decorrelates output"); **B.04/B.52**
  MSN lateral-inhibition WTA (global gain, not pairwise whitening). No catalog entry yet for a *learned
  anti-Hebbian decorrelating lateral* — `graded_lateral` is the project's own realization of the Pehlevan/Földiák
  motif (the closest catalog functional kin is E.05 center-surround decorrelation).
- Project code: `learned_graded_cortex_fair_test.py:162` (`learn_simmatch` — explicit W+M),
  `_l1_centered_online_pca_probe.py:41` (`oja_subspace` — implicit `−y²W` lateral),
  `_l1_phaseA_end_to_end_spiking.py:146` (the "bounded-Hebbian no-lateral" arm = normalized-input + integrated
  readout), `_phaseB_input_mean_bridge.py` (the Phase-3 thaw runner), `_phaseB_offdiagonal_derisk.py` (full ZCA
  collapses / low-rank whitening = host +0.44).
