# Generative-sequence frontier (Spine A) — SURROGATE-GRADIENT FINETUNE ON THE BRIDGE — scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings re-verification for an OWNER-CHOSEN NEW MECHANISM (the standing
> research gate fires under condition **(d) new mechanism class** — surrogate-gradient *training of on-bridge weights*
> has never been built; the project has only trained STANDALONE differentiable SNNs, never the bridge). **NO `sim/`
> edits, NO experiments, NO GPU.** Single deliverable = this doc. The controller should trust-but-verify the **[VERIFY]**
> items, push, and present the recommendation before building.
>
> **The pivot this scopes:** the exact-replication consolidation hit a per-layer clip-compression wall — the graded
> readout un-saturates (Phase A) but the cumulative fidelity *deterministically* compresses `0.846 → 0.620 → 0.288`
> through the stacked `clip(a@W,0,1)` stages, and population coding is a literal no-op on that noiseless readout
> (`2026-06-22-genseq-loopstep3-popcode-NEGATIVE-deterministic-readout.md`). The owner chose the robust SOTA path: stop
> fighting *exact replication of installed-verbatim weights*; **TRAIN the on-bridge weights via surrogate-gradient so the
> bridge's ACTUAL (graded/spiking) output matches the teacher.** This is SpikingBrain's continued-training step, and it
> is the SAME on-bridge learning the C2 (grow + no-forget) stage needs.

---

## 0. One-paragraph answer (the rest is the evidence)

**The finetune is SURPASSABLE cheaply, and — decisively — it does NOT need a differentiable bridge.** The wall is now
known to be *deterministic* per-layer signal compression through stacked `clip(a@W,0,1)` nonlinearities (popcode std
measured at exactly `0.00e+00` → noiseless). A deterministic, differentiable function is exactly what you can *train
around*. The cheapest, highest-fidelity path is the **HYBRID (option b): train the stacked-MLP weights in a
*bridge-faithful* differentiable forward that REPLICATES the bridge's deterministic analog path** — the graded forward
`a_{L+1} = clip(a_L @ W_L, 0, 1)` that the runner *already* wrote and validated as the on-bridge ground truth
(`_genseq_loopstep3_graded_derisk.py:107-117`, `offbridge_graded_forward`). That forward is *trivially* differentiable
in any autograd framework (it is `clip`+matmul, no spikes, no `g·(V−E)`), distil the float transformer's per-layer
activations into it layer-by-layer, then **install the trained weights onto the bridge with NO `sim/` edit** (re-inject
via `inject_explicit_wiring`, which builds the CSR from `initial_weights`, `bridge.py:2393/2468/2491`, or overwrite
in-place via `set_pathway_weights`, `bridge.py:2986`). Because the bridge's graded path provably realizes `clip(a@W,0,1)`
to high fidelity *per layer* (Phase A L0 = 0.846/0.865), weights trained to make THAT function match the teacher will
match the teacher on the bridge. **Option (a) — making the bridge's own forward differentiable (autograd or a
hand-written backward through `_run_one_simulation_step`) — is a real multi-week `sim/` rewrite and is NOT needed unless
the HYBRID's bridge-replication fidelity is shown insufficient.** The training TARGET should be **layerwise activation
distillation** (cheapest, most stable, matches the per-layer metric the consolidation already uses) before the next-token
end-task. **The C2 overlap is genuine and load-bearing:** the differentiable bridge-faithful trainer + the gradient
machinery + the Phase-1.4 no-forget gating compose into exactly the "learn new on-bridge weights without disturbing the
frozen old ones" mechanism C2 needs — so this is foundation, not throwaway. **Cheapest de-risk: train the 3-layer MLP
slice in the bridge-faithful `clip(a@W,0,1)` trainer to distil the teacher's per-layer activations → install on the
bridge → does cumulative analog-Spearman recover to ≥0.8?** GO ⇒ the consolidation wall is surpassed with NO `sim/` edit;
NEGATIVE ⇒ escalate to option (a) (the differentiable-bridge `sim/` edit) — only then.

---

## 1. THE CORE QUESTION — ranked: how do you get a surrogate gradient to update the consolidated weights so the bridge's output matches the teacher?

The literal blocker: **the bridge is a FORWARD simulator, not differentiable.** `_run_one_simulation_step`
(`bridge.py:5715`+) executes the spiking/graded dynamics imperatively (CSR matvecs, `fused_*` kernels, in-place CuPy
state updates) with **no autograd tape and no backward**. To update weights by gradient, you need a differentiable
forward whose weights you then install on the bridge. Three families, cheapest-first:

### (b) HYBRID — train in an already-differentiable forward that REPLICATES the bridge's deterministic dynamics, then install — **#1, NO `sim/` edit, cheapest + most faithful for THIS wall** ✅

- **Mechanism:** the consolidation wall is the *graded* path's stacked `clip` compression. The runner already wrote the
  bridge's deterministic ground-truth forward — `offbridge_graded_forward(Ws, oh, n_blocks)`:
  `a = clip(a @ Ws[L], 0.0, 1.0)` per block (`_genseq_loopstep3_graded_derisk.py:107-117`). The popcode-NEGATIVE proved
  the on-bridge `a_cont` readout is byte-deterministic (`within_pop_std_max = 0.00e+00`,
  `2026-06-22-...popcode-NEGATIVE...md` line 3, 17) and the graded Phase A confirmed the bridge reproduces this analog
  forward per-layer at L0 = 0.846/0.865 (`...graded-PARTIAL...md` table). **⇒ the bridge ≈ a deterministic
  `clip(a@W,0,1)` MLP for the linear-ops half.** Re-implement that exact function in autograd (PyTorch, which the convert
  step already uses — `2026-06-22-genseq-convert-GO-spiking-generates.md` runs PyTorch/CUDA), make `W_L` `requires_grad`,
  and backprop the per-layer distillation loss through `clip`+matmul (clip has a clean sub-gradient — pass-through inside
  `(0,1)`, zero outside; no surrogate even needed for the *graded* half).
- **Why it dissolves the wall:** the `0.846 → 0.620 → 0.288` compression is the *installed-verbatim float transformer
  weights* being a poor match for what the stacked-`clip` chain can carry. **Train the weights FOR the `clip` chain** and
  each layer's output is optimized to match the teacher *given* the clip saturations of the layers below — the
  compression is absorbed into the learned weights (the standard ANN→SNN / quantization-aware-training move: don't fight
  the nonlinearity, train through it).
- **Faithfulness:** HIGH for the graded/linear half (the trainer's forward = the bridge's deterministic readout exactly,
  by construction — same `clip`, same composition). The one honest gap: the trainer's `clip(a@W,0,1)` is the *idealized*
  analog forward; the live bridge adds the `g·(V−E)` driving-force sub-linearity, per-block gain calibration, and AdEx
  settling (the runner's `non_spiking` integrator regime + greedy per-block gain, `:323-340,:411-455`). So the trainer is
  faithful to the *target representation* but not byte-identical to the live conductance dynamics — the de-risk MEASURES
  the residual (install → re-run the existing graded metric). If the `g·(V−E)` residual matters, fold a 1-parameter
  per-block affine (the calibration the runner already does) into the trainer, or add the driving-force term to the
  trainer forward (still differentiable, still NO `sim/` edit).
- **Cost / `sim/`-edit scope:** **NONE.** Pure reuse — a new runner that (i) loads `cortex_10M_seed42.npz`, (ii) builds
  the autograd `clip`-MLP, (iii) distils per-layer activations, (iv) installs trained `W_L` via the existing
  `inject_explicit_wiring` / `set_pathway_weights`, (v) re-runs the existing graded fidelity metric. Day-scale.

### (a) Make the consolidated-slice forward DIFFERENTIABLE on the bridge — autograd or hand-written backward through the step — **#2, the real `sim/` edit, only if (b) is insufficient** ⚠️

- **Mechanism:** add a surrogate-grad-aware backward for the consolidated slice's forward path so the *live bridge
  dynamics* (graded `a_cont` + spike + `g·(V−E)` + refractory + dt) are themselves differentiable and the finetune loop
  trains the bridge directly. Two sub-variants: **(a1)** a hand-written backward for the slice's specific op chain
  (graded matvec + `clip` + conductance + AdEx update), wrapping the existing forward — mirrors `bptt_snn.backward_unroll`
  but for the bridge's *graded* path; **(a2)** rebuild the slice forward in an autograd framework that traces the CuPy
  ops (heaviest, effectively re-implements `_run_one_simulation_step` differentiably).
- **Faithfulness:** HIGHEST (trains the actual live dynamics, including `g·(V−E)` and refractory). But it is the only
  option that needs a protected `sim/` edit, and a *large* one — the bridge has no backward and was never written to be
  differentiable; this is a multi-week build (the parent scoping flags it "NOT on the critical path unless the cheap
  calibration loses too much," `2026-06-22-genseq-loopstep3-consolidation-scoping.md:303-305`).
- **`sim/`-edit scope:** the deepest of the three — see §2. **Reach ONLY after (b) is measured NEGATIVE.**

### (c) LOCAL rule on the bridge — feedback alignment / forward-forward / e-prop — **#3, no full backprop, lower fidelity** ⚠️

- **Mechanism:** avoid exact backprop entirely. **Feedback alignment** (Lillicrap 2016) replaces the transposed weight
  matrix in the backward with a fixed random matrix — a *local-ish* update that needs only a forward + a random
  projection of the error (no differentiable forward through the whole chain). **Forward-forward** (Hinton 2022) trains
  each layer with a local goodness objective on positive/negative data — *no backward at all*, each layer learns from its
  own activity. **e-prop** (Bellec 2020) is the biologically-grounded online surrogate-gradient approximation using
  eligibility traces — and the bridge *already has eligibility traces* (`fused_eligibility_trace_decay`,
  `cp_plasticity_rate_gain`), so e-prop is the most substrate-native of the three.
- **Faithfulness:** LOWER for deep credit assignment (feedback-alignment + forward-forward both lose accuracy vs backprop
  on deep nets), but **HIGHEST biological plausibility** (and e-prop is the BRAIN-BASED-ONLY-aligned answer — it is how a
  real cortex would do surrogate-gradient credit assignment, local + online). For a 3-layer MLP the depth penalty is
  mild.
- **`sim/`-edit scope:** small-to-moderate — a local-update step block (e-prop reuses the eligibility infra; feedback
  alignment needs a fixed random feedback array + a per-layer error injection). **Strategically valuable later** (it is
  the brain-based learning rule the project ultimately wants for C2), but for *this* consolidation de-risk it is heavier
  than (b) and lower-fidelity, so it is #3.

**Ranking verdict:** **(b) HYBRID first** (no edit, cheapest, faithful-to-the-target-representation, day-scale) →
**(a) differentiable-bridge** only if (b)'s install fidelity is insufficient (the real `sim/` edit) → **(c) local rules**
as the brain-based C2 learning rule for the *grow + no-forget* stage (strategically the endpoint, not the cheapest
consolidation de-risk). This ordering matches the prior scoping's (c)-was-#3 placement
(`2026-06-22-genseq-consolidation-past-saturation-scoping.md:183-202`) but sharpens it: the popcode-NEGATIVE's
*determinism* finding is exactly what makes the HYBRID (train the deterministic `clip` chain) the cheap winner.

---

## 2. The `sim/`-edit scope IF (a) is needed (the differentiable bridge) — minimal, additive, guarded, default-off

If (b) misses and option (a) is reached, the edit must follow the project's additive-guarded-default-off discipline
(every recent `graded` / `nmda_slow` / `coincidence` block is the template — `None`-guarded, byte-identical when off).
**What exactly must differentiate** (the minimum):

- **The consolidated slice's forward op chain, exposed with a backward.** For the *graded* consolidation the differentiable
  forward is small: `a_cont = clip((v − rest)/scale, 0, 1)` (`bridge.py:6144-6147`) → masked graded matvec
  `Wᵀ @ a_cont` into `g_e`/`g_i` (`:6148-6175`) → `g·(V−E)` current (`fused_conductance_decay_and_current`, `:6059`) →
  the AdEx/integrator update. The backward is the chain rule through those four ops — `clip` (pass-through in band, 0
  outside), a sparse matvec transpose (already computed forward as `_WgT`), the `g·(V−E)` product, and the membrane
  update (a linear leak in the `non_spiking` integrator regime — *no surrogate needed if non-spiking*; the surrogate
  (`atan_surrogate`, `surrogate_grad.py:22`) is only needed if the slice spikes).
- **Scope it to the slice, default-off.** A new `cfg.enable_bridge_finetune_backward` flag + a `neuron_mask` (the
  consolidated slice indices), exactly like the masked-RF-op precedent (`rf_kick(neuron_mask=...)`, CLAUDE.md
  "sliced RF ops"). When off → unreached, byte-identical (the 18/18 conversational tests must pass verbatim, as the
  graded/RF edits did). The backward only needs to run during a *finetune* call, never in normal stepping.
- **Reuse, don't re-derive, the backward math.** `bptt_snn.backward_unroll` / `bptt_snn_gpu.backward_unroll_xp` already
  implement the hard-reset surrogate + recurrent chain rule for a *spiking* LIF (`bptt_snn.py:180-256`); the bridge edit
  adapts that to the *graded* op chain (simpler — the dominant nonlinearity is `clip`, not the Heaviside). The CuPy port
  is fp32-validated against the numpy reference (`bptt_snn_gpu.py:9-11`).
- **What must NOT change:** the spike matvec, the `fused_*` kernels, the Izhikevich/HH/AdEx paths — all byte-unchanged
  (the backward is a *separate* code path gated by the flag, reading the same forward intermediates).

**Honest cost:** even minimal, this is the heaviest option — exposing a backward through live conductance dynamics is a
new differentiable subsystem in `sim/`, owner-byte-reviewed, multi-week. **It is the fallback, not the plan.**

---

## 3. The training TARGET — layerwise activation distillation vs the next-token end-task — RECOMMEND layerwise

**RECOMMEND: layerwise per-layer activation distillation, FIRST.** Reasons, all grounded in the existing pipeline:

- **It matches the metric the consolidation already uses.** The wall is *measured* per-layer (cumulative analog-Spearman
  `[0.846, 0.620, 0.288]`, `...popcode-NEGATIVE...md` line 12). Distilling each on-bridge layer's `a_cont` to the
  teacher's matched activation directly optimizes that exact quantity — the de-risk's GO/NO-GO reads out cleanly.
- **It is the most stable / cheapest gradient.** Per-layer distillation is a sequence of *independent shallow* regressions
  (`min_W ‖clip(a_L@W) − teacher_{L+1}‖`), each well-conditioned, no deep BPTT through the whole stack, no long-horizon
  credit assignment, no exploding/vanishing through stacked `clip`s. (The end-task next-token loss must backprop through
  ALL layers + the readout — far harder, and the very long-unroll instability `bptt_snn` was built to manage.)
- **The teacher activations already exist.** The off-bridge references are written: the *graded* target
  `offbridge_graded_forward` = `clip(a@W,0,1)` per block (`_genseq_loopstep3_graded_derisk.py:107-117`), AND the trained
  net's own per-layer activations (the float transformer Gen-F, `2026-06-22-genseq-convert-GO...md`; the LIF
  `forward_unroll` membrane, `offbridge_spiking_membrane`, `:120-132`). The distillation target is the float
  transformer's per-layer activation (what the on-bridge layer should compute), regressed into the `clip`-chain forward.
- **SOTA precedent.** ANN→SNN distillation and feature/activation distillation (Hinton 2015; the QCFS / activation-matching
  family the convert step cites) all distil *activations* layerwise for stability before/instead of end-task finetune.
  SpikingBrain's continued-training likewise aligns intermediate representations, not only the LM head.

**Then (optional, second) the next-token end-task** as a *polish* once layerwise has recovered per-layer fidelity — a
small end-to-end finetune to recover any cross-layer interaction the greedy layerwise pass left on the table. But the
GO/NO-GO of the consolidation de-risk is decided by the **layerwise** recovery; the end-task is downstream.

---

## 4. The C2 overlap — this on-bridge learning IS the mechanism C2 (grow + no-forget) needs; scope so it isn't throwaway

**The overlap is genuine and load-bearing — this finetune is the C2 foundation, not a one-off consolidation hack.** C2 =
"grow + no catastrophic forgetting" = *learn new on-bridge weights for new knowledge WITHOUT disturbing the frozen old
weights.* That is precisely a gradient-based on-bridge weight update plus a no-forget constraint. The three pieces compose:

- **The differentiable bridge-faithful trainer (this work) IS the "learn new on-bridge weights" half.** Whether (b)
  HYBRID (train-then-install) or (a) (train-on-bridge), the deliverable is a validated path to *set on-bridge weights by
  gradient to match a target* — exactly what C2's growth step does for each new concept/skill.
- **The Phase-1.4 BRANCH A no-forget machinery IS the "without disturbing the old" half — and it ALREADY EXISTS,
  multi-seed-validated.** CLAUDE.md: Phase-1.4 BRANCH A = **5/6 seeds ≥80% retention, mean 103%** on the catastrophic-
  forgetting eval (`continual_forgetting_eval.py`); the consolidation (Phase-1.3, CLS / sleep-replay) transfers W→A into
  cortex with **3/3 strict-anti-cheat** retention. The mechanism is the **per-pathway plasticity gate** (`cp_plasticity_
  rate_gain=0` freezes weight UPDATES for a tagged pathway; `set_plasticity_gate(name, value)` thaws/freezes at runtime;
  the masked-clip fix `bridge.py:6673/6990/7253` keeps a frozen synapse byte-identical even under the global clips). So a
  C2 growth step trains the NEW slice's weights (gate open) while the OLD slices are gate-frozen → no forgetting *by
  construction*.
- **Scope the overlap so this isn't throwaway:** build the finetune trainer to **(i)** target a *named slice* (the
  consolidated generator slice now; an arbitrary new slice later), **(ii)** respect the plasticity-gate freeze on all
  other slices (train only the open-gate slice's weights), **(iii)** install via the same `inject_explicit_wiring` /
  `set_pathway_weights` the lineage/auto-growth path already uses (`sim/auto_growth.py` TierPromoter + weight-transfer;
  `sim/lineage.py` persistent state). Then the *same* trainer that surpasses the consolidation wall is the C2 growth
  primitive: add a slice → train it (gate-open) against its new target → freeze its gate → the old slices were never
  touched. **The de-risk should explicitly verify the no-forget compose** (after the finetune, a frozen conversational
  slice is byte-identical and the no-confab moat is intact — the standing co-residence anti-cheat).

⇒ The finetune is **directly on the C2 critical path**: it is the missing "learn on-bridge weights by gradient" half, and
Phase-1.4's gate-freeze is the already-validated "no-forget" half. Building (b) first delivers the consolidation fix AND
the C2 growth primitive in one artifact.

---

## 5. The cheapest-first DE-RISK + GO/NO-GO + the `sim/`-edit verdict

**Top-ranked de-risk = (b) HYBRID, layerwise activation distillation, NO `sim/` edit.** The smallest decisive experiment:

1. **Train.** Load `cortex_10M_seed42.npz` (the 4-layer 66→2048→2048→2048→66 net the consolidation runs on). Build the
   **bridge-faithful differentiable forward** = the runner's own `offbridge_graded_forward` chain, `a_{L+1} =
   clip(a_L @ W_L, 0, 1)`, in PyTorch with `W_L.requires_grad` (the convert step already uses PyTorch/CUDA — reuse it).
   For each block L (greedy, layer-by-layer, holding earlier trained blocks fixed): minimize
   `‖clip(a_L @ W_L, 0,1) − teacher_{L+1}‖²` where `teacher_{L+1}` is the float transformer's matched per-layer
   activation on the 6-char calibration set (+ a held-out char set for the specificity check). Optionally fold the
   per-block affine gain the runner already calibrates (`per_layer_e_gain`, `:286-289`) into the trainer so the operating
   point matches the live bridge.
2. **Install.** Re-inject the trained `W_L` into a fresh bridge via the EXISTING `inject_explicit_wiring` (the graded
   signed-split wiring builder, `_genseq_loopstep3_graded_derisk.py:261-308`, with `initial_weights = trained W_L`) —
   **NO `sim/` edit** (`inject_explicit_wiring` builds the CSR from `initial_weights`, `bridge.py:2468/2491`).
3. **Measure.** Re-run the EXISTING graded fidelity metric (`onbridge_block_analog` + the per-layer/cumulative
   analog-Spearman + the matched/mismatched specificity anti-cheat, `:343-587`) on the installed trained weights. Single
   decisive existence run on CuPy first (`feedback_gpu_not_numpy`); ≥6 seeds only once it becomes a variable claim.
4. **Anti-cheat (mandatory):** (i) the matched-vs-mismatched cross-input specificity margin must re-open (the trained
   weights compute each char's SPECIFIC mapping, not a generic high-activation pattern — the runner's existing
   `anti_cheat_specificity`); (ii) a **no-learning / random-target control** (distil to SHUFFLED teacher activations →
   must NOT recover) to prove the recovery is from real distillation; (iii) if the no-forget compose is tested, a
   co-resident conversational slice stays byte-frozen + moat intact.

**GO / NO-GO:**
- **GO** if installed-on-bridge **cumulative analog-Spearman ≥ 0.8** across the 3 stacked blocks (vs the verbatim-install
  0.288), AND the specificity margin re-opens (matched ≫ mismatched), AND the shuffled-target control stays at chance.
  ⇒ the consolidation clip-compression wall is SURPASSED via the HYBRID, **NO `sim/` edit** → resume the loop-step-3
  ladder (attention #2 → full forward #3) → C2, and the trainer is the C2 growth primitive.
- **PARTIAL** if it recovers above the verbatim 0.288 but < 0.8 (e.g. the trainer's idealized `clip` forward leaves a
  `g·(V−E)` / settling residual the install can't close) → add the driving-force term / per-block affine to the trainer
  forward (still differentiable, still NO edit) and re-measure; if still < 0.8 → escalate the END-TASK finetune (a small
  end-to-end next-token polish through the `clip`-chain) before any `sim/` edit.
- **NO-GO (route to option (a))** ONLY if a bridge-faithful trained `clip`-chain CANNOT be made to match on the bridge —
  i.e. the live conductance dynamics diverge irreducibly from the idealized `clip(a@W,0,1)` forward even after the
  driving-force term is added. THEN the differentiable-bridge `sim/` edit (option (a), §2) is justified — train the
  *actual* live dynamics. **This is not expected**, because the bridge's graded path provably realizes `clip(a@W,0,1)`
  per-layer at L0 = 0.846 already; the only open question is whether trained weights close the *cumulative* gap, which a
  faithful trainer is designed to do.

**The `sim/`-edit verdict:** **NONE for the cheapest path (b).** The graded transmission path, the `graded=True` wiring
flag, `inject_explicit_wiring`, `set_pathway_weights`, and the off-bridge `clip`-forward target ALL already exist; the
HYBRID is pure reuse + a PyTorch training loop. A `sim/` edit (option (a), additive-guarded-default-off, §2) is the
fallback reached ONLY on a measured NO-GO — and is itself bounded (a backward for the slice's graded op chain, reusing
`bptt_snn`'s validated math).

---

## 6. Honest verdict — surpassable + how cheaply, vs a deep multi-week build

**SURPASSABLE, cheaply, with NO `sim/` edit, via the HYBRID — because the popcode-NEGATIVE turned the wall into a
*deterministic, differentiable* function you can train through.** Specifically:

- **The genuine residual is deterministic per-layer compression** (`0.846 → 0.620 → 0.288` through stacked `clip`), NOT
  noise (popcode std = `0.00e+00`) and NOT a fundamental point-neuron limit — it is *installed-verbatim float weights
  being a poor fit for the `clip` chain*. That is the textbook case for *training the weights for the substrate*
  (quantization-aware / ANN→SNN distillation), not a wall.
- **The cheapest surpass is the HYBRID:** the bridge's deterministic analog forward is ALREADY written as a differentiable
  `clip(a@W,0,1)` chain (`offbridge_graded_forward`); distil the teacher's per-layer activations into it; install the
  trained weights via the existing wiring API. **NO `sim/` edit, day-scale, faithful-to-the-target-representation.** This
  is the SOTA path (SpikingBrain continued-training; ANN→SNN activation distillation) realized with the project's own
  validated machinery.
- **The differentiable-BRIDGE rewrite (option a) is the genuine multi-week `sim/` build — and it is the FALLBACK, not the
  plan.** It is reached only if the HYBRID's install fidelity is measured insufficient; even then it is bounded (a
  guarded backward for the slice's graded op chain, reusing `bptt_snn`'s fp32-validated surrogate math). The local-rule
  path (option c, e-prop/feedback-alignment) is the *brain-based* learning rule the project ultimately wants for C2 —
  strategically the endpoint, but lower-fidelity for deep credit assignment and heavier than (b) for this de-risk.
- **C2 is not throwaway:** the same differentiable bridge-faithful trainer + the already-validated Phase-1.4 gate-freeze
  no-forget machinery compose into the C2 grow-without-forgetting primitive. Building (b) first delivers BOTH the
  consolidation fix and the C2 growth step in one artifact.

**Bottom line for the controller:** the surrogate-grad finetune is **surpassable cheaply** — run the HYBRID de-risk (train
the bridge-faithful `clip`-chain to distil per-layer activations → install → re-measure the existing graded metric;
day-scale, NO `sim/` edit, NO GPU-architecture risk). GO if installed cumulative analog-Spearman ≥ 0.8 with the
specificity margin re-opened and the shuffled-target control at chance. The differentiable-bridge `sim/` edit is the
bounded fallback on a measured NO-GO, and the trainer doubles as the C2 growth primitive regardless.

---

## 7. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / JSON read):**
- **The wall is deterministic per-layer `clip` compression, not noise** — `2026-06-22-genseq-loopstep3-popcode-NEGATIVE-
  deterministic-readout.md` line 3 (`within_pop_std_max=0.00e+00`), lines 9-13 (per-block `[0.846, 0.620, 0.288]`, FLAT
  in n_per), lines 19-22 (the graded `a_cont` readout is "deterministic and noiseless" → "deterministic signal
  compression through the stacked saturating `clip` nonlinearities"). Read in full.
- **The graded path un-saturates + carries L0 ≈ 0.846/0.865** — `2026-06-22-genseq-loopstep3-graded-PARTIAL-diagnosis-
  confirmed.md` table (graded best scale=20: per-block `[0.865, 0.596, 0.327]`, `a_cont` NOT saturated). Read in full.
- **The off-bridge GRADED target is `clip(a@W,0,1)`, deterministic + differentiable** —
  `research/runners/_genseq_loopstep3_graded_derisk.py:107-117` (`offbridge_graded_forward`: `a = np.clip(a @ Ws[L],
  0.0, 1.0)` per block); the on-bridge readout it matches is `a_cont = clip((v-rest)/scale,0,1)` (`:379`, `bridge.py:
  6144-6147`). Read in full.
- **The bridge is a FORWARD simulator with NO backward** — `_run_one_simulation_step` (`bridge.py:5715`+) executes
  imperative CSR matvecs + `fused_*` kernels + in-place CuPy updates; no autograd tape, no `.backward()`. The graded step
  block `bridge.py:6128-6175`.
- **Installing trained weights needs NO `sim/` edit** — `inject_explicit_wiring` builds the monolithic CSR from
  `initial_weights` (`bridge.py:2393` signature, `:2468` `all_w.extend([float(x) for x in group["initial_weights"]])`,
  `:2491` `coo_matrix(...).tocsr()`); `set_pathway_weights` overwrites edge weights in-place
  (`bridge.py:2986`, returns count updated).
- **The surrogate-grad + BPTT infra exists and is differentiable (standalone, not the bridge)** —
  `sim/surrogate_grad.py` (`atan_surrogate` `:22`, `fast_sigmoid_surrogate` `:45`, `softmax_grad` `:67`);
  `sim/bptt_snn.py` (`forward_unroll` `:89`, `backward_unroll` hard-reset surrogate + recurrent chain rule `:180-256`);
  `sim/bptt_snn_gpu.py` (CuPy port, fp32-validated vs numpy `:9-11`). Their forward is a SPIKING LIF
  `v = leak*v*(1-s)+x@W; s=Heaviside(v-thr)` (`bptt_snn.py:81-86`) — NOT the bridge's graded `clip(a@W,0,1)` analog path
  nor the `g·(V−E)` driving force.
- **(c) was ranked #3 in the prior scoping; the parent §4.2 fallback is the differentiable-bridge edit** —
  `2026-06-22-genseq-consolidation-past-saturation-scoping.md:183-202` (surrogate-grad-on-bridge "the deeper guarded
  `sim/` use, only if (a)+(b) miss"; "does NOT keep the trained net's weights verbatim — it changes them");
  `2026-06-22-genseq-loopstep3-consolidation-scoping.md:303-305` (parent §4.2 option 3 = "a LIF/AdEx-LIF forward
  consistent with `bptt_snn_gpu`", "NOT on the critical path unless the cheap calibration loses too much"). Read.
- **The convert step runs PyTorch/CUDA (so the HYBRID trainer reuses it)** — `2026-06-22-genseq-convert-GO-spiking-
  generates.md` §Scope ("`research/runners/_genseq_convert_derisk.py`, PyTorch/CUDA"). Read in full.
- **C2 no-forget machinery exists + is multi-seed-validated** — CLAUDE.md: Phase-1.4 BRANCH A (5/6 seeds ≥80% retention,
  mean 103%, `continual_forgetting_eval.py`); Phase-1.3 CLS consolidation (3/3 strict-anti-cheat); the per-pathway
  plasticity gate (`cp_plasticity_rate_gain`, `set_plasticity_gate`, the masked-clip fix `bridge.py:6673/6990/7253`);
  `sim/auto_growth.py` TierPromoter + weight-transfer; `sim/lineage.py` persistent state.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing]** That training the idealized `clip(a@W,0,1)` chain and installing yields cumulative
   analog-Spearman ≥ 0.8 *on the live bridge* — i.e. the `g·(V−E)` driving-force sub-linearity + AdEx settling + per-block
   gain do not re-open a gap the idealized trainer can't see. The de-risk MEASURES this directly (install → re-run the
   existing graded metric); the mitigation (fold the driving-force/affine term into the trainer forward, still NO edit)
   is in hand. **This is the one genuine uncertainty** — the trainer is faithful to the target representation, not yet
   proven byte-faithful to the live conductance dynamics.
2. **[VERIFY]** That the float transformer Gen-F's per-layer activations are cleanly extractable as the distillation
   target on the same 4-layer slice (the `cortex_10M_seed42.npz` is the *failed/overfit* BPTT generator per
   `2026-06-22-genseq-step0-C1-consolidation-GO.md` §Next; the *working* generator is the non-spiking Gen-F — confirm
   which net's activations are the intended teacher for the consolidation slice, and that its per-layer activations are
   exposed). The step-0/graded runners load `cortex_10M_seed42.npz`; the convert step converts Gen-F — reconcile.
3. **[VERIFY — fidelity ceiling]** That greedy LAYERWISE distillation (vs full end-to-end) recovers the *cumulative* (not
   just per-layer) score — greedy layerwise can leave cross-layer interactions on the table; the optional end-task polish
   is the named recovery, and the de-risk measures the cumulative directly.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-genseq-loopstep3-popcode-NEGATIVE-deterministic-readout.md` (the wall: deterministic
  `clip` compression `[0.846,0.620,0.288]`, popcode no-op, std=0.00 — the finding that motivates training-through — read
  in full).
- `research/findings/2026-06-22-genseq-loopstep3-graded-PARTIAL-diagnosis-confirmed.md` (graded un-saturates, L0=0.865,
  cumulative 0.327 — read in full).
- `research/runners/_genseq_loopstep3_graded_derisk.py` (`offbridge_graded_forward` `:107-117` = the differentiable
  target; signed-split wiring builder `:261-308`; `onbridge_block_analog` + fidelity metric + specificity anti-cheat
  `:343-587`; greedy per-block gain `:411-455` — read in full).
- `sim/bridge.py`: forward-only step (`:5715`+), graded analog step block (`:6128-6175`, `a_cont` `:6144-6147`),
  `inject_explicit_wiring` (`:2393`, `initial_weights` → CSR `:2468/2491`), `set_pathway_weights` (`:2986`), masked-clip
  no-forget fix (`:6673/6990/7253` per CLAUDE.md) — read the relevant spans.
- `sim/surrogate_grad.py` (`:22/45/67`), `sim/bptt_snn.py` (forward `:89`, backward `:180-256`, spiking forward `:81-86`),
  `sim/bptt_snn_gpu.py` (`:9-11/45-173`) — read in full (the differentiable STANDALONE SNN infra; NOT the bridge).
- `research/findings/2026-06-22-genseq-consolidation-past-saturation-scoping.md` (the prior scoping; (c) surrogate-grad
  ranked #3, `:183-202` — read in full).
- `research/findings/2026-06-22-genseq-loopstep3-consolidation-scoping.md` (parent ladder; §4.2 surrogate-grad fallback =
  differentiable-bridge edit, `:303-305` — read).
- `research/findings/2026-06-22-genseq-convert-GO-spiking-generates.md` (Gen-F = working generator, PyTorch/CUDA convert,
  `cortex_10M` = the failed BPTT net — read in full); `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md`
  (single-layer 0.918, the npz the consolidation loads — read).
- CLAUDE.md: Phase-1.4 BRANCH A no-forget (5/6 ≥80%); Phase-1.3 CLS consolidation (3/3 strict); plasticity-gate machinery;
  `sim/auto_growth.py` / `sim/lineage.py` (the C2 grow + persistence infra).

### Current literature / SOTA (verified via the parent scopings' primary-source pass + standard ML)
- **Continued spiking-LM training (the chosen path's precedent)** — SpikingBrain-7B, CAS/BICLab 2025, arXiv 2509.05276
  (the only spiking LM at scale; trains the converted spiking model, does not install verbatim).
- **Surrogate-gradient BPTT** — Neftci-Mostafa-Zenke 2019 ("Surrogate Gradient Learning in Spiking Neural Networks",
  arXiv 1901.09948); Zenke-Vogels 2021 (SuperSpike / ATan surrogate — the `sim/surrogate_grad.py` source).
- **ANN→SNN + activation/feature distillation (the layerwise-target recommendation)** — Hinton-Vinyals-Dean 2015 (KD,
  arXiv 1503.02531); the QCFS / activation-matching ANN→SNN family the convert step cites; quantization-aware training
  (train-through-the-nonlinearity) — Jacob et al. 2018, arXiv 1712.05877.
- **Local / backprop-free credit assignment (option c)** — feedback alignment (Lillicrap et al. 2016, Nat. Commun.);
  forward-forward (Hinton 2022, arXiv 2212.13345); e-prop (Bellec et al. 2020, Nat. Commun. — online surrogate-gradient
  with eligibility traces, the substrate-native local rule the bridge's eligibility infra already supports).
