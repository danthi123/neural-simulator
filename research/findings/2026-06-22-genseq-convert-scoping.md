# Generative-sequence frontier (Spine A) — CONVERT scoping: ANN→SNN conversion of the working non-spiking Gen-F into a SPIKING generator that still generates coherent NOVEL text (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings/literature scoping for the NEXT step of the generative-sequence
> frontier (Spine A, loop-step 3 "consolidate"'s upstream prerequisite). **NO `sim/` edits, NO training, NO GPU.**
> Single deliverable = this doc. Every load-bearing project claim re-verified against the repo (file:line). SOTA bounded
> by a fresh June-2026 literature pass (MBE / LAS / ECMT / QCFS, abstracts + full-text read). Builds on the parent
> frontier scoping (`2026-06-22-generative-sequence-frontier-scoping.md`, §C1/§3b) and step-0 GO
> (`2026-06-22-genseq-step0-C1-consolidation-GO.md`); does NOT re-derive them. The controller should trust-but-verify
> the **[VERIFY]** items, then push + present before building. This is a SCOPING/DECISION doc, NOT a brain-based result
> and NOT a commitment to build.

---

## 0. One-paragraph answer (the rest is the evidence)

**The convert step is MUCH cheaper and lower-risk than the parent scoping assumed — because Gen-F is a *standard
PyTorch decoder-only GPT* (`nn.MultiheadAttention` + `LayerNorm` + `GELU`), which is exactly the architecture the 2025
SOTA converts *training-free and near-losslessly to spikes*.** The parent scoping treated the convert as an open
sub-problem because it conflated Gen-F with the project's stacked-LIF BPTT nets; the actual `sim/tiny_transformer.py`
artifact is a vanilla GPT (verified `tiny_transformer.py:11-64`), and the trained checkpoint is loadable and **3.45M
params** (not the "~6M" the finding rounds to — exact param count verified). Three independent 2025/2024 methods —
**MBE** (training-free, AAAI 2026: GPT-2 WikiText-2 22.34→22.69 ppl, **+1.57%** at T=16), **LAS** (conversion-only,
**code released**: GPT-2 WikiText-103 16.53→16.79, **+1.6%**, T=16), and **ECMT** (ACM-MM 2024, **code released**) —
each explicitly converts the three hard nonlinear ops (Softmax, LayerNorm, GELU) and preserve the pretrained weights
**with no gradient on the source model**. So the cheapest path is: keep Gen-F's trained weights verbatim, run a
training-free activation-range *calibration* (a minibatch, no backprop), and realize attention/LN/GELU as spiking
operators — landing at ≤~3% perplexity cost. **The decisive cheap de-risk is one afternoon's run**: load the shipped
`generator_f_gate.ckpt.s42.real.pt` checkpoint, swap the model's `forward` for a *spiking-rate* forward (a faithful
discrete-time approximation of softmax/LN/GELU at T=16), and run the **byte-unmodified Gen-F gate** (`generator_f_gate
.py` + the frozen `subword_lm_gate_core.py` bars 0.20/1.5/0.5/0.20 + abs-competence floor) — does the SPIKING version
still clear held-out ppl < vocab, beat the word-shuffle control, and produce non-degenerate coherent novel text (vs the
measured 0-novel composer wall)? **GO** = spiking ho-ppl within ~10–20% of the 6.1 ANN ppl AND still clears the gate's
relative + novelty + copy bars, 3 seeds. The genuine residual risk is real but bounded: **every SOTA paper reports
*perplexity*, not *generation coherence*** — a 3% ppl rise can disproportionately hurt free sampling — and the bridge
has **no LIF neuron model** so the on-bridge *consolidation* (a later step) still has the named dynamics gap step-0 sized
to a single gain calibration at 0.92 fidelity. The fallbacks are ordered and cheap-first: (1) raise T (8→16→32, the
standard latency/fidelity knob); (2) post-conversion surrogate-grad finetune of the spiking forward (a small guarded
`sim/` edit); (3) if attention *specifically* refuses to convert cheaply, fall back to a **feedforward** generator,
which step-0 already showed converts at **0.92** with a single scalar — but note a feedforward LM is a real capability
downgrade from Gen-F's attention-based coherence.

---

## 1. What IS Gen-F's architecture, EXACTLY (verified against the code, not the finding's rounding)

**Gen-F is a standard PyTorch decoder-only GPT — NOT spiking, NOT the project's LIF/BPTT stack.** Verified by reading
the actual module `sim/tiny_transformer.py`:

| Property | Value | Source (file:line) |
|---|---|---|
| Class | `TinyGPT(nn.Module)` — decoder-only causal LM | `tiny_transformer.py:35` |
| Layers | **4** transformer blocks (`n_layer=4`) | `tiny_transformer.py:36,48` |
| Width | **d_model=256** | `tiny_transformer.py:36` |
| Attention | **YES** — `nn.MultiheadAttention(d, n_head=4, batch_first=True)`, causal mask via `torch.triu(...,diagonal=1)` | `tiny_transformer.py:15,25-31,36` |
| Norm | **LayerNorm** ×3 per block (ln1, ln2) + final lnf | `tiny_transformer.py:13,16,28,32,50` |
| MLP | `Linear(d,4d) → GELU → Linear(4d,d) → Dropout` | `tiny_transformer.py:18-20` |
| Block | pre-norm residual: `x + attn(ln1(x)); x + mlp(ln2(x))` | `tiny_transformer.py:28-32` |
| Embeddings | token `nn.Embedding(V,d)` + learned positional `nn.Embedding(block,d)` | `tiny_transformer.py:44-45,60-61` |
| Head | `Linear(d, V, bias=False)` (untied) | `tiny_transformer.py:51` |
| Context length | **block_size = 128** tokens | `tiny_transformer.py:36,55-59` |
| Tokenizer | **BPE, vocab 512** (`sim/bpe_tokenizer.py`; `encode`/`decode`/`vocab_size`/`load`/`save` present) | `tiny_transformer_train.py:36,52,58`; `bpe_tokenizer.py:73-115` |
| **Params (REALIZED)** | **3,454,976 (3.45M)** — verified two ways: `sum(v.numel())` over the shipped state_dict = 3,454,976, and the closed-form `tok+pos+4·block+lnf+head` for V=513/d=256/L=4 = 3,454,976 | checkpoint `generator_f_gate.ckpt.s42.real.pt` |
| Training | **full all-positions autoregressive** next-token CE (`y=data[i+1:i+1+block]`, `F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))`) — AdamW lr 3e-4, cosine, 12000 steps, batch 64 | `tiny_transformer_train.py:106,134-135,68-69,201-208` |
| Corpus | **TinyStories** (`data/corpus/tinystories.txt`, 8 MB; gate default `--corpus tinystories`) | `generator_f_gate.py:105,160-168`; corpus file on disk |
| Held-out ppl | **~6.1** (vs uniform-random 513 → ~84× better); coherent story-shaped English | `2026-05-17-generator-F-...PASS.md` |
| Checkpoint | loadable `{model, optim, sched, step:12000, loss_history, torch_rng}`, 41.6 MB (weights+AdamW state); seeds 42/43/44 + paired `.bpe.json` + `.ctl` word-shuffle controls all on disk | `tiny_transformer_train.py:112-117`; files verified |

**Two honest corrections this scoping inherits:**
1. **Param count: the finding says "~6M" but the realized net is 3.45M.** d=256/L=4/V=512/block=128 mathematically and
   empirically yields 3.45M. (~6M would need d≈360 or L≈8.) Report 3.45M; it does not change the conclusion (still in
   the TinyStories <10M coherent-English regime), but the controller should correct the round number.
2. **The full-autoregressive-loss "gap" the parent scoping flagged is NOT in Gen-F.** The parent scoping warned that
   `cortex_pretraining.py:259` trains on the **last-position target only** — but that is the *BPTT-LIF* trainer, a
   *different* artifact. **Gen-F's trainer uses the all-positions loss** (`tiny_transformer_train.py:106,134-135`), so
   Gen-F is already a proper next-token LM. The "fix the last-position limitation" item applies to Spine B (spiking
   from-scratch), not to converting Gen-F.

**Why the architecture is decisive for convert difficulty:** Gen-F has exactly the three "hard" nonlinear ops a
Transformer→SNN conversion must handle — **scaled-dot-product softmax attention, LayerNorm, and GELU** — and nothing
exotic (no RoPE, no RMSNorm, no MoE, no flash-attn fusion). That is the *canonical* target of the 2025 conversion
literature (§2), so the convert is a SOLVED-class problem, not a research frontier.

---

## 2. The cheapest ANN→SNN conversion path that PRESERVES generation, specific to Gen-F

**Gen-F is a Transformer, so the hard parts are attention/softmax + LayerNorm + GELU — and these are exactly what the
2025 SOTA solves training-free.** Ranked cheapest-first:

### Rank 1 (cheapest, the recommended de-risk path) — training-free activation-calibration conversion (MBE / LAS / ECMT class)

Keep Gen-F's trained weights **verbatim**; convert the *activations* to a discrete-time spiking/rate form at a small
timestep budget T, with a one-pass **calibration** (record the input ranges of each nonlinear op on a TinyStories
minibatch — **no gradient on the model**). The three published methods each handle Gen-F's three ops:

| Op in Gen-F | How the SOTA converts it | Method(s) |
|---|---|---|
| **Softmax** (attention) | change-of-base `eˣ = 2^⌊x·log₂e⌋·2^frac` (integer part = hardware add, fractional part = basis neurons); reciprocal via IEEE-754 mantissa approx + spike-multiply (MBE). LAS reconstructs max-subtraction incrementally across T then approximates exp + reciprocal with Hierarchically-Gated (HG) neurons. | MBE (2508.07710), LAS (2505.09659) |
| **LayerNorm** | decompose into mean–variance normalization + inverse-sqrt scaling; inv-sqrt approximated by basis/HG neurons; variance via spike-based FP multiply | MBE, LAS |
| **GELU** (MLP) | piecewise/basis approximation over partitioned input sub-intervals (MBE: N=4 bases, M=10k sample points; LAS: HG sub-neurons per range) | MBE, LAS, ECMT (Multi-Threshold neuron) |

**Measured cost on GPT-2 (Gen-F's exact family), at T=16 timesteps, training-free, weights preserved:**
- **MBE:** WikiText-2 22.34→**22.69** (+0.35 ppl, **+1.57%**); WikiText-103 22.65→23.41 (+3.36%). [VERIFY — full-text
  read this pass.]
- **LAS:** WikiText-103 16.53→**16.79** (+0.26 ppl, **+1.6%**); 13–16 steps suffice, ≤11 degrades. **Code:
  `github.com/lc783/LAS`.**
- **ECMT:** first high-accuracy Transformer→SNN conversion (ACM-MM 2024), Expectation-Compensation + Multi-Threshold
  neuron, ~1% accuracy loss at 4 steps on ViT. **Code: `github.com/h-z-h-cell/Transformer-to-SNN-ECMT`.**

⇒ **For Gen-F specifically, the expected conversion cost is ≤~3% perplexity at T≈16, training-free.** That is the
cheapest path and it is what the de-risk (§3) should implement — start by *porting the math* (a discrete-time spiking
forward for `TinyGPT`), reusing LAS/ECMT released code as the reference, with NO change to Gen-F's weights and NO `sim/`
edit (the de-risk runs the converted forward in PyTorch, not on the bridge — bridge consolidation is the *separate*
later loop-step that step-0 already de-risked).

### Rank 2 — QCFS-style activation replacement + light recalibration (the feedforward-friendly classic)

QCFS (quantization-clip-floor-shift, Bu et al. 2023, **code `github.com/putshua/SNN_conversion_QCFS`**) replaces ReLU
with a clip-floor activation whose expected conversion error to a rate-SNN is zero, at ultra-low T (4 steps). It is the
parent scoping's named "threshold-balancing/QCFS" path and is the *standard* for the feedforward parts (the MLP/head);
it does **not** by itself solve softmax/LN (Rank-1's contribution). Use as the activation-replacement substrate that
Rank-1's nonlinear-op handlers sit on top of. Marginally more involved than Rank-1's pure post-hoc calibration because
QCFS canonically wants the activation swapped *before* a short recalibration.

### Rank 3 (robust fallback, a guarded `sim/` edit) — post-conversion surrogate-grad finetune

If Rank-1 calibration alone loses too much *generation* quality (ppl OK but sampling degrades — the genuine residual
risk, §4), do a few epochs of surrogate-grad BPTT on the *spiking* forward (the project already has the surrogate-grad
machinery: `sim/surrogate_grad.py` ATan/fast-sigmoid, `sim/bptt_snn_gpu.py`). This is the most robust fidelity fix and
the justified place for a small additive, default-off, byte-identical-when-unused guarded edit — but it is a *fallback*,
not the first move (it costs a training run; Ranks 1–2 do not).

**Why this ordering (not the parent scoping's "convert is an open sub-problem"):** the parent scoping ranked conversion
generically because it had not yet established Gen-F is a vanilla GPT. With that established, the convert collapses to
"apply a 2025 training-free Transformer→SNN method to a 3.45M GPT" — a published, code-released, ≤3%-ppl operation. The
sub-problem is *engineering a faithful discrete-time forward + calibration*, not *research*.

---

## 3. The decisive cheap CONVERT de-risk — "does the converted SPIKING Gen-F still generate coherent novel text?"

**Goal:** the smallest experiment that answers the one open question, reusing the shipped Gen-F checkpoint + the
byte-frozen Gen-F gate, with **no retraining of the LM** and (at this stage) **no bridge / no `sim/` edit**.

### Design (cheapest-first, PyTorch-only)

- **Input:** the **already-trained** `generator_f_gate.ckpt.s{42,43,44}.real.pt` + matching `.bpe.json` (on disk;
  3.45M-param GPT, ho-ppl 6.1). NO retrain.
- **The convert:** implement a *spiking-rate* forward for `TinyGPT` at timestep budget T (start T=16) — a faithful
  discrete-time approximation of softmax, LayerNorm, GELU (the Rank-1 math; reference LAS/ECMT released code), with
  Gen-F's weights frozen and copied verbatim. One-pass **calibration** records each nonlinear op's input range on a
  TinyStories train minibatch (no gradient). This lives in a *new runner* `_genseq_convert_genf_probe.py` (reuse-only;
  imports `TinyGPT`, `BPETokenizer`, and the gate functions).
- **The metrics — reuse the BYTE-UNMODIFIED Gen-F gate verbatim** (`generator_f_gate.py` + the frozen
  `subword_lm_gate_core.py` bars; the gate's `_heldout_nll`/`_generate`/`distinct_ngram_ratio`/`verbatim_copy_fraction`
  already accept any object with a `model(x)`-style forward — swap the ANN `model` for the converted spiking forward).
  Compute on the held-out TinyStories split, per seed:
  1. **held-out ppl (spiking)** vs the ANN's 6.1 and vs **abs-competence floor uniform_ppl = vocab (513)** — the
     pre-registered FAIL-CLOSED bar (`gs_verdict`, `subword_lm_gate_core.py:70-74`). MUST clear < 513.
  2. **conversion-loss delta:** spiking ho-ppl / ANN ho-ppl (the SOTA reports ≤~1.03; report it directly).
  3. **real-structure-vs-shuffle:** spiking ho-ppl ≤ 0.8 × the word-shuffle control's ppl (the load-bearing
     `_GS_PPL_MARGIN=0.20` bar; the `.ctl` shuffle checkpoints are on disk).
  4. **generalization:** spiking ho-ppl ≤ 1.5 × spiking train-ppl (`_GS_GENERALIZATION_MAX`).
  5. **non-degenerate:** distinct-trigram of generated ids ≥ 0.5 (`_GS_DISTINCT_MIN`).
  6. **not-copying:** verbatim 8-gram copy fraction ≤ 0.20 (`_GS_COPY_MAX`).
- **The NOVELTY check (vs the measured 0-novel wall):** the composer's measured wall is novel-composition **0.0 / ratio
  1.0** (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) — a *retrieval* system that emits only stored
  facts. Gen-F is a *free LM*, so the correct novelty evidence is: (a) generated text is **non-degenerate** (distinct
  ≥0.5) AND **not verbatim copy** (≤0.20) AND **beats the word-shuffle control** — i.e. it produces genuinely-novel
  grammatical sequences it was never shown verbatim, the categorical opposite of the composer's 0-novel. Concretely
  assert: the converted model, prompted with a held-out 8-token prefix, generates a continuation whose 8-grams are
  largely **absent from the training corpus** (copy ≤0.20) yet **coherent** (clears the gate). That is "still generates
  coherent NOVEL text" operationalized with the existing bars — no new metric invented.
- **Anti-cheats (carry verbatim):** the abs-competence floor is FAIL-CLOSED (the Gen-S false-PASS lesson, baked into
  `gs_verdict`); the word-shuffle control is the load-bearing one (`generator_f_gate.py:20-26`, checkpoints on disk);
  **≥3 seeds mandatory and unbypassable** (`gs_aggregate_multiseed`, can only strengthen); bars **never tuned**
  (recomputed from JSON); decode-and-read the actual generated text and characterize its true coherence ceiling (the
  Gen-F smell-test discipline — scrutinize a PASS *harder* than a FAIL).

### GO / NO-GO

- **GO** if, **3/3 seeds**: spiking ho-ppl < 513 (abs floor) AND spiking-ppl/ANN-ppl ≤ ~1.2 (conversion within ~20%;
  the SOTA gets ~1.03 — 20% is a generous bridge-agnostic margin for a first faithful port) AND the gate's
  real-structure + generalization + non-degenerate + not-copying bars all clear AND the decoded text is coherent
  story-shaped English (the Gen-F ceiling). ⇒ a SPIKING generator that still generates coherent novel text → proceed to
  loop-step 3 (consolidate onto the bridge — step-0-de-risked path).
- **PARTIAL** if ppl clears the abs floor but the relative/coherence bars degrade (e.g. distinct collapses, or
  ppl/ANN > 1.2) → escalate cheap-first: raise T (16→32), then Rank-3 surrogate-grad finetune (the guarded edit).
- **NO-GO** if the spiking forward cannot beat uniform-random ppl at any feasible T → the conversion math is wrong for
  this regime; debug the nonlinear-op approximators against the LAS/ECMT reference before any further investment.

### Local cost (3090)

**Hours, CPU-or-1×3090, NO training.** The convert is inference + a calibration minibatch; the gate's eval is
teacher-forced ppl over ~2000 windows + 200-token sampled generation × 3 seeds. The shipped checkpoints remove the
12000-step train cost entirely. `SIM_BACKEND` is irrelevant at this stage (PyTorch forward, no bridge). **No cloud.**

---

## 4. Honest risk + fallback

### Where conversion typically loses generation (the genuine residuals)

1. **[VERIFY — most load-bearing] The SOTA reports PERPLEXITY, not GENERATION coherence.** A +1.6–3.4% ppl rise is
   near-lossless for *teacher-forced* scoring, but free autoregressive *sampling* compounds per-token error over 200
   tokens, and small distributional shifts near the argmax can disproportionately degrade coherence (repetition loops,
   topic drift). **No paper this pass directly measured generated-text quality post-conversion** — they measure ppl /
   downstream accuracy. So "ppl converts at ≤3%" does **not** guarantee "generation stays coherent"; the de-risk's
   distinct-trigram + copy-fraction + read-the-text checks are precisely there to catch this. This is the honest open
   question the de-risk answers.
2. **Attention / long-range dependencies are the costliest op.** Softmax conversion (incremental max-subtraction +
   exp + reciprocal across T) is the most timestep-hungry part; at low T it degrades first, and Gen-F's coherence is
   attention-borne. The MBE paper's named failure modes apply: **Global Sub-Optimality** (basis neurons fit poorly in
   the high-curvature near-zero region where Transformer activations concentrate) and **Excessive Dependence on
   Initialization**. Mitigation = raise T (the standard knob) — but T multiplies inference cost.
3. **The ppl→generation-quality gap at the TinyStories ceiling.** Gen-F is *already* at its coherence ceiling
   (locally-grammatical, globally-wandering — `2026-05-17` finding); any conversion degradation eats into a thin margin,
   so a 3% ppl rise could visibly worsen the (already-imperfect) global coherence even while clearing the gate.
4. **The bridge dynamics gap (deferred to consolidation, not this de-risk).** The bridge has **NO LIF**
   (`enums.py:8-15` = IZHIKEVICH/HH/ADEX/RESONATE_AND_FIRE — verified). Converting Gen-F to a *generic* spiking forward
   (this de-risk) is bridge-agnostic; realizing it *on the bridge* (loop-step 3) re-introduces the LIF↔AdEx/Izh/RF gap.
   Step-0 already sized that to a **single global gain calibration → 0.92 fidelity, no `sim/` edit**
   (`2026-06-22-genseq-step0-...GO.md`), but step-0 was layer-0/one-hot/positive-weights-only; full multi-layer +
   signed (E/I) attention weights on the bridge is the named downstream concern. **Conversion produces *signed* attention
   logits and weights** — and step-0 flagged signed-weight routing as an open conversion concern — so this is the
   load-bearing fidelity question for the *consolidation* step that follows this de-risk.

### Fallbacks (ordered cheap-first)

| If… | Fallback | Cost |
|---|---|---|
| ppl clears floor but coherence degrades at T=16 | **raise T** (16→32→64) — the standard latency/fidelity knob | inference-only, hours |
| coherence still degrades at high T | **Rank-3: post-conversion surrogate-grad finetune** of the spiking forward (reuse `surrogate_grad.py`/`bptt_snn_gpu.py`); justified small guarded default-off `sim/` edit IFF the finetune must run with the bridge's neuron model in the forward | a finetune run (≤ hours–day, 3090) |
| **attention specifically** won't convert cheaply (softmax is the wall) | **pretrain a small FEEDFORWARD generator instead of a Transformer** — step-0 already showed a feedforward LIF net converts at **0.92 with a single scalar gain, NO `sim/` edit**. **HONEST COST:** a feedforward LM (n-gram-window MLP) is a real capability *downgrade* — it loses Gen-F's attention-borne multi-sentence coherence (cf. Gen-E n-gram = local fragments only). Use only if attention is truly the blocker; it trades the coherence Gen-F won for conversion-ease. | a new pretrain (hours–≤30 h) + its own gate |
| the whole convert underperforms vs just keeping the ANN | **Spine A's honest scope check:** the deliverable is *a spiking generator on the one bridge* (C1). If conversion genuinely can't preserve generation, the honest finding (per BRAIN-BASED-ONLY) is that the point-/rate-spiking realization of a Transformer generator has a measured cost — itself a publishable CLS/neuromorphic-translation result, and the trigger to weigh Spine B (spiking-from-scratch, the cloud path) vs accepting the documented cost. | — |

---

## 5. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / file read in full):**
- Gen-F = vanilla PyTorch decoder-only GPT (MHA + LayerNorm + GELU, causal mask, learned pos-emb, untied head):
  `sim/tiny_transformer.py:11-78`, read in full.
- Gen-F realized **3.45M params** (not ~6M): `sum(numel)` over `generator_f_gate.ckpt.s42.real.pt` model dict =
  3,454,976, AND closed-form param math = 3,454,976. Checkpoint loadable `{model,optim,sched,step:12000,...}`.
- Gen-F trains with **all-positions autoregressive CE** (`tiny_transformer_train.py:106,134-135`) — NOT the
  last-position `cortex_pretraining.py:259` limitation (which is a *different* artifact). Read in full.
- Corpus = **TinyStories** (`data/corpus/tinystories.txt` on disk, 8 MB; gate `--corpus tinystories` default
  `generator_f_gate.py:105`); `gen_f_train.txt` head is coherent TinyStories.
- The **frozen gate bars** the de-risk reuses: `subword_lm_gate_core.py` — 0.20/1.5/0.5/0.20 + abs-competence floor
  `uniform_ppl` (`:10-26,61-107`); ≥3-seed unbypassable (`:110-123`). The gate harness `generator_f_gate.py`
  (`_heldout_nll:29`, `_generate:64`, word-shuffle control `:20-26`, checkpoints+`.ctl` on disk). Read in full.
- The measured 0-novel composer wall (the novelty baseline the converted LM must beat): novel-composition 0.0 / ratio
  1.0, moat 20/20 — `2026-06-22-generation-novelty-categorical-gap-MEASURED.md`, read in full; novelty metric defs in
  `generation_novelty_probe.py:45-90`.
- C1 facts: `inject_explicit_wiring` at `bridge.py:2393`; `NeuronModel` has **no LIF** (`enums.py:8-15`); step-0
  consolidation GO at 0.92 via single gain, layer-0/one-hot/positive-only caveats —
  `2026-06-22-genseq-step0-C1-consolidation-GO.md` + `_genseq_step0_bridge_load_probe.py`, read in full.
- BPE tokenizer API present (`encode`/`decode`/`vocab_size`/`load`/`save`): `bpe_tokenizer.py:73-115`.

**SOTA (fresh June-2026 web pass; abstracts + key full-texts read):**
- **MBE** (training-free Transformer→SNN, AAAI 2026 / arXiv 2508.07710): GPT-2 WikiText-2 22.34→22.69 (+1.57%),
  WikiText-103 22.65→23.41 (+3.36%), T=16, training-free (calibration-only, no source-model gradient); handles
  Softmax/LayerNorm/GELU via Multi-basis Exponential Decay neurons; failure modes EDI + GSO (near-zero curvature).
  **Full-text read this pass.**
- **LAS** (loss-less ANN-SNN for spike-driven LLMs, AAAI 2026 / arXiv 2505.09659): conversion-only, GPT-2 WikiText-103
  16.53→16.79 (+1.6%), T=16 (13–16 suffice), HG neurons for softmax/LN/GELU; **code `github.com/lc783/LAS`**.
  Full-text read.
- **ECMT** (ACM-MM 2024): first high-accuracy Transformer→SNN conversion, Expectation-Compensation + Multi-Threshold
  neuron, ~1% loss at 4 steps; **code `github.com/h-z-h-cell/Transformer-to-SNN-ECMT`**.
- **QCFS** (Bu et al. 2023): quantization-clip-floor-shift activation, zero expected conversion error, 4-step latency;
  **code `github.com/putshua/SNN_conversion_QCFS`** — the feedforward-activation substrate.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing] That conversion preserves *generation coherence*, not just perplexity.** Every SOTA
   number is ppl/accuracy; none directly measured post-conversion free-generation quality. This is precisely what the
   §3 de-risk's distinct/copy/read-the-text checks measure — it is the hypothesis the experiment tests, not a settled
   result.
2. **[VERIFY] The exact spiking-forward math for Gen-F's specific ops at T=16** is taken from the MBE/LAS abstracts +
   full-text, not re-implemented this pass. The de-risk should reference the LAS/ECMT released code as the
   implementation oracle rather than re-deriving.
3. **[VERIFY — deferred to consolidation] Signed (E/I) attention-weight routing on the bridge.** Step-0 was
   positive-weights-only; conversion produces signed weights, and the bridge routes an excitatory source's negative
   weights on the same channel (wrong sign vs a Transformer summing signed weights). This is the load-bearing fidelity
   question for loop-step 3 (consolidate), NOT for this convert de-risk (which runs in PyTorch).
4. **The "~6M" → 3.45M correction** is verified here; the controller should propagate it (and that the
   last-position-loss limitation does not apply to Gen-F).

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `sim/tiny_transformer.py` (the Gen-F architecture — `TinyGPT`, MHA+LN+GELU, `:11-78`).
- `research/runners/tiny_transformer_train.py` (the trainer — all-positions AR loss `:106,134-135`; AdamW `:68`;
  checkpoint `:112-117`).
- `research/runners/generator_f_gate.py` (the gate harness + word-shuffle control `:20-26`; `_heldout_nll:29`,
  `_generate:64`; `--corpus tinystories :105`).
- `research/runners/subword_lm_gate_core.py` (the FROZEN bars `:10-26,61-123`).
- `sim/bpe_tokenizer.py` (`:73-115`), checkpoints `generator_f_gate.ckpt.s{42,43,44}.{real,ctl}.pt` + `.bpe.json` +
  `data/corpus/tinystories.txt` (all on disk).
- `research/findings/2026-05-17-generator-F-small-transformer-LM-PASS.md` (Gen-F PASS, ho-ppl 6.1, coherence ceiling).
- `research/findings/2026-06-22-generation-novelty-categorical-gap-MEASURED.md` (the 0-novel wall) +
  `research/runners/generation_novelty_probe.py` (`:45-90`).
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md` +
  `research/runners/_genseq_step0_bridge_load_probe.py` (C1 install path, 0.92 fidelity, single-gain calibration, the
  named caveats).
- `research/findings/2026-06-22-generative-sequence-frontier-scoping.md` (the parent frontier scoping; §C1/§3b SOTA).
- `sim/bridge.py` (`inject_explicit_wiring:2393`), `sim/enums.py` (`NeuronModel:8-15` — no LIF),
  `sim/surrogate_grad.py` + `sim/bptt_snn_gpu.py` (the Rank-3 finetune machinery).

### Current literature (June 2026 pass)
- **MBE — Training-Free ANN-to-SNN Conversion for High-Performance Spiking Transformers** — Wang et al., arXiv
  2508.07710 (AAAI 2026): GPT-2 WikiText-2 +1.57% / WikiText-103 +3.36% at T=16, training-free; Softmax/LayerNorm/GELU
  via MBE neurons.
- **LAS — Loss-less ANN-SNN Conversion for Fully Spike-Driven LLMs** — Chen et al., arXiv 2505.09659 (AAAI 2026):
  conversion-only, GPT-2 WikiText-103 +1.6% at T=16; HG neurons; **code github.com/lc783/LAS**.
- **ECMT — Towards High-performance Spiking Transformers from ANN to SNN Conversion** — Huang et al., ACM-MM 2024
  (~1% loss at 4 steps; Expectation-Compensation + Multi-Threshold neuron; **code
  github.com/h-z-h-cell/Transformer-to-SNN-ECMT**).
- **QCFS — Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency SNNs** — Bu et al., 2023 (clip-floor-shift
  activation, 4-step latency; **code github.com/putshua/SNN_conversion_QCFS**).
- **SpikeGPT** — Zhu et al., arXiv 2302.13939 (the Spine-B direct-spiking-training reference, 216M → WikiText-2 ppl
  18–19); **TinyStories** — Eldan & Li, arXiv 2305.07759 (<10M params → coherent multi-paragraph English).
