# Scaled Subword Spiking Language Model — Design (Generator-S)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). User directive
> (2026-05-17): "do option 3, you autonomously research and select the
> mechanism... I expect you to go even a whole week working on your own
> with no input... primary constraints still stand: local hardware
> only, self-contained POST-training (no runtime external dep); a
> public/open-source training corpus off the internet IS authorized."
> Same non-negotiable anti-cheat: pre-registered held-out gate with
> permuted/shuffled controls, fixed bars never tuned post-hoc, honest
> negative propagated, no overclaiming, no config-cranking a terminated
> mechanism, self-contained at RUNTIME, local RTX 3090. Continuous
> autonomous arc — do NOT stop to ask between increments.

## Why this is genuinely different (NOT config-cranking terminated Inc-3)

Seven honest negatives terminated specific mechanisms. The single
converged diagnosis: **training-signal poverty under pure
self-containment** — the sim was its own weak, circular teacher
(self-comprehension judge: G1 AUC 0.775, G1.5 0.40, P 0.475;
order-intrinsic readback near-noise), and where a real gradient
existed it was over an impoverished self-distilled / repo-prose
corpus (Inc-1/2/3 char-BPTT: memorization, not held-out
generalization).

Inc-3's pre-registered terminus is precisely **char-level + impoverished
self-contained text → memorization**. Generator-S changes the *two
exact axes the Inc-3 diagnosis implicated*, not a config knob:

1. **Subword/BPE tokenization** (NOT char-level). Char models memorize
   surface sequences and were "phonetically confused" (Phase 2.3a
   NEGATIVE root cause). Subword tokens are the representational
   substrate every real LM uses; this is a *different language
   representation*, not a hyperparameter of Inc-3.
2. **A real, large, public corpus** (TinyStories / WikiText-103 — what
   open-weights LLMs train on), now explicitly authorized — NOT the
   "repo findings prose + self-distilled teacher" that *caused* Inc-3's
   memorization.
3. **Non-circular held-out generation gate** — held-out next-token
   perplexity + automatic generation-quality metrics, NEVER the
   circular self-comprehension judge (the diagnosed weakness of
   G1/G1.5/P) and NEVER train-loss (the Inc-3 memorization artifact).

It reuses ONLY the *validated* spiking BPTT core (Phase 2.1/2.2:
`cortex_pretraining.train_shakespeare`, `bptt_snn_gpu`,
`surrogate_grad` — loss 14.1→2.24 on Tiny Shakespeare, 41.5s on the
3090, infra reviewed sound). DRY: nothing in the validated core is
reimplemented. The genuinely-open, never-tested question: **does a
subword spiking LM trained by surrogate-grad BPTT on a real public
corpus at real scale generate coherent held-out text** — Phase 2.2
only proved the stack *learns* (train loss ↓); Phase 2.3a only tested
*feature-transfer to the 4-word binding task at 134K params*.
Generation at scale on a real corpus was never attempted.

## Evidence grounding (falsify-cheaply, BEFORE scaling big)

- Validated BPTT spiking core is ON main (commit 6a4d309): `sim/
  bptt_snn_gpu.py` (LIFLayerXP, forward/backward_unroll_xp, CuPy/numpy
  backend), `sim/surrogate_grad.py` (ATan), `sim/char_tokenizer.py`,
  `research/runners/cortex_pretraining.py` (`train_shakespeare`:
  init std 2.0 first / 0.5 later, per-sample CE→backward→SGD). 4
  green test files pin it.
- `data/tinyshakespeare.txt` (1.1 MB) is present locally → zero-network
  grounding corpus to validate the *scaled subword pipeline* end-to-end
  before pulling the large corpus.
- Hardware: Phase 2.2 = 41.5 s for a 134K-param toy. The 3090 (24 GB)
  has ~100–1000× headroom in params×corpus×context for a small spiking
  LM. Kill-safe atomic checkpoint pattern (`sim.train_checkpoint`)
  already exists and is reused (user games/resumes).

## Thesis

A **subword spiking neural language model**: BPE tokenizer (trained
once on the corpus → a static merge table = data, self-contained at
runtime), an N-layer LIF SNN trained by surrogate-gradient BPTT
(Neftci 2019, catalog) for next-subword-token prediction on a real
public corpus, generating text autoregressively from its own local
weights (sampling its rate-coded output distribution). Self-contained
at runtime: a trained SNN is just weights + the merge table; no
external LLM, no runtime corpus. Honest scope: target is *coherent
small-LM generation* (TinyStories-class), explicitly NOT GPT-4-class;
the framing never overclaims.

## Architecture (net-new is small; validated core reused unchanged)

Reuse UNCHANGED (DRY): `bptt_snn_gpu` forward/backward unroll +
surrogate gradient; `cortex_pretraining`'s per-sample CE/grad/SGD loop
shape; `sim.train_checkpoint` atomic kill-safe save/resume;
`song_g1_core`-style pure pre-registered verdict discipline (fixed
bars, permuted control) reused as the *pattern* (a new pure gate
module, not a reimplementation of g1's bars).

**Net-new (small, well-scoped):**
1. `sim/bpe_tokenizer.py` — a minimal, pure, deterministic byte-level
   BPE (train merges on corpus; encode/decode; save/load merge table
   as JSON). No external tokenizer dependency (self-contained). ~200
   lines, fully CPU-unit-testable.
2. `research/runners/corpus_fetch.py` — fetch + cache an authorized
   public corpus (TinyStories primary; WikiText-103 alt) to
   `data/corpus/`; deterministic clean + train/held-out split. Network
   ONLY at fetch time (training resource, not runtime). Idempotent
   (cached → no re-download).
3. `research/runners/scaled_subword_lm_train.py` — scaled multi-layer
   subword SNN trainer. The per-epoch loop is a DRY mirror of
   `cortex_pretraining.train_shakespeare` (same backend/init/CE/SGD);
   the ONLY additions: subword vocab from the BPE tokenizer, CLI
   layer/width/T/context, atomic per-epoch checkpoint + auto-resume,
   OOM batch-halving, KeyboardInterrupt→clean checkpointed exit.
4. `research/runners/subword_lm_gate.py` + a pure scoring core — the
   pre-registered held-out generation gate (below). Pure scoring is
   CPU-TDD'd; the runner is import/signature-smoke + the gate itself.

## Data flow

Fetch+cache corpus → train BPE once (merge table cached) → tokenize
train/held-out splits → scaled SNN BPTT pretraining (kill-safe,
background) → autoregressive generation (sample rate-coded output) →
pre-registered held-out gate (perplexity + controls) → honest
propagation → next increment.

## Pre-registered anti-cheat gate (FIXED bars, never tuned)

Cheap-first decisive slice (scoped like prior B-probes — this slice's
multi-seed gate decides whether the line is pursued; NOT a months
buildout):

1. Train the scaled subword SNN on the real corpus train split to
   convergence-or-budget (kill-safe).
2. **Held-out next-token perplexity** on the untouched held-out split
   (never trained on).
3. **Pre-registered FIXED bars (never tuned post-hoc):**
   - **G-S.A (real-structure):** held-out perplexity ≥ **20%** lower
     than a **shuffled-token control** (same model trained identically
     on token-shuffled corpus — kills n-gram/memorization artifacts),
     AND
   - **G-S.B (generalization):** held-out perplexity within **1.5×** of
     train perplexity (NOT pure memorization — the Inc-3 failure mode),
     AND
   - **G-S.C (generation non-degeneracy):** greedy + sampled
     continuations of held-out prompts have distinct-trigram ratio ≥
     **0.5** and ≤ **20%** verbatim n-gram copy from train (not
     degenerate repetition, not corpus regurgitation).
   - **Multi-seed ≥3** (single-seed is NOT a pass; same discipline as
     the whole arc).
4. The bars (0.20 / 1.5× / 0.5 / 0.20) are FROZEN in a pure module the
   moment this design is committed, in a sidecar, and NEVER recomputed
   at gate time. A pure `subword_lm_gate_core` holds only
   decode/perplexity/control/verdict glue; the verdict is computed by
   fixed-bar logic mirrored from the `song_g1_core` discipline
   (pattern reuse, bars are this gate's own pre-registered constants,
   not g1's 0.10/0.5).
5. **LOAD-BEARING no-harm:** Generator-S is a *separate* trainer/model
   — it does NOT touch the validated grounded-memory bridge, G.20
   ensembles, or any shipped runner. No-harm here = "purely additive
   new files; the validated deliverable is byte-untouched" (verified
   by: no edits to sim/bridge.py, g20_*, the validated runners; full
   existing test suite stays green).

PASS ⇒ a self-contained subword spiking LM genuinely generates
coherent held-out text → scale further (bigger corpus/params/context;
then Generator-C: wire this spiking language cortex onto the validated
grounded-memory arch). FAIL ⇒ honest negative propagated; this
*specific* mechanism (subword spiking BPTT generation at this scale)
is recorded falsified; **immediately proceed autonomously to the next
genuinely-different mechanism** (e.g. B: distillation from a local
open-weights teacher into the spiking net; or a different
catalog-grounded sequence substrate) — do NOT stop, do NOT
config-crank Generator-S.

## Honest ceiling / risks (no overclaiming)

- Spiking LMs are harder to train than ANN LMs; held-out fluency may
  plateau below ANN parity. The gate measures *real held-out
  generative generalization vs. controls*, not SOTA parity — a maxed
  honest FAIL is a real finding and triggers the next mechanism, not a
  config-crank.
- Scope is small-LM coherent generation (TinyStories-class), NOT
  GPT-class; deliverable framing never overclaims "an LLM."
- Self-contained at RUNTIME is preserved (weights + static BPE merge
  table only); the corpus and any teacher are TRAINING-time only.
- Hardware: 3090; kill-safe resumable; long runs background; user
  games/resumes; ASCII-only prints (Windows cp1252).

## Pre-staged next mechanisms (no stopping between)

- **Generator-S PASS →** Generator-C: corpus-pretrained spiking
  language cortex integrated with the validated grounded-memory +
  no-confabulation arch (Phase 2.3 done right at scale).
- **Generator-S FAIL →** Generator-D: knowledge distillation from a
  *local* open-weights teacher (Phi-3/Llama-3.2/Qwen2.5, training-time
  teacher only; runtime = trained spiking net) — strongest possible
  training signal, explicitly pre-registered, same gate discipline.
- Both pre-staged so the autonomous arc never stalls on a verdict.

## Scientific basis (catalog)

Surrogate-gradient BPTT for spiking nets (Neftci/Mostafa/Zenke 2019);
BPE subword tokenization (Sennrich 2016); rate-coded readout;
Pulvermüller distributed cortical word ensembles (the integration
target); Marr/McClelland CLS (Generator-C no-forgetting). Curriculum
/ corpus scale per the open-weights small-LM literature (TinyStories,
Eldan & Li 2023 — small models CAN generate coherent English).

## Out of scope (YAGNI)

No external LLM/corpus/teacher at RUNTIME ever. No config-cranking any
terminated mechanism (Inc-1/2/3 char-BPTT, G1/G1.5/P controllers,
order-intrinsic). No char-level (the diagnosed failure substrate). No
months-class buildout in this slice — the pre-registered multi-seed
gate decides; FAIL → next pre-staged mechanism immediately.
