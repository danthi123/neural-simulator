# Generator-F — Small From-Scratch Transformer LM (self-contained, local, no-cheat) — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17: a week of autonomous work, no stopping/asking, no
> config-cranking a terminated mechanism, self-contained at RUNTIME,
> local 3090, public training corpus authorized, FULL FREEDOM on
> architectural work). This is the evidence-mandated Generator-F.

## Why this is the genuinely-different, evidence-MANDATED mechanism

Generator-D's pre-registered finding explicitly localized the
conversational-generation bottleneck to **the surrogate-grad LIF
spiking substrate itself** (NOT signal poverty — the real corpus
fixed that; NOT the teacher — the distillation target was competent
ppl ~15; distillation closed ~99.3% of the absolute-ppl gap yet still
0/3). Generator-D pre-registered the decision-relevant next question:
*is the SPIKING constraint the wall?* The cheap ESN reservoir probe
(non-spiking but untrained-recurrent) was near-random (471/513) —
inconclusive because a reservoir is not a trained neural LM.

**Falsify-cheaply probe (recorded `bzzzmy1se`) answers it
DECISIVELY:** a tiny from-scratch **non-spiking Transformer**
(d_model 128, 2 layers, 4 heads, ~few-hundred-K params), trained
**20 seconds on the RTX 3090**, 1500 steps, 300K-token TinyStories
slice, reaches **held-out ppl 21.95 vs uniform-random 513 (~23x
better)** and generates **visibly coherent** simple-story text
("One day, the bird went to ... and thought, he ... decided to see
toys. They just and played together all day.", with `<|endoftext|>`
story boundaries) — a clear step-change above the Generator-E
n-gram's local fragments ("awasmiled... lifefrightlts"). Every one
of the 9 neural negatives was spiking / order-blind-pool /
self-contained-signal-poverty; a standard small Transformer on the
authorized corpus was NEVER tried in this arc. **The spiking
substrate (surrogate-grad LIF BPTT at feasible local scale) WAS the
wall.** A standard small Transformer trained on the authorized public
corpus is the evidence-mandated, in-constraints, north-star-targeting
mechanism.

## Honest transparency about the architectural departure (no cheat, no overclaim)

Generator-F uses a **standard small Transformer**, NOT the project's
biology-grounded spiking substrate. This is a deliberate,
fully-transparent, user-authorized choice:
- The user explicitly granted "full freedom on architectural work...
  you autonomously research and select the mechanism"; the only hard
  constraints are local hardware + self-contained POST-training +
  no-cheat. A locally-trained Transformer whose runtime artifact is
  just its own weights satisfies ALL of these (no external LLM, no
  external dependency, no runtime corpus; the public corpus is
  training-time only and explicitly authorized).
- The 9-negative arc + Generator-D's localization + this probe
  TOGETHER are the honest evidence that the biology-grounded spiking
  substrate is **terminally negative for self-contained generation**
  at feasible local scale. Generator-F does not pretend otherwise; it
  is the answer to "what self-contained no-cheat mechanism CAN reach
  the north-star," not a retro-justification of the spiking line.
- The project's **validated biology-grounded asset — the trustworthy
  grounded continual memory with no-confabulation abstention —
  remains separate, untouched, and the primary validated
  contribution.** Generator-F is a distinct language-generation
  capability; it does NOT replace or relitigate that asset, and
  Generator-G (noted, not detailed) is the natural synthesis (ground
  the Transformer's generation on the no-confabulation memory).
- This is reported with NO overclaim: the honest ceiling is
  **small-Transformer / TinyStories-class coherent SIMPLE-STORY
  generation** (Eldan & Li 2023), explicitly NOT GPT-4-class
  reasoning, NOT long-context, NOT general conversation. It IS
  "conversational capabilities similar to a very small ... LM" at
  the small-LM ceiling — which is precisely the user's stated
  north-star scope ("a very small yet SOTA LLM").

## Thesis

A small decoder-only Transformer LM, trained from scratch via
backprop on the authorized public corpus (TinyStories — the corpus
designed by Eldan & Li 2023 to demonstrate that ~1-30M-param models
generate coherent simple stories), tokenized with the project's own
self-contained BPE, self-contained at runtime (the artifact is the
trained weights + BPE merge JSON; zero external dependency, no
external LLM, no runtime corpus), judged by the SAME unmodified
HARDENED `subword_lm_gate_core` (the gate that correctly FAILed 9
neural attempts + Generator-D and correctly PASSed/bounded
Generator-E) PLUS a coherence/quality read of the actual generated
text (honest-ceiling characterization, shown not described).

## Architecture (PyTorch; standard, minimal, self-contained)

Net-new:
1. `sim/tiny_transformer.py` — a minimal, self-contained decoder-only
   GPT (token+positional embedding, N pre-LN blocks of causal MHA +
   GELU MLP, final LN + tied/untied LM head). Pure PyTorch (`torch`
   is available + CUDA on the 3090). PURE-constructible + a tiny CPU
   forward/shape unit test (deterministic seed -> fixed output shape;
   causal mask correctness: position t logits independent of t+1
   inputs). Save/load = `torch.save`/`load` of `state_dict` + a
   sidecar JSON of the (small) hyperparams -> self-contained runtime
   artifact.
2. `research/runners/tiny_transformer_train.py` — kill-safe BPTT
   trainer: BPE on the train split (reuse `sim.bpe_tokenizer`),
   next-token cross-entropy, AdamW, cosine LR, gradient clipping;
   atomic per-checkpoint resume (reuse `sim.train_checkpoint`
   contract via a torch-state adapter, OR a `.pt` + `.meta.json`
   atomic os.replace mirror — kill-safe, user games/resumes);
   OOM-safe (catch CUDA OOM -> halve batch). FIXED pre-registered
   config (frozen BEFORE the decisive run, NOT tuned post-hoc).
3. `research/runners/generator_f_gate.py` — DRY mirror of the
   `subword_lm_gate.py` orchestration: per seed, fetch+split (cached
   TinyStories), BPE, train the Transformer (real) AND an identical
   Transformer on the BPE-invariant word-shuffle control, held-out
   ppl (real/control/train via teacher-forced CE), generate from
   held-out prompts, `distinct_ngram_ratio` + `verbatim_copy_fraction`,
   `gs_verdict(..., uniform_ppl=V)` (HARDENED gate_core, fail-closed
   without it), `gs_aggregate_multiseed` (>=3 seeds). Records the
   actual generated text for the mandatory coherence smell-test. NO
   new bar; gate_core/song_g1_core byte-UNMODIFIED.

Reuse UNMODIFIED (DRY): `sim.bpe_tokenizer`,
`research.runners.corpus_fetch`, the HARDENED
`research.runners.subword_lm_gate_core` (FROZEN bars
0.20/1.5/0.5/0.20 + `_GS_ABS_COMPETENCE_PPL_RATIO=1.0`, >=3 seeds),
the `subword_lm_gate.py` orchestration shape + `_word_shuffle`.

## Pre-registered gate (SAME hardened gate_core; bars FROZEN, never tuned)

Multi-seed >=3, every seed: held-out ppl beats uniform-random
(abs-competence floor — the bar Generator-S failed) AND beats the
BPE-invariant word-shuffle control by >=20% AND held-out <= 1.5x
train AND distinct-trigram >= 0.5 AND verbatim-copy <= 0.20.
**MANDATORY post-run anti-cheat smell-test (scrutinize a nominal
PASS HARDER than a FAIL, the Generator-S lesson):** recompute from
the recorded JSON (no re-run, no bar-tuning) that held-out ppl is
genuinely competent (<< uniform; ideally the probe-indicated ~15-25
range), the word-shuffle margin is genuine (a Transformer that
memorized would fail generalization; one that ignores order would
fail the shuffle margin), verbatim-copy is genuinely low (a
Transformer CAN memorize+regurgitate a small corpus — the copy bar
is load-bearing here too), AND **read the actual generated text** and
characterize its true coherence ceiling honestly (small-Transformer
TinyStories-class, NOT GPT-class) — never spun. PASS (scrutinized
genuine) => the project HAS a self-contained, local, no-cheat small
neural LM that generates coherent simple text and clears the SAME
rigorous gate 9 attempts failed: the north-star within the honest
small-LM ceiling -> Generator-G (ground it on the validated
no-confabulation memory). FAIL or honest-ceiling-too-low =>
propagate the precise honest characterization; the converged picture
stands. Either outcome decision-relevant.

## Honest ceiling / risks (no overclaiming)

- TinyStories-class coherent simple-story generation is the realistic
  ceiling (Eldan & Li 2023): grammatical, locally+story-coherent
  simple English; NOT reasoning, NOT long-context, NOT general
  conversation, NOT GPT-class. Reported strictly as that, with
  verbatim samples, never spun.
- A small Transformer on a bounded corpus CAN memorize/regurgitate ->
  the hardened verbatim-copy + generalization + word-shuffle bars are
  load-bearing here too; the mandatory smell-test explicitly checks
  regurgitation (the Generator-E + Generator-S lessons).
- Self-contained at RUNTIME strictly preserved: artifact = trained
  weights + BPE JSON; corpus + training are training-time only;
  zero external dependency / no external LLM at inference.
- Hardware: probe = 20s for a 2-layer toy; the FIXED pre-registered
  decisive config is a still-small but properly-trained model
  (minutes-to-low-hours on the 3090), kill-safe resumable; my
  wall-clock hand-estimates are unreliable, the measured per-step
  cost is. ASCII-only prints; Windows cp1252.

## Pre-staged successor (arc never stalls)

- Generator-F genuine PASS (scrutinized, honest ceiling) => Generator-G:
  ground the small Transformer's generation on the validated
  grounded-memory + no-confabulation abstention (the synthesis: a
  self-contained, local, no-cheat agent that generates coherent
  simple text AND refuses to confabulate beyond what it grounds) —
  the honest realization of the conversational goal within the
  small-LM ceiling.
- Generator-F FAIL / ceiling-too-low => propagate the precise honest
  finding; the converged conclusion (no self-contained no-cheat
  generator reaches even small-LM coherence at feasible local scale)
  becomes terminal and the validated grounded-memory + no-confab
  agent is the deliverable. No config-crank; no stop.

## Scientific basis (catalog + literature)

Vaswani 2017 (Transformer); Eldan & Li 2023 "TinyStories" (small
LMs generate coherent stories — the corpus + the scale this targets);
backprop / AdamW; BPE (Sennrich 2016, the project's own
self-contained tokenizer). The hardened anti-cheat gate (held-out
generalization, BPE-invariant word-shuffle control, verbatim-copy
bound, absolute-competence floor, multi-seed) is the adjudicator. The
validated grounded-memory + no-confabulation (G.20 / Tonegawa engram
/ CLS) remains the separate biology-grounded asset for Generator-G.

## Out of scope (YAGNI)

No external dependency at RUNTIME ever. No new bar; HARDENED gate_core
byte-UNMODIFIED. No config-cranking any terminated mechanism. No
scaling beyond the FIXED pre-registered cheap-decisive config in this
slice (the probe already grounds feasibility+coherence; the gate
decides; Generator-G synthesis + larger scale are later increments,
noted not detailed). Honest ceiling stated up front and never spun.
