# Self-contained generative speech for the sim — definitive design

> **Status:** APPROVED, autonomous execution authorized (user, 2026-05-16:
> "up to you... as long as the sim is self-contained during actual use
> post-training, LLM use for training is fine"). This is the single,
> definitive design. Supersedes the staged G1/G2 draft and all its
> correction layers (`2026-05-16-generative-rescoped-design.md`),
> which are retained only as decision trail.

## The one hard rule (everything follows from it)

**After training, the sim is entirely self-contained.** At runtime:
only the sim's own trained weights + its own grounded-memory/
abstention. No templates, no external/local LLM as speaker or
interpreter, no network — ever — in the use path. Training MAY use
public corpora and a local LLM as a distillation teacher; none of
that ships into runtime.

## What we keep vs. discard (honest)

- **Discard from the UX path:** hand-written templates (cheating —
  human authored the language). Templates survive only as test
  scaffolding/oracles.
- **Discard:** any runtime LLM (speaker OR interpreter). Hard NO-GO.
- **Keep + reuse:** Stage-1's genuine validated asset — grounded
  cross-bridge retrieval + the no-confabulation abstention gate.
  This is the sim's *own* mechanism (not external), so it stays in
  the runtime path as the honesty layer.
- **Keep + reuse:** the project's own Phase-2 surrogate-grad BPTT
  sequence network (`sim/bptt_snn*.py`, `surrogate_grad.py`,
  `char_tokenizer.py`, `cortex_pretraining.py` on `path-f-hybrid`),
  already proven to learn real text locally (Tiny Shakespeare, loss
  14.1→2.24, this 3090). This is the sim's own generator.

## Architecture (single path)

```
TRAINING (offline, may use local teacher + public corpus)
  public text corpus ─┐
                       ├─▶ sim's own seq net (Phase-2 BPTT)  ──▶ trained weights
  local LLM teacher ───┘   trained with: LM loss on corpus           (self-contained
  (next-token distrib.)    + DISTILLATION loss vs teacher logits      artifact)
                                                              
RUNTIME (100% self-contained — only the sim)
  user ─▶ sim's trained seq net ──draft tokens──▶ own grounded/
          (generates its own                       abstention layer ──▶ reply
           language)                               (drops/withholds spans the
                                                    sim's own memory can't
                                                    support → no confabulation)
```

The generator is the sim's learned weights. The honesty layer is the
sim's own validated retrieval/abstention. Nothing external at use.

## Why distillation is central (not optional)

A net trainable locally on a 3090 over public text alone will be
weak. **Knowledge distillation from a local LLM teacher** (matching
the teacher's next-token distribution during training) is the
standard, legitimate way a small model punches above its size — and
it is fully compatible with the rule: the teacher is used **only at
training time**; the shipped artifact is the sim's own distilled
weights, self-contained. This is the genuine best shot at non-trivial
fluency without cheating.

## Honest ceiling (stated up front, no overclaiming)

Local-3090-trainable + distilled ≠ cloud-LLM fluency, and likely
below a modern small LLM. The deliverable is **the sim genuinely
producing its own language, self-contained, and refusing to
confabulate** — integrity, not parity. Output will start humble
(Shakespeare-scale character/word generation) and improve with
corpus/distillation scale, capped by local compute. This is a
multi-increment research build, not a one-shot.

## Incremental plan with falsifiable gates (project discipline)

Each increment has an anti-cheat gate; a failed gate is an honest
finding, not papered over. Cheapest-decisive first.

**Increment 1 — Foundation on `main` (no network).** Bring the
Phase-2 generator infra onto `main` (cherry-pick/port), reproduce
its known-good local training on the existing local Tiny-Shakespeare
artifact. *Gate:* held-out loss reduction reproduces (≫ n-gram /
permuted baseline). Confirms the sim's own generator works on
current code. No LLM, no fetch.

**Increment 2 — Distillation teacher (one-time training-time fetch).**
Add a local LLM teacher (transformers+torch are already installed;
one-time public open-weights fetch, training-time only, never shipped)
+ a distillation loss. *Gate:* distilled net's held-out perplexity
beats the Increment-1 no-distill baseline by a real margin
(anti-cheat: also vs n-gram; teacher never in the eval/runtime path).

**Increment 3 — Self-contained runtime + honesty layer.** Wire the
trained net's generation through the sim's own grounded-retrieval +
abstention so it generates AND refuses to confabulate, with a
hard offline/no-external assertion. *Gate:* scripted use shows
fluent own-generated text; an un-grounded prompt is refused, not
fabricated; zero external calls (enforced by code+test). Templates
appear only in the test oracle, never the output.

**Increment 4+ — Scale** corpus/model/distillation within local
limits; honest perplexity + grounded-honesty tracking.

## Components / files (anticipated)

- Port: `sim/bptt_snn_gpu.py`, `sim/surrogate_grad.py`,
  `sim/char_tokenizer.py`, `research/runners/cortex_pretraining.py`
  (+ their tests) from `path-f-hybrid` → `main`.
- New: `research/runners/distill_teacher.py` (training-time-only
  local LLM logits provider; offline-enforced; never imported by
  runtime), `sim/distill_loss.py` (KD loss), and a self-contained
  `research/runners/sim_speaker.py` (trained net → own
  grounding/abstention → reply; no external imports).
- Reuse: `abstention_gate`, `_query_top`, `SharedPoolMember`,
  `claim_segment`/grounding logic from Stage-1 (the honesty layer).
- Tests: pure-logic CPU (KD loss, tokenizer, grounding decision);
  offline-enforced smoke for runtime; falsifiable perplexity gates.

## Testing & anti-cheat

- The runtime smoke MUST assert no network + no LLM import in the
  use path (code-enforced, not convention).
- Distillation gate MUST show the *self-contained* net (teacher
  absent) beats baseline — proving the capability is in the sim's
  own weights, not borrowed at eval.
- Honesty: an un-grounded prompt MUST be refused by the sim's own
  abstention, demonstrated (the Stage-1 moat carried forward).

This design is intentionally honest about being a hard, humble,
multi-increment research build. Its value is a genuinely
self-contained sim that speaks its own learned language and does not
lie — the integrity the user is explicitly buying over faked fluency.

## Increment-2 approach refinement (2026-05-16) — data distillation, not logit distillation

Feasibility gate: GO. Qwen2.5-0.5B-Instruct (Apache-2.0, fastest
sensible PoC teacher, user-approved "use the fastest") fetched +
cached locally (~988 MB); torch+CUDA on the 3090 ready;
`transformers` present. One-time training-time fetch only.

**Honest technical refinement:** the Increment-1 generator is
CHARACTER-level (CharTokenizer, vocab ~94); the teacher is
subword/BPE (~150K vocab). Direct next-token *logit* distillation
across mismatched token spaces is ill-posed. The correct, cited
method is **sequence-level / data distillation** (Kim & Rush 2016,
"Sequence-Level Knowledge Distillation"): at TRAINING time the
teacher generates a clean text corpus; the student trains its OWN
char-level weights on that corpus. This:
- sidesteps the tokenizer-space mismatch entirely,
- keeps the rule exactly (teacher only generates training text at
  training time; student is self-contained; teacher never in
  runtime, never an interpreter),
- has a clean falsifiable gate: the distilled SELF-CONTAINED student
  (teacher absent at eval) must beat the Increment-1 no-distill
  baseline on held-out loss/perplexity by a real margin — proving
  the capability lives in the student's own weights.

Increment 2 therefore = (a) training-time-only teacher text
generator (Qwen, offline after the one-time fetch), (b) generate a
distillation corpus, (c) train the Increment-1 student on it,
(d) anti-cheat gate vs the no-distill baseline (teacher absent).
