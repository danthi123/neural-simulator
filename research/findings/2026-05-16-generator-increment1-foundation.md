# Generator Increment 1 — foundation validated on main (anti-cheat-gated, honest/humble)

## TL;DR

The project's own Phase-2 surrogate-grad BPTT generator is ported to
`main`, works, and **provably learns real local text**, beating a
permuted-character control. This validates the FOUNDATION only — it
is NOT yet conversational, distilled, or grounded. Absolute fluency
at smoke scale is low (expected). No network, no LLM, no templates.

## What landed

- Phase-2 infra ported `path-f-hybrid` -> `main` (5 modules + 4 test
  files), **28/28 ported tests green on main** (independently
  re-verified). Self-contained, no API drift.
- Zero-download local corpus (`local_corpus.py`): the repo's own
  ~296K-word English findings prose, deterministic, no network.
- Foundation anti-cheat gate (`generator_baseline_smoke.py`): reuses
  the validated `train_shakespeare` (DRY); trains REAL vs
  permuted-char control.

## Anti-cheat gate result (GPU, local 3090, seed 42)

| | loss start | loss end |
|---|---|---|
| REAL corpus | 15.00 | **4.44 (-70.4%)** |
| PERMUTED control | 14.75 | 5.70 |

REAL is **22.1% below PERMUTED** → the generator learns genuine
sequential structure, not noise-fitting. **GATE: PASS.**

## Honest bounds (no overclaiming)

- Absolute REAL end-loss 4.44 is only *marginally* below uniform
  chance (ln 94 ≈ 4.54). The smoke config is deliberately tiny
  (one 64-unit hidden layer, 20 epochs, 150K chars). This validates
  the **mechanism** (own generator learns real structure on main,
  anti-cheat-controlled) — NOT fluency. Real language quality
  requires Increment 2 (distillation) + scale, capped by local
  compute.
- This is a FOUNDATION increment. NOT conversational. NOT distilled.
  NOT yet wired to the grounded/abstention honesty layer. No claim
  beyond "the sim's own generator demonstrably learns real local
  text on main."

## Next (honest)

Increment 2: local LLM distillation teacher (training-time only,
one-time public-weights fetch) + KD loss; falsifiable gate =
distilled self-contained net beats this no-distill baseline. Then
Increment 3: self-contained runtime + grounding/abstention.

## Files

- ported: `sim/bptt_snn*.py`, `sim/surrogate_grad.py`,
  `sim/char_tokenizer.py`, `research/runners/cortex_pretraining.py`
- new: `research/runners/local_corpus.py`,
  `research/runners/generator_baseline_smoke.py`
- result: `research/findings/raw/g11_bg/generator_baseline.json`
- design: `docs/plans/2026-05-16-self-contained-generator-DESIGN.md`
