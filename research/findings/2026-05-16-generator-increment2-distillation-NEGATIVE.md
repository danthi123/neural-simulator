# Generator Increment 2 — data distillation: honest NEGATIVE (student-capacity bottleneck)

## TL;DR

Sequence-level / data distillation (Kim & Rush 2016) — a local
teacher generates a clean English corpus, the project's own
char-level spiking student trains on it — **did NOT lift the student
at proof-of-concept scale.** The pre-registered controlled gate
FAILED. This is a real, honest negative; it is not papered over and
the gate was not re-tuned to force a pass.

## Controlled gate (teacher absent at eval, equal char budget, same config)

| Corpus (110K chars, 1×64 char-SNN, 25 epochs, seed 42) | end loss |
|---|---|
| REAL-baseline (repo's own English prose) | 4.178 |
| DISTILLED (teacher-generated English) | **3.961** |
| PERMUTED (char-shuffled distilled — anti-cheat control) | 3.917 |

- DISTILLED vs baseline: **+5.2%** (gate required ≥10% → FAIL)
- DISTILLED vs PERMUTED: **−1.1%** (distilled did NOT beat shuffled
  noise → the decisive failure)

## What this actually means (the honest, useful finding)

The damning result is not the +5.2% miss — it is **DISTILLED ≈
PERMUTED**. At this PoC scale the char-level spiking student is **not
learning genuine sequential structure** from the distilled corpus any
better than from shuffled characters. Better (cleaner, more general)
training data cannot help a student too small / under-trained to
exploit sequence structure at all.

This also **sharpens an Increment-1 caveat honestly**: Inc-1's
"foundation PASS" showed REAL text 22% below its permuted control on
the repo corpus. That result does **not reproduce** here — the same
tiny config on a *different* corpus fails the permuted control
(−1.1%). So Inc-1's structure-learning was **corpus-specific and not
robust**, not a general capability. The earlier "foundation
validated" must be read with this correction: the mechanism runs and
*can* show a structure signal on one corpus at one seed, but it is
fragile at this scale — not a dependable foundation yet.

## Diagnosis: the bottleneck is student capacity, not corpus

Data quality is no longer the limiter (the distilled corpus is real,
coherent English). The limiter is **student model capacity under the
local compute budget**: a 1-hidden-layer, 64-unit char-level spiking
net at 25 epochs cannot learn robust sequence structure regardless of
how good the text is. This is exactly the honest ceiling the design
doc stated up front ("local-3090-trainable will be weak; integrity
not parity").

## What was NOT done (anti-cheat discipline)

- The pre-registered gate (≥10% vs baseline AND vs permuted) was
  **not relaxed** after seeing the numbers.
- The student config was **not** repeatedly cranked up until a pass
  appeared (garden-of-forking-paths / gate-tuning cheat). A single,
  transparently-reported capacity scan is legitimate future science
  but is **next-increment scope**, not a retroactive fix.
- Teacher absent at eval; no fabricated corpus; the negative stands.

## Honest status of the generator path

- Increment 1 (foundation): runs on `main`; structure signal is
  **fragile / corpus-specific at PoC scale** (corrected here).
- Increment 2 (data distillation): **NEGATIVE at PoC scale** —
  student-capacity-bound, not corpus-bound.
- The mechanism is not disproven in principle; the honest open
  question is **student scale** (more layers/hidden, word/subword
  tokenizer, far more epochs) within local-only compute — and
  whether useful generation is even reachable on a single 3090. That
  is a strategic scope question (how much local compute to spend on
  a capability that may stay humble), surfaced for the user, not
  auto-pursued by config-cranking.

## The robust, validated asset is unchanged

Generation is unproven. The project's genuinely validated,
non-fragile contribution remains the **trustworthy continual
memory** — many concepts, no catastrophic forgetting, and a
no-confabulation abstention (it refuses to make things up). That
result is anti-cheat-validated and stands; this negative does not
touch it.

## Files

- `research/runners/distill_gate.py`,
  `research/datasets/distill_corpus.txt` (teacher-distilled, 112K)
- `research/findings/raw/g11_bg/distill_gate.json`
- Design: `docs/plans/2026-05-16-self-contained-generator-DESIGN.md`
