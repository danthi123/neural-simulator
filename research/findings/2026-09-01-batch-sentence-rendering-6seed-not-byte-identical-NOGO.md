---
type: finding
status: no-go
date: 2026-09-01
mechanism: batch-sentence-rendering
lane: composer
runner: research/runners/_batch_sentence_rendering_derisk.py
artifacts:
  - research/findings/raw/_batch_sentence_rendering_6seed/s42.json
  - research/findings/raw/_batch_sentence_rendering_6seed/s43.json
  - research/findings/raw/_batch_sentence_rendering_6seed/s44.json
  - research/findings/raw/_batch_sentence_rendering_6seed/s100.json
  - research/findings/raw/_batch_sentence_rendering_6seed/s101.json
  - research/findings/raw/_batch_sentence_rendering_6seed/s102.json
---

# Batch sentence rendering (board #110): 6-seed re-verify finds NOT byte-identical — flip STAYS OFF

## Why this doc exists

`2026-09-01-batch-sentence-rendering-rich-chat-2x-to-3x-byte-identical.md` measured `BRAIN_RICH_BATCH_RENDER`
at a SINGLE seed (42) across three `max_sentences` configurations (n=2/4/6) and reported
`equivalence_ok=True` in all three, naming "production-scale / multi-topic confirmation" as the next rung
before flipping the default. This task (auto-flip-policy sweep of two owner-named GO faculties) re-ran the
SAME runner (`research/runners/_batch_sentence_rendering_derisk.py`, unmodified) at the mandated 6 seeds
(42, 43, 44, 100, 101, 102) as the guard for the flip. It fails: **3 of 6 seeds are NOT
byte-identical between the batched and sequential render paths**, and the divergence is not cosmetic —
it is a genuine content difference (a different subset of facts kept, or an entirely empty reply where the
sequential path said something). The auto-flip policy requires byte-identical output as a hard precondition
(this is a pure infra speedup, not intended to change WHAT the brain says); this measurement refutes that
precondition at half the tested seeds, so `BRAIN_RICH_BATCH_RENDER` stays default-OFF.

## Reproduced first: the single-seed claim holds at seed 42

Re-running seed 42 exactly (`--seed 42 --reps 7`, default `--max-sentences 6`) reproduces the prior finding's
qualitative shape: `baseline_median_s=3.318`, `batched_median_s=1.171`, `speedup_x=2.83` <!--derived-->,
`equivalence_ok=True`, `off_never_calls_batched=True`, `off_poisoned_matches_baseline=True`,
`single_fact_never_batches=True` (`research/findings/raw/_batch_sentence_rendering_6seed/s42.json`). The
mechanism, the speedup, and the OFF-safety all check out at this seed, exactly as claimed.

## The 6-seed guard: equivalence fails on 3/6 seeds

| seed | n_facts | baseline median (s) | batched median (s) | speedup | equivalence_ok |
|---|---|---|---|---|---|
| 42  | 6 | 3.318 | 1.171 | 2.83x <!--derived--> | True  |
| 43  | 6 | 4.023 | 1.688 | 2.38x <!--derived--> | **False** |
| 44  | 6 | 1.753 | 1.484 | 1.18x <!--derived--> | **False** |
| 100 | 6 | 3.341 | 1.026 | 3.26x <!--derived--> | True  |
| 101 | 6 | 3.569 | 1.390 | 2.57x <!--derived--> | **False** |
| 102 | 6 | 4.204 | 1.316 | 3.20x <!--derived--> | True  |

(Full numbers in each seed's own artifact under `research/findings/raw/_batch_sentence_rendering_6seed/`.)
Speedup and the OFF-safety poison tests (`off_never_calls_batched`, `off_poisoned_matches_baseline`,
`single_fact_never_batches`) hold on all 6 seeds — those two properties are genuinely robust. Equivalence is
NOT: 3/6 seeds (50%) diverge. This is why the runner's own `Verdict` still prints `GO` for every seed (it
`disables`, rather than `requires`, the equivalence check, by original design, treating the divergence as an
acceptable documented residual) — but the auto-flip policy's guard for THIS task is explicitly "byte-identical
output, a pure speedup, no content change", which is a stricter bar than the runner's own internal verdict.
Measured against that stricter, task-specified bar, the flip is a NO-GO.

## The divergence is content-level, not stylistic

Inspecting `kept`/`dropped`/`paragraph` on the 3 failing seeds' own artifacts (`s43.json`, `s44.json`,
`s101.json` under `research/findings/raw/_batch_sentence_rendering_6seed/`):

- **seed 43**: sequential (OFF) keeps 1 fact ("Neurons have synapses.");
  batched (ON) keeps **0** facts — the whole reply goes empty (`paragraph_on=''`) while the sequential path
  said something.
- **seed 44**: sequential (OFF) keeps 4 facts ("Spikes fired neurons. Neurons have synapses. Synapses store
  weights. Memory needs sleep."); batched (ON) again keeps **0** facts — a fully empty reply.
- **seed 101**: sequential (OFF) keeps `["weights", "hold", "memory"]`; batched (ON) keeps a DIFFERENT fact,
  `["neurons", "have", "synapses"]` — not a superset/subset, a disjoint substitution.

This matches the mechanism the original finding's own docstring already named as an honest, un-quantified
risk: the installed spiking ops draw graded-read pool-SEM noise via `torch.randn(shape, generator=SPK.gen)`
SHAPED by the full batch tensor, so a batched forward's per-sequence noise realization is not
bitwise-identical to a lone forward's even at the same reseed. What this run adds is that the divergence is
not a rare 1-in-12 tail event that never crosses a decision boundary (the seed-42 measurement's honest but
narrow claim) — at seeds 43/44/101 it crosses far enough to flip a VERIFY accept/reject decision on
EVERY gathered fact in the turn, producing a materially different (sometimes silently empty) reply for the
identical conversation state, seed, and prompt. A user toggling this flag on could get a visibly worse or
blank answer purely from an infra-level batching choice — exactly what "byte-identical, pure speedup" is
supposed to rule out.

## Verdict

**NO-GO on flipping `BRAIN_RICH_BATCH_RENDER` default-ON.** The mechanism's OFF-safety (poison-tested) and
its speedup are solid and reproduce cleanly across all 6 seeds; its claimed byte-identical equivalence does
NOT generalize past the single seed it was first measured on. `_batch_render_enabled()`
(`research/runners/rich_answer_composer.py`) is unchanged (`os.environ.get("BRAIN_RICH_BATCH_RENDER")`,
default OFF) — verified against the live source before and after this run.

## Next rung (not attempted here — out of scope for this task)

The gap is plausibly closeable without abandoning the mechanism (per this repo's standing law, a wall names a
method failure, not a capability to drop): e.g. draw the batched forward's pool-noise per-sequence with an
independently-seeded generator slice matched to the lone-forward draw (so a batched item's noise realization
is bitwise IDENTICAL to what that same item would draw standalone), rather than one `torch.randn` call shaped
by the whole batch. That is a real engineering change to `SpikingQwenFaculty._generate_batch`
(`research/runners/_grounded_lang_integration_derisk.py`), not attempted in this task (scoped to
verify-and-flip, not re-engineer). Until it lands and is itself 6-seed byte-identical-verified, the flip
stays OFF.
