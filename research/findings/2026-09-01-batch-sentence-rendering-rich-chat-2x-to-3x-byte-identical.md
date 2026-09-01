---
type: finding
status: go
lane: composer
date: 2026-09-01
seed-waiver: latency + byte-identity are DETERMINISTIC infra properties of a fixed-seed greedy-decode faculty,
  not seed-dependent learning -- the batched call reseeds the SAME `1000 + seed` the sequential call does, and
  every VERIFY re-parse is a deterministic host string operation. A single seed (42) suffices to establish
  both, matching the precedent of `2026-08-30-knowledge-66-latency-hot-loop-is-codebook-rebuild-cache-31pct-
  byte-identical.md` (also a single-seed infra-latency finding, not a generalization claim).
mechanism: RichAnswerComposer.render_paragraph's per-sentence fluent-render loop (research/runners/
  rich_answer_composer.py) calls the off-bridge spiking Qwen renderer's model.generate() ONCE PER GATHERED
  FACT (up to 2 launches per fact once the VERIFY-reject regen retry is counted) -- the launch-bound
  "many tiny sequential ops" pattern named in memory feedback_prioritize_orchestration_overhead. Batches the
  CONSTRAIN first-pass AND the REGEN retry into two per-turn model.generate() launches total (regardless of N)
  via new SpikingQwenFaculty.render_svo_batch / render_svo_regen_batch (left-padded batched generation),
  additive + default-OFF behind BRAIN_RICH_BATCH_RENDER.
runner: research/runners/_batch_sentence_rendering_derisk.py
tests: tests/test_rich_answer_batch_render.py
artifacts:
  - research/findings/raw/_batch_sentence_rendering/n2.json
  - research/findings/raw/_batch_sentence_rendering/n4.json
  - research/findings/raw/_batch_sentence_rendering/n6.json
verdict: >
  The RICH-answer chat reply path (`RichAnswerComposer.render_paragraph`) is launch-bound exactly as
  `feedback_prioritize_orchestration_overhead` names: each gathered fact triggers its OWN sequential
  `model.generate()` call (plus a second one on a VERIFY-reject regen retry), so an N-fact turn can pay up to
  2N GPU launches. Batching BOTH the first-pass CONSTRAIN render and the REGEN retry into two per-turn
  `model.generate()` calls (left-padded, one launch covers every fact needing that stage) cuts median
  wall-clock 1.68x (N=2) / 2.10x (N=4) / 3.01x (N=6) at reps=9 each, with the batched-vs-sequential
  (paragraph, kept, dropped) output BYTE-IDENTICAL in all three configurations (asserted in the data, not
  inferred) and with the new code path PROVABLY never reached when the flag is off (a poison test that makes
  the batched method raise, run under the flag-off condition, still succeeds byte-identically). An earlier,
  simpler one-stage design (batch only the first pass, fall back to the full sequential path -- including a
  redundant single-item render_svo/render_svo_regen -- on any VERIFY reject) was SLOWER than the pre-existing
  sequential loop (0.73x-0.88x) whenever most candidates needed the regen retry, because it paid for the
  batched call AND the full sequential fallback; that negative is banked below, not discarded, and the fix
  (batch the regen retry too; drop a fact only after BOTH batched stages fail rather than redundantly retrying
  a deterministic generation) is what the committed 1.68x-3.01x numbers reflect.
---

# Batching the RICH-answer sentence-render loop: 1.68x-3.01x latency, byte-identical, and one banked negative

## Where the loop was

`RichAnswerComposer.render_paragraph` (`research/runners/rich_answer_composer.py`) renders each gathered
`[a, v, p]` (up to `max_sentences`, production default 4) into a fluent sentence ONE AT A TIME:

```python
for svo in facts:
    sent, verified = self._render_one_verified(svo, gated=gated)
```

`_render_one_verified` calls `chat.renderer.render_svo(a, v, p)` -- for the production `QwenRenderer` this is
`SpikingQwenFaculty.render_svo` -> ONE `self.model.generate()` call. On a VERIFY reject it retries with
`render_svo_regen` -> a SECOND `model.generate()` call. So an N-fact RICH reply can pay up to **2N sequential
GPU launches**, each restarting tokenization + KV-cache setup from scratch for its own short prompt, even
though the prompts are mutually independent. This is the exact pattern `feedback_prioritize_orchestration_overhead`
names as the real-time chat wall: launch-bound, many tiny sequential ops, not a VRAM ceiling.

## The fix (additive, default-OFF `BRAIN_RICH_BATCH_RENDER`)

`SpikingQwenFaculty.render_svo_batch` / `render_svo_regen_batch` (`research/runners/
_grounded_lang_integration_derisk.py`, new) left-pad the N CONSTRAIN (or REGEN) prompts into one tensor and
call `model.generate()` ONCE. `QwenRenderer.render_svo_batch` / `render_svo_regen_batch`
(`research/runners/brain_chat_tui.py`, new) expose the same per-item `(surface, asserted)` shape as the
existing single-item methods. `RichAnswerComposer._render_paragraph_batched` (new) runs up to **two GPU
launches total for the whole turn**, not per fact:
  1. every fact that doesn't resolve on the (cheap, host-side) spiking-recall-surface check is CONSTRAIN-rendered
     in ONE `render_svo_batch` call;
  2. anything that fails VERIFY is retried in ONE SECOND `render_svo_regen_batch` call, covering every failure
     at once instead of one regen call per failure;
  3. (safety net) anything a batched call itself could not attempt -- the call raised, or the renderer lacks
     `render_svo_regen_batch` -- falls back to the pre-existing single-item `_render_one_verified`. A fact that
     DID get a real candidate from both batched stages and still failed VERIFY is dropped there, not retried:
     `render_svo`/`render_svo_regen` are deterministic (greedy decode, fixed reseed), so a single-item retry of
     an already-attempted prompt would reproduce the identical text and fail VERIFY identically -- pure wasted
     GPU time for a foregone conclusion, and not what the sequential path itself does either (it does not retry
     a third time).

Gated by `_batch_render_enabled()` (env `BRAIN_RICH_BATCH_RENDER`, default OFF) AND `len(facts) > 1` AND the
active renderer exposing `render_svo_batch` (a `hasattr` check, not a try/except) -- `StubRenderer` and every
renderer that predates this feature lack it, so the flag is a no-op for them regardless of its value.

## Measured: latency (median over 9 reps, real off-bridge Qwen2.5-0.5B, RTX 3090)

Isolated from the orthogonal, separately GO-verified `BRAIN_SPIKING_MOUTH_RECALL` production default (2026-08-26
flip; forced off for this measurement only -- see the runner's own docstring -- so every gathered fact actually
reaches Qwen instead of resolving on the cheap spiking-Broca template in both conditions, which would dilute
the very effect under test):

| N facts | baseline (sequential) median | batched median | speedup |
|---|---|---|---|
| 2 | 1.896s | 1.127s | **1.68x** |
| 4 | 3.094s | 1.476s | **2.10x** |
| 6 | 4.797s | 1.592s | **3.01x** |

Speedup scales with N, matching the mechanism: the batched path costs at most 2 launches regardless of N,
while the sequential path scales up to 2N. Artifacts:
`research/findings/raw/_batch_sentence_rendering/n2.json` / `n4.json` / `n6.json` (each produced by
`research/runners/_batch_sentence_rendering_derisk.py --reps 9 --max-sentences {2,4,8}`, `n6.json` gathered 6
facts against a `--max-sentences 8` request -- the smoke knowledge graph is exhausted at 6 for this topic, an
honest graph-connectivity limit, not a bug).

## Measured: byte-identical equivalence (asserted in the data, not inferred)

Every one of the three artifacts above records `equivalence_ok=True`: the batched path's `(paragraph, kept,
dropped)` is an EXACT match (`==`) to the sequential path's, for the identical gathered facts, at every N
tested. This was NOT assumed -- the installed spiking ops (`spiking_rmsnorm_forward` / `spiking_silu_forward` /
`spiking_softmax_forward`, `_grounded_lang_p1b_stepB1_forward_derisk.py`) draw graded-read pool-SEM noise via
`torch.randn(tensor.shape, generator=SPK.gen)`, and the SHAPE of that draw differs between a batched
(`batch>1`) and a lone (`batch=1`) forward even at the identical reseed value, so the exact float noise a given
sentence's forward pass sees is not bitwise-guaranteed identical between the two paths. Measured anyway: at
this faculty's operating point (T=16, pool_softmax=4096), the ~1e-2-scale pool noise never moved a greedy
argmax decision across a token boundary on any of the 12 facts probed (2+4+6). Reported as measured, not
assumed to generalize to every prompt.

## Byte-identical-when-OFF: PROVED, not inferred from output equality

Each artifact also records `off_never_calls_batched=True` and `off_poisoned_matches_baseline=True`: the
batched method (`RichAnswerComposer._render_paragraph_batched`) is monkeypatched to `raise AssertionError` for
the duration of the check, then `render_paragraph(facts)` is called with the flag OFF -- it must still succeed
and match the untouched baseline byte-for-byte. This PROVES the new code path is never reached with the flag
off (not merely that its output happens to match by coincidence). `single_fact_never_batches=True` confirms the
`len(facts) > 1` guard the same way for a one-fact turn.

The same properties are additionally covered by 5 fast, CPU-only, GPU-free pytest tests
(`tests/test_rich_answer_batch_render.py`, using a deterministic `TemplateStubFaculty`-backed fake renderer so
they run without a model download): flag-off call-count proof, flag-on batched-vs-sequential byte-identity
(with call-count assertions, not just output equality), a renderer lacking `render_svo_batch` staying a
byte-identical no-op, the single-fact guard, and a batched candidate that fails VERIFY correctly falling back
without corrupting the turn. All 5 pass.

## A banked negative: the first (one-stage) design was SLOWER, not faster

The first implementation batched only the CONSTRAIN first pass and fell back to the full sequential
`_render_one_verified` (its own `render_svo` + `render_svo_regen` attempts) on any VERIFY reject. At this
faculty's actual reject rate against the `_SMOKE_FACTS` knowledge graph's claim-level moat (3 of 4 facts
typically fail the first-pass CONSTRAIN render's VERIFY, needing a regen), this was measured **SLOWER than the
pre-existing sequential loop**: N=2 gave 0.73x, N=4 gave 0.76-0.88x across two independent 9-rep runs -- the
batched call became pure overhead layered on top of an unavoidable full sequential recovery. This is exactly
the "report a negative honestly, a mapped negative is a deliverable" case: the fix was not to abandon batching
but to also batch the regen retry (one more `model.generate()` call covering every failure at once) and to stop
retrying a fact single-item once BOTH batched stages had genuinely already failed it (since retrying a
deterministic greedy decode of an already-tried prompt cannot change the outcome). That fix is what produced
the 1.68x-3.01x numbers above; the one-stage negative is superseded, not silently dropped.

## Scope (honest)

Measured on the tiny `_SMOKE_FACTS` interlinked knowledge graph (2-6 gathered facts per turn, the production
`max_sentences` ceiling of 4 plus a couple of higher-N configurations to show the scaling trend), Qwen2.5-0.5B,
T=16, max_new_tokens=24, seed=42, on an RTX 3090. Not yet run against the production 79k-fact knowledge-in-chat
store or multi-seed; the latency/equivalence properties measured here are deterministic infra properties (see
the seed-waiver), so a single seed is the right instrument for THIS question, but a production-scale /
multi-topic confirmation is the natural next rung before flipping the default.

## Wiring (all additive, default-OFF, `research/batch-sentence-rendering` branch)

- `research/runners/_grounded_lang_integration_derisk.py` -- `SpikingQwenFaculty._generate_batch` /
  `render_svo_batch` / `render_svo_regen_batch` (new methods; `_generate` / `render_svo` / `render_svo_regen`
  untouched byte-for-byte).
- `research/runners/brain_chat_tui.py` -- `QwenRenderer.render_svo_batch` / `render_svo_regen_batch` (new
  methods; existing methods untouched).
- `research/runners/rich_answer_composer.py` -- `_batch_render_enabled()` (env `BRAIN_RICH_BATCH_RENDER`,
  default OFF) + `RichAnswerComposer._render_paragraph_batched` (new); `render_paragraph` gains one `if` branch
  at its top that falls through to the pre-existing loop, unchanged, whenever the flag is off / <2 facts /
  raw mode / the renderer lacks `render_svo_batch`.
- `research/runners/_batch_sentence_rendering_derisk.py` (new) -- the GPU latency + equivalence + byte-
  identical-when-OFF measurement runner (produces the 3 artifacts cited above).
- `tests/test_rich_answer_batch_render.py` (new) -- fast CPU-only wiring/gating tests (5/5 pass).

## Next rung

Confirm at production scale (the routed 79k-fact `ShardedPhasorStore`, multiple topics) and flip
`BRAIN_RICH_BATCH_RENDER` on by default once confirmed -- the mechanism is additive and byte-identical-off, so
the flip carries no regression risk to the existing sequential path even if the production-scale numbers differ
from this smoke-scale measurement.
