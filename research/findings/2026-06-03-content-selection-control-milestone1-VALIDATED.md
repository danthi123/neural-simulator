# Content-selection / dialogue-Control — Milestone 1 VALIDATED (2026-06-03)

**Result: RESOLVES (Milestone-1 mechanism validation).** A structured Control layer — the brain's
prefrontal "Control" role in Hagoort's Memory-Unification-Control model — produces coherent,
on-topic, non-repeating multi-turn dialogue on top of the validated concept substrate, clearly
beating a no-control retrieval-only baseline, on both the substrate's real documented associations
and a richer synthetic graph, 5/5 seeds.

## What was built

`research/runners/content_selection.py` (+ `_eval.py`, + `tests/test_content_selection.py`, 19 tests).
The controller is the only new logic; all content (concepts, associations) is reused. Three Control
functions, the faithful decomposition of PFC Control:
- **Context buffer** — a fading record of which concepts have been discussed (the discourse model).
- **Relevance** — how strongly a candidate is associated with the active context, read from the
  substrate's **learned associations** (concept codes are orthogonal-by-design, so relevance can't
  come from code similarity — it comes from the learned association graph; this is the faithful
  reading of PFC relevance-biasing).
- **Inhibition-of-return** — suppress recently-said content (avoid repetition).

## The controlled eval

Controller vs a fair no-control baseline (strongest single-step associative retrieval, no context /
no inhibition) on the same association graph, scored by four metrics over the dialogue transcript:
`on_topic`, `non_repetition`, `turn_to_turn` coherence, `topic_progression`. Two graphs: the
substrate's **real documented multitag associations** (apple-big, apple-cat, dog-small, dog-river,
cat-hot, river-cold, big-hot, small-cold — the validated 90% multitag pairs) and a **clearly-labelled
synthetic** four-topic graph for a richer-scale test. Seeds 42-46.

Mean delta (controller - baseline):

| metric | REAL | SYNTHETIC | meaningful? |
|---|---|---|---|
| on_topic | +0.500 | +0.409 | **yes** |
| turn_to_turn | +0.500 | +0.905 | **yes** |
| non_repetition | +0.700 | +0.833 | by-construction |
| progression | +0.700 | +0.833 | degeneracy guard |

Seeds passing (both meaningful deltas > 0 AND progression >= 0.5): **5/5 on both datasets.**

Example transcripts (the human transcript honesty guard — they read coherently):
- `rain -> cloud -> storm -> wind -> sky -> sun -> warm` (controller) vs `cloud,cloud,cloud,...` (baseline)
- `song -> melody -> voice -> sing -> rhythm -> drum -> tune` vs `melody,melody,...`
- real: `apple -> big -> cat -> hot` vs `big,cat,cat,big,...`

## Honesty (this is a validation, not a surprising emergence)

- The controller is **designed** to select associated, non-repeating content, so `non_repetition` and
  `topic_progression` are near-guaranteed by its hard-inhibition construction. The **meaningful**
  coherence signal is `on_topic` + `turn_to_turn` — where the controller could fail on a bad graph
  but does not (it selects genuinely associated concepts; the baseline repeats, giving ~0
  turn-to-turn). The eval validates that the mechanism **works on the real substrate** and quantifies
  the gap; it establishes the baseline + metrics + harness that Milestones 2-3 re-run.
- The **smell-test caught a real flaw and it was fixed**: the first eval runner let the controller
  wander into a disconnected concept cluster once it exhausted the topic's associations; the runner
  now enforces the same on-topic guard as the proper dialogue runner, and the transcripts read
  cleanly.
- **Scope caveats:** the real association substrate is small (8 documented pairs); the controller is
  deterministic, so "multi-seed" here varies only the baseline; "coherence" is measured by
  association-based proxies plus a human transcript read, not full human judgement at scale.

## Staging (next)

Milestone 2 — replace the structured context buffer with a spiking dlPFC region (sustained activity),
re-run this exact eval. Milestone 3 — all three Control functions spiking (top-down biasing +
spike-frequency-adaptation inhibition). Each re-runs the same eval, so faithfulness can't silently
cost coherence. Design: `docs/plans/2026-06-03-content-selection-dialogue-control-design.md`.

This is the first working piece of the **Control** layer — deciding *what to say* so dialogue stays
coherent across turns — biology-grounded (MUC/PFC), built reuse-only on the validated substrate,
honestly scoped, with the spiking-faithful path laid out.
