---
type: plan
status: live
date: 2026-06-02
---

# Biology-faithful, data-efficient conversational agent — roadmap — 2026-06-02

## The organizing insight (owner-clarified)
The bottleneck to richer biology-faithful conversation is NOT VRAM (headroom confirmed) and NOT fundamentally
"compute for LLM-scale data." Trillions of tokens is the LLM's BRUTE-FORCE requirement; a child acquires
language from ~10-50M words (~5-6 orders of magnitude less) because the brain has DATA-EFFICIENT structures:
multimodal grounding, hippocampal fast-binding + consolidation, compositionality (systematic generalization),
curriculum/critical periods, predictive learning, and structural priors. Human-scale data is tractable on the
3090 (hours-days). So the missing piece is the data-efficient LEARNING STRUCTURES that let human-scale data
suffice -- the project's "missing structures" mandate, on hardware we already have. Today's generative overfit
was the symptom of LACKING these, not a hardware wall.

## What is already validated (biology-faithful components)
- Comprehend: Hebbian conjunctive parser (voice-invariant role assignment), multi-seed.
- Memory + composition: VSA bind/unbind, KB >= 30 facts (perfect), 320-concept substrate, multi-seed.
- Produce: generate-by-composition (ordered sentence from a composed meaning; generalizes to novel meanings;
  numpy len 3-5 + spiking len 3 on the real 320 substrate).
- Data-efficiency mechanisms present (separately): grounding (visual cortex + motor pools + embodied-Hebbian),
  hippocampus + SWR consolidation (no catastrophic forgetting), composition, curriculum (plasticity gating /
  critical periods).
=> The conversational COMPONENTS exist biology-faithfully. The frontier moved UP: integrate + learn + scale.

## Roadmap (cheap-first at every step; honest negatives = deliverable)
1. **Integrated conversational loop (tangible, NEAR-TERM -- build first).** Wire the validated components into
   a single loop: comprehend (parse) -> update/retrieve memory -> compose a response meaning -> PRODUCE a full
   ordered-sentence response (generate-by-composition). The tangible brain-analogue agent that converses in
   composed sentences and persists what it learns. Reuse-by-import; numpy demo first, then spiking.
2. **Data-efficient LEARNING (the real frontier).** Integrate the data-efficiency mechanisms to LEARN language
   from human-scale data where the generic net overfit. Cheap-first load-bearing tests, each isolating ONE
   brain mechanism's data-efficiency contribution:
   - compositional prior -> few-shot systematic generalization (learn parts, recombine);
   - grounding -> word-meaning learnable from few grounded examples (vs text-statistics);
   - hippocampal fast-binding + consolidation -> one-shot acquisition without forgetting;
   - curriculum -> simple->complex staging reduces sample complexity.
   The hypothesis under test: with these, a small vocabulary + grammar is learnable + generalizes from
   human-scale (hundreds-thousands of examples) data, where the generic net needs orders more and overfits.
3. **Content selection / dialogue planning (LONG-TERM).** PFC-style control (Hagoort "Control") over what to
   compose-and-produce given context -- the genuine hard frontier where open-ended conversational intelligence
   lives. Honestly bounded: open-ended LLM-breadth needs scale; goal-directed/retrieval-driven selection is
   tractable now and is a real (narrower) conversational agent.

## Discipline
Biology-faithful (catalog + primary sources read directly -- Kandel PDF local); cheap-first before any big
build; scrutinise a PASS harder than a FAIL; honest negatives propagated to both remotes; reuse-by-import; no
protected/frozen/moat-module edits; no non-biological shortcuts (except brief testing/validation). Brainstorm/
design before each major build.

## Immediate next
Build (1) the integrated conversational loop demo (tangible artifact), then begin (2)'s cheap-first
data-efficiency tests. Proceeding autonomously.
