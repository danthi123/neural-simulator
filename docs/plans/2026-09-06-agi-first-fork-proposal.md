# AGI-First Fork — strategic-direction proposal (2026-09-06)

**Status:** APPROVED in principle by the owner 2026-09-06 (time-boxed, reversible experiment). This doc is the SPEC
the fork branch starts from. `main` remains the strict biology path, untouched. See memory
`project_agi_first_fork_openness_2026_09_06` for the origin conversation.

## The bet (unchanged goal, reframed method)
The project is fundamentally an attempt at **true AGI via digital recreation of the human brain.** The owner's
thesis: transformers/LLMs are a **developmental dead-end** — a single frozen training pass + context-bound
experience *structurally* prevents open-ended development; the industry band-aids (RAG, longer context,
fine-tuning) but hits a wall at any reasonable compute/cost. The differentiator of THIS project is a system that
**develops**: learns continuously, grounds meaning in sensation/action/consequence, integrates faculties in one
substrate, and is intrinsically motivated.

**The fork's reframe of the METHOD:** biology as **design guide + inductive bias**, NOT a bit-exact spec.
*Biology guides; function judges; compute is unconstrained.* North-star wording on the fork:
**"the brain's AGI-relevant architecture, realized as efficiently as the hardware allows."**

## HARD INVARIANTS — do NOT relax (these ARE the bet; eroding one = the slide back to the dead-end)
1. **Continual / lifelong learning** — learns through use; never a frozen one-shot-trained artifact.
2. **Grounding** — learns from sensing a world, acting, and consequence (text can be part of experience; isolated
   text-prediction is not the goal).
3. **Integration** — one interacting substrate; faculties are not siloed modules bolted together.
4. **Emergence** — faculties DEVELOP from a learning substrate, not hand-coded one at a time.
5. **Intrinsic motivation** — affect / drives / curiosity produce self-direction.
6. **Honesty boundary** (ethical, orthogonal to capability) — every self-report is a FUNCTIONAL read-out; NEVER
   assert phenomenal/felt experience. Kept always; matters MORE as capability grows.

## WHAT FLEXES on the fork (vs the strict biology path)
- **`speed < faithfulness` → function-first.** Efficient functional approximations are allowed WHERE they preserve
  the six invariants — even if not neuron-for-neuron faithful.
- **Bit-exact biological fidelity is not required** (exact spiking dynamics, seconds-long plateau timing,
  never-a-host-shortcut). Biology sets the architecture/inductive-bias; it does not gate every implementation
  detail.
- **The all-SPIKING requirement is the key experiment to relax:** allow a *continual-learning, integrated neural
  substrate* that need not be spike-exact (rate-based / other efficient dynamics) IF the invariants hold. This is
  the biggest departure and the most likely source of acceleration — treat it as a hypothesis to test, not a
  given.
- **Compute is unconstrained.** The 3090 is a REFERENCE, not a cap; AWS / multi-GPU / a rack are all fair game.
  Rigorously demonstrating a compute requirement is itself a deliverable.
- **Methodological rigor (gates, 6-seed, doc-sync, push-both) is KEPT** — the fork is still scientific; only the
  biological-fidelity strictness flexes, not the honesty of measurement.

## FAILURE-GUARDS
- Any change that erodes one of the six invariants is out of bounds — that is the definition of failing the fork
  (becoming another frozen, context-bound, siloed, hand-built system).
- If the fork can only make progress by giving up continual-learning / grounding / emergence, that is a NEGATIVE
  result worth banking (it would say the strict path's constraints were load-bearing for the AGI properties).

## STRUCTURE (reversible)
- Branch `agi-fork` off `main`. `main` = strict biology path, preserved intact for diff + cherry-pick-back.
- The fork carries its OWN governing docs — a fork-specific `research/coordination/live_state.md` + a fork
  CLAUDE.md section reflecting the relaxed constraint set. `main`'s strict constraints are NOT edited.
- **Time-box: a few days.** Capture learnings continuously as findings ON THE FORK.
- **Exit decision at the end:** (a) merge the learnings/approach back, (b) continue the fork, or (c) return to the
  biology path carrying the banked learnings. All three are acceptable; (c) with real learnings is a success, not
  a failure.

## SUCCESS CRITERIA (the few-days bar — judged, not vibed)
- **Primary:** does the AGI-first approach (function-over-fidelity + big compute) make *faster and/or more general*
  progress on the six invariants than the strict-biology path would in the same window?
- **The honest ledger:** for each relaxation used, what did it BUY (speed / capability / generality) vs COST
  (biological plausibility / faithfulness)? A relaxation that bought nothing is banked as "the strict version was
  fine"; one that bought a lot is a candidate to bring back.

## PROPOSED FIRST MOVE (refine when the branch opens)
Attack the strict path's real long-pole — **scaffold retirement (1 → ~50, the slowest tier because it's
faithfulness-bound)** — with an **emergence / continual-learning substrate**: ONE substrate that *learns many
faculties from experience* rather than hand-retiring 50 host shortcuts to spiking one at a time. This most directly
tests the fork's thesis (does relaxing fidelity + adding compute let faculties EMERGE at a rate the hand-built
strict path can't match?), and it targets invariants #1 (continual learning) and #4 (emergence) head-on. Candidate
second move: a genuinely continuous "learn-through-conversation" loop (the continual-learning differentiator LLMs
structurally lack), unconstrained by the all-spiking rule.

## What this doc does NOT do
It does not change `main`'s operating constraints (still re-injected as non-negotiable each turn). The fork is a
deliberate branch with its own rules; switching the live constraints happens only by editing the fork's governing
docs on that branch.
