# AGI-FORK — operative charter (relaxed-constraint branch)

**This branch (`agi-fork`) operates under the RELAXED constraint set below, NOT `main`'s strict-biology
constraints.** `main` is the strict-biology path, preserved intact for diff + cherry-pick-back. Full spec +
rationale: [docs/plans/2026-09-06-agi-first-fork-proposal.md](docs/plans/2026-09-06-agi-first-fork-proposal.md).
Origin conversation: memory `project_agi_first_fork_openness_2026_09_06`. Owner-approved, **time-boxed (~a few
days), reversible.** Anyone (Claude or the local downtime agent) working this branch follows THIS charter.

## The bet (goal unchanged, METHOD reframed)
True AGI via digital brain-recreation. Transformers/LLMs are a **developmental dead-end** — a frozen one-shot
training pass + context-bound experience *structurally* prevent open-ended development. The differentiator of this
project is a system that **DEVELOPS**: learns continuously, grounds meaning in sensation/action/consequence,
integrates faculties in one substrate, is intrinsically motivated. **Biology guides; function judges; compute is
unconstrained.** North-star wording: *"the brain's AGI-relevant architecture, realized as efficiently as the
hardware allows."*

## HARD INVARIANTS — do NOT relax (these ARE the bet; eroding one = the slide back to the dead-end)
1. **Continual / lifelong learning** — learns through use; never a frozen one-shot-trained artifact.
2. **Grounding** — learns from sensing a world, acting, and consequence (text can be part of experience; isolated
   text-prediction is not the goal).
3. **Integration** — one interacting substrate; faculties are not siloed modules bolted together.
4. **Emergence** — faculties DEVELOP from a learning substrate, not hand-coded one at a time.
5. **Intrinsic motivation** — affect / drives / curiosity produce self-direction.
6. **Honesty boundary** — every self-report is a FUNCTIONAL read-out; NEVER assert phenomenal/felt experience.
   (Ethical, orthogonal to capability; kept always; matters MORE as capability grows.)

## WHAT FLEXES here (vs `main`'s strict path)
- **`speed < faithfulness` → function-first.** Efficient functional approximations allowed WHERE they preserve the
  six invariants — even if not neuron-for-neuron faithful.
- **Bit-exact biological fidelity is NOT required** (exact spiking dynamics, seconds-long plateau timing,
  never-a-host-shortcut). Biology sets architecture / inductive bias; it does not gate every implementation detail.
- **The all-SPIKING requirement is RELAXED** — a continual-learning, integrated substrate need not be spike-exact
  (rate-based / other efficient dynamics allowed IF the invariants hold). Biggest departure + most likely
  accelerant; **treat as a hypothesis to test, not a given.**
- **Compute is unconstrained.** The 3090 is a REFERENCE, not a cap; AWS / multi-GPU / a rack are all fair game.
  Rigorously demonstrating a compute requirement is itself a deliverable.

## KEPT — rigor is NOT relaxed
Gates, **6-seed validation**, doc-sync, **commit BOTH remotes via `tools/push_both.sh` (NEVER `--no-verify`)**,
honest measurement. Only biological-fidelity strictness flexes — not the honesty of measurement.

## FAILURE-GUARD
Any change that erodes one of the six invariants is out of bounds — that is the definition of failing the fork
(becoming another frozen, context-bound, siloed, hand-built system). If the fork can only make progress by giving
up continual-learning / grounding / emergence, that is a **NEGATIVE result worth banking** (it would say the strict
path's constraints were load-bearing for the AGI properties).

## FIRST MOVE (design in flight)
An **EMERGENCE / CONTINUAL-LEARNING SUBSTRATE** — ONE substrate that *learns many faculties from experience* rather
than hand-retiring ~50 host shortcuts to spiking one at a time (the strict path's slowest, faithfulness-bound
tier). Directly tests the fork's thesis (does relaxing fidelity + adding compute let faculties EMERGE at a rate the
hand-built path can't match?) and targets invariants #1 + #4 head-on. **Concrete architecture + minimal decisive
experiment are being determined by the `agi-fork-first-move-design` workflow (5 diverse proposals → adversarial
judge → synthesis); its synthesis fills in the build plan here.**

## EXIT (end of the time-box)
(a) merge the learnings/approach back, (b) continue the fork, or (c) return to the biology path carrying the banked
learnings. All three acceptable; (c) with real learnings is a success, not a failure. **Capture learnings
continuously as findings ON THIS BRANCH.**
