# Next phase: prove who owns the computation

*2026-09-25. Adopted by the owner after two outside AI reviews of the project and the development agent's replies
(this session). Live board: [GAP_CLOSURE_MISSION.md](../../GAP_CLOSURE_MISSION.md). Builds on
[2026-09-19-roadmap-with-a-permanent-llm-mouth.md](2026-09-19-roadmap-with-a-permanent-llm-mouth.md); nothing in
this plan has been built yet.*

## The question for this phase

**Which part of the system caused the answer, and can we demonstrate that causally?**

It is not enough that the neural version passes a benchmark: the neural version has to be the reason it passes.
Qwen2.5-0.5B-Instruct stays the permanent articulation mouth. The target architecture is that the brain owns
semantic content, affect, honesty, intent and commitment, and Qwen only turns the selected communicative state into
language. Scaffold count is not the primary progress measure. The lesion-verified load-bearing fraction stays; a
separate measurement of host computational ownership is added beside it.

## 1. Host-decision seam map

Map every host-side decision in one live chat turn, content and commitment first. For each seam record:

- the host operation;
- the neural state or input immediately before it;
- the decision the host makes;
- whether the neural substrate already represents the relevant choice;
- the host override effect;
- current replacement difficulty;
- research value.

Keep separate: candidate generation, candidate scoring, candidate admission/filtering, winner selection,
commitment, composition, routing. Check in particular that the host does not secretly fix the candidate set before
a supposedly neural selector chooses. **Record a host-share baseline before retiring any scaffold.**

## 2. Host-share measurement

Three levels, kept conceptually separate:

- **Decoding:** can the host decision be predicted from the neural state before it? A useful screen, never
  evidence that the neural system made the decision (if the host takes an argmax of neural scores, the scores
  trivially predict the winner).
- **Substitution:** replace the host decision with a neural-derived one and observe the downstream effect;
  identifies whether the seam is consequential.
- **Intervention (the primary causal measurement):** hold the relevant neural state fixed, change the host
  decision, and measure whether downstream behaviour follows the host or returns toward the neural preference.

Report host share beside, not instead of, the load-bearing fraction. Keep a **learned-content fraction** separate
from a **neural-decision fraction**: Qwen wording brain-owned content is acceptable; Python selecting content and
asking Qwen to word it is not.

## 3. Selection before composition or routing

Use the existing ignition bus and basal-ganglia selection; do not build a new selector. The first target is neural
ownership of selection and commitment. Key test: competing candidates whose relative value changes with internal
state (for example hunger makes candidate B win, satiation makes A win) with no host priority rule. The strongest
version: the host receives the neural winner/commitment signal and never inspects semantic labels to choose among
candidates itself.

## 4. Long-delay temporal credit

A delayed-cue two-choice benchmark:

- cue A or B at the start, then removed;
- a delay that varies (e.g. 0.5x to 8x a base delay, so the network cannot learn a clock);
- distractors during the delay; the terminal state carries no information about the cue;
- a final distractor that can deliberately suggest the wrong answer;
- the terminal choice, then delayed reward; a shuffled-reward control;
- eligibility traces that can be ablated.

The key control separates **persistent cue memory** from **temporal credit assignment**: a network that holds A in
mind through the delay is not thereby assigning a later reward to the earlier causal event. Where practical,
include several intervening actions/events so reward must be credited to an earlier event, not to whatever is
active at reward time. Preregister the expected signatures for the full, trace-ablated, immediate and
misleading-final-cue conditions.

## 5. Sleep transitive inference

A->B learned in context X, B->D in context Y; A and D never co-occur in waking experience. After sleep, present A
alone in a novel context Z and ask whether the system infers D. Preregister at least:

1. normal replay;
2. wake only (no sleep);
3. replay with temporal/episode order scrambled, preserving activity statistics as far as practical;
4. replay switched off.

The novel-context test reduces the chance that the result is context or cue association. Where practical, add a
control that matches marginal activation and co-occurrence statistics while changing the relational chain. Also
read the representations before and after sleep, and check that a replay lesion removes A->D while leaving A->B
and B->D intact. These gates belong in the prioritized-memory preregistration.

## 6. Faithfulness-first, but falsifiable

Biological faithfulness stays the default architectural principle. Biological completeness is a hypothesis about
where capability comes from, not evidence that a mechanism is necessary. For every major capability, report
**which mechanisms were necessary for it, demonstrated by controlled ablation**, using the six-seed lesion
discipline. The point is not to drop mechanisms because one benchmark does not need them; it is to accumulate
evidence about which mechanisms support which capabilities. (Written into CLAUDE.md, non-negotiable 4.)

## 7. Execution order

1. Map host decisions, starting with content and commitment.
2. Establish host-share baselines.
3. Remove host authority from one selection seam, preferably ignition/BG selection.
4. Run the long-delay temporal-credit benchmark.
5. Run the sleep transitive-inference preregistration.
6. Then the persistent world / competing-needs work.

The credit and sleep experiments can run on the shared pool while the seam work proceeds.
