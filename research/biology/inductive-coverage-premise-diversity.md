---
type: biology
id: inductive-coverage-premise-diversity
mechanism: Category-based induction -- the strength of a generalization to a superordinate category scales with how much of that category's representational variety the premises COVER (premise diversity), not with premise count alone
status: established-psychology-neural-substrate-derisk
last_verified: 2026-08-26
current_finding: research/runners/_inductive_coverage_derisk.py
current_status: "Premise-diversity (coverage) effect realized on a two-region SimulationBridge as rate-Hebbian property learning over shared-category-core population codes. The effect EMERGES from the substrate's own soft-bound Hebbian saturation (a synaptic form of the Ch-17 normalization companion process): diverse premises spread potentiation across more subcategory cores, and because w after 2 co-activations < 2x w after 1 (soft-bound concavity), spreading beats concentrating -- diverse-2 > within-2 at MATCHED premise count. No host similarity formula: the readout is the property region's graded population depolarization through the LEARNED concept->property synapses."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "allow the context of a stimulus to modify"
    note: "Ch 17 Sensory Coding -- inhibitory networks 'allow the context of a stimulus to modify the strength of excitation evoked by that stimulus, an important process called normalization.' This is the COMPANION PROCESS: a saturating/normalizing readout is what makes broad coverage beat concentrated potentiation. In this build the same concavity is supplied by the soft-bound Hebbian rule's own saturation (delta = rate*(max-w))."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "makeup of the population"
    note: "Ch 30 -- 'The makeup of the population of such primitives then determines which structural constraints are imposed on learning ... a behavior for which the [motor] system has many primitives will be easy to learn.' Kandel states the COVERAGE principle directly for motor primitives: the breadth of the representational population governs what generalizes. Category-based induction is the semantic-memory instance of the same principle."
  - author_year: "Osherson, Smith, Wilkie, Lopez & Shafir 1990"
    citation: "Category-Based Induction. Psychological Review 97(2):185-200."
    note: "The canonical similarity-COVERAGE model (confirmed via WebSearch; not in the local corpus). Argument strength = (a) similarity of premise to conclusion category + (b) COVERAGE = the degree the premise category is similar to instances of the lowest superordinate spanning premise and conclusion. Premise DIVERSITY raises coverage -> stronger generalization. Neural grounding exists: 'Category-based induction from similarity of neural activation' (Cog Affect Behav Neurosci 2013)."
constraints_config:
  # The coverage effect (diverse-2 > within-2) requires the concept->property weights to sit in the
  # SUB-SATURATION regime of the soft-bound Hebbian rule -- if weights saturate to hebbian_max, w2 ~= w1 and
  # spreading no longer beats concentrating (diverse ~= within). Keep hebbian_rate * epochs below the point
  # where category-core weights pin to hebbian_max. This is the OPERATING POINT the biology implies.
  enable_hebbian_learning: true
  enable_stdp: false
implemented_by:
  - research/runners/_inductive_coverage_derisk.py
findings: []
---

# Premise diversity is coverage, and coverage is a concavity over shared population codes

**The claim the code must respect.** In category-based induction (Osherson et al. 1990) the conclusion is a
generalization to a superordinate category ("all birds have property P"). The strength of that generalization
rises with premise **coverage** -- how much of the superordinate's representational variety the premises are
similar to -- and diverse premises (robin + penguin) cover more than concentrated ones (robin + sparrow) even at
the SAME premise count. Kandel states the population-coverage principle outright for motor control: "[t]he makeup
of the population of such primitives then determines which structural constraints are imposed on learning ... a
behavior for which the [motor] system has many primitives will be easy to learn." The breadth of the active
population governs generalization.

**Why this needs a companion process.** On overlapping population codes (a shared per-category core + per-
subcategory cores + unique tails), a property is learned by Hebbian coincidence (concept co-active with the
property assembly). If potentiation were LINEAR in the number of premises that fire a neuron, concentrating two
premises on one subcategory would tie with spreading them across two -- the total subcategory mass is equal.
Coverage only beats concentration when the readout (or the write) is **concave / saturating**. Kandel names the
cortical version: inhibitory networks "allow the context of a stimulus to modify the strength of excitation
evoked by that stimulus, an important process called normalization." A normalizing (saturating) response makes
two moderate signals worth more than one large one.

**Where the concavity comes from here (brain-based, not a host formula).** The soft-bound Hebbian rule already
supplies it: `delta_w = rate*(w_max - w)`, so a synapse driven by TWO co-activations reaches `w2 < 2*w1` (the
increment shrinks as `w` grows). A category-core neuron fired by both premises carries `w2`; a subcategory-core
neuron fired by ONE premise carries `w1`. Reading generalization as the property assembly's graded population
depolarization over held-out category members (one per subcategory, spanning the superordinate):

- **within-subcategory** (2 premises, same subcat): one subcategory core at `w2`, the rest at 0.
- **diverse** (2 premises, different subcats): two subcategory cores at `w1` each.

The diverse arm wins iff `2*w1 > w2`, which the soft-bound concavity guarantees. Coverage beats concentration
because the substrate's own saturation is concave -- the Ch-17 normalization, realized synaptically.

## What this entry cannot catch

The operating point is load-bearing (see `constraints_config`): if the weights saturate to `w_max`, then
`w2 ~= w1` and the diverse/within gap closes. That is a property of `hebbian_rate * epochs`, not a single
numeric default, so no gate can assert it -- the runner's own verdict measures the gap and fails NO-GO if it is
absent. Also: premise MONOTONICITY (2 premises > 1 premise) is a SEPARATE Osherson effect that co-exists here
(within-2 also beats 1-premise via the same concavity on the category core), so the diversity claim is stated as
the MATCHED-premise-count contrast diverse-2 vs within-2, which isolates coverage from count.
