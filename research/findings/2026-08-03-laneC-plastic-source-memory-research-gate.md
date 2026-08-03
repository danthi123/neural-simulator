---
type: research-gate
status: active
date: 2026-08-03
mechanism: laneC-plastic-source-memory
---

# Lane C plastic source memory: research gate

## Role In The Whole Brain

The conversation path needs to distinguish a fluent-looking recall from a recall that is supported by the brain's own
memory of where the answer came from. This is not a second factual database. It is a learned episodic association that
reinstates source-linked answer activity from the live cue, then sends agreement or conflict toward ACC/aPFC and the
self-schema. The self-schema may downgrade speech when the recalled candidate conflicts with that reinstated activity;
it may never create an answer after the existing hard abstention moat has rejected the cue.

## Why This Gate Fired

Two engineering levers have already attacked the same familiar-but-wrong failure:

1. `source_consistency_floor` reads the exact Python fact record selected by the composer. It catches the measured
   errors but is an explicit metadata shortcut.
2. `neural_source_consistency` writes a second RF/FHRR copy of the fact at storage time. It removes the direct metadata
   read and passes the six-seed battery, but its complete source trace is still written by construction rather than
   learned.

The next step therefore must change the mechanism, not tune another confidence threshold.

## Biological Constraint

Source memory associates an item with the context in which it was encoded. Human work separates hippocampal
pattern-separation signals from source-memory reinstatement in posterior CA1 and connected cortex
(Stevenson et al., 2020, J Neurosci, doi:10.1523/JNEUROSCI.0564-19.2019). <!--derived--> Sparse hippocampal episodic codes and
activity-biased engram allocation provide a plausible way to keep episodes distinct. Retrieval and monitoring recruit
distributed hippocampal-prefrontal assemblies (Domanski et al., 2023, Curr Biol, doi:10.1016/j.cub.2023.02.007), <!--derived--> while
anterior prefrontal regions are repeatedly implicated in monitoring retrieved source information (Simons et al., 2008,
J Cogn Neurosci, doi:10.1162/jocn.2008.20036). <!--derived-->

The repository already has the required low-level analogue: sparse concept/episode assemblies and zero-initialized,
gated Hebbian pathways that learn associations from co-firing (`_D_sparse_heteroassoc.py`). That mechanism is already
multi-seed and permutation-control clean.

## First Mechanism To Test

Build an opt-in `PlasticSourceMemory` on a `SimulationBridge`:

- A complete live proposition `(query kind, cue, recalled candidate)` receives a sparse assembly in each of several
  independent banks. The allocation is a deterministic developmental scaffold and does not consult an observed-fact
  table.
- External-source populations are separate from proposition content.
- Proposition-to-source synapses begin at zero and change only while the source-learning gate is open.
- An explicit experience event co-activates the proposition and external-source populations for a bounded learning
  window. Calling the primary memory's `store()` method alone does not create source evidence.
- Retrieval stimulates only the live proposition assembly, freezes plasticity, and reads source-population spikes.
  Candidate agreement is the normalized population support across the independent banks.
- Production receives spike-derived source support only. The exact stored answer, main composer's `source_fact`, and
  primary-memory index are unavailable to the source inference path.

This first rung may allocate sparse cue assemblies with a deterministic developmental rule. That remaining scaffold
must be named: the episode allocation is structural, while the cue-to-answer content is learned. A later rung should
make allocation competition and co-residence with the live one-brain substrate explicit.

## Required Controls And Gates

The mechanism is not promotable unless all of these are measured:

- **Default-off identity:** no source bridge or behavior change when the option is disabled.
- **Weight-change proof:** the gated cue-to-answer pathway starts at zero and changes after experience.
- **No-learning control:** with the learning gate disabled, the proposition cannot reinstate source support.
- **Permutation control:** when candidates are deliberately permuted during source learning, source support follows
  the permuted experience. This proves that fixed codes or labels do not carry the support decision.
- **Pathway lesion:** disabling transmission through the learned pathway collapses source support while the main recall
  stays intact.
- **Inference purity:** source retrieval accepts only the cue and live candidate; it does not read `kb`, `source_fact`,
  gold correctness, or the unpermuted answer.
- **Production monotonicity:** source monitoring can only retain or downgrade a matched answer. It cannot turn a moat
  miss into content, and self-schema is never invoked on hard misses.
- **Held-out six seeds:** the stressed known-fact battery records wrong assertions, correct retention, source mismatch
  false positives, disabled-learning behavior, permutation behavior, and lesion behavior over seeds
  `42 43 44 100 101 102`.

## Promotion Bar

Cheap-first is one seed. Promotion requires all six seeds, zero added false accepts, zero self-schema invocations on
hard moat misses, zero wrong assertions in the measured familiar-wrong set, and zero source-mismatch false positives
on correct recalls. At least 95% of evaluable source cues must recover their learned answer before candidate comparison.
Disabled-learning and pathway-lesion controls must fall below 20% recovery, and the permutation arm must follow the
permuted teaching above 90%.

Passing this gate would retire the engineered full-fact RF echo as the preferred source-consistency signal. It would
not close biological honesty: deterministic episode allocation, a separate source bridge, host population readout, and
the final source-conflict-to-ACC/aPFC co-resident pathway would remain on the scaffold ledger.
