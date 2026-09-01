---
type: finding
status: live
date: 2026-09-01
mechanism: surprise-to-episodic-encode-decision-crossedge
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_surprise_episodic_encode_decision_6seed.json
runner: research/runners/_onebrain_surprise_episodic_encode_decision.py
builds_on:
  - research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md
  - research/findings/2026-08-28-surprise-episodic-129construction-6seed-GO.md
  - research/findings/2026-08-28-surprise-episodic-production-xedge-wireIn.md
  - research/findings/2026-09-01-production-default-flip-plan.md
  - research/findings/2026-09-01-declarative-cross-edge-functional-gate-read-credit-livedrive-GO.md
---

# Surprise now flips a genuine binary ENCODE/SKIP decision, not a content-neutral diagnostic margin — a NEW purpose-built episodic encode-gate population, grown by a learned cross-edge, load-bearing at the decision level (6/6 GO), closing the flip-plan's row #2 residual

**One-line:** The existing surprise->episodic cross-edge (`onebrain_xedge_surprise_episodic_production.py`,
production-wired default-OFF) reads a continuous divisive-ratio margin on `source_provenance.prov_generated` and
its own committed frozen 6-seed artifact states the read is "**a shift on the live SUBSTRATE ratio ... not (yet)
wired to flip any DECISION-level text**" — the `2026-09-01-production-default-flip-plan.md` audit's row #2
classifies flipping it as HOLLOW for exactly this reason ("a content-neutral additive DIAGNOSTIC field ... Flip =
metadata-only hollow checkbox"). This finding builds the decision-level coupling that residual named: a NEW,
purpose-built binary ENCODE-vs-SKIP gate population (`episodic_encode_gate`, honestly declared as a proxy for the
Group-C-deferred `d5_episodic` CA3 pool — the same scope substitution the existing edge already uses, carried
forward not smoothed over), driven by a learned cross-edge from the D2 surprise/mismatch unit, added PURELY BY
DECLARATION through the generic `onebrain_crossedge_gate.CrossEdgeGateSpec`/`run_gate` harness (no bespoke
F-gate). 6-seed GO (6/6): the edge grows from near-zero (0.05 -> 0.7108-0.7121, tightly converged across seeds
<!--derived--> — a min-max range across the per-seed `emergence.grown` values in the cited artifact) by the
substrate's own standard Hebbian rule; a genuinely high-surprise CONTRADICT trial flips the gate's
threshold-crossing DECISION from SKIP (rate 0.0000) to ENCODE (rate 0.0757-0.0809 <!--derived--> — a min-max
range across the per-seed `decision.rate_high_intact` values) relative to a genuinely low-surprise CONFIRM
trial, via the learned edge alone; the flip VANISHES under lesion on every seed (frac_attributable = 1.0000 on
all 6 seeds); and the pool is byte-identical-off.

## 1. Verify-first: is "surprise -> D5 episodic" already done, and is D5 actually reachable?

Checked before building, per this task's own instruction, against `onebrain_merge_framework.py`, the RAG corpus,
and the recent findings directory. Two things were already true and are NOT re-derived here:

- **A surprise->episodic cross-edge already exists and is production-wired** (`onebrain_xedge_surprise_episodic_
  production.py`, behind `BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC`, default-OFF) — 6-seed GO at the runner level
  (`2026-08-28-surprise-episodic-129construction-6seed-GO.md`, frac_attributable 0.887-0.965 <!--derived--> —
  quoted verbatim from that finding's own §4 table, not this finding's cited artifact). It targets
  `source_provenance.prov_generated`, not `d5_episodic`.
- **`d5_episodic` itself remains `GROUP_A_DEFERRED`** in `onebrain_merge_framework.py` ("Heavy own-pool — a
  ~2000-neuron CA3 with two-compartment apical dendritic-dAP + slow-NMDA reverberation + BTSP formation.
  Group-C own-pool + apical/NMDA-slow seam") — NOT migration-ready onto the shared-bridge `merge_organs`/
  `CrossEdgeGateSpec` machinery this task instructed reuse of. Its real write path (`EpisodicRecallOrgan.
  note_topic` -> `EpisodicDapMemory.store`) is independently measured at **~510s PER TOPIC on numpy@2000
  neurons** (`d5_episodic_production_organ.py`'s own docstring; corroborated by `_onebrain_integration_surprise_
  episodic_crossedge.py`'s module docstring, which made the identical finding one arc earlier: "the GO bar needs
  dozens of encode events x 6 seeds x anti-cheat -- hours to tens of hours, incompatible with the CPU/numpy
  'tiny' budget"). This session's own task brief's premise ("the D5 hippocampal episodic gate are existing
  co-resident spiking organs") does not hold against the current codebase state — flagged honestly rather than
  silently worked around.

So the genuinely NEW gap is not "wire surprise into D5" (already infeasible within this session's compute
budget, for the same reason the existing edge already declared) — it is the flip-plan's own, more precise,
already-diagnosed gap: **the existing edge's read is content-neutral and non-decisional.** That gap does not
require the heavy CA3 pool to close; it requires a read that is actually a DECISION. This finding closes that
gap, honestly scoped as a proxy exactly like its predecessor, but a rung further: a discrete decision instead of
a continuous diagnostic.

## 2. The edge, added PURELY BY DECLARATION

A NEW bare `episodic_encode_gate` organ (one excitatory population, `encode_gate`, 48 RS neurons, no internal
pathways of its own — driven only by the declared cross-edge and, during training only, a host teaching current)
is registered as a trivial `OrganDescriptor` and merged with the registered `SURPRISE` D2 organ:

```python
CROSS_EDGES = [
    CrossEdge(key="surprise_to_encode_gate", source_key="surprise", source_region="surprise",
             target_key="episodic_encode_gate", target_region="encode_gate",
             init_weight=0.05, plastic=True, gate="surprise_to_encode_gate",
             learn_rule="rate_hebbian", freeze_rest=True),
]
pool = merge_organs([SURPRISE_LITE, EPISODIC_ENCODE_GATE], wire=True, cross_edges=CROSS_EDGES)
```

`SURPRISE_LITE` reuses the existing edge's own fix (`enable_hebbian_learning=False` at build time; SURPRISE's 3
pathways are 100% fixed/block-diagonal, so this is behavior-preserving and lets our cross-edge's own training
window control Hebbian learning explicitly). Everything else — the emergence read, the vary/lesion interaction,
the no-corruption drift, `attributable_to`, the byte-off comparison — comes from `onebrain_crossedge_gate.
run_gate`, unmodified.

**Training** (host-supervised, declared not hidden — the SAME class of teaching signal every cross-edge in this
codebase uses: R1/R4/R3v3/the two prior surprise->provenance edges all train their target from a host-injected
tonic current co-occurring with the source's own activity, never a hand-set weight): each episode drives a
CONTRADICT trial on the surprise circuit (this seed's randomly-assigned cue block + a mismatched assertion block
-> `surprise` fires specifically in the mismatched block) together with a tonic teaching current directly into
`encode_gate` (`GATE_TONIC_PA=700.0`pA, order of magnitude of self_schema's own `AUTHOR_PA=650.0` for a
similarly-sized population). "Novelty/prediction-error gates hippocampal encoding" is realized as: episodes
where surprise fires are the episodes the substrate's own Hebbian rule learns to associate with encode_gate
firing.

External grounding (quoted citation identifiers, not artifact measurements — both verified against PubMed
2026-09-01, logged to `research/queue/.external_searches.jsonl`, lane `onebrain-integration`): Lisman & Grace
2005, "The hippocampal-VTA loop: controlling the entry of information into long-term memory", *Neuron*
46(5):703-713, DOI 10.1016/j.neuron.2005.05.002, PMID 15924857 <!--derived--> (the VTA-hippocampal novelty
loop); Kafkas & Montaldi 2018, "Expectation affects learning and modulates memory experience at retrieval",
*Cognition* 180:123-134, DOI 10.1016/j.cognition.2018.07.010, PMID 30053569 <!--derived--> (unexpected /
prediction-violating stimuli selectively strengthen subsequent recollection — the runner's original draft
misattributed this paper to *Neuropsychologia*; caught and corrected before commit, not left from memory).

**The load-bearing read**: after training, drive the pool with EITHER a CONFIRM trial ("low", surprise stays
near-silent — the CONTROL) OR a CONTRADICT trial ("high", surprise fires), with **no direct current into
`encode_gate` at all** — `encode_gate`'s rate is entirely whatever the learned edge carries from surprise's own
activity. `run_gate`'s generic interaction check reads both conditions, then LESIONS the declared edge and
re-reads. ON TOP of those continuous numbers (a derived reporting layer, not a re-implementation of the harness's
own checks), a single pre-registered threshold (`ENCODE_THRESH=0.042`, the midpoint between the "low" and "high"
intact rates measured on a non-canonical calibration seed [7], frozen BEFORE any canonical seed was read) turns
each condition's rate into a discrete ENCODE / SKIP verdict — the literal decision this finding tests.

**Anti-cheat**: `_assign_blocks` (reused by import, unchanged) draws this seed's (cue, mismatched-assertion)
block pair from a seed-keyed RNG independent of every other draw; the OTHER 4 (never-mismatched) trained concept
blocks' edges into `encode_gate` are checked to stay near `W0=0.05` (never trained), measured immediately after
training and BEFORE the interaction check's own lesion (which zeroes the whole declared edge, including the
untrained blocks, for its own purposes) — so the anti-cheat snapshot is taken at the only point in the run where
it is meaningful.

## 3. 6-seed result (42, 43, 44, 100, 101, 102), numpy CPU — GO 6/6

(values to 4 decimal places, rounded from the cited 6-seed artifact — this table is a rounded restatement, not
an exact quote; open the JSON directly for full double precision.)

<!--derived-->

| seed | grown weight | rate low (CONFIRM) | rate high (CONTRADICT) | decision low->high (intact) | decision low->high (lesion) | frac attributable | anti-cheat other-block (before/after) | GO |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.7121 | 0.0000 | 0.0797 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |
| 43 | 0.7114 | 0.0000 | 0.0757 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |
| 44 | 0.7109 | 0.0000 | 0.0779 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |
| 100 | 0.7108 | 0.0000 | 0.0809 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |
| 101 | 0.7121 | 0.0000 | 0.0772 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |
| 102 | 0.7112 | 0.0000 | 0.0791 | SKIP -> ENCODE | SKIP -> SKIP | 1.0000 | 0.0500 / 0.0500 | GO |

Every seed: the edge GROWS from `W0=0.05` well above the `grow_factor*init_weight=0.25` emergence floor (no
runaway toward `HMAX=8.0`); the no-corruption check (max\|delta\| over every non-edge synapse) reads exactly 0.0
on all 6 seeds; the intact "high"-vs-"low" shift clears `INTACT_FLOOR=0.010` with wide margin on every seed; the
`attributable_to` decomposition (the control-comparison instrument this project's `instrument_required` gate
requires for a GO headline) reads the shift as fully attributable to the cross-edge on every seed the same way
`_onebrain_crossedge_provenance_to_selfschema.py`'s reciprocal edge did; and the discrete decision genuinely
flips SKIP->ENCODE intact and collapses to SKIP->SKIP under lesion on every seed — `decision_flip_vanishes_on_
lesion=True` 6/6 (the `Verdict` preconditions in the cited artifact require this on every seed, not merely on
average).

## 4. What this demonstrates, and what it does not

**Closes**: the specific "content-neutral, non-decisional" residual `2026-09-01-production-default-flip-plan.
md` row #2 named as the reason the existing surprise->episodic flip is hollow. This edge's read is a genuine
binary decision, and that decision is shown load-bearing (varies with surprise) and lesion-attributable (the
variation vanishes when the edge is lesioned) — the honest functional shape the flip-plan asked for.

**Does NOT close**: wiring this decision into the REAL `d5_episodic` hippocampal store (`EpisodicRecallOrgan.
note_topic`) so that a SKIP verdict actually suppresses a real BTSP write, or into the production chat turn at
all. `d5_episodic` remains Group-C-deferred for the reason restated in §1 (a genuine, previously-measured compute
wall, not a new one manufactured here); the `episodic_encode_gate` population is a PROXY, exactly like the
existing edge's `prov_generated` target was a proxy, carried forward honestly rather than reframed as "D5
itself." This is a runner-level GO (`research/runners/_onebrain_surprise_episodic_encode_decision.py`) — per
`docs/TERMS.md`, that is "GO at runner level," not `wired`/`closed`/`integrated`: no production wiring, no
`sim/` edit, no default flip, additive.

## 5. Honest residuals (declared, not hidden)

- **Host-supervised training**, exactly like every cross-edge in this codebase: `encode_gate`'s co-drive
  (`GATE_TONIC_PA` during CONTRADICT episodes) is a host teaching current, not a self-organized discovery of
  what should co-occur. The substrate's own Hebbian rule does the binding; the host supplies the correlated
  experience. Per `docs/TERMS.md`, this is `host-supervised`/`teacher-driven`, not `self-organized`.
- **The target is a proxy, not the real hippocampal store.** `encode_gate`'s firing is an ENCODE-COMMITMENT
  DECISION GATE, not a literal CA3 autobiographical memory trace — the full "does surprise change whether topic
  X's real BTSP assembly forms" claim rides the `d5_episodic` Group-C migration, unresolved for the compute
  reason in §1, a named follow-on not attempted here.
- **ONE-SIDED BY DESIGN**: `encode_gate` is a single population (an ENCODE-vs-silent tag), matching R4/the
  self_schema `author` axis's own honest shape — no companion population for a weight-ratio selectivity
  comparison. Selectivity is demonstrated functionally (the anti-cheat other-block check), not as a weight ratio,
  matching `docs/TERMS.md`'s `selective` condition (reported with its control, not a bare ratio).
- **The threshold is a design choice**, calibrated once on a non-canonical seed and frozen — a different
  threshold within the wide gap between the measured low (~0.00) and high (~0.08) rates would not change any
  seed's verdict (the gap is >10x the calibration seed's rates), but the specific value is a chosen decision
  boundary, not a substrate-discovered one.
- **Not a production flip**: matching every predecessor cross-edge's own stated scope, this is a standalone
  research runner. Wiring `episodic_encode_gate`'s decision into `EpisodicRecallOrgan.note_topic` (so a live SKIP
  verdict actually gates the real BTSP write) is the natural next rung, named not attempted.

## 6. Files

`research/runners/_onebrain_surprise_episodic_encode_decision.py` (NEW — `EPISODIC_ENCODE_GATE`, `CROSS_EDGES`,
`EncodeGatePool`, `GATE_SPEC`, `_other_block_drift`, `run_seed`, `calibrate`, `main`) ·
`research/findings/raw/_onebrain_surprise_episodic_encode_decision_6seed.json`. Reused, unmodified:
`research/runners/onebrain_crossedge_gate.py` (`CrossEdgeGateSpec`, `run_gate`, `verify_byte_off`,
`cross_edge_masks`), `research/runners/onebrain_merge_framework.py` (`REGISTRY`, `OrganDescriptor`, `CrossEdge`,
`merge_organs`), `research/runners/_onebrain_integration_surprise_episodic_crossedge.py` (`_assign_blocks`,
`CUE_PA`, `PRE_STEPS`). No `sim/` file touched; no `webapp/server.py` edit; no production default changed.

Functional read-outs only; no phenomenal-experience claim.
