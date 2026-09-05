---
type: finding
status: no-go
date: 2026-09-05
mechanism: d5-depth-hold-direct-read-surfacing-gate
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_rank24_quick_flips_verify.py --check d5-direct-read — reuses
  research.runners._d5_graded_flip_soak's EXACT scenario builder (EpisodicRecallOrgan + the cat/dog/bird 4-turn
  conversation) by import (no duplicated scenario logic), then asks: does ANY single fixed depth_hold threshold
  separate "dog was consolidated" (t4_dog, ON) from "bird was never consolidated" (t4_bird, ON) on every seed?
runner: research/runners/_rank24_quick_flips_verify.py
external: NO-EXTERNAL-NEEDED — an in-repo falsification of a specific proposed retirement mechanism via direct
  measurement of the substrate's own existing graded read; no literature question (the read itself,
  `depth_hold = mean-held max(cp_v_apical − v_hold, 0)`, is unchanged and already biologically anchored by the
  arc this finding does not touch).
artifacts:
  - research/findings/raw/_rank24_quick_flips/d5_direct_read_6seed.json
---
# The scaffold-map's rank-24 "d5 depth_hold direct-read" retirement is NO-GO: raw depth_hold is NOT comparable ACROSS topics, so no single threshold can replace the per-topic `_CONSOLIDATED_TOPICS` host-set gate (measured, 6/6 seeds)

Artifact: `research/findings/raw/_rank24_quick_flips/d5_direct_read_6seed.json` (6-seed; `GO=False`).

**One line.** `research/coordination/scaffold_retirement_backlog.md` rank 24 proposed retiring the host
`_CONSOLIDATED_TOPICS` surfacing-gate set by "reading the existing graded `depth_hold` value directly at
reply-generation time against a calibrated threshold" — scored `readiness: High` by the scaffold-shortcut-map
(`w9sn9wn4b`). Measuring it directly falsifies that: each topic's DG-formed CA3 assembly has its OWN baseline
apical-latch magnitude (varies with assembly composition/size), so a topic that was **never** consolidated can
read a **higher** absolute `depth_hold` than a different topic that **was** consolidated. No production code was
changed — the existing host `_CONSOLIDATED_TOPICS` set-membership gate
(`webapp/continuous_engine.py`, `research/runners/d5_episodic_production_organ.py::_topic_consolidated`) remains
the mechanism; this finding documents why a proposed replacement for it does not ship.

## The measurement

For each seed, the standard cat/dog/bird scenario runs with the flag ON (dog gets `mark_recall`'d then
`consolidate_used_memory`'d; bird and cat never do). Three numbers per seed: `dog_baseline` (dog's `depth_hold`
at t2, before consolidation), `dog_consolidated` (dog's `depth_hold` at t4, after), and `bird_baseline` (bird's
`depth_hold` at t4 — a topic that completed `in_memory=True` but was **never** consolidated, the case the gate
must correctly REJECT).

<!--derived-->
| seed | dog_baseline | dog_consolidated | dog_rise | bird_baseline (never consolidated) | per-seed threshold exists? |
|------|-------------:|------------------:|---------:|------------------------------------:|:---------------------------:|
| 42   | 29.963       | 30.909             | +0.946   | 30.559                               | yes (window 30.559–30.909)  |
| 43   | 29.315       | 29.630             | +0.315   | 30.394                               | **NO** — bird's baseline (30.394) exceeds dog's consolidated value (29.630) |
| 44   | 29.741       | 30.240             | +0.499   | 30.813                               | **NO** — bird's baseline (30.813) exceeds dog's consolidated value (30.240) |
| 100  | 27.029       | 27.456             | +0.427   | 29.955                               | **NO** — bird's baseline (29.955) exceeds dog's consolidated value (27.456) |
| 101  | 30.018       | 30.344             | +0.326   | 30.407                               | **NO** — bird's baseline (30.407) exceeds dog's consolidated value (30.344) |
| 102  | 27.153       | 27.437             | +0.284   | 30.868                               | **NO** — bird's baseline (30.868) exceeds dog's consolidated value (27.437) |

Full artifact: `research/findings/raw/_rank24_quick_flips/d5_direct_read_6seed.json`. **Only 1 of 6 seeds is even
individually separable** (seed 42, and only barely — a 0.35mV window). On the aggregate bar a real production
constant actually needs to clear — `max(bird_baseline across seeds) < min(dog_consolidated across seeds)` — the
runner reports `max_bird_baseline_never_consolidated=30.868` vs `min_dog_consolidated=27.437`: the never-touched
baseline EXCEEDS the consolidated value by **3.4mV**, the opposite of the required ordering. `GO=False`.

Seed 43 alone is already decisive on its own: **bird was never touched by consolidation, yet its baseline
`depth_hold` (30.39mV) is higher than dog's value AFTER a full consolidation pass (29.63mV)**. Any threshold that
correctly accepts dog-consolidated on seed 43 (≤29.63) would ALSO accept bird's never-consolidated baseline on
the SAME seed (30.39 > any such threshold) — a false positive: the reply would claim a rising "recall strength"
for a memory that was never strengthened. Five of six seeds show this same inversion; seed 42 is the only
near-miss, and even it only works in isolation, not against the other five seeds' bird baselines.

## Why: depth_hold has no natural zero or shared scale across topics

`depth_hold` is a MEAN over a topic's OWN held-out assembly cells (`mean-held max(cp_v_apical − v_hold, 0)`).
Each topic's assembly is a DIFFERENT emergent DG-selected population (`assembly_sizes` differ per topic per
seed, e.g. seed 42: [33, 16, 23]) whose composition sets its OWN baseline plateau-depth magnitude — a
one-shot BTSP encode alone (no learn-through-use) already leaves different topics at different absolute
`depth_hold` levels, apparently driven by assembly-composition variance that is comparable in size to (and on
seed 43, LARGER than) the entire learn-through-use rise the mechanism is trying to detect. This is the
CLAUDE.md wall-reframe question answered directly: the "companion process" the raw-threshold proposal implicitly
assumed away is **a per-topic baseline** — the host `_CONSOLIDATED_TOPICS` set is not incidental bookkeeping
standing in for a signal that already exists in comparable form; it is standing in for a NORMALIZATION the
substrate does not yet supply (there is no on-substrate signal for "this topic's OWN rest depth_hold" to divide
or subtract against at read time).

## What would actually work (fresh work, not a quick flip)

A depth_hold-based direct read is not ruled out in principle — but it needs a genuine PER-TOPIC baseline to
compare against (e.g., a value captured at first formation and re-derived from something the substrate itself
carries, such as assembly size or a per-assembly resting conductance), which is a real mechanism-design question,
not a constant. Recording a per-topic baseline FLOAT the host still has to bookkeep would not obviously be a
retirement of host bookkeeping either — it substitutes a boolean host set for a float host ledger. This is
banked as a fresh-work item, not attempted here.

## Scope honesty

No `sim/` edit; no production code changed. `research/runners/_rank24_quick_flips_verify.py` adds a read-only
diagnostic (`--check d5-direct-read`) that measures the existing `depth_hold` read through the existing
`EpisodicRecallOrgan`/`recall_disclosure` machinery; it does not add any new flag or gating path to
`webapp/continuous_engine.py` or `d5_episodic_production_organ.py` — shipping an inert flag for a mechanism this
same runner disproves would be worse than not shipping one. The existing `_d5_graded_flip_soak.py` 6-seed
no-regression soak (the CURRENT production mechanism, `_CONSOLIDATED_TOPICS`) was re-run this session
(seeds 42/43, GO on both, matching its own prior verdict) as the baseline this finding compares against — see
the sibling verification note in the rank-24 commit.
