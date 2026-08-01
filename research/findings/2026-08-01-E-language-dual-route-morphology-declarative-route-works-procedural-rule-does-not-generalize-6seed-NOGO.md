---
type: finding
status: contributing
date: 2026-08-01
mechanism: dual-route-morphology
lane: E-language
artifacts:
  - research/findings/raw/lanes/morph/morph_6seed_aggregate.json
---

# E·Language: dual-route past-tense — the DECLARATIVE route works (blocking + over-regularization + permuted-collapse) but the PROCEDURAL rule does NOT generalize to novel stems — 6-seed NO-GO (0/6)

<!--derived-->
**One-line verdict:** the Stage-2 [CPU] Language de-risk — Pinker–Ullman words-and-rules on the D sparse
heteroassociative SPIKING pool (procedural default "-ed" rule vs declarative irregular whole-form store, competing
via FS/WTA) — resolves to **NO-GO on all 6 seeds**, but a *diagnostic* NO-GO. The **declarative route is solid**:
irregulars block the rule (irr_acc 0.857), lesioning the declarative store drives over-regularization to "-ed"
(0.952), and permuting the stem→whole-form binding collapses irregular retrieval to chance (0.024) — the
Marcus/Pinker blocking + over-regularization dissociation is realized on spikes. But the **procedural rule does NOT
generalize**: novel/held-out stems inflect correctly only 0.188 of the time (gate ≥0.90) — the "wug test" fails.
Novel stems (wug/blick/dax/gorp) are captured by entrenched irregular whole-form attractors (`wug→slept`,
`blick→ran`, `cook→came`) instead of taking the default affix. Ran on the pool concurrently with the GPU crux
(parallelism directive).

Artifact: `research/findings/raw/lanes/morph/morph_6seed_aggregate.json` (backend numpy/CPU; per-seed raw beside it).

## Result — 6 seeds {42,43,44,100,101,102}

<!--derived-->
| read-out (6-seed mean) | value | gate | reading |
|---|---|---|---|
| **reg_acc** (novel-stem rule generalization) | 0.188 | ≥0.90 | **FAILS** — the rule is not stem-independent |
| irr_acc (irregular blocking) | 0.857 | blocks | declarative store outcompetes the rule (correct) |
| overreg_rate_lesion (lesion → over-reg) | 0.952 | high | lesion the store → irregulars regularize to "-ed" (correct) |
| permuted_binding_irr_acc (anti-cheat) | 0.024 | collapse | permute the binding → irregular retrieval → chance (correct) |
| GO | 0/6 | — | — |

## What this maps (a HALF-realized dual route)

<!--derived-->
Three of the four gate conditions — the ones that belong to the **declarative/exception route** — pass cleanly and
6/6: irregulars are stored, they BLOCK the default rule, lesioning them produces the signature over-regularization
("goed"/"runned"), and the learned synaptic binding is load-bearing (permuting it collapses retrieval). That is a
real spiking realization of the exception half of words-and-rules. What fails is the **procedural/rule route**: a
default "-ed" that should apply stem-independently to ANY stem, including never-seen pseudo-stems. In a single
shared heteroassociative pool the novel stems do not reach the affix — they fall into the basin of the strong
entrenched irregular whole-form attractors instead. The rule is being represented as *lookup that did not
generalize*, not as a stem-independent operation.

## Honest scope + next (no capability abandoned)

<!--derived-->
This is a NO-GO for realizing BOTH routes in ONE shared heteroassociative pool at this operating point — a verdict
on the METHOD, not on "productive morphology." The mechanism to test next follows directly from the diagnosis and
from the theory: Pinker–Ullman posits the rule and the exceptions live in *separate systems* (a procedural
basal-ganglia rule vs a declarative temporal-lobe store), precisely so the default is not captured by the
whole-form attractors. The next lever is a SEPARATE procedural route for the affix — a stem-independent
PAST→"-ed" binding that does not share the pool with the whole-form store, so the default wins for novel stems and
the declarative store only intervenes (blocks) for the entrenched ones. The declarative half being already solid
means the next arc is narrow: give the rule its own route. Characterized boundary with the next mechanism named.
