---
type: finding
status: negative
claim_check: measured
date: 2026-09-05
mechanism: delta-rule error-corrective write (`--recurrence deltanet`) on the linattn mouth — wt103 broad-domain s43 direction-test vs the linattn baseline
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/findings/raw/_emerge_wkv_lm_deltanet_wt103_scale_s43.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
builds_on:
  - research/findings/2026-09-05-deltanet-delta-rule-write-on-linattn-BUILT-wt103-direction-test-queued.md
  - research/findings/2026-09-05-mouth-broad-domain-fluency-deep-research-ladder.md
verdict: >
  NO-GO on the pre-registered single-seed direction bar (a labeled DIRECTION-TEST, not a 6-seed headline). The
  delta-rule erase-before-write (`deltanet`, rung-3 of the mouth broad-domain ladder) was run on the exact
  byte-identical wt103 config as the linattn baseline at s43. It lifts the deep-context (positions 10-99) margin
  vs a fair trigram from linattn's floor to a still-negative value — a small, real, but SUB-THRESHOLD improvement
  (below the pre-registered "lift the deep bucket by >=0.05 -> escalate to 6-seed" gate; exact figures in the
  marked body). It stays below the trigram at every depth >=2 (it only wins the trivial depth-1 bucket), so the
  erase-before-write ALONE does not cross the broad-domain bound. Per THE LAW this is a METHOD verdict on the
  delta-rule as a STANDALONE lever, not a capability wall: it is orthogonal to (composes with) the predictive-coding
  OBJECTIVE (rung-1, a wt103 A/B running now) and CAPACITY (rung-2), so its small lift may stack — the honest read
  is "a real but insufficient-alone component," to be revisited if the objective A/B lifts the floor. Not escalated
  to 6-seed on its own (the direction bar was not met). Nothing wired; additive/default-off arm, no production change.
---

# Delta-rule (deltanet) on the mouth, wt103 broad-domain: a modest sub-threshold lift — NO-GO alone

## What ran
`research/runners/_emerge_wkv_lm_derisk.py --recurrence deltanet` on the byte-identical wt103 config as the linattn
baseline (`--pred-aux-weight 0.0`, d_model=192, n_layers=2, 3M sentences / 2.5M train / 4 epochs, s43), harvesting
`research/findings/raw/_emerge_wkv_lm_deltanet_wt103_scale_s43.json` (11921 s on the GPU queue). This is the
direction-test the delta-rule BUILD finding queued; it ran post-reboot from the surviving pre-crash queue.

## Derived — margins vs trigram (s43; all values direct reads of the two cited artifacts)
<!--derived: deltanet by-depth from _emerge_wkv_lm_deltanet_wt103_scale_s43.json; linattn baseline from _emerge_wkv_lm_linattn_wt103_scale_s43.json; the lift is their difference -->
| depth bucket | 1 | 2 | 3 | 4-5 | 6-9 | 10-99 |
|---|---|---|---|---|---|---|
| deltanet margin_vs_trigram | +1.098 | -0.498 | -0.403 | -0.352 | -0.329 | -0.251 |
| linattn baseline (deep 10-99) | | | | | | -0.286 |

Deep-bucket (10-99) lift of the delta-rule over plain linattn: **-0.251 vs -0.286 = +0.035** — a real improvement,
but below the pre-registered +0.05 direction bar, and still negative (sub-trigram). Deltanet wins only the trivial
depth-1 bucket, like every prior arm.

## Reading it (no-defer)
The erase-before-write DOES help the broad-domain deep-context read (the sub-threshold lift tabled above), consistent with its design target
(interference reduction), but not enough alone to clear the bar or cross the trigram at s43. Because it is a
WRITE-RULE change orthogonal to the training OBJECTIVE (rung-1) and CAPACITY (rung-2), its small lift is a
candidate to STACK rather than a dead end. NEXT: read the rung-1 objective A/B (linattn `--pred-aux-weight 1.0`,
running); if the objective lifts the floor, test objective+delta-rule together; independently, the capacity sweep.
Not escalated to 6-seed on its own (single-seed direction bar not met).

## Honest scope
Single-seed (s43) direction-test — labeled as such, not a 6-seed generalization claim (the single_seed discipline).
Additive default-off arm; no production change; the mouth default remains linattn.
