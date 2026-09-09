---
type: finding
status: live
claim_check: measured
date: 2026-09-09
mechanism: mouth token-scaling — same-domain FineWeb-Edu training-corpus scale (linattn d192)
lane: language
seeds: [43]
seed-waiver: single-seed DIRECTION test (does same-domain token supply lift the deep margin?) — no
  generalization claimed from one seed; the decisive multi-seed / multi-capacity confirmation is the AWS
  capacity x token-supply grid (spec 2026-09-06 §5b), gated on this direction GO.
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.json
builds_on:
  - research/findings/2026-09-06-mouth-token-scaling-fineweb-pipeline-and-grid-spec.md
  - research/findings/2026-09-05-mouth-token-scaling-step1-simplewiki-domain-mix-NO-GO.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
---

# Mouth token-scaling STEP-2 direction test: same-domain FineWeb-Edu training LIFTS the deep-context margin by ~+0.2 across every deep bucket — GO; STEP-1's NO-GO was a domain-MIX artifact, not a token-supply failure

## The test
The owner-decided fork (2026-09-05) is TOKEN-SCALING: the 2026-09-01 6-seed GO showed more unique same-domain
tokens give monotonic deep-context improvement, but wt103 is ~exhausted, so the decisive same-domain scale test
needed a corpus download (FineWeb-Edu, approved). This is that test's DIRECTION cell (spec 2026-09-06 §5a): train
linattn d192 on `data/corpus/fineweb_edu.txt` (2.5M passages), eval on the SAME byte-comparable held-out
instrument (wt103 deep-context `margin_vs_trigram`) as the -0.286 baseline and the -0.312 STEP-1 result. <!--derived-->
Bar: deep-bucket (depth 10-99) margin lift >= +0.03 off the -0.286 baseline. <!--derived-->

## Result — decisive lift on EVERY deep bucket (artifact `research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.json`, seed 43, d192, ~4.8h)
`margin_vs_trigram` by depth bucket, this run vs the wt103-only baseline (baseline row from the cited
2026-09-06/2026-09-05 findings' tables):

<!--derived-->
| depth | wt103-only baseline | FineWeb-Edu (this run) | lift |
|---|---|---|---|
| 2      | -0.570 | -0.387 | +0.183 |
| 3      | -0.454 | -0.282 | +0.172 |
| 4-5    | -0.402 | -0.182 | +0.220 |
| 6-9    | -0.356 | -0.161 | +0.195 |
| 10-99  | -0.286 | -0.082 | +0.204 |

The decisive deep bucket (10-99, n=183,709) lifts **+0.204**, ~7x the +0.03 bar; the lift is CONSISTENT across <!--derived-->
every deep bucket (+0.17 to +0.22). Shallow depth-1 is ~unchanged (0.989 -> 0.948, both strongly positive — the <!--derived-->
easy regime was never the question). `wkv_perm` (the shuffled-context control) stays ~8.6-9.0 nats — far worse
than `wkv` (~5.3-5.9) — so the model is genuinely USING context, not memorising position.

## What it means
1. **The token-supply lever is confirmed on the SAME-DOMAIN scale test**, not just the 2026-09-01 proxy: 3x more
   same-domain (FineWeb-Edu) training tokens lift the deep-context margin substantially and uniformly. This is a
   GO on the DIRECTION the owner's fork bet on.
2. **STEP-1's NO-GO (-0.312 <!--derived-->, wt103+simplewiki) was a domain-MIX artifact, not a token-supply failure** — adding
   a simpler out-of-domain corpus hurt; adding MORE of the comparable-difficulty domain helps. The distinction
   the STEP-1 finding flagged is now empirically resolved.
3. **The margin is still NEGATIVE at deep context (-0.082 at 10-99): the mouth is still sub-trigram there**, but
   climbing decisively toward crossing it. This is a direction GO (more tokens help), NOT yet a "crosses the
   trigram at deep context" claim — that is what the scale-up grid tests.

## NEXT (no-defer): the AWS capacity x token-supply grid (spec §5b)
The direction is GO, so escalate token supply + capacity to push the deep margin toward/past 0: Stage A
(d96/d192 x {0.4B, 2B}) then, if it keeps lifting, Stage B (10B). The 2B/10B cells need a >=70 GB-RAM instance
(passage-pool RAM ~69 GB at 2B) so they run on AWS, not the 46 GB local box; the d96 x 0.4B cell (~14 GB RAM)
runs locally. HONEST BOUND: this is 1 seed at 1 capacity/token-point — the grid is the decisive multi-cell
confirmation. Qwen remains the declared open-prose scaffold until the mouth crosses the trigram at deep context
on the broad domain. Functional read-out only; not a phenomenal claim.
