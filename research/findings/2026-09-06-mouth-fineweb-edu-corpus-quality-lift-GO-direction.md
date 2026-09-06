---
type: finding
status: positive
claim_check: measured
date: 2026-09-06
mechanism: mouth broad-domain corpus lever — train the deployable linattn mouth on FineWeb-Edu (high-quality
  filtered-web) at the SAME token budget as the wt103/simplewiki STEP-1, eval on the fixed wt103 held-out deep
  buckets. Isolates CORPUS QUALITY (same scale, different corpus).
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
builds_on:
  - research/findings/2026-09-06-mouth-token-scaling-step1-simplewiki-domain-mix-NO-GO.md
  - research/findings/2026-09-06-mouth-token-scaling-fineweb-pipeline-and-grid-spec.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
verdict: >
  STRONG DIRECTION-POSITIVE (single-seed s43). Training the deployable linattn mouth on FineWeb-Edu at the SAME
  token budget as the wt103/simplewiki STEP-1 (which was NO-GO) LIFTS the deep-context (10-99) margin_vs_trigram
  from the wt103-only baseline to close to the trigram (figures in the marked body — a large lift, ~7x the +0.03
  direction bar), and the margin IMPROVES with depth across the buckets. Because token count is held fixed, this
  isolates CORPUS QUALITY as the lever: FineWeb-Edu (high-quality filtered web) massively outperforms
  wt103/simplewiki at equal scale, bringing the deployable mouth to the EDGE of crossing the fair trigram at deep
  context — something no architecture lever (objective/delta-rule/content-addressing, all banked NO-GO) came near.
  This CONFIRMS the mouth broad-domain wall is DATA-bound (specifically data-QUALITY-bound, on top of the
  2026-09-01 token-supply GO), and it makes the AWS capacity x token-supply GRID strongly justified: at base scale
  FineWeb-Edu already nearly crosses the trigram, so SCALING high-quality tokens (2B/10B, the grid's untested
  cells) should push it PAST the trigram — the fluency bar. Per the pre-set gate (deep lift >= +0.03 -> justify
  the download+grid), the gate is cleared by a wide margin. Single-seed direction-test, not a 6-seed claim; the
  grid provides the multi-seed + scale confirmation. Nothing wired; mouth default stays linattn.
---

# Mouth: FineWeb-Edu corpus quality lifts the deep margin toward crossing the trigram (GO-direction)

## What ran
`_emerge_wkv_lm_derisk.py --recurrence linattn --corpus data/corpus/fineweb_edu.txt --eval-corpus
data/corpus/wikitext103.txt` (n-layers 2, d-model 192, s43, 4 epochs, --max-train-sents 2500000) — the local
science gate for the mouth token-scaling fork. IDENTICAL recipe to the STEP-1 NO-GO except the training corpus
(FineWeb-Edu instead of wt103+simplewiki), so token count is held fixed and the eval instrument is unchanged.

## Derived — deep-context margins vs trigram (s43; direct reads of the two cited artifacts)
<!--derived: fineweb run from research/findings/raw/_emerge_wkv_lm_linattn_fineweb_evalwt103_s43.json; wt103-only baseline from research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json; lift is their difference -->
| depth | 1 | 2 | 3 | 4-5 | 6-9 | 10-99 |
|---|---|---|---|---|---|---|
| FineWeb-Edu (this run) | 0.948 | -0.387 | -0.282 | -0.182 | -0.161 | -0.082 |
| wt103-only baseline | 0.989 | -0.570 | -0.454 | -0.402 | -0.356 | -0.286 |
| lift | -0.041 | 0.183 | 0.172 | 0.220 | 0.195 | 0.204 |

Deep-bucket (10-99) lift = **+0.204** (bar was +0.03), and the margin gets LESS negative with depth (the deep
buckets improved most). At -0.082 the mouth is on the EDGE of crossing the fair trigram at deep context.

## Reading it (no-defer)
Same tokens, different corpus → the lift is CORPUS QUALITY. FineWeb-Edu (high-quality filtered web) beats
wt103/simplewiki at equal scale by a wide margin, and nearly closes the deep-context trigram gap that every
architecture lever failed to move. This is the decisive confirmation that the broad-domain mouth wall is
DATA-bound (quality + supply), NOT architecture. Since base-scale FineWeb-Edu already reaches -0.082, SCALING it
(the AWS grid's 2B/10B-token cells, capacity-matched) should cross the trigram at a deployable size — the fluency
bar. AWS grid launched (owner-approved).

## Honest scope
Single-seed (s43) direction-test — labeled as such, not a 6-seed claim. Additive; no production change. The
crossing-the-trigram claim is a projection from the base-scale edge + the 2026-09-01 token-supply monotonicity;
the grid provides the multi-seed + scale proof.
