# Stage-1 dress rehearsal at 1.5B (s318, DEV, every flag): harness runs end to end; two instrument defects fixed before the TEST launch

DEV only, one seed: an instrument check, not evidence. Design `brain/research/2026-11-19-multiweek-life-design.md`
§13. Code b83a6c036 (`--pruned-a --cue-ladder --interview` with `--span --transfer --frozen-probe --weekend`, 15 days,
full probes on nights 1, 5, 10 and 15), defaults (B). AWS g5.xlarge (A10G, fp16), batch 20261120-1917, host
ip-172-31-45-106. Data: `_scratch/brainv2/aws/20261120-1917/life_s318_S1FULL.json.{json,md}` (not committed).

## Runs end to end

- rc 0 in 4.0 h: 15 trunk nights plus 3 L-prunedA branch nights, about 640-680 s per night plus about 2 min of
  probes per day. Pool CPU nights for the same life take about 1530-1670 s (s315-317 DEV run checks).
- L-prunedA: cut at L11, 8 A episodes deleted (1 per item, store 68 -> 60, no mismatch, no kept mentions).
- Cue ladder: 19 cells, 1-6 items per cell per arm. The instrument separates fetching from use. For example, a context
  cue fetched the right episode ("I always do the crossword on the train") while the reply named another told fact
  ("Patagonia"). So with one seed, R_cue is about 0 in most cells.
- Interviews: 3 nights x 3 arms x 15 replies, 165 units. The read pipeline (blind prompts -> sonnet extractor ->
  `life_interview_read.py`) runs with 0 missing rows (`_scratch/stage1/ivread318/read.md`). Readings: AUROC 0.33 (null
  0.50); invented memory 0.92 intact vs 1.00 both-off; over-abstention 0.00; 5 asserted experience claims in intact
  (for example "I stayed up too late watching a documentary about bridges."). The KNOW question is answered mostly by
  the hub reply "Of course, you're Dan, my friend." whatever the topic. That is a brain defect (the hub), so the spec
  wording was NOT changed after seeing it.

## Instrument defects found, fixed on DEV before any TEST read

1. **Savings null.** The pretrained both-off cortex already rates A above B: raw both-off G is +3.4, against raw
   prunedA G +5.7. So raw G mixes item difficulty with savings, and the §5 "both-off G about 0" bar cannot hold by
   construction. Fix: the registered G is per-item null-subtracted, `G_null_sub` (and `dR_null_sub`), in
   `life.savings_table`. On s318 (strict scorer, P0 unread) G_null_sub is +2.28, +2.39 and +1.92 on branch nights 1-3.
2. **Language guard missing.** The §5 per-night KL/NLL against pretrained on held-out ordinary text was never in the
   harness. Built as `--lang-guard`.
3. **Per-seed norm kill unusable.** The adapter norm grows every night under the defaults: 59 on night 1, 672 on
   night 15, and above 3x night 1 from night 4. The §10 clause would end every life near night 4. It is dropped and
   the norm is reported; the kill is collapse on 2 consecutive probe nights, or KL above 3x night 1 (§13).

## Other DEV observations (for Stage 1, no claim)

- H int/off falls from 5-6/6 on nights 1-2 to 1-3/6 by nights 13-15; wrong fetches rise to 7-8 of 8-10 by nights
  14-15 (the hub). Collapse at most 0.4 (intact, night 3 only).
- Lived replies grow stuck tails ("I always do the flight." on L11 of the branch).

Also fixed: `tools/aws_brain_batch.sh`. A stopped g5 could not restart (InsufficientInstanceCapacity), so results were
stranded until a retry. `AWS_BRAIN_GRACE_MIN` now keeps the instance up after the jobs end, so a pull loop copies
results first. `AWS_BRAIN_STATE` lets several batches run at once.
