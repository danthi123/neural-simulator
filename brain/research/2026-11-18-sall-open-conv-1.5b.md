# Pre-registration: the S_all memory key judged on open conversation (1.5B), milestone 11-30 "recall from reworded questions"

Written 2026-11-18, before any run. Capability: Memory selection; board #282. Primary measure: the open-conversation
eval (`brain/eval/open_conv.py`), read whole.

**Record brief.**
- The open-conversation transcripts already have this gap in them. Count is from the banked default rows, using the new
  `right_episode` read-out applied to the old rows:
  - AWS and pool default arms, seeds 1100-1125 (`results/2026-11-1[56]-*/oc{,d}_s*.rows.json`).
  - With S0 on day 2, the hippocampus fetches the right telling for about 70% of direct questions, but for only 15
    of 90 reworded ones (17%).
  - Reworded replies are right 47/90 anyway, because the slept cortex knows the fact.
  - Meaning-based selection is what the milestone asks for, and S0 does not have it.
- S_all (all-layer matrix-delta key, `brain/hippocampus_keys.py`) has a long record:
  - On heavy rewordings it selected the right episode 0.69 vs 0.33 (0.1B) and 0.76 vs 0.10 (0.4B)
    (`results/2026-11-03-meaning-selection*.md`).
  - At 1.5B it leads S0 by about one told fact in every run at no false-recall cost, but fell short of its registered
    margins:
    - key default K2: +0.83 vs a bar of 1 (`results/2026-11-14-key-default-1.5b.md`);
    - contrast on the mixed day: +0.00 (`results/2026-11-14-sall-contrast-mixed-1.5b.md`).
  - Those tests were scripted probes; it has never been judged on open conversation.
- Known risk: the 1.5B bar (0.22 / 0.01, `KEY_BARS_BY_SIZE`) was fit on unslept reads, and the night changes S_all
  scores. The slept-bar refit was preregistered (`research/2026-11-15-sall-slept-bar-1.5b.md`) and never run.
  - So day-2 S_all may fire less, or on the wrong episode.
  - The fire rate on recall turns (`recall_fired`) and needless recall are reported to expose this.
  - Known lure: "my aunt" can route to the grandmother episode, 4/6 seeds unslept with no contrast.
- Prerequisites met: hippocampus with a cortex-state key, sleep in the live brain.
  - Host decision: none new. `right_episode` is a teacher-side read-out of which telling the hippocampus returned; it
    decides nothing.
  - Expected: selection up clearly. Reply-level reworded recall up a little, because the slept cortex already answers
    half.

**Method.** `brain/eval/open_conv.py --model 1.5b --seed <s>`.
- Arm d: default (S0).
- Arm k: `--key S_all --key-bar size`.
- Fresh seeds 1126-1131, on pool CPU fp32: pool40 1126/1127, pool41 1128/1129, pool42 1130/1131.
- Both arms of a seed run on the same node.

**Criteria (six seeds, intact arm, day 2).**
- S1 (meaning-based selection): reworded right_episode summed over the six seeds, k >= d + 4 (of 18), and k >= d on
  >= 5/6 seeds.
- S2 (replies not worse):
  - reworded + direct ok, summed, k >= d - 1;
  - abstentions on memory questions ("not sure", "don't know") k <= d + 2.
- S3 (no new harm):
  - false + invented on never-told questions, summed, k <= d + 2;
  - needless recall on open turns k <= d + 3;
  - intrusions k <= d + 2.
- S4 (reading): one fresh sonnet reader per seed sees the two day-2 conversations under shuffled letters and scores
  each 1-10. S4 holds if k >= d on >= 5/6 seeds.
- Lesion read-out (reported, not a criterion): the hippocampus-off arm is the same brain with the key's effect
  removed. k's reply gain over d in the intact arm should not appear in the hippocampus-off arm.

**Decision.**
- S1-S4 hold: flip the 1.5B default key to S_all with the size bar, after a land-result skeptic and six seeds on
  42/43/44/100/101/102 with the weekly as a report. Record the milestone item as met for selection.
- S1 holds and S2-S4 hold except S4: selection is met as a read-out, no flip. The next method is the slept-bar refit
  (the preregistered stage 1).
- S1 fails with fire on recall turns low (k recall_fired < d - 4): the unslept bar is the cause. Run the slept-bar
  refit, then this test again on fresh seeds.
- S1 fails with fire normal: S_all does not select by meaning after a night at 1.5B. Bank it NO-GO on the method.
  Next: a key read from the cortex's own question representation (a self-key on the slept cortex).
