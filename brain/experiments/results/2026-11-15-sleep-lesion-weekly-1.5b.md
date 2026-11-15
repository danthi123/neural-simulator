# Result: after sleep, switching off the night's synapses visibly changes the weekly replies (1.5B, six seeds, pool CPU): L0-L3 HOLD

Prereg: `brain/research/2026-11-15-sleep-lesion-weekly-1.5b.md` (L0-L3, written after a report-only smoke and before
any registered seed). Runs: pool40 (s42, s43), pool41 (s44, s100), pool42 (s101, s102), `device cpu` fp32,
`brain.eval.weekly --model 1.5b --shared-sleep --sleep-lesion`, all rc 0. Transcripts:
`brain/eval/transcripts/sleep-lesion/sl_s*.md`. Scored by the weekly scorer as written (six keyed questions).

## Keyed correct in session 2 (of 6)

| seed | intact | hippocampus off | hippocampus + night's synapses off | third arm replies differing from intact |
|---|---|---|---|---|
| 42 | 3 | **2** | 0 | 6 |
| 43 | 6 | 5 | 0 | 6 |
| 44 | 5 | 5 | 0 | 6 |
| 100 | 6 | 6 | 0 | 6 |
| 101 | 5 | 6 | 0 | 6 |
| 102 | 5 | 5 | 0 | 6 |
| mean | 5.00 | 4.83 | 0.00 | 6.00 |

Report-only seeds (same code, not criteria, not pooled): smoke 200-202: intact 6/6/3, hippocampus off 6/5/4, third
arm 0/0/0; 203-205 (run after the registered seeds): intact 5/6/6, hippocampus off 5/6/6, third arm 0/0/0, 6/6 keyed
replies changed on each. On s203 the third arm also declines the story ("I don't know any stories about foxes.").

## Criteria

- **L0 holds.** Three "session 2" sections on every seed; `run_shared` copies one slept home into all arms and asserts
  `sleeps >= 1` with nothing pending before each arm (`brain/eval/weekly.py:77-93`).
- **L1 holds.** 6 of 6 keyed replies differ from intact, on 6/6 seeds.
- **L2 holds.** Third arm keyed correct 0/6 on 6/6 seeds.
- **L3 holds.** Hippocampus-off arm >= 4/6 on 5/6 seeds; s42 fails (2/6).
- **VERDICT (registered): L0-L3 HOLD.** Milestone day-14 item "a lesion visibly changes replies" is met after sleep,
  by a lesion of the night's synapses.

## What the replies show (fresh sonnet skeptic: SURVIVES, all counts reproduce; wording narrowed to necessity)

- With the night's synapses off, the cortex answers as the pretrained model: "I'm sorry, but I don't have access to
  information about your cat." (s42), a generic own name ("ChatGPT", "Samantha", "Assistant", "Alex", "Lily"), and a
  wrong friend's name ("Sam"). Ordinary replies (fox story, feeling) stay fluent in that arm on every seed.
- So the night's adapters are NECESSARY for the told facts that survive the hippocampus lesion: removing them (with
  the hippocampus already off) removes the facts. This is necessity, not specificity: the lesion removes all adapter
  effects, and there is no control arm with another night's or random adapters.
- The lesion touches only the adapters: `set_lesions(consolidated_off=True)` sends `forward`, `read` and `speak` to
  the raw pretrained network; the birth state is read from the raw cortex and session 2 starts from a fresh Brain, so
  the two lesion arms differ only by the adapters (skeptic's code read).
- Engagement is recorded at the reply level only (no call count or lesion trace field); the replies differ on 36/36
  keyed rows. No "I don't know about that." in any transcript.

## Caveats

- **s42's night degraded the cortex on CPU.** Intact and hippocampus-off arms answer the fox story with "A red
  Volvo." and "Once upon a time, there was a red Volvo.", and feeling with "I drive a red Volvo." That is the
  false-recall-after-sleep defect at its worst; it is why s42 fails L3. The same seed on AWS L4 (weekly 11-15) did not
  show it.
- Own name after the night (intact): right on 2/6 seeds (s43, s100); "Dan" on s42 ("Dan is my friend."), s44, s102;
  "Marguerite" on s101. Worse than the AWS weekly (4/6). Hardware differs (CPU fp32 vs L4 fp16); not pooled.
- Host: `tools/where_ran.py` reports unknown (nothing stamps it); pool40/41/42 rests on the job launch record. Not
  pooled with the AWS weekly numbers.

## Next

- Own name and false recall after sleep stay the open defects (rank diagnosis running, `brain/research/2026-11-15-rank-diagnosis-1.5b.md`).
- `--sleep-lesion` stays a weekly option (default OFF); turning it on for the weekly record is a separate call.
