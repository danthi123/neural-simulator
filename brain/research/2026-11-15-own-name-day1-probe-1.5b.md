# Pre-registration: why the recalled self episode does not carry the own name on day 1 (1.5B, no training, pool CPU)

Written 2026-11-15 before any 1.5B run of the probe (a 0.1B smoke ran first, below). Capability: Self and identity;
board #280 (high-use facts never wrong). A diagnosis, not a method: no switch, no default changes.

**Record brief.**
- Day 1, turn 1 ("Hi! What's your name?"): right on 2/6 registered seeds, though the right self episode is recalled
  on 6/6 (`results/2026-11-15-weekly-1.5b-milestone.md` addendum). Wrong: "[name]", "David", "Sam", "Bob".
- Gain 0.5 at 1.5B was fitted on told facts only (`results/2026-11-13-gain-refit-1.5b.md`,
  `2026-11-13-live-gain-1.5b.md`). Its one self check (`gain_identity_check.py`) was greedy, one run: 0.5 answered
  "My name is Nova.", 4 garbled it ("Nova, what's your name?"). The self episodes' gain was never measured under the
  brain's own sampling (`speak`: temperature 1, top_p 0.3, penalties 0.4).
- Turn 1 starts from the birth state on every seed (asserted in the probe), so seeds differ only in the sampler.
- Scoping workflow (code path, record, ranked methods, sonnet critic; 2026-11-15): strength, not selection, is the
  likely cause; ranked methods: strength-weighted reinstatement of permanent episodes, an answer-only delta, a layer
  band; a readout-sharpening rule was dropped (host rule on sampling).
- 0.1B smoke (pool40, 3 samples): name sampled 0/3 at gain 0 and 0.5, 3/3 at 4; the first token " I" leads at 0.5
  ("I'm called Alex."), so a reply can skip the "My name is" frame.
- Lever count on the own-name defect: 0 (first diagnosis). Prerequisites met; host decision none.

**Method.** `python -m brain.experiments.own_name_probe --model 1.5b --samples 12`, pool CPU fp32. Gains 0, 0.25,
0.5, 0.75, 1, 1.5, 2, 3, 4; prompts live ("Hi! What's your name?") and bare ("What's your name?"); delta full and
answer-only (norm-matched). Read-outs: top-5 first tokens; P(" Nova" | "Assistant: My name is") and its rank; 12
sampled replies with the brain's own speak defaults (seeds 1000-1011), share naming Nova.

**Criteria (live prompt, full delta = what the brain does now).**
- D0: the recalled episode is the self episode for both prompts; gain 0 names Nova in <= 1/12 samples.
- D1, strength: some gain g in (0.5, 4) names Nova in >= 10/12 samples with replies that are not garbled (by reading:
  a sentence giving the name, not "Nova, Nova." or the question echoed), while 0.5 names it in <= 6/12.
- D2, frame: at 0.5, P(" Nova" | "My name is") is rank 1 but the first token is not " My" in most samples.
- D3, cue mismatch: bare names Nova in >= 4 more of 12 samples than live at 0.5.
- D4, answer-only: answer-only names Nova in >= 3 more of 12 samples than full at the same gain, at 0.5 or at the
  best D1 gain.

**Decision.**
- D1 holds: next is a method that makes permanent (high-use) episodes reinstate more strongly (strength-weighted
  gain), pre-registered with six live seeds and the told-fact controls (told episodes keep 0.5).
- D4 holds (with or without D1): the answer-only delta joins that test as a second arm.
- D3 holds: the cue form matters; the method's test must ask with the greeting.
- None holds: strength is not the cause; next is reading the self episode's delta (what it stores) before any method.

**Expected.** D1 (strength): gain 4 named it 3/3 at 0.1B, and 0.5 was fitted to a different job.
