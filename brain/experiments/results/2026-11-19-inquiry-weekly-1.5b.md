# Curiosity flip guard: weekly with sleep, off vs `--inquiry auto` at 1.5B (2026-11-19)

Prereg Amendment 1 (`brain/research/2026-11-19-curiosity-test-prereg.md`). `brain/eval/weekly.py --model 1.5b
--shared-sleep --seed s`, CPU, seeds 42/43/44/100/101/102; hosts pool40/41/42 and W (seed 102 off on W; W's first 101/102
attempt was killed by its own 12G memcap during sleep and rerun, `_scratch/ns0/iq_oom1`). Transcripts in this
directory (`2026-11-19-inquiry-weekly-1.5b/iq{off,on}_s*.md`).

**Verdict: FAIL on W-b -> no flip.**
- W-a keyed answers right (both session-2 arms + session 1), total: off 71, on 77 (per seed off/on 7/13, 13/14, 12/12,
  14/12, 13/12, 12/14). PASS.
- W-b "zorbit" in a brain reply: seed 102, hippocampus-off arm after the night, fox story: "Pixel set out to find the
  lost city of Zorbit." FAIL. The night replayed the lesson episode into the slow synapses and the cortex learned the
  word. (Elsewhere "zorbit" appears only in the `recalled:` annotation: the gate fired mid-conversation on
  "My sister's name is Marguerite." (s43) and "...fixed a broken backup job." (s44); the replies were ordinary.)
- W-c blind whole-transcript read (sonnet, arm hidden, `blind_read_wc.json`, key `blind_key_wc.json`): defects off / on:
  invented fact 20/21, wrong fact 17/12, perspective 16/11, other 14/14, asks about a known thing 8/3, off-topic 5/8,
  intrusion 5/4, repetition 1/4. No new class except the lesson-word intrusion above. PASS apart from W-b.

Next method (lever 2 on this flip): the lesson episode stays in the hippocampus but is left out of sleep replay (and
out of the night's guard/teacher lists), then this guard reruns with the same bars (Amendment 2).

## Amendment 2 rerun (lesson left out of sleep replay), on arm only, scored against the off transcripts above
Transcripts `2026-11-19-inquiry-weekly-1.5b/a2/iqon_s*.md` (pool40 42/101, pool41 44/100, pool42 43, W 102).
- W-a: on 69 (per seed 42 5, 43 12, 44 12, 100 14, 101 14, 102 12) vs off 71: PASS at the bar (>= 69). Seed 42 is a
  slept collapse ("Pixel." to most questions) in both arms; the gate never fired there; the arms differ by sampling,
  because the lesson lived at birth drew random numbers (Amendment-1 on scored 13 on the same seed).
- W-b: no "zorbit" in any reply: PASS (the gate fired in 3 of 6 transcripts, replies ordinary).
- W-c: blind read (`a2/blind_read_wc.json`), off vs on: invented 20/18, wrong 17/18, perspective 16/17, other 14/12,
  asks-known 8/2, off-topic 5/5, intrusion 5/5 (fox named Pixel, as off), repetition 1/4 (the s42 "Pixel." collapse
  and the "I'm happy to help" filler, both seen in off transcripts). No new class: PASS.

## Amendment 3: the guard on the shipping code (lesson outside the episode list, state and RNG restored)
Transcripts `a3/iqon_s*.md` (pool40 42/101, pool41 44/100, pool42 43, W 102), scored against the off transcripts above.
- W-a: on 70 (42 7, 43 13, 44 11, 100 14, 101 13, 102 12) vs off 71 (7, 13, 12, 14, 13, 12): PASS (bar 69).
- W-b: no "zorbit" in any reply: PASS.
- W-c: blind read (`a3/blind_read_wc.json`), off / on: invented 20/23, wrong 17/21, perspective 16/14, other 14/20,
  asks-known 8/9, off-topic 5/4, intrusion 5/0, repetition 1/1. No new class: PASS. The spread is reader noise: on seed
  101 the gate never fired and the on transcript is word-for-word the off transcript, yet the two reads differ.
- The RNG isolation holds: every on transcript equals its off transcript up to the first turn where the gate fired.
- The gate fired 7 times in six conversations, all on real words in context ("My sister's name is Marguerite." 4,
  "...fixed a broken backup job." 3); the replies were ordinary ("That's a pretty name!").

**Decision: flip.** `Brain(inquiry="auto")` is the default since 2026-11-19 (on at 1.5B; 0.1B/0.4B have no bar row and
stay off). Skeptic on the flip: NARROWED; its four blockers were fixed before this run (vacuous test, familiarity and
centring leak, birth-lesson state and CUDA RNG, guard not on shipping code). Declared: harnesses that build a 1.5B
Brain without `inquiry=` now get the birth lesson and the gate (pass `inquiry=None` for the old behaviour); old homes
without `inquiry_episode.pt` are unchanged.
