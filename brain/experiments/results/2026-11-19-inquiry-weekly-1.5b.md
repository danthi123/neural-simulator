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
