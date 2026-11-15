# Pre-registration: M4b, the frame guard minus told-topic cues, against false recall after sleep (1.5B, six seeds, AWS)

Written 2026-11-15 before any run of the method. Capability: Sleep consolidation + Metacognition (false recall).

**Record brief.**
- M4, the frame guard in sleep (`results/2026-11-14-near-miss-guard-1.5b.md`), is NOT A LEAD.
  - It cut false recall from 6.83 to 3.83 of 12 (lower on 11/12 seeds).
  - It cost told recall (7.17 -> 6.00) and paraphrase (-1.5).
  - About half of the removed intrusions became abstentions or inventions.
  - It refused some told facts (told-row abstentions 0 -> 7 of 72).
- The matched weaker-night control (`results/2026-11-15-weak-night-control-1.5b.md`) found it SELECTIVE: a shorter
  night keeps false recall at 9.00 vs the guard's 5.50 at matched slept told. A weaker night does not cut false
  recall at all. Selectivity is narrowed: the told match was by one item, and about 60% of the gain was abstention.
- The told cost has a named cause in the M4 prereg ("the method's own cost"). Some frame cues ask about a TOLD topic
  ("Who is my best friend?") and hold the pretrained "I don't have that information" on it.
- Lever count on this defect: 8 with this one. It is the second form of the method family the control said is
  worth keeping (11-03 plan, M4; CLS interleaving).

**Mechanism (`Brain(near_miss=4, near_miss_frame="FT")`, switch default OFF).** Frame F as before. A kept cue is
also dropped when its possessed word ("my X", with a hand-written modifier list skipped) appears in what the user told
(the user part of the brain's own stored episodes). This is a brain-side rule (`brain/near_miss.py` `told_topic`;
SCAFFOLDS.md row). The probe's test-hygiene filter is unchanged. On the registered M4 cue logs, the rule would drop
16 of 113 kept cues (told topics from all 12 facts, an upper bound; the night only knows its slept 6). This is
descriptive, and no outcome was read. Host decision: none.

**Method.** As M4 (`mixed_day_probe.py --arms c,s,r --self-q`, 1.5B, `--steps 480 --lr 5e-4 --kl-weight 3`, same
slept halves, seeds 42/43/44/100/101/102, one AWS batch).
- New: `nmft_s*` (`--near-miss 4 --near-miss-frame FT`).
- Off arm: the banked `nmoff_s*` of batch 20261114-2031. It is deterministic: it reproduced the 11-14 tellmix exactly.
- Reproduction check: `nmoff_s42` is rerun in this batch and must equal the banked file in told/never per arm.
- Rows are re-scored before `near_miss_table.py --on nmft` (which adds N5-N6).

**Criteria (arm s, six-seed means, of 12 unless said).**
- N0: the nmoff_s42 rerun reproduces the banked file; nmft logs >= 1 kept cue per seed and frame FT.
- N1-N4: as M4.
  - N1: false <= 0.5 x off and lower on >= 5/6.
  - N2: told >= 0.85 x off.
  - N3: para >= off - 1.
  - N4: self >= off - 0.5.
- N5 (new): invented answers on never-told rows (not a told fact, not an abstention) <= off + 1. The guard must not
  swap intrusions for inventions.
- N6 (new): abstentions on told rows <= off + 0.5. It must not refuse taught facts.

**Decision.**
- GO = N0-N6: land-result, then the flip question with an answer-rate ruling, since abstentions on never-told rows
  are the intended direction.
- LEAD = N0, N2-N6, and false lower by >= 2 on >= 5/6.
- NOT A LEAD otherwise.

**Expected.** Told and paraphrase costs shrink (fewer told-topic cues), false recall falls a little less than under M4,
and inventions do not rise. If the told cost stays the same, it is not carried by the told-topic cues: the guard and
the told bindings compete in one rank-8 adapter (the capacity risk named in M4), and the next method is M5.
