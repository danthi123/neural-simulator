# 2026-11-03 S_all meaning key in the live Brain (switch, default OFF) -- PRE-REGISTRATION (written before any run)

Branch `research/sall-in-brain-1103` (from `research/landing-1103`). Capability: memory selection
(`brain/PREREQUISITES.md`). Record: `2026-11-03-meaning-selection-realpath.md` (S_all lead, dev-fit bar 0.185 / 0.05,
recommendation: a live-brain test before any flip) and `2026-11-03-live-brain-12facts.md` (the probe this reuses).

## Change
- `Brain(key=None)`: None = the key the home stored, else "S0". "S0" builds the plain `Hippocampus` (default; the
  saved `memory.pt` has exactly the old fields). "S_all" builds `KeyedHippocampus(key="S_all", familiarity=0.185,
  margin=0.05)` (`Brain.KEY_BARS`); episodes keep no separate key (read from the stored delta). A non-S0 key is saved in
  `memory.pt` (`"key"`); an explicit `key=` wins at load; opening an S_all store as S0 rebuilds each S0 key from the
  stored fp16 delta. The feeling-of-knowing bar (`KNOW_BAR_HIPPOCAMPUS` 0.22, abstain switch, default OFF) was fit for
  S0 and is NOT refit here: abstention with S_all is untested.
- Unit tests `brain/tests/test_brain_key_switch.py` (stand-in cortex, no weights): default Brain vs the
  `research/landing-1103` Brain: same replies, infos and saved store (texts, deltas, keys, state) byte-equal; S_all bar
  and no stored key; persistence and switching both ways; S_all similarity = centred cosine of all-layer deltas.

## Probe (`brain/experiments/sall_live_probe.py`)
0.1B (`rwkv7-g1d-0.1b`), CPU, Brain sleep defaults (240 steps, lr 5e-4, kl 1.0, rank 8). Seeds 42/43/44/100/101/102.
Day 1 as the 12-fact probe (`live_brain_probe.tell_day`, default key), copy before the restart, the restart sleeps once.
Both keys answer from the SAME homes (storage does not depend on the key): paired arms differ only in the key.
Arms: a_S0 / a_Sall (slept, intact; PRIMARY), b_S0 / b_Sall (never slept, intact; secondary), c_hippo_off (slept,
hippocampus off; lesion). Questions: 12 originals, 12 PARA rewordings, 16 never-told (4 `sleep_probe.NEVER_TOLD` +
12 `facts_bank.NEVER_TOLD`, test bank, never used to fit). Fresh Brain per question, sampled reply, seeded per question.
Strict scorer `strict_scorer.score` (primary, unchanged); `correct_answer` reported. False recall = any of the 12 key
words as a word in a never-told reply (/16). Read-outs: selection (recalled episode = the asked fact's telling),
never-told fire (any episode recalled).

## Criteria (fixed now; instrument not changed after the first run)
- P1 (paraphrase): strict paraphrase hits a_Sall > a_S0 by mean >= +1.0 /12, and a_Sall >= a_S0 on >= 5/6 seeds.
- P2 (no false-recall rise): mean false recall a_Sall <= mean a_S0 + 0.5 (/16), and a_Sall > a_S0 on <= 2/6 seeds.
- P3 (originals not lost): mean strict recall a_Sall >= mean a_S0 - 0.5.
- Reported, not criteria: the same three on b (hippocampus alone); selection and never-told fire per arm; correct_answer
  counts; lesion c vs a_Sall (the hippocampus-off brain must answer differently; the reply is what is scored).
Verdict: P1-P3 all hold = a lead for S_all in the live brain (not a GO, not a default flip: one probe, 12 facts, 0.1B,
one sampled reply per question, the S_all bar fit offline on dev facts). Any failure is reported as that criterion failing.
Known limits now: the bar 0.185 / 0.05 was fit on offline one-sentence cues (no conversation state); the paired design
means day 1 was lived with S0 (only spontaneous recall during the telling could differ under S_all).

---
# RESULTS (six seeds, 0.1B, CPU; numbers from `2026-11-03-sall-in-brain/seed*.json`, table by `sall_live_table.py` -> `table.txt`)
Instrument unchanged after the pre-registration commit (one pipeline smoke at 2 sleep steps, seed 7, not scored here).
Unit tests: `brain/tests/test_brain_key_switch.py` 5/5 pass (default Brain byte-equal to `research/landing-1103` on the
stand-in cortex). Check of the instrument: a_S0 reproduces the 12-fact probe's arm a exactly (recall 6.50, paraphrase
5.50, same per-seed values), b_S0 its arm b (6.33 / 3.83).

Mean over seeds [42,43,44,100,101,102]. Strict scorer; false recall /16; selection counts are identical on every seed
(storage and cues are deterministic; seeds change only sleep and the sampled reply).

| arm | recall /12 | paraphrase /12 | false recall /16 | correct_answer recall / para | sel told / para | never-told fired /16 |
|---|---|---|---|---|---|---|
| a_S0 (slept) | 6.50 [5,6,9,6,7,6] | 5.50 [5,5,6,5,6,6] | 13.00 [14,13,14,13,13,11] | 1.83 / 2.17 | 11 / 7 | 14 |
| a_Sall (slept) | 5.83 [4,6,7,6,6,6] | 5.33 [5,5,4,6,6,6] | 13.00 [14,14,8,15,14,13] | 3.17 / 4.00 | 9 / 5 | 4.5 |
| b_S0 (no sleep) | 6.33 | 3.83 | 9.33 | 5.00 / 3.33 | 11 / 7 | 14 |
| b_Sall (no sleep) | 4.83 | 2.17 | 5.33 | 3.83 / 2.17 | 9 / 5 | 6 |
| c_hippo_off (slept) | 5.50 | 3.33 | 11.83 | 6.00 / 3.67 | 0 / 0 | 0 |

## Criteria (primary arm a)
- P1 FAIL: paraphrase a_Sall - a_S0 = -0.17 (needed >= +1.0); a_Sall >= a_S0 on 5/6 seeds.
- P2 FAIL: false recall difference +0.00 (within +0.5) but a_Sall > a_S0 on 4/6 seeds (allowed <= 2).
- P3 FAIL: recall a_Sall - a_S0 = -0.67 (allowed >= -0.5).
- Secondary b (hippocampus alone, not a criterion): paraphrase -1.67 (0/6 seeds >=), false recall -4.0 (lower on 6/6), recall -1.50.
- Lesion: hippocampus off changes the reply on 103/240 questions vs a_Sall (sampled replies; c answers 5.5 / 3.3 from the slept synapses).

## Verdict
Not a lead: S_all at the offline bar 0.185 / 0.05 does not raise strict paraphrase recall in the live brain at 0.1B
(5.33 vs 5.50), and false recall is not lowered in the slept brain. What the read-outs show:
1. The key changes WHICH and WHETHER, as offline: S0 fires on 10/12 rewordings, 3 of them on the wrong episode (e.g. "What's
   the name of my kitty?" -> the brain's own name episode), and on 14/16 never-told questions; S_all fires on 5/12
   rewordings, all on the right episode, and on 4.5/16 never-told. The bar is too strict for live cues (it was fit on
   one-sentence offline cues): S_all selects 2 fewer originals (9 vs 11) and 2 fewer rewordings (its 5 are a subset of S0's 7 correct ones).
2. Fewer hippocampal misfires do not reach the reply after sleep: false recall in the slept brain stays 13/16 with either
   key, and the hippocampus-off brain already says a fact word on 11.8/16 never-told questions. Sleep's consolidated
   synapses, not hippocampal selection, are the main source of false recall here. Without sleep (b) S_all cuts false
   recall 9.3 -> 5.3 /16 at a cost of 1.7 rewordings.
3. The correct_answer scorer (reported, not a criterion) rates a_Sall higher (recall 3.2 vs 1.8, paraphrase 4.0 vs 2.2):
   why (e.g. echoes or wrong-subject answers after S0 misfires) was not analysed. Not used to rescue the verdict.
Caveats: one sampled reply per question; 12 facts; the strict scorer's known errors (12-fact writeup); selection
differences are one deterministic outcome per arm, not six independent ones.
Next step: keep the switch OFF. Refit the S_all bar on live-brain cues (dev facts told in conversation, cue read with
the conversation state), then rerun this probe unchanged; and put the false-recall control on the consolidated path
(the gate in front of sleep's synapses), since the key cannot reduce false recall that comes from the cortex.
