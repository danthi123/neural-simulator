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
