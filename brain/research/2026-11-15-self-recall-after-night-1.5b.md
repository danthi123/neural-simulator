# Pre-registration: why "What's your name?" recalls the sister telling after the night (1.5B, no training)

Written 2026-11-15 before any 1.5B run of the probe. Capability: Self and identity; board #280. A diagnosis, no
method: it reads the brain's own recall competition on slept homes and changes nothing.

**Record brief.**
- Session 2 of the weekly (after the night): "What's your name?" recalls "My sister's name is Marguerite." on 6/6
  registered seeds in both self-completion arms; the self episode never wins (`results/2026-11-15-self-completion-
  weekly-1.5b.md`). On day 1, turn 1 (only the three self episodes stored) it wins on every seed.
- The default key S0 is the layer-0 matrix-memory delta, centred on the mean stored key, weighted by tanh(strength)
  (`brain/hippocampus.py`). S_all (every layer, size-fit bar) scored self questions 1.67 vs 0.67 on an unslept 1.5B
  day (`results/2026-11-14-key-default-1.5b.md`), not chosen as default (K2 short by 0.17).
- 0.1B smoke (pool40, 4-step night, not a result): S0 scores are identical through the slept and the raw cortex; S_all
  scores differ slightly.
- Prerequisites met; host decision none.

**Method.** `brain/experiments/self_recall_probe.py` on copies of the slept intact homes the pool's report-only weekly
runs left (`~/brain-lane/workSC_{off,on}_s<seed>/_weekly_intact`, seeds 200-205: CPU fp32, after session 2). Arms: key
S0 / S_all (size bar) x cue read through the slept cortex (live) / the pretrained cortex (raw). Cues: "What's your
name?", "Who are you?", "Hi again! Do you remember my name?", plus two told-fact cues as controls. Six seeds, 200-205,
from the off-arm homes (the on-arm homes reported beside them). Not the registered seeds: their homes were on AWS.

**Criteria ("What's your name?", off-arm homes).**
- P0 (engagement): S_all live and raw scores differ on every home; S0 live vs raw reported (a zero difference means
  the night does not reach the layer-0 key, which is then itself a finding).
- P1 (the night's read causes it): S0 raw recalls the self episode on >= 5/6 homes while S0 live does not.
- P2 (the stored competition causes it): S0 recalls the sister telling (or no self episode) in both reads on >= 5/6.
- P3 (S_all is the next method): S_all live recalls the self episode, passing its bar, on >= 5/6 homes, and both told
  controls still recall their telling.

**Decision.** P1: next method reads the recall cue with the pretrained cortex. P2 and P3: next method is a
pre-registered S_all test for the weekly's self questions. P2 without P3: the next method changes what the self
episode is keyed by (for example its question). Otherwise reported per home.

**Expected.** P2 (the smoke shows S0 unchanged by the night); P3 uncertain (the smoke had S_all lose too, at 0.1B).
