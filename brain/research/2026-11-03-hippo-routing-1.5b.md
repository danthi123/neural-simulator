# Hippocampal routing at 1.5B: diagnosis and a pre-registered fix (2026-11-03)

Capability: memory selection (pattern completion). Branch `claude/hippo-route-1103`.

## Record brief
- Known: the live 1.5B run (`results/2026-11-03-live-brain-1.5b.md`, caveats 1-2) recalls the right day-1 episode for
  6/12 told questions on every seed; arm a (hippocampus on + sleep) 4.50 < arm c (hippocampus off) 9.33.
- Known: the layer-0 key S0 already selected badly at 0.4B offline (forced orig 0.45 vs S_all 0.99,
  `results/2026-11-03-meaning-selection-realpath.md`). S_all in the live brain failed at 0.1B
  (`2026-11-03-sall-in-brain.md`, `2026-11-03-sall-live-bar-refit.md`), where S0 was still 11/12 right.
- New here: the same measurement at 1.5B, the live-path check, and an oracle-routing gain check.
- Prerequisites: a 1.5B cortex (met), sleep at 1.5B (met, arm c 9.33). Host decision: none new; the bar is a fitted
  scaffold constant like the existing KEY_BARS. `tools/before_you_build.sh` reported a stale local corpus, so the
  record above was read directly from the result files.

## Measurement (CPU fp32, deterministic, no sleep, no speech)
`brain/experiments/hippo_route_probe.py` stores the 12 facts exactly as the live Brain does (3 permanent self
episodes + the 12 tellings, from the birth state) and scores every cue under every key the code supports (S, att, ffn,
last, pool per layer and `_all`). Dev bank (DEV_FACTS36, seeds 42/43/44) only for bars. Artifacts:
`brain/experiments/results/2026-11-03-hippo-routing-1.5b/` (`route_1.5b.json.gz`, `table_1.5b.txt`, `check_*.json`,
`gain_1.5b.json`). Table: `python -m brain.experiments.hippo_route_table <route_1.5b.json.gz> --fit`.

Forced top-1 = right episode, of 12 (no bar):

| key | test told | test PARA | dev orig / para_a / para_b (mean of 3) | told margin mean (min) |
|---|---|---|---|---|
| S0 (default) | 7 | 5 | 10.0 / 5.7 / 5.0 | +0.057 (-0.093) |
| **S_all** | **11** | **11** | 11.7 / 11.7 / 8.7 | +0.164 (-0.008) |
| S17 (best single layer) | 12 | 11 | 12.0 / 12.0 / 10.3 | +0.217 (+0.014) |
| S8 | 11 | 10 | 11.7 / 11.7 / 10.7 | +0.229 (-0.015) |
| pool_all | 7 | 5 | 7.3 / 5.3 / 6.3 | +0.036 |
| ffn_all | 5 | 4 | 7.3 / 4.7 / 5.0 | -0.039 |
| att_all | 3 | 4 | 6.0 / 4.3 / 5.0 | -0.053 |
| last_all | 3 | 4 | 5.7 / 4.0 / 5.3 | -0.047 |
At 0.1B the same probe gives S0 12/8 and S_all 11/10 (smoke), so S0's failure is a size effect of the layer-0 key.

Live path (`hippo_route_check.py`, a real Brain and `recall()`), told questions right / wrong / none, PARA right,
never-told (16) fired:
- S0, default bar 0.12/0.02: **6 / 5 / 1**, PARA 3, never fired 11. Same 6 misroutes as the 1.5B JSONs on every seed
  (cat -> Tomasz, dog -> Marguerite, best friend -> son's birthday, work -> Volvo, brother -> Tomasz; grandmother
  blocked by the margin, gap 0.004). The 0.1B-fit bar causes 1 of the 6 losses; the key causes 5.
- S_all, 0.1B bar 0.185/0.05: 10 / 0 / 2, PARA 5, never fired 8.
- S_all, 1.5B dev-fit bar 0.22/0.01: **9 / 0 / 3**, PARA 5 / 0 wrong, never fired 5.

Oracle routing (`hippo_gain_probe.py`, right episode reinstated, unslept 1.5B, greedy): correct of 12 at gain
0/1/2/3/4 = 0/2/3/8/**10** (told), 0/3/5/7/10 (PARA). The 0.1B gain 4 is right at 1.5B; the defect is routing.

## Diagnosis
The hippocampus at 1.5B fetches the wrong memory because its key (the layer-0 fast-memory change) no longer
separates the tellings at this size (told margin +0.06, 5 of 12 negative). Every all-layer matrix-memory key
separates them (S_all margin +0.16; no wrong episode fires at either S_all bar). Residual-stream and token-shift keys
are worse than S0. The bar plays a small part (1 of 6). When S_all does not fire, nothing is reinstated and the slept
cortex answers alone, which is arm c's strong path: that is the "defer to the slept cortex when unsure" behaviour,
with no new gate.

## Fix (behind a switch, default OFF)
`Brain(key="S_all", key_bar="size")`: the existing S_all key with a bar refit for 1.5B on the DEV bank only
(`Brain.KEY_BARS_BY_SIZE[("S_all","1.5b")] = (0.22, 0.01)`, dev J 0.55). `key_bar="default"` (default) keeps every
existing bar. S_all rather than S17: S_all needs no layer chosen on 1.5B data and has a record at 0.1B and 0.4B; S17
was the top on the test bank, a selection on the evaluation set. Unit tests: `brain/tests/test_key_bar_switch.py`
(42 brain tests pass). Pipeline smoke (not scored): the probe at 1.5B on CPU, 2 sleep steps, `--max-q 1`, seed 42: arms
a-g run; a routes the cat question to Tomasz, f and g to Pixel.

## PRE-REGISTRATION (written before any run; bars fixed now)
Probe `live_brain_probe.py` unchanged for arms a-e (same seeds, settings and scorer as live15), plus `--fix-key S_all
--fix-bar size`: arm **f_fix_sleep** (as a, S_all at the 1.5B bar) and **g_fix_nosleep** (as b, same key). Model 1.5B,
`--device cuda`, `--steps 480 --lr 5e-4 --kl-weight 3`, seeds 42/43/44/100/101/102. Packet
`brain/experiments/jobs/2026-11-03-aws-hippo-route.txt` (6 jobs). Primary measure `strict_scorer.correct_answer`
(`recall_correct`). Table: `python -m brain.experiments.hippo_route_live_table <dir> --glob 'hroute_s*.json'`.
- F1 (instrument, arm g only -- unslept, what the CPU measured): g routes >= 8/12 told questions to the right episode
  and <= 1/12 to a wrong one on every seed (CPU measurement 9/0; one question of slack, since 'What car do I drive?'
  clears the bar by 0.003). Fails = the GPU routing differs; the run is INVALID for this question, not a negative.
  Arm f's routing (read through the slept cortex) is an OUTCOME, reported per seed: bad routing there shows up as
  F2/F3 failing, not as INVALID. (Amended 11-03 before any run, on review.)
- F2: mean recall f - mean a >= 2.0 and f > a on >= 5/6 seeds.
- F3: mean recall f >= mean c - 1.0 (the hippocampus no longer costs recall after sleep).
- F4: mean false recall f <= mean a + 0.5 (of 4).
- F5: mean recall g - mean b >= 2.0 and g > b on >= 5/6 seeds (the hippocampus alone works).
- Verdict: F1-F5 all hold on six seeds = LEAD for `key="S_all", key_bar="size"` at 1.5B (not a GO, no default flip
  without land-result). Any failing criterion is reported as that criterion failing.
- Reported, not criteria: paraphrase f/g vs a/b/c, echo, a-e against the live15 numbers, per-seed seconds,
  `cuda_peak_bytes` (< 24 GiB or the seed is invalid). Invalid seeds are rerun with the same command.
- Expected: f about 8-10, g about 7 (9 right fires x ~0.8 correct). F3 is the uncertain one.

## Known limits
The bar was fit on unslept reads (arm f reads cues through the slept cortex; routing was identical in a and b at
live15); one sampled reply per question; 12 test facts; F4 uses only 4 never-told questions, and S_all at this bar
fires on 5/16 never-told cues (1 of the 4 in the probe), so false recall is not fixed here (abstention is still the
missing prerequisite).
