# Gate misfire at 1.5B: does the "not known" gate fire on told facts? (2026-11-03) -- PRE-REGISTRATION (committed before any run)

Why: the told-gate test (`2026-11-03-told-gate-1.5b.md`) showed that when the "not known" gate fires, the lived
"I don't know" lesson overrides a fact the brain had just recalled. The lesson is therefore safe only if the gate
rarely fires on told questions. That gate's misfire rate at 1.5B under the fitted bars was never measured (reviewer).

## Record brief
- Known: the gate is `Brain.knows`: known iff hippocampal familiarity (best strength-weighted centred-cosine score over
  stored episodes, `Hippocampus.familiarity`) >= bar. Bars on record, none fit as a gate at 1.5B:
  S0 `KNOW_BAR_HIPPOCAMPUS = 0.22` (fok rule, fit at 0.1B on the 36-question set); S_all F bar 0.104 (abstain v2,
  `results/2026-11-03-abstain-v2/bars.json`, leave-one-seed-out on slept 0.1B DEV); S_all 1.5B selection bar 0.22
  (`KEY_BARS_BY_SIZE`, fit for RECALL on the hippo-routing DEV bank, `brain/research/2026-11-03-hippo-routing-1.5b.md`).
- Known (hippo routing, 1.5B, unslept, live recall with margin): S0 fetches the right episode for 6/12 told; S_all at
  0.22/0.01 for 9/12 (3 none), and fires on 5/16 never-told. Never measured: the familiarity-only gate rates.
- Prerequisites: 1.5B cortex (met); sleep at 1.5B (met, live-brain 1.5B). The gate is a fitted host bar (declared
  scaffold, SCAFFOLDS.md feeling-of-knowing rows); no new host decision. `tools/before_you_build.sh` reported a stale
  local corpus, so the record above was read from the result files directly.
- Expected: S0 fails (its told margins are small at 1.5B); S_all at the 0.1B bar 0.104 keeps told facts but catches
  too few never-told; S_all at 0.22 is borderline. After sleep the cue is read through the slept cortex while the stored
  keys were read unslept, so familiarity may shift (direction unknown).

## Setting (fixed now)
- Script `brain/experiments/gate_misfire_probe.py` (measurement only; no Brain default changes). Per seed: fresh home,
  `Brain(cortex=1.5B, sleep_seed=seed, sleep_on_restart=False)`, day 1 = the 12 `facts.FACTS` tellings through `hear`
  (as live_brain_probe / confab_m2 day 1; no lesson episode stored: the F gate excludes it anyway). Then the gate signal
  (familiarity, no speech, nothing stored) for every question under both keys on the SAME episodes: S0 (the brain's
  own hippocampus) and S_all (a `KeyedHippocampus` sharing the episodes, as `abstain_v2.keyed`; identical to
  `Brain(key="S_all", key_bar="size")` familiarity). Conditions: `unslept`, then `slept` after one `Brain.sleep()`
  (1.5B SLEEP_DEFAULTS: 480 steps, lr 5e-4, KL 3, rank 8). Unslept signals do not depend on the seed (storage and
  familiarity are deterministic reads); the six seeds matter for the slept condition.
- Model "1.5b" (`rwkv7-g1l-1.5b-20261031-ctx34304.pth`), `--device cuda`, W 3090 via `tools/gpu_queue.sh`. Seeds
  42/43/44/100/101/102. Outputs `_scratch/brainv2/gate-misfire/gm_s<seed>.json`, copied to
  `2026-11-03-gate-misfire-1.5b/`; `report` writes `report.json`.
- Question sets: told = the 12 told questions (`FACTS`); para = their 12 PARA rewordings (`facts.PARA`);
  never = DEV never-told (36, `confab_m2.DEV_NEVER`) + v3 never-told (24, `confab_m2.FRESH_NEVER3`) = 60.
  DEV split for the sweep: dev_told = `abstain_in_brain.FRESH_TOLD` (12 other rewordings of the told questions) and
  dev_never = DEV_NEVER. Test split for the sweep: told, para, v3 never.
- Gate fires on a question iff familiarity < bar. Misfire = fires on a told question. Catch = fires on a never-told one.

## Pre-registered bars (fixed now), three (key, gate bar) pairs, each in unslept and slept
- S0 @ 0.22 (Brain default gate bar), S_all @ 0.104 (abstain v2 F bar, per seed from bars.json), S_all @ 0.22
  (the 1.5B size bar, i.e. the S_all/size fix used as a gate).
- "Gate usable for the lesson" iff told misfire <= 1/12 as the mean over the six seeds (12 told questions) AND
  never-told catch >= 80% (of 60) on >= 5/6 seeds. Else "not usable". PARA misfire is reported against the same 1/12
  (secondary, not in the verdict).
- Sweep (a measurement, not a choice of default): per key and condition, on the DEV split pooled over seeds, the
  lowest bar whose dev_never catch >= 80%; reported: its dev_told misfire (the best achievable told misfire at >= 80%
  catch), and, at that dev-chosen bar, the held-out told / PARA misfire and v3 catch. Nothing is chosen on the test
  split. Also reported: told-vs-never AUROC per key and condition.
- Invalid (not a negative): a crashed or OOM seed is rerun unchanged.

## Known limits stated now
12 told facts only; familiarity read minutes after storage (no day of decay); one sleep; dev_told are rewordings of
the same 12 facts (no separate told-fact bank), so the sweep's dev and test told items share facts; the gate here is
familiarity-only (no margin, no recollection test).

## Result (2026-11-03)
Run: W GPU (RTX 3090) via gpu_queue, six seeds, about 6 min per seed (sleep 253-269 s), `cuda_peak_bytes` 10.6 GiB.
CPU smoke first (0.1B, seed 42, 3 questions per set): ran end to end. Files: `2026-11-03-gate-misfire-1.5b/`
(`gm_s<seed>.json` = every familiarity value; `report.json` from `gate_misfire_probe report`). Numbers read from `report.json`.

Pre-registered bars. Per seed: told misfire (of 12) / PARA misfire (of 12) / never-told catch (of 60).
Unslept values are identical on all six seeds, as stated in the setting (deterministic reads).

| key @ bar | cond | told misfire mean | PARA misfire mean | catch mean | seeds catch >= 80% | per seed 42/43/44/100/101/102 | verdict |
|---|---|---|---|---|---|---|---|
| S0 @ 0.22 | unslept | 4/12 (33%) | 5/12 | 55% | 0/6 | 4/5/33 on every seed | not usable |
| S0 @ 0.22 | slept | 4/12 (33%) | 5/12 | 55% | 0/6 | 4/5/33 on every seed | not usable |
| S_all @ 0.104 | unslept | 2/12 (17%) | 2/12 | 18% | 0/6 | 2/2/11 on every seed | not usable |
| S_all @ 0.104 | slept | 1.83/12 (15%) | 2.5/12 | 33% | 0/6 | 2/3/20, 2/3/18, 1/0/25, 2/2/19, 2/4/19, 2/3/16 | not usable |
| S_all @ 0.22 (size fix) | unslept | 3/12 (25%) | 7/12 | 92% | 6/6 | 3/7/55 on every seed | not usable |
| S_all @ 0.22 (size fix) | slept | 5.67/12 (47%) | 8.5/12 | 93% | 6/6 | 5/8/55, 6/9/54, 6/8/57, 6/9/56, 6/9/57, 5/8/56 | not usable |

- Verdict (pre-registered): the gate is NOT usable for the lesson under any key, bar or condition. The misfire bar
  (<= 1/12) fails everywhere; the catch bar (>= 80% on >= 5/6 seeds) holds only for S_all @ 0.22.
- Sweep on the DEV split (measurement; lowest bar with dev never-told catch >= 80%, pooled seeds):

| key | cond | dev bar | dev told misfire | held-out told / PARA misfire | held-out v3 catch |
|---|---|---|---|---|---|
| S0 | unslept | 0.248 | 33% | 33% / 67% | 83% |
| S0 | slept | 0.248 | 33% | 33% / 67% | 83% |
| S_all | unslept | 0.190 | 25% | 17% (2/12) / 50% | 79% |
| S_all | slept | 0.168 | 26% | 29% / 49% | 76% |

  The best told misfire at >= 80% catch is 2-4 of 12, not <= 1. The other way round: the bar that misfires on at most
  1 of the 12 told questions catches 30% (S0), 13% (S_all unslept), 12-47% (S_all slept) of never-told questions.
- Separation (AUROC told vs never-told): S0 0.73 (PARA 0.64), S_all unslept 0.83 (PARA 0.68), slept 0.78 (0.66).
  Forced top-1 right episode, told, pooled of 72: S0 42, S_all 66 unslept / 62 slept.
- Which told questions misfire (S_all @ 0.22, unslept): dog (0.196, right episode), best friend (0.098, right
  episode), work (0.082, wrong episode). Two of three are cases where the hippocampus would fetch the right memory:
  the gate would suppress a correct recall.
- Sleep: S_all familiarity of told cues drops after one sleep (misfire 3 -> 5-6 at 0.22) while never-told stays
  about the same: the cue is read through the slept cortex, the stored keys are from day 1. S0 does not move at all
  with sleep (layer-0 change not altered by the adapters; observed, not explained here).
- Meaning: familiarity alone does not separate told from never-told questions well enough at 1.5B (AUROC 0.73-0.83)
  for a gate whose firing overrides recall. Any bar that catches most never-told questions fires on 2-6 of the 12 told
  questions, and the lived lesson then replaces those answers with "I don't know". The lesson must not be wired to
  this gate. Next method (not a gate retune): arbitration in which a confident recall wins over the lesson (the gate
  only reinstates the lesson when no episode passes the selection bar and margin), or a recollection test that uses
  the recalled content; re-measure told DK with that, not familiarity alone.
- Limits: as stated; 12 told facts (a misfire count step is 1/12); dev_told shares facts with the test told items;
  keys stored unslept and read minutes later (no decay); familiarity-only gate (no margin, no recollection).
