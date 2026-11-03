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
