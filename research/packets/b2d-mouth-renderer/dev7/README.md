# B2d packet: dev-seed record (seed 7, 2026-09-30, Claude)

Dev seed only: nothing here is a gate-seed result or a claim about the faculties. The files are under
`research/findings/raw/_load_bearing/_shards/b2d0929-{stubid-dev7-e8a0d0c,stubid-dev7-f5e3732,mouth-dev7}/` and
`research/findings/raw/_b2d_mood_render_probe/dev7_{stub,production}/`; `summarize_dev7.json` beside this file is the
machine-read summary (`summarize_dev7.py`), written once the two optional dev GPU checks cancelled at handoff have run
(HANDOFF section 11).

## 1. The renderer change leaves the stub path byte-identical (the prereg's acceptance check)

The three mouth rows' stub-arm cells were run at dev seed 7 from two exported git-archive trees: the change's
commit `e8a0d0cb1` and its parent on main `f5e3732fe` (`run_stub_identity.sh`, numpy, one BLAS thread, the nine
adequate LB flags, `--repeats 2`).

| row | files compared | byte-identical | lb.json verdict (both) |
|---|---|---|---|
| affect-coloring | intact_a, intact_b, lesion, lesion.rep0, lb.json | 5 of 5 | regressed (valence_sign, tone_token) |
| affect-appraisal-interoceptive | same | 5 of 5 | regressed (valence_sign) |
| affect-drives-response | same | 5 of 5 | regressed (affect_drives.reason only: an SD field, dropped by S1) |

Unit-level pins of the same claim, over every probe turn: `tests/test_arm_renderer_production_path.py`.

## 2. The mouth arm at dev seed 7 (LB_ARM_RENDERER=production, cupy, from the git-archive tree of e8a0d0cb1)

- The production renderer resolved correctly: every arm's probe turn reads `renderer` = `off-bridge Qwen-0.5B
  (spiking forward)` and `_arm_renderer.mode` = `production`; every sidecar reads `git_archive`, both manifest checks
  true, `SIM_BACKEND=cupy`, `LB_ARM_RENDERER=production`; source `tiny-demo +LTM`. `tools/b2d_mouth_score.py` runs end
  to end with all preconditions passing (fields pin set to the dev commit).
- affect-coloring on `emo`: **MOUTH-SILENT** (0 Qwen generation calls in all four arms). The reply is the
  curiosity/abstention template ("I don't know about that. My curiosity is piqued ..."), identical to the stub arm's.
  As predicted (Amendment 1, A1.3).
- affect-drives-response on `well`: the intact arms' reply ('the wolf bites the apple') is an assertion echo with 0
  Qwen calls, so this row is also MOUTH-SILENT at dev seed 7. A1.3's first prediction for this row
  (MOUTH-NOT-LOAD-BEARING) did not hold; its stated alternative did. The first attempt's lesion arm was killed by
  SIGTERM mid-build (see section 4); its re-run with `LB_RESUME_SKIP_EXISTING=1` was cancelled at handoff (optional,
  HANDOFF section 11).
- affect-appraisal-interoceptive on `emo`: same turn and same reply as affect-coloring; not run on the GPU at the dev
  seed (the six-seed arm runs it).
- Cost: about 10 min per arm on cupy (brain build, Qwen load and calibration, the turn), so about 40 min per cell;
  RSS about 8 GB per worker, 12.4 GB peak per LB job. 18 gate cells: about 12 GPU-hours.

So the registered mouth arm is expected to read MOUTH-SILENT on all three rows at the gate seeds: on these probe
turns the production Qwen mouth never speaks.

## 3. Why the Qwen mouth is silent at the console default, and the remap candidate (NEEDS-CLAUDE #3)

At the console's defaults (renderer auto, rich off) a bounded-transitive recall reply is spoken by the spiking recall
mouth (`BRAIN_SPIKING_MOUTH_RECALL`, default ON since 2026-08-26, `research/runners/brain_chat_tui.py`
`spiking_recall_surface`), which pre-empts the Qwen renderer and runs with the stub renderer too. Its surface carries
its own mood-tone marker from a spiking rate read of the affect level (`_apply_mouth_mood_tone`). The B2d stub arm
therefore already speaks through the production recall mouth; what its registered affect turns lack is a turn that is
both mood-engaged and a recall.

`research/runners/_b2d_mood_render_probe.py` (dev seed only) runs the `emo` message and then "what does the dog
chase" in the same session, through the battery's own worker, intact / intact rebuild / `BRAIN_AFFECT_LESION=1`:

| renderer mode | intact_a | intact_b (null) | lesion | reading |
|---|---|---|---|---|
| stub (numpy) | `the dog chases the cat! — worth going further here.` | same | `the dog chases the cat — worth going further here.` | reply changed; null clean; attributable 1.0 |
| production (cupy) | `the dog chases the cat!` (0 Qwen calls) | same | not run (cancelled at handoff; HANDOFF section 11) | intact arms agree |

One dev seed, one probe: a candidate for a remap amendment (a registered two-turn affect probe, checked at a second
dev seed, then six seeds), not a result. The changed field is the answer text only (the tone marker); who composed
the marker is the host-share plan's question, not this probe's.

## 4. Incidents found while running this (each logged in research/FAILURE_LOG.md)

- A refused `tools/mem_ok.sh` makes the GPU queue drop a line (three dev lines dropped in one second). The mouth job
  lines now wait up to 90 min for memory before refusing (`make_jobs.py`), and a mouth cell that never ran is
  re-queued, not re-run (`rerun_mouth_cell.sh`).
- `tests/test_game_pause_resume.py` kills live GPU jobs (it runs `gpu_queue.sh pause --now` with the real nvidia-smi)
  and SIGSTOPs real `research.runners` processes. Two dev workers died with rc=143 while other sessions ran pytest
  selections. A fix task was filed. Until it lands, a mouth cell killed this way is re-run through step 9's table.
- The GPU queue holds every job while the local-llm unit is active, and `llm.sh yield` refuses while any draft
  worktree has staged changes; this session used `llm.sh off` plus the `.local_llm_was_on` restore marker.
