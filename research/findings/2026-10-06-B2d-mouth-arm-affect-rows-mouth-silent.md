---
type: finding
status: live
date: 2026-10-06
lane: instrument
seeds: [42, 43, 44, 100, 101, 102]
prereg:
  - research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_load_bearing/_shards/b2d0929-mouth/b2d_mouth_score.json
  - research/findings/raw/_load_bearing/_shards/b2d0929-mouth/PIN.txt
  - research/packets/b2d-mouth-renderer/dev7/README.md
---

# B2d mouth arm: the three affect rows could not be measured through the Qwen mouth (2026-10-06)

Registered arm `b2d0929-mouth` of the B2d pre-registration (Amendment 1, A1.2), at P = `5818dc629c5c0cd3cbb370c0c6cc16b314805b03`,
scored by the registered `tools/b2d_mouth_score.py` (unchanged since the pin; the 30 cited blobs match P), six seeds
(42/43/44/100/101/102), Qwen2.5-0.5B (`QwenRenderer`) on cupy, `LB_ARM_RENDERER=production`. Every number below is read from `research/findings/raw/_load_bearing/_shards/b2d0929-mouth/b2d_mouth_score.json` and the arm
files beside it. Reported separately from the
stub arm and never averaged into B2d's S0/S1/S2. Reviewed by Claude 2026-10-06 (cell finals re-derived from the raw arm
files and reproduced by a re-run of the scorer).

## 1. Result (as scored, then what it does and does not say)

| row (probe turn) | cells | MOUTH-SILENT | defined | defined and reply-load-bearing | reading |
|---|---|---|---|---|---|
| affect-coloring (`emo`) | 6 | 6 | 0 | n/a | MOUTH-SILENT |
| affect-appraisal-interoceptive (`emo`) | 6 | 6 | 0 | n/a | MOUTH-SILENT |
| affect-drives-response (`well`) | 6 | 5 | 1 (seed 100) | 0 of 1 | MIXED |

Battery verdict COMPLETE: 17 of 18 cells MOUTH-SILENT, 1 DEFINED, 0 FAIL, no cell needing a re-run, no precondition failed.
Nothing was re-run, so there is no attempt-1 directory.

**This is not a negative result about the affect faculties.** MOUTH-SILENT means the Qwen generation entry points ran 0
times on the probe turn in all four arms: the reply on that turn never came from the mouth, so this arm cannot measure
the row on this turn. The registered scorer says the same ("named, never counted as 0 and never as a pass"), so the 17
silent cells are UNDEFINED for the question "does the affect faculty change what the mouth writes". The one defined cell
reads 0 of 1, which is not a rate. No row is shown reply-load-bearing through the mouth, and no row is shown NOT
load-bearing through it.

## 2. Silent is not the same as the instrument failing

The instrument read above zero where it should, and read zero where the reply was not a Qwen reply:

- Every arm's probe turn resolved to the Qwen renderer (`off-bridge Qwen-0.5B (spiking forward)`, mode `production`) on
  all 72 arm files, every sidecar reads `git_archive`, both manifest checks true, `SIM_BACKEND=cupy`, no BRAIN_ key in
  `lb.json`; a cell where the handler had resolved the stub would be class M, and there are none.
- The Qwen call counter reads 1 in all four arms of the one defined cell (seed 100, `well`, reply `Wolf bites apple.`),
  inside this same arm, and a real-path positive control at dev seed 7 counted 1 call (`dev7/README.md` section 5).
- The replies explain the zeros. On `emo` the reply in every arm is the curiosity/abstention template ("I don't know
  about that. My curiosity is piqued ...", `abstained` true). On `well` five of six seeds echo the stored assertion
  ("the wolf bites the apple") with no renderer call.

Predictions (A1.3, written before any mouth output): affect-coloring and affect-appraisal-interoceptive MOUTH-SILENT on 6 of 6:
matched. affect-drives-response MOUTH-NOT-LOAD-BEARING 0 of 6: missed (the row's stated alternative, MOUTH-SILENT if `well`
does not pass through the renderer, held on 5 of 6 seeds; seed 100 rendered through Qwen and is unexplained).

## 3. What the arm files do show about the affect path

- On `emo` the affect faculty is engaged (appraisal hits on the affect words in every arm): in 4 of 6 seeds (43, 44, 100, 101)
  the intact arm computes a tone level 2 and a manner instruction ("warm, friendly sentence") that the lesion arm does
  not; in seeds 42 and 102 the intact arm computes tone level 0 and no instruction. Where the instruction exists, the
  abstention template drops it: nothing downstream consumed it on this turn, so the reply is identical in all arms.
- On `well` the affect channel is neutral by construction (tone level 0 and an empty manner template in both arms, all
  six seeds). The single defined cell therefore tests a turn on which the affect lesion cannot be expected to change the reply.
- A turn that is both mood-engaged and rendered does not exist among the registered turns (A1.3's own conclusion: a
  property of the probe set). The dev-seed-7 mood-render probe (`emo` then "what does the dog chase") changed the reply
  under the affect lesion in stub mode (tone marker only; one seed, not a result), but the production-mode arm of that probe
  made 0 Qwen calls too: at the console defaults the spiking recall mouth (`BRAIN_SPIKING_MOUTH_RECALL`, default on)
  speaks bounded recalls before the Qwen renderer is reached, and its own mood-tone marker is the affect path there.

## 4. Caveats the reading carries

1. **The affect organ ran on its fallback.** The workstation export at P was built without `research/findings/raw/`
   (checked: `_affect_learned_valence_map.json` is absent there), the omission behind the 2026-10-04 Stage-1 C4 void
   (FAILURE_LOG 2026-10-04 and 2026-10-06). The silent readings do not depend on the valence source, and `well` carries no
   affect words, so no reading here changes; but the affect organ measured is the raw-norm variant, and a mood-engaged
   rendered probe must run from an export that ships the map (`export_local_revision.sh` now does).
2. **One cell is a mixed-attempt cell**: `s42/affect-coloring` (FAILURE_LOG 2026-10-06). intact_a, intact_b and lesion are
   from the registered re-run (10-04 23:22-23:39); `lb.json` and lesion.rep0 survive from the unregistered 10-01 run. The
   re-run was killed (rc=143) at 23:47. Every arm file that exists reads 0 Qwen calls and the same template reply, so the
   cell reads MOUTH-SILENT either way, but it is not a clean registered cell; a clean re-run is requested. Were that cell
   treated as not final, the row would read PENDING and the battery INCOMPLETE until it lands.
3. The ruling to copy the unregistered 10-01 cells aside before the re-run (`b2d-1001-cells`) was not carried out (the
   directory does not exist), so no 10-01 versus re-run byte-equality determinism check exists for the other 17 cells.
4. The scorer's stub column is partial (stub-arm cells had landed for 8 of 18 cells at scoring time, all reading S2
   `pass`: reply unchanged); the stub shards are not on this branch, and the stub arm (258 cells) is not complete. This finding
   is the mouth arm only; the B2d battery verdict waits for the stub arm.
5. Known counter limit (scorer docstring): the open-ended generator (`BRAIN_OPEN_ENDED`, default off) calls `model.generate`
   directly, so a reply it wrote would read MOUTH-SILENT. It was off.

## 5. What it means for the roadmap

The three affect rows stay TRACE-ONLY-INSTRUMENT-LIMIT in the reply-level rescore: unmeasured at the reply, not refuted.
The load-bearing fraction does not move. The fact the arm exposes is about where the Qwen mouth speaks: on the registered
affect turns (an unknown-word abstention and an assertion echo) the reply is host-template text that no affect lesion
reaches, and on bounded recalls the spiking recall mouth, not Qwen, speaks. To show an affect faculty reply-load-bearing
through Qwen, a probe needs a turn the Qwen renderer actually speaks and on which the affect state is engaged. The
next step is a remap amendment (a registered two-turn probe, dev seeds first, from an export that ships the valence map),
not more seeds of this arm.

## 6. Honesty

These are replay-level lesion readings of a functional channel. They say nothing about experience; the self-reports of
the system on these turns are read-outs of monitors, not claims of feeling.
