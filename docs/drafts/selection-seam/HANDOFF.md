# HANDOFF: BG live-seam retirement (plan step 3), for the local model

*2026-09-29, Claude (lane A). Execute top to bottom. No step needs judgment; where one would, it is a STOP with a
NEEDS-CLAUDE item. Do not change any number, bar, arm or file listed here. At most one fix round per failing step,
and only the fix written in the step.*

Worktree: `/home/dant123/Projects/sim/.claude/worktrees/selection-seam-need-reversal`, branch
`research/selection-seam-need-reversal`. Python: `/home/dant123/Projects/sim/.venv/bin/python` (written `PY` below).
Everything below runs from the worktree root unless a step says otherwise.

## What is built (staged, not committed)

| file | region | what |
|---|---|---|
| `research/runners/bg_action_selection_production_organ.py` | `BEGIN neural-seam` .. `END neural-seam` (end of file) | flag `BRAIN_BG_NEURAL_SEAM` (default off): no pre-filter, no host salience, heard/hold -> D1 synapses, action latch commit |
| `research/runners/_gnw_norgan_bus_derisk.py` | 2 lines in `norgan_hop` + `BEGIN neural-seam read` .. `END` | flag `BRAIN_GNW_UNIQUE_IGNITION_READ` (default off): commit only on a single ignited slot, no argmax |
| `webapp/server.py` | the `faculty:bg-action-selection` block | calls `decide_action_neural` only when the flag is on |
| `research/runners/host_share_bg_seam.py` | new | live-turn arms, wrappers, `score`, registered `verdict` + `BARS` |
| `research/runners/_bg_neural_seam_derisk.py` | new | organ-level DEV de-risk |
| `tests/test_bg_neural_seam.py` | new | byte-identical-off (hashes from main 3c2907b85), dispatch, no-salience, behaviour |
| `research/runners/_need_reversal_selection_seam.py` + test | new | toy mechanism de-risk ONLY (never gate seeds) |
| `tools/lab.py::membrane_bound_check` + `tests/test_membrane_bound_check.py` | new function | membrane range check (COMMITTED 81d5cf2a9) |
| `research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md` | new, UNTRACKED | commit it ALONE (step 4) |

Other lane's region, do not edit: the long-delay-credit lane owns plasticity on the `bg_value_afferent_plasticity`
pathways (heard/hold -> D1). This branch keeps them frozen.

## Step 0: preconditions (check; STOP if any fails)

```bash
export PY=/home/dant123/Projects/sim/.venv/bin/python
cd /home/dant123/Projects/sim && $PY tools/parallel_audit.py | sed -n 2p      # must NOT read UNDER-PARALLELIZED on compute, or it is after 18:30
git -C /home/dant123/Projects/sim cat-file -e research/host-share-baselines:research/runners/host_share.py && echo LANE_C_OK
```
If `LANE_C_OK` does not print: STOP, NEEDS-CLAUDE #1.

## Step 1: tests (all must pass)

```bash
cd /home/dant123/Projects/sim/.claude/worktrees/selection-seam-need-reversal
SIM_BACKEND=numpy $PY -m pytest -q tests/test_bg_neural_seam.py tests/test_need_reversal_selection_seam.py tests/test_membrane_bound_check.py
```
Any failure: STOP, NEEDS-CLAUDE #2 (paste the failing test name and assertion).

## Step 2: commit A (code, tests, drafts, DEV raw)

```bash
git add research/runners/bg_action_selection_production_organ.py research/runners/_gnw_norgan_bus_derisk.py \
  webapp/server.py research/runners/host_share_bg_seam.py research/runners/_bg_neural_seam_derisk.py \
  research/runners/_need_reversal_selection_seam.py tests/test_bg_neural_seam.py tests/test_need_reversal_selection_seam.py \
  research/FAILURE_LOG.md research/coordination/lane_builds.jsonl \
  docs/drafts/selection-seam research/findings/raw/_need_reversal_seam research/findings/raw/_bg_neural_seam \
  research/findings/raw/_provenance/runs.jsonl
git status --short | grep -v '^[AM] ' ; git commit -F- <<'MSG'
feat(bg seam): default-off neural admission, value and latch commit for the live BG seam + unique-ignition bus read

BRAIN_BG_NEURAL_SEAM=1 removes the host pre-filter, the host salience and the first-crossing commit read from the
production SPEAK/STAY-SILENT seam (heard/hold -> D1 synapses, action latch). BRAIN_GNW_UNIQUE_IGNITION_READ=1 removes
the norgan_hop argmax. Both default off; flag-off byte identity pinned by hashes from main 3c2907b85. Live-turn
scorer research/runners/host_share_bg_seam.py (lane C instrument). DEV only; no gate seed run. Review of the local
model's N=5 BG work (an explicit-step overshoot, not a wall) and the toy need-reversal de-risk (withdrawn as a prereg).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
```
If blocked ONLY by `compute-idle-persistent` or `lane-starvation`: wait, re-run Step 0, retry once. Any other gate:
STOP, NEEDS-CLAUDE #3 (paste the gate output). Never `--no-verify`, never write a waiver.

## Step 3: bring in lane C's instrument

```bash
git merge --no-edit research/host-share-baselines
SIM_BACKEND=numpy $PY -c "import research.runners.host_share as C; print(C.classify_seam(1.0, 0.0, 1.0))"   # prints TRANSDUCED
```
Merge conflict or a different print: `git merge --abort`, STOP, NEEDS-CLAUDE #1.

## Step 4: commit B (the preregistration ALONE)

```bash
git add research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md
git commit -m "prereg(bg seam): live BG seam retirement, six gate seeds, G1-G5 + preconditions" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
bash /home/dant123/Projects/sim/tools/push_both.sh research/selection-seam-need-reversal
SHA=$(git rev-parse HEAD); echo $SHA
```

## Step 5: provision and queue the six gate seeds (pool, CPU)

```bash
cd /home/dant123/Projects/sim
bash tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
for s in 42 43 44 100 101 102; do
  bash tools/pool_queue.sh add "cd ~/derisk-pool/revisions/$SHA && bash tools/mem_ok.sh 8 2 && mkdir -p research/findings/raw/_bg_live_seam/gate && bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.host_share_bg_seam --run-all research/findings/raw/_bg_live_seam/gate --seed $s --allow-gate-seeds" \
    --checked "prereg research/findings/2026-09-29-bg-live-seam-retirement-PREREGISTRATION.md (${SHA:0:9}); BG live seam, seed $s, 7 fresh-brain arms; record: no prior live measurement of this seam; mem_gb=8"
done
bash tools/pool_queue.sh list | tail -8
```
Expected cost: 7 arms x ~25-30 min per seed (one tiny-demo ChatBrain build each), ~3.5 h per seed on one core.

## Step 6: wait for 42 result files, then score

`pool-sync.service` copies results into `research/findings/raw/_bg_live_seam/gate/s<seed>/<arm>.json`.
```bash
cd /home/dant123/Projects/sim/.claude/worktrees/selection-seam-need-reversal
ls research/findings/raw/_bg_live_seam/gate/s*/*.json | grep -vc prov      # must be 42
$PY -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/gate \
  --seeds 42 43 44 100 101 102 --out research/findings/raw/_bg_live_seam/gate/score.json
```
Fewer than 42 files after 12 h: re-queue only the missing seeds with the Step 5 line (the runner skips existing arm
files). Still missing after one re-queue: STOP, NEEDS-CLAUDE #4.

## Optional DEV check (any time; not a verdict)

A cupy DEV run of seed 7 was queued on the GPU at ~15:10 (`research/findings/raw/_bg_live_seam/dev_cupy/s7/`, 7 arm
files when done). When all 7 exist, score it and paste the printed line into WORKLOG.md; a DEV result changes nothing
in Steps 2-8:
```bash
HOST_SHARE_INSTRUMENT_DIR=/home/dant123/Projects/sim/.claude/worktrees/host-share-baselines/research/runners \
  $PY -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/dev_cupy --seeds 7
```
(After Step 3 the `HOST_SHARE_INSTRUMENT_DIR=...` prefix is not needed.)

## Step 7: outcome -> verdict (read `verdict` and `gates` in score.json; do not recompute)

| score.json | write in the finding title and board line |
|---|---|
| `verdict` GO | "BG live seam: host admission, score and commit read retired behind default-off flags (6-seed GO); learned-content fraction 0; flags still OFF" |
| `verdict` NO-GO, only G1 false | "production BG host decision is not consequential on these turns (G1); the retirement claim is not tested" |
| `verdict` NO-GO, G2a or G2b false | "neural seam does not reproduce the production behaviour (G2); retirement NOT shown" |
| `verdict` NO-GO, G3 false | "reply not caused by the neural value route (G3); retirement NOT shown" |
| `verdict` NO-GO, G4 false | "latch read is not threshold-invariant (G4); the host read still owns part of the commit" |
| `verdict` NO-GO, G5 false | "bus argmax does work on live turns (G5); the unique-ignition read changes answers" |
| `verdict` UNDEFINED | "UNDEFINED: <the failed precondition from `preconditions`>"; not a negative |

Any NO-GO or UNDEFINED: write the finding as below, then STOP, NEEDS-CLAUDE #5 (do not start a fix).

## Step 8: finding, commit, board

1. Write `research/findings/2026-09-30-bg-live-seam-retirement-RESULT.md` (frontmatter `type: finding`,
   `status: live`, `runner: research/runners/host_share_bg_seam.py`, `artifacts: [research/findings/raw/_bg_live_seam/gate/score.json]`).
   Body: the Step 7 sentence; a table copied from `score.json` `pooled` and `n`; `production_seam_class`; the six
   `gates`; the reported-only numbers (S, D, `first12_change`, `readout`); and these fixed caveats: "wired, not learned
   (learned-content fraction 0)", "flags default OFF; flipping them is an owner decision", "the organ is its own
   SimulationBridge, not the one-brain substrate".
2. Commit the raw directory and the finding together; push with `tools/push_both.sh`.
3. Board: one line in `GAP_CLOSURE_MISSION.md` CURRENT STATE with the Step 7 sentence and the finding path; Vikunja
   via `tools/vikunja.sh` as the `vikunja` skill says. Do not flip any default.

## Separate staged work to commit (same Step 0 precondition)

- Engine fix, branch `research/conductance-overshoot-guard`, worktree
  `/home/dant123/Projects/sim/.claude/worktrees/conductance-overshoot-guard`: flag `enable_bounded_conductance_integration`
  (default off), 4 tests pass on numpy (off byte-identical; overshoot -247.7 mV off vs -72.2 mV on). Commit its staged
  files with its WORKLOG's message, push with `tools/push_both.sh research/conductance-overshoot-guard`. cupy tests not
  run: queue once with `bash tools/gpu_queue.sh add 'cd <that worktree> && SIM_BACKEND=cupy $PY -m pytest -q tests/test_bounded_conductance_integration.py'`.
- Amend (do not retract) the draft finding on branch `research/draft-seam-map` (worktree
  `/home/dant123/Projects/sim/.claude/worktrees/draft-seam-map`): append the section below verbatim to
  `research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md`, Markdown-only commit, push that branch.

> ## Amendment 2026-09-29 (Claude review): the N=5 "tonic-firing wall" is an integration overshoot
> With four `commit_fs -> commit` GABA-A projections of weight 40 onto each commit pool, one explicit step drives the
> commit membrane to -377 mV; the pools then fire at the refractory ceiling with arousal off and the thalamus silent.
> Scaling the cross-inhibition to 40/(N-1) = 10 removes it at the default tonic (commit 0.83-1.0, dev seed 7). The
> GPi 500 / thal 45 point avoids the overshoot only because it silences the thalamus; the gpi 50 / thal 45 "window"
> is the same -386 mV overshoot. The six-seed run (`_gpi_fix_6seed.json`, dirty tree, no prereg) is DEV. "0% of the
> selection is host-owned" does not follow: the candidate scores are host-computed. Review:
> `docs/drafts/selection-seam/REVIEW-of-local-model-nchannel-bg.md` on `research/selection-seam-need-reversal`.

## Stop conditions (any one: stop and write the NEEDS-CLAUDE item into WORKLOG.md)

1. Any test in Step 1 fails. 2. A gate other than compute-idle-persistent / lane-starvation blocks a commit.
3. The merge in Step 3 conflicts. 4. Results missing after one re-queue. 5. `verdict` is not GO.
6. Anything asks you to change a bar, an arm, a constant in the `neural-seam` region, or a default flag.

## NEEDS-CLAUDE

1. Lane C's `research/runners/host_share.py` is not committed on `research/host-share-baselines`, or merging it conflicts.
2. A Step 1 test fails.
3. A real gate blocks commit A or B.
4. Gate-seed results missing after one re-queue.
5. The verdict is NO-GO or UNDEFINED (a new method is Claude's call; no-defer applies).
6. Known open items for Claude, not for the local model: (a) the learned heard/hold -> D1 mapping (owned by the
   long-delay-credit lane); (b) arousal is not needed for a content SPEAK commit in the neural seam (DEV organ, 12 seeds: 0.10-0.75
   commit without arousal on content), so the Gate-A arousal anti-cheat is weaker there; (c) the organ is not on the
   one-brain substrate; (d) flipping either flag on by default; (e) the silent GPe/STN in Gate-A v2 (a tonic-GPe
   variant); (f) run the engine flag against this seam (latch pools reached +68..+145 mV at motor->latch 80 before the
   weight was lowered to 40).
