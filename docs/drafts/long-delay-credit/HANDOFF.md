# HANDOFF: long-delay temporal credit on the production BG organ (plan step 4, lane B)

Execution packet for the local model. Follow the steps in order, run the commands exactly as written, and STOP at every
stop condition. Nothing here needs judgment; anything that does is on the NEEDS-CLAUDE list at the end.

- Worktree: `/home/dant123/Projects/sim/.claude/worktrees/long-delay-credit-spiking` (branch
  `research/long-delay-credit-spiking`, cut from main `f52c6d246`). Run every command from the worktree unless stated.
- Preregistration: `research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md` (frozen bars, section 6;
  what may be tuned, section 7). History and the audit response: `docs/drafts/long-delay-credit/SPIKING-WORKLOG.md`.
- Code: `research/runners/bg_corticostriatal_credit.py` (the default-off `BRAIN_BG_CREDIT` extension),
  `research/runners/bg_action_selection_production_organ.py` (two new functions at the end of the file, nothing else
  changed), `research/runners/long_delay_credit_bg.py` (the runner), `research/runners/long_delay_credit_bg_verdict.py`
  (the REGISTERED SCORER for the gate), `research/runners/long_delay_credit_bg_instrument.py` (the mechanical
  section-7 instrument scorer). Tests: `tests/test_long_delay_credit_bg.py`, `tests/test_bg_credit_off_identical.py`.
- Rate-level reference (the local model's overnight draft, reviewed in the WORKLOG): `research/runners/_draft_long_delay_credit.py`,
  `tests/test_draft_long_delay_credit.py`, `docs/drafts/long-delay-credit/{PREREG-DRAFT,WORKLOG}.md`,
  `research/findings/raw/_draft_long_delay_credit_dev*.json` (+ `.prov.json`).

Shell: the multi-line loops below are bash (`declare -A`, arrays). Run them in bash (`bash -c '...'` from zsh).

## Step 0: preconditions (every session)

```bash
cd /home/dant123/Projects/sim && .venv/bin/python tools/parallel_audit.py | head -3
```
If it prints `UNDER-PARALLELIZED` the commit gates `compute_idle_persistent` / `lane_starvation` will block non-exempt
commits. Do not retry in a loop; do Step 5/6 work that does not commit, or wait (the block lifts about 18:30 on
2026-09-29 per the coordinator). Never use `--no-verify`, never write a waiver.

## Step 1: tests (must pass before any commit)

```bash
cd /home/dant123/Projects/sim/.claude/worktrees/long-delay-credit-spiking
bash /home/dant123/Projects/sim/tools/mem_ok.sh 2 && OMP_NUM_THREADS=1 SIM_BACKEND=numpy bash /home/dant123/Projects/sim/tools/memcap.sh 2 -- \
  /home/dant123/Projects/sim/.venv/bin/python -m pytest -q tests/test_long_delay_credit_bg.py tests/test_bg_credit_off_identical.py tests/test_draft_long_delay_credit.py -p no:cacheprovider
```
Expected: 14 passed. Any failure: STOP -> NEEDS-CLAUDE item 1.

## Step 2: commits, in this order (pathspec commits; never `git add -A`)

C1, the rate-level reference plus the lane registration:
```bash
git add research/coordination/lane_builds.jsonl research/runners/_draft_long_delay_credit.py tests/test_draft_long_delay_credit.py \
  docs/drafts/long-delay-credit/PREREG-DRAFT.md docs/drafts/long-delay-credit/WORKLOG.md \
  research/findings/raw/_draft_long_delay_credit_dev.json research/findings/raw/_draft_long_delay_credit_dev.json.prov.json \
  research/findings/raw/_draft_long_delay_credit_dev_sweep.json research/findings/raw/_draft_long_delay_credit_dev_sweep.json.prov.json
git commit -m "draft(long-delay credit): the local model's rate-level draft, kept as the reference for expected signatures

Built overnight by the local model (27B) on research/draft-long-delay-credit; reviewed in
docs/drafts/long-delay-credit/SPIKING-WORKLOG.md (its trace-ablation dissociation is an algebraic identity
of the feature design). Owner ruling: the capability claim needs the spiking substrate.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
C2, the preregistration ALONE (it must precede every run it governs):
```bash
git add research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md
git commit -m "prereg(long-delay credit): delayed credit on the production BG organ, frozen bars before any build run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
C3, the build:
```bash
git add research/runners/bg_corticostriatal_credit.py research/runners/bg_action_selection_production_organ.py \
  research/runners/long_delay_credit_bg.py research/runners/long_delay_credit_bg_verdict.py \
  research/runners/long_delay_credit_bg_instrument.py tests/test_long_delay_credit_bg.py tests/test_bg_credit_off_identical.py \
  docs/drafts/long-delay-credit/SPIKING-WORKLOG.md docs/drafts/long-delay-credit/HANDOFF.md
git commit -m "feat(bg credit): BRAIN_BG_CREDIT (default off) corticostriatal three-factor credit on the production selector + the benchmark runner, scorers and tests

Off path byte-identical to f52c6d246 (asserted in data by tests/test_bg_credit_off_identical.py).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
bash /home/dant123/Projects/sim/tools/push_both.sh research/long-delay-credit-spiking
```
If a commit is blocked ONLY by `compute_idle_persistent` or `lane_starvation`: leave it staged, write one line in the
WORKLOG, and retry after Step 0 clears. Any other gate message: STOP -> NEEDS-CLAUDE item 2. Do NOT stage
`research/findings/raw/_provenance/runs.jsonl` (it holds this worktree's discarded dev runs).

C4 (whenever the dev verification in `research/findings/raw/_long_delay_credit_bg/dev/` is finished): commit its
`*.json` and `*.log` alone with message `data(long-delay credit): dev seed 7 positive-control verification (not evidence)`.

## Step 3: provision the pool at the committed revision

```bash
SHA=$(git rev-parse HEAD); echo $SHA
bash /home/dant123/Projects/sim/tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
```
If it refuses because the revision does not contain origin/main: `git fetch origin && git merge origin/main` (a
two-parent merge), re-run Step 1, `git commit` (the merge), push (Step 2 last line), provision again. A merge conflict:
STOP -> NEEDS-CLAUDE item 3.

## Step 4: instrument sweep (prereg section 7; DEV seeds 7-15 only; positive controls only)

One job per (tag, schedule, seed). Tags in priority order (the dev verification below showed no critic dip at the
defaults, so the critic tags come first): `base` (none), `strio60` (`--w-outc-strio 60`), `snc10` (`--w-strio-snc 10`),
`strio60snc10` (both), `ctx40` (`--w-ctx-msn 40`), `ctx10` (`--w-ctx-msn 10`), `lr10` (`--lr 10`), `lr2` (`--lr 2`),
`mem20` (`--w-mem-msn 20`).
```bash
SHA=$(git rev-parse HEAD)
declare -A FL=([base]="" [strio60]="--w-outc-strio 60" [snc10]="--w-strio-snc 10" [strio60snc10]="--w-outc-strio 60 --w-strio-snc 10" [ctx40]="--w-ctx-msn 40" [ctx10]="--w-ctx-msn 10" [lr10]="--lr 10" [lr2]="--lr 2" [mem20]="--w-mem-msn 20")
for tag in base strio60 snc10 strio60snc10 ctx40 ctx10 lr10 lr2 mem20; do for sch in credit_immediate late_immediate sanity_early_ch0; do for s in 7 8 9 10 11 12 13 14 15; do
bash /home/dant123/Projects/sim/tools/pool_queue.sh add "cd ~/derisk-pool/revisions/$SHA && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.long_delay_credit_bg --schedule $sch --arms full --seeds $s --n-train 200 --n-test 100 ${FL[$tag]} --out research/findings/raw/_long_delay_credit_bg/sweep/${tag}_${sch}_s$s.json  #checked:lane B long-delay credit instrument sweep (prereg 2026-09-29 section 7), dev seed $s, positive-control schedule only; mem_gb=2"
done; done; done
bash /home/dant123/Projects/sim/tools/pool_queue.sh list | tail -3
```
When all 243 files are back (pool-sync pulls them into `/home/dant123/Projects/sim/research/findings/raw/_long_delay_credit_bg/sweep/`):
```bash
cp -r /home/dant123/Projects/sim/research/findings/raw/_long_delay_credit_bg/sweep research/findings/raw/_long_delay_credit_bg/
/home/dant123/Projects/sim/.venv/bin/python -m research.runners.long_delay_credit_bg_instrument \
  --tags base,strio60,snc10,strio60snc10,ctx40,ctx10,lr10,lr2,mem20 --dir research/findings/raw/_long_delay_credit_bg/sweep \
  --out research/findings/raw/_long_delay_credit_bg/sweep/instrument_report.json; echo rc=$?
```
rc=2 (no tag passes every criterion): STOP -> NEEDS-CLAUDE item 4. rc=0: the printed `"chosen"` tag is the
configuration. A job that crashes twice: STOP -> NEEDS-CLAUDE item 5.

## Step 5: Amendment A (freeze the constants; nothing else may change)

1. If `chosen` is not `base`, edit the default(s) in `CreditParams` (`research/runners/bg_corticostriatal_credit.py`)
   that the tag names (`w_outc_strio`, `w_strio_snc`, `w_ctx_msn`, `lr` or `w_mem_msn`) to the tag's value(s). Re-run Step 1. Commit that file alone:
   `fix(bg credit): freeze <param>=<value> from the section-7 instrument sweep`.
2. `SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python -c "import research.runners.long_delay_credit_bg as B; print(B.RUNNER_SHA256)"`
3. Append to the prereg, then commit the prereg alone (`prereg(long-delay credit): Amendment A`), push:
```
## AMENDMENT A (<date>): constants frozen before any gate seed
- chosen instrument configuration: <tag> (<param>=<value> or none), selected mechanically by
  research/runners/long_delay_credit_bg_instrument.py (report: research/findings/raw/_long_delay_credit_bg/sweep/instrument_report.json).
- runner SHA-256 (runner + extension module): <sha256>. Bars in section 6: unchanged.
- seen before registration: only the section-7 positive controls on dev seeds 7-15. No prediction arm was run.
```
Commit the sweep JSONs + the report in a separate `data(...)` commit AFTER the amendment commit.

## Step 6: gate runs (seeds 42 43 44 100 101 102), after Amendment A is committed and provisioned (Step 3 again)

```bash
SHA=$(git rev-parse HEAD)
JOBS=(
 "credit full,shuffled_yoked" "credit trace_ablated_integral" "credit trace_ablated_peak" "credit misleading_final_cue"
 "credit memory_lesioned" "credit critic_lesioned" "credit reward_to_da_lesioned"
 "credit dose_m0.5_tau300.0" "credit dose_m2.0_tau300.0" "credit dose_m8.0_tau300.0"
 "credit dose_m0.5_tau1000.0" "credit dose_m2.0_tau1000.0" "credit dose_m8.0_tau1000.0"
 "credit dose_m0.5_tau3000.0" "credit dose_m2.0_tau3000.0" "credit dose_m8.0_tau3000.0"
 "credit_immediate full" "credit_immediate trace_ablated_integral"
 "late_delayed full,shuffled_yoked" "late_delayed trace_ablated_integral" "late_delayed misleading_final_cue" "late_delayed memory_lesioned"
 "late_immediate full" "late_immediate trace_ablated_integral" "late_immediate memory_lesioned")
for s in 42 43 44 100 101 102; do for j in "${JOBS[@]}"; do set -- $j; tag=${2//,/+}
bash /home/dant123/Projects/sim/tools/pool_queue.sh add "cd ~/derisk-pool/revisions/$SHA && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.long_delay_credit_bg --schedule $1 --arms $2 --seeds $s --out research/findings/raw/_long_delay_credit_bg/gate/${1}_${tag}_s$s.json  #checked:lane B long-delay credit GATE run, preregistered 2026-09-29 + Amendment A, seed $s; mem_gb=2"
done; done
```
150 jobs. Each takes roughly 40-90 min on one pool core (400 train + 200 test + 40 probe trials).

## Step 7: score with the registered scorer

```bash
cp -r /home/dant123/Projects/sim/research/findings/raw/_long_delay_credit_bg/gate research/findings/raw/_long_delay_credit_bg/
/home/dant123/Projects/sim/.venv/bin/python -m research.runners.long_delay_credit_bg_verdict \
  research/findings/raw/_long_delay_credit_bg/gate/*.json --runner-sha256 <SHA-256 from Amendment A> \
  --out research/findings/raw/_long_delay_credit_bg/verdict.json
```
A `runner hash mismatch` exit: STOP -> NEEDS-CLAUDE item 6. Commit the gate JSONs + verdict.json in one `data(...)`
commit (the prereg is already committed, so `prereg_before_run` passes).

## Outcome -> verdict table (write exactly this; the finding goes to NEEDS-CLAUDE item 7 for the prose)

| scorer `verdict` | what it means | board / next action |
|---|---|---|
| GO | G1-G9 pass: the selector credits a 1-4 s delayed reward to the early choice, measured on the early event, with reach bought by a slower tag at matched magnitude, and a held cue drives a terminal choice whose delayed reward is credited | NEEDS-CLAUDE 7 (finding + verify-go); then the rung in prereg section 8 |
| NO-GO | at least one evidence gate measured and failed | record the failing gates from verdict.json in the WORKLOG; NEEDS-CLAUDE 8 (the next method, THE LAW) |
| UNDEFINED | a gate could not reach 5 defined seeds (silent full arm, weight at a bound, memory hold < 0.9) | NEEDS-CLAUDE 9 (instrument failure, never a pass) |

Also copy `mechanisms` (NECESSARY / CONTRIBUTING / NOT NECESSARY per mechanism) and `signatures` S1-S4 from
verdict.json into the WORKLOG verbatim.

## Stop conditions (stop, write one WORKLOG line, hand to Claude)

Test failure; a commit blocked by any gate other than compute_idle_persistent / lane_starvation; a merge conflict;
instrument scorer rc=2; a pool job crashing twice; a runner-hash mismatch; any scorer verdict (GO, NO-GO, UNDEFINED:
the write-up and the next step are Claude's); any wish to change a bar, a schedule constant, an arm, or a seed list.

## NEEDS-CLAUDE

1. A test fails.
2. A non-compute gate blocks a commit.
3. `git merge origin/main` conflicts.
4. No instrument configuration passes section 7 (the substrate's operating point, or the eligibility mode, needs a new
   method; Gate-B history says credit on this selector is hard even at short gaps).
5. A pool job crashes twice (read the `.out` on the node).
6. Runner hash mismatch at scoring.
7. GO: write the finding (TERMS.md check, verify-go skill), sync the board and summary docs.
8. NO-GO: name the next method (candidates: a held-choice/commitment trace to the reward, tag-and-capture, replay,
   dendritic plateau eligibility) and preregister it.
9. UNDEFINED: diagnose the instrument.
10. Merging to main, flipping any default, or wiring the extension into the production chat path.

## Dev verification already run (DEV seeds 7-9, not evidence)

`research/findings/raw/_long_delay_credit_bg/dev/inst1_{credit_immediate,late_immediate}_full_s7.{json,log}`
(positive-control schedules only, dev seed 7, 200 training trials, default constants). It does not replace the Step 4
sweep; Amendment A may only use the Step 4 report. Reading: instrument items pass (commit rates 0.90-1.00, exploration
0.43-0.67, memory hold/clear/decoy resistance 1.0/0 Hz/1.0, tonic DA offset 2e-5); the critic makes NO dip (unrewarded
DA min 0.498 vs the 0.47 bar), and both positive controls FAIL at the defaults: credit_immediate success 0.51 with D1
weights driven up and D2 weights to the floor on both channels (unsigned drift), late_immediate terminal success 0.00
with full responding (the terminal commit goes to the wrong channel: the anti-mapping the standalone circuit also
showed). If the Step 4 sweep finds no passing tag, NEEDS-CLAUDE item 4 applies, with this reading as the starting point.
