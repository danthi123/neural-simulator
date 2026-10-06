# HANDOFF: word-role-commit (spiking commit, application A1; execution packet for Hermes)

Written 2026-10-06 by Claude. Execute top to bottom; use no judgment beyond the tables. Plan: `docs/drafts/spiking-commit/PLAN.md`.
Prereg (frozen bars, battery, selection rule, fixed sentences): `research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md`.
Organ: `research/runners/word_role_parser_organ.py` (COMMIT_GRID) + battery/CLI `research/runners/word_role_commit_organ.py`.
Wiring: `research/runners/brain_chat_tui.py` `_neural_question_parse` (flags `BRAIN_WORD_ROLE_COMMIT`, `BRAIN_WORD_ROLE_COMMIT_OP`,
`BRAIN_WORD_ROLE_COMMIT_LESION`; default OFF). LB rows: `research/runners/lbf_rows/word_role_commit.py`. Scorer: `tools/s2p_score.py`
packet `word-role-commit`. Job lines: `docs/drafts/s2p/make_jobs.sh word-role-commit ...`.

Rules: you never edit the organ, the wiring, the rows, the scorer, the prereg, the bar-sanity specs/records or a job line; you never
flip a default. Verdicts you write are PROVISIONAL until Claude confirms (runbook verdict rule). Placement per RULING
`compute-placement`: S runs tests and scorers only; every organ or LB line goes to the pool at a provisioned pin.

Every block is ONE Bash call. Set first in each: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees; K=docs/drafts/spiking-commit; RAWP=research/findings/raw/_s2p/word-role-commit`.

## 0. Start condition, worktree, pin, provision

Start when `git -C $P log --oneline -1 main -- research/runners/word_role_commit_organ.py` prints a commit.

```bash
cd $P && git worktree add -b research/word-role-commit $WT/word-role-commit main && cd $WT/word-role-commit && ln -s $P/.venv .venv && ln -s $P/data data
PIN=$(git rev-parse HEAD); mkdir -p $K && printf 'P\t%s\t%s\n' "$PIN" "$(date -u +%FT%TZ)" > $K/PINS.tsv
SIM_NO_PROVENANCE=1 nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_word_role_commit.py tests/test_word_role_parse.py tests/test_s2p_score.py
git add $K/PINS.tsv && git commit -m "ops(word-role-commit): pin P=${PIN:0:9}" && echo "$PIN"
```

| result | do |
|---|---|
| a test fails | STOP. NEEDS-CLAUDE "[word-role-commit] test <name> fails at <sha>" |
| all pass, commit made | provision P: `mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$PIN`; wait for `ok`/`already` in `/opt/host-status/sim-provision/$PIN.json`; `failed` with no `next_retry_utc`: STOP, NEEDS-CLAUDE |

Every line below is pinned to P, the main commit the worktree was cut from. `make_jobs.sh` accepts a HEAD that differs from P
only under docs/drafts and research/findings/raw, so the PINS, jobs and data commits do not move the pin.

## 1. Step 1: dev-seed operating-point sweep (no verdict)

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && S2P_JOBS_OUTD=$K/jobs bash docs/drafts/s2p/make_jobs.sh word-role-commit sweep $PIN
git add $K/jobs && git commit -m "ops(word-role-commit): step-1 sweep lines at ${PIN:0:9}"
cd $P && bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "word-role-commit step 1 dev seeds 1-8: prereg research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md ; mem_gb=2 ; max_age_h=72" </dev/null || echo "ADD FAILED: $l"; done < '"$WT/word-role-commit/$K/jobs/sweep_${PIN:0:7}.txt"
```

Expect 8 lines (any other count: STOP, NEEDS-CLAUDE). Results: `$P/$RAWP/<P7>/sweep/op<I>_s<S>.json`, I = 0..5, S = 1..8 (48 files).

## 2. Select (when all 48 sweep files are in)

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && P7=${PIN:0:7} && mkdir -p $RAWP/$P7 && cp -a $P/$RAWP/$P7/sweep $RAWP/$P7/
$PY tools/s2p_score.py select --packet word-role-commit --sweep-dir $RAWP/$P7/sweep --out $RAWP/$P7/selection.json; echo rc=$?
git add $RAWP/$P7 && git commit -m "data(word-role-commit): step-1 sweep + selection (dev seeds; no verdict)"
```

| rc | do |
|---|---|
| 0 (SELECTED, op = I) | step 3. WORKLOG the table row (`$K/WORKLOG.md`) |
| 1 (NONE) | method banked: PROVISIONAL finding `research/findings/<date>-word-role-commit-operating-point-NONE.md` (the table only); NEEDS-CLAUDE "[word-role-commit] no COMMIT_GRID row passes dev 1-8" |
| 3 (UNDEFINED) | re-queue ONLY the seeds in `missing` once, re-run this block; still 3: STOP, NEEDS-CLAUDE |

## 3. Step O lines, canary, queue

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && OP=$($PY tools/s2p_score.py op --packet word-role-commit --selection $RAWP/${PIN:0:7}/selection.json) && echo op=$OP
S2P_JOBS_OUTD=$K/jobs bash docs/drafts/s2p/make_jobs.sh word-role-commit O $PIN $OP && git add $K/jobs && git commit -m "ops(word-role-commit): step O lines at ${PIN:0:9}, op=$OP"
cd $P && J=$WT/word-role-commit/$K/jobs && bash tools/pool_canary.sh run pool41 "$(head -1 $J/O_${PIN:0:7}.txt)" --wait-s 3600; echo canary-O rc=$?
bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "word-role-commit step O: prereg research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md ; mem_gb=2 ; max_age_h=120" </dev/null || echo "ADD FAILED: $l"; done < '"$J/O_${PIN:0:7}.txt"
```

Expect 6 O lines. Canary rc != 0: STOP, NEEDS-CLAUDE with the canary log path.

## 4. Score O (when the 6 files are in)

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && P7=${PIN:0:7} && mkdir -p $RAWP/$P7/verdicts && cp -a $P/$RAWP/$P7/O $RAWP/$P7/
$PY tools/s2p_score.py O --packet word-role-commit --dir $RAWP/$P7/O --selection $RAWP/$P7/selection.json --pin $PIN --out $RAWP/$P7/verdicts/O.json; echo rc=$?
git add $RAWP/$P7 && git commit -m "data(word-role-commit): step O + verdict"
```

| rc | do |
|---|---|
| 0 PASS | step 5 |
| 1 FAIL | do NOT run R. PROVISIONAL NO-GO finding naming each failing (seed, case), and whether only `read_invariant` failed (prereg section 6 row 5); NEEDS-CLAUDE |
| 3 UNDEFINED | re-queue the UNDEFINED seeds once; still 3: STOP, NEEDS-CLAUDE |

## 5. Step R lines, canary + engagement check, queue

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && OP=$($PY tools/s2p_score.py op --packet word-role-commit --selection $RAWP/${PIN:0:7}/selection.json)
S2P_JOBS_OUTD=$K/jobs bash docs/drafts/s2p/make_jobs.sh word-role-commit R $PIN $OP && git add $K/jobs && git commit -m "ops(word-role-commit): step R lines at ${PIN:0:9}, op=$OP"
cd $P && J=$WT/word-role-commit/$K/jobs && bash tools/pool_canary.sh run pool42 "$(head -1 $J/R_${PIN:0:7}.txt)" --wait-s 7200; echo canary-R rc=$?
```

Expect 12 R lines. Then open the canary shard's `lesion_*.json` and one `intact_a_*.json` and check the probe turn's
`word_role_trace`: `route` = `word_role_commit`, `error` null, `commit_lesioned` true in the lesion arm and false in the intact arm,
`commit.read_invariant` true in the intact arm. Any of these wrong, or the canary rc != 0: STOP, NEEDS-CLAUDE (quote the trace).
Otherwise queue:

```bash
cd $P && PIN=$(awk '$1=="P"{print $2}' $WT/word-role-commit/$K/PINS.tsv) && J=$WT/word-role-commit/$K/jobs
bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "word-role-commit step R: prereg research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=168" </dev/null || echo "ADD FAILED: $l"; done < '"$J/R_${PIN:0:7}.txt"
```

## 6. Score R (when the 12 shards are in)

```bash
cd $WT/word-role-commit && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && P7=${PIN:0:7}
$PY tools/s2p_score.py R --packet word-role-commit --tag s2p-R-wc-$P7 --pin $PIN --selection $RAWP/$P7/selection.json --out $RAWP/$P7/verdicts/R.json; echo rc=$?
git add $RAWP/$P7/verdicts && git commit -m "data(word-role-commit): step R verdict"
```

| rc | do |
|---|---|
| 0 PASS | PROVISIONAL finding `research/findings/<date>-word-role-commit-gate.md` with the prereg section 7 sentences verbatim, the O and R tables, the reported diagnostics (dual-role reads, host-rule agreement, latched min / unlatched max per seed) and the residuals of prereg section 2; merge per RULING `routine-merges`; NEEDS-CLAUDE "[word-role-commit] O+R PASS PROVISIONAL, verify-go due" |
| 1 FAIL | PROVISIONAL NO-GO finding (per-seed reply verdicts); NEEDS-CLAUDE |
| 3 UNDEFINED | if the reason is `engagement-*`: STOP, NEEDS-CLAUDE (do not re-run). Otherwise re-queue the UNDEFINED shards once; still 3: STOP, NEEDS-CLAUDE |

Do not start A2 (wm-focus-bind) or any flip: Claude writes those packets.
