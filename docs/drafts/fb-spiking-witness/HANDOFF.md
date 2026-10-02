# HANDOFF: fb-spiking-witness (session-2 packet; execution packet for Hermes)

Written 2026-10-01 by the cloud Claude session 2. Execute top to bottom; use no judgment beyond the tables. Runbook
task: T18. Prereg (frozen bars, battery, selection rule, fixed sentences): `research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md`. Organ: `research/runners/tom_spiking_witness_organ.py`. Wiring:
`webapp/false_belief_chat.py` (default OFF; flag `BRAIN_FALSE_BELIEF_SPIKING_WITNESS`, operating point `BRAIN_FALSE_BELIEF_WITNESS_OP`, lesion `BRAIN_FALSE_BELIEF_WITNESS_LESION`). LB rows:
`research/runners/lbf_rows/fb_spiking_witness.py`. Scorer: `tools/s2p_score.py` (tests `tests/test_s2p_score.py`). Job lines: `docs/drafts/s2p/make_jobs.sh`.

What it moves: "who saw the move". Today the host `_Scenario.present` dict decides it and gates the belief write. Here a spiking presence/absence flip-flop holds the tracked agent's presence, and either a coincidence relay (GRID 0-3) or a frozen threshold read of that population (GRID 4-7) gates the write. Dev seeds 1-8 (Claude, local): GRID 4-6 pass all 10 cases; the relay points did not ignite the belief member, so expect step 2 to select GRID 4 (readout); the finding must say which mode ran.

Rules for this packet: every step below is routine for you EXCEPT where a row says NEEDS-CLAUDE. You never edit the
organ, the wiring, the rows, the scorer, the prereg or a job line; you never flip a default (packet P4 does, after this
packet's step R reads PASS). RULING `s2p-canary` lets you run the pool canary yourself (step 4). Verdicts you write are
PROVISIONAL until Claude confirms a PASS (runbook verdict rule); a NO-GO or UNDEFINED follows RULING `results-without-pass`.

Every block is ONE Bash call. Set first in each: `P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python;
WT=$P/.claude/worktrees; K=docs/drafts/fb-spiking-witness; RAWP=research/findings/raw/_s2p/fb-spiking-witness`.

## 0. Start condition and worktree

Start when `git -C $P log --oneline -1 -- tools/s2p_score.py` prints a commit (the owner landed
`claude/session2-packets`). Then:

```bash
cd $P && git worktree add -b research/fb-spiking-witness $WT/fb-spiking-witness main && cd $WT/fb-spiking-witness && ln -s $P/.venv .venv && ln -s $P/data data
PIN=$(git rev-parse HEAD); mkdir -p $K && printf 'P\t%s\t%s\n' "$PIN" "$(date -u +%FT%TZ)" > $K/PINS.tsv
SIM_NO_PROVENANCE=1 nice -n 10 $PY -m pytest -q -p no:cacheprovider tests/test_fb_spiking_witness.py tests/test_s2p_score.py
git add $K/PINS.tsv && git commit -m "ops(fb-spiking-witness): pin P=${PIN:0:9}" && echo "$PIN"
```

| result | do |
|---|---|
| a test fails | STOP. NEEDS-CLAUDE "[fb-spiking-witness] test <name> fails at <sha>" |
| all pass, commit made | provision P: `mkdir -p $P/research/queue/provision.req && touch $P/research/queue/provision.req/$PIN` (RULING `compute-placement`), wait for `ok`/`already` in `/opt/host-status/sim-provision/$PIN.json` |

## 1. Step 1: the dev-seed operating-point sweep (no verdict)

```bash
cd $WT/fb-spiking-witness && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && bash docs/drafts/s2p/make_jobs.sh fb-spiking-witness sweep $PIN
git add $K/jobs && git commit -m "ops(fb-spiking-witness): step-1 sweep lines at ${PIN:0:9}"
cd $P && bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "fb-spiking-witness step 1 dev seeds 1-8: prereg research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md ; mem_gb=2 ; max_age_h=72" </dev/null || echo "ADD FAILED: $l"; done < '"$WT/fb-spiking-witness/$K/jobs/sweep_${PIN:0:7}.txt"
```

Expect 8 lines. Results land in `$P/$RAWP/<P7>/sweep/` (64 files `op<I>_s<S>.json`, I = 0..7, S = 1..8).

## 2. Select the operating point (when all 64 sweep files are in)

```bash
cd $WT/fb-spiking-witness && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && P7=${PIN:0:7} && mkdir -p $RAWP/$P7 && cp -a $P/$RAWP/$P7/sweep $RAWP/$P7/
$PY tools/s2p_score.py select --packet fb-spiking-witness --sweep-dir $RAWP/$P7/sweep --out $RAWP/$P7/selection.json; echo rc=$?
git add $RAWP/$P7 && git commit -m "data(fb-spiking-witness): step-1 sweep + selection (dev seeds; no verdict)"
```

| rc | do |
|---|---|
| 0 (SELECTED, `op` = I) | step 3 with OP=I. WORKLOG the table row |
| 1 (NONE) | the method is banked: write a PROVISIONAL finding `research/findings/<date>-fb-spiking-witness-operating-point-NONE.md` (the selection table, nothing else), merge it per RULING `routine-merges`, NEEDS-CLAUDE "[fb-spiking-witness] no operating point passes dev seeds 1-8; table at <branch @ sha>; next method per prereg section 3" |
| 3 (UNDEFINED) | re-queue ONLY the seeds named in `missing` once (runbook section 3 rule), re-run this block; still 3: NEEDS-CLAUDE |

## 3. Steps O and R: job lines (gate seeds)

```bash
cd $WT/fb-spiking-witness && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && OP=$($PY tools/s2p_score.py op --packet fb-spiking-witness --selection $RAWP/${PIN:0:7}/selection.json) && echo op=$OP
bash docs/drafts/s2p/make_jobs.sh fb-spiking-witness O $PIN $OP && bash docs/drafts/s2p/make_jobs.sh fb-spiking-witness R $PIN $OP
git add $K/jobs && git commit -m "ops(fb-spiking-witness): step O + R lines at ${PIN:0:9}, op=$OP"
```

Expect `O_<P7>.txt` with 6 lines and `R_<P7>.txt` with 12 lines; any other count: STOP, NEEDS-CLAUDE.

## 4. Canary, then queue O and R

The pool refuses gate-seed lines until a canary of the same runner at the same revision passed (runbook section 3).
RULING `s2p-canary`: run them yourself, one per runner module, then queue:

```bash
cd $P && PIN=$(awk '$1=="P"{print $2}' $WT/fb-spiking-witness/$K/PINS.tsv) && J=$WT/fb-spiking-witness/$K/jobs
bash tools/pool_canary.sh run pool41 "$(head -1 $J/O_${PIN:0:7}.txt)" --wait-s 3600; echo canary-O rc=$?
bash tools/pool_canary.sh run pool42 "$(head -1 $J/R_${PIN:0:7}.txt)" --wait-s 7200; echo canary-R rc=$?
bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "fb-spiking-witness step O: prereg research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md ; mem_gb=2 ; max_age_h=120" </dev/null || echo "ADD FAILED: $l"; done < '"$J/O_${PIN:0:7}.txt"
bash -c 'while IFS= read -r l; do bash tools/pool_queue.sh add "$l" --checked "fb-spiking-witness step R: prereg research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md ; mem_gb=8 ; max_age_h=168" </dev/null || echo "ADD FAILED: $l"; done < '"$J/R_${PIN:0:7}.txt"
```

| result | do |
|---|---|
| canary rc 0 for both, every add succeeds | continue; WORKLOG the time |
| a canary rc 3 (inconclusive) | run it once more on the other node; still 3: NEEDS-CLAUDE |
| a canary rc 1 (FAIL) | STOP. NEEDS-CLAUDE "[fb-spiking-witness] canary FAIL at <sha>: <its log path>" (a code defect, Claude's) |
| an add fails | STOP that file, NEEDS-CLAUDE with the refusal text; never edit a line |

## 5. Score O and R (when 6 O files and 12 LB shards have landed)

```bash
cd $WT/fb-spiking-witness && PIN=$(awk '$1=="P"{print $2}' $K/PINS.tsv) && P7=${PIN:0:7} && cp -a $P/$RAWP/$P7/O $RAWP/$P7/
mkdir -p research/findings/raw/_load_bearing/_shards && cp -a $P/research/findings/raw/_load_bearing/_shards/s2p-R-fb-$P7 research/findings/raw/_load_bearing/_shards/
$PY tools/s2p_score.py O --packet fb-spiking-witness --dir $RAWP/$P7/O --selection $RAWP/$P7/selection.json --pin $PIN --out $RAWP/$P7/verdicts/O.json; echo O rc=$?
$PY tools/s2p_score.py R --packet fb-spiking-witness --tag s2p-R-fb-$P7 --pin $PIN --selection $RAWP/$P7/selection.json --out $RAWP/$P7/verdicts/R.json; echo R rc=$?
git add $RAWP/$P7 research/findings/raw/_load_bearing/_shards/s2p-R-fb-$P7 && git commit -m "data(fb-spiking-witness): steps O and R at ${PIN:0:9} (PROVISIONAL)"
```

UNDEFINED (rc 3): the prereg's one re-run per not-DEFINED cell (move the old file or shard dir aside to
`_attempt1_*`, re-queue the identical line, log it in `$K/reruns.tsv`), then score again; still 3: NEEDS-CLAUDE.

| O | R | do |
|---|---|---|
| PASS | PASS | write the finding `research/findings/<date>-fb-spiking-witness-gate.md` (PROVISIONAL banner; numbers only from O.json/R.json; the prereg section 6 sentence for the mode that ran), then verify-go per RULING `verify-go` (two MiMo Pro refuters), NEEDS-CLAUDE "[fb-spiking-witness] O PASS + R PASS at research/fb-spiking-witness @ <sha>; confirm + merge; unblocks P4 (s2p-flip)" |
| PASS | FAIL | PROVISIONAL finding (which seeds, which rule: intact-wrong / lesion-kept / control-violated), merge per `routine-merges`, NEEDS-CLAUDE "[fb-spiking-witness] R FAIL: <rule> on <seeds>" (the wiring, not the organ, is the defect) |
| FAIL | any | PROVISIONAL finding with the failed cases per seed, merge per `routine-merges`, NEEDS-CLAUDE "[fb-spiking-witness] O FAIL: <cases> on <seeds>" |

Board line (WORKLOG + RESULT line, Claude syncs the board): `<date> fb-spiking-witness: O <v>, R <v> at op <I> (<mode>). Finding: <path>.`

## Stop conditions

A gate blocks a commit for any reason other than the compute gates; a test fails; a pool job fails twice outside the
re-run rule; anything would need an edit to code, rows, scorer, prereg or a job line: STOP, NEEDS-CLAUDE.
