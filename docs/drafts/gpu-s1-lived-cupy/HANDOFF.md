# HANDOFF: Stage-1 C3 (`lived`) cupy replication on the workstation GPU (runbook T24; informational)

Written 2026-10-02 by cloud Claude (session 4). For Hermes on S. Rows marked NEEDS-CLAUDE: one line under NEEDS
CLAUDE in the runbook, then stop this task.

**What it is.** The Stage-1 exit gate (T6, GATE `c238ff325`) measures C3, lived-fact recall, on numpy, where the D5
episodic write is forced on (`BRAIN_EPISODIC_STORE=1`) because on numpy the production build defers it (about 510 s
per topic). The prereg names this a declared departure and registers the follow-up this packet runs (section 10: "a
cupy replication of `lived` without the forced write is an optional follow-up through `tools/gpu_queue.sh`, never a
gate arm"; HANDOFF NEEDS-CLAUDE #11). On cupy the write is on by default (`webapp/server.py::_episodic_store_ok`), so
the group's `BRAIN_EPISODIC_STORE=1` equals the production GPU default there: this is the production GPU path.

**What it is not.** Not a gate arm, not a verdict on C3, never averaged with or substituted for the gate. numpy and
cupy at the same seed build different networks (docs/ENGINE_REFERENCE.md, the cross-backend seed trap): read it as
six more networks run through the same protocol on the production substrate. Its own directory,
`research/findings/raw/_stage1_exit/gpu_lived_cupy/`, is never the gate's.

**Ruling (Claude, this session).** Run it now, not "after the pool gate has landed" (#11's wording): it uses no pool
time, and the gate's bars are frozen, so nothing here can move them. Do not read or report its numbers until T6's
gate is scored (RULING `s1-lived-cupy`): this keeps the gate's write-up blind to it.

**Code.** The gate's own runner at the gate's own pin, `c238ff325`, through `tools/gpu_remote.sh add --rev`
(session 4; the pin predates the bridge). Harness identical to the gate's (stub renderer, no LLM, tiny-demo + LTM,
`PYTHONHASHSEED=0`, one BLAS thread) except `--backend cupy`. Arms: `intact_a`, `intact_b`, `lesion_recall` +
`.rep0`, `lesion_episodic` + `.rep0`, `permuted` (the runner's `arms_for("lived")`). Seeds 42 43 44 100 101 102 (gate
seeds run under the committed Stage-1 prereg, which `gpu_add.sh` checks at the pin). Job lines:
`docs/drafts/gpu-s1-lived-cupy/make_jobs.sh` (2 dev lines, 42 gate lines), each under `tools/mem_ok.sh 8 4` (waits up to
90 min for RAM) and `tools/memcap.sh 12`.

**Expected GPU time.** Unmeasured on cupy; the dev smoke measures it. On numpy a `lived` arm takes about 65 min, most
of it the forced write, which takes seconds on cupy: expect roughly 10-20 min per arm, so about 7-14 GPU-h for 42.
Pause-safe: `pause --now` kills the running arm and re-queues its line; the arm re-runs whole (the runner writes the
shard only at the end; a stale `.part.json` is overwritten).

Every block is ONE Bash call. Set first in each:
`P=/home/dant123/Projects/neural-simulator; PY=$P/.venv/bin/python; WT=$P/.claude/worktrees;
GATE=c238ff325a38e40a68b1f9751822630ff729377b; O=research/findings/raw/_stage1_exit/gpu_lived_cupy`

## 0. Start

Start when `git -C $P log --oneline -1 -- docs/drafts/gpu-s1-lived-cupy/make_jobs.sh` prints a commit (the owner
landed session 4). Then:

```bash
git -C $P worktree add -b research/s1-lived-cupy $WT/s1-lived-cupy main 2>&1 | tail -1
git -C $P branch -a --contains $GATE | head -3
git -C $P cat-file -e $GATE:tools/local_llm/gpu_add.sh && echo guard-at-pin
```

No branch listed, or no `guard-at-pin`: STOP, NEEDS-CLAUDE "[T24] gate pin not reachable on S". WORKLOG:
`docs/drafts/gpu-s1-lived-cupy/WORKLOG.md` on `research/s1-lived-cupy` (create; append after every step).

## 1. Dev smoke (seed 7: intact_a, and intact_b for the null on cupy)

The aggregate reads a seed UNDEFINED when `intact_a` and `intact_b` (two same-seed rebuilds) differ on a turn. On
cupy that is not guaranteed (GPU arithmetic order), so the smoke checks it before 42 lines are spent.

```bash
cd $P && bash docs/drafts/gpu-s1-lived-cupy/make_jobs.sh dev | while IFS= read -r line; do bash tools/gpu_remote.sh add --rev $GATE "$line" </dev/null || echo ADD-FAILED; done
cd $P && bash tools/gpu_remote.sh yield --max-hours 2
```

When both `$P/$O/dev/s7/lived__intact_a.json` and `..._intact_b.json` have arrived (sim-sync; check next session):
`$PY $P/docs/drafts/gpu-s1-lived-cupy/devcheck.py $P/$O/dev/s7` (backend, brain source, corpus, hash seed, turn
errors in both arms, and the null; prints `GO <wall_s>` or `STOP <reasons>`).

| result | do |
|---|---|
| `GO <wall_s>` with `wall_s` <= 2400 | step 2 |
| `GO <wall_s>` with `wall_s` above 2400 (42 arms would take over 28 GPU-h) | step 2 anyway, and an INFO line "[T24] cupy lived arm <wall_s> s" |
| `STOP` naming `null diffs` | STOP, NEEDS-CLAUDE "[T24] cupy null fails at dev seed 7: <labels>" (every gate seed would read UNDEFINED) |
| any other `STOP` | STOP, NEEDS-CLAUDE "[T24] dev smoke: <the printed line>" |
| a job ended `DONE(rc!=0)` (`bash tools/gpu_remote.sh log 80`), or `ADD-FAILED` | STOP, NEEDS-CLAUDE "[T24] dev smoke: <rc or refusal and the last error line>" |

## 2. Queue the 42 gate-seed lines, then free the card

```bash
cd $P && n=0; while IFS= read -r line; do bash tools/gpu_remote.sh add --rev $GATE "$line" </dev/null || { echo "ADD-FAILED at $((n+1))"; break; }; n=$((n+1)); done < <(bash docs/drafts/gpu-s1-lived-cupy/make_jobs.sh gate); echo "queued $n"
cd $P && bash tools/gpu_remote.sh yield --max-hours 12
```

If T23's mouth lines are queued, these run after them (FIFO): yield again next session until 42 shards are in.
`ADD-FAILED`: STOP, NEEDS-CLAUDE with the message and the count (never re-add queued lines). A line that ended
`DONE(rc!=0)` (log): re-queue that one line ONCE, unchanged, with the same `add --rev` command; a second failure
stays missing (the aggregate reads it UNDEFINED for that seed).

## 3. Aggregate (after T6's gate is scored, and 42 shards or the queue holds none of these lines)

```bash
ls $P/$O/s*/lived__*.json | wc -l
mkdir -p $WT/s1-lived-cupy/$O && rsync -a --exclude dev $P/$O/ $WT/s1-lived-cupy/$O/ && rsync -a $P/$O/dev $WT/s1-lived-cupy/$O/
cd $WT/stage1-exit && $PY -m research.runners.stage1_exit_gate --aggregate $WT/s1-lived-cupy/$O --seeds 42,43,44,100,101,102 --out $WT/s1-lived-cupy/$O/aggregate.json > $WT/s1-lived-cupy/$O/report.md; echo rc=$?
$PY -c "import json;g=json.load(open('$WT/s1-lived-cupy/$O/aggregate.json'))['groups']['lived'];print('lived verdict:',g.get('verdict'))"
```

(`$WT/stage1-exit` is T6's worktree; its runner is byte-identical to the pin's: `git -C $WT/stage1-exit diff --quiet
$GATE HEAD -- research/runners/stage1_exit_gate.py && echo same`. Not `same`: STOP, NEEDS-CLAUDE.) The report lists the
other criteria as missing or UNDEFINED: they were not run here, by design.

## 4. Write up (PROVISIONAL, informational)

`research/findings/<date>-stage1-lived-cupy-replication-INFORMATIONAL.md` on `research/s1-lived-cupy`, with the
runbook's PROVISIONAL line under the H1, and only these sentences, numbers from the aggregate:
- "Informational cupy replication of Stage-1 criterion C3 (lived-fact recall), the gate's runner at `c238ff325`,
  seeds 42 43 44 100 101 102, the production GPU default (episodic write on by default). Not a gate arm."
- "On cupy the `lived` group reads <verdict> (<k> of 6 seeds pass; per seed: <seed: PASS/FAIL/UNDEFINED and its
  reasons>)."
- "The numpy gate's C3 reads <T6's lived verdict> (T6 finding). The two are different networks at the same seeds
  (docs/ENGINE_REFERENCE.md, cross-backend seed trap): a difference is not a backend effect on one network."
- "What this does not show: any gate verdict; C1, C2 or C4; that numpy and cupy agree cell by cell."
Then paste `report.md`'s `lived` block below them, unedited. Commit the finding, `$O` (shards, sidecars,
`aggregate.json`, `report.md`) and the WORKLOG; merge per RULING `routine-merges` (an informational result, RULING
`results-without-pass`); RESULT line "[T24] RESULT: cupy lived <verdict> (<k>/6); numpy gate C3 <verdict>;
research/s1-lived-cupy @ <sha>".
