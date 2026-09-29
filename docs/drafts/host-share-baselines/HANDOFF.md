# HANDOFF — host-share baselines (plan step 2): execution packet for the local model

Written 2026-09-29 by Claude. Execute top to bottom. Do not change code, bars, turns or seams. At most ONE fix round per
issue; anything not covered here goes on the NEEDS-CLAUDE list at the end and you stop that line of work.

- Worktree: `/home/dant123/Projects/sim/.claude/worktrees/host-share-baselines` (branch `research/host-share-baselines`).
  Below, `W=` that path, `P=/home/dant123/Projects/sim` (the primary checkout, on main; never switch its branch),
  `PY=/home/dant123/Projects/sim/.venv/bin/python` (the worktree has no `.venv`).
- Never merge to main. Never `--no-verify`. Never write waiver files. Push only with `bash tools/push_both.sh`.
- Set these first in every new shell:
  `W=/home/dant123/Projects/sim/.claude/worktrees/host-share-baselines; P=/home/dant123/Projects/sim; PY=/home/dant123/Projects/sim/.venv/bin/python`
- The shell may be zsh: run every multi-line block exactly as given, inside `bash -c '...'` where shown.

## 0. State at handoff

- Everything is STAGED in the worktree, nothing committed (`gates/compute_idle_persistent` blocked commits).
- The prereg `research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md` is UNTRACKED on purpose: it is
  committed ALONE in step 4 (`gates/prereg_before_run`).
- Dev verification (dev seed 7) is DONE and PASSED (14:17-14:41; shards staged in
  `$W/research/findings/raw/_host_share/dev/s7/`; prereg section 10 already filled). Step 2 is only for a re-check.
- A cupy dev replication (seed 8) is queued on the GPU queue; it is informational only. Ignore it for every verdict.

## 1. When to start

Start step 3 only when commits are possible: after 18:30 on 2026-09-29, OR when
`cd $P && $PY tools/parallel_audit.py | grep -c "UNDER-PARALLELIZED"` prints 0. Before that, only step 2 may run.

## 2. Dev verification (DONE by Claude, PASS; re-run these checks only if a shard file is missing)

```bash
cd $W
ls research/findings/raw/_host_share/dev/s7/*.json          # expect the three shard files listed in section 0
$PY - <<'EOF'
import json, glob
for p in sorted(glob.glob("research/findings/raw/_host_share/dev/s7/*.json")):
    if p.endswith(".prov.json"): continue
    d = json.load(open(p)); ev = d["events"]
    seams = sorted({e["seam"] for v in ev.values() for e in v})
    applied = sorted({e["seam"] for v in ev.values() for e in v if e.get("applied")})
    srcs = sorted({str(r.get("source")) for r in d["responses"].values()})
    errs = [k for k, r in d["responses"].items() if "_error" in r]
    print("- `%s`: seams recorded %s; applied %s; sources %s; errored turns %s; seeds_used %s." % (
        p.split("/")[-1], seams, applied, srcs, errs, d.get("seeds_used")))
EOF
```

Paste the printed lines under "## 10. Dev verification" in the prereg, replacing its placeholder paragraph. Then:

- PASS if: every file exists; `recall__intact_b.json` records seams `k-select`, `bg-select`, `organ-c`, `metacog`,
  `ignition`; the bg-select file lists `bg-select` as applied; the v-choice file lists `v-choice` as applied; every
  source is `tiny-demo +LTM`; no errored turns.
- If a file is missing and no `host_share --worker` process is running (`ps aux | grep "[h]ost_share --worker"`):
  re-run that one shard once with the command below, then re-check. If it fails again: STOP (NEEDS-CLAUDE #1).

```bash
cd $W && bash -c 'bash tools/mem_ok.sh 7 && bash tools/memcap.sh 9 -- '"$PY"' -u -m research.runners.host_share --shard --seed 7 --group GROUP --arm ARM --out research/findings/raw/_host_share/dev/s7/GROUP__ARMFILE.json'
# GROUP/ARM/ARMFILE: recall/intact_b/intact_b ; recall/bg-select:intervene/bg-select_intervene ; compete/v-choice:substitute/v-choice_substitute
```

- Any other failure of the PASS list: STOP (NEEDS-CLAUDE #1). Do not edit code.

## 3. Commit 1 (code, review, evidence) and push

```bash
cd $W
$PY -m research.runners.host_share --selftest | tail -1        # must print: SELFTEST OK
$PY -m research.runners.host_share_schema | tail -1           # must print: SCHEMA SELFTEST OK
$PY -m pytest -q -p no:cacheprovider tests/test_host_share.py  # must print: 10 passed
git checkout -- research/findings/raw/_provenance/runs.jsonl  # do not commit the shared provenance log
git add research/coordination/lane_builds.jsonl research/runners/host_share.py research/runners/host_share_schema.py \
  research/runners/_host_share_draft_confound_check.py tests/test_host_share.py research/FAILURE_LOG.md \
  docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md docs/drafts/host-share-baselines/WORKLOG.md \
  docs/drafts/host-share-baselines/HANDOFF.md \
  research/findings/raw/_host_share/draft_confound_dev_s1-7.json research/findings/raw/_host_share/draft_confound_dev_s1-7.json.prov.json
rm -f research/findings/raw/_host_share/dev/s7/*.log.prov.json   # stray sidecars stamped on log files
git add $(ls research/findings/raw/_host_share/dev/s7/*.json | grep -v '\.log\.' | grep -v '\.part')   # shards + sidecars only
git status --short | grep PREREG                              # must print ONE line starting with "??" (prereg untracked)
git commit -m "build(host-share): the plan-step-2 host-share instrument (6 seams, intervention-classified schema, sham + held-fixed checks) + seam-map review + dev verification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
bash tools/push_both.sh research/host-share-baselines
```

If the commit is blocked ONLY by `compute_idle_persistent` or `lane_starvation`: wait 30 min and retry (max 6 tries),
then STOP (NEEDS-CLAUDE #2). Any other gate failure: STOP (NEEDS-CLAUDE #2). Do not use `--no-verify`.

## 4. Bring main in (only if needed), then commit 2 = the prereg ALONE

```bash
cd $W && git fetch origin main && git merge-base --is-ancestor origin/main HEAD && echo CONTAINS-MAIN || echo NEEDS-MERGE
```

If it prints `NEEDS-MERGE`: `git merge --no-edit origin/main`. If the merge reports a conflict: `git merge --abort`
and STOP (NEEDS-CLAUDE #3). After a clean merge, re-run the three checks at the top of step 3; any failure: STOP
(NEEDS-CLAUDE #3). Then:

```bash
cd $W
git add research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md
test "$(git diff --cached --name-only)" = "research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md" && echo OK-ONLY-PREREG || echo "STOP: something else is staged (git reset -q HEAD -- <that path>, then re-check)" 
git commit -m "prereg(host-share): six-seed host-share baseline, frozen seams/arms/bars (committed before any gate seed)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
bash tools/push_both.sh research/host-share-baselines
SHA=$(git rev-parse HEAD); echo $SHA > docs/drafts/host-share-baselines/GATE_SHA   # the pinned gate revision
```

`GATE_SHA` is a local note; it is committed with the results in step 8.

## 5. Provision the pool at the gate revision

```bash
cd $P && SHA=$(cat $W/docs/drafts/host-share-baselines/GATE_SHA) && bash tools/pool_provision.sh --revision $SHA --isolated pool41 pool42
for n in pool41 pool42; do ssh -o BatchMode=yes $n 'ls ~/Projects/sim-data/knowledge_bundles/ 2>/dev/null' | grep -q wikidata_100k && echo "$n LTM OK" || echo "$n LTM MISSING"; done
```

- If `pool_provision.sh` prints `REFUSED ... does not contain origin/main` (main moved after step 4): STOP
  (NEEDS-CLAUDE #3). Do not merge after the prereg commit: the gate must run at the commit that adds the prereg.
- If any node prints `LTM MISSING`: run `bash tools/pool_sync_assets.sh <node>` once and re-check; still missing:
  STOP (NEEDS-CLAUDE #4). V6 would fail on that node.

## 6. Queue the 108 gate jobs (one per seed x group x arm)

```bash
cd $W && SHA=$(cat docs/drafts/host-share-baselines/GATE_SHA) && $PY -m research.runners.host_share --emit-pool-jobs $SHA > /tmp/host_share_gate_jobs.tsv && wc -l /tmp/host_share_gate_jobs.tsv   # must print 108
cd $P && bash -c 'while IFS=$'"'"'\t'"'"' read -r cmd why; do bash tools/pool_queue.sh add "$cmd" --checked "$why" || echo "ADD-FAILED: $cmd"; done < /tmp/host_share_gate_jobs.tsv'
bash tools/pool_queue.sh list | head -3
```

Any `ADD-FAILED`: re-run the loop once for those lines only (the duplicate guard refuses lines already queued). Still
failing: STOP (NEEDS-CLAUDE #5). Append to the WORKLOG: the SHA, the queue time, and "108 jobs queued".

**ETA.** One job ~40-60 min (a numpy build + 4 turns; ~6.3 GB RSS). The pool has 2 live nodes with 15 GB each (1-2
jobs per node). Expect ~20-45 h wall for all 108. Record the start time; do not re-queue a running job (check
`ssh <node> 'ps aux | grep [h]ost_share'` before declaring anything stuck; `POOL_JOB_MAX_AGE` is 12 h).

## 7. Wait for completion

Results sync back automatically into `$P/research/findings/raw/_host_share/gate/` (`pool-sync.service`).

```bash
find $P/research/findings/raw/_host_share/gate -name '*.json' ! -name '*.prov.json' ! -name '*.part*' ! -name 'aggregate.json' | wc -l   # 108 when complete
```

When the count stops rising for 12 h below 108: list the missing shards with step 8's aggregate (`missing_shards`),
re-queue exactly those lines from `/tmp/host_share_gate_jobs.tsv` ONCE, wait again. Still missing after that: score
anyway (the verdict will be INVALID-OR-INCOMPLETE) and add NEEDS-CLAUDE #6.

## 8. Score with the registered scorer, write the result, commit

```bash
cd $W && mkdir -p research/findings/raw/_host_share/gate && rsync -a $P/research/findings/raw/_host_share/gate/ research/findings/raw/_host_share/gate/
$PY -m research.runners.host_share --aggregate research/findings/raw/_host_share/gate --seeds 42,43,44,100,101,102 --out research/findings/raw/_host_share/gate/aggregate.json
$PY -m research.runners.host_share --report research/findings/raw/_host_share/gate/aggregate.json > /tmp/host_share_report.md
cat /tmp/host_share_report.md
```

Write the finding with this exact header, then the report block unchanged, then the one interpretation line per seam
from section 9 (copy the sentence for each seam's class; add nothing else):

```bash
cd $W && D=$(date +%F) && F=research/findings/${D}-host-share-baselines-RESULT.md && {
printf -- '---\ntype: finding\nstatus: live\ndate: %s\nlane: C-self-workspace\nmechanism: host-decision share baseline, six seeds, scored by the registered scorer\nseeds: [42, 43, 44, 100, 101, 102]\nrunner: research/runners/host_share.py\nartifacts:\n  - research/findings/raw/_host_share/gate/aggregate.json\n---\n\n# Host-decision share baselines: result (%s)\n\nPrereg: research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md (gate revision %s). Scored by `python -m research.runners.host_share --aggregate ... --seeds 42,43,44,100,101,102`.\n\n' "$D" "$D" "$(cat docs/drafts/host-share-baselines/GATE_SHA)"
cat /tmp/host_share_report.md
printf '\n\n## Per-seam reading (fixed sentences from HANDOFF section 9)\n\n'
} > $F && echo "now append one line per seam from HANDOFF section 9 to $F"
```

Then commit and push:

```bash
cd $W && git add research/findings/raw/_host_share/gate research/findings/*-host-share-baselines-RESULT.md docs/drafts/host-share-baselines/GATE_SHA docs/drafts/host-share-baselines/WORKLOG.md
git commit -m "data(host-share): six-seed host-share baseline -- verdict <paste the aggregate verdict>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && bash tools/push_both.sh research/host-share-baselines
```

## 9. Outcome -> what to write (no other interpretation)

Validity (from the report's first line and "Validity" line):

| aggregate says | do |
|---|---|
| `BASELINE-VALID` | write the finding (step 8) and one board line (below). Done. |
| `INVALID-OR-INCOMPLETE`, only `missing shards > 0` | re-queue missing shards once (step 7), re-score. Still missing: finding marked INVALID-OR-INCOMPLETE + NEEDS-CLAUDE #6. |
| V1 pass False (sham/determinism failed on 2+ seeds) | finding marked INVALID; NEEDS-CLAUDE #7. Do not re-run. |
| V6 pass False (a source other than `tiny-demo +LTM`) | finding marked INVALID; NEEDS-CLAUDE #4. |
| V3 False, or PYTHONHASHSEED/seed threading False | finding marked INVALID; NEEDS-CLAUDE #7. |
| V4 False | finding marked INVALID; NEEDS-CLAUDE #7. |

Per-seam sentence (copy the one matching each seam's class; `<seam>` = the seam name):

| class | sentence |
|---|---|
| HOST-OWNED | `<seam>`: host-owned. Flipping the host decision moved the reply on at least 80% of events. |
| HOST-OWNED (replaceable) | `<seam>`: host-owned, and the neural-derived rule gave the same reply on these probes, so it is a replacement candidate. |
| NEURAL-OWNED | `<seam>`: neural-owned. The reply returned to the neural preference when the host decision was flipped. |
| MASKED | `<seam>`: masked. A later stage overrode the flipped host decision; this seam is not what decides the reply. |
| MIXED | `<seam>`: mixed; no single owner on these probes. |
| UNDEFINED | `<seam>`: undefined (see its UNDEFINED reasons). |

For organ-c, metacog and ignition add: "(call-graph check, not pooled)".

Board line (append under CURRENT STATE in `$P/GAP_CLOSURE_MISSION.md`? NO: never edit the primary checkout's files.
Append it to the WORKLOG instead; Claude moves it to the board):
`<date> host-share baseline (plan step 2): <verdict>; measured panel host_decision_share <x>, neural_decision_fraction <y>; classes k-select <c>, v-choice <c>, bg-select <c>; content learned <l>, python-selected <p>. Finding: <path>.`

## 10. Stop conditions (stop the line of work, write the NEEDS-CLAUDE item into the WORKLOG, do nothing else on it)

- A gate blocks a commit for any reason other than compute-idle/lane-starvation, or those persist past 6 retries.
- A merge conflict, or a selftest/test failure after a merge.
- Dev verification fails its PASS list twice.
- A pool node lacks the LTM bundle after one asset sync.
- Queue adds fail twice.
- Any validity failure in section 9 other than missing shards.
- Anything would require editing `research/runners/host_share*.py`, the prereg, or the thresholds.

## 11. NEEDS-CLAUDE (judgment items; do not attempt)

1. Dev verification failure (a seam not live, an intervention not applied, an errored turn).
2. A gate failure other than compute-idle / lane-starvation.
3. Merge conflicts with main, or tests failing after the merge; or the pool refusing a revision already holding the
   prereg.
4. A pool node without the LTM tier (V6).
5. Pool queue adds failing.
6. Gate shards still missing after one re-queue.
7. V1 / V3 / V4 / hash / seed-threading failures: diagnosis.
8. After a BASELINE-VALID result: sync the summary docs + board (`sync-documentation`, `vikunja`), and hand the
   bg-select and ignition numbers to the step-3 builder as its "before" values.
9. Production fix: thread `_brain_chat_seed()` into the value-choice critic, the BG selector and the shared salience
   organ (FAILURE_LOG 2026-09-29 row).
10. The deliberation gate (seam-map C10-C13) looks inert on the default path (stale `_last_gnw_bus`); decide the fix.
11. k-select "order" events are masked by value choice at the reply; decide whether a consensus-level measurement
    (`consensus_intact` / `consensus_intervene`, already recorded per event) should be preregistered as a follow-up.
12. The draft seam map (`research/draft-seam-map`) should absorb the review's corrections before it leaves drafts.
