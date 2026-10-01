# B2b: ruling on the two seed-102 cells (s102/discourse-register, s102/episodic-memory)

Written 2026-09-29, before the actions below were taken. Governing text: `research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md`,
Amendment 1 (A1.2 to A1.8) and Amendment 2. F = `a308f1e09babcc9ed096c3c8046d00040391368c`, tag `b2b0924-base`.

## Question

The board (GAP_CLOSURE_MISSION.md, TUESDAY item 3) and the local-model runbook ("Parked for Claude", B2b item b) say the
two cells "were dispatched twice (00:54 pool2, 13:15 pool1/pool2)", left partial arm files and no `lb.json`, and ask
whether the 13:15 dispatch was the cell's one A1.4 re-run.

## Evidence

**1. The 00:54 dispatches were B2a, not B2b.** `research/queue/dispatch.log` lines 36849 and 36850 (2026-09-25 00:54:07 and
00:56:35, pool2) carry the reason "B2a re-run of an invalid s102 cell", `cd ~/derisk-pool/revisions/9db7613296...` (M1)
and `--out research/findings/raw/_load_bearing/_shards/b2a0924/s102/<faculty>/lb.json`. Both landed there: `lb.json`
at 01:53 and 02:59, sidecar `git_sha` 9db7613296. They belong to the B2a battery (tag b2a0924, revision M1) and are
not attempts of any b2b0924 cell.

**2. Each B2b cell was dispatched exactly once.** The only b2b0924-base lines for these cells in `dispatch.log`,
`pool.queue.claims`, `pool.running` and the done/malformed/unchecked files:

| cell | dispatch.log | claim (epoch) | node | sidecar `host` |
|---|---|---|---|---|
| s102/episodic-memory | line 37364, 13:15:26 | 2074 (1790356526) | pool2 (AWS i-00fb44c02f4ffedba) | ip-172-31-47-57 |
| s102/discourse-register | line 37365, 13:16:40 | 2075 (1790356600) | pool1 (AWS i-0f79777d16725fa28) | ip-172-31-47-37 |

Node-to-instance mapping: `research/queue/.aws_pool1` and `.aws_pool2`. The sidecar hosts match the hosts that
`research/coordination/b2b0924_corpus_sha256.tsv` records for pool1 and pool2. No fragment of either line appears in
the dispatcher-fragment audit (`research/findings/raw/_dispatcher_fragment_audit/node_evidence/`, 0 hits for s102).

**3. The attempt ran F, and three of its five files landed.** Primary checkout, `_shards/b2b0924-base/s102/<faculty>/`:
`intact_a_*`, `intact_b_*` and `lesion_*_s102.json`, each with a sidecar that records `git_sha` F in full,
`source_kind` git_archive, both manifest checks true, `SIM_BACKEND=numpy`, and no BRAIN_* key except BRAIN_CHAT_SEED.
Missing: `lesion_*_s102.json.rep0` and `lb.json` (a complete cell of this row has five files; see s42, s43, s101).

| cell | arm starts (UTC, from sidecars) | last file landed (UTC) | arm length |
|---|---|---|---|
| discourse-register | 17:16:41, 17:32:54, 17:48:58 | 18:04:42 | 15.7 to 16.0 min |
| episodic-memory | 17:15:27, 17:45:20, 18:15:18 | 18:43:33 | 28.3 to 29.7 min |

**4. How the attempt ended: both AWS nodes were stopped by the idle-stopper minutes after the next arm would have ended.**
`research/queue/aws_idle_stop.log`: `2026-09-25T18:24:26Z i-0f79777d16725fa28 (pool1) idle >= 20m, no runner -- STOPPING`
and `2026-09-25T19:15:24Z i-00fb44c02f4ffedba (pool2) idle >= 20m, no runner -- STOPPING`. On a 16-vCPU node, one
single-threaded job reads as CPU-idle; the only guard was `pgrep -f '[r]esearch\.runners'`, so each stop means that
no runner was alive at that moment. The missing fourth arm would have ended at about 18:20:40Z (discourse-register) and
19:12Z (episodic-memory), 3.8 and 3.4 min before the stops. The five load readings logged between the two stops, when
only pool2 was running, read 1.0, 1.0, 1.14, 1.08, then 0.04: one single-threaded job ran until after about 19:04Z.
The code that stopped the nodes had no sync-before-stop step. That step merged at 15:48:57 EDT the same day
(559f705b7), after both stops. `pool_sync` has reported pool1 and pool2 "UNREACHABLE (skipped)" ever since. Both
instances are still `stopped`, not terminated (`.aws_spend_ledger.jsonl`, 2026-09-29). Their `job_status.log`
records for these two jobs were never pulled. The copy in the fragment audit ends at 07:35 (pool1) and 10:00 (pool2)
on 09-25.

**Most likely:** each job finished and wrote its `rep0` arm and `lb.json` to the stopped instance's disk, 3 to 4 min
before the stop and before the next 15-min `pool_sync`. **Not verified:** the process's exit is unobserved. A crash in
the fourth arm cannot be ruled out without the instance's disk.

## Ruling

1. **The 13:15 and 13:16 dispatches are each cell's FIRST B2b attempt, not its re-run.** There is no earlier b2b0924
   attempt (item 2), so there is nothing for them to re-run. They also meet none of A1.4(b)'s conditions: they were
   queued, not run by direct ssh on a named node; they carry no `mem_ok.sh`/`memcap.sh` guard; and they are not logged
   in `b2b0924_reruns.tsv`. The one re-run of each cell is unused.
2. **A1.4(c) does not apply.** It covers a cell with no dispatch record and no `lb.json`. These cells have a dispatch
   record and ran F (item 3). A re-queue would not be free. This attempt counts.
3. **Attempt 1 is UNDEFINED, class E** (A1.3: "no `lb.json`"). It is not class C or class I: no verdict exists to
   classify. The first host is pool1 (ip-172-31-47-37) for discourse-register and pool2 (ip-172-31-47-57) for
   episodic-memory.
4. **Procedure:** apply A1.4(a) and A1.4(b). Move attempt 1 aside, then run the one re-run by direct ssh on a named F
   node other than the first host: discourse-register on pool41 (node-dl4g243), episodic-memory on pool42
   (node-dl5d243). Each command is the job line unchanged from `cd` to `--out`, with
   `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` in front of its `env`. Log both in `b2b0924_reruns.tsv`.
   These cells are not the dispatcher-accident class, so Amendment 2's queued `pool_node=` path does not apply, and
   the re-runs go by direct ssh. Both pool41 and pool42 hold F (`.source_revision` git_archive, manifest
   8faea3e6...) and the recorded corpus hash 7a00272e... (re-hashed 2026-09-29 12:10 EDT), so A1.2 rule 4 holds.

## The judgment call (flagged for the owner)

The prereg does not name the case "the attempt probably finished, but its last files are on a stopped node".

- **Reading A: retrieve before ruling.** The `--out` path is on the node, so start pool1 and pool2, pull the files,
  and judge attempt 1 by A1.2. Cost: two r7i.4xlarge for about 15 min. Also, while running they are live dispatch
  targets, because `.pool_extra_nodes` still lists them. If `lb.json` is there and passes A1.2, each cell is DEFINED
  on its first attempt and no re-run is spent.
- **Reading B: class E now; spend the re-run.** A1.2 to A1.5 are applied to the files that the aggregate glob reads
  in the primary checkout (A1.4(a) names that glob). Node copies appear in the prereg only as things to move aside.
  A1.5 says INCOMPLETE is "final, not a waiting state", so the procedure never waits on an unreachable node.
- **Chosen: Reading B**, the one the text most directly supports. It needs no cloud spend and no change to node
  state. The two readings differ only if the re-run itself is not DEFINED. Then attempt 1 is the only other evidence,
  and starting pool1/pool2 to read it is the owner's call. If attempt 1 is ever retrieved, it is kept and reported as
  attempt 1, never scored (A1.4(a)). F on numpy with one thread is deterministic, so a retrieved attempt 1 also works
  as a cross-check on the re-run, as the three torn-cell re-runs did.

## Pending node-side move (A1.4(a)) for the stopped nodes

The node copies cannot be moved while pool1 and pool2 are stopped. `pool_sync` skips unreachable nodes, so the local
move holds for as long as they stay stopped. **Before pool1 or pool2 next starts**, with the torn-cell Steps 1a/1b in
`research/coordination/b2b0924_reruns_commands.txt`, run:

```
ssh -n -o BatchMode=yes -F research/queue/.pool_ssh_config pool1 'cd ~/derisk-pool/revisions/a308f1e09babcc9ed096c3c8046d00040391368c && mkdir -p research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s102 && mv research/findings/raw/_load_bearing/_shards/b2b0924-base/s102/discourse-register research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s102/discourse-register'
ssh -n -o BatchMode=yes -F research/queue/.pool_ssh_config pool2 'cd ~/derisk-pool/revisions/a308f1e09babcc9ed096c3c8046d00040391368c && mkdir -p research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s102 && mv research/findings/raw/_load_bearing/_shards/b2b0924-base/s102/episodic-memory research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s102/episodic-memory'
```

Then pull each moved directory into the primary checkout's `_b2b0924_attempt1/b2b0924-base/s102/<faculty>/` so that
attempt 1's `rep0`, `lb.json` and the node's `job_status.log` record are kept and reported. If a node started before
these moves, `rsync -au` would copy only files missing locally or older locally. After the re-run lands, every
attempt-1 file name exists locally with a newer mtime, so nothing is overwritten. Before the re-run lands, attempt 1's
`lb.json` would reach the canonical path: do not start pool1 or pool2 until both re-runs have landed.

## Actions taken (2026-09-29)

- **A1.4(a), primary checkout, 12:14:49 EDT:** `_shards/b2b0924-base/s102/{discourse-register,episodic-memory}/` moved
  whole to `_load_bearing/_b2b0924_attempt1/b2b0924-base/s102/`. Each holds three arm files and their sidecars. No
  node reachable by `pool_sync` holds either directory. pool41's and pool42's F trees and `~/derisk-pool/sim` were
  checked; pool40 has no route. The node-side move on pool1 and pool2 is PENDING, with the commands above.
- **A1.4(b), direct ssh, not queued:** discourse-register on pool41, started 12:15:07 EDT; episodic-memory on pool42,
  started 12:17:46 EDT. Each command is the `b2b0924_base_jobs.txt` line for the cell. A `diff` showed it identical
  to that line once the guard is removed. The guard `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` is put
  in front of `env`. Each job runs detached under `setsid` with `POOL_JOB_ID=b2b0924-rerun-s102-<faculty>-<epoch>`
  and `POOL_JOB_MEM_GB=8` in its environment, so the dispatcher's node budget counts its 8 GB. Neither variable is
  BRAIN_*, so A1.2 rule 3 is unaffected. Output goes to `~/derisk-pool/b2b0924_reruns/s102_<faculty>.out` on the
  node. The start record and the exit code go to `~/derisk-pool/b2b0924_reruns/job_status.log`.
- **Logged** in `research/coordination/b2b0924_reruns.tsv`, one row per cell (result PENDING until it lands).
- **Other cells checked:** an independent A1.2 check over all 258 lines of `b2b0924_base_jobs.txt`, without
  `lb_shard.py`, finds the other 214 coverable cells DEFINED. These two are the only UNDEFINED coverable cells, so
  no other cell needs a re-run.

## Outcome (2026-09-29 14:10 EDT)

- **s102/discourse-register** (pool41, rc 0 at 13:12): DEFINED, verdict regressed (load-bearing).
- **s102/episodic-memory** (pool42, rc 0 at 14:08): DEFINED, verdict regressed (load-bearing).
- In both cells, the re-run's intact_a, intact_b and lesion arm files parse to JSON equal to attempt 1's three
  landed files from pool1/pool2. On this evidence the choice between Reading A and Reading B did not change the result.
- **Battery B2b (A1.5): COMPLETE.** 216/216 coverable cells DEFINED, 0 FAIL cells. Metric label: load-bearing
  fraction at F a308f1e09 (2026-09-24 production default). Per seed n_load_bearing/n_exercised: 28/34 (s42, s43),
  29/34 (s44, s100, s101, s102). Mean 0.843, SD 0.014 <!--derived-->. Robust core 28, union 29; the only faculty
  between 1 and 5 seeds is affect-marker-spiking-wta (4/6). R3: learned-referent reads `not-covered:thin` on all
  six seeds. multiref.error turns: 0 of 252.
- **A1.9 against B2a (M1, 168/168 B2a cells valid under rules 1-3):** over the same 28 rows, no faculty's count
  differs and no (faculty, seed) cell differs; B2b over those rows has robust core 24, union 25, the same as B2a.
  The eight added rows: affect-appraisal-interoceptive, affective-tom, causal-whatif and multiref-competition are
  load-bearing on 6/6; d5-consolidate, gnw-bus, sleep-replay and spiking-anaphor are exercised and load-bearing
  on 0/6. They move the mean from B2a's 0.949 to 0.843.
- **Deviation:** A1.4(a)'s node-side move on pool1/pool2 was not possible before the re-runs, because the nodes are
  stopped. Its purpose, preventing rsync resurrection, holds while they stay stopped. It is pending in
  `research/coordination/b2b0924_reruns_commands.txt` (bottom block).
- **Caveat carried with the metric (found 14:20 by the B2d draft):** comprehension-monitor's 6/6 load-bearing is an
  organ crash, not a lesion effect. In every lesion arm the comprehension organ raised a `ValueError` (shape
  mismatch 7002 vs 1032), so the decision field was missing while the reply stayed the same. B2b's rule 5 reads only
  the lb.json verdict, so the cells stay DEFINED and the prereg verdict stays COMPLETE. Without that row the robust
  core would be 27 and each seed's fraction one load-bearing faculty lower (reported, not re-scored:
  `load_bearing_with_lesion_arm_organ_error` in b2b_score.json; research/FAILURE_LOG.md 2026-09-29).
- **Caveat carried with the metric:** `lb_shard.py aggregate` records `ltm_mode: off`, but every intact_a arm file
  reads turn source `tiny-demo +LTM` (208/208 at 12:30, all four hosts), so B2b ran with the sim-data LTM bundle.
- Scored by `tools/b2b_score.py`, which does not import lb_shard. Output:
  `research/findings/raw/_load_bearing/_b2b0924_score/b2b_score.json`, with a preconditions block, per-cell states
  and a sha256 manifest. It agrees number by number with the registered
  `research/findings/raw/_load_bearing/_shards/b2b0924-base/aggregate.json` (`lb_shard.py aggregate --pin F`:
  216 checked, 216 valid).
