---
type: finding
status: audit
claim_check: measured
date: 2026-09-25
lane: pool-infrastructure
mechanism: audit of every pool job that the pre-096dfdae0 dispatcher claimed from a PARTIAL queue line (the
  revision probe's ssh, run without -n, drained part of pop_job's candidate stream and the next read resumed
  mid-line) -- what each fragment executed, on which node, what it wrote, whether it reached the primary
  checkout, and whether any committed finding or aggregate depends on it
artifacts:
  - research/findings/raw/_dispatcher_fragment_audit/2026-09-25-fragments.json
  - research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json
  - research/findings/raw/_load_bearing/_shards/b2a0924/s100/open-ended-generation/lb.json.prov.json
---

# Dispatcher fragment jobs: what ran, what it wrote, what depends on it (audit, 2026-09-25)

## Plain statement

Until commit 096dfdae0 (2026-09-25 11:04), `tools/pool_autodispatch.sh` could claim the TAIL of a queue line and run
it as if it were a whole job. This audit finds every such claim and traces each one to its node, its exit status, and
its outputs.

- **14 fragments were claimed and dispatched**, from 2026-09-24 16:04:49 to 2026-09-25 07:35:36, all on the two AWS
  nodes (pool1: 9, pool2: 5). A further **5 fragments were set aside unrun** by the dispatcher's `#checked:` gate.
- **9 of the 14 ran nothing.** bash could not find their first word as a command (exit 127).
- **1 ran only a missing script (exit 2).** It is line 1638, the one the review cited. Its first command,
  `tools/assert_flipped_defaults.py`, does not exist in the unpinned node tree (git_archive d7b2a2bb5), so the
  `&&` chain stopped. No mkdir ran and no load-bearing measurement ran.
- **4 ran a real load-bearing measurement (exit 0)** in the UNPINNED `~/derisk-pool/sim` tree of their node, at the
  wrong code revision. These are claims 1454, 1627, 1631 and 1635. The review's list missed all four, because
  each begins with a plausible `NAME=value` word. Each wrote one cell, `pool_sync` pulled each cell into the primary
  checkout, and a later run of the FULL line, pinned to the registered revision, replaced each cell.
- **Every fragment's full line also ran, in full and pinned, afterwards.** No fragment removed its own full line from
  the queue. Every one of those later runs that has finished exited 0. Only the `.json` tail (line 1916) still has
  candidate full lines running or queued at the audit snapshot (2026-09-25 11:33).
- **No committed finding is at risk.** Five items go to the owner (see "Findings at risk"). One of them is not a
  fragment at all: six SETTLE A2 wiring lines ran nothing.

## Scope

- **Window.** The revision probe (`revision_available`) reached main in merge 561efa586 (2026-09-23 22:44:43). The
  dispatcher restarted at 22:53:21 (`[pool-dispatch] started 22:53:21` in dispatch.log). It restarted with the fix
  at 2026-09-25 11:05:52. The scan covers the WHOLE `pool.queue.claims`: 2005 lines at the 11:33:10 snapshot, as
  recorded in the audit JSON's `provenance.inputs`. It finds zero fragments before the window and zero among the 22
  claims made after the fixed restart.
- **Nodes.** Fragments reached only pool1 and pool2. The mini-PCs (pool40/41/42) received none.
- **Read-only.** No queue file, node, shard cell or finding was modified. Node files were read with
  `ssh -n -o BatchMode=yes -o ConnectTimeout=8` (both AWS nodes were up), and copies were kept in session scratch.

## Method

1. **Detector.** A claim is a fragment if its job text is a PROPER SUFFIX of another known full line. Known lines are
   every claim, the live `pool.queue`, `.unchecked` and `.malformed`, and a fragment is never counted as a parent.
   A second scan lists claims whose first word is not a normal command start. Tool:
   `tools/audit_pool_fragment_claims.py` (test: `tests/test_audit_pool_fragment_claims.py`).
2. **Cross-checks on the detector.** All 14 fragments immediately follow a `revision <sha> not provisioned on <same
   node>` line in dispatch.log. That is the one moment the bug can act. Across the window, 19 dispatches follow such a
   line: the 14 fragments, 4 ordinary full lines, and the first A2 line (see below). `pool.running` has 753
   entries in the window at the snapshot, and every one matches a claim.
3. **Execution and exit status.** Each claim is mapped to its `pool.running` record (node and executed text). The
   base64 of that text is looked up in the node's `~/derisk-pool/sim/job_status.log` v2 records (rc).
4. **Writes.** On each node, every file under `~/derisk-pool/sim` (excluding `.venv`) newer than the tree's
   provisioning marker `.pool_environment.json` was listed (pool1: 2026-09-24 13:41Z; pool2: 15:56Z). The node tree's own provenance log (`runs.jsonl` under
   `research/findings/raw/_provenance`) was read. The list contains exactly the four measurement cells below, their
   provenance records, `__pycache__`, `job_status.log` and `autodispatch.out`, and nothing else.
5. **Pull-back.** `research/queue/pool_sync.log` lists the files each sync pulled from a node's main tree. Its runs
   were aligned to the `pool-sync.service` journal's finish times.
6. **Pin rule.** `tools/lb_shard.py`'s own `cell_prov_fails` was run on each fragment-produced cell (copied from the
   node) and on every local cell of `b2a0924` (pin M1 9db7613296c3d02a161b36fb15da3983188b7902) and `b2b0924-base`
   (pin F a308f1e09babcc9ed096c3c8046d00040391368c).
7. **Citations.** Findings, the board and git history (all refs) were searched for the affected cells and for any
   file whose sidecar `argv[0]` lies in an unpinned node tree.

Reproduce (paths are session-local copies of the node files):

```
.venv/bin/python tools/audit_pool_fragment_claims.py --json <out> --node-status <dir of pool1/pool2 job_status.log> \
  --scan-shards b2a0924=9db7613296c3d02a161b36fb15da3983188b7902 \
  --scan-shards b2b0924-base=a308f1e09babcc9ed096c3c8046d00040391368c --node-outputs <dir of copied node cells>
```

## Every fragment

In the table below, a fragment word that ends in a JSON filename is split before `.json`. This stops the claim
checker from reading it as a citation. The exact texts are in the audit JSON. All times are EDT. Every fragment
ran with its node's `~/derisk-pool/sim` as the working directory, with no revision pin.

| claims line | claimed | node | probe that swallowed the bytes | fragment starts | what bash ran | rc | wrote | the full line's real run(s) |
|---|---|---|---|---|---|---|---|---|
| 1454 | 09-24 16:04:49 | pool1 | 49a089d8 on pool1 | `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LB_…` | `load_bearing_fraction --only open-ended-generation --seed 100` at git_archive 5d10431c6, SIM_BACKEND unset | 0 | cell `b2a0924/s100/open-ended-generation` (lb.json, sidecar, oed_distributional_s100.json) | 1579 (operator re-run, 19:23:01 pool2, rc 0) and 1582 (19:24:30 pool2, rc 0), both at M1 |
| 1455 | 09-24 16:06:07 | pool1 | 49a089d8 on pool1 | `ed/lb` `.json` | nothing: command not found | 127 | nothing | tail of a b2a0924 wm-binding-advanced line; every candidate (s43/s44/s100/s101) ran later, rc 0 |
| 1461 | 09-24 16:11:27 | pool1 | 49a089d8 on pool1 | `n-ended-generation/lb` `.json` | nothing | 127 | nothing | b2a0924 open-ended-generation; all 6 seed candidates ran later, rc 0 |
| 1473 | 09-24 16:45:08 | pool1 | 49a089d8 on pool1 | `tive-memory/lb` `.json` | nothing | 127 | nothing | b2a0924 prospective-memory; 4 candidates ran later, rc 0 |
| 1625 | 09-24 20:21:49 | pool2 | 4da72fd2 on pool2 | `/raw/_load_bearing/…/s43/vision-identity-spiking-hmax/lb` `.json` | nothing | 127 | nothing | 1846 (09-25 05:36:30 pool2, rc 0) |
| 1627 | 09-24 20:28:58 | pool2 | 4da72fd2 on pool2 | `ENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LB_…` | `load_bearing_fraction --only d5-consolidate --seed 43` at git_archive d7b2a2bb5, SIM_BACKEND unset | 0 | cell `b2b0924-base/s43/d5-consolidate` (lb.json + sidecar; verdict `unmapped-in-battery`) | 1852 (09-25 05:43:01 pool2, rc 0), at F |
| 1630 | 09-24 20:39:43 | pool1 | 4da72fd2 on pool1 | `dmodel-forward/lb` `.json` | nothing | 127 | nothing | b2b worldmodel-forward; all 4 candidates ran later, rc 0 |
| 1631 | 09-24 20:39:51 | pool1 | d460b449 on pool1 | `RIVE_PROBE=1 LB_DISCOURSE_REGISTER_DRIVE_PROBE=1 …` | `load_bearing_fraction --only causal-whatif --seed 42` at git_archive 5d10431c6, SIM_BACKEND unset | 0 | cell `b2b0924-base/s42/causal-whatif` (lb.json + sidecar; `unmapped-in-battery`) | 1806 (09-25 03:38:28 pool1, rc 0), at F |
| 1635 | 09-24 20:40:22 | pool2 | 4da72fd2 on pool2 | `RM_PROBE=1 .venv/bin/python -u -m …` | `load_bearing_fraction --only affect-appraisal-interoceptive --seed 42` at git_archive d7b2a2bb5, SIM_BACKEND unset | 0 | cell `b2b0924-base/s42/affect-appraisal-interoceptive` (lb.json + sidecar; `unmapped-in-battery`) | 1810 (09-25 03:48:21 pool2, rc 0), at F |
| 1637 | 09-24 20:45:14 | pool1 | 4da72fd2 on pool1 | `coding/lb` `.json` | nothing | 127 | nothing | b2b da-gated-encoding; all 4 candidates ran later, rc 0 |
| 1638 | 09-24 20:55:50 | pool2 | 4da72fd2 on pool2 | `& .venv/bin/python tools/assert_flipped_defaults.py && mkdir …` | `POOL_CHECKED_REASON=… &` (backgrounded assignment), then python on a script absent at d7b2a2bb5 | 2 | nothing (no mkdir, no provenance record) | 1752 (09-25 00:19:36 pool2, rc 0), at F |
| 1647 | 09-24 20:58:04 | pool1 | 4da72fd2 on pool1 | `earing_fraction --only curiosity-followup …` | nothing | 127 | nothing | 1791 (09-25 02:52:42 pool2, rc 0) |
| 1909 | 09-25 07:31:27 | pool2 | 5b5ea1b7 on pool2 | `nce/lb` `.json` | nothing | 127 | nothing | b2b self-initiated-utterance s44 (1948) or s100 (1991); both ran later, rc 0 |
| 1916 | 09-25 07:35:36 | pool1 | 5b5ea1b7 on pool1 | `.json` | nothing | 127 | nothing | tail of any of 83 candidate b2b lines: 62 ran later with rc 0, 17 were still running at the snapshot, and 4 are still queued |

**Short tails have ambiguous parents.** A tail such as `.json` matches many lines. A candidate counts as the parent
only if it was still queued at the fragment's claim time, meaning it was claimed later or is queued now. Each such
fragment ran nothing, so the ambiguity affects only the bookkeeping, not any output.

**Quarantined, never run (5).** Each of these follows a failed probe. Each begins inside the `#checked:` reason, so
it carries no `#checked:` and the record-check gate set it aside. dispatch.log keeps the first 96 characters (lines
36451, 36474, 36477, 36637, 36682). `pool.queue.unchecked` no longer holds them; it was rewritten on 2026-09-25 at
04:05. The last two are tails of B2b lines and of `ca3_superposed_fact_attractor` lines. The first three are tails
of lines that are in neither the claims nor the live queue. Those lines were never dispatched, and a quarantine
cannot remove a real line.

## The four cells a fragment wrote

| cell | fragment | node tree | fragment's result | pulled into the primary checkout (sync run finished) | replaced by | pin rule on the fragment copy | local copy now |
|---|---|---|---|---|---|---|---|
| `b2a0924/s100/open-ended-generation` | 1454 | pool1, 5d10431c6 | `regressed`, load_bearing true | 09-24 16:17:51, and again 19:34:56 | run 1582, lb.json mtime 09-24 19:25:06 | excluded: git_sha is not M1, env.SIM_BACKEND unset, oed file has no sidecar | pinned M1 run, started 23:24:31Z |
| `b2b0924-base/s43/d5-consolidate` | 1627 | pool2, d7b2a2bb5 | `unmapped-in-battery` | 09-24 20:35:31 | run 1852, lb.json 09-25 06:00:18 | excluded: git_sha is not F, SIM_BACKEND unset | pinned F run |
| `b2b0924-base/s42/causal-whatif` | 1631 | pool1, 5d10431c6 | `unmapped-in-battery` | 09-24 20:50:52 | run 1806, lb.json 09-25 04:34:49 | excluded: same two fields | pinned F run |
| `b2b0924-base/s42/affect-appraisal-interoceptive` | 1635 | pool2, d7b2a2bb5 | `unmapped-in-battery` | 09-24 20:50:52 | run 1810, lb.json 09-25 04:10:08 | excluded: same two fields | pinned F run |

- **No real artifact was overwritten by a fragment, on a node or locally.**
  - On a node, a fragment wrote only under `~/derisk-pool/sim`. The real run wrote only under
    `~/derisk-pool/revisions/<pin>`.
  - Locally, `pool_sync` uses `rsync -au`, so the newer file wins. In each cell the fragment ran BEFORE any real run
    of that cell. No earlier real copy existed for it to replace, and the later real copy replaced it.
  - For `b2a0924/s100/open-ended-generation`, the fragment's lb.json and oed_distributional_s100.json are JSON-equal
    to the pinned run's (parsed comparison). Only the sidecar differs.
- **The local shard trees are clean.** The pin rule passes all 186 local `b2a0924` cells at M1 and all 147 local
  `b2b0924-base` cells at F. No sidecar in either tag has an `argv[0]` outside `~/derisk-pool/revisions/`.
- **Provenance.** The node-tree `runs.jsonl` records the 4 fragment runs (run_ids 1790280290-1098350,
  1790296139-1555371, 1790296792-1720678, 1790296822-1575672). None of these run_ids appears in the primary
  checkout's provenance log or in any shard sidecar. `pool_sync` excludes `_provenance/`. No commit on any ref
  contains an LB sidecar from `~/derisk-pool/sim`.

## What depends on these cells

- **B2a** (`2026-09-25-production-default-battery-B2a-FAIL.md` and `...-B2a-rescored-PASS.md`).
  - These findings rest on `research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json`. That aggregate
    was produced with `--pin` M1, and its provenance block reads `status: verified`, with 168 cells checked and 0
    invalid.
  - The only B2a cell a fragment wrote is `s100/open-ended-generation`. The B2b pre-registration review caught it on
    2026-09-24. At ~19:20 the cell was moved to `.claude/worktrees/_rescue_tmp_2026-09-24/b2a_offrev_cell`, and a
    pinned re-run was queued (1579).
  - The committed sidecar of that cell (commit 50916e654) records M1 and a start of 23:24:31Z.
  - Even the fragment's content equals the pinned content.
- **B2b.** No finding cites a `b2b0924-base` cell. No B2b aggregate exists yet, and no B2b file is in git.
- **Nothing else.** No other runner's output was written by a fragment.

## Findings at risk

**No committed finding is at risk.** Nothing here was retracted or rewritten. The five items below go to the owner.

1. **B2b: which attempt is "first" for three cells.** The cells are `s43/d5-consolidate`, `s42/causal-whatif` and
   `s42/affect-appraisal-interoceptive`.
   - The B2b prereg (A1.4(c)) says a torn line that DID run is class E. Class E goes through A1.4(a)/(b): move the
     first attempt aside, then re-run once by direct ssh on a DIFFERENT host, logged in
     `research/coordination/b2b0924_reruns.tsv`. That file does not exist.
   - In each of these cells, the torn run was followed by the unaltered full line, which ran at F via the dispatcher
     on the SAME host (pool2, pool1, pool2).
   - Reading 1: the torn run is the first attempt. Then each F run is an unlogged same-host re-run.
   - Reading 2: the torn run was not the registered job line. Then each F run is the first attempt and the cell
     stands as it is now.
   - The owner should settle this before the B2b aggregate is scored. Per A1.4(c), the other B2b fragments ran nothing
     and are not attempts.
2. **SETTLE A2 never ran (not a fragment).** Claims 1923-1928 are six complete lines that each begin with a label:
   `A2 wiring seed <s>: mem_gb=8 && cd ~/derisk-pool/revisions/5b5ea1b7… && … --run-wiring …`.
   - On pool2 at 09:59:54-10:00:26, bash ran `A2`, got exit 127 and skipped the whole `&&` chain.
   - pool2 has no `_affect_marker_settle_congruence/wiring` output and no such process. No later claim re-ran them.
   - The 11:06 board entry (5e1c78128) says "SETTLE A2 six seeds dispatched to pool2 at 10:00".
   - This is logged in `research/FAILURE_LOG.md`.
3. **The fragment outputs are still on the nodes.** The four cells above still exist in pool1's and pool2's
   `~/derisk-pool/sim` trees. `pool_sync` re-pulls a node file whenever the local copy is missing.
   - This happened on 2026-09-24. The operator moved the bad B2a cell aside at ~19:20, and the 19:34:56 sync pulled
     the fragment copy straight back from pool1. It was replaced only because a newer pinned copy already existed
     on pool2. That copy was written at 19:25:06, and pool_sync pulls pool2 after pool1.
   - The pin rule would exclude such a copy, so the result would be an excluded cell, not a silent wrong value.
   - Recommendation: remove those four cell directories from the two unpinned trees. This read-only audit did not
     remove them.
4. **The 2026-09-24 failure-log row has the wrong cause.** That row (commit dcc2c9a49) blamed an unlocked append for
   the "torn queue line", and it was closed by making `pool_queue.sh` append under a lock.
   - The lock did not stop the fragments. Ten more were dispatched after that row, from 09-24 20:21 to 09-25 07:35.
   - The real cause is the probe's stdin drain, fixed in 096dfdae0, which has its own row.
   - The owner may want to annotate the older row.
5. **B2b has no recorded pin.** `b2b0924-base` has no `PIN.txt`. Its aggregate must be run with
   `--pin a308f1e09babcc9ed096c3c8046d00040391368c`. `gates/lb_battery_provenance` blocks a new unverified
   `aggregate.json`, so this cannot pass silently. With the pin, every fragment copy is excluded (see the table).

## Adjacent observation (not caused by the fragments)

`tools/lb_shard.py`'s covered-by-parent window takes its upper bound from the `lb.json.prov.json` file mtime. A git
checkout resets that mtime.

- The tracked sidecar for `b2a0924/s100/open-ended-generation` reads 2026-09-25 00:43:26, which widens that cell's
  window to about 5 h (the aggregate records `[1790292271, 1790311407]`).
- The content-match requirement still binds the covered file to the parent's own result, so no admission depends on
  the wide window here.
