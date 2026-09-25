---
type: preregistration
status: preregistered
date: 2026-09-24
lane: load-bearing
mechanism: the #1 metric (lesion-verified load-bearing fraction) on the full 50-row registry at F (origin/main
  a308f1e09babcc9ed096c3c8046d00040391368c), at the production default and again with the flip candidate
  BRAIN_LEARNED_REFERENT_LEXICON=1 in every arm of every shard (flip-rule criterion (c) for the learned referent lexicon)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any b2b0924-base or b2b0924-flipcand job or shard exists. F is the measured
  revision; the commit that adds this file is a documentation-only child of F. Amendment 1 (committed before any job was
  queued) withdraws the flipcand arm; B2b runs the base arm only.
---

# Battery B2b: the full registry at F, with and without the learned referent lexicon (pre-registration)

> **Amendment 1, committed before any job was queued, withdraws the flipcand arm.** B2b runs the base arm only (tag
> b2b0924-base). R1, R3's flag reading, the manipulation check, the integrity smoke and "What a PASS licenses" below are
> void; the validity rule, the re-run procedure and the verdict are replaced. Read Amendment 1 at the end first.

## Why

B2a (`research/findings/2026-09-24-production-default-battery-B2a-PREREGISTRATION.md`, tag b2a0924) measures revision M1
9db761329 with the 38-row registry of that revision. Since M1, main gained the row-registry hook (`research/runners/lbf_rows/`)
with three row modules, plus several default-OFF lanes. B2b measures the whole registry at F.

Its second arm answers criterion (c) of the owner's flip rule (2026-09-23) for the first flip candidate that already holds
criterion (a): the learned referent lexicon, `research/findings/2026-09-24-language-learned-referent-production-route-GO-6seed.md`.
The rule: a validated feature flips default-ON only with (a) a 6-seed GO finding, (b) a SOUND review, (c) no regression in a
combined battery with it ON, and (d) a production-default validation run.

## Frozen revision and registry

- F = `a308f1e09babcc9ed096c3c8046d00040391368c` (origin/main when this was filed). Every job runs from
  `~/derisk-pool/revisions/<F>`.
- The registry at F is `load_bearing_fraction.FACULTY_LESIONS` after the hook merge (`LBF_ROW_MERGE_REPORT`): 50 rows
  <!--derived-->. Kinds: 33 neural-lesion, 3 whether-disable, 6 thin, 1 mechanism-only, 4 in-process, 3 proposed.
- Row modules merged: learning, live_organs, reward_value_afferent, with no import error. Rows parked by their own modules stay
  out: self-schema, reasoning-transitive-chat, tom-false-belief. Eight proposed_lesions_conflict_kb rows collide with base rows
  and the hook drops them; the base rows stand.
- Coverable rows (neural-lesion and whether-disable; these enter the fraction): 36. Sharded rows (`lb_shard.py`
  MEASURABLE_KINDS, which adds thin and mechanism-only): 43 per seed, so 258 shards per arm and 516 in all <!--derived-->.
- Against M1: no row removed and no kind changed. Twelve rows added. Eight are coverable: affect-appraisal-interoceptive,
  affective-tom, causal-whatif, gnw-bus, multiref-competition and spiking-anaphor (neural-lesion); d5-consolidate and
  sleep-replay (whether-disable). Four are thin: learned-referent, onebrain-xedge, spiking-qroute and
  surprise-salience-snc-afferent (the last arrived with the A10 merge 11bfd7232). M1 had 28 coverable and 31 sharded rows.

## Arms (fixed)

Both arms: `--no-fixes` (no fix flag, so every mechanism runs at its production default), the adequate probe set (the
`lb_shard.py` default: nine LB_* probe flags), `--repeats 2`, numpy backend, one BLAS/OMP thread, seeds 42 43 44 100 101 102,
the same 50-row registry. Each job line keeps the B2a layout:
`cd <revision dir> && .venv/bin/python tools/assert_flipped_defaults.py && <the lb_shard.py line>`.

```
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2b0924-base     --no-fixes --probe-set adequate --root '~/derisk-pool/revisions/<F>'
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2b0924-flipcand --no-fixes --probe-set adequate --extra-env BRAIN_LEARNED_REFERENT_LEXICON=1 --root '~/derisk-pool/revisions/<F>'
python tools/lb_shard.py aggregate --tag b2b0924-base     --seeds 42 43 44 100 101 102
python tools/lb_shard.py aggregate --tag b2b0924-flipcand --seeds 42 43 44 100 101 102
```

For the same seed and row, a base line and a flipcand line differ in two places only: the tag in the output path, and one
token in the `env` prefix, `BRAIN_LEARNED_REFERENT_LEXICON=1`.

**How the flag reaches the intact and the lesion arm of every shard (read in the code at F).** `lb_shard.py jobs --extra-env`
puts the flag in the `env` prefix of the shard process. Every arm is a subprocess started by
`onebrain_regression_battery._spawn_arm` with `env=dict(os.environ)`. Its worker (`_collect_worker`) only overlays the row's
driving `base_env` and, in the lesion arm, the row's lesion flag. No row at F uses BRAIN_LEARNED_REFERENT_LEXICON as its lesion
flag or in a driving env; its only mention in `load_bearing_fraction.py` and `research/runners/lbf_rows/` is the note of the
learned-referent row.

So in flipcand the intact arm, the null-control rebuild, the lesion arm and its repeat all carry the flag, and in base none
does. A probe that returns before any `_spawn_arm` call (the open-ended distributional ruler) runs inside the shard process and
sees the same environment. The existing `--extra-env` path does the job; no new mechanism is needed.

**Guard.** `tools/assert_flipped_defaults.py` runs before the `env` prefix, in the node's own environment. It checks that F
contains the three 2026-09-23 flips and that no BRAIN_* variable is set on the node. It does not see the flipcand flag, by
design. Static check on the job text at generation: base lines carry no `BRAIN_*=` token, and flipcand lines carry exactly
one, `BRAIN_LEARNED_REFERENT_LEXICON=1`.

**Cost of the flag.** With the flag on, the first `extract_referents` call in each arm process builds the deployment lexicon
(`lexicon_spiking_frame_category.get_lexicon()`: an 8,000,000-character prefix of `data/corpus/tinystories.txt`, lexicon seed
42 in every arm whatever BRAIN_CHAT_SEED is). One standalone build at F on the local box took about 23 s and 0.39 GB peak RSS
(not an artifact). Jobs declare the same memory as B2a, `mem_gb=6`, the `tools/pool_runner_mem.tsv` value for
load_bearing_fraction.

**Integrity smoke (declared here, no weight).** Before dispatch, one flipcand line for a row whose organ calls
`extract_referents` is run at seed 7, which is not a battery seed, into a scratch directory outside both tags. It checks only
that arms build with the flag on and records peak memory. No criterion reads it.

## Validity of a cell (written before any shard exists)

A (faculty, seed) cell of an arm is DEFINED only if all of these hold:
- its `lb.json` exists, and its provenance sidecar records `git_sha` F, `git_dirty` false and `SIM_BACKEND=numpy`;
- in flipcand, the sidecar's env holds `BRAIN_LEARNED_REFERENT_LEXICON=1`; in base, the sidecar's env holds no BRAIN_* key;
- the report is not UNRELIABLE, and `null_control_clean` is not False;
- the verdict is a measured outcome: `regressed`, `pass` or `not-exercised`.

Any other verdict (`arm-build-failed`, `lesion-knob-missing`, `noisy`, `noisy-null-control`, `unmapped-*`, `artifact-*`,
`seed-missing-in-artifact`) makes the cell UNDEFINED. `lb_shard.py aggregate` lists an `arm-build-failed` cell as present and
not load-bearing, so this check reads the per-seed `verdicts` in `per_faculty` and the sidecars, not only
`incomplete_faculties`. An UNDEFINED cell is re-run once with its own unchanged job line at F. It is never scored as 0 and
never as a pass.

## Criteria (written before any shard exists)

- **R1 (no regression with the flag on):** for each of the 36 coverable faculties, n_load_bearing (the number of seeds, out of
  six, with `load_bearing` true) in b2b0924-flipcand is not lower than in b2b0924-base. numpy shards are deterministic, so
  there is no tolerance: one seed lower fails R1 for that faculty. The two arms differ only by the flag, so a drop is
  attributed to the flag. A drop is reported with the host of each cell; if the two hosts differ, both cells are re-run once on
  one host, and that pair decides R1 for the cell.
- **R2 (complete):** in both arms, each of the 36 coverable faculties has six DEFINED cells, with clean null controls. Missing
  or UNRELIABLE is UNDEFINED, never 0.
- **R3 (reported, not gated): the learned-referent row itself.** The row is kind `thin`. `measure_faculty` returns
  `not-covered:thin` for a thin row before it builds any arm, so R3 will read `not-covered:thin` in both arms by construction.
  B2b therefore carries no evidence on whether the lexicon is load-bearing in chat; that evidence is the 6-seed GO finding
  (lesion recovery 0.0 on every seed). Making the row measurable is a registry change (its own note: a single-flag row
  becomes expressible once the lexicon ships default-ON) and is not part of B2b.
- **Verdict, in this order:** FAIL if R1 fails for any faculty with six DEFINED cells in both arms (named). Otherwise
  INCOMPLETE if any faculty still has an UNDEFINED cell after its one re-run, in either arm (cells named); INCOMPLETE is not a
  PASS. Otherwise PASS.

## Reported (not gated)

- Per arm: robust core (members and count), union, mean fraction and SD over the six seeds (per seed, n_load_bearing over
  n_exercised, as `lb_shard.py aggregate` computes it), and the backend, host mix and ltm_mode each aggregate records.
- Per faculty: n_load_bearing in base against flipcand, including any rise.
- Manipulation check: for each shard, the harness's `compare()` over the row's decision fields between the base intact arm and
  the flipcand intact arm, plus whether any turn's `answer` differs. The rows most likely to differ are those whose organs call
  `extract_referents` (the D6 multi-referent WM organ and the activity-silent WM organ). If no intact arm differs anywhere,
  the flag was inert on this probe set, and a PASS says only that it broke nothing the probes reach.
- Comparison with B2a (the aggregate that `lb_shard.py aggregate --tag b2a0924` writes; it does not exist yet at filing). The denominator changes: B2a
  covers the 28 coverable rows at M1, and B2b covers 36. Reported side by side: (i) B2b-base over the same 28 rows, with
  per-faculty n_load_bearing against B2a (a change is attributed to code merged between M1 and F, and is not bisected here);
  (ii) B2b-base over all 36, with the eight added rows named. If B2a is not complete when B2b is scored, the comparison is
  reported as pending, never estimated.

## What a PASS licenses

Criterion (c) of the owner's flip rule for BRAIN_LEARNED_REFERENT_LEXICON: no regression in a combined battery with it ON.
Nothing more. It does not meet (b), the SOUND review, or (d), the production-default validation run on the flipped default.
It is not a claim that the lexicon is load-bearing in chat (R3 cannot measure that), and it licenses no other flag. A FAIL
keeps the flag default-OFF and names each faculty that dropped.

## Residuals

- The six brain seeds share one lexicon trained at seed 42, as in the GO finding: flipcand varies the brain seed, not the
  lexicon.
- Latency is not measured; the lexicon build adds about 23 s to each arm process in flipcand.
- The 2026-09-24 midnight plan (steps S26-S28) sketched B2b at the head of research/integration-0924 with tags b2b-rows and
  b2b-caps. This battery is frozen at origin/main instead, with tags b2b0924-base and b2b0924-flipcand, and holds one flip
  candidate.

## Amendment 1 (2026-09-24, before any b2b0924 job was queued or run)

Committed before any line of either tag entered the pool queue. At this commit no b2b0924 shard, cell or smoke exists.
It follows the pre-queue adversarial review of this document, whose verdict was "amend before queueing". Each part below
says which text above it replaces.

### A1.1 The flipcand arm is withdrawn

- No b2b0924-flipcand line is queued or run. The integrity smoke, which was declared for flipcand, is withdrawn with it.
  The 258 flipcand lines stay in `research/coordination/b2b0924_jobs.txt` as filed and are never queued. The queue source
  is now `research/coordination/b2b0924_base_jobs.txt`: the 258 base lines of that file, same text, same order.
- Reason. The review measured at F that BRAIN_LEARNED_REFERENT_LEXICON=1 changes the referent list on 60 of 112 probe
  turns, in the turn groups of 23 of the 36 coverable rows. Method: `get_lexicon()` built once (seed 42, the primary
  checkout's tinystories.txt), then `extract_referents` with and without it over every probe turn; standalone, no
  artifact written. The lexicon admits question and function words as referents: 'what' on every what-question, 'who',
  'before', 'most', 'today'. On `tom_fb` it pushes 'anne' out of the referent cap.
- R1 counted only seeds that stay load-bearing, so it could not see a change of that kind in the intact replies, and a
  PASS of R1 would not have shown criterion (c). The candidate goes back to its lane with this measurement. Criterion (c)
  for it is untested here, and nothing in B2b licenses its flip.
- Void above: R1; R3's comparison of the two arms; the manipulation check; the integrity smoke; the flag's cost and
  residuals; "What a PASS licenses". B2b is now the first measurement of the full 50-row registry at F, at the production
  default. Everything in "Arms (fixed)" that concerns the base arm stands: `--no-fixes`, the adequate probe set,
  `--repeats 2`, numpy, one BLAS/OMP thread, seeds 42 43 44 100 101 102, 43 sharded rows per seed, 258 shards, and 36
  coverable rows in the fraction.

### A1.2 Validity of a cell (replaces the list under "Validity of a cell")

A (faculty, seed) cell of b2b0924-base is DEFINED only if all of these hold:
1. `lb.json` exists at its job line's `--out` path. Its sidecar `lb.json.prov.json` records `git_sha` F in full,
   `source_kind` `git_archive`, `source_manifest_verified_at_start` true, `source_manifest_verified_at_exit` true, and
   `SIM_BACKEND=numpy` in its env.
2. Every arm sidecar in the shard directory (`intact_a_*`, `intact_b_*`, `lesion_*.json` and `lesion_*.json.repN`, each
   `.prov.json`) records the same `git_sha`, `source_kind` and both manifest checks.
3. The `lb.json` sidecar's env holds no BRAIN_* key. An arm sidecar's env holds none except BRAIN_CHAT_SEED, which
   `main()` sets after import. The 471 B2a sidecars on the primary checkout at this commit read exactly this way (100
   `lb.json` sidecars with no BRAIN_* key, 371 arm sidecars with BRAIN_CHAT_SEED only).
4. The node that ran it holds the recorded corpus hash (A1.6).
5. The report is not UNRELIABLE, `null_control_clean` is not False, and the verdict is `regressed`, `pass` or
   `not-exercised`.

`git_dirty` is no longer checked. In a git_archive revision directory it is hard-coded False (the `_source_snapshot`
fallback in `research/runners/__init__.py`), so it carries no information. Rows of kind thin and mechanism-only return
`not-covered:*` before any arm is built; they are reported and are outside this rule.

### A1.3 Why a cell is UNDEFINED: three causes, read from the files

- **E, the cell did not run F as registered:** no `lb.json`; a sidecar fails rule 1, 2 or 3; the node's corpus hash
  differs (rule 4); or the process was killed (memory or otherwise).
- **C, the code at F failed on the row:** verdict `arm-build-failed`, `lesion-knob-missing`, `lesion-not-effective`,
  `unmapped-*`, `artifact-missing`, `artifact-unreadable`, `seed-missing-in-artifact`, `noisy` or `noisy-null-control`;
  or the report is UNRELIABLE, or `null_control_clean` is False. A numpy shard on one thread is deterministic, so a null
  control or a repeat that disagrees is a property of the build at F, not of the node.
- **I, the probe ran at F and could not decide the row:** any other verdict, for example `probe-inadequate:*`,
  `content-dependent-effect`, `off-organ-route`, `contrast-undefined` or `nonspecific-lesion`.

### A1.4 Re-run procedure (replaces "re-run once with its own unchanged job line")

Each UNDEFINED cell gets exactly one re-run.
- **(a) Move the first attempt out of the aggregate glob.** `lb_shard.py aggregate` reads
  `research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/<faculty>/lb.json` (every seed and faculty). Before the re-run, the whole shard directory moves to
  `research/findings/raw/_load_bearing/_b2b0924_attempt1/b2b0924-base/s<seed>/<faculty>/`, both on the node that ran it
  (under `~/derisk-pool/revisions/<F>/`) and in the primary checkout. Both moves are needed: `tools/pool_sync.sh` pulls
  with `rsync -au`, which copies back any remote file missing locally, so a local-only move is undone at the next
  15-minute sync. The first attempt is kept and reported, never scored.
- **(b) The re-run is not queued, because the dispatcher cannot pin a host.** It runs by direct ssh on one named node
  provisioned with F, never the host of the first attempt, in `~/derisk-pool/revisions/<F>`. The command is the job line
  unchanged from `cd` to `--out`, with `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` put in front of its
  `env`: the dispatcher does not see a direct run, so the node's memory is checked first and the run is capped. The
  re-run's `lb.json` lands at the canonical path and reaches the aggregate through the normal sync. A local copy of F
  counts as a named host only if it has the layout `pool_provision.sh` makes (its `git archive` of F plus
  `.source_revision` and `.source_manifest.sha256`), so that its sidecars can meet A1.2. A git worktree at F cannot,
  because its sidecars record no git_archive source. Every re-run is logged in `research/coordination/b2b0924_reruns.tsv`:
  cell, first host, first cause and verdict, re-run host, command, start time, result.
- **(c) A cell that never ran is not a failed attempt.** If a line went stale in the queue, was lost to a torn line, or
  never reached a node (no dispatch record and no `lb.json`), its line is queued again and this does not use up the
  re-run. The stale line is first moved out of `research/queue/pool.queue`, because `pool_queue.sh`'s duplicate guard
  refuses an identical queued command. At most two such re-queues per cell; after that the cell is MISSING. A torn line
  that did run shows in its provenance (rule 1 or 2 fails): that is class E and goes through (a) and (b).
- **Host of an attempt:** the node named for its dispatch in `research/queue/dispatch.log`, or in the re-run log,
  checked against the `host` field that sidecars record at F.

### A1.5 Verdict (replaces R1, R2 and "Verdict, in this order")

Final state of each coverable cell, once the procedure has run:
- **DEFINED:** the first attempt or the re-run meets A1.2. A DEFINED re-run replaces an UNDEFINED first attempt.
- **FAIL cell:** class C on the first attempt and on the re-run, on two different hosts.
- **Otherwise not DEFINED:** UNDECIDED if the last attempt is class I; MISSING if it is class E or the cell never ran;
  C-UNCONFIRMED if exactly one attempt is class C.

Battery verdict, in this order:
- **FAIL** if any coverable cell is a FAIL cell, each named with its verdicts, hosts and error text. A FAIL cell is
  decided from its own two attempts, so FAIL never waits on other cells.
- **COMPLETE** if all 216 coverable cells (36 rows x 6 seeds) are DEFINED.
- **INCOMPLETE** otherwise, naming every cell that is not DEFINED and its state. INCOMPLETE is final, not a waiting
  state, once every UNDEFINED cell has had its re-run and every cell that never ran its re-queues. INCOMPLETE is not
  COMPLETE.

Per faculty, n_load_bearing (seeds out of six with `load_bearing` true) is a point value only when all six cells are
DEFINED. Otherwise it is reported as the interval [DEFINED load-bearing count, that count plus the cells not DEFINED].
A cell that is not DEFINED is never scored 0 and never as a pass. The robust core and the union are computed over
DEFINED cells, and each faculty whose membership an interval leaves open is named. R3 stays reported: the
learned-referent row reads `not-covered:thin` by construction. Also reported, not gated: per shard, the number of turns
in its arm files that carry a `multiref.error` (a D6 failure the production path records and continues past).

### A1.6 Corpus (new)

F pins the code, not the data. `data/corpus/tinystories.txt` is untracked and rsynced from the primary checkout by
`pool_provision.sh`, and the CORPUS GUARD checks only that the file exists. After provisioning, its sha256 is recorded
for each node's `revisions/<F>` copy in `research/coordination/b2b0924_corpus_sha256.tsv`, committed before the first
line is queued. The primary checkout's copy reads
`7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec` at this commit. A cell from a node whose recorded
hash differs from it is class E. A node re-provisioned during the battery is hashed again before it takes more cells.

### A1.7 Memory (replaces "Jobs declare the same memory as B2a")

Every base line declares `mem_gb=8` through its `--checked` reason. The dispatcher reads the first `mem_gb=N` anywhere in
the line, reserves that much on the node for `POOL_GROWTH_WINDOW_S`, and counts it against the node for the job's life.
The job lines themselves carry no `mem_gb` token and no memcap. 8 is above the `tools/pool_runner_mem.tsv` value of 6
for load_bearing_fraction, which was measured on 2026-09-23 for rows of the 38-row registry; the twelve rows added at F
have no measured peak. Declaring too much costs packing on the 15 GB nodes (pool41, pool42), not validity. Direct re-runs
use `mem_ok.sh 8 2` and `memcap.sh 12` (A1.4 b). A first attempt that dies of memory is class E.

### A1.8 Queueing in waves (new)

`tools/pool_autodispatch.sh` skips any queue line older than `POOL_JOB_MAX_AGE`. The live dispatcher (the
pool-dispatch.service user unit) sets no override, so the default of 43200 s (12 h) applies. Age is measured from the
epoch that `tools/pool_queue.sh` writes as the line's first field when the line is added. A skipped line stays in the
queue and never runs. At 19:22 EDT on 2026-09-24, 151 lines were queued ahead of B2b (45 of them B2a), and
`research/queue/dispatch.log` shows between 5 and 41 dispatches in each full hour from 09:00 to 19:00 that day. Queued at
once, the tail of the 258 lines would pass 12 h.

So the base arm is queued in three waves of 86 lines, two whole seeds each, in job-file order: s42 and s43, then s44
and s100, then s101 and s102. `research/coordination/b2b_queue_next_wave.sh` queues the next wave only when fewer than 20
b2b0924-base lines are still fresh in the queue, and records every add in
`research/coordination/b2b0924_base_waves.tsv`. There is no `FRONT=1`: B2a finishes first. After each wave, every queued
b2b0924-base line is checked to start with `cd ~/derisk-pool/revisions/<F> && ` and to end with `/lb.json  #checked:`,
which catches the torn-line class that hit B2a.

### A1.9 Comparison with B2a (amends "Reported")

Items (i) and (ii) stand, for the base arm only. A B2a cell enters the comparison only if its sidecars meet A1.2 rules
1 to 3 with M1 `9db7613296c3d02a161b36fb15da3983188b7902` in place of F, because B2a's own prereg has no per-cell SHA
rule. B2a's s100/open-ended-generation cell is excluded until its re-run lands. Its first run came from a torn queue line
that lost its revision pin, and it ran `git_sha` 5d10431c6 in `~/derisk-pool/sim`, not M1 (the 2026-09-24 row in
`research/FAILURE_LOG.md`, main commit dcc2c9a49). Each per-(faculty, seed) difference is listed with both hosts. It is
attributed to code merged between M1 and F only when both cells are DEFINED, and it is not bisected here. B2a's
sidecars have no `host` field (M1 predates it), so a B2a cell's host is read from `research/queue/dispatch.log` alone.

### A1.10 What a COMPLETE verdict licenses (replaces "What a PASS licenses")

Reporting the #1 metric, the lesion-verified load-bearing fraction, at F over the full registry: per seed n_load_bearing
over n_exercised, its mean and SD, the robust core, the union, and each faculty's count. Nothing more. It licenses no
default flip and says nothing about BRAIN_LEARNED_REFERENT_LEXICON. A FAIL names each row that the shipped revision
cannot measure as registered; each is a defect at F for that row's lane.

### A1.11 Correction: the after-wave shape check does not catch the torn-line class it names (2026-09-25, dispatcher-fragment audit)

A1.8 says the after-wave check ("every queued b2b0924-base line ... starts with `cd ~/derisk-pool/revisions/<F> && `
... and ... ends with `/lb.json  #checked:`") "catches the torn-line class that hit B2a." It does not, for any of
the ten b2b0924-base fragments the audit found (research/findings/2026-09-25-dispatcher-fragment-jobs-audit.md).
That audit traced the actual tearing to `pop_job`'s READ, not to the queued line: the bug (fixed in 096dfdae0) made
a revision probe's `ssh` (run without `-n`) drain part of `pop_job`'s own candidate stream, so the NEXT `read`
resumed mid-line and dispatched a TAIL of a queue line as if it were a whole job. The line that stayed queued the
whole time (and that A1.8's own shape check would have scanned) was never torn -- it is the intact FULL line,
dispatched correctly, later, in full. A1.8's check inspects exactly that intact copy and would read PASS on every
one of the ten cases; it cannot see a fragment that a mid-read glitch invented from bytes the queue file never
separately stored. See the audit's "Findings at risk" item 1 for the resulting open question (which of the
fragment-touched cells' later, correctly-pinned runs counts as the "first" attempt under A1.4).
