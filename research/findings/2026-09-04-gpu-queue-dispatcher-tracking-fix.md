---
type: finding
status: positive
date: 2026-09-04
mechanism: tools/gpu_queue.sh dispatcher tracking-loss fix -- a ground-truth GPU-residency check
  (nvidia-smi compute-apps, cross-referenced against /proc cmdline) gates every dispatch and every
  `pause --now` kill, instead of trusting gpu.running's single self-reported pid; plus two fd/errexit
  hygiene fixes (fd-8 lock inheritance, unguarded kill under set -e) found while proving the fix works
lane: infrastructure (compute-lane reliability, not a research/science finding -- no seeds apply)
verdict: FIXED and selftested (isolated scratch-dir harness, 5/5 clean repeat runs, both failing
  directions proven). NOT YET LIVE in production -- the running systemd-managed daemon (pid 1696,
  continuous since 2026-08-30) only picks up this fix on its next restart, which this change
  deliberately does NOT trigger (23 real jobs were running; see Deployment below).
artifacts:
  - research/findings/raw/2026-09-04-gpu-queue-dispatcher-tracking-fix-selftest.json
---

# gpu_queue.sh dispatcher tracking-loss fix: standalone jobs, double-start risk, and a broken `pause --now`

## The report

"The dispatcher's `current` showed one pid while the actual GPU compute proc was a DIFFERENT pid, so
`game.sh on` (non-force) couldn't pause it, and there's a risk of the dispatcher starting a second job
while the first still holds VRAM." (owner, this session, referencing `tools/gpu_queue.sh` /
`tools/game.sh`.)

## Root cause: `gpu.running` is a single self-reported pid, never reconciled against reality

`tools/gpu_queue.sh`'s dispatcher (`daemon()`) records exactly one pid per job -- the `$!` of
`setsid bash -c "$job" &` -- into `research/queue/gpu.running`, then blocks on `wait "$jpid"` to know
when the job is "done". Three independent, concretely-evidenced ways that record can stop matching
reality:

**1. A dead dispatcher orphans its job, permanently and silently (proven historically).** The
`__daemon)` case's `trap 'rm -f "$DPID"' EXIT` cleans up only the dpid file -- never `gpu.running`,
never the job itself. If the daemon process dies for any reason (crash, a manual `stop`+`start`, or
now that `gpu-queue-dispatch.service` runs `__daemon` directly under `Restart=always`/`RestartSec=15`,
any nonzero/signalled exit) while a job is genuinely still running, that job is simply orphaned:
reparented, still running, still holding VRAM, with NOTHING recording its pid anywhere. Grepping
`research/queue/gpu_queue.log` for a `START:` line with no matching `DONE(rc=...)` before the next
`dispatcher up` line finds exactly this pattern four times: 2026-08-22 00:21, 2026-08-26 21:35,
2026-08-27 06:41, and 2026-08-27 17:50.

**2. The VRAM-headroom guard cannot catch a lost job at this workload's actual footprint.** Even when
tracking is lost, `MIN_FREE` (default 3000 MiB) is the ONLY thing standing between "a job is already
resident" and "start another one anyway" -- and real jobs in `gpu_queue.log` use 300 MiB-3.4 GB on a
24 GB card. Checked live during this session: 3280 MiB used, 20822 MiB free, while one job was
genuinely resident -- comfortably enough "free" for the guard to wave through a second, third, even a
fourth job stacked on the same card.

**3. Live, present-tense proof the same bookkeeping class silently drifts.** `research/queue/gpu_queue.dpid`
currently reads `3471176` (a dead pid, no matching "dispatcher up" log line, mtime 2026-09-01) while the
actual live daemon is pid `1696` (systemd MainPID, continuous since 2026-08-30, confirmed via
`ps`/`systemctl show`). `status`'s `daemon_alive()` still reports "up" correctly here because it falls
back to a flock probe -- but `status` was, before this fix, printing the wrong pid as if it were
authoritative. This is the identical failure class (a single self-reported pid, unreconciled) manifesting
in the adjacent `dpid` record instead of `gpu.running`, discovered without needing a contrived
reproduction.

`game.sh on` (non-force) is exactly `gpu_queue.sh pause --now`, which reads `gpu.running` and kills that
one recorded pid's process group. When the record is wrong or absent for any of the three reasons above,
`pause --now` has nothing correct to act on -- matching the report precisely, and matching why
`game.sh on --force`'s separate, independent `pgrep`-based sweep already existed as a manual workaround
for exactly this class of problem.

One hypothesis from the initial triage -- that `$!` records a bash wrapper's pid rather than the
`setsid`-launched job's actual leaf pid -- was tested directly and **not** reproduced for the job shapes
this queue actually runs today (`ENV=val python ...` and `cd DIR && ENV=val python ...` both collapse via
bash's tail-exec optimization into a single pid, verified with `ps`/`pgrep` and cross-checked against the
live running job's actual `nvidia-smi --query-compute-apps` pid, which matched). The fix below does not
depend on that hypothesis being true, and defends against it anyway (see "Descendant defense" below).

## The fix

**`gpu_resident_brain_pids()`** (new): cross-references `nvidia-smi --query-compute-apps=pid` against
each candidate pid's own `/proc/<pid>/cmdline`, matching the same `python.*(research\.runners|webapp)`
pattern `game.sh`'s `gpu_python_procs()` already uses -- ground truth, independent of any file this
project writes itself. Guarded by the same `timeout 8` hang-safety the existing `freevram()` already
uses (the 3090-off-the-bus case).

**Dispatch guard (closes double-start, mechanism 1+2):** before popping and launching a new job,
`daemon()`'s contention loop now also requires `gpu_resident_brain_pids()` to be empty -- not just
`freevram() >= MIN_FREE`. If a brain-loading process is resident that the daemon's own record doesn't
(transitively) cover, `_adopt_resident()` overwrites `gpu.running` with the discovered pid (so `status`
and `pause` see the truth, logged once, not spammed) and the daemon waits rather than dispatching. This
check runs on every dispatch attempt, including a freshly-(re)started daemon's very first one, so it
covers both an inherited orphan and a truly-standalone launch outside the queue -- with no special-casing
needed for "startup" vs "steady state". `GPU_QUEUE_NO_RESIDENCY_GUARD` is a test-only bypass to prove the
failing direction; never set in production.

**`pause --now` (closes mechanism 3 / the reported symptom directly):** the kill target set is now the
union of (a) `gpu.running`'s recorded pid and its own process-group (the common, verified case), (b) its
live descendants via `pgrep -P` (descendant defense, in case a future job shape doesn't collapse to one
pid the way today's do), and (c) every pid `gpu_resident_brain_pids()` reports. A stale or absent
`gpu.running` record no longer hides the real job from a plain (non-`--force`) `pause --now`.

**Group-kill safety (found while testing the fix above, not in the original report):** `pause --now`
group-kills (`kill -TERM -pgid`) only when a target is confirmed to be its own process-group leader
(`pgid == pid`) -- true for anything descended from this queue's own `setsid`-launched jobs. A pid
surfaced purely by the residency scan has unknown provenance and is signalled individually, never as a
whole group, so it can never take out an unrelated shell/session that happens to share a process group
with a bare (non-isolated) process.

**Singleton-lock fd-inheritance fix (found while testing, the more serious latent bug):** fd 8 (the
`__daemon`-held singleton `flock`) is inherited by every forked child by default. The dispatched job
itself is the dangerous case: without an explicit close, a job that survives its daemon's death keeps
the lock "held" for its entire remaining runtime -- hours, for the multi-day campaigns this queue runs --
so a freshly-restarted daemon could never win the singleton race to even reach the new residency guard.
Every subprocess `daemon()` forks (the `sleep` calls, the queue-pop subshell, and critically
`setsid bash -c "$job"`) now carries `8>&-`.

**`errexit` safety in the kill sweep (found while testing):** `pause`'s case arm runs under the file's
top-level `set -e` (unlike `daemon()`, which explicitly disables it). A bare `kill ... "$t" 2>/dev/null`
on an already-dead target has a nonzero exit status even with its error message suppressed, which
aborted the whole sweep at the first already-gone pid, silently skipping every later target. Every kill
in the sweep is now `|| true`-guarded.

**`status`:** now shows the resolved dispatcher pid or an explicit "lock-held but the recorded dpid is
stale" note (rather than confidently printing a dead pid), and surfaces any `gpu_resident_brain_pids()`
entry `gpu.running` doesn't cover, so a divergence is visible before someone needs `pause` to find out
the hard way.

## Verification

`tools/gpu_queue.sh --selftest` gained two new tests (fully isolated in a scratch dir via
`GPU_QUEUE_DIR`, plus a fake `nvidia-smi` script via the new `GPU_QUEUE_NVIDIA_SMI` override -- the real
production queue and daemon are never touched, matching the existing TEST A/B isolation contract):

- **TEST C** proves the double-start guard: a `setsid`-isolated fake brain process is spawned and
  deliberately NOT recorded in `gpu.queue`/`gpu.running` (simulating an orphan or a standalone launch).
  With the guard active, a second queued job correctly stays queued (peeks-don't-pops holds). With
  `GPU_QUEUE_NO_RESIDENCY_GUARD=1` (the failing-direction proof), the same fake process does NOT block
  it -- confirming TEST C detects the real mechanism, not a tautology.
- **TEST D** proves `pause --now` reaches the genuinely-resident process even when `gpu.running` names
  an unrelated, already-dead pid (the deliberately-worst-case reconstruction of the reported symptom).

Both new tests exposed the three "found while testing" bugs above before I trusted the fix -- TEST C's
own failing-direction check initially failed to reproduce until the fd-8 leak was fixed (a `sleep` was
found, via `fuser`/`lsof`, still holding the lock a killed daemon was supposed to have released), and
TEST D exposed the `errexit`-under-`set -e` bug directly (traced with `bash -x`).

Full suite result (this session, 5/5 repeat runs clean, exit 0 every time):
`research/findings/raw/2026-09-04-gpu-queue-dispatcher-tracking-fix-selftest.json`.

`bash -n tools/gpu_queue.sh` passes. The live production queue (`research/queue/gpu.queue`,
`gpu.running`, the running job's pid) was diffed before/after this entire session and is byte-identical
-- confirmed untouched throughout.

## Deployment (deliberately not done here)

This fix lives on branch `research/gpu-queue-tracking-fix` and is not yet running in production: the
live `gpu-queue-dispatch.service` (pid 1696) only re-reads `tools/gpu_queue.sh` on its next start, and
23 real jobs were queued/running behind it during this session -- restarting it was explicitly out of
scope ("do NOT disrupt the LIVE gpu_queue"). Landing this fix on `main` and restarting the service (a
brief window where the current job would need to be either let finish or `pause --now`'d and requeued
cleanly, which the fix itself now makes reliable) is the natural next step, at the owner's discretion.
