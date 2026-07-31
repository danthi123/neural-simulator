# Failure log — one line per NEWLY-NOTICED failure mode

**The rule this file exists to enforce: noticing a failure is judgement, but CLOSING it is not.** Add a line the
moment a new failure mode is noticed. `tools/gates/coverage.py` then BLOCKS until that line names a gate, or
explicitly declares why the class cannot be gated.

Without this, closure depended on remembering to build a gate — which is the exact dependency the whole gate
system exists to remove. Every gate before this one was added because I happened to notice and happened to act.

**Format** — one row, `gate:` must name a module in `tools/gates/` OR start with `NOT-GATEABLE:` plus a reason.

| date | failure | gate |
|---|---|---|
| 2026-07-31 | a queued job's command is invalid (flag does not exist) — 9 jobs dispatched, died on argparse | `pool_queue` argparse validation |
| 2026-07-31 | a job dies mid-run and nothing notices — dispatcher logged launches, not exit status | dispatcher `job_status.log` + heartbeat |
| 2026-07-31 | queue METADATA (`#checked:`) executed as part of the command, breaking a brace group | dispatcher strips at pop time |
| 2026-07-31 | two queued jobs meant to differ are byte-identical (loop variable not used in the command) | `NOT-GATEABLE: the one-variable rule is enforced inside experiment.py for ARMS; a queue of independent jobs has no equivalent. Candidate: hash queued commands and flag exact duplicates.` |
| 2026-07-31 | a gate's own corpus fallback silently undoes the hook's scoping | `coverage` (selftest requirement) |
| 2026-07-31 | pool artifacts land outside the provenance door's watched directory | `tools/pull_pool_results.sh` (stamps from dispatch.log at pull time) |
| 2026-07-31 | steering docs (roadmap/board/ROADMAP.md) carry NO frontmatter, so they are invisible to doc_type / status / mechanism gates — 0 of 287 plans declare a type | `doc_type` (once they declare `type:`; roadmap done, 286 plans pending) |
| 2026-07-31 | the roadmap's 16 wall-ledger rows name NO mechanism id, so per-mechanism status cannot flow into the plan and the ledger stays hand-maintained (one row was stale the same day it was written) | `NOT-GATEABLE yet: needs mechanism entries to exist first (1 of ~16). Candidate: a gate requiring every wall-ledger row to name a mechanism id, once the registry covers the walls.` |
| 2026-07-31 | lane_check read gpu.queue ONLY, so pool-staged CPU-lane work was invisible and it reported starvation already fixed | `tools/lane_check.py` (reads both queues) |
| 2026-07-31 | CPU lanes sat 194 min unserved while the heartbeat correctly alarmed every 15 min — a TRUE alarm ignored because it only reports | `gates/lane_starvation` (BLOCKING; waiver auto-expires in 6h) |
