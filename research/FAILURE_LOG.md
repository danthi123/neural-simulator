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
