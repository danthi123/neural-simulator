---
status: aborted
lane: gateb-v14-performance
date: 2026-08-04
type: finding
---

# V14 Stage A Moving-Worktree Attempt Aborted

The first launch of the amended 2,000-step performance matrix was stopped
before its first worker returned a timing result. During that worker, the
controller committed CPU-only local-model lifecycle changes in the same
candidate worktree. The measured simulator files were unchanged, but the Git
revision visible to the worker changed from `c672b1708` to `8274ca4de`.

This does not constitute a performance result. The empty provisional summary
was relabeled `NOT_EVALUATED`, and the artifact records
`status: aborted-no-results`. The preserved artifact is
`research/findings/raw/v14_stageA_performance_c672b1708_ABORTED.json`.

The replacement matrix runs the same committed harness and specification from
an immutable detached candidate worktree at `c672b1708`, against the existing
detached control at `6c9034991`. Subsequent controller commits cannot alter
either measured source.
