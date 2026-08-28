---
type: finding
status: partial
lane: continuous-life/generation
date: 2026-08-28
mechanism: unchanged (no code edits this session) — production-scale port of the on-substrate generative
  attractor-wander dendritic dAP bistable-latch blend-completion (board #104 rung 2), default-OFF live idle-tick
  wire-in via `BRAIN_CONTINUOUS_IDEATE_SPIKING`; this session's change is OPERATIONAL — root-caused why the
  2026-08-27 staged 6-seed cupy production verify never ran, reconfirmed no regression, and correctly re-staged it
  on the PRIMARY gpu.queue against `main` (not an agent worktree)
seeds: [42]
instrument: research/runners/_generative_attractor_wander_onsubstrate_derisk.py (regression smoke only, non-emergent
  n_ca3=400 path); no new instrument
runner: research/runners/_generative_attractor_wander_onsubstrate_derisk.py
artifacts:
  - research/findings/2026-08-28-generative-wander-production-verify-restaged-after-queue-loss.md
  - research/findings/raw/_generative_attractor_wander_onsubstrate/regression_smoke_2026-08-28_seed42.json
  - research/findings/raw/_generative_attractor_wander_onsubstrate/batch1.json
  - research/queue/gpu.queue (append, on the PRIMARY checkout — not committed by this session, see below)
---
# The 2026-08-27 staged 6-seed cupy production-scale generative-wander verify never ran (queue entry lost); root-caused, regression-reconfirmed, and correctly RE-STAGED on the PRIMARY gpu.queue against `main` — PARTIAL, no verdict yet

## State of the lane (checked this session, before touching anything)

[[2026-08-27-generative-wander-production-scale-PARTIAL]] landed the production-scale port (n_ca3=2000, emergent
DG-selected, BTSP-formed membership) of the board #104 rung-2 generative attractor-wander, plus its default-OFF
live idle-tick wire-in (`webapp/continuous_engine.py`'s `BRAIN_CONTINUOUS_IDEATE_SPIKING`), on branch
`research/generative-wander-production` (commit `a4e2a015e`). That commit **is already an ancestor of `main`**
(`git merge-base --is-ancestor a4e2a015e main` → yes; `git log main..origin/research/generative-wander-production`
is empty) — the code + wiring are NOT the pending rung. The finding explicitly conditioned any production-default
flip on a staged 6-seed `--emergent` cupy GPU verify (the directory
`research/findings/raw/_generative_attractor_wander_onsubstrate/` plus the filename
`production_n_ca3_2000_6seed.json`, deliberately not written as one contiguous path here — same reason the source
finding itself split it, and this restaging finding again below — so this NOT-YET-EXISTING path is not misread as
an already-cited artifact), claimed queued at `gpu_queue.sh add`, depth 11, at write time.

A same-day LATER finding, [[2026-08-27-production-default-flips-session-verification-no-flips-landed]]
(Candidate 2, its own summary table), independently checked and confirmed: that output file still does not exist,
and the `research/generative-wander-production` branch (both `origin` and `gitea` remotes) has no follow-up commit
past `a4e2a015e` — the verify never landed. This session re-confirmed the same absence (file absent; branch HEAD
unchanged) and went one step further: found out WHY, then fixed it, rather than re-describing the same gap a third
time.

## Root cause (this session's investigation)

`git log --all --oneline -p -- research/queue/gpu.queue | grep generative_attractor_wander_onsubstrate` returns
**zero hits, across every commit in the repository's history** — the queue-add was never captured by a commit, in
either direction (add or pop). `research/queue/gpu_queue.log` and `research/queue/dispatch.log` (the dispatcher's
own runtime logs, which print a `START:` line with the FULL job string the instant a job is popped) also contain
**zero** mentions of this runner — the daemon never dispatched it. The two existing queue backups
(`gpu.queue.wedge-bak`, `gpu.queue.harvest-bak`) both predate the 2026-08-27 15:28 window and don't contain it
either. Conclusion: the append the 2026-08-27 finding narrated either did not actually persist to the PRIMARY
`/home/dant123/Projects/sim/research/queue/gpu.queue` (the file the shared singleton dispatcher — verified alive,
`tools/gpu_queue.sh status` → `dispatcher: up`, this session — actually polls), or was overwritten by a later
queue rewrite before the daemon reached it. `research/queue/gpu.queue` is an **uncommitted, mutable, per-checkout
working-tree file** (git-tracked but not committed on every write — the existing "queue(gpu): bump..." commits are
periodic syncs, not per-add), so an append made against a throwaway agent worktree's own copy is not the same file
the shared daemon watches unless the command explicitly targets the PRIMARY root. This is a plausible, sufficient
explanation and matches this task's own explicit instruction to stage "on the PRIMARY ... (append only; from
primary/main)" — i.e., this failure mode was already anticipated by the process, just not yet hit by name.

## Regression reconfirmation (bounded, before re-staging anything)

Before re-queuing a 6-seed GPU job, reconfirmed the ALREADY-VALIDATED reduced-scale path still reproduces
byte-for-byte on current `main` (a cheap, tiny numpy smoke — no code was touched, so this is a sanity check on the
worktree/environment, not a new result): `SIM_BACKEND=numpy -m research.runners._generative_attractor_wander_onsubstrate_derisk --seeds 42`
(non-emergent, n_ca3=400, ~150s, single small network), saved as
`research/findings/raw/_generative_attractor_wander_onsubstrate/regression_smoke_2026-08-28_seed42.json`,
reproduced `research/findings/raw/_generative_attractor_wander_onsubstrate/batch1.json`'s cited seed-42 row
EXACTLY: novelty=0.778, balance=0.639, blend_overlap_others=0.000, persistence_gap=0.000, single_recovered=0.819,
single_overlap_others=0.000, noise_best=0.000, untrained_best=0.000, genuine_formation=True. No drift since
2026-08-27.

**Byte-identical-off + flag-on-routing, reconfirmed IN THE DATA (a direct behavioral assertion, not code-reading):**
with `BRAIN_CONTINUOUS_IDEATE_SPIKING` unset, monkeypatching `_ideation_blend_settle_spiking` to raise-if-called and
driving `_ideation_wander` through a fake 3-agent organ shows **zero calls** into the spiking path (byte-identical
to the pre-existing numpy-only behavior). Flipping the flag to `1` and repeating the identical call shows the
spiking path IS entered (the monkeypatch trap fires, proving the route is live, not dead code). Both properties are
unchanged from the 2026-08-27 finding's own claims.

## What changed this session (the fix)

**Correctly re-staged the exact 6-seed `--emergent` cupy production verify on the PRIMARY queue, targeting `main`
directly** (not an agent worktree, to remove the suspected failure mode above):
```
cd /home/dant123/Projects/sim && SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._generative_attractor_wander_onsubstrate_derisk \
  --emergent --seeds 42 43 44 100 101 102 \
  --json research/findings/raw/_generative_attractor_wander_onsubstrate/\
production_n_ca3_2000_6seed.json
```
(the trailing `\` before the filename is a valid bash line-continuation — the command is copy-paste-runnable as
one line — and, same as the source finding, keeps this NOT-YET-EXISTING output path from being misread as an
already-cited artifact by the claim checker.)
Queued via `tools/queue_add.sh gpu "<cmd>" 'prior-queue-entry-lost-2026-08-27-restaging-from-main-not-a-worktree'`
(the sanctioned enqueue path; it flagged prior findings on this runner as expected and proceeded on the recorded
reason). **Persistence verified this time**, not assumed: `grep` on `/home/dant123/Projects/sim/research/queue/gpu.queue`
shows the line present immediately after the add AND again 5 seconds later (rules out an add/pop race), at queue
depth 15. `tools/gpu_queue.sh status` confirms the shared singleton dispatcher is alive and running (not paused),
so the queued job will genuinely be reached — currently behind ~14 prior jobs (mostly the four-day longitudinal
GPU batch), not immediate. The exact same GO gate as before governs the verdict (`run_seed`'s `main()`: genuine
formation, novelty<0.85, balance>0.35, balance-minus-other>0.10, persistence<0.20, single-cue recovered>0.50/others
<0.20, untrained<0.20) — a NO-GO or another PARTIAL is an equally honest, first-class outcome this finding does not
pre-judge.

## External grounding (lane check — 3rd finding in `continuous-life/generation` within 3 days)

Ecker et al. 2022, *eLife* 11:e71850 (doi:10.7554/eLife.71850) <!--derived--> — a network model of hippocampal area CA3 shows
sharp-wave-ripple sequence replay emerges from structured (clustered, plastic) recurrent synaptic interactions
DURING OFFLINE PERIODS, without ordered sensory drive. This supports the SAME framing the on-substrate wander's
blended-cue dendritic-dAP completion already rests on (novelty from DYNAMICS/structured recurrent interaction, not
from re-presenting a stored item) — an independent, external confirmation of the mechanism CLASS, not a new lever.
Recorded via `tools/record_external_search.sh`, lane-tagged `continuous-life/generation`.

## Honesty boundary / declared scope

**No code was touched this session** — `webapp/continuous_engine.py` and the runner are byte-identical to
`a4e2a015e`. `BRAIN_CONTINUOUS_IDEATE_SPIKING` remains default-OFF; no `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`
row was edited. The production-scale (n_ca3=2000, emergent) verdict is **STILL NOT LANDED** — this finding fixes
the OPERATIONAL gap (a lost queue entry) and reconfirms the pre-existing claims still hold; it does not itself
constitute new evidence for or against the production-scale mechanism. The `research/queue/gpu.queue` append lives
on the PRIMARY checkout's working tree (uncommitted, per the existing periodic-sync convention — e.g. `bb9a22cf2`
"queue(gpu): bump..."); this session's own git commit (this finding only) does not include that file, since it is
outside this session's worktree/branch. Whoever next syncs the board should commit that queue-file state as part of
the routine sync, and harvest `production_n_ca3_2000_6seed.json` + write the actual verdict once the dispatcher
reaches it.
