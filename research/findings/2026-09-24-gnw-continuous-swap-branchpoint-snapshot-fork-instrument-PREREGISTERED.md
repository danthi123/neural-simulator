---
type: preregistration
status: preregistered
date: 2026-09-24
mechanism: A GENUINE branch-point snapshot/fork/compare instrument for the GNW continuous-swap lane
  (`research/runners/_gnw_continuous_branchpoint_verify.py`), built ONE substrate, driven ONCE to the
  establish-A -> evict/admit-B branch point, then FORCE-forked onto two independently-built substrates via
  the existing, already-shipped `_full_snapshot`/`_full_restore` utility
  (`research.runners._p1_2_workspace_deliberation_loop_derisk`) plus a captured STD host-state snapshot and
  the interpreter-global RNG state (OU noise draws from it every step; not itself `cp_`-prefixed).
lane: GNW / global workspace continuous cross-turn ignition (PARKED 2026-09-23, docs/.vikunja_sync
  "GNW continuous-thought-swap lane PARKED after 3 review rounds")
seeds: [42, 43, 44, 100, 101, 102]
verdict: PREREGISTERED -- no run has happened yet under this file. Seed 42 will run LOCALLY under
  `tools/memcap.sh`/`tools/mem_ok.sh` immediately after this commit; seeds 43/44/100/101/102 are staged on
  the pool (pool41 + pool42, isolated revision), not awaited synchronously.
artifacts: []
external: Mongillo, Barak & Tsodyks (2008), Science 319:1543 -- unchanged citation for the Tsodyks-Markram
  STD eviction effector this instrument verifies the FORK of (no new biological claim; see "what this does
  not claim" below).
builds_on:
  - the parked branch `origin/research/gnw-thought-swap-drive` (never merged to main; its own file
    `research/runners/_gnw_swap_continuous_recency_derisk.py` records TWO retracted fix rounds inside its
    OWN fork-control hash, `_std_state_hash` -- read directly off that file for the full retraction history)
  - docs/.vikunja_sync, 2026-09-23 entry "GNW continuous-thought-swap lane PARKED after 3 review rounds":
    "next attempt (when resumed) verifies against a genuine branch-point snapshot of every cp_* array" --
    this file is that attempt.
review_corrections_applied:
  - "round 2 (2026-09-24, journal wf_16244c9d-ca5, agentId a6dd960fd33710903, verdict fix-required): (1) G2 and
    G4 (and G1) relabelled INTEGRITY SMOKE -- each passes by construction on this substrate/backend and has no
    realistic failing outcome today; G3 is now stated as the only discriminating gate. (2) added the
    pre-registered NEGATIVE CONTROL the review named: G3-neg runs the S2-equivalent fork (S4) WITHOUT reloading
    the saved RNG state and requires the post-run hash to MISMATCH S1's -- without it G3 passing was never shown
    to mean anything. (3) committed the probe that produced the '32/32 live cp_* arrays' and 'CSR .copy()
    independent' preconditions (`scratchpad/probe_cp_types.py`) together with its provenance rows, instead of
    citing an uncommitted script. (4) the runner docstring's `seed_go` formula and 'gated ONLY on G1-G4' text
    now match the code (which already AND-ed in G5); `tools.lab.lever` for G5 now uses `required=False` so an
    unmoved lever records a NO-GO for that seed instead of raising and aborting the whole --six-seed aggregate
    with no artifact. Also added, same round: a `SIM_BACKEND=numpy` assertion in `main()` (the RNG capture is
    only the stream OU noise reads from under that backend), and a pool-aggregation note (the five pool-staged
    seeds each write their own `--smoke` JSON; `pooled_go` is assembled by the orchestrator from all 6 files,
    not produced by any single run)."
---

# GNW continuous-swap branch-point snapshot/fork instrument: pre-registration (before any run)

**This is a pre-registration only**, committed before the first run this file governs
(`tools/gates` prereg-before-run discipline). It fixes the mechanism, the exact instrument, the gates that
must hold, and what this runner does NOT claim, before seed 42 (local) or any pool-staged seed is read for
verdict purposes.

## Why this is the genuine next rung, not a re-derivation of the parked lane

`bash tools/before_you_build.sh "GNW continuous branch point snapshot fork verification"` (this session)
surfaced no prior attempt at a full `cp_*`+STD+RNG branch-point fork in this lane or any other; the closest
hits are the shipped `#77`/`#85` swap mechanism itself (unchanged, reused-by-import here) and the parked
branch's own retracted fix rounds. `git log --all --grep` for `branch.*snapshot`/`cp_.*fork` before this
session returns nothing that builds the instrument this file builds.

**The wall question, asked first.** What does the real system run alongside "compare two branches of a
continuous substrate" that the parked lane replaced with a constant? Answer, read directly off the parked
branch's own code (not hypothesized): it replaced a GENUINE FORK with **two independently-built substrates
at the same seed**, and replaced "verify they are the same state" with **one narrow array-family hash**
(`std.deps[*].x` only) taken, in its first two fix rounds, at the WRONG point in the timeline (after the two
arms had already diverged, or — in its second fix round — before divergence but still covering only that one
array family). Neither ever asked whether the two "independent" substrates were IN FACT the same state, only
whether one small READOUT of them matched. That is the missing companion process: a real fork needs an actual
COPY operation forcing two substrates into one state, and a verification that spans everything the substrate's
own future depends on -- not a hope, however seeded, that a rebuild reproduces it.

## The instrument

Build substrate S1 once; drive it, continuously, to the branch point (establish A, evict A / admit B). At
that point, capture:
  - `_full_snapshot(S1.bridge)` -- EVERY `cp_*` array on the live bridge (the existing, already-shipped
    generic utility; reused unchanged, no `sim/` edit).
  - the STD host state (`std.deps[k].x`, `.boost` per loop) -- NOT `cp_`-prefixed, but load-bearing: it is
    what `RecurrenceDepression.apply()` writes INTO `cp_connections.data` every step.
  - `np.random.get_state()` -- OU noise (`cfg.enable_ou_process`) draws `cp.random.randn(n)` every step from
    the INTERPRETER-GLOBAL RNG stream, not a per-object generator; two forks with identical `cp_*`+STD state
    still diverge the instant either takes a noisy step unless this is captured and replayed too.

FORK: build S2 and S3 fresh (same seed, same architecture -- matching shapes are all `_full_restore` needs)
and FORCE-RESTORE the captured branch point onto each. S1 is untouched by this and is itself the third leg.

## Gates (each with its realistic failing outcome, `docs/BUILD_LANE_CHECKLIST.md`)

Round-2 review found three of the five original gates pass BY CONSTRUCTION on this substrate/backend and were
being counted as evidence anyway (`docs/BUILD_LANE_CHECKLIST.md`: "if a check passes by construction ... it
is an INTEGRITY SMOKE -- label it so, never count it as evidence"). They are labelled as such below. **G3 is
the only gate that can actually tell a good fork from a bad one**, and it is now paired with a pre-registered
NEGATIVE CONTROL (G3-neg) so a G3 pass is evidence rather than an artifact of a control never exercised in its
failing direction.

- **G1 `cp_completeness`** [INTEGRITY SMOKE] -- measures whether any non-`None` `cp_*` attribute on the built
  bridge is absent from `_full_snapshot`'s output. Measured empirically before writing the runner
  (`scratchpad/probe_cp_types.py`, committed alongside this file with its provenance): 32 live (non-`None`)
  `cp_*` arrays on the substrate `build()` produces TODAY, all captured; the other 126 known `cp_*` attribute
  names on the bridge class read `None` because their subsystems are off. Because `build()` is unchanged by
  this instrument, this cannot show a different number on THIS pre-registration -- it is a regression check
  against a FUTURE edit to `build()` (e.g. turning on STDP/BTSP/HH channels), not live evidence today.
- **G2 `fork_restore_exact_s2` / `_s3`** [INTEGRITY SMOKE] -- re-snapshots S2 (resp. S3) immediately after
  `_full_restore` writes the captured branch point onto it, using the SAME keys and filter that produced the
  source hash. `_full_restore` assigns `getattr(bridge, k)[:] = arr` (or an in-place CSR restore) per key;
  reading the same keys back with the same filter reproduces exactly what was just written, and a genuine
  shape/key mismatch would raise an exception inside `_full_restore` itself, not surface as a hash mismatch
  here. There is no code path under which the restore silently writes something different and G2 still
  returns True, so a match is not evidence the fork is scientifically correct -- only that the assignment ran
  without raising. That claim rests on G3.
- **G3 `control_fork_determinism`** [THE DISCRIMINATING GATE] -- FAILS if, from the two forks (S1 and S2, each
  only G2-consistent, not yet shown scientifically identical), loading the SAME saved RNG state before each
  and running the IDENTICAL control action (propose A again) on both does not reproduce a byte-identical
  post-run full-state hash AND identical verdict fields. This is the one gate here with a demonstrated failing
  direction (G3-neg, next) -- a pass is the genuine claim "S1 and S2 are, operationally, the same branch."
- **G3-neg negative control for G3** [REQUIRED -- proves G3 is not vacuous] -- the SAME control action, from
  the SAME restored branch point, on a FOURTH fork (S4), but WITHOUT reloading the saved RNG state first (OU
  noise draws `cp.random.randn(n)` every step from the interpreter-global stream, already advanced past the
  branch point by S1's and S2's own control runs). `g3_negative_control_mismatches` must be True -- the
  un-RNG-matched run must DIVERGE from S1's post-run hash -- or a G3 pass on that seed is not evidence the RNG
  capture is load-bearing, and `seed_go` is False regardless of G3's own result.
- **G4 `no_aliasing`** [INTEGRITY SMOKE on `SIM_BACKEND=numpy`] -- checks the STORED branch-point snapshot's
  own hash is unchanged after S1 and S2 are each driven forward past the branch point: the direct regression
  test for the parked lane's 1st-fix-round bug class. On the numpy backend it cannot fail: `numpy.ndarray.copy()`
  and `scipy.sparse.csr_matrix.copy()` are always genuine deep copies, verified standalone in
  `scratchpad/probe_cp_types.py` (mutate-after-copy leaves the copy untouched) -- so `_full_snapshot`'s
  `.copy()` cannot alias on this backend. Kept as an executable regression check, but not counted as evidence
  today.
- **G5 `carryover_ok`** -- FAILS if `x_A` at the branch point is not measurably below 1.0 (the STD lever that
  the branch point is supposed to exercise never actually moved). Uses `tools.lab.lever` with
  `required=False`: an unmoved lever is a real, reportable per-seed NO-GO, not a process crash that aborts the
  whole `--six-seed` aggregate with no artifact (the previous `required=True` draft would have raised
  `LeverError` inside the six-seed list comprehension the instant one seed's `x_A` landed at exactly 1.0).

`seed_go` = G1 AND G2(S2) AND G2(S3) AND G3 AND G3-neg AND G4 AND G5, all True. `pooled_go` = `seed_go` on 6/6
seeds. Only G3, G3-neg and G5 have a realistic failing outcome on this substrate/backend today; G1, G2 and G4
are executable regression checks against future edits, gated but not evidence on their own, and must not be
cited as "the fork was verified" by themselves.

**Backend assertion:** the RNG capture (`np.random.get_state()`/`np.random.set_state()`) only actually
controls the stream OU noise draws from when `cp is np`, i.e. `SIM_BACKEND=numpy`. Under CuPy, OU noise draws
from a separate `cp.random` stream this capture never touches, so G3/G3-neg would read as false
failures/passes for a mis-configuration, not the instrument. `main()` now asserts the backend and exits loudly
rather than letting that happen silently.

**Pool aggregation:** the five pool-staged seeds each run their own `--smoke` invocation and write their own
per-seed JSON -- there is no `--six-seed` process on the pool holding all 6 seeds at once. `pooled_go` for the
full battery is therefore assembled AFTER all 6 per-seed JSON files exist (1 local + 5 pool), by applying the
same `n_go == 6` / `Verdict.require` logic `run_six_seed` applies in-process; this is the orchestrator's job
when collecting the pool artifacts, not something any single run of this file produces.

## What this instrument does NOT claim

This is a verification-instrument finding, not a mechanism finding. It does not re-open
`carryover_causal_at_near_threshold` (still banked NO-GO on the parked, never-merged branch: lesioning ONLY
the STD carryover never rescues the near-threshold swap decision on any of 6 seeds) and does not assert the
`BRAIN_GNW_SWAP_CONTINUOUS` flag is production-load-bearing. `recent`/`fresh` swap verdicts at the shipped
safety operating point (`SALIENT_PA`) are reported for continuity with the parked lane's own safety claim,
but `seed_go`/`pooled_go` gate on G1 through G5 PLUS the G3-neg negative control (the instrument itself). If
they hold 6/6, the honest claim this finding can make is narrow and mechanical: *"a full `cp_*`+STD+RNG
snapshot taken at this branch point, forked onto two independent substrates, is verifiably the same state,
running the same action from it is verifiably deterministic, AND a control that skips the RNG replay
verifiably diverges"* -- i.e. the fork the parked lane's science depended on is now genuine and shown to
discriminate, not merely asserted or checked in one direction. Whether that verified fork then supports ANY
mechanism/recency claim is explicitly left to a SEPARATE, later finding (per "isolate a genuinely different
method" and "do not stage a run whose pre-registration already predicts failure" -- this file predicts
nothing about the mechanism, only about the instrument).

## Compute plan

Seed 42: LOCAL, under `bash tools/memcap.sh <measured_gb> --` after `tools/mem_ok.sh` passes (this substrate
is a ~960-neuron workspace; the exploratory probe run before writing this file peaked well under 1 GB RSS at
`--smoke`, so 2 GB is used as the memcap ceiling with margin). Seeds 43/44/100/101/102: staged on the pool
(`tools/pool_queue.sh add`, pool41 + pool42, `--checked` line naming the measured mem_gb and the verify-first
note above), not awaited synchronously in this session. No GPU queue job: this runner is CPU/numpy-only by
the SAME design as the substrate it reuses (`SIM_BACKEND=numpy`, small workspace) -- there is no GPU-bound
step to route through `tools/gpu_queue.sh`.
