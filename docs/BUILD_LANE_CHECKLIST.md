# Build-lane checklist — what adversarial review keeps rejecting

Paste-by-reference into every build-agent prompt (`read docs/BUILD_LANE_CHECKLIST.md before building`). Earned on
2026-09-23: 12 build lanes in one day, and **every first-round build came back fix-required** from its adversarial
reviewer — almost always for one of the classes below. Each extra fix round costs ~1-2M agent tokens and hours of
wall-clock; catching these in round one is the cheapest parallelism there is.

## Before building
- **Verify-first, including UNTRACKED artifacts.** `bash tools/before_you_build.sh "<defect>"`, `git log --all --grep`,
  AND `ls research/findings/raw | grep <kw>` — a banked 4/6 NOT-GO sat untracked and a lane re-ran it as "never run".
- **The target may already be shipped.** Check the ledger and the load-bearing battery's faculty list.

## The gate (the #1 rejection class)
- **Before writing code, write one line per evidence gate: "this gate FAILS if <realistic outcome>".** If the only
  input to the measured reply is the thing the lesion disables, or the lesion arm cannot reach the same route, or a
  field is set by a host template of the arm label, there is no realistic failing outcome — redesign the probe before
  building it. (2026-09-23: 5 of 6 first-round builds were rejected for exactly this — hold-query read-out,
  swap lead template, DA lesion pinned to tonic, curiosity G4/G5, comparator G3.)
- **It must be able to fail.** If a check passes by construction (the only drive into a pool is the edge you lesion;
  a relay passes a sign it was given; a 'no lost detection' check at a strength that already detects everything),
  it is an INTEGRITY SMOKE — label it so, never count it as evidence.
- **UNDEFINED is never a pass.** `x is None or x > t` is a bug; use `tools.lab.undefined_if_empty`.
- **Nulls are distributions,** not one fixed shuffle (a single permutation fails/passes by construction for n≈7).
- **The unit of replication is the seed or an independent session** — serially dependent samples in one session are
  not exchangeable; p-values over them are anti-conservative.
- **Lesion the SPECIFIC claimed edge,** not the whole organ; hold everything else byte-identical.
- **Contrast gates test the contrast:** if an ON arm is credited, the OFF control must be required NOT to show it.
- **Constants are not fit on the evaluation seeds** — freeze from other seeds/prior commits, or declare in-sample
  and add held-out seeds.
- **Report the mechanism's own state, not an arg-max of the evaluation metric** over a sweep (that is tuning).
- **Pre-register in its own commit BEFORE any run it governs;** amendments get an AMENDMENT LOG naming what had been seen.
- **Do not stage a run whose pre-registration already predicts failure** — fix the design first.

## Identity and honesty
- **Byte-identical off is ASSERTED IN DATA** by exact compare (== / sha256) against a PINNED pre-change SHA — not
  `allclose`, not "the code path looks unchanged", not a moving ref like `origin/main` (tautological after merge).
- **Declare every host shortcut** (argmax/argsort over spike rates, host population codes, host PPMI weights, host
  decay/capture rules, host-wired teacher pathways). A host rule that decides the outcome is not credited to the brain.
- **Terminology per docs/TERMS.md** — "robust core" is not grown by a default-off flag; say "load-bearing under the
  adequate probe with an opt-in flag" (Option-C pair).

## Compute and repo hygiene
- **Each arm/variant gets its own `--out` directory** — the battery's intermediate files are named by probe group.
- **Every LOCAL python you start — including throwaway exploration scripts — runs under `bash tools/memcap.sh <GB> --`**
  after `tools/mem_ok.sh <GB>` passes. The box is shared with other lanes, the GPU queue and the owner's desktop
  (2026-09-23: two uncapped 8.5 GB exploration scripts left 3 GB free next to a GPU-queue job; one was killed).
- **Remote runs need the corpus** (`load_bearing_fraction` refuses without it; provisioners ship it).
- **GPU-queue lines start with `export XDG_RUNTIME_DIR=/run/user/1000;`** when the runner self-checks memcap.
- **Never commit live `research/queue/*` state** — merging it re-injects jobs.
- **"Merge main" means `git merge origin/main`: a TWO-parent commit.** Copying main's files into a one-parent commit
  leaves the merge-base stale, so the branch conflicts with main again (2026-09-23 curiosity fix round). Before
  pushing: `git merge-tree --write-tree origin/main HEAD` must exit 0.
- **Never SendMessage a Workflow subagent** (it resumes a duplicate in the same worktree).
- **A repo-wide idle-compute gate blocking YOUR commit is the orchestrator's problem, not yours.** Do NOT queue filler
  (new seed batches of a closed result, a spare seed "for replication", an "autofill" continuation) to clear
  `compute_idle_persistent` / `lane_starvation`, and do not write a waiver. Leave the work committed-staged in your
  worktree, and return with `commit_blocked_by: <gate>`; the orchestrator stages real work or waives with evidence.
  (2026-09-23: a scoring agent queued three such jobs; they were removed before they took pool RAM from D6 arms.)
- **Pool lines declare their memory:** put `mem_gb=<measured peak RSS, rounded up>` in the `--checked` text. The
  dispatcher reserves it per node (15 GB nodes; a missing hint reserves 1 GB and invites an OOM burst).
