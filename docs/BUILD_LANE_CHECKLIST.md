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
- **Remote runs need the corpus** (`load_bearing_fraction` refuses without it; provisioners ship it).
- **GPU-queue lines start with `export XDG_RUNTIME_DIR=/run/user/1000;`** when the runner self-checks memcap.
- **Never commit live `research/queue/*` state** — merging it re-injects jobs.
- **Never SendMessage a Workflow subagent** (it resumes a duplicate in the same worktree).
