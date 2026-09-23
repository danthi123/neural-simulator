---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-23
mechanism: harvest read of the v2 pre-registered gate (paired per-seed difference in LEARNED_spkwta_held
  against a flat control AND a width-matched flat null) for conjunctive S2.5 binding (fixed-random pairwise,
  and competitive-selection) AT the satdiv capability-GO operating point (--s2-norm satdiv --s2-satdiv-sigma 8
  --s2-satdiv-scale 760 --ridge 1.0 --n-glimpses 6 --heldout-position --scramble-null). No new mechanism,
  runner, or sim/ edit in this finding -- three of the four arms (control/none, fixed, competitive) were
  already run and committed on research/perception-lane-next (HEAD 7aafbf22da246eeaf4b8db6302629f59220191ad,
  NOT merged here); the 4th arm (widthctrl_n1152, the decisive one) was NOT on that branch at 7aafbf22d --
  `tools/pool_sync.sh` pulled it fresh from pool42 during this harvest. This finding is the first read of the
  4th (decisive) arm against the bands.
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: BINDING-NEGATIVE(fixed); BINDING-POSITIVE-UNCONFIRMED(competitive) -- exhaustive, mutually-exclusive
  per arm, per the v2 gate. Neither arm reaches a confirmed BINDING-POSITIVE.
artifacts:
  - research/findings/raw/lanes/perception/conjbind_none_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (control, arm 1)
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (width-matched null, arm 2, decisive)
  - research/findings/raw/lanes/perception/conjbind_fixed_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (arm 3)
  - research/findings/raw/lanes/perception/conjbind_competitive_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json (arm 4)
  - all four *.prov.json sidecars (committed alongside)
external: NO-EXTERNAL-NEEDED -- this is a harvest read of already-run, already-committed pool artifacts
  against an existing pre-registration; no new claim requiring literature support.
builds_on:
  - research/findings/2026-09-23-vision-configural-binding-at-satdiv-GO-operating-point-PREREGISTERED.md
    (on research/perception-lane-next, HEAD 7aafbf22d -- NOT merged to main; read via `git show`, not copied,
    since the pre-registration itself is lane-branch property this harvest does not alter)
  - re-review of that branch (verdict: fix-required, prior_issues_resolved: false) -- issues 4 and 5 below
    are the ones this finding must resolve before reading arm 2, per this harvest task's own instruction
---

# Perception CPU-lane harvest: reading the satdiv-GO 4-arm binding gate

**Artifacts read, all four landed (three already committed on `research/perception-lane-next` HEAD `7aafbf22d`,
NOT merged here; the fourth pulled fresh by this harvest, see below):**
`research/findings/raw/lanes/perception/conjbind_none_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json`
(control, on the lane branch), `research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json`
(width-matched null, **NOT on the lane branch at `7aafbf22d` — pulled by `tools/pool_sync.sh` during this
harvest from pool42**), `research/findings/raw/lanes/perception/conjbind_fixed_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json`
(fixed-random binding, on the lane branch), `research/findings/raw/lanes/perception/conjbind_competitive_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json`
(competitive-selection binding, on the lane branch).

## What this is and is not

This is a **harvest**, not a new build. `research/perception-lane-next` (HEAD `7aafbf22d`) ran a fix round
that staged 4 arms of the pre-registered v2 gate; its own re-review (`wf_69a14678-7c1`, journal line 6) came
back **fix-required**, `prior_issues_resolved: false`, over five issues. Two of those issues are structural
defects in the gate itself that had to be fixed **before reading the decisive width-matched arm** — which,
per this harvest's own instructions, is done here, now, because that arm had not yet been read by anyone
when this session started `bash tools/pool_sync.sh` and found it landed.

## Status of the width-matched null (arm 2) at harvest time

`tools/pool_sync.sh` (this session) pulled
`conjbind_widthctrl_n1152_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json` from `pool42` (revision
`f10e13f33b`). Its provenance shows all 6 seeds completed cleanly (`elapsed_seconds: 827.8`, `git_dirty:
false`). **This is the first time this arm's numbers have been read anywhere** — the fix-round build agent
staged it and returned before it finished; the re-reviewer confirmed via `ssh pool42 ps aux` that it was
still running (PID 3238771) and explicitly deferred any verdict. So per the harvest task's instruction ("fix
the bands BEFORE reading the width-matched arm if not already done"), the fix comes first, below, then the
numbers.

## Two post-hoc, non-material amendments (written in this same commit, alongside reading all four arms)

**Neither amendment is a pre-registered fix made in a prior commit before arm 2 was read** — both are written
here, in the same commit that reads all four arms, so the "fixed BEFORE reading arm 2" ordering this harvest
originally claimed cannot actually be checked from the commit history. They are recorded as **post-hoc**
amendments. Neither one is material: re-deriving tau from the clean control reproduces the existing value
exactly (0.04, unchanged), and the precedence fix does not move any arm into or out of the overlap zone it
disambiguates (fixed lands at `mean_d_width = -0.0469`, well clear of the zone; competitive at `+0.2257`, also
clear). Both are declared here for the record, not credited as having governed how arm 2 was read.

**Re-review issue 5 (tau must be locked from the clean control before arms 3/4 are read, or a dated
amendment is required):** the clean-provenance control's own per-seed `LEARNED_spkwta_held`
(`[0.4792, 0.5208, 0.4896, 0.4792, 0.5104, 0.3854]`) is **byte-identical** to the originally-banked control
the placeholder tau was derived from (confirmed by the re-reviewer via exact compare, reconfirmed here by
direct read of both arrays). Sample SD = 0.0481, SE = 0.0197, 2×SE = 0.0393. Re-deriving tau from this
now-clean control therefore reproduces the same value: **tau = 0.04, unchanged**. This is the amendment: tau
is locked at 0.04 on the clean-provenance control, dated 2026-09-23, **declared here post-hoc, in the same
commit that reads all four arms** (not in a prior commit before arm 2, as an earlier version of this finding
claimed). (The lane branch's own fix round read arms 3/4 before the control had landed, which the re-review
correctly flagged as a broken ordering on that branch — this harvest does not inherit that violation because
it reads all four arms together, after all four existed, but its own tau-lock is likewise not staged ahead of
the read it governs.)

**Re-review issue 4 (bands are not mutually exclusive: `mean_d_width in [-tau, 0]` plus a `capability_go`
regression satisfies both NEUTRAL and the second NEGATIVE clause):** the precedence rule fixed here is
**BINDING-NEGATIVE is evaluated first** (either its OR-clause: `mean_d_width < -tau`, or `capability_go`
regression AND `mean_d_width <= 0`); only if neither NEGATIVE clause fires is NEUTRAL or POSITIVE considered.
This does not change any number below — it only removes the ambiguity for future arms that might land in the
overlap zone.

## The four arms, `LEARNED_spkwta_held`, per seed

| seed | control (arm1) | widthctrl (arm2) | fixed (arm3) | competitive (arm4) |
|---|---|---|---|---|
| 42  | 0.4792 | 0.3854 | 0.3333 | 0.5208 |
| 43  | 0.5208 | 0.2917 | 0.3021 | 0.6354 |
| 44  | 0.4896 | 0.3958 | 0.2917 | 0.5208 |
| 100 | 0.4792 | 0.2917 | 0.2604 | 0.4792 |
| 101 | 0.5104 | 0.3438 | 0.3125 | 0.6667 |
| 102 | 0.3854 | 0.3750 | 0.3021 | 0.6146 |
| **mean** | **0.4774** | **0.3472** | **0.3004** | **0.5729** |
| `capability_go` (n/6) | 5 | 0 | 0 | 4 |
| `scramble_null_pass` | 1.0 | 1.0 | 1.0 | 1.0 |

`scramble_null_pass = 1.0` on every seed of every arm — the anti-cheat precondition for trusting any of the
bands holds throughout.

**Width alone (widthctrl − control), reported descriptively, not gated:** mean **−0.1302**
(`[-0.0938, -0.2291, -0.0938, -0.1875, -0.1666, -0.0104]`). Going from a 96-unit flat pool to a 1152-unit flat
pool with **no** binding structure *hurts* held-out decode by 13 points on average — more units alone do not
help this task; if anything they hurt (plausibly overfitting/interference on the held-out split with 12× the
free readout weights and no binding-imposed structure).

## Reading the decisive metric: `mean_d_width`

| arm | `mean_d_ctrl` | `mean_d_width` (decisive) | `capability_go` vs control (5) | verdict |
|---|---|---|---|---|
| fixed | **−0.1771** | **−0.0469** | 0/6 (regresses) | **BINDING-NEGATIVE** |
| competitive | **+0.0955** | **+0.2257** | 4/6 (regresses) | **BINDING-POSITIVE-UNCONFIRMED** |

**Fixed-random pairwise binding is unambiguously BINDING-NEGATIVE.** `mean_d_width = -0.0469` is below `-tau`
(`-0.04`) by itself (first OR-clause fires cleanly, no precedence ambiguity here), and separately
`capability_go` collapses from 5/6 to 0/6 while `mean_d_width <= 0` (second clause also fires). Binding via a
fixed-random pairwise scheme actively hurts held-out decode relative to an equal-width flat pool, not just
relative to the narrower control. This reinforces
`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §2.1's own named fallback ("retire STDP V2/IT and
standardize on the validated V1→pooler codon") with a second, independent operating point — for this specific
binding scheme.

**Competitive-selection binding is BINDING-POSITIVE-UNCONFIRMED, not a confirmed POSITIVE.** Its
`mean_d_width = +0.2257` clears `+tau` by more than 5×, and `mean_d_ctrl = +0.0955` also clears `+tau` — both
halves of the POSITIVE test's primary AND-condition pass, decisively. But the gate's own secondary
requirement — "per-seed `capability_go` for X does not regress below the control's count" — **fails**:
control passes `capability_go` on 5/6 seeds, competitive on only 4/6 (seeds 42 and 101 fail `capability_go`
despite `LEARNED_spkwta_held` of 0.5208 and 0.6667 respectively, both *above* the control's own mean). This is
exactly the scenario the gate's `POSITIVE-UNCONFIRMED` sub-case exists to catch — an arm that wins decisively
on the continuous metric while under-performing on the binary per-seed threshold count — and per the
pre-registration this is reported as **UNCONFIRMED, treated as NEUTRAL for the roadmap fallback question**,
not silently rounded up to POSITIVE. It needs the adversarial re-verify the gate itself requires before any
claim, and remains a live, open lead — not closed either way.

**Secondary diagnostic (`RATE_lin_ceiling_held`, reported, not gating):** fixed `mean_d_ctrl = -0.1354`,
`mean_d_width = +0.0642`; competitive `mean_d_ctrl = -0.0261`, `mean_d_width = +0.1736`. The host-ridge
ceiling tracks the same qualitative pattern as the spiking readout for both arms, so this is not a
spiking-vs-rate divergence — the spiking WTA readout genuinely tracks what the underlying rate code offers.

## Correcting two process claims on the lane branch (context, no edit made here) -- CORRECTED after re-review

The re-review (issue 1) found that the fix round's "silently DROPPED from the queue" diagnosis, written into
both the pre-registration's AMENDMENT LOG and a new `research/FAILURE_LOG.md` row, is **wrong**: the fixer
checked its own worktree's git-tracked copy of `research/queue/pool.queue`, but the live queue that
`pool_queue.sh`/`pool_autodispatch.sh` actually read is hardcoded to
`/home/dant123/Projects/sim/research/queue/pool.queue` in the main checkout — a different file. The original
jobs were never dropped; they dispatched on time (09:25:27 and 09:27:12) and produced the committed
artifacts. **The previous version of this section had the two runs backwards, and misdescribed this harvest's
own artifact-copying as read-only when it was not — both are corrected here.**

The `research/perception-lane-next` branch's own committed `conjbind_fixed_...json.prov.json` carries
`run_id: 1790169928-3439243` (13:25:28 UTC = **09:25:28 EDT**) — that IS the original, first `pool41` dispatch,
and it is the one the lane branch has always held. This harvest's own earlier working copy carried a
*different* `run_id: 1790171996-3244110` (13:59:56 UTC = **09:59:56 EDT**, `elapsed_seconds: 1017.2` vs. the
original's `841.7`) — that second file was written when `tools/pool_sync.sh` pulled a duplicate re-dispatch
from pool42 and **overwrote** this worktree's copy of the lane branch's file with it, which also means this
harvest's earlier claim to have copied all four arms' artifacts "read-only and unmodified" was not true for
this one file. **Fixed here:** the lane branch's original artifact and `.prov.json` (run_id `1790169928-3439243`,
09:25:28 EDT, `elapsed_seconds: 841.7`) are restored into this harvest's committed copy; the per-seed
`LEARNED_spkwta_held` values are identical between the two runs (only `elapsed_seconds` and provenance differ),
so no verdict or number in this finding changes. The duplicate dispatch itself burned **roughly 17 minutes**
of pool compute (`elapsed_seconds: 1017.2`, not the "~14 minutes" this section previously estimated — that
number conflated the gap between the two dispatch *start times* with the actual wasted run length). The
correction to file on the lane branch (a worktree agent reading a stale tracked copy of shared queue state,
not a `flock` race) is unaffected by this fix. As of this harvest, the live queue
(`/home/dant123/Projects/sim/research/queue/pool.queue`) is confirmed empty — no further duplicate of any of
these four jobs is pending. This harvest does not edit the lane branch's docs; the correction is recorded
here for whoever next touches that branch.

## Bottom line

Read against the fixed v2 bands: **fixed-random pairwise binding is BINDING-NEGATIVE** (a second, independent
operating point on top of 09-17's own diagnosis that this task's signal is fine distributed cosine modulation
a hard conjunction bank is the wrong tool for). **Competitive-selection binding is
BINDING-POSITIVE-UNCONFIRMED** — a real, large lead on the decisive width-isolated metric that the gate's own
secondary check (a `capability_go` regression on 2 of 6 seeds) keeps from being called a confirmed win. Per
the gate's own text, this needs an adversarial re-verify before any "surpass" claim, and per
`docs/TERMS.md` is reported as PARTIAL, not GO.
