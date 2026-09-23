---
type: finding
status: complete
date: 2026-09-23
mechanism: continuous (no-restore) cross-turn ignition for the GNW held-topic swap workspace
  (webapp/gnw_thought_swap.py::ThoughtSwapWorkspace.observe, de-risk
  research/runners/_gnw_swap_continuous_recency_derisk.py) -- board #77/#85's own named next rung.
lane: laneC (GNW / integration-to-production)
verdict: GO on SAFETY ONLY (6/6 seeds, now including a REAL restore-mode comparison arm and hold/lesion/LRU-slot-
  reuse turns). The recency/mechanism claim this finding originally made is RETRACTED and BANKED AS NO-GO (see
  "2026-09-23 FIX ROUND" below) -- refuted by the runner's own neural lesion arm, 6/6 seeds.
artifacts:
  - research/runners/_gnw_swap_continuous_recency_derisk.py
  - research/findings/raw/_gnw_swap_continuous_recency_smoke.json
  - research/findings/raw/_gnw_swap_continuous_recency_6seed.json
  - webapp/gnw_thought_swap.py (continuous_enabled() + the isolate= thread-through, additive production glue)
  - tests/test_gnw_swap_continuous_byte_identical.py (byte-identity regression test, new this fix round)
verification: |
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_swap_continuous_recency_derisk --six-seed \
    --json research/findings/raw/_gnw_swap_continuous_recency_6seed.json
  -> verdict=GO seed_go 6/6, swap 6/6, branch 6/6, branch_hash 6/6 (now GATED, correctly timed -- see "2ND FIX
     ROUND"), carry 6/6, blind 6/6, no_regression 6/6, full_conv 6/6, det 6/6
     [near-threshold dissociation 1/6, carryover_causal 0/6 -- RETRACTED as a mechanism, NOT gating]
  .venv/bin/python -m pytest tests/test_gnw_swap_continuous_byte_identical.py -q  -> 4 passed (now pinned to a fixed
    pre-branch SHA, not a moving merge-base -- see "2ND FIX ROUND")
  Ran LOCALLY (CPU, ~3min total for the six-seed sweep after this fix round's extra arms, a 960-neuron toy substrate
  -- "unit test / tiny smoke", not a full-brain build; tools/mem_ok.sh not required per CLAUDE.md's own compute-lane
  rules). Pool cross-machine reproduction remains STAGED, NOT queued (see "Compute" below) -- an existing tooling
  gap (`research/FAILURE_LOG.md` 2026-09-23), not attempted again this round.
---

# GNW swap CONTINUOUS cross-turn ignition: SAFE to enable (6/6 seeds) -- the recency-trace mechanism claim is RETRACTED

**One line.** Making the held-topic workspace's substrate genuinely continuous across turns (no per-turn restore)
does not change any TESTED swap-vs-hold verdict, now measured against a real restore-mode arm and on turn shapes
(holds, the board-#85 lesion path, LRU slot reuse) the original version of this finding never exercised. The
mechanism claim this finding originally made -- that the carried-over synaptic state is a reply-relevant "recency
trace" -- does **not** survive its own lesion test and is retracted below, not merely softened.

## 2026-09-23 FIX ROUND (adversarial review verdict=fix-required; every issue addressed)

This finding was built, reviewed, and is being corrected in the same session. The review found five defects; all
five are fixed or retracted here, not argued around:

1. **Byte-identity was FALSE for the default path.** `webapp/gnw_thought_swap.py` added an unguarded
   `"continuous": continuous_enabled()` key to `ThoughtSwapWorkspace.observe()`'s info dict. Because
   `swap_drives_chat.observe_turn` does `out = dict(info)` and the drives path is DEFAULT-ON
   (`_SWAP_DRIVES_DEFAULT_ON=True`, `webapp/server.py`), every default `/api/brain-chat` turn after the first
   gained a NEW `resp["swap_drives"]["continuous"] = False` key. **Fixed**: the key is now only added when
   `BRAIN_GNW_SWAP_CONTINUOUS` is truthy. **Verified in the data**, not inferred from reading the code:
   `tests/test_gnw_swap_continuous_byte_identical.py` hashes a 9-turn conversation (holds, swaps, the lesion path,
   LRU slot reuse) run through the branch's `ThoughtSwapWorkspace` against the SAME conversation run through
   `webapp/gnw_thought_swap.py` as it exists at `origin/main`'s merge-base, with the flag unset -- exact dict
   equality on every turn, all 4 tests pass.
2. **The no-regression gate never measured what it claimed.** It compared two CONTINUOUS arms (RECENT, FRESH) to
   each other and to a hard-coded `True` -- an `isolate=True` restore-mode arm was never run. **Fixed**: the gate
   now builds two fresh substrates and calls `run_intention_swap(..., isolate=True)` on each (`restore_recent`,
   `restore_fresh`) -- a REAL restore-mode decision for the identical proposal -- and requires the continuous arm's
   verdict AND post-window rate to match it, 6/6 seeds.
3. **Hold turns, the board-#85 lesion path, and LRU slot reuse were never tested.** The gate only ever ran swap
   turns. **Fixed**: a new check (`full_conversation_no_regression`) runs the ACTUAL production glue
   (`webapp.gnw_thought_swap.ThoughtSwapWorkspace`, not just the low-level primitive) through a 9-turn script --
   two same-topic holds, a swap back to a previously-evicted topic, a 4th and 5th topic forcing LRU slot reuse past
   `N_PATTERNS=3`, and a lesioned turn immediately followed by its un-lesioned re-proposal -- comparing
   continuous-ON to continuous-OFF on the swap-vs-hold VERDICT (not internal numeric state, which legitimately
   differs once continuity is on). Identical on every turn, 6/6 seeds.
4. **The seed-42 "existence proof" is refuted by the runner's own lesion arm.** The original version called a
   near-threshold dissociation (RECENT fails to re-ignite, FRESH succeeds) at seed 42 evidence that the STD
   carryover causes a recency-driven swap failure. It is not: forcibly resetting ONLY A's carryover back to 1.0
   immediately before re-proposing it (`nt_recent_lesioned`) does **not** rescue the swap at seed 42
   (`nt_recent_lesioned["swapped"] == nt_recent["swapped"] == False`), and this null holds on **all 6 seeds**
   (`carryover_causal_at_near_threshold` = 0/6, including the one seed with the raw dissociation). If wiping the
   carryover to a virgin state does not change the outcome, the carryover was never what blocked the swap -- the
   dissociation is a pattern-identity (A vs C) or per-seed-heterogeneity confound, not a recency effect. **This
   claim is RETRACTED, not merely downgraded to "not yet reproducible."**
5. **Terminology.** `branch_identical` compared 4 scalar read-outs and was described as "byte-identical branch
   state" -- `docs/TERMS.md` requires a hash or exact compare for that word. A genuine hash check
   (`branch_state_hash_match`, full STD-state-array SHA256) was added. `branch_identical` is now described only as
   a floating-point-tolerance match, never byte-identical. **⛔ This item's "fix" for `branch_state_hash_match` was
   ITSELF wrong and was corrected in the 2ND FIX ROUND below — see there for what actually happened.**

## 2026-09-23 2ND FIX ROUND (adversarial re-review verdict=fix-required; re-review reproduced the bug)

The re-review found the 1st round's own hash-check fix was broken, plus the byte-identity test's reference becoming
tautological after merge. Both are now fixed, verified by direct reproduction (not argued around):

1. **`branch_state_hash_match` was computed AFTER the two arms had already diverged, so it could never match.** In
   `evaluate_seed`, the hash line ran after `recent` (re-propose A on S1) and `fresh` (propose C on S2) had already
   executed, isolate=False -- i.e. AFTER S1 and S2 had taken different steps. Comparing two states that have already
   diverged is not a fork control; it is guaranteed to disagree. **The 1st round's stated root cause -- "threaded-
   BLAS floating-point non-associativity under `OMP_NUM_THREADS=2`, confirmed bit-identical under
   `OMP_NUM_THREADS=1`" -- was FALSE.** The re-review ran `evaluate_seed(42)` under `OMP/OPENBLAS/MKL_NUM_THREADS=1`
   and reproduced `branch_state_hash_match=False` there too, at the same (wrong) hash point. **Fixed**: the hash
   (`branch_hash_1`/`branch_hash_2`) is now taken immediately after `_fresh_branch` returns, BEFORE `recent`/`fresh`
   run -- the true branch point. Verified directly (both this fix and the reproduction of the original bug, in one
   script, same session): hashing at the true branch point gives IDENTICAL hashes (`f0b84eb9...`) under BOTH
   `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=2` -- matching the re-review's own manual side-experiment hash exactly --
   while hashing the SAME two builds AFTER they diverge gives two DIFFERENT hashes, regardless of thread count. This
   also directly refutes the BLAS-thread-count explanation: thread count does not move the true-branch-point hash
   at all; only the (wrong) hash *timing* did. `branch_state_hash_match` now GATES `seed_go`/`pooled_go`, 6/6.
2. **The byte-identity regression test's reference becomes tautological after merge.** `git merge-base HEAD
   origin/main` equals `HEAD` itself once this branch is merged into main, so the test would compare
   `webapp/gnw_thought_swap.py` to itself and could never fail post-merge. **Fixed**: pinned to a fixed, non-moving
   SHA (`5b718e73c`, the last commit to touch this file before this arc; verified an ancestor of both `HEAD` and
   `origin/main`) instead of a branch-relative merge-base. All 4 tests still pass against the pinned reference.
3. **`branch_identical`'s scalar comparison remains a weak, secondary check** (not the fork control) -- the docstring
   and code comments now say this explicitly; `branch_state_hash_match` is the real fork control.
4. **`no_regression_at_production_pa` mostly re-confirms `swaps_correct` at the saturating drive strength**
   (`SALIENT_PA`): every arm swaps 6/6 and every arm's post-window rate plateaus at the same 0.3333, so this check
   is weak evidence on its own. The runner's docstring, code comments, and `Verdict.require` messages now say
   explicitly that `full_conversation_no_regression` (which exercises holds/lesion/LRU-reuse, non-saturating turn
   shapes, through the real production glue) is the check that actually carries the safety weight.

**Reproduction of both bugs (and the fixes), run this session:** a script builds two substrates from the same seed,
hashes them at the true branch point (matches, `f0b84eb9...`, both thread counts) and then again AFTER letting the
arms diverge (does not match, either thread count) -- proving the divergence-timing bug directly rather than
inferring it from a comment. `tests/test_gnw_swap_continuous_byte_identical.py -q` -> 4 passed against the pinned
SHA. `research/runners/_gnw_swap_continuous_recency_derisk.py --six-seed` -> `branch_state_hash_match` 6/6, now
gated; `pooled_go` remains `True`.

Separately (not one of the review's five, but relevant to why claim 4 is retracted rather than merely
recalibrated): **the biological framing was never supported.** `STD_TAU_D=250ms`, and production advances ZERO
simulated time between HTTP turns -- a real inter-turn gap of seconds is 8-40*tau_D (full recovery), so citing
Mongillo, Barak & Tsodyks 2008 (a real-time-elapsing ~1s working-memory trace) as the substrate for a turn-to-turn
carryover was not supported even before the lesion refutation. That framing is dropped, not softened. The next
method (not attempted this round) is either (a) find the actual seed-42/102 near-threshold confound directly
(likely per-pattern Izhikevich-heterogeneity margin, not carryover), or (b) model real elapsed wall-clock time as
inter-turn free-run recovery before any recency-trace claim is re-attempted.

## Why this is the genuine next rung, not a re-derivation

`bash tools/before_you_build.sh "GNW continuous cross-turn ignition recency swap workspace no restore"` surfaced
`2026-08-19-gnw-swap-into-chat-GO.md:73` directly — its own honest-limit #3 (renumbered #1 in the current source):
*"A truly continuous cross-turn ignition (no restore) is the named next rung."* `git log --all --grep` and a scan of
`research/findings/*gnw*`/`*swap*`/`*thought-swap*` confirmed the SURROUNDING capability — making the swap DRIVE the
live reply — is already merged to `origin/main`, `DEFAULT-ON` (`_SWAP_DRIVES_DEFAULT_ON=True`,
`webapp/server.py`), and independently re-verified byte-identical against today's code by
`2026-09-05-rank11-topic-swap-scaffold-backlog-item-already-integrated.md`, which also confirms it is already a row
in `research/runners/load_bearing_fraction.py`'s `FACULTY_LESIONS` battery (`swap-drives-response`). Building that
again would duplicate shipped work.

## The mechanism (reuse-by-import; NO `sim/` edit)

`research/runners/_gnw_neural_swap_intention_derisk.py::run_intention_swap` already exposes `isolate=False` ("a
CONTINUOUS run, 0 restore calls"). Protocol per seed: establish topic A (one necessary cold-start `isolate=True`),
then swap continuously (`isolate=False`) A→B, evicting A via the same recurrence-weakening STD the shipped
mechanism already uses; A's loop is left with a depleted resource variable `x_A < 1`. Two controlled arms branch
from this point: **RECENT** re-proposes A; **FRESH** proposes C (never held, `x_C == 1`). This fix round adds two
more arms at the SAME branch point: **restore_recent**/**restore_fresh**, a genuine `isolate=True` replay of the
identical proposal, giving the no-regression check something real to compare against.

## What is robustly TRUE (the GO, 6/6 seeds, SAFETY ONLY)

1. **The carryover is real, not a label** (a raw computational fact -- see the biology note above for why this is
   NOT claimed as a validated recency trace). `x_A` at the branch point measures 0.73–0.78 across all six seeds.
2. **Restore mode is PROVABLY blind to it**: `std.reset()` (what `isolate=True` performs) wipes it to exactly 1.0
   for every pattern, every seed (`restore_blind`, 6/6).
3. **Safe against a REAL restore-mode arm.** Continuous mode's verdict matches an actual `isolate=True` run of the
   identical proposal, 6/6 (`no_regression_at_production_pa`, fixed this round).
4. **Safe on untested turn shapes too.** The production glue module, run through holds, the board-#85 lesion path,
   and LRU slot reuse, reaches the identical verdict continuous-ON vs continuous-OFF, 6/6
   (`full_conversation_no_regression`, new this round).
5. **Determinism** (build-twice Izhikevich hash) holds 6/6; the branch fork matches to floating-point tolerance
   (1e-9) on the population-level decision scalars, 6/6 (`branch_identical`, no longer called byte-identical), AND
   is now GENUINELY byte-identical at the true branch point (full-STD-state SHA256, `branch_state_hash_match`,
   taken BEFORE either arm diverges), 6/6 -- see "2ND FIX ROUND" for the correction of the earlier mis-timed hash /
   false BLAS-artifact diagnosis.

This is what makes the new `BRAIN_GNW_SWAP_CONTINUOUS` flag (default-off, additive) a low-risk, VERIFIED-safe
addition. **It ships as plumbing only — it carries no capability claim.**

## What is RETRACTED, not "not yet true" (docs/TERMS.md)

The original version of this finding reported a near-threshold dissociation at seed 42 (`--near-threshold-pa`,
1500pA) as an "existence proof" of a recency-driven swap failure. **This is refuted, not unreproduced.** The
runner's own neural lesion (`std.deps[A].x[:] = 1.0` immediately before re-proposing A) does not rescue the swap
at seed 42, and `carryover_causal_at_near_threshold` = 0/6 across all seeds, including the one with the raw
dissociation. **`swap-continuous-recency` was never added to `load_bearing_fraction.py`'s `FACULTY_LESIONS`
battery, and still is not** — there is no reproducible state→reply dependency to add. The neural lesion tool
remains in the runner for whichever future session investigates the actual seed-42/102 confound.

## Anti-cheats

- **Controlled fork:** RECENT and FRESH are built from the SAME seed through the IDENTICAL establish+A→B sequence;
  `branch_identical` (6/6) confirms the two builds match at the decision-relevant scalars before the one proposal
  that differs. The real fork control is `branch_state_hash_match` (6/6, GATED) -- a full-STD-state SHA256 taken at
  the true branch point, before either arm diverges (see "2ND FIX ROUND" for the correction of the earlier
  mis-timed hash and its false BLAS-artifact diagnosis).
- **The lever moved:** `x_A_at_branch < 1.0 - 0.05` is void-checked before anything downstream is interpreted.
- **Restore-blindness is a code-level check:** exact float equality to 1.0 (1e-12), not "close to."
- **The causal claim is REQUIRED to survive its own lesion, not merely reported alongside it** (fix #4) — this is
  the anti-cheat that caught the original overclaim.
- **No sim/ edit:** `git diff sim/` is empty.
- **Production wiring is additive, reversible, and now test-verified byte-identical when off** (fix #1).

## Compute

Ran locally (CPU, `SIM_BACKEND=numpy`, a 7-region/960-neuron toy substrate): the six-seed sweep with this round's
added restore/full-conversation arms took ~3 minutes wall. No `mem_ok.sh`/`memcap.sh`/pool routing required at this
scale. **Pool cross-machine reproduction remains staged, not queued** — `research/FAILURE_LOG.md` (2026-09-23)
records the tooling gap (`pool_queue.sh add`'s reachability probe validates only the SHARED `~/derisk-pool/sim`
checkout, never an `--isolated --revision <sha>` one) as NOT-GATEABLE this round rather than editing shared,
concurrently-used pool infra mid-session; a named follow-up (`POOL_CHECK_ROOT` override) is recorded there.

## Files

`research/findings/raw/_gnw_swap_continuous_recency_6seed.json` / `_smoke.json` (full per-seed data, regenerated
this fix round with the new gates). `research/runners/_gnw_swap_continuous_recency_derisk.py` (restore arms,
full-conversation safety check, lesion-causality field, retraction docstring). `webapp/gnw_thought_swap.py`
(byte-identity fix + retraction-updated docstrings). `tests/test_gnw_swap_continuous_byte_identical.py` (new).
`research/FAILURE_LOG.md` (pool tooling gap, NOT-GATEABLE entry).
