# HANDOFF: B2d reply-level rescore finalized, comprehension-monitor crash fixed, `tools/b2d_score.py` built

Worktree `/home/dant123/Projects/sim/.claude/worktrees/b2d-prereg`, branch `research/b2d-reply-level` (created
2026-09-29 from `research/b2d-prereg-2026-09-29` @ `3c2907b85`, per the owner-coordinator's explicit instruction for
this arc; **note the branch-name mismatch** against `.claude/worktrees/b2-rulings/HANDOFF.md` §6, which names
`research/b2d-prereg-2026-09-29` for the same worktree -- that document predates this session's work and was not
updated here; reconcile branch names before merging either to main, or push both and let the owner pick).

Paths: `P=/home/dant123/Projects/sim` (primary checkout; shards land here, e.g. `b2b0924-base`'s raw arm files
used to validate `tools/b2d_score.py` below), `W=$P/.claude/worktrees/b2d-prereg` (this worktree).

## 0. What is already done (do not redo)

- **Finding, finalized and staged**: `research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md`.
  Applies every correction the reply-level rescore's adversarial skeptic checks made to the adjudicator's headline
  (5 of 14 checks REFUTED and corrected: confidence-forthcomingness reclassified instrument-limit not
  tautological; source-provenance-honesty trace-only ONLY for the registered probe (already reply-load-bearing on
  the chain-route turn); swap-drives-response's separately-run `LB_SWAP_DRIVE_PROBE` cited as already
  reply-load-bearing 6/6; wm-binding-advanced's two already-tried remediations named as closed dead ends; the
  headline's episodic-memory/s102 count corrected). Headline: B2b reply robust core 17/33 (trace core 27/28 once
  comprehension-monitor's broken lesion arm is excluded), flip-adequate reply core 15/25. Of 9 genuine trace-only
  rows (comprehension-monitor is UNDEFINED, not trace-only), 7 are instrument limits, 2 are tautological.
- **`docs/TERMS.md`**: added the `load-bearing` term (reply-level default meaning, code condition, fallback
  wording) + a dated note under "Adding a term" citing this session's overclaim.
- **B2d prereg amended** (`research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md`,
  staged, NOT YET COMMITTED even once -- the original filing itself was still uncommitted when this session
  started): added "Mouth-channel arm" (new tag `b2d0929-mouth`, GPU, Qwen2.5-0.5B/`QwenRenderer`, greedy decoding
  already default, for affect-coloring/affect-appraisal-interoceptive/affect-drives-response; NEEDS-CLAUDE:
  `_collect_worker` at `onebrain_regression_battery.py`~902 hardcodes `renderer="stub"`, so the renderer needs
  threading through before this arm can run) and "Probe remaps for instrument-limit rows" (dev-seed-first, per
  CLAUDE.md's 42/43/44-dev discipline; 3 rows already have unmerged reply-level evidence to integrate --
  confidence-forthcomingness, source-provenance-honesty, swap-drives-response -- and 4 need a genuinely new probe
  -- da-mode, multiref-competition, the two comprehension learned-cue rows -- open-ended-generation already has
  its NO-GO reply instrument, no new probe needed). Also updated "Feasibility measured at filing" with REAL S1
  numbers (see below) and marked Step 5 DONE with its scope reduction stated honestly.
- **`tools/b2d_score.py` + `tests/test_b2d_score.py`, DONE, 22/22 tests pass.** Computes S0 (verbatim `lb.json`
  read), S1 (self-describing fields excluded, via a direct arm-file diff over
  `research/coordination/b2d_decision_fields.json`'s `kept` fields -- NOT a harness replay; see the file's own
  docstring for the stated scope reduction against the prereg's literal "identity replay" description) and S2
  (reply-level, exactly as pre-registered), plus **A1.2 rule 7** (an organ `error` key on a row's own probe turn,
  in any arm, forces class C -- never `regressed`). Validated against REAL B2b arm files on disk (read-only, no
  brain built): S0 robust core 27/28 (comprehension-monitor correctly excluded via rule 7), S2 robust core 17/18
  (matches the finding's adjudicated number exactly), **S1 robust core 20, union 22, mean 0.724** (new --
  recorded in the prereg's "Feasibility measured at filing"). Per-faculty spot checks match the finding's per-row
  analysis exactly (e.g. affect-drives-response S0 6/6 -> S1 0/6 pass; da-mode-drives-response S0 6/6 -> S1 2/6,
  the genuine non-tautological subset).
- **Comprehension-monitor lesion-twin crash: FIXED.** `research/runners/comprehension_production_organ.py`
  `_snapshot_rest` now requires `comp is self.comp` before taking the shared-pool-snapshot branch (was: branched
  only on the organ's own `self._shared`, so the standalone lesion twin wrongly borrowed the pool's N=7002 rest
  snapshot into its own N=1032-neuron arrays). `tests/test_comprehension_lesion_twin_snapshot.py`, 4/4 passing,
  verified BOTH directions (fails on the pre-fix condition via `git stash`, passes post-fix; intact-path snapshot
  proven byte-identical to the shared pool's snap, unchanged by the fix). `research/FAILURE_LOG.md` row added
  (2026-09-29, comprehension-monitor entry, gate = the new test + B2d's rule 7). **Re-measuring
  comprehension-monitor's actual load-bearing status is NOT done** (needs a lesion run, which needs either B2d's
  own arm or a targeted probe -- see NEEDS-CLAUDE below).
- **Staged, NOT committed** (compute-idle-persistent / lane-starvation gates -- see §2): `docs/TERMS.md`,
  `research/FAILURE_LOG.md`, `research/runners/comprehension_production_organ.py`, the 5 pre-existing B2d prereg
  files (now amended), `research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md`,
  `tools/b2d_score.py`, `tests/test_b2d_score.py`, `tests/test_comprehension_lesion_twin_snapshot.py`.

## 1. Re-derive before trusting anything above (read-only, no brain build)

```bash
cd $W
SIM_BACKEND=numpy $P/.venv/bin/python -m pytest tests/test_comprehension_lesion_twin_snapshot.py tests/test_b2d_score.py -q
$P/.venv/bin/python tools/check_docs.py
$P/.venv/bin/python tools/b2d_score.py --tag b2b0924-base --pin a308f1e09babcc9ed096c3c8046d00040391368c \
    --seeds 42 43 44 100 101 102 --base $P/research/findings/raw/_load_bearing/_shards \
    --fields research/coordination/b2d_decision_fields.json
```

| prints | do |
|---|---|
| 26 tests pass (4 + 22), check_docs OK, b2d_score S0 robust_core_n 27 / S1 20 / S2 17 | matches this HANDOFF; proceed |
| any different number | STOP. NEEDS-CLAUDE: "b2d_score re-derivation changed: <summary>" (a shard file changed on disk since this was written, or a code regression) |

## 2. Commit order (only when gates allow; never `--no-verify`, never write a waiver)

Two independent, non-overlapping commits (matches CLAUDE.md's own carve-out: `tools/**`/`tests/**` clear the
idle-compute gate before other paths do):

```bash
cd $W
# Commit A -- code + tests only (most likely to clear the gate first)
git add tools/b2d_score.py tests/test_b2d_score.py tests/test_comprehension_lesion_twin_snapshot.py \
    research/runners/comprehension_production_organ.py
git commit -m "$(cat <<'EOF'
fix(comprehension): the lesion twin's rest snapshot now checks comp identity, not just organ pool-attachment; add tools/b2d_score.py (S0/S1/S2 + A1.2 rule 7)

_snapshot_rest branched on self._shared alone, so the standalone lesion twin wrongly borrowed the shared pool's
N=7002 rest snapshot into its own N=1032-neuron arrays whenever the organ ran pool-attached -- every retained
comprehension-monitor lesion arm since 2026-09-21 crashed on this (88/88), and the harness's except-branch read
the crash as a "6/6 load-bearing" regression. Fixed: the shared snapshot is used only when comp is self.comp;
every other bridge (the twin) always takes its own 40-step settle. Verified both directions
(tests/test_comprehension_lesion_twin_snapshot.py, 4/4): fails on the pre-fix condition, passes post-fix; the
intact path's snapshot is proven byte-identical to the shared pool's own snap, unchanged by the fix.

tools/b2d_score.py (tests/test_b2d_score.py, 22/22) computes the B2d pre-registration's S0 (raw), S1
(self-describing fields excluded, per research/coordination/b2d_decision_fields.json) and S2 (reply-level)
scores plus A1.2 rule 7 (an organ error on a row's probe turn forces class C, never regressed) directly from
shard arm files on disk -- no harness import, no brain build (pinned by a source/AST-level test). Validated
read-only against the real b2b0924-base shard tree: S0 core 27/28, S1 core 20/22 (new), S2 core 17/18 (matches
the reply-level rescore finding's adjudicated number exactly).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"

# Commit B -- findings + coordination docs
git add research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md \
    research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md \
    research/coordination/b2d_jobs.txt research/coordination/b2d_decision_fields.json \
    research/coordination/b2d_corpus_sha256.tsv \
    research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt \
    docs/TERMS.md research/FAILURE_LOG.md HANDOFF.md
git commit -m "$(cat <<'EOF'
docs(load-bearing): reply-level rescore finalized (5 skeptic corrections applied) + B2d prereg amended (mouth-channel arm, probe remaps) + docs/TERMS.md load-bearing entry

research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md applies every adversarial skeptic
correction to the adjudicated headline: confidence-forthcomingness reclassified instrument-limit (a separately-run
LTM-elaboration protocol already shows 6/6 GO); source-provenance-honesty trace-only ONLY for the registered
probe (already reply-load-bearing on the chain-route turn, default-on since 2026-09-01); swap-drives-response's
preregistered LB_SWAP_DRIVE_PROBE already ran reply-load-bearing 6/6, cited not folded into the headline;
wm-binding-advanced's two prior remediations named as closed dead ends; the two 2026-09-29 seed-102 A1.4 re-runs
counted. Headline: B2b reply robust core 17/33 (trace core 27/28 once comprehension-monitor's broken lesion arm
is excluded); flip-adequate reply core 15/25. Of 9 genuine trace-only rows, 7 are instrument limits, 2 tautological.

docs/TERMS.md: added "load-bearing" (reply-level default; a decision-field-only result must say "trace-level
load-bearing" explicitly), citing this overclaim.

B2d prereg (still its first commit overall): added the mouth-channel arm (GPU, Qwen2.5-0.5B, NEEDS-CLAUDE code
change named) and probe remaps for the genuine instrument-limit rows (dev-seed-first), and recorded
tools/b2d_score.py's real S1 numbers on B2b (core 20/22, mean 0.724).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
git status
```

| commit result | do |
|---|---|
| both succeed | `bash tools/push_both.sh research/b2d-reply-level`, then verify with `git ls-remote` (the script does this itself) |
| blocked by `compute_idle_persistent` or `lane_starvation` | wait (per the b2-rulings HANDOFF, budget frees ~18:30, or when a pool/GPU job is running); retry commit A first (tools/tests clear sooner per CLAUDE.md) |
| blocked by any other gate | STOP. NEEDS-CLAUDE with the gate's exact message -- do not waive, do not `--no-verify` |

## 3. Provisioning / queue commands (for when B2d itself is ready to queue -- NOT done by this session)

Unchanged from the pre-registration's own "Prepared commands" (steps 0-9); this session did not provision, queue,
or run anything. Quick reference:

```bash
# pool (CPU arms, the main b2d0929 registry) -- prereg Steps 3-4, 6
cd $P && git fetch -q origin && git merge-base --is-ancestor origin/main 3c2907b85c10fe918ddc81b4c69de8888c9f321b \
    && bash tools/pool_provision.sh --revision 3c2907b85c10fe918ddc81b4c69de8888c9f321b --isolated pool41 pool42
# (corpus hash recording: prereg Step 4, already done 2026-09-29 14:22, staged in b2d_corpus_sha256.tsv)
bash research/coordination/b2d_queue_next_wave.sh --dry-run   # NEEDS-CLAUDE: script not yet written (Step 6)

# GPU (mouth-channel arm, new tag b2d0929-mouth) -- NEEDS-CLAUDE: renderer threading in _collect_worker first
tools/gpu_queue.sh start   # no-op if the one daemon is already live
tools/gpu_queue.sh add 'SIM_BACKEND=cupy BRAIN_CHAT_RENDERER=qwen .venv/bin/python -u -m research.runners.onebrain_regression_battery --only affect-coloring --seed 42 --renderer qwen --repeats 2 --out research/findings/raw/_load_bearing/_shards/b2d0929-mouth/s42/affect-coloring/lb.json'
# ^ the --renderer flag does not exist yet; this is the SHAPE the command takes once the NEEDS-CLAUDE change lands
tools/gpu_queue.sh status
```

## 4. Scoring commands (`tools/b2d_score.py`, built this session)

```bash
.venv/bin/python tools/b2d_score.py --tag b2d0929 --pin 3c2907b85c10fe918ddc81b4c69de8888c9f321b \
    --seeds 42 43 44 100 101 102 --base research/findings/raw/_load_bearing/_shards \
    --fields research/coordination/b2d_decision_fields.json \
    --code-root ~/derisk-pool/revisions/3c2907b85c10fe918ddc81b4c69de8888c9f321b \
    --corpus research/coordination/b2d_corpus_sha256.tsv --reruns research/coordination/b2d_reruns.tsv \
    --dispatch-log research/queue/dispatch.log \
    --compare-tag b2b0924-base --compare-pin a308f1e09babcc9ed096c3c8046d00040391368c \
    --out research/findings/raw/_load_bearing/_shards/b2d0929/b2d_score.json
```

## 5. Outcome -> verdict table

| `b2d_score.py` prints | do |
|---|---|
| `verdict: COMPLETE`, `preconditions_failed: []` | write the findings doc, labeled "at H" and "stub renderer" for the main arm; report S0, S1 (headline), S2 beside it, per faculty |
| `verdict: FAIL`, `fail_cells` non-empty | name each FAIL row in the findings doc as a row the shipped revision cannot measure as registered; do not average it in |
| `verdict: INCOMPLETE` | apply A1.4 re-runs to every UNDEFINED cell (prereg Step 7); re-score; do not report a headline yet |
| any `preconditions_failed` entry (code-root blob mismatch, host-check failure) | STOP. Fix the precondition (wrong `--code-root`, a re-pin needed) before trusting any number |
| `organ_error_cells` non-empty for a row OTHER than comprehension-monitor | NEEDS-CLAUDE: a new organ crash, not the one this session fixed -- add a `research/FAILURE_LOG.md` row before proceeding |

## NEEDS-CLAUDE (open; do not act on these without the owner or a fresh Claude session)

1. **`_collect_worker` renderer threading** (`research/runners/onebrain_regression_battery.py`~902): hardcodes
   `renderer="stub"`; needed before the mouth-channel arm (§3) can run at all. Acceptance: with `renderer` left at
   its default, the existing 258-cell b2d0929 arm reproduces byte-identical to a run before the change.
2. **Harness-level S1 identity replay** (the prereg's original "How S1 is computed" design: import
   `research.runners.load_bearing_fraction` at H, monkeypatch `_spawn_arm_raw`/`compare()`). `tools/b2d_score.py`'s
   S1 is an independent direct-arm-file recomputation instead (stated honestly in its docstring); a harness replay
   would let a future run cross-check the two. Not blocking -- S1 as built is validated against real B2b data and
   matches this session's finding's per-row analysis exactly.
3. **Re-measure comprehension-monitor** now that its lesion arm no longer crashes. Needs either a fresh lesion run
   (any battery) or, cheaper, a single targeted re-run of just this row at a few dev seeds first.
4. **Dev-seed verification for the 4 genuinely-new probe remaps** (da-mode, multiref-competition, the two
   comprehension learned-cue rows) named in the B2d prereg's "Probe remaps" section -- none has been run even once,
   including at a dev seed.
5. **`research/coordination/b2d_queue_next_wave.sh`** (prereg Step 6) is not written.
6. **Step 1's SD-call re-check** (someone other than the three original audits re-reads
   `research/coordination/b2d_decision_fields.json`, pragmatic-implicature's borderline N/SD call first) is still
   open from before this session; unaffected by anything done here.
7. **Branch-name mismatch** (see the header): reconcile `research/b2d-reply-level` (this session, per explicit
   instruction) against `research/b2d-prereg-2026-09-29` (named in `.claude/worktrees/b2-rulings/HANDOFF.md` §6 for
   the same worktree) before either merges to main.
8. **swap-drives-response's 5 uncommitted seed artifacts** (seeds 43/44/100/101/102 of `LB_SWAP_DRIVE_PROBE`, at
   revision `7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`) sit untracked in a working tree somewhere -- find and commit
   them (or re-run at a fresh H-pinned revision) before citing more than the seed-42 leg as verified.
9. **B2b's A1.4 ruling on the two seed-102 re-runs** (discourse-register, episodic-memory) is a pending board
   decision this session's finding depends on for its headline numbers; if voided, both rows leave both B2b cores
   and every number citing 27/28 trace-core or 17/33 reply-core in the new finding needs a one-line correction.
