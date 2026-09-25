---
type: preregistration
status: preregistered
date: 2026-09-25
lane: load-bearing
mechanism: flip-rule criterion (c) for the PAIR BRAIN_DA_TAG_CAPTURE=1 + BRAIN_SLEEP_REPLAY_CAPTURE=1 (dopamine-gated
  synaptic tag-and-capture of chat-written store blocks, webapp/da_tag_capture_chat.py, plus the sleep-onset replay
  route, webapp/sleep_replay_capture.py) -- the full load-bearing registry measured twice at ONE pinned revision F2
  (origin/main fd29040db19987819461693aaf385977e45840ef), a base arm at the production default and a flipcand arm
  with the pair ON, compared pair by pair (row x seed)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any b2c0925 job, shard or smoke exists, and before any run of the
  load-bearing harness with either flag ON at any revision. Design (i) chosen (paired arms at F2); design (ii)
  (flipcand at F2 against B2b's base arm at a308f1e09) rejected with reasons. Nothing is queued by this document.
---

# Battery B2c: the registry with the DA-capture + sleep-replay pair ON, paired with its base at one revision (pre-registration)

## Why

The owner's flip rule (2026-09-23): a validated feature flips default-ON only with (a) a 6-seed GO finding, (b) a
SOUND independent review, (c) no regression in the combined production battery with it ON, and (d) a
production-default validation. This battery is leg (c) for one candidate, and the candidate is a PAIR.

- `BRAIN_DA_TAG_CAPTURE` holds (a) twice: `research/findings/2026-09-24-da-tag-capture-chat-wire-6seed-GO-runner-level-ltm-off.md`
  and `research/findings/2026-09-25-da-tag-capture-chat-wire-ltm-on-GO-runner-level-6seed.md` (GO 6/6, sign-flip p 1/64).
- Alone it is not a candidate. The LTM-on finding reads `ordinary_fact_flip_forgetting` True on all six seeds: a fact
  told once, plainly, is recalled the next day under today's default and lost with the flag ON (DA prereg
  `research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md`, Amendments 3 and 5).
- `BRAIN_SLEEP_REPLAY_CAPTURE` is the companion that closes that loss: `research/findings/2026-09-25-sleep-replay-capture-rc-GO-6seed.md`,
  GO 6/6 on RC1-RC6 (RC1: the ordinary fact is correct next day with the route, abstains without it), prereg
  `research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md`. It is inert without `BRAIN_DA_TAG_CAPTURE`.
- Context, not part of the pair: the awake-rest route (`research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md`,
  NO-GO, `BRAIN_AWAKE_REPLAY_CAPTURE` stays OFF) and the r2 downscaling rule
  (`research/findings/2026-09-25-sleep-replay-capture-r2-NO-GO-6seed.md`, NO-GO, `BRAIN_SLEEP_DOWNSCALING` stays OFF).
  Neither flag is set in any B2c arm.

B2c answers (c) for the two flags together. It does not answer (b) or (d), and it says nothing about either flag alone.

## Design: (i) paired arms at a new pin F2. Design (ii) is rejected

**(i)** 258 base cells and 258 flipcand cells, both at F2, same seeds, same rows, same probe set; each (row, seed)
pair differs only by three env tokens. **(ii)** flipcand at F2 scored against B2b's base arm at F = a308f1e09, which
is still running (134 of 258 cells had landed by 11:11 EDT on 2026-09-25). (ii) would save one arm, about 100 core-h.
It is rejected for five reasons.

1. **The code between F and F2 is not shown identical in data.** 46 Python files under `sim/`, `webapp/` and
   `research/runners/` differ. Fifteen main merges touch the 27 of them that production or the harness runs (table
   below). Each merge declares its flags default-OFF, but "declared default-OFF" is a reading of code, not a
   measurement. The only whole-feature OFF counterfactual in data covers the pair's own code as of 24380cbe4 (r2 item
   3, IDENTICAL). Nothing covers the other merges together.
2. **Such a check has already failed once, in this lane.** The DA lane's pinned offcheck (Amendment 4 of its prereg)
   read `replies_identical: false` with `BRAIN_DA_TAG_CAPTURE` unset on both trees. The store and the ledger scenario
   were identical. The reply JSON differed across a gap of 19 `webapp/server.py` hunks, about 14 of them from unrelated
   default-OFF merges. F to F2 adds 63 lines to that file alone. Hash tests showing that default-OFF code cannot change
   base cells do not exist for this range.
3. **One F-to-F2 change adds a provenance sidecar for the open-ended-generation row's own output, not every base
   shard directory** (narrowed, fix round: the F->F2 diff of `load_bearing_fraction.py` calls `declare_output` only
   for `oed_distributional*.json`, load_bearing_fraction.py:1190-1198). `research/runners/__init__.py` and
   `research/runners/load_bearing_fraction.py` (merge 56c0465c9) give that one row's file its own provenance sidecar;
   B2b's cells at F for it rely on the covered-by-parent rule instead. A cell at F and its twin at F2 can match in
   verdict there, never as artifacts -- but this reaches the 6 open-ended-generation cells (one row x six seeds), not
   the other 42 rows' shard directories. Reasons 1, 2 and 5 are enough on their own to reject design (ii); this one
   is scoped to its actual reach.
4. **The identity check (ii) needs would be partial and costly.** One full seed of base re-run at F2 is 43 cells, about
   17 core-h (B2b's measured cost per seed). Its pass would certify one seed of 43 rows, not six, and its fail would
   require (i) anyway. B2b's own base arm is not complete; any B2b cell that ends not DEFINED leaves the matching (ii)
   comparison undecidable.
5. **A reply change cannot be attributed across two revisions.** B2b's Amendment 1 showed that load-bearing counts
   alone miss a flag that changes intact replies (60 of 112 probe turns, for the learned referent lexicon). B2c
   therefore gates on reply changes (R2, R3 below). Across F and F2 such a change could be the pair or the drift; at one
   revision it can only be the pair.

(i) costs about 205 core-h (see "Cost"), and its base arm is also the #1 metric at F2, which B2b cannot give. The
cross-revision comparison (B2b base at F against B2c base at F2, cell by cell) is REPORTED below as a drift
measurement. It is the identity check (ii) would have needed, obtained at no extra cost. It never scores the pair.

**Code files that differ between F and F2 and that production or the harness runs** (from
`git log --first-parent a308f1e09..fd29040db`; left out: documentation, findings, tests, tools, and the 19 runner
modules no production module imports, 18 of them `_`-prefixed plus `chat_time_plasticity_audit.py`):

| merge | files | declared default |
|---|---|---|
| 19a56914d REQUIRED_ENV opt-in rows | `research/runners/lbf_rows/{__init__,reasoning_transitive_chat,tom_false_belief_chat}.py` | rows opt-in; registry unchanged |
| 100bb3839 A10 follow-ups | `research/runners/lbf_rows/reward_value_afferent.py`, `webapp/da_mode_drives_chat.py`, `webapp/reward_value_afferent_chat.py` | `BRAIN_REWARD_VALUE_AFFERENT` OFF |
| 2a84f80e1 open-ended gated turn | `research/runners/lbf_rows/open_ended_gated.py`, `webapp/open_ended_gated_turn.py`, `webapp/server.py` | `BRAIN_OPEN_ENDED_GATED` OFF |
| d68a607af, db8db2a5b, d51e9c88b lexicon rounds 1-3 | `research/runners/lexicon_frame_junction.py`, `research/runners/lexicon_spiking_frame_category.py` | lexicon flags OFF |
| 3bdf8b619 sleep-replay capture | `webapp/da_tag_capture.py`, `webapp/da_tag_capture_chat.py`, `webapp/sleep_replay_capture.py` | the pair's own code, OFF |
| 55823d1bd, a878d274e chat-time plasticity audit | `research/runners/{onebrain_merge_framework,surprise_production_organ,worldmodel_production_organ,_spiking_expectation_rpe_derisk,_affective_world_model_derisk}.py` | local-freeze flags OFF |
| 56c0465c9 oed provenance | `research/runners/__init__.py`, `research/runners/load_bearing_fraction.py` | provenance only (changes shard files) |
| 2af73bfdd gap#4 clamp companion | `sim/bridge.py`, `sim/config.py` | `bdsp_pbar_ratio_tau_ms` 0.0 |
| 4b4b35774 SETTLE A2 wiring | `webapp/affect_drives_chat.py`, `webapp/server.py`, `research/runners/_affect_marker_settle_congruence.py` | `BRAIN_AFFECT_MARKER_CONGRUENCE` OFF |
| 9d1329c35 sleep route r2 | `research/runners/onebrain_regression_battery.py`, `webapp/da_tag_capture_chat.py`, `webapp/sleep_replay_capture.py` | sub-flags OFF; label-only groups |
| f793b6945 awake-replay capture | the same battery file, `webapp/awake_replay_capture.py`, `webapp/da_tag_capture{,_chat}.py` | `BRAIN_AWAKE_REPLAY_CAPTURE` OFF |
| df12ec1cc sleep load renorm | the same battery file, `webapp/sleep_replay_capture.py` | `BRAIN_SLEEP_LOAD_RENORM` OFF |

## Frozen revision F2 and registry

- **F2 = `fd29040db19987819461693aaf385977e45840ef`**, origin/main when this was filed (a documentation-only commit
  over merge df415f6ca). `git merge-base --is-ancestor` confirms it contains 3bdf8b619 (sleep-replay capture merge),
  2ac0fb245 (the DA chat wire merge), cce3c1dbd (DA Amendment 3, the LTM-on arm's code) and 269ae8f76 (the revision
  the rc family ran at). Of these four, F = a308f1e09 contains only 2ac0fb245.
- **Why the newest main and not the earliest candidate (3bdf8b619).** A flip, if licensed, lands on main at or after
  F2, so F2 is closer to what would ship. F2's version of the pair includes r2's night schedule and awake mark. Per the
  module docstring these still run exactly one epoch for any one-night protocol. The r2 counterfactual read the whole
  feature IDENTICAL when OFF at 24380cbe4; the awake-replay and load-renorm merges edited the same files after that.
  F2 also carries the oed sidecar fix, so no B2c cell needs the covered-by-parent rule.
- **F2's ON path is READ as matching the GO code, not MEASURED to (fix round; see "Residuals").** `git diff --stat`
  over `da_tag_capture{,_chat}.py`, `sleep_replay_capture.py` and `onebrain_regression_battery.py` reads +353/-17
  from 269ae8f76 (the rc family's own GO revision) to F2, and +613/-12 from cce3c1dbd (the LTM-on DA Amendment 3 GO
  revision) to F2. The paragraph above accepts this on the module docstring's word ("still run exactly one epoch") --
  the same kind of check design (ii) is rejected for at reason 1. This battery's PASS therefore does not by itself
  show F2's ON path reproduces the GOs; see "Residuals" for the identity check this still needs before a flip.
- **Registry at F2.** `load_bearing_fraction.FACULTY_LESIONS` after the row hook: 50 rows, the same keys and kinds as
  at F (computed at filing, F from a `git archive` tree, F2 from a checkout at F2; identical). Coverable (neural-lesion + whether-disable):
  36. Sharded (`lb_shard.py` MEASURABLE_KINDS): 43 per seed, so 258 cells per arm and 516 in all <!--derived-->. The
  43 rows are frozen, in registry order, in `research/coordination/b2c_make_jobs.sh`.

## Arms (fixed)

Both arms: `--no-fixes`, the adequate probe set (the nine LB_* flags), `--repeats 2`, numpy, one BLAS/OMP thread,
seeds 42 43 44 100 101 102, the 43 frozen rows. Each line keeps B2b's layout:
`cd ~/derisk-pool/revisions/<F2> && .venv/bin/python tools/assert_flipped_defaults.py && mkdir -p <shard dir> && env <env prefix> .venv/bin/python -u -m research.runners.load_bearing_fraction --only <row> --seed <s> --repeats 2 --out <shard dir>/lb.json`.

- **b2c0925-base.** No BRAIN_* token. Each line equals the matching line of `research/coordination/b2b0924_base_jobs.txt`
  once the revision and the tag are substituted (checked with `cmp` over all 258 lines at filing).
- **b2c0925-flipcand.** Exactly three tokens in the env prefix: `BRAIN_DA_TAG_CAPTURE=1`
  `BRAIN_DA_TAG_CAPTURE_CLOCK=turn` `BRAIN_SLEEP_REPLAY_CAPTURE=1`. Removing them and renaming the tag gives the base
  twin exactly (checked by `b2c_make_jobs.sh` at generation).
- **Queue source.** `research/coordination/b2c0925_jobs.txt`, 516 lines: for each seed in order, for each row, the base
  line followed by its flipcand twin.

**Why the turn clock.** The ledger's world clock belongs to the environment (host code for the world is legitimate).
Its production default, `wall`, is machine time since the ledger was built. In a battery that is the pool node's compute
time, which varies with host and load. Flipcand cells would stop being deterministic. Any gap of more than five minutes
of compute between two observed turns (`continuous_engine.SLEEP_IDLE_SEC` = 300 s) would also start a sleep episode
mid-conversation. `turn` makes each observed turn last 30 s of world time. Every GO run of both flags used it
(`_da_tag_capture_chat_probe.py`:
`ON = {BRAIN_DA_TAG_CAPTURE: 1, BRAIN_DA_TAG_CAPTURE_CLOCK: turn}`), and so does the harness's own
`_DA_TAG_CAPTURE_ENV`. The rc family's arms are `{**ON, **RC}`, so the sleep route's GO ran on it too. The flip
would change two brain flags; the clock default stays `wall`. B2c does not test the wall clock (see "Residuals": leg
(d) must).

**How the flags reach every arm (read in the code at F2).** `lb_shard.py jobs --extra-env` puts them in the shard
process's env. Every arm is a subprocess from `onebrain_regression_battery._spawn_arm` with `env=dict(os.environ)`.
Its worker overlays only the row's driving `base_env` and, in the lesion arm, the row's lesion flag. No row at F2 uses
any of the three tokens as its lesion flag. No row's `base_env` sets them: `LB_DA_TAG_CAPTURE_PROBE` is not in the
adequate set. So every flipcand arm carries the pair and no base arm does. `assert_flipped_defaults.py` runs before the
env prefix, in the node's own environment, and sees neither arm's flags.

## What the pair can reach in this battery (read in the code at F2, before any cell exists)

- `observe_chat_turn` (each chat turn, after the DA-encoding read) builds the session's ledger on the first turn,
  treats every store block already present (build-time knowledge) as unmanaged, integrates the ledger to the turn's
  world time, and rewrites every managed block.
- `after_store_chat` (response assembly) registers each block the turn wrote and rewrites it as `w = b + f*inc`
  (b a seeded baseline, f the early-phase factor, 1 at the write). A chat-written block differs from base from the
  next turn on.
- The sleep route runs only when the ledger clock passes sleep onset, five minutes after the last observed turn. On
  the turn clock that happens only across a world step. `tick_chat` runs only on the engine's idle tick, which in this
  harness only the world steps call.

Consequences by row, from each coverable row's driving turn group at F2 under the adequate probe set:

| class | coverable rows | what the pair can do before the driving reply |
|---|---|---|
| UNEXPOSED (26) | affect-appraisal-interoceptive, affect-coloring, affect-drives-response, affect-marker-spiking-wta, affective-tom, bg-action-selection, comprehension-learned-animacy-cue, comprehension-learned-verb-selects, comprehension-monitor, confidence-forthcomingness, curiosity-followup, da-gated-encoding, da-mode-drives-response, gnw-bus, gnw-multistep-deliberation, metacog-monitor, multiref-competition, noncontradiction-gate, pragmatic-implicature, reconsolidation, self-initiated-utterance, source-provenance-honesty, surprise-monitor, vision-identity-spiking-hmax, worldmodel-forward; open-ended-generation (distributional ruler, no chat turn) | no managed block exists before the reply (right: no `base + weight_factor*inc` rewrite happens), but the ledger still runs on the driving turn -- `on_store`+`advance` call `_write`, which unconditionally sets `comp._store_dirty`/`_store_csr = None`/`_persistent_dirty` and clears `_csr_cache` (and `_seq_dirty`/`_fused_dirty` where those apply) even with zero managed blocks (da_tag_capture.py:497-511, fix round: corrected from "nothing but building the ledger"); R2 catches any reply change this causes |
| IN-CONVERSATION (8) | causal-whatif (7-turn group), prospective-memory (5), discourse-register (3), common-ground-drives, episodic-memory, spiking-anaphor, swap-drives-response, wm-binding-advanced (2 each) | the rewrite `w = base + weight_factor*inc` of blocks written earlier in the same conversation, where `base` is a SEEDED complex-Gaussian baseline the SAME magnitude as `inc` (`BETA_BASELINE=1.0`, da_tag_capture.py:69, fix round: this is not a small perturbation of the intact weight, the two terms are comparable in size); `weight_factor >= exp(-3/90)`, about 0.967 <!--derived--> at most, for the oldest block (6 turns = 3 min of world time old) at the driving turn; R2 catches any resulting change |
| NIGHT (2) | d5-consolidate (d5c group), sleep-replay (slp group) | a 24 h world step: early-phase decay, the SWR epoch, capture; the only rows where the pair's designed effect can reach the reply |

**Which rows may legitimately change: the two NIGHT rows only.** There a memory ability may improve or degrade, and
R3 scores the direction. Everywhere else the pair is not designed to change a reply, and a change counts as a
regression (R2). The da-gated-encoding row is measured on `well` (field `da_encoding.on`), which the ledger never
writes. The battery therefore cannot show the pair load-bearing; the GO findings do that.

**R3's real reach is narrower than "the two NIGHT rows" (fix round).** In B2b's own base cells at F, `d5c_teach`
and `slp_teach2` do not resolve the patient on any of the six seeds, so the wolf and owl facts R3's CORRECT target
checks for (`["wolf", "chase", "rabbit"]`, `["owl", "chase", "mouse"]`) are never stored in either arm; only
ABSTAIN and OTHER are reachable there, so R3 can catch a downgrade into confabulation but cannot catch the loss of
a fact that WAS stored -- the facts `slp_teach1`/`slp_teach3` DO store (fox/hare, hawk/vole) are never the target of
any recall turn. See "What a PASS licenses" and the pre-registered fox/hawk read-back under "Reported" below.

## Validity of a cell (per arm; B2b Amendment 1.2 with F2 in place of F, plus the expected env)

A (row, seed) cell of either arm is DEFINED only if all of these hold:
1. `lb.json` exists at its job line's `--out`. Its sidecar records `git_sha` F2 in full, `source_kind` `git_archive`,
   both `source_manifest_verified_*` flags true, and `SIM_BACKEND=numpy`.
2. Every arm sidecar in the shard directory records the same.
3. Env. Base: the `lb.json` sidecar holds no BRAIN_* key; arm sidecars hold BRAIN_CHAT_SEED only. Flipcand: every
   sidecar holds exactly the three pair tokens with those values, plus BRAIN_CHAT_SEED on arm sidecars, and no other
   BRAIN_* key. Rules 1-3 are what `tools/lb_shard.py aggregate --pin <F2>` checks, with `--expect-env` for flipcand
   (added on this branch; tested both directions in `tests/test_lb_shard_expect_env.py`).
4. The node that ran it holds the recorded corpus hash ("Corpus").
5. The report is not UNRELIABLE, `null_control_clean` is not False, the verdict is `regressed`, `pass` or
   `not-exercised`, and (flipcand only, fix round: widened from a single top-level field, which missed two live
   failure paths):
   - **no turn of any arm file carries an `error` key ANYWHERE under `da_tag_capture`**, at any depth -- both
     top-level (`da_tag_capture.error`, `after_store_chat` raising, webapp/server.py's second try/except around it)
     AND nested under `observe` (`da_tag_capture.observe.error`, `observe_chat_turn` raising: webapp/server.py's
     FIRST try/except around it catches the error into `da_tag_capture_info`, which then lands at
     `resp["da_tag_capture"]["observe"]`, not at the top level a single-field check would read);
   - **for a NIGHT row** (d5-consolidate, sleep-replay) **whose flipcand arm's `n_managed_blocks` > 0 at the recall
     turn**, that same turn's `da_tag_capture.sleep_replay_capture.n_epochs` is also >= 1 -- its absence with blocks
     already managed means the sleep route did not run that night;
   - **the cell's dispatch log does not contain "DA tag-and-capture tick failed"** -- webapp/continuous_engine.py's
     idle-tick handler around `tick_chat` only LOGS this (a `warning`, not a raise), which is where the night's SWR
     epoch actually runs, so none of the JSON fields above can see a tick that failed there; a pool cell's own
     `autodispatch.out` (or the smoke's direct-ssh capture) is what this check reads.

   Without this widening a flipcand cell whose pair crashed mid-turn or whose sleep route silently never fired could
   still read DEFINED and score PASS. `research/coordination/b2c_smoke_check.py` implements the same three checks
   for the pre-wave smoke (see "Integrity smoke").

Causes of an UNDEFINED cell are B2b's A1.3 classes. **E**: the cell did not run F2 as registered (rules 1-4, a killed
process). **C**: the code at F2 failed on the row (the A1.3 list, UNRELIABLE, a dirty null control, an `error` key
anywhere under `da_tag_capture` in a flipcand arm, a NIGHT row managing a block without a matching
`sleep_replay_capture.n_epochs` >= 1 at recall, or a dispatch log carrying "DA tag-and-capture tick failed"). **I**:
the probe ran and could not decide.

## Re-run procedure (B2b A1.4, adapted)

- **One re-run per UNDEFINED cell.** First the whole shard directory moves to
  `research/findings/raw/_load_bearing/_b2c0925_attempt1/<tag>/s<seed>/<row>/`, on the node and in the primary
  checkout. The re-run goes by direct ssh on a named F2-provisioned node other than the first host, with
  `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` in front of the job line's `env`. It is logged in
  `research/coordination/b2c0925_reruns.tsv`: cell, first host, cause and verdict, re-run host, command, start, result.
- **A line that never ran is re-queued, at most twice, and does not use up the re-run** (A1.4 c).
- **Pair re-run (new).** A pair scored as a regression below whose two cells ran on different hosts (sidecar `host`)
  is re-run for BOTH arms, one after the other, on one named F2 node, with the same wrapper; that same-host pair
  decides. Both first attempts move to attempt1 and are reported. A regression scored with both cells on one host is
  decided as it stands (numpy on one thread; the null controls are the run-to-run check).

## Pair states and the regression rule (written before any cell exists)

Cell final states follow B2b A1.5: DEFINED, FAIL cell (class C on both attempts, two hosts), UNDECIDED, MISSING,
C-UNCONFIRMED. For each of the 216 coverable pairs (36 rows x 6 seeds):
- **DECIDED** if both cells are DEFINED.
- **FLIP-FAIL** if the flipcand cell is a FAIL cell and the base cell is DEFINED. This counts as a regression: with the
  pair ON the row can no longer be measured.
- **UNDEFINED** otherwise, named with both cells' states. It is never scored as a regression and never as a pass.

A DECIDED pair is read on two things.

- **(a) Load-bearing status**, `load_bearing` in each cell's `lb.json`.
- **(b) The intact reply at the driving turn.** The turn is `turn` in `lb.json`; the reply is that turn's response in
  the cell's single `intact_a_*` arm file (not its `.prov.json`). Compared: the row's registered decision fields as the harness's
  `compare()` compares them (its noise fields excluded), plus the top-level `abstained` and `answer` (exact string). The
  pair's own trace key `da_tag_capture` is excluded. For open-ended-generation (no arm file), (b) compares the
  `lb.json` entry on `lb_shard._CONTENT_BOUND_FIELDS`.

Instrument check behind the exact `answer` compare: in the 126 B2b base cells at F that have one `intact_a` and one
`intact_b` file, the null-control rebuild's driving-turn `answer`, `abstained` and `recalled_svo` equal intact_a's in
all 126 <!--derived-->. Run-to-run variation on one host does not reach these fields.

Scoring:
- **R1, load-bearing loss.** Base `load_bearing` true, flipcand false: REGRESSION.
- **R2, reply change outside the NIGHT rows.** For the 34 other coverable rows, any difference in (b) is a
  REGRESSION. It is labelled LEAK on an UNEXPOSED row and IN-CONVERSATION on the eight exposed rows. The label is
  diagnostic; both count.
- **R3, recall direction on the NIGHT rows** (d5-consolidate, sleep-replay). The driving turn's outcome in each arm is
  CORRECT if `abstained` is false and `recalled_svo` equals the told triple: `["wolf", "chase", "rabbit"]` for
  `d5c_recall2`, `["owl", "chase", "mouse"]` for `slp_recall`. It is ABSTAIN if `abstained` is true, and OTHER
  otherwise (an answer that is not the told fact, the confabulation class). Order: CORRECT > ABSTAIN > OTHER.
  Flipcand lower than base: REGRESSION. Higher: IMPROVEMENT, reported. The same class with a difference in (b):
  NEUTRAL-CHANGE, reported. **d5-consolidate's own gate (fix round):** score it under R3 only when its flipcand
  arm's `n_managed_blocks` > 0 at the recall turn (a fact was actually stored to consolidate). It is expected to be
  0 -- `d5c_teach`'s patient never resolves in B2b's base cells on any of the six seeds -- and when it is, any
  difference in (b) at `d5c_recall2` is scored under R2 instead: with nothing stored, a reply change there cannot be
  a memory effect, only a leak.
- **Load-bearing gain.** Base false, flipcand true: GAIN, reported. On a row outside the NIGHT rows it is also flagged
  as unexpected from the mechanism.
- **No netting.** An IMPROVEMENT or a GAIN never offsets a REGRESSION on another row or seed.

**Battery verdict, in this order:**
- **FAIL** if any DECIDED pair is a REGRESSION (R1, R2 or R3, host-confirmed by the pair re-run where the hosts
  differ), or any pair is FLIP-FAIL. Each is named with both cells' values, hosts, labels and error text. A FAIL
  keeps both flags default-OFF.
- **INCOMPLETE** if any of the 216 pairs is UNDEFINED once every re-run and re-queue has run (each named). It is final,
  and it is not a PASS.
- **PASS** otherwise.

## Scoring code (a start precondition)

`tools/b2c_score.py` must be merged, UNCHANGED, to `origin/main` before wave 1 (fix round: `b2c_queue_next_wave.sh`'s
precondition check now runs `git cat-file -e origin/main:tools/b2c_score.py` plus `git diff --quiet origin/main --
tools/b2c_score.py`, not a bare file-exists test, so a local-only or uncommitted copy still refuses the queue).
`b2c_queue_next_wave.sh` refuses to queue while it is absent, so no rule is coded after any cell is seen. It
implements exactly the rules above, nothing else.
- **Inputs:** both tags' shard trees, the pin F2, the expected env, and each cell's dispatch log (`autodispatch.out`
  for a pool cell, the direct-ssh capture for the smoke) for the rule-5 tick-failure grep.
- **Validity:** `lb_shard.cell_prov_fails` for rules 1-3, the corpus record for rule 4; rule 5 reads `lb.json` for
  the report/null-control/verdict checks AND, flipcand only, every turn of every arm file for an `error` key
  anywhere under `da_tag_capture` (top-level or nested under `observe`), the NIGHT recall turn's
  `sleep_replay_capture.n_epochs` when that arm's `n_managed_blocks` > 0, and the cell's dispatch log for
  "DA tag-and-capture tick failed" (an idle-tick error webapp/continuous_engine.py only LOGS -- no JSON field sees
  it). `research/coordination/b2c_smoke_check.py` implements this same rule-5 logic for the smoke and is the
  reference the scorer's tests are checked against.
- **Rows:** each row's decision fields from `load_bearing_fraction` at F2 with the adequate-probe remaps, cross-checked
  against `lb.json`'s `turn`.
- **Output:** a `score.json` in `research/findings/raw/_load_bearing/_b2c0925_score/`, holding every pair's state,
  class, (a) and (b) values, hosts and label.
- **Selftest:** must fail in the failing direction of each rule: an R1 loss, an R2 LEAK and an IN-CONVERSATION
  change, an R3 downgrade; an R3 upgrade scored as IMPROVEMENT and not as a regression; an UNDEFINED pair never
  scored; FLIP-FAIL; a nested `da_tag_capture.observe.error` scored class C (not silently DEFINED); a NIGHT cell
  with `n_managed_blocks` > 0 and `sleep_replay_capture.n_epochs` 0 or absent at recall scored class C; a cell whose
  dispatch log carries "DA tag-and-capture tick failed" scored class C even with clean JSON fields; a d5-consolidate
  pair with 0 managed blocks in flipcand scored under R2, not R3, on any reply difference.

## Reported (not gated)

- Per arm: robust core, union, mean fraction and SD over the six seeds, backend, host mix and `ltm_mode`, as
  `lb_shard.py aggregate` writes them. B2c-base is the #1 metric at F2.
- The pair table: every pair's state and class, with the hosts of both cells.
- Manipulation read, from the flipcand intact arms: which rows' arms carry `da_tag_capture` at all, and
  `n_managed_blocks` before the driving turn. For the NIGHT rows, also `sleep_replay_capture` (`n_epochs`, per-block
  read-back R) and the recall outcome in both arms on every seed. The slp group tells three facts before one night; the
  rc GO's scope was one fact per conversation, so this is the first multi-fact night read of the route. If no flipcand
  intact arm on any seed manages a block, the finding says in its verdict line that the pair rewrote nothing the probes
  reach and a PASS says only that the flags broke nothing there.
- **Pre-registered fox/hawk read-back (fix round).** Since the recall target R3 gates on (wolf/owl) is never stored
  (see "Which rows may legitimately change"), the read-back that IS informative on retention is on the facts that
  ARE stored: sleep-replay's fox/hare block (`slp_teach1`) and hawk/vole block (`slp_teach3`). For each, on every
  seed and both arms, report `sleep_replay_capture`'s per-block `R` at every epoch and `da_tag_capture`'s per-block
  `z_mean`/`frac_synapses_z_gt_half` at the `slp_recall` turn, so a reader can see whether the route captured and
  read back a fact that was actually written, independent of the untestable wolf/owl target. Not gated: this reports
  retention where it CAN be measured, it does not substitute for an R3-style pass/fail on it.
- Answer changes in lesion arms, for pairs whose load-bearing status is unchanged.
- **Drift F to F2:** for each (row, seed) whose B2b-base cell (at F, per B2b's own rule) and B2c-base cell are both
  DEFINED, whether (a) and (b) are equal. A difference is attributed to the F-to-F2 merges, not bisected here, and
  never used to score the pair.
- Wall time per shard per arm from the sidecars (the pair's cost per arm process), and peak memory where recorded.

## Provenance

- Every job runs from `~/derisk-pool/revisions/<F2>` as `tools/pool_provision.sh --isolated` builds it: `git archive`
  of F2 plus `.source_revision`, `.source_manifest.sha256` and `.provisioned_ok`. A git worktree does not qualify.
- F2 is provisioned on pool1 and pool2 only. They are the same AWS instance type (r7i.4xlarge), and the dispatcher's
  `revision_available` check then hands B2c lines to no other node. A cell that lands elsewhere is still valid under
  the cell rules; the pair re-run handles cross-host pairs.
- Aggregates are written with `lb_shard.py aggregate --pin <F2>` (base with an explicitly empty `--expect-env`,
  flipcand with the three tokens). Both then carry `provenance.status: verified`, which gate LBP requires for a
  committed `aggregate.json`.

## Corpus (B2b A1.6)

F2 pins the code, not the data. After provisioning, `research/coordination/b2c_record_corpus.sh pool1 pool2` hashes
`data/corpus/tinystories.txt` in each node's `revisions/<F2>` copy. It writes `research/coordination/b2c0925_corpus_sha256.tsv`
only when every node matches the primary checkout, holds F2 and carries `.provisioned_ok`, and that file is committed
before wave 1. The primary checkout's copy read `7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec` at
B2b's filing. A cell from a node whose hash differs is class E; **so is a cell from a node missing from
`b2c0925_corpus_sha256.tsv` entirely** (fix round: rule 4, "Validity of a cell" #4, reads "holds the recorded
hash" -- a host absent from the record has no hash to hold, which is the same failure as a mismatched one, not a
lesser one). A node re-provisioned during the battery is hashed again before it takes more cells.

## Memory

Every line declares `mem_gb=8` through its `--checked` reason, B2b's value. The flipcand arm adds an isolated spiking
D1 reader and the ledger to each arm process; their size is unmeasured. The integrity smoke runs under a hard 8 GB cap.
If it is killed for memory, `mem_gb` is raised by an amendment before wave 1.

## Integrity smoke (declared, no weight)

Before wave 1, `research/coordination/b2c_smoke.sh` runs two flipcand lines at seed 7 (not a battery seed) by direct
ssh on pool1, into `research/findings/raw/_load_bearing/_b2c0925_smoke/`, which no aggregate reads. Each is the
battery's own s42 flipcand line with the seed and the output directory swapped, PRECEDED by the same pinned
`assert_flipped_defaults.py` guard every battery job line carries. The rows are sleep-replay (NIGHT) and
causal-whatif (the longest in-conversation group). The remote out_dir is rsync'd back to this checkout (fix round),
and `research/coordination/b2c_smoke_check.py` then scores it against the same checks rule 5 of "Validity of a
cell" uses: `da_tag_capture` appears with no `error` key at any depth (top-level or nested under `observe`),
`n_managed_blocks`, `sleep_replay_capture.n_epochs` >= 1 at `slp_recall` once a block is managed, no "DA
tag-and-capture tick failed" line in the remote's captured stdout+stderr, and the LTM tier the arm actually built
against (from the pulled sidecar's `env.BRAIN_DATA_ROOT`, reported). Each line must also complete under `memcap.sh
8`. No criterion reads this smoke. Its results will be seen before the battery runs; this document is committed
first.

## Queueing

`research/coordination/b2c_queue_next_wave.sh`, modelled on `b2b_queue_next_wave.sh`, is added to the heartbeat cycle
once the battery is started. It queues six waves of 86 lines, one seed per wave, each base line followed by its
flipcand twin, so the two cells of a pair dispatch together. It queues the next wave only when fewer than 20 B2c lines
are still fresh in the queue (`POOL_JOB_MAX_AGE` 12 h, as in B2b A1.8). It adds no `FRONT=1`, so B2b finishes first.
It refuses (exit 2) until the start preconditions exist:
- the committed corpus record;
- `PIN.txt` (F2) for both tags, and `EXPECT_ENV.txt` (the three tokens) for flipcand only, under the primary
  checkout's shard tree;
- `tools/b2c_score.py`, merged UNCHANGED to `origin/main` (fix round: checked with `git cat-file -e` + `git diff
  --quiet` against `origin/main`, not a bare file-exists test -- see "Scoring code").

**Human precondition the script cannot check (fix round; see "What a PASS licenses"): verify-go items B3 and B4 on
the pair have run.** `b2c_queue_next_wave.sh` has no way to verify another finding's status, so this is not coded
into it -- whoever runs "Prepared commands" below confirms it first.

Before each add it checks the line statically: the pinned `cd`, the guard, and the env tokens (none for base, exactly
three for flipcand). `--status` re-checks every queued B2c line for those rules and the `#checked:` tail. It never
re-queues or moves a stale line.

## Cost and wall time (estimate from the record)

- **B2b base cells at F, from their sidecars.** 134 had landed by 11:11 EDT (seeds 42 and 43 complete, 44 at 41 of 43,
  100 at 7). A complete seed took 16.4 and 17.7 core-h. The median shard ran 16.4 min, the mean 21.9 min, the longest
  146 min (episodic-memory). B2a: 186 cells, 95.5 core-h, longest 248 min.
- **B2c:** about 17 core-h x 6 seeds x 2 arms, so about 205 core-h, plus the pair's overhead in the flipcand arm
  (unmeasured).
- **At B2b's observed throughput:** about 10.7 cells per hour (134 cells in 12.5 h of dispatch, excluding the
  2.4 h dispatcher starvation of 2026-09-25). 516 cells then take about 48 h.
- **With pool1 and pool2 serving B2c alone, corrected (fix round).** `tools/pool_autodispatch.sh`'s `node_is_idle`
  caps a node at `cores - 1` single-threaded `-m research.runners` processes (no `POOL_JOBS_PER_NODE` override on
  the live dispatch service), and the pgrep pattern it counts against that cap matches BOTH processes an LB job
  runs -- the `load_bearing_fraction` parent and the arm worker it spawns via
  `onebrain_regression_battery._spawn_arm`. A 16-core node therefore holds at most 8 concurrent LB jobs, not 16 (the
  naive `cores - 1` reading), confirmed by B2b's own sidecars (at most 8 concurrent cells landed per AWS host
  there). Two nodes: **16 concurrent cells, not 28.**
- **Revised floor.** 204 core-h (6 seeds x 2 arms x 17 core-h) over 16 concurrent cells is about 13 h of dispatch
  <!--derived-->, plus a tail of 2-4 h for the longest cells (146-248 min observed in B2b/B2a): **about 15-17 h**,
  not 10-12 h.
- **AWS cost, corrected.** About $1.0 per node-hour of compute (the spend ledger's own rate) x two nodes x about
  15-17 h each is **about $32-35** <!--derived-->, not $20-25; about $100 if both nodes instead run the full 48 h
  B2b-throughput estimate. The owner's $50/day cap applies to the daily rate, and the tooling enforces it.

## What was seen before filing (disclosure)

- **Nothing of B2c.** No b2c0925 line, shard or smoke exists. No run of the load-bearing harness with either flag ON
  exists at any revision.
- **Read.** The three GO findings and the two context NO-GO findings named above; DA prereg Amendments 3-5; the
  header of the sleep-replay prereg; B2b's prereg and wave script; B2a's prereg and its rescored finding.
- **B2b base cells at F** (untracked, primary checkout):
  - sidecar timings and hosts of the 134 landed cells, for the cost estimate;
  - `lb.json` verdicts of five rows: sleep-replay and d5-consolidate `pass`, not load-bearing, on seeds 42-44;
    episodic-memory load-bearing on 42-43; causal-whatif and prospective-memory load-bearing on 42-44;
  - the seed-42 `intact_a` files of d5-consolidate and sleep-replay. Every recall turn abstains ("I haven't learned
    about mentioned"). `d5c_teach` and `slp_teach2` do not resolve the patient, so the wolf and owl facts are not
    parsed; `slp_teach1` and `slp_teach3` return fox/hare and hawk/vole. This set the expectation that R3 has little
    room to show an improvement on this probe. R3 was written symmetric and was not tuned to it;
  - the null-control `answer` check above (126 cells, base only).
- **Computed at filing:** the registry at F and at F2 (identical); the static turn-group table at F2; B2a's aggregate
  re-run with the modified `lb_shard.py` (`--pin` M1, no expected env), which reproduced the committed
  `research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json` byte-identical (`cmp`); the B2c base lines
  against B2b's job file (`cmp`, identical after substitution).

## What a PASS licenses

Criterion (c) of the owner's flip rule for the PAIR, both flags together, at F2, under the scripted turn clock, over
the 36 coverable rows and the adequate probe set. That means no load-bearing loss, no reply change outside the NIGHT
rows, and no recall downgrade reachable by R3 in them. It licenses nothing more:
- not (b), a SOUND review of the pair;
- not (d), a production-default validation on the live server path with both flags as defaults and the wall clock;
- not either flag alone. DA alone loses an ordinary fact overnight; the sleep route is inert without it;
- not `BRAIN_AWAKE_REPLAY_CAPTURE`, `BRAIN_SLEEP_DOWNSCALING` or `BRAIN_SLEEP_LOAD_RENORM`;
- not a claim that the pair is load-bearing or helps memory in chat. That evidence is the GO findings'.
- **not a test of overnight RETENTION for the facts that ARE stored** (fix round). R3's CORRECT target (the wolf/owl
  triples) is never reachable: `d5c_teach` and `slp_teach2` fail to resolve their patient at teach time on all six
  of B2b's base-cell seeds, so neither fact is ever stored in either arm. R3 can therefore only catch a downgrade in
  the ABSTAIN/OTHER split (confabulation), never the loss of a fact that WAS stored. The facts that ARE stored
  (fox/hare from `slp_teach1`, hawk/vole from `slp_teach3`) are read back and reported (see "Reported" below,
  pre-registered) but are not R3's gate, so a PASS says the reported read-back did not visibly worsen, not that
  overnight retention was gated end-to-end;
- **not an ON-path identity check** (fix round; see "Residuals"). A PASS here does not by itself show F2's flipcand
  code reproduces the GO revisions' behavior, and does not by itself license a flip without that check;
- **scoped to LTM-off** (fix round; see "Residuals" "LTM, pinned") -- not the LTM-on DA arm the flip-deciding GO ran.

A FAIL keeps both flags default-OFF and names each regressed pair. INCOMPLETE is not a PASS.

**Precondition on running this battery at all (fix round, owner-directed order): B2c is not queued before verify-go
items B3 and B4 have run on the pair.** The adversarial verify-go review of this pair
(`research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md` on `research/pair-verify-go`,
verdict SOUND-WITH-ISSUES) reads leg (b) as NOT YET MET and its section 7, "What must happen, in order", lists B3
and B4 (after B1's correction note and B2's owner decision) as still open before leg (b) counts as met: **B3** a
registered production-path arm on the WALL clock (a realistic day: several turns, two or more facts told at
different times, three or more pauses of 5 min or more, then a night and a multi-night idle, gated on no
resurrection of a decayed fact, no confabulation and the ordinary/salient outcomes -- "until this runs, the
multi-epoch regime is unmeasured"); **B4** a registered salient-vs-neutral contrast inside one family with both
flags intact (long-delay salient vs neutral) plus a waking-only DA lesion leaving the SWR DA edge intact. B2c
answers leg (c), a different leg, and does not technically depend on B3/B4 -- but the review's own order (section 7
lists B1-B4 under leg (b), C1-C3 under leg (c), in that sequence) puts leg (b)'s open items before leg (c)'s compute
spend, so this document's "Prepared commands" below do not run until B3 and B4 have landed on the pair's review.

## Residuals

- **The wall clock is untested.** Under the production default a battery's world time would be compute time. In live
  chat a pause of five minutes or more starts a sleep episode, with its SWR epoch. Leg (d) must exercise a real session
  with pauses on both sides of five minutes.
- **The battery barely exercises the pair's designed effect.** At F both night recalls abstain in base on seed 42, and
  two of the three slp facts parse. B2c checks for regressions; it does not measure the pair's benefit.
- **A multi-fact night is outside the rc GO's scope.** The slp group puts two or three facts into one night's replay
  and PRP budget. B2c reports that read and does not gate it.
- **F2's ON path needs an identity check before either default flips (fix round).** This battery's PASS does not
  include one -- see "Frozen revision F2 and registry" above for the diff sizes it is currently accepted on a
  docstring reading alone. Before a PASS here is used to flip `BRAIN_DA_TAG_CAPTURE` or `BRAIN_SLEEP_REPLAY_CAPTURE`
  default-ON, re-run at F2, at one seed: the rc family's arm (`{**ON, **RC}`) and the LTM-on DA Amendment 3 arm,
  compared turn-by-turn against their committed GO artifacts (`da_tag_capture`, `sleep_replay_capture` fields and
  the reply). Absent that check, a PASS below is scoped to "no regression in the registry at F2's code", not to "the
  code that earned the GOs is unchanged" -- and by itself does not license a flip.
- **R3 cannot detect forgetting of the facts that ARE stored (fix round).** See "Which rows may legitimately
  change" and "What a PASS licenses" -- R3 only reads the wolf/owl target, which is never reached, so a PASS here
  says nothing about overnight retention of fox/hare or hawk/vole.
- **LTM, pinned (fix round).** Both arms build whatever the F2 tree builds on the node; the pool's r7i.4xlarge nodes
  cannot hold the 100k-entry LTM tier (it exceeded pool RAM per the LTM-off GO), so in practice both arms build
  LTM-OFF. This is now recorded rather than left implicit: `research/coordination/b2c_smoke_check.py` and the
  aggregate's own `ltm_mode` field (already derived from `BRAIN_DATA_ROOT` presence in the sidecar) both report it,
  and **a PASS below is scoped to LTM-off** -- it says nothing about the LTM-on DA arm the flip-deciding GO ran (see
  the bullet above). The pair writes the buffer only regardless of tier (the LTM-on GO; `TieredFactStore` routes
  writes to the buffer), so LTM attachment cannot change what the ledger REWRITES, only what a recall turn can READ.
- **No row uses the pair's own lesion flags.** `BRAIN_DA_ENCODING_LESION` (da-gated-encoding's flag) also pins the
  ledger's D1 input, so in flipcand that row's lesion arm differs from base's in one more respect. R1 and R2 read it
  like any other row.
- **Hosts.** pool1 and pool2 are the same instance type. No cross-host numpy difference has been measured here; the
  pair re-run confirms any regression whose cells ran on different hosts.

## Prepared commands (not run by this document; run from the primary checkout after this branch is merged)

```
F2=fd29040db19987819461693aaf385977e45840ef
# -1. PRECONDITION (fix round, owner-directed order): verify-go items B3 and B4 on the DA-capture + sleep-replay
#     pair (research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md section 7, leg (b)) have
#     RUN -- B3 the registered wall-clock production-path day, B4 the salient-vs-neutral + waking-only-DA-lesion
#     contrast. NONE of the commands below run until both are landed and their own findings are committed; this is
#     a precondition on running this file's commands at all, not merely on interpreting the result.
# 0. precondition: tools/b2c_score.py (section "Scoring code") built, self-tested and merged, UNCHANGED, to
#    origin/main (git cat-file -e origin/main:tools/b2c_score.py && git diff --quiet origin/main -- tools/b2c_score.py)
# 1. provision F2 on the two AWS pool nodes only (ALLOW_STALE: main will have moved past F2; --isolated leaves
#    ~/derisk-pool/sim untouched)
POOL_PROVISION_ALLOW_STALE=1 bash tools/pool_provision.sh --revision $F2 --isolated pool1 pool2
# 2. corpus record (writes b2c0925_corpus_sha256.tsv only if every node matches); commit it
bash research/coordination/b2c_record_corpus.sh pool1 pool2
# 3. record PIN.txt for both tags and EXPECT_ENV.txt for flipcand, and prove the queue source unchanged
bash research/coordination/b2c_make_jobs.sh --record-pin research/coordination/b2c0925_jobs.check.txt
cmp research/coordination/b2c0925_jobs.check.txt research/coordination/b2c0925_jobs.txt && rm research/coordination/b2c0925_jobs.check.txt
#    (the two lb_shard calls that script makes, with the 43 frozen rows passed as --faculties <rows>:)
#    python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2c0925-base --no-fixes --probe-set adequate \
#        --repeats 2 --root '~/derisk-pool/revisions/'$F2 --faculties <rows> --pin $F2
#    python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2c0925-flipcand --no-fixes --probe-set adequate \
#        --repeats 2 --root '~/derisk-pool/revisions/'$F2 --faculties <rows> --pin $F2 \
#        --extra-env BRAIN_DA_TAG_CAPTURE=1 BRAIN_SLEEP_REPLAY_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn
# 4. integrity smoke (seed 7, direct ssh on pool1, hard 8 GB cap); --print shows the two commands first. Pulls the
#    remote out_dir back via rsync and scores it with research/coordination/b2c_smoke_check.py (rule-5-equivalent
#    checks: no error under da_tag_capture at any depth, sleep-replay n_epochs>=1 at slp_recall once a block is
#    managed, no logged tick-failure, LTM tier recorded) -- non-zero exit stops here, before wave 1.
bash research/coordination/b2c_smoke.sh --print pool1
bash research/coordination/b2c_smoke.sh pool1
# 5. waves: check, then add the second line to the heartbeat cycle beside b2b_queue_next_wave.sh; commit
#    research/coordination/b2c0925_waves.tsv whenever it moves
bash research/coordination/b2c_queue_next_wave.sh --dry-run
bash research/coordination/b2c_queue_next_wave.sh
# 6. aggregates (after the last re-run), then the registered scorer
python tools/lb_shard.py aggregate --tag b2c0925-base --seeds 42 43 44 100 101 102 --pin $F2 --expect-env
python tools/lb_shard.py aggregate --tag b2c0925-flipcand --seeds 42 43 44 100 101 102 --pin $F2 \
    --expect-env BRAIN_DA_TAG_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn BRAIN_SLEEP_REPLAY_CAPTURE=1
python tools/b2c_score.py --pin $F2 --out-dir research/findings/raw/_load_bearing/_b2c0925_score
```
