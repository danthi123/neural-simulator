---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: the v3 DA-gated synaptic tag-and-capture ledger (webapp/da_tag_capture.py SynapticTagCaptureLedger) wired into the live /api/brain-chat store path and the continuous engine's idle/sleep tick behind BRAIN_DA_TAG_CAPTURE (default OFF, webapp/da_tag_capture_chat.py), read as a next-day recall turn in the load-bearing battery
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_tag_capture_chat/design_seed7.json
---

# PRE-REGISTRATION — DA tag-and-capture wired into chat, with a next-day battery turn (2026-09-23)

Committed on its own, BEFORE any gate-seed run. The code it governs (wiring, battery groups, runner, gates) is in the
commit immediately before this one (`a201293f5`) on branch `research/da-tag-capture-chat-wire`; every constant below is
fixed there.
Terms follow `docs/TERMS.md`.

## Why

`da-gated-encoding` reads NOT load-bearing (0/6) in the #1-metric battery. Its probe is one fresh turn (`well`,
field `da_encoding.on`, True in both arms), and the battery has no next-day turn. The v3 mechanism acts on
PERSISTENCE (Bethus, Tse & Morris 2010: D1/D5 blockade spares encoding and immediate recall and changes ~24 h
retention), and it is GO only at runner level
(`2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level.md`): nothing in `/api/brain-chat`
built its ledger. So no battery turn could let the lesion show, whether or not the brain's contribution is real.

## What was built (default OFF, byte-identical off)

- `webapp/da_tag_capture_chat.py`: the SAME v3 ledger (same constants, same a-priori gamma calibration from the
  5-min Go-boundary protocol, same spiking D1 `write_gain` reader, seed 42) on the chat composer. Three hooks:
  `observe_chat_turn` after the DA-encoding install in `webapp/server.py` (integrate the store synapses to the turn's
  world time, then schedule this turn's D1 drive from the brain's own DA level); `after_store_chat` where the reply is
  assembled (register the turn's new store blocks: increment = the composer's own DA-gated write, baseline = the
  seeded pre-existing synapse strength); `tick_chat` in `continuous_engine.tick_idle_sessions` after the Turrigiano
  pass (night-time decay and capture run on the brain's own idle/sleep tick).
- `SynapticTagCaptureLedger` gains `block_offset` (build-time blocks are not managed) and `sync_from_store` (another
  writer's pure rescale is absorbed into base + increment; a non-scale rewrite is re-tagged as a fresh write). With
  `block_offset=0` and no sync call the v3 runner path is unchanged (asserted in the offcheck, below).
- Battery groups (label-only, `onebrain_regression_battery._EXTRA_TURNS`, none in `PROBE_TURNS`):
  `datc` salient telling -> night -> recall; `datn` neutral telling -> night -> recall; `datci` / `datni` the same
  tellings -> recall at once. The night is a WORLD STEP (`_WORLD_STEPS`), never a brain_chat turn: the environment
  clock jumps 24 h, then `continuous_engine.tick_idle_sessions` runs once at that time with the same four getters the
  server loop passes, so it is a sleep-depth tick (idle >= SLEEP_IDLE_SEC).
- `load_bearing_fraction.py` flag `LB_DA_TAG_CAPTURE_PROBE` (default OFF): remaps da-gated-encoding to `datc_recall`,
  fields `recalled_svo` / `abstained`, `BRAIN_DA_TAG_CAPTURE=1` + `BRAIN_DA_TAG_CAPTURE_CLOCK=turn` on BOTH arms; the
  lesion is the row's existing `BRAIN_DA_ENCODING_LESION`.
- `research/runners/_da_tag_capture_chat_probe.py`: the 10-arm per-seed runner, `grade_seed` (the gates below),
  `--aggregate`, `--selftest`, `--offcheck`.

### The conversations (the world; texts fixed in `onebrain_regression_battery`)

SALIENT (`datc`, `datci`): "Guess what, something unbelievable happened at the circus today!" / "You will never
believe this crazy story, it is absolutely amazing!" / "the cat chases the ball" / "Everyone in the audience was
screaming and laughing in total shock!" / "Honestly it was the most astonishing spectacle anybody has ever witnessed!"

NEUTRAL (`datn`, `datni`): "the cat is here" / "the ball is here" / "the cat chases the ball" / "the cat is here" /
"the ball is here".

Recall: "what does the cat chase". Both tellings have five turns (identical world time); the stored fact is the
identical plain sentence in both. Only the surrounding content differs, which is the natural manipulation (behavioral
tagging: a novel/salient event near a weak encoding; Moncada & Viola 2007; Wang, Redondo & Morris 2010).

### Design-time observations (seen BEFORE this pre-registration; seed 7, not an evaluation seed)

Artifact: `research/findings/raw/_da_tag_capture_chat/design_seed7.json` (three exploratory worker runs, seed 7,
`BRAIN_LTM_SHIP_DEFAULT=0`). These chose the fact and set G6's ratio; no evaluation seed was run.

- The fact must be comprehensible to the tiny-demo brain. "the zebra swallowed the violin" (the v3 fact) is
  short-circuited by the comprehension monitor as out-of-vocabulary words and never stored; "the bird eats the worm"
  is short-circuited as role-ambiguous (two animate nouns). "the cat chases the ball" (in-vocabulary, animate agent,
  inanimate patient, a new binding: the build-time fact is dog-chase-cat) was comprehended and stored in both tellings.
<!--derived-->
- Seed 7 (values rounded from the design artifact), companion ON, one process, all four groups: salient DA 0.79 / 0.94 / 0.76 (fact) / 0.93 / 1.04, D1
  activation 0.41 / 0.66 / 0.31 / 0.53 / 0.75, PRP p_max 0.021; neutral DA 0.38 / 0.51 / 0.49 / 0.20 / 0.05, D1
  activation 0 / 0.10 / 0 / 0 / 0, p_max 0.0008. Immediate recall correct in both tellings. After the night the
  salient fact block had every synapse's z above 1/2 and was recalled; the neutral fact block had z ~ 0 and the brain
  abstained. So the design is not predicted to fail on seed 7; it is not known on the evaluation seeds.
- The D1 pool read at DA 0.507 gave activation 0.10 (the tonic-rate noise floor), so a lesioned arm (DA seen by D1
  pinned to 0.5 on every turn) can accumulate a p_max of a few thousandths against the intact ~0.02. G6's ratio is
  therefore 0.25, not v3's 0.10 (v3's intact p_max was ~3x larger because its salient conversation was 8 min long).
- A tiny-demo brain WITH the default LTM tier (wikidata_100k) exceeded an 11 GB memory cap while loading and could not
  be run under the RAM limits of this box or the 15 GB pool nodes. Without it a build peaks near 1.1 GB. The arms
  therefore run with `BRAIN_LTM_SHIP_DEFAULT=0` (below).

### LTM tier off in the measured arms (declared deviation from the production default)

The tiny-demo production brain wraps its composer in `TieredFactStore(buffer, ltm)`. `_maybe_acquire` writes the
conversation's facts to the BUFFER composer, which is the only store the ledger manages (`store_composer` unwraps the
wrapper); a buffer abstain falls through to the routed wikidata shard. The measured arms turn the LTM tier off on every
arm (`--ltm off`, `BRAIN_LTM_SHIP_DEFAULT=0`), so a next-day buffer abstain is final. The battery flag
`LB_DA_TAG_CAPTURE_PROBE` keeps the production default (LTM on); its reading under LTM is NOT covered by these gates and
is reported only when a battery run with the flag happens on a machine with the RAM.

## Arms (per seed; each a fresh brain in its own subprocess; seed threaded by BRAIN_CHAT_SEED)

ON = {BRAIN_DA_TAG_CAPTURE=1, BRAIN_DA_TAG_CAPTURE_CLOCK=turn}; OFF = {BRAIN_DA_TAG_CAPTURE=0, ...CLOCK=turn};
LES = {BRAIN_DA_ENCODING_LESION=1}.

| arm | group | env |
|---|---|---|
| sal_night_intact_a, sal_night_intact_b | datc | ON |
| sal_night_lesion | datc | ON + LES |
| neu_night_intact, neu_night_lesion | datn | ON / ON + LES |
| sal_imm_intact, sal_imm_lesion | datci | ON / ON + LES |
| neu_imm_intact | datni | ON |
| sal_night_off_intact, sal_night_off_lesion | datc | OFF / OFF + LES |

Recall outcome of the recall turn: `correct` (recalled_svo == [cat, chase, ball]), `abstain` (recalled_svo
null and abstained), `confab` (any other recalled_svo), `undefined` (error / anything else).

## Gates (per seed; `grade_seed` implements them verbatim)

- **G0 null control.** sal_night_intact_a and _b give the same recall outcome, recalled_svo, abstained, ledger state
  at recall (world_t_h, n_turns, p, p_max, n_managed_blocks, rescales, rewrites, gamma, a_go) and fact-block summary.
  Failing -> seed UNDEFINED.
- **P1 precondition.** sal_imm_intact and neu_imm_intact both `correct` (the fact is stored and immediately readable
  in both tellings). Failing -> seed UNDEFINED (never a pass or a fail).
- **G1 the load-bearing claim.** sal_night_intact_a `correct` AND sal_night_lesion `abstain`.
- **G2 the lesion spares immediate recall.** sal_imm_lesion `correct`.
- **G3 selectivity.** neu_night_intact `abstain` (the plainly-told fact is not kept overnight).
- **G4 companion-off contrast.** sal_night_off_intact and sal_night_off_lesion give the same outcome and
  recalled_svo (without the companion the lesion does not change the next-day reply: today's battery reading).
- **G5 no confabulation** in any arm.
- **G6 the lesion held at measurement.** sal_night_lesion PRP p_max < 0.25 x sal_night_intact_a p_max. Intact p_max
  0 or missing -> seed UNDEFINED.
- Any arm error or `undefined` outcome -> seed UNDEFINED.

Seed verdict: UNDEFINED if any UNDEFINED condition holds; else GO iff G1-G6 all hold; else NO-GO.
Aggregate (6 seeds): GO iff 6/6 seeds GO; PARTIAL iff 4-5 GO and no seed NO-GO; otherwise NO-GO; INCOMPLETE if any
seed is missing. Reported, not gated: `tools.lab.attributable_to` of the recall-decision change (lesion vs null rebuild;
lesion effect with the companion ON vs OFF), one-sided sign-flip p over seeds (intact vs lesion; salient vs neutral), the
per-turn DA levels and D1 activations, the fact block's z state.

## What each gate can and cannot show (honest in advance)

- **The lesion half of G1 follows from the wiring once the edge is cut**: BRAIN_DA_ENCODING_LESION pins the DA the D1
  pool sees to tonic, the pool fires at its tonic rate (a ~ 0), no PRP is made, the fact's early phase decays. That
  half is an integrity check of the lesion, not evidence. The evidence is the INTACT arm: whether the brain's own DA
  over this conversation, read by the spiking D1 pool, drives the per-synapse late phase past its unstable point
  under the pre-registered gamma (a real failure mode: too little or too short a drive and the intact arm abstains
  too), together with G3 (a plain telling must NOT be kept), G2 (the lesion must not damage encoding) and G4 (without
  the companion the lesion must not change the reply).
- **G3 largely tests the DA contrast**, as in v3: if the neutral telling holds DA at or below tonic, the D1 pool gets
  no drive and G3 cannot fail through the synaptic dynamics. Reported per seed via the neutral turns' DA levels.
- **A GO makes da-gated-encoding load-bearing on the battery ONLY under the opt-in flag pair** (LB_DA_TAG_CAPTURE_PROBE
  + BRAIN_DA_TAG_CAPTURE): the robust core is not grown by a default-off flag.

## Host shortcuts in the decision path (declared)

- The per-synapse tag / PRP / late-phase equations are host-integrated ODEs (v3 constants, pre-registered there).
- The D1 rate -> activation normalization is host arithmetic on a measured spike count.
- TURN_DRIVE_H: one observed turn drives the D1 pool for a fixed 30 s of world time (the environment's turn length);
  a turn that short-circuits before the DA read drives nothing.
- `sync_from_store` bookkeeping; unmanaged build-time blocks (block_offset).
- The world clock and the 24 h night (environment). The battery's scripted "turn" clock makes world time independent
  of machine speed.
- The parse boundary (`_maybe_acquire`'s host SVO extractor + lemmatizer) that decides the fact is an assertion.
The late-phase variable is a synaptic state, not a replay path: a GO here is not "consolidation" in the TERMS sense.

## Byte-identical off (asserted in data before the gate seeds run)

`--offcheck --pinned-sha f35196e66` runs the salient next-day conversation in-process with BRAIN_DA_TAG_CAPTURE unset
on a `git archive` of the pinned pre-change SHA and on the branch, and requires EXACT sha256 equality of every reply
(sorted-key JSON) and of the composer's store synapses, plus an identical hash for a deterministic synthetic v3-ledger
scenario (block_offset=0, no sync) on both trees. Both trees run with `BRAIN_LTM_SHIP_DEFAULT=0` (as the arms). If the
pinned tree itself is not reproducible run-to-run the check reads UNDEFINED, never a pass. Artifact:
`offcheck.json` in the same raw directory as the design artifact (written before the seed-42 run).

## Compute

Local 1-seed smoke (seed 42, `--ltm off`) under `tools/mem_ok.sh 12` + `tools/memcap.sh 12`; it is the seed-42 gate
run, not a separate exploratory run. Then seeds 43, 44, 100, 101, 102 staged on the pool (`pool41`, `pool42`), one seed
per job, `--workers 3`, output `research/findings/raw/_da_tag_capture_chat/seed<s>.json`, then `--aggregate`.

## AMENDMENT LOG

### Amendment 1 (2026-09-23) — the D1 reader was NOT isolated from the shared production write-gain cache

**What had already been seen before this amendment** (adversarial review v2:dd14adaf7 of commit `caf0c9a0b`,
against the seed-42 artifact in `research/findings/raw/_da_tag_capture_chat/seed42.json`): every companion-ON
arm's `tag_capture_at_recall` reports `gamma` and `d1_a_go` from `ChatTagCapture`'s D1 read. The intact arms read
`gamma=46.549, d1_a_go=0.1314`; all three lesion arms (`sal_night_lesion`, `neu_night_lesion`, `sal_imm_lesion`) <!--derived-->
read `gamma=32.774, d1_a_go=0.1867` instead of the SAME value the intact arms got. A synthetic reproduction (no <!--derived-->
brain, in a throwaway process) confirmed the mechanism: `ChatTagCapture` built its `SpikingD1Activation` via
`_da_write_gain_spiking_derisk._get_reader(seed, False)`, the identical `(seed, lesion)` cache key production's
own intact-arm `spiking_write_gain` read uses when `BRAIN_DA_ENCODING_SPIKING_GAIN` is on (default). Whichever
caller reached that cache entry FIRST in the process decided its calibration. In an intact arm production's own
gain read typically built it first, consuming whatever ambient (non-reseeded) global RNG state the rest of the
turn had left; in a `BRAIN_DA_ENCODING_LESION` arm production pins the gain to 1.0 and never calls `_get_reader`
at all, so `ChatTagCapture` was always the sole, first builder there, deterministically inside `_private_rng`.
So the "same a-priori gamma calibration" claim above and the module docstring's "never perturbs another organ's
RNG stream" claim were both FALSE for the shared-cache path: this is a build-order confound, not evidence that
the DA-gate lesion itself changes the ledger's calibration, and it broke `docs/BUILD_LANE_CHECKLIST.md`'s "lesion
the SPECIFIC claimed edge, hold everything else byte-identical."

**Fix (commit `daa4b382d`, this branch).** `_da_write_gain_spiking_derisk._get_isolated_reader(seed, tag)`: a
cache namespace production's write-gain path never reads or writes. `SpikingD1Activation(..., isolated=True)`
builds/fetches from it instead of the shared `(seed, lesion)` cache; `ChatTagCapture` now passes `isolated=True`
(`isolated_tag="da_tag_capture_chat"`). Verified in a throwaway process: reproducing the ambient-RNG-consumption
asymmetry above with the OLD code gives the exact reported gamma/d1_a_go split (46.549/0.1314 vs 32.774/0.1867); <!--derived-->
with the NEW isolated path both arms read `gamma=32.774, d1_a_go=0.1867` — identical. <!--derived-->

**Standing check added.** `grade_seed` (`research/runners/_da_tag_capture_chat_probe.py`) now computes
`G_isolation_gamma_consistent`: gamma and d1_a_go must be equal (abs diff < 1e-6) across every companion-ON arm
at a seed; a mismatch makes the WHOLE SEED `UNDEFINED` (folded into the existing `undefined` condition, never
silently scored GO or NO-GO on a confounded calibration). A record with no `env` / no `gamma` (the pre-existing
synthetic `grade_seed` selftest patterns, or any older artifact predating this field) is exempt, not penalized,
so old coverage does not regress. Selftest adds 4 checks proving both directions (consistent -> gate passes and
the seed reads its ordinary verdict; one companion-ON arm's gamma differs -> the gate fails and the seed reads
UNDEFINED even though every other gate's inputs still match the designed-GO pattern); 23/23 selftest checks pass.

**What this amendment does NOT change:** the gates (G0-G6), the arms, the conversations, the fact, or any
threshold above — only the D1 reader's cache isolation and the new standing consistency check. No `sim/` edit,
no default flipped.

**Governs:** a fresh seed-42 re-run (superseding the confounded `seed42.json`, which is retained on disk for the
record but its `tag_capture_at_recall.gamma`/`d1_a_go` values are VOID per this amendment — do not cite them), a
re-run of `--offcheck` from a clean committed branch HEAD (the prior `offcheck.json` was run against an
uncommitted working tree 74 s before its own code commit, per the same review, and predates the origin/main merge
that followed), and the 5 seeds (43/44/100/101/102) restaged at the fixed revision. The 5 lines previously staged
at `d4484acfc` are confounded by the same defect and are superseded, not reused.

**LTM-on (production default) configuration:** unmeasured, as already declared above under "LTM tier off in the
measured arms" — restated here because the review asked this be stated plainly rather than left implicit: no
gate in this document, before or after this amendment, reads the LTM-on battery row, and none should be reported
as measured until a run under `BRAIN_LTM_SHIP_DEFAULT` unset (or `=1`) actually executes on hardware with enough
RAM for the default LTM tier.

### Amendment 2 (2026-09-24) — `aggregate()` trusted a stale stored verdict; a false "re-verified" claim; the
### `--offcheck` pin predated this branch's own merges; two more artifacts from the pre-fix run are VOID

**What had already been seen before this amendment** (adversarial re-review v2:2a37f2493 of commit `5d3810f2d`,
`safe_to_merge: false`, `verdict: fix-required`):

**1. RECORD CORRECTNESS (blocking).** `aggregate()` (`research/runners/_da_tag_capture_chat_probe.py`) read each
seed's STORED `gates.seed_verdict` off disk and never re-graded it. The confounded
`research/findings/raw/_da_tag_capture_chat/seed42.json` (run at `c4c62d066`, before the D1-reader-isolation fix
`daa4b382d`) stores `seed_verdict: "GO"`. Re-grading its own `arms` data with the CURRENT `grade_seed` reads
`UNDEFINED` (`G_isolation_gamma_consistent` is `False`: the confound Amendment 1 fixed is present in exactly this
artifact). No post-fix seed-42 run was staged into the production `--out` dir at the time of the review, so once
the 5 pool seeds landed, `--aggregate research/findings/raw/_da_tag_capture_chat` would have reported
`complete: True` with a confounded seed-42 `GO` silently counted.

**Fix.** `aggregate()` now calls `r["gates"] = grade_seed(r)` on every row before reading `seed_verdict`, so a
stale on-disk grade from before a grading-logic fix can never outlive that fix. Verified directly against the
real artifact: `--aggregate` over a directory holding only the unmodified, still-`GO`-labelled
`seed42.json` now reports `seed_verdicts: {"42": "UNDEFINED"}` (run on pool41 against a throwaway hard-linked copy
of the revision tree with only this file swapped in, so the shared revision the pool dispatcher uses was not
touched). **Test added**: `tests/test_da_tag_capture_chat_aggregate.py`
(`test_aggregate_regrades_a_stale_stored_go_row_as_undefined`,
`test_aggregate_regrading_is_idempotent_for_an_already_current_row`), plus a matching `--selftest` check
(`"aggregate: re-grades a stale stored-GO row to the current UNDEFINED verdict"`); 24/24 selftest checks pass.

**2. FALSE CLAIM in the staged queue metadata (corrected here, not by editing the live queue file).** The 5 pool
lines restaged by Amendment 1 (and the one still queued for seed 102 as of this amendment) carry `#checked:` text
that reads "...seed 42 re-verified under the fix." **This was false when written and remains false as of this
amendment** (2026-09-24T04:34Z): the only seed-42 run under the fix is the supplementary run in
`research/findings/raw/_da_tag_capture_chat_verify/seed42/`, still in flight on pool41 (PID `4069973`, 3rd of 10
arms building at last check), with no `seed42.json` written yet. It has NOT landed. This document, not the queue
file, is the durable record of that correction; `research/queue/*` is live dispatcher state and out of scope for
this branch's commits. Once that run completes, harvest it as the production `seed42.json` (superseding the
confounded one per Amendment 1) before the 6-seed `--aggregate` is read as a verdict.

**3. `--offcheck` baseline predated this branch's own two `origin/main` merges.** `PINNED_SHA` (and the module
docstring's example) defaulted to `f35196e66`, a SHA from BEFORE this branch merged `origin/main` twice (`975165f26`
at `36a175534`, and the branch now sits on `origin/main` further still). `git diff --stat f35196e66 36a175534`
touches `webapp/server.py` (+55/-1); a difference against `f35196e66` cannot be attributed to this branch's own
change. **Fix:** `PINNED_SHA = "36a175534"`, the merge-base of this branch and `origin/main` (verified via
`git merge-base HEAD origin/main`). **Disclosure, not yet resolved:** a re-run of `--offcheck` from a clean
committed branch HEAD against the OLD (wrong) pin `f35196e66`, launched by the prior fix round (PID `2767315`,
log only, never committed as an artifact) completed with `replies_identical: false` / `byte_identical_off: false`
— a MISMATCH. Given point 3's own reasoning, this mismatch is not interpretable as evidence about this branch
(the pin itself was wrong), but it is also not yet superseded by a run at the correct pin: **a re-run of
`--offcheck --pinned-sha 36a175534` is PENDING**, blocked as of this amendment by sustained local RAM contention
(`tools/mem_ok.sh` refused 4 GB, 6 GB and 12 GB locally; `git archive` of an arbitrary SHA needs a full git
checkout, which the CPU pool nodes do not have, so this check cannot be offloaded to the pool). **This branch's
merge is NOT byte-identical-off-verified at the correct pin yet** — do not read the OFF path as unchanged until
`offcheck.json` exists in `research/findings/raw/_da_tag_capture_chat/` at `pinned_sha: "36a175534"` reporting
`byte_identical_off: true`.

**4. Voiding scope extended.** Amendment 1 voided `seed42.json`'s `tag_capture_at_recall.gamma`/`d1_a_go` fields.
The same confounded pre-fix run (`d4484acfc`, before `daa4b382d`) also produced two more artifacts that must be
read as VOID for the same reason, not cited as evidence of anything about the D1-isolation fix or about
`da-gated-encoding`'s load-bearing status: `research/findings/raw/_da_tag_capture_chat/lbf_row_s42_ltmoff` (the
`load_bearing_fraction --only da-gated-encoding` battery row for seed 42, `git_sha: d4484acfc`, `git_dirty: true`,
`load_bearing: true` — this row does not exercise `ChatTagCapture`'s shared-cache path directly, but it was built
in the same dirty, pre-isolation-fix working tree and is superseded by the same revision boundary), and every file
under `research/findings/raw/_da_tag_capture_chat/seed42/` (the ten per-arm response JSONs `seed42.json`'s
`gates` were computed from). No document currently cites either path, but this amendment names them VOID now so
none does later without the same `⛔` this amendment carries.

**What this amendment does NOT change:** the gates (G0-G6, `G_isolation_gamma_consistent`), the arms, the
conversations, the fact, or any threshold above — only `aggregate()`'s re-grading, the `--offcheck` pin, and the
voiding scope. No `sim/` edit, no default flipped, no `research/queue/*` edit.

**Governs:** every future `--aggregate` read of `research/findings/raw/_da_tag_capture_chat` (must re-grade, per
point 1); the harvested seed-42 artifact once the in-flight `_verify` run lands (per point 2); a future
`--offcheck` run (must use `--pinned-sha 36a175534`, per point 3); and any reader of `lbf_row_s42_ltmoff` or
`research/findings/raw/_da_tag_capture_chat/seed42/*.json` (VOID, per point 4).

### Amendment 3 (2026-09-24, branch `research/da-tag-capture-ltm-on`) — registering the flip-deciding LTM-ON
### arm, and a new REPORTED (non-gating) diagnostic for board #227 item (c): does the flip cost an ordinary
### fact its overnight survival relative to today's production default?

Committed on its own, BEFORE either of the two things it registers has been run. Governs the code committed
immediately before it on this branch (the two `research/runners/_da_tag_capture_chat_probe.py` additions
below); every constant is fixed there.

**1. The LTM-ON arm (the flip-deciding read, restated as a formal registration).** The 2026-09-24 GO finding
(`2026-09-24-da-tag-capture-chat-wire-6seed-GO-runner-level-ltm-off.md`) scored ONLY the buffer-only arm
(`--ltm off`, `BRAIN_LTM_SHIP_DEFAULT=0`) on all 6 seeds; the production default attaches the tiered
`wikidata_100k` LTM (`BRAIN_LTM_SHIP_DEFAULT` unset, `--ltm on` on this runner), and that configuration has
never been run. This amendment registers it under the IDENTICAL instrument, unchanged: the same 10 (now 11,
per point 2) arms, the same seeds `[42, 43, 44, 100, 101, 102]`, the same gates (`G0`, `P1`, `G1`-`G6`,
`G_isolation_gamma_consistent`) verbatim from `grade_seed`, and the same `--aggregate` 6/6 combine rule. No
gate, arm, seed, fact, or conversation text differs between the LTM-off and LTM-on registrations; only the
`--ltm` flag (and therefore whether `TieredFactStore` wraps the buffer in a routed `wikidata_100k` shard)
differs. A `GO` under this registration is the flip-deciding read the prior finding's own "what this GO does
NOT show" section named as still outstanding; a `NO-GO` or `UNDEFINED` blocks the flip exactly as it would
have under the LTM-off registration.

**Declared reasoning for why the buffer-only measurements below (point 2) are expected, not merely hoped, to
transfer to LTM-on unchanged (checked, not assumed, by running point 2's new arm under `--ltm on` too, in the
SAME 6-seed batch as point 1 -- no separate run).** `TieredFactStore.store()` (`research/runners/
tiered_fact_store.py`) routes every WRITE to the buffer only; the LTM shard is read-only fallback on a buffer
ABSTAIN (`_tiered`). `webapp/da_tag_capture_chat.py` `store_composer()` unwraps `TieredFactStore` to
`.buffer` explicitly and the ledger only ever manages blocks written to THAT composer. So whether the LTM tier
is attached cannot change which blocks the ledger manages or how they decay -- the LTM tier is inert with
respect to this mechanism by construction, not by measurement. The one thing that could differ is DA level at
the TELLING turn (if an LTM-backed recall earlier in the conversation changed downstream affect/expectation
state) -- none of the arms below query the LTM before telling the fact, so this channel is not exercised
either. Both premises are checked directly by running the SAME `neu_night_off_intact` / `neu_night_intact`
contrast under `--ltm on`, not left as an unverified inference.

**2. `neu_night_off_intact` + `ordinary_fact_flip_forgetting` (board #227 item (c), REPORTED, never gating).**
Code added this branch, before this amendment, in `research/runners/_da_tag_capture_chat_probe.py`:
- A new arm, `neu_night_off_intact` (group `datn`, env `OFF`): the plain telling's OWN companion-OFF control
  -- today's production default (no DA-tag-capture wiring reachable at all) tells the SAME neutral fact,
  sleeps, and is asked. The salient group already had this OFF control (`sal_night_off_intact`); the neutral
  group did not, so nothing in this instrument could show what the flip actually costs an ORDINARY fact
  relative to today's baseline.
- A new `grade_seed` field, `ordinary_fact_flip_forgetting`: `True` iff `neu_night_off_intact` recalls
  correctly (today's baseline: the plain fact is never touched by anything that decays it) AND
  `neu_night_intact` does not (the companion-ON arm; `G3_neutral_not_kept` already requires this arm to
  ABSTAIN as the mechanism's OWN by-design selectivity). `False` when the ON arm ALSO recalls correctly (no
  cost). `None` on any `seed*.json` committed before this branch (the arm did not exist; `aggregate()`'s
  re-grade of the six already-committed LTM-off files must read `None` here, not crash or silently score 0).

**Why `True` here is an EXPECTED reading of a working mechanism, not evidence against the GO.** `G3` already
requires the companion-ON neutral arm to abstain overnight -- that is the behavioral-tagging selectivity the
whole mechanism is FOR (Moncada & Viola 2007; a plain telling near no salient event is not consolidated). A
`True` `ordinary_fact_flip_forgetting` reading on every seed is therefore the ALREADY-KNOWN G3 result restated
from the flip's own vantage point: it makes explicit, in the permanent record, that flipping
`BRAIN_DA_TAG_CAPTURE` to production-default trades "every buffer-taught fact persists indefinitely" (today,
measured: `research/runners/rf_phasor_composer.py`'s store has no decay path and `TieredFactStore.
promote_buffer_to_ltm()` is declared "NOT auto-invoked in v1", so nothing removes a buffer entry absent this
ledger) for "only a DA-salient telling persists." This is a real, load-bearing behavior change the flip
decision must weigh with eyes open -- board #227 item (c) asks for exactly this visibility, not a fix. No
other existing route was found to already cover it: `BRAIN_SLEEP_REPLAY` (`2026-08-26-gap5-sleep-replay-
production-wirein-GO.md`) reactivates the EPISODIC organ's CA3 topic assemblies, a structurally separate
store from the composer `store_conns` this ledger manages, and would not rescue a forgotten SVO fact here.

**Excluded from `core`/`seed_verdict` by construction** (verified in `--selftest`, both directions): this
field can never flip a seed between `GO`/`NO-GO`/`UNDEFINED`, and its `None` reading on an old artifact is
inert under `aggregate()`'s re-grade. It changes no threshold, arm, or gate this document already registered.

**Compute (registered before any run).** Seed 42 first, solo, to obtain a MEASURED (not estimated) peak RSS
for `--ltm on` before dispatching the other 5 -- this box's own convention (`docs/BUILD_LANE_CHECKLIST.md`
"pool lines declare their memory... a measured peak, rounded up"). Design-time evidence already on record
(this document's own "LTM tier off in the measured arms" section) put a full `--ltm on` build over an 11 GB
cap and over the 15 GB pool-node cap at BUILD time; the isolated `ShardedPhasorStore` build at the same
78,857-fact scale separately measures ~3.78 GB marginal RSS above a ~1.1 GB no-LTM baseline
(`2026-09-05-rank6-knowledge-core-substrate-write-scaled-derisk-mixed.md`), so the empirically-observed >11 GB
full-build cost is NOT fully explained by the store's own marginal footprint alone -- an open gap, not
papered over with a confident number. Runs on the AWS `r7i.4xlarge` CPU pool (128 GB RAM, 16 vCPU; declared
in `GAP_CLOSURE_MISSION.md`'s compute-lanes section), `SIM_BACKEND=numpy`, one seed per instance-job, each
under `tools/memcap.sh` at a cap this document does not fix in advance (the seed-42 job picks a conservative
cap given the ~128 GB ceiling; the remaining 5 jobs' cap is corrected to the seed-42 MEASURED peak before they
are queued, per the same convention `research/FAILURE_LOG.md` already recorded a violation of on
2026-09-24 for a different lane's guessed `mem_gb`). No job in this compute plan is queued by this amendment.

**What this amendment does NOT change:** the gates (`G0`-`G6`, `G_isolation_gamma_consistent`), the existing
arms, the conversations, the fact, or any threshold in this document or Amendments 1-2 -- only registers the
LTM-ON arm and adds the new arm + reported field above. No `sim/` edit, no default flipped, no existing
`seed*.json`'s stored verdict is read as anything other than what Amendment 2's re-grade already made it.

**Governs:** the LTM-ON 6-seed run (point 1) once queued; the `neu_night_off_intact` arm and
`ordinary_fact_flip_forgetting` field on every run after this commit (point 2); and any reader of the prior
GO finding, who must read its "what this GO does NOT show" LTM-on caveat as still accurate until point 1's
own run lands and is scored.
