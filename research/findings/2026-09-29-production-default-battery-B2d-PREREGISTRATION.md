---
type: preregistration
status: preregistered
date: 2026-09-29
lane: load-bearing
mechanism: the #1 metric (lesion-verified load-bearing fraction) on the full 50-row registry at H (origin/main
  3c2907b85c10fe918ddc81b4c69de8888c9f321b), production default, with the self-describing lesion fields removed from
  the decision fields by a post-hoc scorer over the arm files, and a reply-level secondary score beside it
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any b2d0929 job, shard or smoke exists. H is the candidate pin, the
  origin/main HEAD at 13:55 EDT on 2026-09-29; the commit that adds this file is a documentation-only child of H.
  H is replaced only by the rule in "Frozen revision" below, by an amendment committed before any line is queued.
---

# Battery B2d: the full registry at H, decision fields without the lesion's own labels, and a reply-level score (pre-registration)

## Why

B2b (`research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md`, Amendments 1 and 2) measures
the 50-row registry at F = a308f1e09 at the production default. An audit of its shards found that 26 coverable rows
read load-bearing on all six seeds by the registered decision fields, but only 16 of them change the reply itself on
the probe turn. Re-derived at filing from the B2b arm files (read-only, method in "Feasibility measured at filing"):
the same 26 and 16.

Part of the gap is built into the instrument. Some decision fields describe the lesion instead of the reply: the
lesion branch of the organ writes them. `da_drives.reason` reads `lesion_collapsed` in every lesion arm of
da-mode-drives-response. A lesion that flips only such a field is load-bearing by construction.

B2d measures the registry at H and adds two things (owner-coordinator directive, 2026-09-29):
1. The headline load-bearing verdict is recomputed with the self-describing fields removed from each row's decision
   fields. The per-row lists are in `research/coordination/b2d_decision_fields.json`.
2. A reply-level secondary score: does the lesion change the reply itself on the probe turn, beyond the noise floor
   that the two intact arms and the two lesion arms give.

The plan for this phase (`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md` §2) asks for host share to be
reported "beside, not instead of" the load-bearing fraction. B2d follows the same rule for its own secondary score,
and its gap rows are an input to that plan's seam map (§1).

## Frozen revision and registry

- **H = `3c2907b85c10fe918ddc81b4c69de8888c9f321b`**, origin/main when this was filed (fetched 13:55 EDT; main had
  not moved since 12:19). Every job runs from `~/derisk-pool/revisions/<H>`.
- **Re-pin rule.** `tools/pool_provision.sh` refuses a revision that does not contain origin/main (lines 60-69), and
  `POOL_PROVISION_ALLOW_STALE` is never used for B2d. If origin/main has moved past H when provisioning runs, H is
  replaced by the origin/main HEAD at that moment (H') through an amendment committed before any line is queued. The
  amendment regenerates `b2d_jobs.txt` and `PIN.txt` with `--pin H'`, recomputes the registry at H' and names any
  change, compares the git blob of every file that `b2d_decision_fields.json` cites between H and H' and re-audits
  each row whose cited file changed, and repeats the identity replay (below) at H'.
- **Registry at H**, computed by importing `research/runners/load_bearing_fraction.py` at H in a git worktree with
  the primary venv and `SIM_NO_PROVENANCE=1`: 50 rows <!--derived-->. Kinds: 33 neural-lesion, 3 whether-disable,
  6 thin, 1 mechanism-only, 4 in-process, 3 proposed. Coverable (neural-lesion and whether-disable, in the
  fraction): 36. Sharded (`lb_shard.py` MEASURABLE_KINDS): 43 per seed, so 258 shards <!--derived-->.
  `LBF_ROW_MERGE_REPORT`: modules learning, live_organs, reward_value_afferent; no import error; the 8
  proposed_lesions_conflict_kb collisions dropped as at F; parked: self-schema (by its module), and the opt-in rows of
  open_ended_gated (3), reasoning_transitive_chat and tom_false_belief_chat, which need their REQUIRED_ENV flag.
- **Against F, measured, not read from commit messages:** the F registry was imported the same way from a
  `git archive` of F. The two are equal in every row key, kind, lesion flag and lesion value, in every
  FACULTY_PROBES row (probe turn, decision fields, thin), and, for the 35 coverable rows measured through brain_chat,
  in the turn group each row's effective probe turn resolves to (turn labels, messages, sessions, resets, percepts,
  rich flags, with the adequate-probe remaps applied). No row is added, removed or changed. What changed between F
  and H is code outside the registry:
  - `969894855` (2026-09-25, owner decision) turns the affect-marker word off the answer surface by default
    (`BRAIN_AFFECT_MARKER_SURFACE`, default OFF, `webapp/affect_drives_chat.py:213`, gating two prepend sites in
    `webapp/server.py`). The marker is still computed and recorded in `affect_drives.lead`. This is the one
    default-path change between F and H that the log names.
  - `eed3652a0` adds the REQUIRED_ENV opt-in mechanism to the row hook; at production defaults the opt-in rows stay
    out, so the registry is unchanged.
  - `5ce76fdfd` gives the open-ended-generation parent-written file its own sidecar (`declare_output`).
  - About 80 other commits touch `webapp/` or `research/runners/`. Their commit messages describe them as
    default-OFF. B2d is the first measurement of that claim on these 35 probe groups (see "Comparison with B2b").

## Arm (fixed)

One arm, tag **b2d0929**. `--no-fixes` (every mechanism at its production default), the adequate probe set (the
`lb_shard.py` default: nine LB_* probe flags, `tools/lb_shard.py:38-41`), `--repeats 2`, numpy backend, one
BLAS/OMP thread, seeds 42 43 44 100 101 102, the 50-row registry at H.

The job file is `research/coordination/b2d_jobs.txt`, 258 lines, generated at filing with
`SIM_NO_PROVENANCE=1 .venv/bin/python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2d0929 --no-fixes
--probe-set adequate --repeats 2 --root '~/derisk-pool/revisions/<H>' --pin <H>`, then
`.venv/bin/python tools/assert_flipped_defaults.py && ` inserted after the `cd`, as B2b did. `lb_shard.py jobs --pin`
wrote `research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt` (H in full), which `aggregate` reads as its
default pin. Checks at filing:
- with the pin replaced by F and the tag by `b2b0924-base`, the file is identical to
  `research/coordination/b2b0924_base_jobs.txt` line for line (`diff`, no output);
- no line carries a `BRAIN_*=` token; all 258 end in `/lb.json`;
- `queue_job_runnable_check` (`tools/queue_job_shape_check.sh`) returns 0 for lines 1, 2, 37, 44, 90 and 258, bare
  and with the `#checked:` reason below appended, and refuses a negative control (line 44 with `B2d s43: ` in front);
- `tools/assert_flipped_defaults.py` exits 0 at H with no BRAIN_* variable set.

## Validity, causes, re-runs and verdict (B2b Amendment 1 carried forward)

Each rule of B2b Amendment 1 applies to B2d with F replaced by H and `b2b0924-base` by `b2d0929`, except where a
change is stated.

- **A1.2 validity of a cell**, rules 1 to 5, unchanged: `lb.json` at its `--out` path; its sidecar and every arm
  sidecar record `git_sha` H in full, `source_kind` git_archive, both manifest checks true and `SIM_BACKEND=numpy`;
  no BRAIN_* key except BRAIN_CHAT_SEED in arm sidecars and none in the `lb.json` sidecar; the node's corpus hash is
  the recorded one; the report is not UNRELIABLE, `null_control_clean` is not False, and the verdict is `regressed`,
  `pass` or `not-exercised`. Added: **rule 7**, no `error` key in the probe-turn parent dict of a compared field (see
  "Self-describing fields at H"). Added: **rule 6**, the arm files the scorer reads (the probe group's `intact_a_*`,
  `intact_b_*`, `lesion_*.json` and `lesion_*.json.rep0`) all exist in the cell directory. `cell_prov_fails` checks
  the sidecars of the files that are present; it does not notice a file that is absent.
- **A1.3 cause classes E, C and I**, unchanged, with one named sub-case of E, **E-stranded**: the line was
  dispatched, the node is stopped or unreachable, and no `lb.json` reached the primary checkout (the case of
  `research/coordination/b2b0924_s102_ruling.md`, on branch research/b2-rulings-2026-09-29).
- **A1.4 re-run procedure**, unchanged: (a) move the first attempt aside on the node and in the primary checkout,
  to `research/findings/raw/_load_bearing/_b2d0929_attempt1/b2d0929/s<seed>/<faculty>/`; (b) one re-run by direct ssh
  on a named node provisioned with H, never the host of the first attempt, with the job line unchanged from `cd` to
  `--out` and `bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 --` put in front of its `env`, logged in
  `research/coordination/b2d_reruns.tsv`; (c) a line that never ran is queued again, at most twice, after the stale
  line is moved out of `research/queue/pool.queue`. Amendment 2's queued `pool_node=` path is allowed for a re-run
  only when the first attempt was a dispatcher accident of the kind that amendment names.
- **A1.5 verdict**, unchanged, with 216 coverable cells (36 rows by 6 seeds): FAIL if any cell is a FAIL cell (class
  C on two different hosts); COMPLETE if all 216 are DEFINED; INCOMPLETE otherwise, which is final once every
  UNDEFINED cell has had its re-run and every never-ran line its re-queues.
- **A1.6 corpus**, unchanged: the sha256 of `data/corpus/tinystories.txt` in each node's `revisions/<H>` copy is
  recorded in `research/coordination/b2d_corpus_sha256.tsv`, committed before the first line is queued. The
  primary checkout's copy reads `7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec` at filing, the
  same value B2b recorded.
- **A1.7 memory**, unchanged: every line declares `mem_gb=8` through its `--checked` reason. See "Memory, cost and
  wall time".
- **A1.8 waves**, re-sized for the lane (see "Memory, cost and wall time"). The correction of 2026-09-25 stands: the
  shape check after each wave catches a malformed staged line, not the pop-time tearing fixed in `096dfdae0`.
- **Stranded results (new, from the s102 ruling).** Sync-before-stop (`559f705b7`, in H) makes the idle-stopper and
  the budget cap pull a node's results and re-check for a runner before stopping it; an unverified sync means no stop
  that cycle. If a node still stops with results unsynced (a manual stop, a failed sync followed by a stop from
  outside these scripts, an instance fault), B2d applies the ruling's Reading B in advance:
  - The cell is class E-stranded at once. Its one re-run is spent by A1.4(b) on another named H node. INCOMPLETE
    stays final and never waits on an unreachable node.
  - The stopped node is not started until every re-run of its stranded cells has landed. Otherwise
    `pool_sync.sh`'s `rsync -au` could bring attempt 1's `lb.json` to the canonical path first.
  - Before the node next starts, its copy of each stranded cell is moved aside on the node, as A1.4(a) does, and
    is then pulled into `_b2d0929_attempt1/`. A retrieved attempt 1 is reported and never scored. Because numpy
    on one thread is deterministic, it also serves as a cross-check of the re-run.
  - Starting a stopped AWS node only to retrieve attempt 1 costs money and makes the node a dispatch target again
    while it runs. That is the owner's call, not part of this procedure.
  - Before scoring, every b2d0929 line in `research/queue/dispatch.log` with no `lb.json` in the primary checkout
    is listed with its node and that node's state, so that E-stranded cells are named and never counted as never-ran
    (A1.4(c) re-queues are only for lines with no dispatch record).

## Three scores

All three read the same cells and the same arm files. Only S1 is the headline.

### S0, raw (as registered; reported)

Per cell, the `load_bearing` of `lb.json` as the harness at H wrote it. Per seed, n_load_bearing over n_exercised,
as `lb_shard.py aggregate` computes it. S0 is reported for continuity with B2a and B2b and for the drift report.

### S1, primary: self-describing fields excluded (the #1 metric at H)

**Self-describing (SD).** A decision field of a row is SD when a statement that runs only with the row's lesion flag
set (a branch on the flag, or on a variable taken directly from it) assigns the field's value, or returns early so
that the field is absent or null, and the value does not depend on any neural read made after the cut. Sub-kinds:
lesion-named literal, branch-forced constant, branch-structural, derived from a forced value, flag echo. A field is
**not** SD when the flag changes weights, inputs or a read upstream and the same code computes the field in both
arms from a neural read, even when the cut is designed to force a known outcome.

**The list.** `research/coordination/b2d_decision_fields.json` gives, per coverable row: the effective probe turn
and decision fields under the adequate probe set (as `measure_faculty` passes them to `compare()`), each field's
class (SD with sub-kind, neural, reply, noise-excluded) with file:line evidence at H, the kept fields, and the git
blob of each cited file at H. The audit behind it is summarized under "Self-describing fields at H".

**How S1 is computed (no harness change; the pin stays H).** The scorer replays `measure_faculty` at H over the arm
files already in each cell directory:
- it imports `research.runners.load_bearing_fraction` from a tree whose harness files have H's git blobs
  (`load_bearing_fraction.py`, `onebrain_regression_battery.py`, `research/runners/lbf_rows/*`, `tools/lab.py`),
  with the nine adequate LB_* flags and `LB_RESUME_SKIP_EXISTING=1` set before import, so that `_spawn_arm`
  (`load_bearing_fraction.py:1295`) loads each arm from disk;
- it replaces `_spawn_arm_raw` with a function that raises, so no brain is ever built and a missing arm file makes
  the cell UNDEFINED (class E, rule 6);
- it wraps the module-level `compare`, so that every `faculties` row passed to it loses the row's SD fields before
  the original `compare()` runs. The treatment comparison (line 1740), the null control through
  `_n_decision_diffs` (line 1513) and the repeat check (line 1757) all call that module-level name, so the
  exclusion applies to all three;
- the result is the cell's S1 verdict and `load_bearing`, with the same meaning as the harness's.

The same replay with an empty exclusion list must reproduce the cell's `lb.json` verdict and diffs exactly (the
**identity replay**). A cell that fails it is UNDEFINED for S1 and named: the scorer is then not reading what the
harness read.

**Rows the post-hoc route cannot score, and what happens to them.**
- open-ended-generation (neural-lesion, in the fraction). Under `LB_OPEN_ENDED_DISTRIB_PROBE` it returns early to
  `measure_open_ended_distributional` (line 1531). That path builds its own WTA bank, compares plausible fractions
  and has no decision fields to label SD. The scorer never replays it, because a replay would compute rather than
  load. Its S1 equals its S0, and the report says so.
- A row whose kept list is empty (every compared field SD) has no field left that can show a change the lesion
  branch did not write. Its S1 cell is UNDEFINED:sd-only, never 0 and never a pass, and is named. The per-seed S1
  fraction is then reported as a point over the scorable rows, together with the interval
  [n_lb / (n_ex + n_sd_only), (n_lb + n_sd_only) / (n_ex + n_sd_only)].
- The only way to score such a row in S1 is a registry change: add a decision field that the lesion branch does not
  write (for example a read-back of the state the organ changed), merged to main and followed by a re-pin. B2d does
  not make that change. It names each sd-only row as a registry defect for that row's lane.

### S2, secondary: reply-level (reported beside S1, never instead of it)

- **Probe turn.** For each coverable row except open-ended-generation, the single turn that `compare()` reads: the
  effective probe turn in `b2d_decision_fields.json` (for example `epi_recall` for episodic-memory, `dr2_c` for
  discourse-register). Earlier turns of the group are not scored.
- **Reply.** The tuple (`answer`, `abstained`, `recalled_svo`) of that turn's response dict in an arm file.
  `answer` is the reply text; the B2b arm files carry no other reply-text field. Canonical form:
  `json.dumps([answer, abstained, recalled_svo], sort_keys=True)`. The text is not normalized: no case folding, no
  whitespace or digit masking. Normalizing would be a choice made after seeing the text. The harness is
  deterministic, and the noise floor below catches any case where it is not.
- **Defined.** An S2 cell is DEFINED only if the cell meets A1.2 rules 1 to 4 and 6; the probe turn exists in all
  four arms (`intact_a`, `intact_b`, `lesion`, `lesion.rep0`) with no `_error` and a string `answer`;
  intact_a equals intact_b (the null floor); and lesion equals lesion.rep0 (the lesion floor). If intact_a and
  intact_b differ, the cell is UNDEFINED:null-noise; if lesion and rep0 differ, UNDEFINED:lesion-noise. An UNDEFINED
  S2 cell is never 0 and never a pass. S2 does not require rule 5, so a cell whose decision fields are noisy can
  still have a defined reply.
- **Reply load-bearing.** A DEFINED S2 cell whose intact_a reply differs from its lesion reply.
- **Reported.** Per seed, n_reply_load_bearing over n_reply_exercised (the DEFINED S2 cells), the mean over the six
  seeds and the SD (population SD, as `lb_shard.py aggregate` computes it). The reply-level robust core (rows reply
  load-bearing on all six seeds with six DEFINED cells) and union. Per faculty, S0, S1 and S2 counts side by side,
  as a point when all six cells are DEFINED and otherwise as the A1.5 interval.
- **Also reported, with no weight.** Which reply field changed: abstained or recalled_svo, or `answer` text only.
  An `answer`-only change is not a wording-only change: discourse-register's `bird was.` against `dog was.` is
  `answer`-only and is a change of content. Also: whether each `answer`-only change survives masking digit runs.
- **Scope.** The reply is the stub renderer's text. The worker defaults to `BRAIN_CHAT_RENDERER=stub` and
  `SIM_DISABLE_LLM=1` (`onebrain_regression_battery.py:884-885`), so S2 reads what the brain hands to the mouth,
  not Qwen's words.

### How the gap between S1 and S2 is read (stated before any B2d shard exists)

- **S1 yes, S2 no:** the lesion changes a decision field that the lesion branch did not write, but not the reply on
  this probe turn. The faculty changes internal state that this probe's reply does not use. Either it acts on later
  turns or other probes, or it observes without driving. This is the "faculty counts only if its lesion changes the
  reply" standard (owner, 2026-08-19) applied to one probe. It is not proof that the faculty never drives a reply.
- **S2 yes, S1 no or sd-only:** the reply changes, but no kept field records why. The registry is missing a field.
- **Both yes, with an `answer`-only change:** the reply changes, often through a host template clause that is keyed
  on the faculty's decision (in B2b: `(Reminder -- ...)`, `My decision-margin reads this as low-confidence...`,
  `(Holding -- nothing salient to add to that.)`). The lesion shows that the neural decision gates the clause. It
  does not show who composed the clause.
- **Input to the host-decision-share plan.** Each row of the last two kinds is an entry for the seam map
  (plan §1: the host operation, the neural state before it, the host decision). The intervention measurement
  (plan §2) is what decides ownership. S2 is a lesion measurement; it cannot measure ownership.

## Mouth-channel arm (new tag `b2d0929-mouth`; GPU, Qwen2.5-0.5B, added 2026-09-29 per the reply-level rescore)

The rescore (`research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md`) classes
affect-coloring, affect-appraisal-interoceptive and affect-drives-response as TRACE-ONLY-INSTRUMENT-LIMIT because no
battery to date has the Qwen mouth in the loop (`_collect_worker` hardcodes `renderer="stub"`). This arm closes that
gap for these three rows only.

- **Renderer.** `qwen` (`research/runners/brain_chat_tui.py`'s `QwenRenderer`, backed by
  `research/runners/_grounded_lang_integration_derisk.py`'s `SpikingQwenFaculty`, Qwen2.5-0.5B-Instruct). Greedy
  decoding is already the faculty's only mode (`model.generate(..., do_sample=False, ...)`, no sampling flag to set).
- **Backend.** `SIM_BACKEND=cupy` (GPU only -- the Qwen renderer warns and runs slow on CPU; this arm never runs on
  numpy). Queued one job at a time through `tools/gpu_queue.sh add '<cmd>'` (CLAUDE.md non-negotiable #9: one
  brain-loading GPU process at a time); never dispatched outside the queue.
- **Rows and turns.** The SAME probe turns already registered for these three rows in the b2d0929 arm (`emo` for
  affect-coloring and affect-appraisal-interoceptive, `well` for affect-drives-response, per
  `b2d_decision_fields.json`'s `probe_turn`), so this arm's S2 is directly comparable to the stub arm's S2 on the
  identical turn -- the only variable that changes is the renderer.
- **Arms.** intact_a, intact_b (the null -- two independent same-seed rebuilds through the Qwen renderer; kept even
  though decoding is greedy, because the null must also catch any upstream non-determinism a generative call
  introduces that a template renderer cannot), lesion, lesion.rep0. Same four-arm shape as S2 above.
- **NEEDS-CLAUDE, not written by this pre-registration.** `_collect_worker`
  (`research/runners/onebrain_regression_battery.py`, ~line 902) passes `renderer="stub"` as a literal to
  `BrainChatRequest`, so `BRAIN_CHAT_RENDERER=qwen` alone never reaches it (the env var it reads at ~line 884 sets
  only a default that this call site ignores). Needed: either thread a `renderer` parameter through
  `_collect_worker`/`_spawn_arm`, defaulting to `"stub"` so every existing caller (b2b, b2d0929, every prior
  battery) stays byte-identical, or a bespoke ~30-line script that calls `brain_chat` directly with `renderer="qwen"`
  for just these 3 rows' turns. Acceptance check either way: with `renderer` left at its default, the existing
  258-cell b2d0929 arm reproduces byte-identical to a run before the change.
- **Scoring.** S2 exactly as defined above, except `answer` is real Qwen prose rather than a stub template, so the
  comparison genuinely exercises the manner-instruction channel (`affect.manner_template`, `affect.tone_level`).
  Reported separately from the b2d0929 headline -- a different renderer is a different instrument, never averaged
  into S0/S1/S2 of the main arm.
- **Cost.** 3 rows x 6 seeds x 4 arms = 72 short Qwen generations (`max_new_tokens` per `QwenRenderer`'s default) --
  minutes on the local 3090, no AWS GPU needed.
- **Prior signal, stated so a GO is not assumed.** A different real mouth (linattn WKV) already read a directional
  NO-GO, positive-asymmetric, on the same affect-tone channel
  (`research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO.md`). This arm
  is the measurement the instrument-limit classification calls for, not a bet on the outcome.

## Probe remaps for instrument-limit rows (dev-seed verification; never queued at the blind seeds until the dev-seed check passes)

Per CLAUDE.md's dev/blind split (42/43/44 dev, 100/101/102 blind) and the reply-level rescore's per-row "fix"
column. Each remap below is checked on ONE dev seed (42) before it is added to any 6-seed job file; none of these
rows is queued at 100/101/102 under a new probe until its own dev-seed check passes and is recorded here or in a
follow-on amendment.

**Rows with an existing, already-run reply-level measurement (integrate; do not re-derive).**
- **confidence-forthcomingness.** The 2026-09-01 LTM-elaboration protocol already reads a genuine 6-seed reply-level
  GO (`research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json`,
  built by `verify_confidence_ltm_loadbearing.py`). Dev-seed step: read seed 42's cell out of that existing output
  (no re-run) and confirm its `answer`/`n_sentences` diff matches what the rescore finding cites, before wiring the
  protocol as this row's B2d S2 probe.
- **source-provenance-honesty.** The chain-route (derived-answer) turn already reads a genuine 6-seed GO
  (`research/findings/2026-09-01-source-provenance-honesty-129-default-on-flip-GO.md`,
  `research/findings/raw/_source_provenance_honesty_flip/flip_verify.json`). Dev-seed step: confirm seed 42's
  `flip_verify.json` cell still reproduces at H (the flags it depends on, `bedb9ad6e`/`e73bea486`, are both
  ancestors of H) before wiring the chain-route turn as this row's B2d S2 probe ALONGSIDE, never instead of, the
  registered `well` turn.
- **swap-drives-response.** `LB_SWAP_DRIVE_PROBE` already reads reply-load-bearing 6/6 at one pinned revision
  (`7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`), with only the seed-42 leg committed to git. Dev-seed step: commit
  the seed-42 artifact, then confirm it reproduces at a fresh H-pinned build (same probe, same seed) before adding
  `LB_SWAP_DRIVE_PROBE=1` to the b2d0929 flag set; the other five seeds' uncommitted working-tree artifacts are not
  trusted without a fresh H-pinned re-run.

**Rows needing a genuinely new probe (the adjudicator's remap list, none run yet).**
- **da-mode-drives-response.** A turn whose intact DA reaches focus (>= `_DA_FOCUS_MAX`,
  `webapp/da_mode_drives_chat.py`) on all 6 seeds, unlike `emo` (4/6 only). Dev-seed step: on seed 42, confirm the
  candidate turn's intact DA reads focus and the intact `answer` carries the focus suffix, before checking whether
  the lesion removes it.
- **multiref-competition.** `hold` immediately followed by a hold-query turn ("who are we talking about"), so
  `answer = jq['readout']` is actually exercised. Dev-seed step: on seed 42, confirm the hold-query turn's intact
  `answer` differs from a no-hold control before comparing lesion vs intact.
- **comprehension-learned-animacy-cue / comprehension-learned-verb-selects.** An ill-formed or role-reversed
  transitive whose only covered noun/verb is the LEARNED (not hand-table) one, so the intact brain abstains or
  repairs and the lesion passes it through unchanged -- the direction one skeptic check already ran live once
  (not at 6 seeds, not committed as a probe). Dev-seed step: on seed 42, confirm the intact arm reads
  `abstained=True` (or a repair `answer`) and the lesion arm does not, before adding either row to a 6-seed job file.
- **open-ended-generation.** The dedicated production-turn reply instrument (not the distributional ruler), which
  already ran and reads NO-GO on its own terms (mean delta 0.094 under the 0.10 floor <!--derived-->, quoted from
  `research/findings/2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md`, reproduced by
  `research/findings/2026-09-25-open-ended-gated-turn-part-a-6seed-gate-NOT-GO-UNDEFINED.md`). No dev-seed step is
  needed to wire it in (it already exists and already ran at 6 seeds); B2d reports its existing NO-GO verdict as
  this row's S2 rather than leaving the row UNDEFINED, and does not re-run it.

**wm-binding-advanced is deliberately absent from this list.** Two remediation attempts already exist
(`LB_WMB_HOLDQUERY_PROBE`, found pass-by-construction and deliberately excluded from load-bearing scoring by its own
code; a third, unnamed probe with the same outcome) and neither closes the row. It stays UNDEFINED in B2d pending a
genuinely new probe, not a re-run of either existing one.

## Self-describing fields at H

`research/coordination/b2d_decision_fields.json` covers all 35 brain_chat coverable rows. Each compared field is
marked SD (with sub-kind), N (neural consequence), R (reply field), NX (noise-excluded, never compared) or UNSURE,
with file:line evidence at H and the git blob of every cited file. Three read-only audits applied the definition
above, and the SD calls were re-read at filing. UNSURE marks a field that is a literal or a host parse identical in
both arms, or that is absent in both arms. UNSURE fields are kept in S1; they cannot decide a verdict on this probe
set, and an S1 verdict that rests only on one would be reported as such.
- **SD fields (8 rows affected):**
  - reconsolidation: `.action` (`lesioned_nowrite`, `research/runners/reconsolidation_production_organ.py:167-169`).
  - discourse-register: `.agent` and `.abstained` (the lesion subclass forces the prev slots to the identity index,
    `research/runners/d3_discourse_event_register_production_organ.py:89-93`).
  - affect-drives-response: `affect_drives.reason` (`webapp/affect_drives_chat.py:487-488`).
  - da-mode-drives-response: `da_drives.reason` (`webapp/da_mode_drives_chat.py:459-460`).
  - affective-tom: `affective_tom.reason` (`research/runners/affective_tom_production_organ.py:319-320`).
  - noncontradiction-gate: `.recalled_yn` (`if self._lesion: return "unknown"`,
    `research/runners/b3_noncontradiction_production_organ.py:178-179`), and `.reject` and `.stored_polarity`,
    which are derived from it.
  - confidence-forthcomingness: `.granted` and `.reason` (whether-disable skips the block,
    `webapp/server.py:6427,6619-6620`).
  - comprehension-monitor: `comprehension.comprehended` (see the organ error below).
- **sd-only rows (S1 UNDEFINED:sd-only):** reconsolidation, discourse-register, affective-tom and
  confidence-forthcomingness. The first three change the reply on 6 of 6 B2b seeds, so S2 carries them;
  confidence-forthcoming changes it on none.
- **Borderline N call, stated:** `pragmatic.enriched_interpretation` reads `undetermined (implicature collapsed)`
  under the lesion. The lesion rebuilds the RSA posterior with one weight at 0, and the same margin code emits the
  string (`research/runners/pragmatic_production_organ.py:183-213`), so the call is N. A reviewer may call it SD.
  S0 is reported beside S1, so that call is visible.
- **Organ error found by the audit (instrument failure in B2b).** In every checked lesion arm of
  comprehension-monitor (s42, s101), `comprehension` is `{"on": true, "error": "ValueError: could not broadcast input
  array from shape (7002,) into shape (1032,)"}`. The lesion twin raises, and the exception fallback in
  `webapp/server.py:6108-6109` drops `comprehended`. The harness read that as a change, so B2b's 6 of 6
  load-bearing for this row is a crash, not a lesion effect. B2d rule (added to A1.2 as rule 7): a cell whose
  probe-turn response carries an `error` key in the parent dict of any compared field, in any arm, is
  UNDEFINED:organ-error, class C. This applies to S0 as it is reported, and to S1 and S2.

## Scoring code: a start precondition

No b2d0929 line is queued until a checker is committed and has passed the acceptance checks below. The B2b checker
(`tools/b2b_score.py`, being written on branch research/b2-rulings-2026-09-29) covers A1.2, A1.5 and A1.9 for B2b.
B2d's checker, **`tools/b2d_score.py`**, may import it or generalize it. The interface is fixed here; the code is
not written by this pre-registration.

```
.venv/bin/python tools/b2d_score.py \
    --tag b2d0929 --pin <H> --seeds 42 43 44 100 101 102 \
    --base research/findings/raw/_load_bearing/_shards \
    --fields research/coordination/b2d_decision_fields.json \
    --code-root <a tree at H: a pool revisions/<H> copy or a git archive of H> \
    --corpus research/coordination/b2d_corpus_sha256.tsv \
    --reruns research/coordination/b2d_reruns.tsv \
    --dispatch-log research/queue/dispatch.log \
    --compare-tag b2b0924-base --compare-pin a308f1e09babcc9ed096c3c8046d00040391368c \
    --out "$B2D_DIR"/b2d_score.json
```

- **Per cell:** A1.2 rules 1 to 6 with the failing items; cause class (E, E-stranded, C, I) and final state
  (A1.5); host; the S0 verdict; the identity-replay result; the S1 verdict and kept-field diffs; the S2 state with the
  four reply tuples' hashes and the changed reply fields.
- **Per seed and per faculty:** S0, S1 and S2 as defined above; robust core and union for each; the sd-only rows
  and the S1 interval; the battery verdict.
- **Drift block:** the same per-cell record for the compare tag at its own pin, and the cell-by-cell comparison.
- **Refuses** unless the `--code-root` harness blobs equal H's, the fields file's `pin` equals `--pin`, and every
  blob the fields file cites equals the blob in `--code-root`.
- **Never builds a brain.** It never writes under `--base` except `--out`.
- **Acceptance, before queueing, committed with the scorer:**
  1. identity replay over B2b's cells reproduces every present `lb.json` verdict and diffs (at filing, the
     prototype below reproduced 209 of 209);
  2. synthetic cells: a diff only in an SD field gives S1 pass; intact_a differing from intact_b in `answer` gives
     S2 UNDEFINED:null-noise, never 0; a missing `.rep0` gives UNDEFINED, class E, rule 6; an all-SD row gives
     UNDEFINED:sd-only; an open-ended-generation cell is never replayed;
  3. a test that fails if the scorer calls `_spawn_arm_raw`.

## Feasibility measured at filing (B2b data, read-only; no B2d result)

The B2b shard tree was copied from the primary checkout to a scratch directory and read there.
- **Identity replay.** H's `measure_faculty`, run through the route above with an empty exclusion list, reproduced
  the recorded verdict and diffs of 209 of 209 B2b brain_chat cells that have an `lb.json`. The one other cell,
  s102/episodic-memory, has no `.rep0` yet; the replay raised on it as designed.
- **Raw B2b** (provisional; B2b's own scoring is pending): mean fraction 0.842, SD 0.013, robust core 27 (26 <!--derived-->
  brain_chat rows and open-ended-generation), union 29; swap-drives-response and wm-binding-advanced read
  not-exercised on every seed.
- **Draft S2 on B2b:** per seed 17/35, 17/35, 18/35, 18/35, 18/35, 17/34; mean 0.502, SD 0.013; reply robust core <!--derived-->
  16, union 18 (affect-marker-spiking-wta on 4 seeds, episodic-memory on 5 defined seeds). There were no noise
  UNDEFINED cells: the null and lesion floors were exact on every cell. Of the 105 reply-changed cells, 29 change
  abstained or recalled_svo, and 76 change `answer` only.
- **Draft S1 on B2b, computed 2026-09-29 by the now-committed `tools/b2d_score.py`** (direct arm-file diff over
  `b2d_decision_fields.json`'s `kept` fields, NOT a harness-internal replay -- see that file's own docstring for the
  documented simplification; re-derivable with the "Step 5" command below, `--tag b2b0924-base --pin
  a308f1e09babcc9ed096c3c8046d00040391368c`; this run's own output was not committed as an artifact, per
  `gates/prereg_before_run`, so re-derive rather than trust this line):
  <!--derived--> per seed 20/34, 20/34, 20/34, 21/34, 20/34, 20/34; mean 0.724, SD 0.020; S1 robust core 20, union 22.
  Rows that flip from S0's "load-bearing" to S1's "pass" once their only diff was self-describing
  (e.g. affect-drives-response, S0 6/6 -> S1 0/6) and rows that flip from S0's blanket "6/6" to a genuine partial
  count once the tautological seeds are excluded (da-mode-drives-response, S0 6/6 -> S1 2/6, matching this session's
  independent finding exactly) both reproduce the reply-level rescore's per-faculty analysis
  (`research/findings/2026-09-29-load-bearing-fraction-reply-level-rescore.md`). sd-only rows (reconsolidation,
  discourse-register, affective-tom, confidence-forthcomingness) correctly read UNDEFINED:sd-only on every seed, not
  0 and not counted in the denominator.

These are B2b numbers at F, computed by draft code that is not yet run as B2d's own registered arm. They are
recorded so that the definitions are shown to be computable on real arm files, and because S1's own acceptance
checks (identity-adjacent sanity: S0 reproduces `lb.json` verbatim; sd-only rows never read 0; a field absent in
every arm never blocks a `pass`) are demonstrated on real data here, not only on synthetic fixtures in
`tests/test_b2d_score.py`. They carry no weight for B2d's own COMPLETE verdict, which is scored only from b2d0929's
own arm files at H.

## Memory, cost and wall time

- **Memory.** An LB worker for episodic-memory at F peaked at 6.9 GB RSS on a 15 GB pool node (measured
  2026-09-29). `mem_gb=8` on every line therefore means one B2d job at a time on pool41 or pool42, 12-core 15 GB
  nodes. The other rows have no measured peak; they are declared at 8 as well, which costs packing, not validity.
- **Work.** B2b took about 16.4 to 17.6 core-hours per seed of 43 rows, so B2d is about 98 to 106 job-hours. 248 of
  B2b's 257 landed cells ran on AWS r7i.4xlarge nodes, so these figures are AWS-speed. On the 8 rows that also ran on
  pool41/pool42, the median pool job took about 1.4 times as long (a small sample). The longest job is
  episodic-memory, about 2.2 h median and 3.0 h at most on AWS.
- **Pool only (pool41 + pool42, one job each):** about 98-106 × 1.4 / 2, roughly 70 to 75 h of wall time, or about
  three days if nothing else uses the two nodes; no cloud cost. Waves: one seed (43 lines) at a time, with the next
  wave queued when fewer than 10 fresh b2d0929 lines remain. At about 3 lines/hour that stays under the 12 h
  `POOL_JOB_MAX_AGE`, provided other pool work does not take the nodes.
- **AWS (r7i.4xlarge: 16 vCPU, 128 GiB, about $1.0071/h, owner cap about $50/day, auto-stop on):** memory allows 16 <!--derived-->
  jobs of 8 GB per node, and CPU allows about 15 single-threaded jobs. With ideal packing, one node needs about
  7 h plus a tail of up to 3 h, about $8 to $11; two nodes need about 4 to 6 h of wall time for the same total cost.
  B2b's observed run was slower than ideal: two nodes, with lb.json files landing from 00:05 to about 14:30 on
  2026-09-25 after the first wave was queued at 19:54 the evening before. Planning figure: at most about 20 h of wall
  time and at most about $40, under the daily cap. Waves: two seeds (86 lines), with the next wave queued when fewer
  than 20 fresh lines remain, as B2b did.
- **Recommended:** AWS plus pool, one arm, with the pool nodes also serving as the named hosts for re-runs. A
  pool-only run is valid, only slower.

## Comparison with B2b (reported, not gated)

- **Drift F to H, cell by cell:** for each of the 216 coverable (row, seed) cells, S0, S1 and S2 at F (from B2b's
  arm files, scored by the same scorer and fields file) against H, with both hosts. A cell enters only if it is
  DEFINED in both batteries, under B2b's A1.2 with F and B2d's with H. A cell that is not final in B2b when B2d is
  scored is reported as pending, never estimated.
- **Rows added or changed F to H:** none (see "Frozen revision"). The denominators are the same 36 rows.
- **Expected difference, named in advance:** affect-marker-spiking-wta. At F, its S2 reads changed on 4 seeds,
  where the marker word was prepended to `answer`. At H, `969894855` keeps the marker off the surface by default, so
  its S2 is expected to read no change. Its S0 is not expected to change, because it compares the recorded
  `affect_drives.lead`.
- **Any other difference** is attributed to code merged between F and H only when both cells are DEFINED. It is
  reported with the commit range and without bisection. Such a difference is itself a finding: a change on the
  default path that its commit did not announce.

## What a COMPLETE verdict licenses

Reporting the #1 metric at H: S1, the lesion-verified load-bearing fraction with the self-describing fields
removed, over the full registry at the production default, per seed with mean and SD, robust core, union and each
faculty's count. S0 and S2 are reported beside it. Every figure is labeled "at H" and "stub renderer".

Nothing more. It licenses no default flip. It makes no claim about replies rendered by Qwen, about who composed a
reply (that is the plan's intervention measurement), or about any probe outside the registered set. A FAIL names
each row the shipped revision cannot measure as registered. An sd-only row is named as a registry defect for its
lane, not as a failure of the faculty.

## Known weak points

1. **The SD audit is a judgment on code, made after B2b's data was seen.** The definition is written down and every
   call cites a line. But which fields to drop was decided by people who knew which rows would lose their load-bearing
   status, and a borderline call (a branch-forced constant against a cut designed to force the same value) moves the
   headline. Mitigation: the per-field list and evidence are committed before any B2d shard exists; S0 is reported
   beside S1, so every call is visible and reversible; a reviewer who disagrees can re-score from the arm files
   without a re-run. Not mitigated: nobody independent has done the audit a second time.
2. **The identity replay was run on F's arm files through H's code.** The registry, the turn groups and the scoring
   path are equal between F and H, so this is expected to hold on B2d's own files, but it is shown only on B2b's.
   The scorer runs the identity replay on every B2d cell and makes a mismatched cell UNDEFINED; it does not assume
   the result.
3. **S2 reads one turn of stub-renderer text.** An exact comparison catches a changed number in a template (for
   example the listener-model probability in the pragmatic clause) as a reply change. It also misses effects that
   show only on later turns or only through Qwen. A row with S1 yes and S2 no may still drive replies elsewhere. The
   gap reading above is limited to this probe for that reason. What S2 measures is whether the brain's lesion reaches
   the text on the probe turn, not whether the reply a user sees differs.

## Residuals

- open-ended-generation is scored by its distributional ruler in S0 and S1 and is outside S2.
- Rows of kind thin, mechanism-only, in-process and proposed stay outside all three scores, as in B2b. learned-referent
  reads `not-covered:thin` by construction.
- The per-row memory declaration is uniform (8 GB) because only episodic-memory has a measured peak.
- The comparison with B2b depends on B2b's final cells. B2b's two seed-102 re-runs were in flight at filing.

## Prepared commands (exact, in order; none is run by this pre-registration)

Run from the primary checkout (`/home/dant123/Projects/sim`) unless a step names the worktree. Each step has a stop
condition; a stop means no later step runs until it is resolved.

**Step 0, done at filing (worktree).** `b2d_jobs.txt` and `PIN.txt` were generated as described under "Arm (fixed)",
and are staged together with this file and `b2d_decision_fields.json`.

**Step 1, NEEDS-CLAUDE.** Have someone other than the three audits re-check the SD calls in
`research/coordination/b2d_decision_fields.json`, pragmatic first. Also log the comprehension-monitor organ error
in `research/FAILURE_LOG.md`, with a gate. Stop: a disputed call is not settled by an amendment before queueing.

**Step 2, commit and push the pre-registration (worktree), once the gates allow it.**
```
cd /home/dant123/Projects/sim/.claude/worktrees/b2d-prereg && git add research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md research/coordination/b2d_jobs.txt research/coordination/b2d_decision_fields.json research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt && git commit -F /tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/b2d_commit_msg.txt && bash tools/push_both.sh research/b2d-prereg-2026-09-29
```
Stop: the pre-commit hook refuses (never `--no-verify`), or `push_both.sh` does not verify both remotes.

**Step 3, provision H on the pool nodes, only while origin/main is an ancestor of H.**
```
cd /home/dant123/Projects/sim && git fetch -q origin && git merge-base --is-ancestor origin/main 3c2907b85c10fe918ddc81b4c69de8888c9f321b && bash tools/pool_provision.sh --revision 3c2907b85c10fe918ddc81b4c69de8888c9f321b --isolated pool41 pool42
```
For the AWS lane, add `pool1 pool2` (or the current AWS aliases) to the same command after the instances are up.
Stop: `merge-base` fails, in which case the re-pin rule applies by amendment and every later step uses H'; or the
provisioner refuses or exits non-zero. Never set `POOL_PROVISION_ALLOW_STALE`.

**Step 4, record the corpus hash for each provisioned node.**
```
bash -c 'set -euo pipefail; H=3c2907b85c10fe918ddc81b4c69de8888c9f321b; T=research/coordination/b2d_corpus_sha256.tsv; [ -f "$T" ] || printf "node\thost\tfile\tsha256\tsource_git_sha\tprovisioned_ok\trecorded_utc\n" > "$T"; for n in pool41 pool42; do out=$(ssh -n -o BatchMode=yes -F research/queue/.pool_ssh_config "$n" "cd ~/derisk-pool/revisions/$H && hostname && sha256sum data/corpus/tinystories.txt"); host=$(echo "$out" | sed -n 1p); sha=$(echo "$out" | sed -n 2p | cut -d" " -f1); printf "%s\t%s\tdata/corpus/tinystories.txt\t%s\t%s\tyes\t%s\n" "$n" "$host" "$sha" "$H" "$(date -u +%FT%TZ)" >> "$T"; done; cat "$T"'
```
Add the AWS aliases to the `for` list if used. Commit the file with a pathspec commit. Stop: any sha256 other than
`7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec`; that node takes no b2d0929 cell until it is
re-provisioned and hashed again.

`B2D_DIR` below is `research/findings/raw/_load_bearing/_shards/b2d0929`; `SCRATCH` is any directory outside the repo.

**Step 5, DONE 2026-09-29 (`tools/b2d_score.py`, `tests/test_b2d_score.py`), with a stated scope reduction from the
interface above.** The committed scorer computes S0 (verbatim, trivially exact), S1 (a direct arm-file diff over
`b2d_decision_fields.json`'s `kept` fields, per row and probe turn) and S2 (reply-level, exactly as defined), plus
A1.2 rule 7, all read-only from the shard tree -- it never imports `research.runners.load_bearing_fraction` and
never builds a brain (pinned by a unit test asserting the module is not imported). **What this does NOT do**: the
"How S1 is computed" section above describes monkeypatching the harness's own `compare()`/`_spawn_arm_raw` so an
EMPTY exclusion list reproduces `lb.json`'s verdicts and diffs byte-for-byte (the literal "identity replay").
`tools/b2d_score.py` does not attempt that: its S1 is an independent recomputation from the arm files, not a replay
of the harness's internal comparison function, so there is no harness-level "identity" to check S1 against -- only
S0 (a verbatim read) is exact by construction. A harness-level replay implementation remains NEEDS-CLAUDE (see
HANDOFF.md); it would let a future run cross-check this file's S1 against the harness's own SD-stripped `compare()`
rather than trusting this file's independent field-level treatment/null/repro logic alone. Run against B2b's cells
(read-only, no B2d result -- this is what "Feasibility measured at filing" above reports):
```
.venv/bin/python tools/b2d_score.py --tag b2b0924-base --pin a308f1e09babcc9ed096c3c8046d00040391368c --seeds 42 43 44 100 101 102 --base research/findings/raw/_load_bearing/_shards --fields research/coordination/b2d_decision_fields.json --code-root ~/derisk-pool/revisions/3c2907b85c10fe918ddc81b4c69de8888c9f321b --out "$SCRATCH"/b2d_score_on_b2b.json
```
Here `--code-root` is a local `git archive` of H if the pool copy is not local; without it, the code-root blob
precondition fails openly (reported, not silently skipped) rather than being treated as passed. Stop: any B2b cell
with an `lb.json` reads a different S0 `verdict`/`load_bearing` than the file itself records (this IS still checked
-- S0 is a verbatim read of `lb.json`, so a mismatch here means a bug in cell enumeration or file matching, not in
comparison logic).

**Step 6, NEEDS-CLAUDE: the wave script.** Write `research/coordination/b2d_queue_next_wave.sh` from
`b2b_queue_next_wave.sh`, with F replaced by H, `TAG=b2d0929`, `JOBS=b2d_jobs.txt`, `LEDGER=b2d_waves.tsv`,
`CORPUS_RECORD=b2d_corpus_sha256.tsv`, a lock file of its own, `WAVE_SIZE` 43 on the pool-only lane or 86 with AWS,
`THRESHOLD` 10 or 20, and
`REASON='prereg research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md (H 3c2907b85); B2d; mem_gb=8'`.
Then run:
```
bash research/coordination/b2d_queue_next_wave.sh --dry-run
```
Stop: the dry run lists any line whose text differs from `b2d_jobs.txt`, or it refuses for want of the corpus record.

**Step 6, fallback without the wave script (one wave, the first 43 lines = seed 42).** Each line goes through the
queue's own gates:
```
bash -c 'R="prereg research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md (H 3c2907b85); B2d; mem_gb=8"; sed -n 1,43p research/coordination/b2d_jobs.txt | while IFS= read -r line; do bash tools/pool_queue.sh add "$line" --checked "$R" </dev/null || { echo "ADD FAILED: $line" >&2; exit 1; }; done'
```
For later waves use `sed -n 44,86p`, `87,129p`, `130,172p`, `173,215p` and `216,258p`, each queued only when fewer
than 10 fresh b2d0929 lines remain in `research/queue/pool.queue`. Log each add's epoch, wave and line number in
`research/coordination/b2d_waves.tsv`. Stop: any add fails, or a queued b2d0929 line does not start with
`cd ~/derisk-pool/revisions/3c2907b85c10fe918ddc81b4c69de8888c9f321b && ` and end with `/lb.json  #checked:`.

**Step 7, while running.** For each UNDEFINED cell, apply A1.4 with the logged command form
`cd ~/derisk-pool/revisions/<H> && .venv/bin/python tools/assert_flipped_defaults.py && mkdir -p <dir> && bash tools/mem_ok.sh 8 2 && bash tools/memcap.sh 12 -- env <the line's env and command unchanged>`,
by direct ssh on a named H node other than the first host. Record it in `research/coordination/b2d_reruns.tsv`.
Stop: a cell reaches FAIL (class C on two hosts). That cell is recorded, and the battery continues, since FAIL never
waits on other cells.

**Step 8, aggregate (S0), after every cell is final.**
```
.venv/bin/python tools/lb_shard.py aggregate --tag b2d0929 --seeds 42 43 44 100 101 102
```
It reads `PIN.txt`. Stop: `provenance.status` is not `verified`.

**Step 9, score (S0, S1, S2, drift).**
```
.venv/bin/python tools/b2d_score.py --tag b2d0929 --pin 3c2907b85c10fe918ddc81b4c69de8888c9f321b --seeds 42 43 44 100 101 102 --base research/findings/raw/_load_bearing/_shards --fields research/coordination/b2d_decision_fields.json --code-root ~/derisk-pool/revisions/3c2907b85c10fe918ddc81b4c69de8888c9f321b --corpus research/coordination/b2d_corpus_sha256.tsv --reruns research/coordination/b2d_reruns.tsv --dispatch-log research/queue/dispatch.log --compare-tag b2b0924-base --compare-pin a308f1e09babcc9ed096c3c8046d00040391368c --out "$B2D_DIR"/b2d_score.json
```
Stop: the scorer refuses (a blob mismatch, a PENDING field, a pin mismatch). The verdict is written as a findings doc
with the labels "at H" and "stub renderer".
