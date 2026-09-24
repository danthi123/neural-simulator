"""LOAD-BEARING FRACTION — the instrument for the #1 project metric (owner-ratified 2026-09-19).

THE METRIC (docs/plans/2026-09-19-roadmap-with-a-permanent-llm-mouth.md §4; memory
project_2026_09_19_permanent_llm_mouth_loadbearing_metric): the FRACTION of production faculties where LESIONING the
brain's contribution provably CHANGES the reply. Operationalises the standing bar (feedback_faculties_must_drive_not_
observe): a faculty is REAL only when varying its state changes the reply AND lesioning it makes that difference vanish.
The permanent LLM mouth is acceptable BY DEFINITION only while this fraction is high + lesion-verified — the LLM words,
the brain decides. When it falls, we have "an LLM wearing a brain", the thing the project rejects.

REUSE, NOT REINVENT. This is the SHIPPED-FACULTY REGRESSION BATTERY (research/runners/onebrain_regression_battery.py)
run with the SECOND ARM = "lesion this faculty's brain contribution" instead of "flip an unrelated flag", and the
verdict INVERTED. It imports and reuses, verbatim:
  * `_spawn_arm(env, turn_labels, out_path)` — build ONE fresh brain (seed 42, numpy/CPU, stub renderer, no LLM) in a
    subprocess with a given env and run probe turns through the REAL webapp.server.brain_chat. (Each lesion is an env
    flag the fresh build reads, so no in-process monkeypatching is needed — the battery worker is reused as-is.)
  * `compare(intact, lesioned, faculties=[row])` — the DECISION-ONLY (categorical) equality check, with the continuous
    NOISE fields (rates/levels/mood/margins/...) already excluded. The battery calls a diff a "regression"; here a diff
    is a CHANGE = the faculty is LOAD-BEARING.
  * `PROBE_TURNS` / `_TURN_BY_LABEL` / `FACULTY_PROBES` — the per-faculty (probe turn, decision fields) registry, and
    the shared-session dependency ordering (hold->held, dr_a->dr_b->dr_c, bc_a->bc_b).

THE VERDICT INVERSION (the whole point). Regression battery: PASS = the two arms decide IDENTICALLY (no unrelated flip
broke the faculty). Load-bearing: LOAD_BEARING = the two arms decide DIFFERENTLY (lesioning the brain's contribution
changed the decision). So per faculty, reusing the SAME compare():
    compare(intact, lesioned) verdict == "regressed" (a decision field differs)  -> LOAD-BEARING
    compare(...)             verdict == "pass"        (present + identical)       -> NOT load-bearing (cosmetic)
    compare(...)             verdict == "not-exercised"(fields absent both arms)  -> NOT-EXERCISED (thin probe)

THE LESION KNOB, PER FACULTY (FACULTY_LESIONS below). The metric wants to cut the BRAIN's contribution while leaving the
organ INSTALLED — not to disable the whole wiring block. Most production organs ship a dedicated `BRAIN_<X>_LESION=1`
knob that does exactly that (e.g. affect_drives_chat.py: BRAIN_AFFECT_DRIVES_LESION cuts the interoceptive->ladder
synapses so the felt mood collapses to ~0 and the affective lead VANISHES, the organ otherwise untouched). Those are the
gold-standard lesions used here. Where a faculty has no dedicated neural-cut knob we record HOW it is lesioned honestly:
  neural-lesion   a dedicated BRAIN_<X>_LESION flag cuts the neural read (the organ stays installed).  [PRIMARY]
  whether-disable no neural-cut knob; the BRAIN_<X>=0 master switch removes the organ (weaker: a byte-identical escape
                  may leave the ANSWER unchanged while only the decision-field presence swings). Noted per row.
  in-process      lesioned only by an in-process monkeypatch (parser.role_of / composer.query / _substrate_recall) —
                  ALREADY covered behaviorally by research/runners/_production_lesion_probe.py (CHOOSE/GENERATE/LEARN);
                  this env-flag instrument does not reach it. NOT a gap in coverage, a gap in THIS harness's mechanism.
  thin            a dedicated lesion flag EXISTS but the brain_chat-only probe set cannot construct the trigger (the
                  gnw-deliberation / value-driven-choice >=2-distinct-patient conflict — see the battery's own note).
  mechanism-only  the flip changes the SUBSTRATE not the ANSWER by design (one-brain-substrate: onebrain==rf answers).
  proposed        no clean lesion knob exists yet; the minimal one to add is named in the note. (NOT faked.)

RELIABILITY — "changed, and not just noise" (the metric's own qualifier). The webapp default builds the tiny-demo brain
and every organ at seed 42 (hardcoded; no per-request seed), so a given env is DETERMINISTIC: re-running an arm yields
byte-identical decisions. The anti-noise guarantee is therefore made EXECUTABLE, not assumed:
  (1) DETERMINISM SELF-CHECK: the intact arm for a turn-group is built TWICE and its decision fields must match. If they
      do not, the harness is non-deterministic and EVERY load-bearing verdict is untrustworthy -> reported loudly, the
      metric is marked UNRELIABLE. (This is the real "not just noise" proof: with the harness proven deterministic, any
      intact-vs-lesion decision diff is attributable to the lesion, not to run-to-run variation.)
  (2) LESION-REPEAT: each lesion arm is (optionally, --repeats>1) rebuilt and its decision fields must reproduce; a
      diff that does not reproduce is flagged `noisy` and EXCLUDED from the load-bearing count.
  (3) CHANGE KIND: a diff is classified `structural` (a field present intact / absent-or-null lesioned, i.e. the
      organ's output is genuinely gated off) vs `value` (both present, categorical value flips). Structural changes are
      inherently robust to any RNG-trajectory shift; value-only changes are still counted but flagged for the record.
  (4) FLAG-RESOLVES GUARD: a declared lesion flag that does NOT appear in source is reported as `lesion-knob-missing`,
      NEVER silently as "not load-bearing" — a no-op flag that reads as "no change" is the exact false-negative this
      metric must not make (the "a check that cannot fail" lesson, applied to a lesion that cannot bite).

HONEST BOUNDARY. Same as the battery's: this is a REACHABILITY + DECISION-DIFFERENCE instrument over the tiny-demo
brain_chat path, not a proof of each faculty's correctness. LOAD_BEARING_FRACTION is reported over the DENOMINATOR of
faculties this harness can actually lesion+exercise (neural-lesion / whether-disable with a driving probe); the
in-process / thin / mechanism-only / proposed faculties are reported SEPARATELY with their reason, never silently folded
into either numerator or denominator.

CPU / tiny-smoke. numpy backend, stub renderer, no LLM (inherited from the battery worker). Every arm is a full-brain
build -> RUN UNDER tools/memcap.sh (the 2026-09-18 OOM guard). Peak RSS is ONE build at a time (arms are sequential);
each arm runs only its faculty's minimal turn-group (1-3 turns), far lighter than the battery's 26-turn single process.

Run (smoke, ONE faculty, capped):
  tools/memcap.sh 20 -- .venv/bin/python -m research.runners.load_bearing_fraction --smoke curiosity-followup \
      --out research/findings/raw/_load_bearing/smoke.json
Verify the EPISODIC DRIVING fix (default-off; flips episodic-memory hollow->load-bearing; cupy strongly preferred so
the forced BTSP write is ~seconds not ~510s/store):
  SIM_BACKEND=cupy LB_EPISODIC_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only episodic-memory --repeats 2 \
      --out research/findings/raw/_load_bearing/episodic_drive.json     # expect load-bearing=1, null-control clean
Verify the SURPRISE CONFIRM fix (default-off; flips surprise-monitor hollow->load-bearing by measuring the CONFIRM turn
where the same-block lesion actually bites, instead of the contra CONTRADICT turn it never reaches; numpy is fine, the
confirm read is ~seconds):
  LB_SURPRISE_CONFIRM_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only surprise-monitor --repeats 2 \
      --out research/findings/raw/_load_bearing/surprise_confirm.json   # expect load-bearing=1, null-control clean,
      # surprise.surprised False(intact) vs True(lesion)
Verify the DISCOURSE-REGISTER DRIVING fix (default-off; flips discourse-register hollow->load-bearing; no forced env —
the register defaults spiking=True on any backend, so numpy CPU is fine, cupy only faster):
  LB_DISCOURSE_REGISTER_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only discourse-register --repeats 2 \
      --out research/findings/raw/_load_bearing/discourse_register_drive.json  # expect load-bearing=1, agent bird/dog, null clean
Verify the COMMON-GROUND DRIVING fix (default-off; flips common-ground-drives hollow->load-bearing; numpy is fine --
the ledger self-pins SIM_BACKEND=numpy, so no cupy/forced-write is needed):
  LB_CG_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only common-ground-drives --repeats 2 \
      --out research/findings/raw/_load_bearing/cg_drive.json     # expect load-bearing=1, null-control clean,
      # common_ground_drives.decision reduce(intact) vs introduce(lesion) on the re-mention turn
Verify the NON-CONTRADICTION DRIVING fix (default-off; flips noncontradiction-gate hollow->load-bearing; NO forced-
write env needed, the boot fact (dog,chase,cat)=AFFIRM is stored on any backend, so numpy is fine):
  LB_NONCONTRADICTION_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only noncontradiction-gate --repeats 2 \
      --out research/findings/raw/_load_bearing/noncontradiction_drive.json  # expect load-bearing=1, null-control clean
Verify the AFFECT-COLORING DRIVING fix (default-off; flips affect-coloring hollow->load-bearing; the Gate-B ladder read
runs on numpy for any turn -> NO cupy needed, NO env-forcing -- just remaps the probe to the strongly-affective 'emo' turn):
  LB_AFFECT_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only affect-coloring --repeats 2 \
      --out research/findings/raw/_load_bearing/affect_drive.json       # expect load-bearing=1, null-control clean
Verify the BG-ACTION-SELECTION DRIVING fix (default-off; flips bg-action-selection hollow->load-bearing by comparing the
structural `bg_select.on` field instead of the confounded top-level `abstained`; numpy is fine — no forced write):
  LB_BG_SELECT_DRIVE_PROBE=1 tools/memcap.sh 20 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only bg-action-selection --repeats 2 \
      --out research/findings/raw/_load_bearing/bg_select_drive.json    # expect load-bearing=1, change_kind=structural
Verify the PROSPECTIVE-MEMORY DRIVING fix (default-off; flips prospective-memory hollow->load-bearing; runs on ANY
backend -- no forced write, BRAIN_PMEM + BRAIN_PMEM_HEBBIAN are default-ON so the intact arm fires on the cue turn;
the driving group is formation -> 3 intervening turns -> cue so the held x cue coincidence reaches its operating point):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' LB_PMEM_DRIVE_PROBE=1 tools/memcap.sh 16 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only prospective-memory --repeats 2 \
      --out research/findings/raw/_load_bearing/pmem_drive.json         # expect load-bearing=1, null-control clean
Verify the OPEN-ENDED-GENERATION DISTRIBUTIONAL probe (default-off; the RULER swap, not a new lesion). Finding
2026-09-21-open-ended-generation-single-turn-not-load-bearing-spiking-plausibility-gate-masks-draw.md: this faculty
reads NOT load-bearing on the single-turn field-diff (a noise-dominated single soft-WTA draw -- the wrong ruler) but
IS robustly load-bearing DISTRIBUTIONALLY (6-seed GO, research/findings/raw/_load_bearing/_followon2_openended_
distributional_6seed.json): ablating the draw's likelihood collapses the plausible-fraction-of-novel-generated-
hypotheses ~0.337->~0.01. This flag reuses that GO'd _followon2 machinery (SpikingWTASampler / _gate_and_collect /
build_world, UNCHANGED) as the ruler for 'open-ended-generation' instead of the webapp brain_chat single-turn path
-- it NEVER calls _spawn_arm / never builds the tiny-demo brain; the "brain build" here is _followon2's own small
unwired WTA bank (still memcap'd per discipline):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' LB_OPEN_ENDED_DISTRIB_PROBE=1 tools/memcap.sh 10 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only open-ended-generation \
      --out research/findings/raw/_load_bearing/oeg_distributional_verify.json
      # expect load_bearing=True, verdict=regressed, null_control_clean=True (intact_a==intact_b, an independent
      # REBUILD at the identical seed -- the substrate's cfg.seed determinism, not a statistical closeness bar),
      # treatment_diffs (|intact-lesion| plausible-fraction gap) >> control_diffs (0)
Verify the WM-BINDING HOLD-QUERY probe (default-off; an adequate probe for wm-binding-advanced, which reads
not-exercised on the default `held` turn). Pre-registration:
research/findings/2026-09-23-wm-binding-holdquery-adequate-probe-PREREGISTRATION.md:
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' LB_WMB_HOLDQUERY_PROBE=1 tools/memcap.sh 12 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only wm-binding-advanced --repeats 2 --seed 42 \
      --out research/findings/raw/_load_bearing/wmb_holdquery/s42/lbf.json
Run (full measurement, capped; defer to a non-gaming window):
  tools/memcap.sh 24 -- .venv/bin/python -m research.runners.load_bearing_fraction \
      --out research/findings/raw/_load_bearing/load_bearing.json
Self-test (no brain build, no cap needed): .venv/bin/python -m research.runners.load_bearing_fraction --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# ── reuse the regression-battery machinery verbatim (no edit to that module) ─────────────────────────────────────
from research.runners.onebrain_regression_battery import (
    PROBE_TURNS,
    _EXTRA_TURNS,
    _TURN_BY_LABEL,
    FACULTY_PROBES,
    _spawn_arm as _spawn_arm_raw,
    _get_path,
    compare,
    faculty_list,
)
# The attribution discipline (tools.lab): a treatment/control PAIR must make an explicit attribution call — measuring
# both arms is not the same as asking whose the difference was (the gap#5 clamp owned 97% of a change nobody subtracted).
# Here: treatment = decision fields changed INTACT-vs-LESION; control (null) = decision fields changed INTACT-vs-INTACT-
# REBUILD (same env, rebuilt). With the harness deterministic the null is 0 -> 100% of the change is the lesion; a
# non-zero null means the "change" is partly run-to-run noise and the load-bearing verdict is NOT clean.
from tools.lab import attributable_to

# RESUME-SKIP (opt-in, env-gated; default OFF -> byte-identical to a fresh run). When LB_RESUME_SKIP_EXISTING is set,
# an arm whose output file already exists AND parses as valid JSON is LOADED instead of rebuilt -- correct because
# _spawn_arm_raw's return value IS `json.load(open(out_path))`, so a loaded arm is identical to a freshly-built one.
# This makes a killed run resumable: the expensive per-faculty brain-builds are reused off disk; only missing or
# truncated arms rebuild. A file truncated by a mid-write kill fails json.load -> falls through to a real rebuild.
_LB_RESUME = os.environ.get("LB_RESUME_SKIP_EXISTING", "").strip().lower() in ("1", "true", "yes", "on")

# ── EPISODIC DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline) ────
# WHY (diagnosis, finding 2026-09-20-hollow-episodic-drive): episodic-memory is isolated-lesion-load-bearing (the dAP
# completion collapses 0.909->0.000 under BRAIN_EPISODIC_LESION) yet reads INTEGRATED-HOLLOW here for a PROBE reason,
# not a wiring reason. Its default probe turn `episodic` ("did we discuss the dog") runs on a FRESH session with NO
# prior storage, so the INTACT recall correctly reads in_memory=False (an honest not-in-memory) -- the SAME output the
# lesion produces (an unformed-weights collapse of a memory that was never formed). The recall GATE already DRIVES the
# reply (webapp/server.py Hook A: answer/abstained/verified/episodic ALL flip with in_memory); it simply never had a
# memory to recall. Compounding it, the BTSP WRITE is cupy-gated (server._episodic_store_ok), so on the numpy probe
# backend nothing is stored even if a storing turn were added. This flag makes the instrument CONSTRUCT the driving
# condition: remap the episodic probe to the store->recall pair (battery turns `epi_store`->`epi_recall`, session
# 'epi2') and FORCE the write with BRAIN_EPISODIC_STORE=1 so it runs on ANY backend (~510s/store on numpy@2000,
# ~seconds on cupy -- the declared latency residual; speed is secondary). Then intact recalls it (in_memory=True ->
# disclosure) and the lesion collapses it (in_memory=False -> "I don't recall") -> the decision field `episodic.in_
# memory` FLIPS -> LOAD-BEARING. OFF (default) -> episodic is measured on the lone `episodic` turn exactly as the
# baseline did (hollow), and no other faculty is touched.
LB_EPISODIC_DRIVE = os.environ.get("LB_EPISODIC_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_EPISODIC_DRIVE_TURN = "epi_recall"      # the referential RECALL turn (its group is store->recall, same session)
_EPISODIC_DRIVE_ENV = {"BRAIN_EPISODIC_STORE": "1"}   # force the BTSP write to execute on the probe backend

# ── SURPRISE CONFIRM PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline) ────
# WHY (diagnosis, finding 2026-09-20-hollow-surprise-monitor-confirm-probe): surprise-monitor is isolated-lesion-load-
# bearing (BRAIN_SURPRISE_LESION zeroes the block-diagonal patient_expected->surprise prediction edges, collapsing the
# 22.8x confirm/contradict separation) yet reads INTEGRATED-HOLLOW here for a PROBE reason, not a wiring reason. Its
# default probe turn is `contra` ("the dog chase the fish") = a CONTRADICT trial: the asserted patient ('fish') lives in
# a DIFFERENT circuit block than the stored one ('cat'), and the lesioned inhibition only ever reached the SAME (stored)
# block -- so on CONTRADICT the surprise pool is un-inhibited INTACT too, and the lesion changes nothing (empirically
# intact_a_contra surprised=true, lesion_surprise_monitor surprised=true -> compare() 'pass' -> not load-bearing). The
# lesion only BITES on a CONFIRM trial (asserted==stored, SHARED block), where the intact prediction cancels excitation
# on that block (surprised=False) and the lesion removes that cancellation (surprised=True). This flag remaps the
# surprise probe to the CONFIRM turn (`confirm` = "the dog chase the cat", already in PROBE_TURNS for metacog-monitor,
# session 'surp', single-turn group -> byte-identical roster, no new turn/session/forced-write env). Grounded in the
# already-produced artifact intact_a_confirm.json: intact confirm surprised=false (surprise_hz 0.0 < threshold 2.629)
# with calib.confirm_before_max=4.398 (the pre-homeostat, PARTIALLY-inhibited confirm rate) already ABOVE 2.629 -> fully
# removing the inhibition (the lesion) fires confirm at >= that rate -> surprised flips False->True -> the decision field
# `surprise.surprised` FLIPS -> LOAD-BEARING. The reply is genuinely driven by it: webapp/server.py gates surprise_prefix
# ("That surprises me -- my mismatch monitor fired ...") on sj['surprised'] and splices it into the answer, so a CONFIRM-
# turn lesion spuriously annotates a plain restatement -- a real user-visible diff the CONTRADICT probe can never expose
# (both arms already carry the notice there). OFF (default) -> surprise is measured on the `contra` turn exactly as the
# baseline did (hollow), and no other faculty is touched. No base_env: the confirm turn is deterministic (homeostat-
# calibrated at build), so the null control is clean with no forced write.
LB_SURPRISE_CONFIRM = os.environ.get("LB_SURPRISE_CONFIRM_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_SURPRISE_CONFIRM_TURN = "confirm"       # the CONFIRM turn (asserted==stored, shared block) where the lesion bites

# ── DISCOURSE-REGISTER DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline)
# WHY (diagnosis, finding 2026-09-20-hollow-discourse-register-drive): discourse-register is isolated-lesion-load-bearing
# (the who-was-before read collapses when the prev spiking slots are silenced) yet reads INTEGRATED-HOLLOW here for a
# PROBE reason, not a wiring reason. Its default probe `dr_c` ('dog chase cat' -> 'then bird chase worm' -> 'who was
# doing it before') has its correct before-agent be 'dog', which is referents[0] — and referents[0] is EXACTLY the
# register's identity index (ident=0, _d3_event_connective_derisk.make_connective_task). The LESION
# (_PrevSilencePairRegister.observe, d3_discourse_event_register_production_organ.py) collapses the held prev slots by
# FORCING them to that same identity index -> forced 'dog'. So the INTACT read ('dog', via the learned RNN shift +
# FS-WTA re-discretization) and the LESION read ('dog', via forced-identity) are the IDENTICAL agent: an index
# collision, not a failure of the read to reach the reply (it demonstrably does — webapp/server.py's before/now
# short-circuit early-returns a JSONResponse carrying discourse_register straight from answer_before). Compared fields
# discourse_register.agent/.abstained therefore see zero diff -> hollow. This flag remaps the probe to the 'dr2' triple
# ('bird chase worm' -> 'then dog chase cat' -> 'who was doing it before'), which SWAPS the roles so the correct
# before-agent is 'bird' = referents[3] != identity 0: intact reads the held prev agent 'bird', the lesion still forces
# 'dog' -> discourse_register.agent FLIPS 'bird' vs 'dog' -> LOAD-BEARING. No base_env needed (the register defaults
# spiking=True on ANY backend, unlike episodic's cupy-gated BTSP write). OFF (default) -> discourse-register is measured
# on the lone `dr_c` turn exactly as the baseline did (hollow), and no other faculty is touched.
LB_DISCOURSE_REGISTER_DRIVE = os.environ.get("LB_DISCOURSE_REGISTER_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_DISCOURSE_DRIVE_TURN = "dr2_c"          # the before-query turn (its group is bird/worm -> shift dog/cat -> before?)
# Static-verification anchors for the ident-collision guard (see selftest). MUST mirror the production register build
# site (brain_chat_tui.py: make_discourse_register(["dog","cat","fish","bird","worm","ball"])) and the identity index
# (_d3_event_connective_derisk.make_connective_task: ident = 0). The whole point of the remap is that the correct
# before-agent index is NOT this identity index (which the lesion forces the held prev slot to).
_DR2_PROD_REFERENTS = ["dog", "cat", "fish", "bird", "worm", "ball"]
_DR2_IDENT = 0
# ── COMMON-GROUND DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline) ─
# WHY (diagnosis, finding 2026-09-20-hollow-common-ground-drives-drive): common-ground-drives is isolated-lesion-load-
# bearing (its own lesion_note: BRAIN_CG_DRIVES_LESION builds the ledger recurrence at weight 0 -> a re-mentioned
# referent can no longer read grounded -> the reduced-reference lead VANISHES) yet reads INTEGRATED-HOLLOW on the
# default probe for a PROBE reason, not a wiring reason. Its default probe turn `well` ("the wolf bites the apple")
# runs on a FRESH session, and 'wolf'/'bites'/'apple' are NOT build-time KB concepts, so gnw_thought_swap._extract_topic
# returns None (no grounded token) -> common_ground_ledger_production_organ.observe_turn takes its `if not topic:
# return {..., "decision": None, ...}` branch IDENTICALLY on the intact AND lesion arms (decision=None==None) -> hollow.
# Even with a grounded first-mention token, the load-bearing divergence only appears on a RE-MENTION of an already-
# grounded referent (intact reads REDUCE via the held NMDA bump; the lesioned recurrence=0 ledger cannot hold it and
# stays INTRODUCE) -- a single fresh first-mention turn can never construct that fork. This flag makes the instrument
# CONSTRUCT the driving condition: remap the common-ground probe to the mention->re-mention pair (battery turns
# `cg_mention1`->`cg_mention2`, session 'cg2') over 'dog' (a build-time KB agent, found by _extract_topic with no
# gate-ordering/OOV issue). Then the INTACT ledger holds the grounded slot (decision=reduce) while the lesioned ledger
# collapses it (decision=introduce) -> the decision field `common_ground_drives.decision` FLIPS -> LOAD-BEARING. No
# forced-write / backend flag is needed: common_ground_ledger_production_organ pins SIM_BACKEND=numpy for its own
# bridge unconditionally, so the driving pair works on the default numpy probe backend (base_env stays {}). OFF
# (default) -> common-ground-drives is measured on the lone `well` turn exactly as the baseline did (hollow), and no
# other faculty is touched.
LB_CG_DRIVE = os.environ.get("LB_CG_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_CG_DRIVE_TURN = "cg_mention2"           # the RE-MENTION turn (its group is mention1->mention2, same session 'cg2')
_CG_DRIVE_ENV: dict = {}                 # no forced write / backend flag needed (the ledger pins SIM_BACKEND=numpy)
# ── NON-CONTRADICTION DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline)
# WHY (diagnosis, finding 2026-09-20-hollow-noncontradiction-gate-drive): noncontradiction-gate is isolated-lesion-
# load-bearing (the 6-seed B3 GO: disabling negation storage flips 0->18 false-accepts, the canonical negation reads
# "yes" on the substrate) yet reads INTEGRATED-HOLLOW here for a PROBE reason, not a wiring reason. Its default probe
# turn is `well` ("the wolf bites the apple") — a fresh TEACH of BRAND-NEW vocabulary (the battery's own comment,
# onebrain_regression_battery.py:176: "wolf/bite/apple are new vocabulary"). So `ask_yes_no("wolf","bite","apple")`
# has NO stored belief and legitimately reads "unknown" on the INTACT substrate too — the SAME value the lesioned
# _RecallShim forces — so on/reject/recalled_yn/asserted_polarity are byte-identical intact vs lesion (accept /
# reject=False / "unknown"). The gate GENUINELY drives the reply (webapp/server.py:5943-5954: reject=True early-returns
# the rejection message + the noncontradiction block; reject=False falls through to the normal reply, block still
# attached), and is genuinely wired to a real recallable belief — the tiny-demo brain stores (dog,chase,cat)=AFFIRM at
# BUILD time (brain_chat_tui `_build*` hear-loop; a build-time store, present on ANY backend), which is exactly why the
# battery's pre-existing `confirm`/`metacog` probes already recall "yes" on it. This flag remaps the noncontradiction
# probe to a turn that ASSERTS the NEGATED form of that boot fact: intact recalls "yes" (stored AFFIRM) -> stored !=
# asserted(NEGATE) -> REJECT (recalled_yn="yes", stored_polarity="AFFIRM"); the lesion forces "unknown" -> ACCEPT
# (recalled_yn="unknown", stored_polarity=None) -> the decision fields FLIP -> LOAD-BEARING. UNLIKE episodic, NO forced-
# write env is needed (the fact is stored unconditionally at boot on every backend), so base_env stays {} in both arms
# -> the NULL control is a plain rebuild and is byte-identical. OFF (default) -> noncontradiction is measured on the
# lone `well` teach turn exactly as the baseline did (hollow), and no other faculty is touched.
LB_NONCONTRADICTION_DRIVE = os.environ.get("LB_NONCONTRADICTION_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_NONCONTRA_DRIVE_TURN = "noncontra_neg"   # a single fresh-session turn: assert NEGATE of the AFFIRM boot fact (dog,chase,cat)
_NONCONTRA_DRIVE_FIELDS = ["noncontradiction.on", "noncontradiction.reject", "noncontradiction.recalled_yn",
                           "noncontradiction.asserted_polarity", "noncontradiction.stored_polarity"]


def _noncontra_probe_parses_negated_boot_fact():
    """STATIC check (no brain build): the driving turn's TEXT parses, through the SAME production organ front-end the
    webapp uses, to the NEGATED form of the AFFIRM boot fact -> (agent,action,patient,polarity)==(dog,chase,cat,NEGATE).
    This is the load-bearing static claim: intact will recall "yes" on (dog,chase,cat) and REJECT a NEGATE assertion,
    while the lesion forces "unknown" and ACCEPTS -> the decision fields diverge. Import is lazy (parse-only; no brain)."""
    try:
        from research.runners.b3_noncontradiction_production_organ import extract_polar_assertion
        turn = _TURN_BY_LABEL.get(_NONCONTRA_DRIVE_TURN)
        if not turn:
            return False
        parsed = extract_polar_assertion(turn[1])   # turn = (label, message, session, reset, percept, rich)
        return parsed == ("dog", "chase", "cat", "NEGATE")
    except Exception:
        return False
# ── AFFECT-COLORING DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline) ──
# WHY (diagnosis, finding 2026-09-20-hollow-affect-coloring-drive): affect-coloring is isolated-lesion-load-bearing (the
# `affect_out` transmission gate collapses the ladder differential to 0.0 under BRAIN_AFFECT_LESION -- the same class of
# neural cut episodic uses) yet reads INTEGRATED-HOLLOW here for a PROBE reason, not a wiring reason. Its default probe
# turn `well` ("the wolf bites the apple") is MOOD-NEUTRAL: appraise_text returns n_hits=0 (none of wolf/bites/apple/the
# pass the Warriner _STRONG_MARGIN salience gate -- "wolf" is |3.9-5|<margin, the rest are not in WARRINER at all), so
# webapp/server.py _update_session_mood HOLDS the prior mood (0.0) and read_differential injects a 0.0 appraisal through
# settle/ramp/drive-off/read REGARDLESS of the lesion flag -- the affect_out gate only bites when there is a NONZERO
# differential to clamp. So BOTH the intact and BRAIN_AFFECT_LESION arms read ~baseline (valence_sign="0", tone_token="")
# -> IDENTICAL -> NOT load-bearing. The reply IS already colored by the ladder read (server.py: valence_sign / tone_token
# / manner_template / _mood_tone_level ALL flow from the neural differential); the probe simply never gives the ladder
# anything to color. This flag remaps the affect-coloring probe to the strongly-affective `emo` turn ("Wonderful! I am so
# happy and delighted, this is fantastic and amazing!") -- ALREADY a self-contained turn in PROBE_TURNS (session 'emo',
# reset=True, its own single-turn group), so NO new turn / dependency chain is added and NO env-forcing is needed (the
# Gate-B ladder read runs on numpy for ANY turn, UNLIKE episodic's cupy-gated BTSP write -> base_env stays {}). On `emo`,
# appraise_text hits n_hits=5 strongly-positive words -> the session mood goes non-neutral -> intact reads valence_sign=
# "+" (a nonzero positive differential) while the lesion clamps affect_out=0 -> differential 0.0 -> valence_sign="0" ->
# the decision fields `affect.valence_sign` + `affect.tone_token` FLIP -> LOAD-BEARING. OFF (default) -> affect-coloring
# is measured on the lone `well` turn exactly as the baseline did (hollow), and no other faculty is touched.
LB_AFFECT_DRIVE = os.environ.get("LB_AFFECT_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_AFFECT_DRIVE_TURN = "emo"      # the strongly-affective turn (its group is ['emo'] -- its own isolated single-turn session)
# ── BG-ACTION-SELECTION DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow
# baseline) ───────────────────────────────────────────────────────────────────────────────────────────────────────
# WHY (diagnosis, finding 2026-09-20-hollow-bg-action-selection-drive): bg-action-selection genuinely DRIVES the reply
# on the `bgdots` probe ('...') — the two-channel spiking basal-ganglia race commits STAY_SILENT and the turn short-
# circuits with a HOLD line (webapp/server.py:4714-4723, which is the ONLY place the `bg_select` key is written). But
# the ONE field the battery row compares is the top-level `abstained` (onebrain_regression_battery.py:272), which is
# CONFOUNDED: the lesion `BRAIN_BG_SELECT_LESION=1` maps to `arousal` (organ:120-121), which skips the entire salience-
# bias barrage (organ:162-177) so the race never commits, `decide_action` returns None, and the turn FALLS THROUGH past
# the BG block to the single-fact path, where a punctuation-only '...' has no comprehensible content -> `answer,
# abstained, verified = "I don't know about that.", True, False`. So BOTH arms read `abstained=True` — the intact arm
# via the BG HOLD short-circuit, the lesion arm via a completely independent no-content abstain -> compare()="pass" ->
# NOT load-bearing, even though the answer TEXT (HOLD_TEXT vs "I don't know about that.") AND the `bg_select` block's
# presence differ. This flag remaps the compared field from the confounded `abstained` to `bg_select.on` — present+True
# only when the BG block's own short-circuit fired (intact), absent on the lesioned fallback -> compare() sees a field
# present intact / absent lesioned -> `regressed`, change_kind `structural` (the gold-standard robust diff). No new turn
# or session is needed (the default `bgdots` turn already puts the race in its designed STAY_SILENT-favored regime); OFF
# (default) -> bg-action-selection is measured on `abstained` exactly as the baseline did (hollow), no other faculty
# touched. Same class of fix as LB_EPISODIC_DRIVE_PROBE: the mechanism is load-bearing on the reply, the instrument's
# (turn, compared-field) pair simply could not see it — here because the field collided with an unrelated abstain path.
LB_BG_SELECT_DRIVE = os.environ.get("LB_BG_SELECT_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_BG_SELECT_DRIVE_TURN = "bgdots"                 # the content-empty turn that puts the BG race in its STAY_SILENT regime
_BG_SELECT_DRIVE_FIELDS = ["bg_select.on"]       # the structural field the independent no-content fallback never sets
# ── PROSPECTIVE-MEMORY DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the hollow baseline) ─────
# WHY (diagnosis, finding 2026-09-20-prospective-memory-drive-v2): prospective-memory is isolated-lesion-load-bearing
# (research/runners/_prospective_memory_production_verify.py rows A/C: the intact latch fires on the cue turn,
# BRAIN_PMEM_LESION collapses the held assembly -> the SAME cue stays silent) yet reads INTEGRATED-HOLLOW here for a
# PROBE reason, not a wiring reason. Its default probe turn `pmem_form` compares field `prospective.held`, which
# form_intention() sets to the compile-time literal True UNCONDITIONALLY (the lesion's real effect, `held_after_lesion`,
# is a DIFFERENT field the probe never compares) -> intact True == lesion True -> `pass` -> hollow. But the FIRST fix
# (a 2-turn formation->cue group comparing `prospective.fired`) ALSO read hollow (treat=0): brain-build-verified, the
# intact arm did NOT fire either. ROOT CAUSE (measured organ-level): prospective memory is an intention held ACROSS
# INTERVENING ACTIVITY and released at a LATER cue -- the SFA/NMDA held x cue coincidence only reaches its operating
# point after the hold is advanced by intervening turns (intact rel_A 0.163@n=0 -> 0.221@n=1 -> 0.340@n=3; FIRE_THR=
# 0.2), so a ZERO-DELAY formation->cue does not fire even intact (there is nothing "prospective" about an immediate
# cue). This flag makes the instrument run the NATURAL prospective protocol: remap the prospective probe to the
# formation -> 3 intervening turns -> cue group (battery turns `pmem_form2`..`pmem_cue`, session 'pmem2') and compare
# `prospective.fired`. NO base_env forcing is needed: BRAIN_PMEM + BRAIN_PMEM_HEBBIAN are default-ON, so the ordinary
# intact build learns the cue->action binding one-shot at formation, holds it across the 3 distractors, and fires on
# the cue turn (fired=True); the BRAIN_PMEM_LESION arm collapses the latch at formation so the cue stays silent
# (fired=False; rel_A ~0.04 at every n) -> the decision field `prospective.fired` FLIPS -> LOAD-BEARING. OFF (default)
# -> prospective is measured on the lone `pmem_form` turn exactly as the baseline did (hollow), no other faculty touched.
LB_PMEM_DRIVE = os.environ.get("LB_PMEM_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_PMEM_DRIVE_TURN = "pmem_cue"            # the CUE turn (its group is formation -> 3 intervening turns -> cue, same session 'pmem2')
# ── OPEN-ENDED-GENERATION DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the hollow baseline) ───
# WHY (diagnosis, finding 2026-09-20-gap-open-ended-generation-v2): the default open-ended probe `rich_open` ("what
# might a dog chase") is integrated-HOLLOW because the tiny KB has ONE 'chase' fact -- (dog,chase,cat) -- already
# stored, so the only reachable (dog,chase,?) patient is novelty-excluded -> _generate_hypothesis abstains in BOTH
# arms (intact == lesion). The v1 fix taught 9 chase facts but STILL read treat=0 on the real brain: the stored
# 'cat' (co-occurrence weight 2 with (dog,chase)) TIED the twice-taught 'rabbit' and won the intact spiking-WTA
# argmax, so the intact draw FIXATED on 'cat' (novelty-excluded) and dead-ended to abstain -- the likelihood
# ablation had nothing to change. This flag remaps the measurement to a TEACH->ASK group ('oe_t1..oe_t9' -> 'oe_ask',
# session 'oe2') that teaches a NATURAL predator-prey chase KB where 'rabbit' is chased by FOUR predators so its
# (dog,chase,rabbit) weight (4) STRICTLY dominates the stored cat's (2): the INTACT likelihood-weighted spiking draw
# then peaks the NOVEL 'rabbit' (volunteers it), while the LESION's uniform draw (BRAIN_SPIKING_DRAW_LESION -> the
# now-honored ablate on draw_from_weights, this branch's wiring fix) has no likelihood bias and selects among all
# novel plausible patients -> the decision field `hypothesis_svo` (+ the rendered `answer`) differs -> LOAD-BEARING.
# OFF (default) -> open-ended is measured on the lone `rich_open` turn exactly as the baseline (hollow); no other
# faculty touched.
LB_OPEN_ENDED_DRIVE = os.environ.get("LB_OPEN_ENDED_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_OPEN_ENDED_DRIVE_TURN = "oe_ask"        # the rich=True open-ended ASK turn (its group is oe_t1..oe_t9 -> oe_ask, one session)
# BOTH arms admit candidates via the host #3E plausibility gate: on the tiny KB the DEFAULT-ON spiking plausibility
# read is too conservative on the weak agent-action edge (_related(dog,chase), co-occurrence 1) to admit ANY novel
# candidate, so _generate_hypothesis abstains in BOTH arms and the draw is MASKED (measured; the v2 diagnosis). This
# base_env is applied to intact AND lesion identically, so the ONLY inter-arm difference remains the draw lesion --
# it ISOLATES the draw's load-bearingness, it does not create it. (Under the default gate the integrated faculty
# abstains -> the honest residual: a richer KB or a less-conservative gate operating point is needed to unmask the
# draw under the default spiking gate.)
_OPEN_ENDED_DRIVE_ENV = {"BRAIN_SPIKING_PLAUSIBILITY": "0"}

# ── SWAP-DRIVES ADEQUATE PROBE (opt-in, env-gated; default OFF -> byte-identical to the pre-change battery) ──────────
# WHY (2026-09-23 all-fixes 6-seed battery: swap-drives-response NOT-EXERCISED on every seed). The default probe turn
# `held` ('the wolf watches the owl', after 'the fox and the wolf walked in') cannot exercise the GNW thought-swap
# drive (board #77/#85, webapp/swap_drives_chat.py) for two probe reasons, not wiring reasons: (1) wolf/owl/fox are
# not build-time KB concepts, so gnw_thought_swap._extract_topic returns None (no_topic_hold -- no swap is ever due);
# (2) the turn is answered by the role-binding REPAIR short-circuit, which returns before server.py attaches
# `swap_drives` -> the fields are absent in BOTH arms (measured: allfixes2/s42/swap-drives-response/*.json). This flag
# remaps the faculty to an ordinary topic conversation on session 'sw2' over the boot facts: `sw_open` ('what does
# the dog chase' -> held topic 'dog'), `sw_hold` (same question: a HOLD, no swap due), `sw_switch` ('what does the cat
# eat' -> a salient competing grounded topic 'cat'). Whether the intact brain swaps on sw_switch is the spiking
# mismatch/eviction/vacancy chain's decision (it can fail: mismatch_held_no_swap), and whether that reaches the reply
# is the server's lead prepend on the single-fact path (it can fail: a short-circuit, or a downstream overwrite).
# The pre-registered gate (research/findings/2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md) is scored by
# _swap_drive_score below: LOAD-BEARING requires the REPLY (`answer`) to differ intact-vs-lesion on sw_switch, a clean
# intact-vs-intact null on all three turns, a reproduced lesion, AND a clean CONTRAST -- the lesion must NOT change
# sw_open/sw_hold, where no swap is due (a lesion that alters replies with no topic change is non-specific, not the
# swap). No base_env (the swap drive is default-ON; the swap workspace self-seeds from BRAIN_CHAT_SEED). OFF (default)
# -> swap-drives-response is measured on `held` exactly as before (not-exercised), and no other faculty is touched.
LB_SWAP_DRIVE = os.environ.get("LB_SWAP_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_SWAP_DRIVE_TURN = "sw_switch"                     # the topic-change turn (group: sw_open -> sw_hold -> sw_switch)
_SWAP_DRIVE_CONTRAST_TURNS = ("sw_open", "sw_hold")  # no swap due on these -> the lesion must NOT change them
_SWAP_DRIVE_FIELDS = ["answer", "swap_drives.swapped", "swap_drives.reason", "swap_drives.lead"]
_SWAP_DRIVE_REPLY_FIELD = "answer"                 # the reply-level field whose change the headline requires
# Static anchor for the topic-extraction check (selftest): MUST mirror research/runners/brain_chat_tui.py
# _build_tiny_demo's `facts` (the tiny-demo KB the battery worker builds). The whole point of the remap is that the
# probe turns' first grounded concept is dog, dog, cat (and that `held`'s is None).
_SW_TINY_DEMO_FACTS = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
                       ("dog", "chase", "cat"), ("cat", "eat", "fish")]


def _swap_probe_topics_static():
    """STATIC check (no brain build): the grounded topic the PRODUCTION extractor (gnw_thought_swap._extract_topic)
    reads off each probe turn, against a stub composer carrying the tiny-demo KB. Returns {label: topic}. Import is
    lazy (webapp.gnw_thought_swap imports the swap de-risk module, which is import-light; no substrate is built)."""
    from webapp import gnw_thought_swap as _GTS

    class _StubComposer:
        kb = [({"agent": a, "action": v, "patient": p}, None) for a, v, p in _SW_TINY_DEMO_FACTS]
    out = {}
    for lab in ("held",) + _SWAP_DRIVE_CONTRAST_TURNS + (_SWAP_DRIVE_TURN,):
        out[lab] = _GTS._extract_topic(_TURN_BY_LABEL[lab][1], _StubComposer())
    return out


def _swap_score_synthetic(nonspecific=False):
    """Run the FULL _score_swap_drive path (not just the pure scorer) on synthetic arms -- added after the s42 smoke
    crashed in _score_swap_drive on an unimported helper that the pure-scorer selftests never reached. Returns
    (load_bearing, verdict)."""
    def turn(swapped, reason, lead, ans):
        return {"answer": lead + ans, "swap_drives": {"swapped": swapped, "reason": reason, "lead": lead}}
    intact = {"sw_open": turn(False, "first_thought", "", "a"), "sw_hold": turn(False, "same_topic_hold", "", "a"),
              "sw_switch": turn(True, "topic_change_swap", "On cat, then — ", "b")}
    lesion = {"sw_open": turn(False, "first_thought", "", "a"),
              "sw_hold": turn(False, "same_topic_hold", "", "a" + ("!" if nonspecific else "")),
              "sw_switch": turn(False, "mismatch_held_no_swap", "", "b")}
    row = ("swap-drives-response", _SWAP_DRIVE_TURN, list(_SWAP_DRIVE_FIELDS), False)
    treat_pf = compare(intact, lesion, faculties=[row])["per_faculty"][0]
    res = {"null_control_clean": True, "verdict": treat_pf["verdict"], "load_bearing": True}
    _score_swap_drive(res, row, treat_pf, intact, dict(intact), lesion, True)
    return res["load_bearing"], res["verdict"]


def _swap_drive_score(treat_verdict, treat_diffs, null_clean, reproduced, contrast_diffs, contrast_exercised):
    """The PRE-REGISTERED decision for the swap-drives adequate probe (pure; selftested on synthetic inputs).
    Returns (load_bearing, verdict). Order matters: every UNDEFINED condition is checked before any positive.
      * switch fields absent in both arms           -> (None, 'not-exercised')
      * a contrast turn's fields absent (intact)     -> (None, 'contrast-undefined')   [UNDEFINED is never a pass]
      * intact-vs-intact null differs (any turn)     -> (None, 'noisy-null-control')
      * lesion changes a no-swap-due contrast turn   -> (None, 'nonspecific-lesion')
      * switch identical intact-vs-lesion            -> (False, 'pass')
      * lesion diff does not reproduce               -> (None, 'noisy')
      * switch differs but NOT in the reply `answer` -> (False, 'trace-only')          [the reply did not depend on it]
      * otherwise                                    -> (True, 'regressed')"""
    if treat_verdict == "not-exercised":
        return None, "not-exercised"
    if not contrast_exercised:
        return None, "contrast-undefined"
    if not null_clean:
        return None, "noisy-null-control"
    if contrast_diffs:
        return None, "nonspecific-lesion"
    if treat_verdict == "pass":
        return False, "pass"
    if not reproduced:
        return None, "noisy"
    if not any(d.get("field") == _SWAP_DRIVE_REPLY_FIELD for d in (treat_diffs or [])):
        return False, "trace-only"
    return True, "regressed"

# ── OPEN-ENDED-GENERATION DISTRIBUTIONAL PROBE (opt-in, env-gated; default OFF -> byte-identical) ──────────────────
# WHY (finding 2026-09-21-open-ended-generation-single-turn-not-load-bearing-spiking-plausibility-gate-masks-draw):
# even the LB_OPEN_ENDED_DRIVE_PROBE remap above (the best single-turn instrument can do) reads treat=0 on the real
# brain -- the single soft-WTA draw is OU-noise-dominated for the WHICH-patient choice at this operating point, so
# ablating the likelihood does not reliably flip ONE turn's volunteered patient. This is an INSTRUMENT mismatch, not
# an inert mechanism: the SAME draw's lesion collapses the aggregate plausible-fraction-of-novel-generated-
# hypotheses 0.337->~0.01 in the _followon2_spiking_wta_sampler_derisk distributional metric (6-seed GO, research/
# findings/raw/_load_bearing/_followon2_openended_distributional_6seed.json), with a SHUFFLED-graph null already
# proving the effect is the real co-occurrence structure, not noise ("the instrument is part of the emulation",
# CLAUDE.md's wall-reframe). THIS flag swaps the RULER for 'open-ended-generation' ONLY: instead of a webapp
# brain_chat single-turn field-diff, it reuses the _followon2 machinery UNCHANGED (SpikingWTASampler /
# _gate_and_collect / build_world -- no re-derivation) to build the SHARED world once, then draw from THREE
# independent samplers at the IDENTICAL seed: intact_a (measured), intact_b (the NULL control -- an independent
# REBUILD at the identical seed/params; per CLAUDE.md's cfg.seed determinism guarantee ("build twice at one seed ...
# identical -> seeded") this must read EXACTLY equal, not merely close), and lesion (ablate_likelihood=True, the
# SpikingWTASampler knob the 6-seed GO already lesions). load-bearing iff the null is CLEAN (0 diff -- a
# determinism claim, not a statistical threshold) AND the lesion's plausible-fraction differs from intact_a's --
# exactly the null-control discipline every OTHER driving probe in this file applies (rebuild-vs-rebuild must
# match; lesion-vs-intact must not), just measured on a DISTRIBUTIONAL fraction instead of a categorical decision
# field. If the null is NOT clean, this is reported honestly (verdict=noisy-null-control, load_bearing=None) --
# never tuned to force a result. OFF (default) -> open-ended-generation is measured on whichever single-turn path
# is active (the baseline `rich_open` turn, or LB_OPEN_ENDED_DRIVE_PROBE's teach->ask remap) exactly as before; this
# path NEVER calls _spawn_arm and NEVER builds the webapp tiny-demo brain, so every other faculty + the battery's
# PROBE_TURNS/roster is untouched.
LB_OPEN_ENDED_DISTRIB = os.environ.get("LB_OPEN_ENDED_DISTRIB_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
# The _followon2 GO's own validated operating point (n_attempts_spiking default); overridable for a faster/slower
# verify without touching the shipped default. Read once at import time, like every other env-gated knob here.
_OED_N_ATTEMPTS = int(os.environ.get("LB_OPEN_ENDED_DISTRIB_N_ATTEMPTS", "800"))
_OED_MIN_EFFECT = 1e-9   # a clean (deterministic) null makes ANY nonzero lesion effect real, not sampling noise

# ── AFFECT->TONE OPEN-OUTPUT RULER (LB_AFFECT_TONE_OPEN_PROBE, default OFF) ─────────────────────────────────────
# Swaps the `affect-coloring` faculty's ruler from the TEMPLATED single-turn decision field (affect.valence_sign /
# affect.tone_token, read off the brain_chat return dict) to the DIRECTIONAL, INDEPENDENT-LEXICON, 6-seed OPEN-output
# tone probe -- the right ruler for a faculty whose real job is steering the FREELY-GENERATED reply (roadmap SS8;
# mirrors LB_OPEN_ENDED_DISTRIB_PROBE, which swapped the open-ended-generation ruler for the same reason). The heavy
# measurement itself (36 fresh-subprocess full-webapp-brain arms on the linattn WKV mouth) lives in
# research.runners._lbf_affect_tone_open_output_derisk -- far too heavy to inline in the battery -- so this ruler
# READS that runner's canonical verdict artifact and surfaces the per-seed directional load-bearing signal into the
# #1 metric's row (read-don't-derive, the verdict.reads discipline). OFF (default) -> affect-coloring is measured on
# the templated field EXACTLY as before; this path builds NO brain and imports nothing new, so every other faculty
# and the off battery are byte-identical.
LB_AFFECT_TONE_OPEN = os.environ.get("LB_AFFECT_TONE_OPEN_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_AFFECT_TONE_OPEN_ARTIFACT = os.environ.get(
    "LB_AFFECT_TONE_OPEN_ARTIFACT",
    "research/findings/raw/_affect_tone_open_output/affect_tone_open_output_verdict.json")

# ── WM-BINDING HOLD-QUERY PROBE (LB_WMB_HOLDQUERY_PROBE, default OFF -> byte-identical) ─────────────────────────
# WHY: wm-binding-advanced (the D6 multi-referent WM organ, lesion BRAIN_MULTIREF_LESION = recur 0, the slow-NMDA
# hold killed) reads NOT-EXERCISED on every seed of the 2026-09-23 6-seed battery. Its default probe `held` ('the wolf
# watches the owl') names only ONE referent the organ's hand lexicon admits ('owl' is not on _REFERENT_NOUNS), so
# judge() returns None, and the turn also exits through the comprehension-repair return, so no `multiref` key is on
# the reply. The organ's only reply path is its HOLD-QUERY read-out: after a turn introduces >=2 referents it knows,
# "who are we talking about" is answered by reading every held referent back off the spiking buffer. This flag remaps
# the faculty to that exchange: 'wmb_intro' ('the fox and the wolf walked in', both on the hand lexicon) ->
# 'wmb_ask' ('who are we talking about'), comparing the REPLY (`answer`). It adds two pre-registered adequacy
# conditions on the INTACT arm and a SPECIFICITY control (the same ask after a ONE-referent intro, 'wmb1', where the
# organ is out of scope -- the lesion must NOT change that reply). Pre-registration:
# research/findings/2026-09-23-wm-binding-holdquery-adequate-probe-PREREGISTRATION.md.
# Honest scope (declared): the referent IDENTITIES carried between turns live in the organ's host codebook
# (`_slot_of_ref`); each load resets the buffer and re-writes, so the spiking hold carries the referents across the
# within-load write->hold->hold span, not across turns. Extraction is the host lexicon; the register read is a host
# argmax over firing rates; the read-out sentence is a host template. OFF -> the `held` row exactly as before.
LB_WMB_HOLDQUERY = os.environ.get("LB_WMB_HOLDQUERY_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_WMB_DRIVE_TURN = "wmb_ask"
_WMB_DRIVE_FIELDS = ["answer"]                    # the REPLY is the pre-registered decision field
_WMB_CONTROL_TURN = "wmb1_ask"
_WMB_REFERENTS = ("fox", "wolf")                  # the two referents 'wmb_intro' introduces (both on the hand lexicon)
# report-only mechanism fields (never gate the verdict; recorded so the read-out is auditable per arm)
_WMB_MECH_FIELDS = ["multiref.kind", "multiref.is_hold_query", "multiref.n_referents", "multiref.recovered",
                    "multiref.all_recovered", "multiref.hold_alive_min", "inner_state_readout", "abstained"]


def _wmb_words(text):
    import re as _re
    return set(_re.findall(r"[a-z']+", (text or "").lower()))


def _wmb_mech(arm, turn):
    """Report-only: the organ's own state on `turn` of `arm` (the pre-registered mechanism fields)."""
    from research.runners.onebrain_regression_battery import _get_path
    r = (arm or {}).get(turn) or {}
    return {f: _get_path(r, f)[1] for f in _WMB_MECH_FIELDS} | {"answer": r.get("answer")}


def _wmb_adequacy(intact_a, ctrl_a, ctrl_b, ctrl_les, lesioned=None):
    """The pre-registered adequacy + specificity gate for LB_WMB_HOLDQUERY_PROBE (pure; no brain build).

    A1 (route): the INTACT 'wmb_ask' reply was produced by the organ's hold-query read-out (multiref.kind=='query',
       is_hold_query True) -- the ask was not intercepted by another route.
    A2 (two referents): the INTACT read-out holds >=2 referents AND the INTACT reply names BOTH 'fox' and 'wolf'.
    S1 (specificity): on the 1-referent control 'wmb1_ask', the INTACT reply is NOT a multiref read-out (organ out of
       scope), intact-vs-intact-rebuild `answer` is identical, and intact-vs-LESION `answer` is identical.
    A1 is required on the LESION arm too when it is passed (review v2:7a3b94367: a lesion ask intercepted elsewhere
    would otherwise count a route change as a hold effect).
    Returns (override_verdict_or_None, report). override is None iff all hold; else a named non-pass verdict.
    NOTE (2026-09-24 relabel): this probe is an INTEGRITY SMOKE, not load-bearing evidence -- its reply template's only
    input is the lesioned buffer, so once A1 holds the treatment is predetermined. measure_faculty never lets it set
    load_bearing (see `integrity_smoke`)."""
    rep = {"A1_route": None, "A2_two_referents": None, "S1_control_out_of_scope": None,
           "S1_control_null_clean": None, "S1_control_lesion_unchanged": None}
    ia = (intact_a or {}).get(_WMB_DRIVE_TURN) or {}
    mr = ia.get("multiref") if isinstance(ia.get("multiref"), dict) else {}
    rep["A1_route"] = bool(mr.get("kind") == "query" and mr.get("is_hold_query") is True)
    if lesioned is not None:
        il = lesioned.get(_WMB_DRIVE_TURN) or {}
        lmr = il.get("multiref") if isinstance(il.get("multiref"), dict) else {}
        rep["A1_route_lesion"] = bool(lmr.get("kind") == "query" and lmr.get("is_hold_query") is True)
        rep["A1_route"] = bool(rep["A1_route"] and rep["A1_route_lesion"])
    rep["A2_two_referents"] = bool((mr.get("n_referents") or 0) >= 2
                                   and all(w in _wmb_words(ia.get("answer")) for w in _WMB_REFERENTS))
    if ctrl_a is None or ctrl_b is None or ctrl_les is None:
        return "arm-build-failed", rep
    ca = ctrl_a.get(_WMB_CONTROL_TURN) or {}
    cb = ctrl_b.get(_WMB_CONTROL_TURN) or {}
    cl = ctrl_les.get(_WMB_CONTROL_TURN) or {}
    if any(isinstance(t, dict) and t.get("_error") for t in (ca, cb, cl)):
        return "arm-build-failed", rep
    cmr = ca.get("multiref") if isinstance(ca.get("multiref"), dict) else {}
    rep["S1_control_out_of_scope"] = bool(cmr.get("kind") != "query")
    rep["S1_control_null_clean"] = bool("answer" in ca and ca.get("answer") == cb.get("answer"))
    rep["S1_control_lesion_unchanged"] = bool("answer" in ca and ca.get("answer") == cl.get("answer"))
    if not rep["A1_route"]:
        return "probe-inadequate:route", rep
    if not rep["A2_two_referents"]:
        return "probe-inadequate:two-referents", rep
    if not rep["S1_control_out_of_scope"]:
        return "control-inadequate", rep
    if not rep["S1_control_null_clean"]:
        return "noisy-null-control", rep
    if not rep["S1_control_lesion_unchanged"]:
        return "off-target-lesion", rep
    return None, rep


# ── WM-BINDING ORDINARY-CONTENT PROBE (LB_WMB_CONTENT_PROBE, default OFF -> byte-identical) ─────────────────────
# WHY (adversarial review v2:7a3b94367): the hold-query probe above cannot fail -- its reply is a template whose only
# input is the buffer the lesion disables. This probe asks the question that CAN fail: does the organ's held state
# change an ORDINARY content reply? Two content-swapped sessions (fox/wolf 'wmc', cat/dog 'wmcx') each LOAD two
# referents on the intro and then ask an ordinary transitive; the organ's only path into that reply is this session's
# held WM focus, co-driven through the one-brain d6->comprehension cross-edge. The lesion is EDGE-CONFINED
# (BRAIN_MULTIREF_LESION_SCOPE=recur on EVERY arm): only the w_k->w_k slow-NMDA synapses of the organ's pools are zeroed
# on the SHARED slice; the shared buffer, read_isolation, and the xedge focus are identical across arms.
# Pre-registration: research/findings/2026-09-24-wm-binding-ordinary-content-probe-PREREGISTRATION.md.
# AMENDMENT C (2026-09-24, filed before any LB_WMB_CONTENT_PROBE seed result was pulled or read): AMENDMENT B
# rescoped the PROSE a T=true/"regressed" GO here must carry (it shows the organ's own w_k->w_k recurrence shapes
# comprehension's read of a host-reinjected, host-timed, host-targeted drive into a POSITIONAL focus pool -- not
# that the organ's HELD REFERENT CONTENT reaches an ordinary reply) but left the COUNTING unchanged: the result
# still landed under the "wm-binding-advanced" key, so a GO here would silently over-credit that faculty's row in
# `load_bearing_fraction`'s numerator/`load_bearing_faculties` list with a claim the prose explicitly disclaims.
# This amendment fixes the counting to match the prose: `measure_wmb_content` reports under the DISTINCT faculty
# key `_WMC_FACULTY_KEY` ("wm-binding-recurrence-drive"), never "wm-binding-advanced", so a GO/pass here can still
# enter the harness's aggregate fraction (FACULTY_LESIONS's existing kind="neural-lesion" wiring is unchanged --
# this is a reporting-label fix, not a re-scoring) but is attributed to its own row, not conflated with the
# hold-query-superseding "wm-binding-advanced" claim. Tested by tests/test_wmb_content_probe_faculty_key_amendment_c.py.
LB_WMB_CONTENT = os.environ.get("LB_WMB_CONTENT_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_WMC_FACULTY_KEY = "wm-binding-recurrence-drive"    # AMENDMENT C: the counted key for measure_wmb_content's result
_WMC_TURNS = {"A": ("wmc_intro", "wmc_drive", ("fox", "wolf")),
              "B": ("wmcx_intro", "wmcx_drive", ("cat", "dog"))}
_WMC_BASE_ENV = {"BRAIN_MULTIREF_LESION_SCOPE": "recur"}   # on EVERY arm (a no-op without BRAIN_MULTIREF_LESION)
_WMC_MECH_FIELDS = ["multiref.kind", "multiref.n_referents", "multiref.recovered", "multiref.hold_alive_min",
                    "multiref.lesion_scope", "comprehension.margin", "comprehension.comprehended",
                    "comprehension.xedge_live_learn.focus", "comprehension.repair", "inner_state_readout", "abstained"]


def _wmb_probe_flags():
    # NB (2026-09-24): LB_WMB_FOCUS_PROBE is recorded on its own record (`measure_wmb_focus`), not here, so this
    # dict -- written into the content/hold-query records -- stays byte-identical to before that probe existed.
    return {"LB_WMB_HOLDQUERY_PROBE": bool(LB_WMB_HOLDQUERY), "LB_WMB_CONTENT_PROBE": bool(LB_WMB_CONTENT)}


def _wmc_mech(arm, turn):
    from research.runners.onebrain_regression_battery import _get_path
    r = (arm or {}).get(turn) or {}
    return {f: _get_path(r, f)[1] for f in _WMC_MECH_FIELDS} | {"answer": r.get("answer")}


def _wmc_gate(arms):
    """The pre-registered gate for LB_WMB_CONTENT_PROBE (pure; no brain build). `arms` = {"A"|"B": {"intact_a",
    "intact_b", "lesion", "lesion_rep": arm-dict-or-None}}. Returns (load_bearing True/False/None, verdict, report).

    Conditions, per content c in (A, B), evaluated in this order (the first failure names the UNDEFINED verdict):
      build   every arm built, no per-turn `_error`                                     -> "arm-build-failed"
      R1      organ IN SCOPE on the intro on EVERY arm: multiref.kind=='maintain', n_referents==2; on the lesion
              arms also multiref.lesion_scope=='recur' (the confined lesion reached the organ) -> "probe-inadequate:route"
      R2      the drive reply is ORDINARY on EVERY arm: no inner_state_readout, no multiref hold-query
                                                                                   -> "probe-inadequate:not-ordinary"
      L       the confined lesion held: lesion intro multiref.hold_alive_min == 0.0 on both lesion arms
                                                                                   -> "lesion-not-effective"
      N       intact_a == intact_b on `answer`                                     -> "noisy-null-control"
      C       the intact reply follows the input: names >=1 of c's own referents and none of the other content's
                                                                                   -> "probe-inadequate:content"
      R       lesion == lesion_rep on `answer`                                     -> "noisy"
    Then T_c = intact_a `answer` != lesion `answer`. Both -> load_bearing True ("regressed"); neither -> False
    ("pass", a real negative); exactly one -> None ("content-dependent-effect", reported, not counted)."""
    rep = {"A": {}, "B": {}}
    for c, (intro, drive, refs) in _WMC_TURNS.items():
        other = _WMC_TURNS["B" if c == "A" else "A"][2]
        a = arms.get(c) or {}
        r = rep[c]
        if any(a.get(k) is None for k in ("intact_a", "intact_b", "lesion", "lesion_rep")) or any(
                isinstance(t, dict) and t.get("_error") for k in ("intact_a", "intact_b", "lesion", "lesion_rep")
                for t in (a.get(k) or {}).values()):
            return None, "arm-build-failed", rep

        def mr(arm, turn):
            m = ((arm or {}).get(turn) or {}).get("multiref")
            return m if isinstance(m, dict) else {}
        r1 = {}
        for k in ("intact_a", "intact_b", "lesion", "lesion_rep"):
            m = mr(a[k], intro)
            ok = m.get("kind") == "maintain" and m.get("n_referents") == 2
            if k.startswith("lesion"):
                ok = ok and m.get("lesion_scope") == "recur"
            r1[k] = bool(ok)
        r["R1_intro_in_scope_every_arm"] = r1
        r2 = {}
        for k in ("intact_a", "intact_b", "lesion", "lesion_rep"):
            d = a[k].get(drive) or {}
            r2[k] = bool(not d.get("inner_state_readout") and mr(a[k], drive).get("kind") != "query"
                         and "answer" in d)
        r["R2_drive_ordinary_every_arm"] = r2
        r["L_lesion_hold_dead"] = bool(all(mr(a[k], intro).get("hold_alive_min") == 0.0
                                           for k in ("lesion", "lesion_rep")))
        ia, ib = a["intact_a"].get(drive) or {}, a["intact_b"].get(drive) or {}
        la, lb = a["lesion"].get(drive) or {}, a["lesion_rep"].get(drive) or {}
        r["N_null_clean"] = bool(ia.get("answer") == ib.get("answer"))
        w = _wmb_words(ia.get("answer"))
        r["C_follows_input"] = bool(any(x in w for x in refs) and not any(x in w for x in other))
        r["R_lesion_reproduced"] = bool(la.get("answer") == lb.get("answer"))
        r["T_reply_changed"] = bool(ia.get("answer") != la.get("answer"))
    for c in ("A", "B"):
        if not all(rep[c]["R1_intro_in_scope_every_arm"].values()):
            return None, "probe-inadequate:route", rep
    for c in ("A", "B"):
        if not all(rep[c]["R2_drive_ordinary_every_arm"].values()):
            return None, "probe-inadequate:not-ordinary", rep
    for cond, verdict in (("L_lesion_hold_dead", "lesion-not-effective"), ("N_null_clean", "noisy-null-control"),
                          ("C_follows_input", "probe-inadequate:content"), ("R_lesion_reproduced", "noisy")):
        if not all(rep[c][cond] for c in ("A", "B")):
            return None, verdict, rep
    ta, tb = rep["A"]["T_reply_changed"], rep["B"]["T_reply_changed"]
    if ta and tb:
        return True, "regressed", rep
    if not ta and not tb:
        return False, "pass", rep
    return None, "content-dependent-effect", rep


def measure_wmb_content(out_dir, seed=42, repeats=2):
    """LB_WMB_CONTENT_PROBE: build intact a/b + confined-lesion + lesion-rebuild arms for BOTH content sessions, then
    `_wmc_gate`. Early-return path (never touches any other faculty's arms).

    AMENDMENT C: reports under `_WMC_FACULTY_KEY` ("wm-binding-recurrence-drive"), NOT "wm-binding-advanced" --
    `spec` is still looked up under the "wm-binding-advanced" FACULTY_LESIONS entry (same lesion flag/value/kind,
    so this still counts as a coverable neural-lesion result in the aggregate fraction), but the reported identity
    is distinct so a GO/pass here never enters "wm-binding-advanced"'s own row in `load_bearing_faculties` /
    `not_load_bearing_faculties` -- see the AMENDMENT C note above LB_WMB_CONTENT for why."""
    spec = FACULTY_LESIONS["wm-binding-advanced"]
    flag, val = spec["flag"], spec["value"]
    _sfx = _seed_suffix(seed)
    res = {"faculty": _WMC_FACULTY_KEY, "turn": _WMC_TURNS["A"][1], "kind": spec["kind"], "flag": flag,
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": _flag_resolves(flag),
           "lesion_env": dict(_WMC_BASE_ENV, **{flag: val}), "wmb_probe_flags": _wmb_probe_flags(),
           "counted_faculty_key": _WMC_FACULTY_KEY, "source_faculty_lesion_key": "wm-binding-advanced",
           "note": ("LB_WMB_CONTENT_PROBE: ordinary transitive after a 2-referent intro, fox/wolf (wmc) + content-"
                    "swapped cat/dog (wmcx); EDGE-CONFINED lesion (BRAIN_MULTIREF_LESION_SCOPE=recur on every arm). "
                    "AMENDMENT C: counted as '%s', NOT 'wm-binding-advanced' -- a GO/regressed here shows the "
                    "organ's own recurrence shapes comprehension's read of a host-reinjected drive into a "
                    "positional focus pool, not that the organ's held referent CONTENT reaches an ordinary reply."
                    % _WMC_FACULTY_KEY)}
    if not res["flag_resolves"]:
        res["verdict"] = "lesion-knob-missing"
        return res
    arms = {}
    for c, (intro, drive, _refs) in _WMC_TURNS.items():
        grp = turn_group(drive)
        fn = "_".join(grp)
        les_env = dict(_WMC_BASE_ENV, **{flag: val})
        arms[c] = {
            "intact_a": _spawn_arm(dict(_WMC_BASE_ENV), grp, os.path.join(out_dir, "intact_a_%s%s.json" % (fn, _sfx))),
            "intact_b": _spawn_arm(dict(_WMC_BASE_ENV), grp, os.path.join(out_dir, "intact_b_%s%s.json" % (fn, _sfx))),
            "lesion": _spawn_arm(les_env, grp, os.path.join(out_dir, "lesion_recur_%s%s.json" % (fn, _sfx))),
            "lesion_rep": _spawn_arm(les_env, grp, os.path.join(out_dir, "lesion_recur_%s%s.json.rep0" % (fn, _sfx))),
        }
    lb, verdict, gate = _wmc_gate(arms)
    res["load_bearing"], res["verdict"], res["wmc_gate"] = lb, verdict, gate
    res["wmc_mechanism"] = {c: {k: {"intro": _wmc_mech(arms[c][k], _WMC_TURNS[c][0]),
                                    "drive": _wmc_mech(arms[c][k], _WMC_TURNS[c][1])} for k in arms[c]}
                            for c in arms}
    if verdict in ("regressed", "pass"):
        diffs = []
        for c in ("A", "B"):
            ia = (arms[c]["intact_a"] or {}).get(_WMC_TURNS[c][1]) or {}
            la = (arms[c]["lesion"] or {}).get(_WMC_TURNS[c][1]) or {}
            if ia.get("answer") != la.get("answer"):
                diffs.append({"field": "answer", "turn": _WMC_TURNS[c][1], "on": ia.get("answer"),
                              "off": la.get("answer")})
        res["diffs"] = diffs
        res["treatment_diffs"] = len(diffs)
        res["control_diffs"] = 0
        res["null_control_clean"] = True
        res["lesion_reproduced"] = True
        res["change_kind"] = _classify_diffs(diffs) if diffs else "none"
    os.makedirs(out_dir, exist_ok=True)
    return res


# ── WM REFERENT->FOCUS BIND ANAPHOR PROBE (LB_WMB_FOCUS_PROBE, default OFF -> byte-identical) ─────────────────────
# WHY (finding 2026-09-24-wm-binding-ordinary-content-probe-6seed-NOGO-held-state-does-not-reach-an-ordinary-reply):
# 6/6 clean negatives -- the ordinary reply read the WM focus through the POSITIONAL CAND_POOLS[0], never WHICH
# referent the organ holds. Its named next mechanism is BRAIN_MULTIREF_FOCUS_BIND (d6_multiref_wm_production_organ:
# cross-turn held state + a cue-driven focus-WTA retrieval that resolves an anaphor to the winning register's
# referent). This probe asks whether the ORDINARY reply follows the HELD referent. Each pair = two sessions that
# introduce the SAME two referents in SWAPPED order (which referent sits in each register differs; the words do not),
# then the SAME anaphor question. T: the two sessions' replies differ. X: under the hold lesion (BRAIN_MULTIREF_LESION
# =1 with SCOPE=recur) they are the same -- the lesion must REMOVE the difference. N: an intact rebuild reproduces.
# Every arm runs all four sessions in one build. ON arms: BRAIN_MULTIREF_FOCUS_BIND=1. OFF arms (the positional route,
# the FAILING DIRECTION): BRAIN_MULTIREF_FOCUS_BIND=0 explicitly -- the same gate must read NOT load-bearing there, or
# the ON verdict is void (probe-inadequate:positional-passes). Counted under its own key with kind
# "neural-lesion-opt-in": it measures a default-OFF mechanism, so it is EXCLUDED from the production fraction.
# Pre-registration: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md.
LB_WMB_FOCUS = os.environ.get("LB_WMB_FOCUS_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_WMF_FACULTY_KEY = "wm-binding-referent-focus"
_WMF_PAIRS = {"A": (("wmf_a1_intro", "wmf_a1_ask"), ("wmf_a2_intro", "wmf_a2_ask")),
              "B": (("wmf_b1_intro", "wmf_b1_ask"), ("wmf_b2_intro", "wmf_b2_ask"))}
_WMF_TURNS = [lab for p in ("A", "B") for s in _WMF_PAIRS[p] for lab in s]
_WMF_ENV = {"on": {"BRAIN_MULTIREF_FOCUS_BIND": "1", "BRAIN_MULTIREF_LESION_SCOPE": "recur"},
            "off": {"BRAIN_MULTIREF_FOCUS_BIND": "0", "BRAIN_MULTIREF_LESION_SCOPE": "recur"}}
_WMF_ARMS = ("intact_a", "intact_b", "lesion", "lesion_rep")
_WMF_MECH_FIELDS = ["multiref.kind", "multiref.n_referents", "multiref.recovered", "multiref.hold_alive_min",
                    "multiref.lesion_scope", "multiref.resolved", "multiref.resolved_register",
                    "multiref.resolved_pool", "multiref.margin", "multiref.register_rates", "multiref.wta_rates",
                    "recalled_svo", "abstained", "inner_state_readout"]


def _wmf_mech(arm, turn):
    r = (arm or {}).get(turn) or {}
    return {f: _get_path(r, f)[1] for f in _WMF_MECH_FIELDS} | {"answer": r.get("answer")}


def _wmf_gate(arms, require_resolution=True):
    """The pre-registered gate for LB_WMB_FOCUS_PROBE (pure; no brain build). `arms` = {"intact_a","intact_b",
    "lesion","lesion_rep": {turn_label: response}}. Returns (load_bearing True/False/None, verdict, report).

    Evaluated per pair p in (A, B), conditions in this order (first failure names the UNDEFINED verdict):
      build  every arm built, no per-turn `_error`                                  -> "arm-build-failed"
      R1     both intros in scope on EVERY arm (multiref.kind=='maintain', n_referents==2); on the lesion arms also
             multiref.lesion_scope=='recur'                                          -> "probe-inadequate:route"
      R2     both asks ORDINARY on every arm (no inner_state_readout, no multiref hold-query, an answer present)
                                                                                      -> "probe-inadequate:not-ordinary"
      RES    (ON only) on both intact arms each ask was resolved by the organ (multiref.kind=='resolve') to one of
             that session's own intro referents                                     -> "probe-inadequate:no-resolution"
             and the two sessions of the pair resolved DIFFERENT referents          -> "probe-inadequate:same-referent"
      L      both lesion arms' intros read hold_alive_min == 0.0, and (ON only) the lesion still holds at the ask:
             its retrieval record exists and every register rate is 0.0            -> "lesion-not-effective"
      N      intact_a == intact_b on every ask `answer`                              -> "noisy-null-control"
      C      (ON only) each intact reply follows its resolution: recalled_svo is None (abstain) or its agent is the
             resolved referent                                                       -> "probe-inadequate:content"
      R      lesion == lesion_rep on every ask `answer`                              -> "noisy"
    Then T_p = the pair's two intact replies differ; X_p = the pair's two LESION replies are the same.
      every pair T and X     -> (True,  "regressed")        the ordinary reply follows the held referent, lesion removes it
      no pair T              -> (False, "pass")             a real negative: the reply does not follow the held referent
      a pair T but not X     -> (None,  "off-organ-route")  the difference survives the hold lesion (not the organ's)
      otherwise              -> (None,  "content-dependent-effect")
    `require_resolution=False` is the OFF (positional-route) evaluation: RES and C are the ON route's adequacy
    conditions and are skipped; every other condition and the verdict table are identical."""
    from research.runners.d6_multiref_wm_production_organ import extract_referents
    rep = {"A": {}, "B": {}}
    if any(arms.get(k) is None for k in _WMF_ARMS) or any(
            isinstance(t, dict) and t.get("_error") for k in _WMF_ARMS for t in (arms.get(k) or {}).values()):
        return None, "arm-build-failed", rep

    def mr(arm, turn):
        m = ((arm or {}).get(turn) or {}).get("multiref")
        return m if isinstance(m, dict) else {}

    def ans(arm, turn):
        return ((arm or {}).get(turn) or {}).get("answer")
    for p, sessions in _WMF_PAIRS.items():
        r = rep[p]
        r1 = {}
        for k in _WMF_ARMS:
            ok = True
            for intro, _ask in sessions:
                m = mr(arms[k], intro)
                ok = ok and m.get("kind") == "maintain" and m.get("n_referents") == 2
                if k.startswith("lesion"):
                    ok = ok and m.get("lesion_scope") == "recur"
            r1[k] = bool(ok)
        r["R1_intros_in_scope_every_arm"] = r1
        r2 = {}
        for k in _WMF_ARMS:
            ok = True
            for _intro, ask in sessions:
                d = arms[k].get(ask) or {}
                ok = ok and not d.get("inner_state_readout") and mr(arms[k], ask).get("kind") != "query" \
                    and "answer" in d
            r2[k] = bool(ok)
        r["R2_asks_ordinary_every_arm"] = r2
        if require_resolution:
            ok = True
            for k in ("intact_a", "intact_b"):
                for intro, ask in sessions:
                    m = mr(arms[k], ask)
                    held = set(extract_referents(_TURN_BY_LABEL[intro][1]))
                    ok = ok and m.get("kind") == "resolve" and m.get("resolved") in held
            r["RES_intact_resolved_to_held"] = bool(ok)
            resolved = [mr(arms["intact_a"], ask).get("resolved") for _intro, ask in sessions]
            r["RES_resolved"] = resolved
            r["RES_pair_differs"] = bool(len(set(resolved)) == 2 and None not in resolved)
        r["L_lesion_hold_dead"] = bool(all(mr(arms[k], intro).get("hold_alive_min") == 0.0
                                           for k in ("lesion", "lesion_rep") for intro, _ask in sessions))
        if require_resolution:
            # the lesion must STILL HOLD at the moment of measurement (docs/TERMS.md `lesion`): on the ON lesion arms
            # the ask turn's own retrieval read must find no live register
            r["L_lesion_holds_at_ask"] = bool(all(
                mr(arms[k], ask).get("kind") == "resolve"
                and max(mr(arms[k], ask).get("register_rates") or [1.0]) == 0.0
                for k in ("lesion", "lesion_rep") for _intro, ask in sessions))
            r["L_lesion_hold_dead"] = bool(r["L_lesion_hold_dead"] and r["L_lesion_holds_at_ask"])
        r["N_null_clean"] = bool(all(ans(arms["intact_a"], ask) == ans(arms["intact_b"], ask)
                                     for _intro, ask in sessions))
        if require_resolution:
            c = True
            for _intro, ask in sessions:
                sv = (arms["intact_a"].get(ask) or {}).get("recalled_svo")
                res_ = mr(arms["intact_a"], ask).get("resolved")
                c = c and (sv is None or (isinstance(sv, list) and len(sv) > 0 and sv[0] == res_))
            r["C_reply_follows_resolution"] = bool(c)
        r["R_lesion_reproduced"] = bool(all(ans(arms["lesion"], ask) == ans(arms["lesion_rep"], ask)
                                            for _intro, ask in sessions))
        (_i1, a1), (_i2, a2) = sessions
        r["T_intact_pair_differs"] = bool(ans(arms["intact_a"], a1) != ans(arms["intact_a"], a2))
        r["X_lesion_pair_same"] = bool(ans(arms["lesion"], a1) == ans(arms["lesion"], a2))
    for p in ("A", "B"):
        if not all(rep[p]["R1_intros_in_scope_every_arm"].values()):
            return None, "probe-inadequate:route", rep
    for p in ("A", "B"):
        if not all(rep[p]["R2_asks_ordinary_every_arm"].values()):
            return None, "probe-inadequate:not-ordinary", rep
    order = []
    if require_resolution:
        order += [("RES_intact_resolved_to_held", "probe-inadequate:no-resolution"),
                  ("RES_pair_differs", "probe-inadequate:same-referent")]
    order += [("L_lesion_hold_dead", "lesion-not-effective"), ("N_null_clean", "noisy-null-control")]
    if require_resolution:
        order += [("C_reply_follows_resolution", "probe-inadequate:content")]
    order += [("R_lesion_reproduced", "noisy")]
    for cond, verdict in order:
        if not all(rep[p][cond] for p in ("A", "B")):
            return None, verdict, rep
    per = {}
    for p in ("A", "B"):
        t, x = rep[p]["T_intact_pair_differs"], rep[p]["X_lesion_pair_same"]
        per[p] = True if (t and x) else (False if not t else None)
    if all(v is True for v in per.values()):
        return True, "regressed", rep
    if all(v is False for v in per.values()):
        return False, "pass", rep
    if any(v is None for v in per.values()):
        return None, "off-organ-route", rep
    return None, "content-dependent-effect", rep


def _wmf_headline(on_result, off_result):
    """Combine the ON verdict with the OFF (positional-route) failing-direction check. The OFF route must NOT read
    load-bearing; if it does, the probe cannot tell the organ's content from position and the ON verdict is void."""
    lb_on, v_on = on_result[0], on_result[1]
    lb_off = off_result[0]
    if lb_off is True:
        return None, "probe-inadequate:positional-passes"
    return lb_on, v_on


def measure_wmb_focus(out_dir, seed=42, repeats=2):
    """LB_WMB_FOCUS_PROBE: 8 builds -- ON {intact a/b, lesion, lesion rebuild} and OFF {the same four} -- each running
    all four anaphor sessions; `_wmf_gate` on each; `_wmf_headline` combines them. Early-return path (never touches
    any other faculty's arms)."""
    spec = FACULTY_LESIONS["wm-binding-advanced"]
    flag, val = spec["flag"], spec["value"]
    _sfx = _seed_suffix(seed)
    res = {"faculty": _WMF_FACULTY_KEY, "turn": "wmf_a1_ask", "kind": "neural-lesion-opt-in", "flag": flag,
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": _flag_resolves(flag),
           "mechanism_flag": "BRAIN_MULTIREF_FOCUS_BIND", "mechanism_flag_resolves":
               _flag_resolves("BRAIN_MULTIREF_FOCUS_BIND"),
           "env": {m: dict(e) for m, e in _WMF_ENV.items()}, "lesion_env": {flag: val},
           "counted_faculty_key": _WMF_FACULTY_KEY, "source_faculty_lesion_key": "wm-binding-advanced",
           "note": ("LB_WMB_FOCUS_PROBE: two order-swapped sessions per pair (dog/cat -> 'what does it chase'; "
                    "cat/bird -> 'what does it eat'); the ordinary reply must follow the HELD referent (T) and the "
                    "confined hold lesion must remove the difference (X). OFF arms = the positional route, must read "
                    "not load-bearing. kind neural-lesion-opt-in: a default-OFF mechanism, excluded from the "
                    "production fraction.")}
    if not (res["flag_resolves"] and res["mechanism_flag_resolves"]):
        res["verdict"] = "lesion-knob-missing"
        return res
    arms = {}
    for mode, base in _WMF_ENV.items():
        les = dict(base, **{flag: val})
        fn = lambda k: os.path.join(out_dir, "%s_%s_wmf%s.json" % (mode, k, _sfx))   # noqa: E731
        arms[mode] = {"intact_a": _spawn_arm(dict(base), _WMF_TURNS, fn("intact_a")),
                      "intact_b": _spawn_arm(dict(base), _WMF_TURNS, fn("intact_b")),
                      "lesion": _spawn_arm(les, _WMF_TURNS, fn("lesion")),
                      "lesion_rep": _spawn_arm(les, _WMF_TURNS, fn("lesion_rep"))}
    on = _wmf_gate(arms["on"], require_resolution=True)
    off = _wmf_gate(arms["off"], require_resolution=False)
    lb, verdict = _wmf_headline(on, off)
    res["load_bearing"], res["verdict"] = lb, verdict
    res["wmf_gate_on"] = {"load_bearing": on[0], "verdict": on[1], "report": on[2]}
    res["wmf_gate_off"] = {"load_bearing": off[0], "verdict": off[1], "report": off[2]}
    res["failing_direction_ok"] = bool(off[0] is not True)
    res["wmf_mechanism"] = {mode: {k: {t: _wmf_mech(arms[mode][k], t) for t in _WMF_TURNS} for k in _WMF_ARMS}
                            for mode in arms}
    if on[1] in ("regressed", "pass"):
        diffs = []
        for p, ((_i1, a1), (_i2, a2)) in _WMF_PAIRS.items():
            ia = arms["on"]["intact_a"]
            if (ia.get(a1) or {}).get("answer") != (ia.get(a2) or {}).get("answer"):
                diffs.append({"pair": p, "field": "answer", "session_1": (ia.get(a1) or {}).get("answer"),
                              "session_2": (ia.get(a2) or {}).get("answer")})
        res["diffs"] = diffs
        res["treatment_diffs"] = len(diffs)
        res["control_diffs"] = 0
        res["null_control_clean"] = True
        res["lesion_reproduced"] = True
    os.makedirs(out_dir, exist_ok=True)
    return res


# ── DA TAG-AND-CAPTURE NEXT-DAY PROBE (opt-in, env-gated; default OFF -> byte-identical) ─────────────────────────────
# WHY (finding 2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level): DA-gated encoding acts
# on PERSISTENCE (Bethus, Tse & Morris 2010), so its default `well` probe (one fresh turn, field `da_encoding.on`, True
# in both arms) can never show a lesion. The v3 synaptic tag-and-capture ledger is now wired into chat behind
# BRAIN_DA_TAG_CAPTURE (webapp/da_tag_capture_chat.py). This flag remaps da-gated-encoding to the SALIENT next-day
# group (onebrain_regression_battery 'datc': surprising news around one plain fact -> a night through the brain's own
# idle/sleep tick -> 'what does the cat chase') and compares the recall turn's `recalled_svo` / `abstained`.
# base_env arms the companion on BOTH arms (and the scripted 30 s/turn world clock), so the only inter-arm difference
# is the existing BRAIN_DA_ENCODING_LESION. This row alone is the battery's intact-vs-lesion + null call; the salient
# vs neutral, spare-immediate-recall and companion-off contrasts are the pre-registered gates of
# research/runners/_da_tag_capture_chat_probe.py. OFF (default) -> da-gated-encoding is measured on `well` as before.
LB_DA_TAG_CAPTURE = os.environ.get("LB_DA_TAG_CAPTURE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_DA_TAG_CAPTURE_TURN = "datc_recall"
_DA_TAG_CAPTURE_FIELDS = ["recalled_svo", "abstained"]
_DA_TAG_CAPTURE_ENV = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}


def _oed_build_shared_world(seed):
    """Build the _followon2 shared world ONCE at `seed` (taxonomy vocab + the real TinyStories co-occurrence corpus
    + the PPMI plausibility graph P/row/tau + the RF composer store of AFFIRM/NEGATE facts) -- lazy import so this
    module carries no import-time dependency on the followon2 de-risk unless LB_OPEN_ENDED_DISTRIB_PROBE is actually
    exercised (byte-identical import graph otherwise). Mirrors _followon2_spiking_wta_sampler_derisk.main()'s own
    world-build call (the SAME `a` defaults: D=64, n_facts=24, n_negated=12, tau_pct=50.0), minus the argparse
    layer. Returns (build_world(...)'s 7-tuple, the `a` namespace)."""
    import argparse as _ap
    from research.runners._followon2_spiking_wta_sampler_derisk import build_world
    from research.runners.option_c_real_cooccurrence_derisk import (
        TAXONOMY_8x8, taxonomy_to_vocab_categories, build_real_cooccurrence)
    a = _ap.Namespace(D=64, n_facts=24, n_negated=12, tau_pct=50.0)
    vocab, cat_ids, _cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    corpus_path = os.path.join(proj, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            "%s missing -- symlink data/corpus/*.txt into this worktree from the main checkout's data/corpus/ "
            "(CLAUDE.md discipline: without the real corpus this organ degrades to standalone, a false negative)."
            % corpus_path)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=5, repeat_cap=40, seed=42,
                                     max_bytes=4_000_000, freq_floor=30, min_facts_per_category=20, verbose=False)
    return build_world(seed, vocab, corpus, a), a


def _oed_plausible_fraction(P, row, tau, proposer, all_stored, seed, ablate, n_attempts):
    """ONE arm of the distributional measurement: build a FRESH SpikingWTASampler (a REBUILD -- a brand-new
    Izhikevich WTA bank, reseeded from `seed`) and run `n_attempts` spiking generative draws, gated through the
    brain's UNCHANGED plausibility/non-contradiction gates (`_gate_and_collect`, verbatim from _followon2). Returns
    (plausible_fraction_of_novel, n_accepted, n_novel_attempts) -- the SAME distributional ruler the 6-seed GO used."""
    from research.runners._followon2_spiking_wta_sampler_derisk import SpikingWTASampler, _gate_and_collect
    sampler = SpikingWTASampler(P, row, tau, seed=seed, ablate_likelihood=ablate)
    raw = sampler.draw(n_attempts)
    rep = _gate_and_collect(raw, proposer, all_stored)
    return rep["plausible_fraction_of_novel"], len(rep["accepted"]), rep["n_novel_attempts"]


def _oed_score(intact_a_frac, intact_b_frac, lesion_frac, min_effect=_OED_MIN_EFFECT):
    """PURE decision logic (no brain build) for the distributional probe, factored out so the selftest can exercise
    the exact decision procedure on synthetic numbers -- mirrors _classify_diffs()/compare() being pure functions
    the selftest already calls directly. TREATMENT = |intact_a - lesion|; CONTROL (the null) = |intact_a - intact_b|
    (an independent rebuild at the identical seed/params). Returns (control_diff, treatment_diff, null_clean,
    load_bearing, verdict). load_bearing is None (never False) when the null is not clean -- an unclean null makes
    the treatment reading untrustworthy, not evidence of absence (the same "noisy-null-control" semantics every
    other faculty's null check already uses)."""
    control_diff = abs(intact_a_frac - intact_b_frac)
    treatment_diff = abs(intact_a_frac - lesion_frac)
    null_clean = control_diff <= 1e-9
    if not null_clean:
        return control_diff, treatment_diff, null_clean, None, "noisy-null-control"
    if treatment_diff <= max(1e-9, min_effect):
        return control_diff, treatment_diff, null_clean, False, "pass"
    return control_diff, treatment_diff, null_clean, True, "regressed"


def measure_open_ended_distributional(out_dir, seed=42, repeats=1, n_attempts=None):
    """The DISTRIBUTIONAL load-bearing measurement for 'open-ended-generation' (LB_OPEN_ENDED_DISTRIB_PROBE).
    Reuses the _followon2 machinery UNCHANGED instead of the webapp brain_chat single-turn path; NEVER calls
    _spawn_arm, so no other faculty's code path is touched and this never builds the tiny-demo brain. Returns a
    result dict shaped like measure_faculty()'s row (the same top-level keys) so run()'s counting/denominator logic
    needs no special-casing -- only the MEANING of diffs/treatment_diffs/control_diffs changes: a distributional
    fraction, not a categorical field-diff count."""
    n_attempts = _OED_N_ATTEMPTS if n_attempts is None else int(n_attempts)
    res = {"faculty": "open-ended-generation", "turn": "oe_distributional (_followon2 world, no webapp turn)",
           "kind": "neural-lesion", "flag": "ablate_likelihood(SpikingWTASampler)",
           # DEVICE STAMP (device-and-cost gate discipline, matches run()'s own report): this side-artifact is
           # written from inside a measurement, not as a runner's direct --out target, so it carries its own
           # backend/device record rather than relying on a provenance sidecar that would not exist for it.
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": None,
           "measurement_ruler": "distributional",
           "note": "LB_OPEN_ENDED_DISTRIB_PROBE: scored by the _followon2 draw-many plausible-fraction-of-novel "
                   "lesion (6-seed GO), NOT the single-turn field-diff -- the single-turn instrument is the wrong "
                   "ruler for this faculty (finding 2026-09-21-open-ended-generation-single-turn-not-load-bearing)."}
    try:
        (comp, affirmed, negated, P, row, tau, _universe), a = _oed_build_shared_world(seed)
        from research.runners._genfrontier_b2_generative_replay_derisk import GenerativeReplayProposer
        import numpy as np
        all_stored = set(affirmed) | set(negated)
        proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                            np.random.default_rng(seed * 7 + 1), use_spiking_sampler=False)
        res["flag_resolves"] = True   # the class + its ablate_likelihood knob imported + built successfully
    except Exception as e:
        res["verdict"] = "arm-build-failed"
        res["flag_resolves"] = False
        res["note"] += "  world-build failed: %r" % (e,)
        return res

    intact_a_frac, intact_a_n, intact_a_novel = _oed_plausible_fraction(
        P, row, tau, proposer, all_stored, seed, False, n_attempts)
    intact_b_frac, intact_b_n, intact_b_novel = _oed_plausible_fraction(   # the NULL: an independent REBUILD
        P, row, tau, proposer, all_stored, seed, False, n_attempts)
    lesion_frac, lesion_n, lesion_novel = _oed_plausible_fraction(
        P, row, tau, proposer, all_stored, seed, True, n_attempts)

    control_diff, treatment_diff, null_clean, load_bearing, verdict = _oed_score(
        intact_a_frac, intact_b_frac, lesion_frac)
    res["control_diffs"] = control_diff
    res["treatment_diffs"] = treatment_diff
    res["null_control_clean"] = null_clean
    res["distributional"] = {
        "intact_a_plausible_fraction": intact_a_frac, "intact_a_n_accepted": intact_a_n,
        "intact_a_n_novel": intact_a_novel,
        "intact_b_plausible_fraction": intact_b_frac, "intact_b_n_accepted": intact_b_n,
        "intact_b_n_novel": intact_b_novel,
        "lesion_plausible_fraction": lesion_frac, "lesion_n_accepted": lesion_n, "lesion_n_novel": lesion_novel,
        "n_attempts": n_attempts, "seed": seed,
    }
    res["diffs"] = [{"field": "open_ended.plausible_fraction_of_novel", "on": intact_a_frac, "off": lesion_frac}]
    res["change_kind"] = "distributional" if treatment_diff > _OED_MIN_EFFECT else "none"
    res["attributable_fraction"] = attributable_to(
        "load-bearing[open-ended-generation:distributional]", treatment_diff, control_diff)

    # LESION-REPEAT (the same anti-noise discipline as the categorical arms): rebuild the lesion arm and require
    # the SAME sign of effect (a repeat that reads NO change while the first read a real one is non-reproducing).
    reproduced = True
    for i in range(max(0, repeats - 1)):
        les2_frac, _, _ = _oed_plausible_fraction(P, row, tau, proposer, all_stored, seed, True, n_attempts)
        if load_bearing and abs(intact_a_frac - les2_frac) <= _OED_MIN_EFFECT:
            reproduced = False
            break
    res["lesion_reproduced"] = reproduced
    if null_clean and load_bearing and not reproduced:
        res["load_bearing"], res["verdict"] = None, "noisy"
    else:
        res["load_bearing"], res["verdict"] = load_bearing, verdict

    _sfx = _seed_suffix(seed)
    try:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(res, open(os.path.join(out_dir, "oed_distributional%s.json" % _sfx), "w"), indent=2, default=str)
    except Exception:
        pass   # provenance convenience only -- never fail the measurement over a write error
    return res


def _affect_tone_open_row(seed_entry, delta):
    """PURE translation (no brain build; selftest-exercisable) of ONE seed's open-output tone gaps into
    (load_bearing, verdict, treatment_diff, control_diff, null_clean). TREATMENT = the intact pos-minus-neg tone
    spread (the mood-driven tone effect on the free reply); CONTROL (the null) = the attribution control's own
    directional gap (a mood-DECOUPLED valence must not reproduce it). load_bearing requires BOTH mood directions
    correct-sign AND a clean control (|control| < delta). A wrong-sign direction -> load_bearing False verdict
    'wrong-sign'; a within-band (null) direction with a clean control -> load_bearing False verdict 'pass' (not
    load-bearing on this seed); an unclean control -> load_bearing None verdict 'noisy-null-control' (untrustworthy,
    never evidence of absence -- the same semantics as _oed_score)."""
    ps, ns = seed_entry.get("pos_state"), seed_entry.get("neg_state")
    treatment = abs(float(seed_entry.get("real_directional_gap", 0.0) or 0.0))
    control = abs(float(seed_entry.get("ctrl_directional_gap", 0.0) or 0.0))
    null_clean = control < delta
    if not null_clean:
        return None, "noisy-null-control", treatment, control, null_clean
    if ps == "wrong" or ns == "wrong":
        return False, "wrong-sign", treatment, control, null_clean
    lb = (ps == "correct" and ns == "correct")
    return lb, ("load-bearing" if lb else "pass"), treatment, control, null_clean


def measure_affect_tone_open_output(out_dir, seed=42, repeats=1):
    """The OPEN-OUTPUT DIRECTIONAL load-bearing measurement for 'affect-coloring' (LB_AFFECT_TONE_OPEN_PROBE).
    READS the canonical 6-seed verdict artifact produced by research.runners._lbf_affect_tone_open_output_derisk
    (the heavy fresh-subprocess linattn-mouth probe) and surfaces THIS seed's directional tone load-bearing signal
    as a faculty row shaped like measure_faculty()'s (same top-level keys), so run()'s counting/denominator logic
    needs no special-casing -- only the MEANING of treatment/control changes: a directional tone gap on the FREE
    reply, not a categorical decision-field diff. Reads-not-derives (the artifact is the instrument); NEVER builds
    a brain and NEVER calls _spawn_arm, so the battery's other faculties are untouched."""
    res = {"faculty": "affect-coloring",
           "turn": "affect_tone_open (linattn WKV mouth FREE reply; _lbf_affect_tone_open_output_derisk artifact)",
           "kind": "neural-lesion", "flag": "BRAIN_AFFECT_LESION",
           "backend": os.environ.get("SIM_BACKEND", "numpy"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": None,
           "measurement_ruler": "open-output-directional-independent-lexicon (6-seed)",
           "note": "LB_AFFECT_TONE_OPEN_PROBE: affect-coloring scored by the DIRECTIONAL open-output tone ruler "
                   "(research.runners._lbf_affect_tone_open_output_derisk), NOT the templated decision field -- "
                   "the single-turn field-diff is the wrong ruler for a steering faculty (roadmap SS8)."}
    apath = _AFFECT_TONE_OPEN_ARTIFACT
    if not os.path.exists(apath):
        proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        apath = os.path.join(proj, _AFFECT_TONE_OPEN_ARTIFACT)
    if not os.path.exists(apath):
        res["verdict"] = "artifact-missing"
        res["flag_resolves"] = True   # the flag + function exist; only the (heavy) precomputed artifact is absent
        res["note"] += ("  artifact %r not found -- run `CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy python -m "
                        "research.runners._lbf_affect_tone_open_output_derisk --controller` first."
                        % _AFFECT_TONE_OPEN_ARTIFACT)
        return res
    try:
        art = json.load(open(apath))
    except Exception as e:
        res["verdict"] = "artifact-unreadable"; res["flag_resolves"] = True
        res["note"] += "  artifact read failed: %r" % (e,)
        return res
    res["flag_resolves"] = True
    delta = float(art.get("delta_preregistered", 0.0))
    per_seed = art.get("per_seed", {})
    entry = per_seed.get(str(seed)) or per_seed.get(seed)
    if not entry or not entry.get("complete"):
        res["verdict"] = "seed-missing-in-artifact"
        res["note"] += "  seed %s not complete in the artifact." % seed
        return res
    lb, verdict, treatment, control, null_clean = _affect_tone_open_row(entry, delta)
    res["treatment_diffs"] = treatment
    res["control_diffs"] = control
    res["null_control_clean"] = null_clean
    res["change_kind"] = "directional-tone" if (null_clean and lb) else "none"
    res["attributable_fraction"] = attributable_to(
        "load-bearing[affect-coloring:open-output-tone]", treatment, control)
    res["diffs"] = [{"field": "open_reply.tone_compound(intact_pos vs intact_neg)",
                     "on": entry.get("tone_pos"), "off": entry.get("tone_neg")}]
    res["lesion_reproduced"] = bool(art.get("determinism_ok"))
    res["overall_6seed_go"] = bool(art.get("GO"))
    res["delta_preregistered"] = delta
    res["load_bearing"], res["verdict"] = lb, verdict

    _sfx = _seed_suffix(seed)
    try:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(res, open(os.path.join(out_dir, "affect_tone_open%s.json" % _sfx), "w"), indent=2, default=str)
    except Exception:
        pass   # provenance convenience only -- never fail the measurement over a write error
    return res


def _spawn_arm(env, turn_labels, out_path):
    if _LB_RESUME and os.path.exists(out_path):
        try:
            return json.load(open(out_path))
        except Exception:
            pass  # truncated/corrupt (e.g. killed mid-write) -> rebuild
    return _spawn_arm_raw(env, turn_labels, out_path)

# A sentinel env var NOTHING reads — the null lesion. Setting it changes no brain behavior; used by the self-test to
# confirm the instrument does NOT report a change when the "lesion" is a no-op (guards against a false-positive harness).
NULL_LESION_FLAG = "BRAIN_LOAD_BEARING_NULL_LESION"

# SEED THREADING (research/seed-threading-lbf, 2026-09-20): every arm build already reads the substrate seed off
# `BRAIN_CHAT_SEED` (webapp/server.py._brain_chat_seed + each per-organ workspace's own `_DEFAULT_SEED`, all
# reading the SAME env var) — see main()/run() below, which set it once for the whole invocation. `_seed_suffix`
# namespaces every per-arm output FILENAME by seed so LB_RESUME_SKIP_EXISTING and the in-memory intact_cache never
# false-skip/collide across a multi-seed sweep sharing one --out directory. seed=42 (the pre-existing hardcoded
# value) keeps the ORIGINAL, un-suffixed filenames — BYTE-IDENTICAL to every existing on-disk artifact and to the
# pre-this-change resume behavior; only a non-42 seed adds the `_s<seed>` tag.
def _seed_suffix(seed: int) -> str:
    return "" if int(seed) == 42 else "_s%d" % int(seed)


# ── the PER-FACULTY LESION MAP ───────────────────────────────────────────────────────────────────────────────────
# faculty_key -> {flag, value, kind, note}. `kind` is one of: neural-lesion / whether-disable / in-process / thin /
# mechanism-only / proposed (see the module docstring). `value` is the env value that LESIONS ("1" for a BRAIN_*_LESION
# neural cut; "0" for a BRAIN_* master-switch disable). `flag=None` for kinds this env-flag harness cannot drive.
# Every flag below was confirmed present in source (grep webapp/ research/runners/); _flag_resolves() re-checks at run
# time so a stale/typo'd flag surfaces as `lesion-knob-missing`, never as a silent "not load-bearing".
FACULTY_LESIONS = {
    # ---- core CHOOSE/RECALL/LEARN: in-process lesions, ALREADY behaviorally covered by _production_lesion_probe ----
    "content-selection":        dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by parser.role_of (junk role -> factual questions ABSTAIN) in _production_lesion_probe "
             "CHOOSE; no env knob. Minimal env lesion to add: BRAIN_SUBSTRATE_PARSE_LESION."),
    "semantic-recall":          dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by composer.query_patient (the battery module's own spiking_recall_lesion demo / "
             "_production_lesion_probe); no env knob. Minimal env lesion to add: BRAIN_COMPOSER_RECALL_LESION."),
    "moat-verify":              dict(flag=None, value=None, kind="in-process",
        note="the no-confab moat: lesioned in-process by chat._substrate_recall (lesion -> the abstain flips to a "
             "keyword confab), _production_lesion_probe CHOOSE. No env knob; minimal add: BRAIN_SUBSTRATE_RECALL_LESION."),
    "in-loop-learning":         dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by chat._substrate_recall (a fact taught this turn disappears), _production_lesion_"
             "probe LEARN. No env knob; minimal add: BRAIN_SUBSTRATE_RECALL_LESION."),
    "one-brain-substrate":      dict(flag="BRAIN_COMPOSER_KIND", value="rf", kind="mechanism-only",
        note="onebrain vs rf give the SAME answers BY DESIGN (ledger lesion_note) — a MECHANISM claim (recall on firing "
             "neurons), not an answer change. activity.composer label flips but that is cosmetic; NOT answer-load-bearing "
             "by construction. Excluded from the fraction."),
    # ---- dedicated BRAIN_*_LESION neural-cut knobs (PRIMARY: cut the neural read, organ stays installed) ----
    "comprehension-monitor":               dict(flag="BRAIN_COMPREHENSION_LESION", value="1", kind="neural-lesion", note=""),
    "comprehension-learned-animacy-cue":   dict(flag="BRAIN_LEARNED_ANIMACY_LESION", value="1", kind="neural-lesion",
        note="cuts the learned animacy cue; on a hand-table-OOV noun ('monkey') comprehension.on swings present/absent."),
    "comprehension-learned-verb-selects":  dict(flag="BRAIN_LEARNED_VERB_SELECTS_LESION", value="1", kind="neural-lesion",
        note="cuts the learned verb-selects cue; on a hand-table-OOV verb ('clean') comprehension.on swings."),
    "noncontradiction-gate":    dict(flag="BRAIN_NONCONTRADICTION_LESION", value="1", kind="neural-lesion", note=""),
    "affect-coloring":          dict(flag="BRAIN_AFFECT_LESION", value="1", kind="neural-lesion",
        note="affect_production_organ.py:242 BRAIN_AFFECT_LESION cuts the Gate-B ladder read."),
    "affect-drives-response":   dict(flag="BRAIN_AFFECT_DRIVES_LESION", value="1", kind="neural-lesion",
        note="cuts the interoceptive->ladder synapses -> felt mood collapses -> the affective lead vanishes. NOTE the "
             "'well' turn is mood-neutral (level 0) intact too, so the acted decision may be unchanged there; the change "
             "shows on an affective turn — a lower-confidence row on the neutral probe."),
    "affect-marker-spiking-wta":dict(flag="BRAIN_AFFECT_MARKER_SPIKING_LESION", value="1", kind="neural-lesion",
        note="cuts the felt-state->marker WTA selection; measured on the strongly-affective 'emo' turn (level>0)."),
    "da-mode-drives-response":  dict(flag="BRAIN_DA_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "da-gated-encoding":        dict(flag="BRAIN_DA_ENCODING_LESION", value="1", kind="neural-lesion", note=""),
    "source-provenance-honesty":dict(flag="BRAIN_SOURCE_PROVENANCE_HONESTY_LESION", value="1", kind="neural-lesion",
        note="server.py get_organ(lesion=source_provenance_lesioned())."),
    "common-ground-drives":     dict(flag="BRAIN_CG_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "swap-drives-response":     dict(flag="BRAIN_SWAP_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "wm-binding-advanced":      dict(flag="BRAIN_MULTIREF_LESION", value="1", kind="neural-lesion", note=""),
    "prospective-memory":       dict(flag="BRAIN_PMEM_LESION", value="1", kind="neural-lesion", note=""),
    "pragmatic-implicature":    dict(flag="BRAIN_PRAGMATIC_LESION", value="1", kind="neural-lesion", note=""),
    "surprise-monitor":         dict(flag="BRAIN_SURPRISE_LESION", value="1", kind="neural-lesion", note=""),
    "metacog-monitor":          dict(flag="BRAIN_METACOG_LESION", value="1", kind="neural-lesion", note=""),
    "worldmodel-forward":       dict(flag="BRAIN_WORLDMODEL_LESION", value="1", kind="neural-lesion", note=""),
    "curiosity-followup":       dict(flag="BRAIN_CURIOSITY_LESION", value="1", kind="neural-lesion",
        note="removes the spiking ASK-pool crave drive -> the novel-abstain follow-up is silenced (DR-1 GO's own lesion "
             "arm). curiosity.curious flips True->False on the 'unknown' abstain turn."),
    "reconsolidation":          dict(flag="BRAIN_RECONSOLIDATION_LESION", value="1", kind="neural-lesion", note=""),
    "episodic-memory":          dict(flag="BRAIN_EPISODIC_LESION", value="1", kind="neural-lesion", note=""),
    "discourse-register":       dict(flag="BRAIN_DISCOURSE_REGISTER_LESION", value="1", kind="neural-lesion", note=""),
    "gnw-multistep-deliberation":dict(flag="BRAIN_GNW_MULTISTEP_LESION", value="1", kind="neural-lesion", note=""),
    "self-initiated-utterance": dict(flag="BRAIN_SELF_INITIATE_LESION", value="1", kind="neural-lesion", note=""),
    "vision-identity-spiking-hmax":dict(flag="BRAIN_VISION_IDENTITY_LESION", value="1", kind="neural-lesion", note=""),
    "bg-action-selection":      dict(flag="BRAIN_BG_SELECT_LESION", value="1", kind="neural-lesion", note=""),
    "open-ended-generation":    dict(flag="BRAIN_SPIKING_DRAW_LESION", value="1", kind="neural-lesion",
        note="cuts the vocab-agnostic spiking DRAW off cp_firing_states -> the generated hypothesis collapses "
             "(plausible-frac ~0.83->~0.04 in the B1 verify). Measured on the rich_open turn."),
    # ---- dedicated lesion flag EXISTS but the brain_chat-only probe cannot construct the trigger (thin) ----
    "gnw-deliberation":         dict(flag="BRAIN_GNW_DELIBERATE_LESION", value="1", kind="thin",
        note="lesion knob exists but the >=2-distinct-patient (agent,action) conflict is unconstructible via brain_chat "
             "(reconsolidation rewrites in place); the battery marks this probe thin. Not exercised by this harness."),
    "value-driven-choice":      dict(flag="BRAIN_VALUE_CHOICE_LESION", value="1", kind="thin",
        note="same root cause as gnw-deliberation: no 2-candidate conflict is constructible via brain_chat teaching."),
    # ---- no dedicated neural-cut knob; the master switch disables the whole organ (weaker) ----
    "confidence-forthcomingness":dict(flag="BRAIN_CONFIDENCE_FORTHCOMING", value="0", kind="whether-disable",
        note="no BRAIN_*_LESION knob; BRAIN_CONFIDENCE_FORTHCOMING=0 removes the organ -> the confidence_forthcoming key "
             "swings present/absent (a structural change, but a disable not a neural cut). Minimal neural lesion to add: "
             "BRAIN_CONFIDENCE_FORTHCOMING_LESION cutting the balance-of-evidence read while leaving the granted/reason "
             "trace attached."),
    # ---- no clean env lesion yet (in-process only or none); minimal proposal named ----
    "anaphora-wm":              dict(flag=None, value=None, kind="proposed",
        note="the compared field activity.roles is the parser's role binding on a DIRECT SVO ('the wolf watches the owl'), "
             "not anaphora resolution; the closest lesion is content-selection's in-process parser.role_of. No clean env "
             "lesion drives activity.roles. Minimal add: BRAIN_SUBSTRATE_PARSE_LESION (shared with content-selection)."),
    "discourse-planner":        dict(flag=None, value=None, kind="proposed",
        note="the rich multi-sentence plan (fields rich/n_sentences) has a neural_planner master switch but no neural-cut "
             "lesion. Minimal add: BRAIN_DISCOURSE_PLANNER_LESION cutting the plan's spiking sentence-count read."),
    "selective-attention-biased-competition":dict(flag=None, value=None, kind="proposed",
        note="the host content_bias lexicon was DELETED (ledger scaffold_retired:YES; learned SpikingFeatureCompat is the "
             "sole source), lesioned in-process by SpikingFeatureCompat.lesion(). No env knob. Minimal add: "
             "BRAIN_BIASED_COMPETITION_LESION reading the weights-cleared twin."),
}


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _faculty_row(key):
    for row in FACULTY_PROBES:
        if row[0] == key:
            return row  # (key, turn_label, fields, thin)
    return None


def turn_group(label):
    """The minimal ordered turn-group needed to reach `label`: all same-session turns up to and INCLUDING it (in
    declaration = execution order). Encodes the battery's shared-session dependencies (hold->held, dr_a->dr_b->dr_c,
    bc_a->bc_b, and the label-only epi_store->epi_recall) so a lesion arm reproduces the SAME session history the intact
    arm sees — a clean per-faculty control. Iterates PROBE_TURNS + _EXTRA_TURNS so the label-only driving pair (which is
    deliberately kept OUT of the default roster) still resolves to its store->recall group."""
    target = _TURN_BY_LABEL[label]
    sess = target[2]
    grp = []
    for t in list(PROBE_TURNS) + list(_EXTRA_TURNS):
        if t[2] == sess:
            grp.append(t[0])
        if t[0] == label:
            break
    return grp


def _flag_resolves(flag):
    """True iff the lesion flag string appears in an ORGAN source file (webapp/ or research/runners/), EXCLUDING this
    runner's own file (where every flag is named in FACULTY_LESIONS). Anti-false-negative guard: a flag that does not
    resolve is a NO-OP lesion that would read as 'not load-bearing' — reported as `lesion-knob-missing` instead. (This
    is presence-based: a necessary condition against typos/stale names, not a proof the flag gates the measured field —
    that is what the measurement itself decides.)"""
    if not flag:
        return False
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (research/ is one below)
    proj = os.path.dirname(root)
    for sub in ("webapp", os.path.join("research", "runners")):
        d = os.path.join(proj, sub)
        try:
            p = subprocess.run(["grep", "-rqlE", "--exclude-dir=__pycache__", "--include=*.py",
                                "--exclude=%s" % os.path.basename(__file__), flag, d],
                               capture_output=True)
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _draw_from_weights_honors_ablate():
    """Code-level check that the production wire-in draw `SpikingWTASampler.draw_from_weights` consults
    `ablate_likelihood` (the v2 wiring fix), so the neural DRAW lesion (BRAIN_SPIKING_DRAW_LESION) actually bites the
    production `_generate_hypothesis` draw and is not a silent no-op. Reads the method body -- a presence check, not
    proof the lesion changes a given reply (the measurement decides that)."""
    import inspect
    try:
        from research.runners._followon2_spiking_wta_sampler_derisk import SpikingWTASampler
        src = inspect.getsource(SpikingWTASampler.draw_from_weights)
    except Exception:
        return False
    return "ablate_likelihood" in src and "np.ones" in src


def _classify_diffs(diffs):
    """structural = a field goes present<->absent/null (organ output gated off); value = both present, value flips."""
    kinds = set()
    for d in diffs:
        on, off = d.get("on"), d.get("off")
        if on is None or off is None:
            kinds.add("structural")
        else:
            kinds.add("value")
    if "structural" in kinds:
        return "structural"
    return "value" if kinds else "none"


# ── the per-faculty lesion measurement ───────────────────────────────────────────────────────────────────────────
def _n_decision_diffs(row, arm_a, arm_b):
    """The number of the faculty's DECISION fields that differ between two arms (categorical, noise excluded). Reuses
    compare() so the exact same field set / noise exclusion / present-vs-absent rules apply."""
    return len(compare(arm_a, arm_b, faculties=[row])["per_faculty"][0]["diffs"])


def measure_faculty(key, out_dir, repeats=1, intact_cache=None, seed=42):
    """Build the INTACT arm TWICE (a, cached per turn-group; b, the NULL control) and the LESION arm for `key`, then
    make the explicit attribution call: TREATMENT = decision fields changed intact-vs-lesion; CONTROL = decision fields
    changed intact-vs-intact-rebuild. load-bearing requires a treatment change (attributed to the lesion) AND a clean
    null control (0 changes intact-vs-intact) — a change that also appears in the null is run-to-run noise, not the
    lesion. Returns the per-faculty result dict.

    `seed` (default 42, byte-identical to before this param existed) is NOT passed to _spawn_arm as an env override
    — the substrate seed is threaded via the process-wide BRAIN_CHAT_SEED env var set once by main()/run() for the
    whole invocation, so every arm build (including this faculty's) already builds at that seed. `seed` here only
    namespaces the OUTPUT FILENAMES (via `_seed_suffix`) so a multi-seed sweep sharing one --out dir cannot collide
    or false-skip under LB_RESUME_SKIP_EXISTING."""
    # OPEN-ENDED-GENERATION DISTRIBUTIONAL RULER (LB_OPEN_ENDED_DISTRIB_PROBE, default OFF): an early return BEFORE
    # any FACULTY_LESIONS lookup / _spawn_arm call -- this path never touches the webapp brain_chat machinery, so
    # every other faculty (and open-ended-generation itself when the flag is off) is byte-identical to before.
    if LB_OPEN_ENDED_DISTRIB and key == "open-ended-generation":
        return measure_open_ended_distributional(out_dir, seed=seed, repeats=repeats)
    # AFFECT->TONE OPEN-OUTPUT RULER (LB_AFFECT_TONE_OPEN_PROBE, default OFF): same early-return discipline --
    # swaps affect-coloring's ruler to the 6-seed directional open-output tone probe (reads the runner artifact),
    # never touching the webapp brain_chat machinery, so every other faculty (and affect-coloring itself when the
    # flag is off) stays byte-identical.
    if LB_AFFECT_TONE_OPEN and key == "affect-coloring":
        return measure_affect_tone_open_output(out_dir, seed=seed, repeats=repeats)
    # WM REFERENT->FOCUS BIND ANAPHOR PROBE (LB_WMB_FOCUS_PROBE, default OFF): same early-return discipline; takes
    # precedence over the content probe and the hold-query smoke when several of these flags are set.
    if LB_WMB_FOCUS and key == "wm-binding-advanced":
        return measure_wmb_focus(out_dir, seed=seed, repeats=repeats)
    # WM-BINDING ORDINARY-CONTENT PROBE (LB_WMB_CONTENT_PROBE, default OFF): same early-return discipline; takes
    # precedence over the hold-query INTEGRITY smoke when both flags are set.
    if LB_WMB_CONTENT and key == "wm-binding-advanced":
        return measure_wmb_content(out_dir, seed=seed, repeats=repeats)
    spec = FACULTY_LESIONS.get(key)
    row = _faculty_row(key)
    res = {"faculty": key, "turn": (row[1] if row else None),
           "kind": (spec or {}).get("kind"), "flag": (spec or {}).get("flag"),
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": None,
           "note": (spec or {}).get("note", "")}
    if row is None:
        res["verdict"] = "unmapped-in-battery"; return res
    if spec is None:
        res["verdict"] = "unmapped-lesion"; return res
    if spec["kind"] in ("in-process", "thin", "mechanism-only", "proposed"):
        # not drivable by this env-flag harness — report the reason, not a fake number.
        res["verdict"] = "not-covered:" + spec["kind"]
        res["flag_resolves"] = _flag_resolves(spec["flag"]) if spec["flag"] else None
        return res
    flag, val = spec["flag"], spec["value"]
    res["flag_resolves"] = _flag_resolves(flag)
    if not res["flag_resolves"]:
        res["verdict"] = "lesion-knob-missing"      # a flag that is not read anywhere -> would be a silent no-op
        res["load_bearing"] = None
        return res

    # EPISODIC DRIVING remap (default-off; see LB_EPISODIC_DRIVE). Make the episodic probe exercise its load-bearing
    # recall path: remap to the store->recall turn (its group is derived below as ['epi_store','epi_recall'] because
    # both are in session 'epi2', declared store-first) and FORCE the BTSP write so it runs on any backend. base_env is
    # applied to BOTH the intact and lesion arms (so the NULL control also stores -> both intact arms read in_memory=
    # True -> clean null; only the lesion collapses it). Every OTHER faculty keeps base_env={} -> byte-identical.
    base_env = {}
    if LB_EPISODIC_DRIVE and key == "episodic-memory":
        row = ("episodic-memory", _EPISODIC_DRIVE_TURN, ["episodic.in_memory"], False)
        base_env = dict(_EPISODIC_DRIVE_ENV)
        res["turn"] = _EPISODIC_DRIVE_TURN
        res["note"] = "LB_EPISODIC_DRIVE_PROBE: store->recall on session 'epi2' + BRAIN_EPISODIC_STORE=1. " + res["note"]
    # NON-CONTRADICTION DRIVING remap (default-off; see LB_NONCONTRADICTION_DRIVE). Remap the noncontradiction probe to
    # a fresh-session turn that ASSERTS the NEGATED form of the AFFIRM boot fact (dog,chase,cat): intact recalls "yes"
    # -> REJECT; lesion forces "unknown" -> ACCEPT -> reject/recalled_yn/stored_polarity diverge. NO forced-write env is
    # needed (the boot fact is stored on any backend), so base_env stays {} -> the NULL control is a plain rebuild
    # (byte-identical). Every OTHER faculty keeps base_env={} -> byte-identical.
    if LB_NONCONTRADICTION_DRIVE and key == "noncontradiction-gate":
        row = ("noncontradiction-gate", _NONCONTRA_DRIVE_TURN, list(_NONCONTRA_DRIVE_FIELDS), False)
        res["turn"] = _NONCONTRA_DRIVE_TURN
        res["note"] = ("LB_NONCONTRADICTION_DRIVE_PROBE: assert NEGATE of the AFFIRM boot fact (dog,chase,cat) on a "
                       "fresh session 'ncontra'; no forced-write env (boot-stored on any backend). " + res["note"])

    # SURPRISE CONFIRM remap (default-off; see LB_SURPRISE_CONFIRM). Make the surprise probe exercise its load-bearing
    # inhibition path: remap from the `contra` turn (a CONTRADICT trial the same-block lesion never touches) to the
    # `confirm` turn (asserted==stored, SHARED block), where the intact prediction cancels the excitation (surprised=
    # False) and the lesion removes that cancellation (surprised=True) -> the decision field `surprise.surprised` FLIPS.
    # No base_env is needed (unlike episodic): the confirm turn is already in PROBE_TURNS and deterministic, so both
    # intact arms read surprised=False -> clean null, only the lesion flips it. Every OTHER faculty keeps base_env={}.
    if LB_SURPRISE_CONFIRM and key == "surprise-monitor":
        row = ("surprise-monitor", _SURPRISE_CONFIRM_TURN, ["surprise.surprised"], False)
        res["turn"] = _SURPRISE_CONFIRM_TURN
        res["note"] = "LB_SURPRISE_CONFIRM_PROBE: measured on the CONFIRM turn 'confirm' (asserted==stored, shared block) where the same-block lesion bites; contra is a different-block CONTRADICT the lesion never reaches. " + res["note"]

    # DISCOURSE-REGISTER DRIVING remap (default-off; see LB_DISCOURSE_REGISTER_DRIVE). Make the discourse probe exercise
    # its load-bearing prev-slot read on a turn whose correct before-agent is NOT the register's identity index (which
    # the lesion forces the held prev slot to). Remap to the 'dr2' triple ('bird chase worm' -> 'then dog chase cat' ->
    # before?), whose group is derived below as ['dr2_a','dr2_b','dr2_c'] (all session 'dr2', declared clause-first) so
    # the lesion arm reproduces the SAME clause history. NO base_env: the register defaults spiking=True on any backend,
    # so nothing needs forcing (unlike episodic's cupy-gated BTSP write) -> every OTHER faculty stays byte-identical.
    if LB_DISCOURSE_REGISTER_DRIVE and key == "discourse-register":
        row = ("discourse-register", _DISCOURSE_DRIVE_TURN, ["discourse_register.abstained", "discourse_register.agent"], False)
        res["turn"] = _DISCOURSE_DRIVE_TURN
        res["note"] = ("LB_DISCOURSE_REGISTER_DRIVE_PROBE: before-agent is referents[3]='bird', not referents[0]='dog'"
                       "==ident, so the lesion's forced-identity fallback is distinguishable from the correct answer. "
                       + res["note"])
    # COMMON-GROUND DRIVING remap (default-off; see LB_CG_DRIVE). Make the common-ground probe exercise its
    # load-bearing audience-design path: remap to the mention->re-mention turn (its group is derived below as
    # ['cg_mention1','cg_mention2'] because both are in session 'cg2', declared mention-first) so the RE-MENTION reads
    # an ALREADY-grounded referent. base_env stays {} (both intact arms + the lesion arm; the ledger self-pins numpy),
    # so the NULL control also re-mentions -> both intact arms read decision=reduce -> clean null; only the lesion (its
    # recurrence built at weight 0) collapses the held slot to decision=introduce. Every OTHER faculty keeps base_env={}
    # -> byte-identical. The compared field narrows to `common_ground_drives.decision` (the reduce/introduce flip); `.on`
    # is True on both arms and `.reason` is absent on a topic'd turn, so neither could discriminate.
    if LB_CG_DRIVE and key == "common-ground-drives":
        row = ("common-ground-drives", _CG_DRIVE_TURN, ["common_ground_drives.decision"], False)
        base_env = dict(_CG_DRIVE_ENV)
        res["turn"] = _CG_DRIVE_TURN
        res["note"] = "LB_CG_DRIVE_PROBE: mention->re-mention on session 'cg2' ('dog'); intact reduce vs lesion introduce. " + res["note"]

    # AFFECT-COLORING DRIVING remap (default-off; see LB_AFFECT_DRIVE). Make the affect-coloring probe exercise its
    # load-bearing ladder read on a turn with a NONZERO mood to color: remap to the strongly-affective `emo` turn (its
    # group is derived below as ['emo'] -- its own single-turn session). No env-forcing (base_env stays {}) -- UNLIKE
    # episodic, the Gate-B ladder read runs on numpy for any turn, so no store or backend gate needs relaxing; the SAME
    # BRAIN_AFFECT_LESION neural cut then collapses affect_out=0 -> a 0.0 differential -> valence_sign flips "+"->"0".
    # Every OTHER faculty keeps base_env={} and its baseline row -> byte-identical.
    if LB_AFFECT_DRIVE and key == "affect-coloring":
        row = ("affect-coloring", _AFFECT_DRIVE_TURN, ["affect.on", "affect.valence_sign", "affect.tone_token"], False)
        res["turn"] = _AFFECT_DRIVE_TURN
        res["note"] = ("LB_AFFECT_DRIVE_PROBE: remap to the strongly-affective '%s' turn (no env-forcing). "
                       % _AFFECT_DRIVE_TURN) + res["note"]

    # BG-ACTION-SELECTION DRIVING remap (default-off; see LB_BG_SELECT_DRIVE). SAME turn ('bgdots'), SAME session,
    # SAME env (base_env stays {}) -> turn_group + both arm builds are byte-identical to the baseline; ONLY the field
    # list changes, from the confounded top-level `abstained` (True in BOTH arms — intact via the BG HOLD short-circuit,
    # lesion via the independent no-content abstain) to `bg_select.on` (present+True only on the intact short-circuit,
    # absent on the lesioned fallback) -> a structural intact-vs-lesion diff. The NULL control (intact vs intact-rebuild)
    # is unaffected: both intact arms fire the short-circuit -> both set bg_select.on=True -> 0 control diffs.
    if LB_BG_SELECT_DRIVE and key == "bg-action-selection":
        row = ("bg-action-selection", _BG_SELECT_DRIVE_TURN, list(_BG_SELECT_DRIVE_FIELDS), False)
        res["turn"] = _BG_SELECT_DRIVE_TURN
        res["note"] = ("LB_BG_SELECT_DRIVE_PROBE: compare bg_select.on (structural) instead of the confounded top-level "
                       "abstained (True in both arms). " + res["note"])

    # PROSPECTIVE-MEMORY DRIVING remap (default-off; see LB_PMEM_DRIVE). Make the prospective probe run the NATURAL
    # prospective protocol -- an intention held ACROSS intervening turns then released at a later cue: remap to the
    # cue turn (its group is derived below as ['pmem_form2','pmem_d0','pmem_d1','pmem_d2','pmem_cue'] because all five
    # are in session 'pmem2', declared formation-first) and compare `prospective.fired` instead of the compile-time-
    # constant `prospective.held`. No base_env is forced: BRAIN_PMEM + BRAIN_PMEM_HEBBIAN are default-ON, so the intact
    # arm learns the binding, holds it across the 3 intervening turns, and fires (fired=True); only the
    # BRAIN_PMEM_LESION arm collapses the latch (fired=False). Every OTHER faculty keeps base_env={} -> byte-identical.
    if LB_PMEM_DRIVE and key == "prospective-memory":
        row = ("prospective-memory", _PMEM_DRIVE_TURN, ["prospective.fired"], False)
        res["turn"] = _PMEM_DRIVE_TURN
        res["note"] = "LB_PMEM_DRIVE_PROBE: formation -> 3 intervening turns -> cue on session 'pmem2'. " + res["note"]
    # OPEN-ENDED-GENERATION DRIVING remap (default-off; see LB_OPEN_ENDED_DRIVE). Remap the open-ended-generation
    # measurement to the teach->ask group ('oe_t1..oe_t9' -> 'oe_ask', session 'oe2', derived below by turn_group)
    # that teaches the predator-prey chase KB, and compare the generative decision fields (`hypothesis_svo` the drawn
    # triple + the rendered `answer`). No base_env needed (the teach turns store via the standard in-loop acquire on
    # any backend); every OTHER faculty keeps base_env={} -> byte-identical.
    if LB_OPEN_ENDED_DRIVE and key == "open-ended-generation":
        row = ("open-ended-generation", _OPEN_ENDED_DRIVE_TURN, ["hypothesis_svo", "answer"], False)
        base_env = dict(_OPEN_ENDED_DRIVE_ENV)
        res["turn"] = _OPEN_ENDED_DRIVE_TURN
        res["note"] = ("LB_OPEN_ENDED_DRIVE_PROBE: teach->ask on session 'oe2' (predator-prey chase KB; novel "
                       "'rabbit' strictly dominates the stored 'cat'); BRAIN_SPIKING_PLAUSIBILITY=0 on BOTH arms so "
                       "the #3E gate admits the candidates (the default spiking gate masks the draw on the tiny KB); "
                       "the draw lesion (BRAIN_SPIKING_DRAW_LESION) is the only inter-arm difference + the honored "
                       "ablate on draw_from_weights. " + res["note"])
    # WM-BINDING HOLD-QUERY remap (default-off; see LB_WMB_HOLDQUERY). Remap to the intro->ask pair ('wmb_intro' ->
    # 'wmb_ask', session 'wmb'), compare the REPLY. No base_env. The adequacy + specificity gate runs after the
    # standard treatment/null/reproduce computation below (it can only turn a verdict UNDEFINED, never into a pass).
    _wmb_on = bool(LB_WMB_HOLDQUERY and key == "wm-binding-advanced")
    if _wmb_on:
        row = ("wm-binding-advanced", _WMB_DRIVE_TURN, list(_WMB_DRIVE_FIELDS), False)
        res["turn"] = _WMB_DRIVE_TURN
        res["note"] = ("LB_WMB_HOLDQUERY_PROBE: 'the fox and the wolf walked in' -> 'who are we talking about' on "
                       "session 'wmb' (reply = the organ's read-back); specificity control = the same ask after a "
                       "1-referent intro (session 'wmb1'). " + res["note"])
    # SWAP-DRIVES ADEQUATE remap (default-off; see LB_SWAP_DRIVE). Remap swap-drives-response to the topic-change turn
    # `sw_switch` (group sw_open -> sw_hold -> sw_switch, session 'sw2', derived below by turn_group) and compare the
    # reply + the swap trace. base_env stays {} -> the arms differ ONLY by BRAIN_SWAP_DRIVES_LESION. The contrast/
    # null/reply scoring is applied after the standard verdict (below), via the pre-registered _swap_drive_score.
    if LB_SWAP_DRIVE and key == "swap-drives-response":
        row = ("swap-drives-response", _SWAP_DRIVE_TURN, list(_SWAP_DRIVE_FIELDS), False)
        res["turn"] = _SWAP_DRIVE_TURN
        res["note"] = ("LB_SWAP_DRIVE_PROBE: sw_open('dog') -> sw_hold('dog', contrast) -> sw_switch('cat') on session "
                       "'sw2'; reply must change on the switch and NOT on the no-swap-due contrast turns. " + res["note"])

    # DA TAG-AND-CAPTURE next-day remap (default-off; see LB_DA_TAG_CAPTURE).
    if LB_DA_TAG_CAPTURE and key == "da-gated-encoding":
        row = ("da-gated-encoding", _DA_TAG_CAPTURE_TURN, list(_DA_TAG_CAPTURE_FIELDS), False)
        base_env = dict(_DA_TAG_CAPTURE_ENV)
        res["turn"] = _DA_TAG_CAPTURE_TURN
        res["note"] = ("LB_DA_TAG_CAPTURE_PROBE: salient news around one plain fact -> a night through the brain's own "
                       "idle/sleep tick -> next-day recall on session 'datc'; BRAIN_DA_TAG_CAPTURE=1 on both arms. "
                       + res["note"])

    grp = turn_group(row[1])
    # cache key includes base_env so a stored (BRAIN_EPISODIC_STORE) intact arm never aliases a plain-{} arm on a
    # shared turn-group (the driving group is unique anyway, but keep the key honest).
    env_sig = ",".join("%s=%s" % (k, v) for k, v in sorted(base_env.items()))
    grp_sig = ",".join(grp) + ("|" + env_sig if env_sig else "")
    _fname = grp_sig.replace(",", "_").replace("|", "__").replace("=", "-")
    intact_cache = intact_cache if intact_cache is not None else {}
    _sfx = _seed_suffix(seed)

    # INTACT arm (base env = all defaults on, plus any driving base_env), built TWICE: `a` (cached per turn-group,
    # shared across faculties on the same turn) and `b` the NULL control (a fresh rebuild at the same seed -> the
    # run-to-run baseline of "no change").
    if grp_sig not in intact_cache:
        # union: base_env carries the driving-probe env (episodic + hollow drives); _sfx threads the per-seed suffix.
        a = _spawn_arm(dict(base_env), grp, os.path.join(out_dir, "intact_a_%s%s.json" % (_fname, _sfx)))
        b = _spawn_arm(dict(base_env), grp, os.path.join(out_dir, "intact_b_%s%s.json" % (_fname, _sfx)))
        intact_cache[grp_sig] = (a, b)
    intact_a, intact_b = intact_cache[grp_sig]

    # LESION arm. union: _sfx threads the per-seed suffix; base_env carries the driving-probe env (both arms share it).
    les_out = os.path.join(out_dir, "lesion_%s%s.json" % (key.replace("-", "_"), _sfx))
    lesioned = _spawn_arm({**base_env, flag: val}, grp, les_out)
    if intact_a is None or intact_b is None or lesioned is None:
        res["verdict"] = "arm-build-failed"; return res
    # A brain that FAILED TO LOAD returns per-turn {'_error': ...} dicts, not None -- which then compared as "fields
    # absent in both arms" and read NOT-EXERCISED: a silent false negative (2026-09-23, an AWS run missing the
    # `experiment` package read not-exercised on all 10 arms in seconds). Surface it as the build failure it is.
    _arm_errs = [t.get("_error") for arm in (intact_a, intact_b, lesioned) if isinstance(arm, dict)
                 for t in arm.values() if isinstance(t, dict) and t.get("_error")]
    if _arm_errs:
        res["verdict"] = "arm-build-failed"; res["arm_error"] = str(_arm_errs[0])[:300]; return res

    treat_pf = compare(intact_a, lesioned, faculties=[row])["per_faculty"][0]
    res["verdict"] = treat_pf["verdict"]        # "regressed" (=changed) / "pass" (=identical) / "not-exercised"
    res["diffs"] = treat_pf["diffs"]
    res["change_kind"] = _classify_diffs(treat_pf["diffs"]) if treat_pf["diffs"] else "none"
    treatment_diffs = len(treat_pf["diffs"])
    control_diffs = _n_decision_diffs(row, intact_a, intact_b)   # the NULL: must be 0 on a deterministic harness
    res["treatment_diffs"] = treatment_diffs
    res["control_diffs"] = control_diffs
    res["null_control_clean"] = (control_diffs == 0)

    # THE ATTRIBUTION CALL (tools.lab): what fraction of the observed decision change is the lesion vs the null control.
    res["attributable_fraction"] = attributable_to("load-bearing[%s]" % key, treatment_diffs, control_diffs)

    # LESION-REPEAT (optional extra anti-noise): rebuild the lesion arm and require the SAME verdict.
    reproduced = True
    for i in range(max(0, repeats - 1)):
        les2 = _spawn_arm({**base_env, flag: val}, grp, les_out + ".rep%d" % i)
        if les2 is None or compare(intact_a, les2, faculties=[row])["per_faculty"][0]["verdict"] != treat_pf["verdict"]:
            reproduced = False
            break
    res["lesion_reproduced"] = reproduced

    # Load-bearing iff: the decision CHANGED under lesion, the change is NOT present in the null control (clean
    # attribution -> the change is the lesion's, not noise), and it reproduces.
    if treat_pf["verdict"] == "not-exercised":
        res["load_bearing"] = None
    elif treat_pf["verdict"] == "pass":
        res["load_bearing"] = False
    elif not res["null_control_clean"]:
        res["load_bearing"] = None
        res["verdict"] = "noisy-null-control"       # the intact arm itself changed run-to-run -> verdict untrustworthy
    elif not reproduced:
        res["load_bearing"] = None
        res["verdict"] = "noisy"
    else:
        res["load_bearing"] = True                  # changed, attributable to the lesion, reproduced

    if _wmb_on:
        # SPECIFICITY control arms (the 1-referent 'wmb1' pair): intact, intact-rebuild, lesion. Then the pre-registered
        # adequacy gate; a failed condition makes the verdict UNDEFINED (load_bearing None), never a pass.
        cgrp = turn_group(_WMB_CONTROL_TURN)
        _cf = "_".join(cgrp)
        ctrl_a = _spawn_arm({}, cgrp, os.path.join(out_dir, "intact_a_%s%s.json" % (_cf, _sfx)))
        ctrl_b = _spawn_arm({}, cgrp, os.path.join(out_dir, "intact_b_%s%s.json" % (_cf, _sfx)))
        ctrl_les = _spawn_arm({flag: val}, cgrp, os.path.join(out_dir, "lesion_%s_ctrl%s.json"
                                                               % (key.replace("-", "_"), _sfx)))
        override, adequacy = _wmb_adequacy(intact_a, ctrl_a, ctrl_b, ctrl_les, lesioned=lesioned)
        res["wmb_adequacy"] = adequacy
        res["wmb_control_group"] = cgrp
        res["wmb_mechanism"] = {"intact_a": _wmb_mech(intact_a, _WMB_DRIVE_TURN),
                                "intact_b": _wmb_mech(intact_b, _WMB_DRIVE_TURN),
                                "lesion": _wmb_mech(lesioned, _WMB_DRIVE_TURN),
                                "control_intact_a": _wmb_mech(ctrl_a, _WMB_CONTROL_TURN),
                                "control_lesion": _wmb_mech(ctrl_les, _WMB_CONTROL_TURN)}
        res["verdict_before_adequacy"] = res["verdict"]
        if override is not None:
            res["verdict"] = override
            res["load_bearing"] = None
        # INTEGRITY SMOKE (review v2:7a3b94367): the hold-query reply's only input is the lesioned buffer, so a change is
        # predetermined once the route is reached. The smoke's own outcome is kept (`integrity_smoke_verdict`); the
        # faculty record NEVER carries load_bearing from it, and its verdict is outside ("regressed","pass") so run()'s
        # fraction excludes it from both numerator and denominator.
        res["integrity_smoke"] = True
        res["integrity_smoke_verdict"] = res["verdict"]
        res["verdict"] = "integrity-smoke"
        res["load_bearing"] = None
        res["wmb_probe_flags"] = _wmb_probe_flags()
    if LB_SWAP_DRIVE and key == "swap-drives-response":
        _score_swap_drive(res, row, treat_pf, intact_a, intact_b, lesioned, reproduced)
    return res


def _score_swap_drive(res, row, treat_pf, intact_a, intact_b, lesioned, reproduced):
    """Apply the pre-registered swap-drives gate to `res` IN PLACE (only called under LB_SWAP_DRIVE_PROBE). Adds the
    contrast measurements (the lesion vs intact on the no-swap-due turns), extends the null control to every turn of
    the group, records each arm's own swap state on every turn (the mechanism's state, not an arg-max of the metric),
    and re-derives (load_bearing, verdict) through _swap_drive_score."""
    fields = row[2]
    c_rows = [("swap-drives-response", t, list(fields), False) for t in _SWAP_DRIVE_CONTRAST_TURNS]
    contrast_diffs, contrast_null_diffs, contrast_exercised = [], 0, True
    for cr in c_rows:
        pf = compare(intact_a, lesioned, faculties=[cr])["per_faculty"][0]
        contrast_diffs += [dict(d, turn=cr[1]) for d in pf["diffs"]]
        contrast_null_diffs += _n_decision_diffs(cr, intact_a, intact_b)
        # the contrast is DEFINED only if the swap trace is present on the intact arm for that turn
        if not _get_path((intact_a or {}).get(cr[1]) or {}, "swap_drives.reason")[0]:
            contrast_exercised = False
    null_clean = bool(res["null_control_clean"]) and contrast_null_diffs == 0
    lb, verdict = _swap_drive_score(treat_pf["verdict"], treat_pf["diffs"], null_clean, reproduced,
                                    contrast_diffs, contrast_exercised)
    state = {}
    for arm_name, arm in (("intact_a", intact_a), ("intact_b", intact_b), ("lesion", lesioned)):
        state[arm_name] = {}
        for t in _SWAP_DRIVE_CONTRAST_TURNS + (_SWAP_DRIVE_TURN,):
            sd = ((arm or {}).get(t) or {}).get("swap_drives") or {}
            state[arm_name][t] = {k: sd.get(k) for k in ("swapped", "reason", "topic", "held_topic_before",
                                                         "held_topic", "lead", "lesioned", "mm_peak", "boost_max")}
            state[arm_name][t]["answer"] = ((arm or {}).get(t) or {}).get("answer")
    res.update({
        "swap_drive_probe": True,
        "reply_changed": any(d.get("field") == _SWAP_DRIVE_REPLY_FIELD for d in treat_pf["diffs"]),
        "contrast_diffs": contrast_diffs,
        "contrast_null_diffs": contrast_null_diffs,
        "contrast_exercised": contrast_exercised,
        "null_control_clean_switch_turn": res["null_control_clean"],
        "null_control_clean": null_clean,          # the faculty's null now covers every turn of the group
        "swap_state": state,
        "verdict_standard_rule": res["verdict"], "load_bearing_standard_rule": res["load_bearing"],
        "load_bearing": lb, "verdict": verdict,
    })


# ── the full measurement ─────────────────────────────────────────────────────────────────────────────────────────
def run(out_dir="research/findings/raw/_load_bearing", only=None, repeats=1, seed=42):
    """`seed` (default 42, byte-identical): the substrate seed for this WHOLE invocation. Callers (main() below) are
    responsible for setting the process-wide BRAIN_CHAT_SEED env var to this same value BEFORE calling run() — this
    function does not set it itself (it may be called directly, e.g. from a test, without the env side effect) —
    `seed` here is threaded only to `measure_faculty` for output-filename namespacing (`_seed_suffix`)."""
    os.makedirs(out_dir, exist_ok=True)
    # DEVICE STAMP (device-and-cost gate): record the backend the arms actually built on. The battery worker inherits
    # this process's SIM_BACKEND (it spawns with dict(os.environ)); default numpy via setdefault. Recorded so the
    # result is auditable without a provenance sidecar (a CPU/GPU mix-up is a different experiment, not a slow run).
    report = {"runner": "research.runners.load_bearing_fraction",
              "metric": "load_bearing_fraction", "repeats": repeats, "seed": seed,
              "backend": os.environ.get("SIM_BACKEND", "numpy"),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              # ENV-KNOB READBACK (tools.verdict.Verdict.knob discipline: a flag PASSED is not the same claim as
              # a flag that REACHED the arm builds -- 2026-09-22, research/lbf-fix-source-provenance-abstain's own
              # verify script reads this back rather than trusting its own invocation command line). Additive-only
              # (a new report key); every existing consumer of this report is unaffected.
              "source_prov_abstain_at_tie_env": os.environ.get("BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE"),
              # same readback for the affect-marker SETTLE companion processes (2026-09-23, D1 lane; additive key).
              "affect_marker_settle_env": os.environ.get("BRAIN_AFFECT_MARKER_SETTLE")}

    keys = only or faculty_list()
    intact_cache = {}
    per = [measure_faculty(k, out_dir, repeats=repeats, intact_cache=intact_cache, seed=seed) for k in keys]
    report["per_faculty"] = per

    # DETERMINISM / NULL CONTROL, aggregated from every exercised faculty's intact-vs-intact-rebuild control. The
    # harness is deterministic iff every null control is clean (0 changes); a dirty null makes that faculty's verdict
    # untrustworthy (its "change" is partly run-to-run noise). This IS the metric's "not just noise" proof, now
    # per-turn-group rather than a single representative turn.
    checked = [p for p in per if p.get("control_diffs") is not None]
    dirty = [p["faculty"] for p in checked if not p.get("null_control_clean")]
    report["determinism"] = {"deterministic": (len(dirty) == 0) if checked else None,
                             "n_null_controls_checked": len(checked), "dirty_null_controls": dirty}
    if dirty:
        report["UNRELIABLE"] = ("intact arm is NON-DETERMINISTIC at seed 42 for %s -> those load-bearing verdicts are "
                                "untrustworthy (the change is partly run-to-run noise); seed the substrate" % dirty)

    # DENOMINATOR = faculties this harness can lesion+exercise (neural-lesion / whether-disable with a driving probe).
    coverable = [p for p in per if p["kind"] in ("neural-lesion", "whether-disable")]
    # not-exercised/missing/noisy excluded. "trace-only" (LB_SWAP_DRIVE_PROBE only: the swap trace changed but the reply
    # did not) is exercised-and-NOT-load-bearing; no other path emits it, so the default battery is unchanged.
    exercised = [p for p in coverable if p["verdict"] in ("regressed", "pass", "trace-only")]
    load_bearing = [p for p in exercised if p["load_bearing"] is True]
    report["counts"] = {
        "n_faculties_total": len(per),
        "n_coverable_env_lesion": len(coverable),
        "n_exercised": len(exercised),
        "n_load_bearing": len(load_bearing),
        "n_not_load_bearing": len([p for p in exercised if p["load_bearing"] is False]),
        "n_not_exercised": len([p for p in coverable if p["verdict"] == "not-exercised"]),
        "n_noisy": len([p for p in coverable if p["verdict"] in ("noisy", "noisy-null-control")]),
        "n_lesion_knob_missing": len([p for p in coverable if p["verdict"] == "lesion-knob-missing"]),
        # reported separately, NOT in the denominator:
        "n_in_process": len([p for p in per if p["kind"] == "in-process"]),
        "n_thin": len([p for p in per if p["kind"] == "thin"]),
        "n_mechanism_only": len([p for p in per if p["kind"] == "mechanism-only"]),
        "n_proposed": len([p for p in per if p["kind"] == "proposed"]),
    }
    n_ex = report["counts"]["n_exercised"]
    report["load_bearing_fraction"] = (report["counts"]["n_load_bearing"] / n_ex) if n_ex else None
    report["load_bearing_faculties"] = [p["faculty"] for p in load_bearing]
    report["not_load_bearing_faculties"] = [p["faculty"] for p in exercised if p["load_bearing"] is False]
    return report


def _wmb_selftest_checks():
    """LB_WMB_HOLDQUERY_PROBE static wiring + pure adequacy-gate logic, in BOTH directions (no brain build)."""
    from research.runners.d6_multiref_wm_production_organ import extract_referents, is_hold_query
    t = _TURN_BY_LABEL
    default_labels = {x[0] for x in PROBE_TURNS}

    def arm(turn, answer, mr=None):
        r = {"answer": answer}
        if mr is not None:
            r["multiref"] = mr
        return {turn: r}
    q2 = {"kind": "query", "is_hold_query": True, "n_referents": 2}
    good_i = arm(_WMB_DRIVE_TURN, "I'm holding 2 referents in working memory at once: fox and wolf.", q2)
    c_ok = arm(_WMB_CONTROL_TURN, "I don't know.")
    return {
        "wmb flag parses to a real bool": isinstance(LB_WMB_HOLDQUERY, bool),
        "wmb turns are label-only (NOT in the default roster)":
            not ({"wmb_intro", "wmb_ask", "wmb1_intro", "wmb1_ask"} & default_labels),
        "wmb drive group is intro->ask": turn_group(_WMB_DRIVE_TURN) == ["wmb_intro", "wmb_ask"],
        "wmb control group is intro->ask": turn_group(_WMB_CONTROL_TURN) == ["wmb1_intro", "wmb1_ask"],
        "wmb intro names exactly the two lexicon referents": extract_referents(t["wmb_intro"][1]) == list(_WMB_REFERENTS),
        "wmb control intro names ONE referent (organ out of scope)": len(extract_referents(t["wmb1_intro"][1])) == 1,
        "wmb ask is a hold-query in both sessions":
            is_hold_query(t["wmb_ask"][1]) and is_hold_query(t["wmb1_ask"][1]) and t["wmb_ask"][1] == t["wmb1_ask"][1],
        "wmb lesion flag resolves in source": _flag_resolves("BRAIN_MULTIREF_LESION"),
        "wmb gate: adequate + specific -> no override":
            _wmb_adequacy(good_i, c_ok, c_ok, c_ok)[0] is None,
        "wmb gate: intercepted ask (no multiref query) -> probe-inadequate:route":
            _wmb_adequacy(arm(_WMB_DRIVE_TURN, "fox and wolf"), c_ok, c_ok, c_ok)[0] == "probe-inadequate:route",
        "wmb gate: reply names one referent -> probe-inadequate:two-referents":
            _wmb_adequacy(arm(_WMB_DRIVE_TURN, "I'm holding: fox.", q2), c_ok, c_ok, c_ok)[0]
            == "probe-inadequate:two-referents",
        "wmb gate: control routed through the organ -> control-inadequate":
            _wmb_adequacy(good_i, arm(_WMB_CONTROL_TURN, "x", q2), arm(_WMB_CONTROL_TURN, "x", q2),
                          arm(_WMB_CONTROL_TURN, "x", q2))[0] == "control-inadequate",
        "wmb gate: control reply changes under lesion -> off-target-lesion (fails closed)":
            _wmb_adequacy(good_i, c_ok, c_ok, arm(_WMB_CONTROL_TURN, "something else"))[0] == "off-target-lesion",
        "wmb gate: control rebuild differs -> noisy-null-control":
            _wmb_adequacy(good_i, c_ok, arm(_WMB_CONTROL_TURN, "other"), c_ok)[0] == "noisy-null-control",
        "wmb gate: a missing control arm -> arm-build-failed": _wmb_adequacy(good_i, None, c_ok, c_ok)[0]
            == "arm-build-failed",
        "wmb gate: LESION arm off the hold-query route -> probe-inadequate:route (A1 on every arm)":
            _wmb_adequacy(good_i, c_ok, c_ok, c_ok, lesioned=arm(_WMB_DRIVE_TURN, "I don't know."))[0]
            == "probe-inadequate:route",
        "wmb gate: lesion arm on the route -> no override":
            _wmb_adequacy(good_i, c_ok, c_ok, c_ok, lesioned=arm(_WMB_DRIVE_TURN, "none", q2))[0] is None,
    }


def _wmc_selftest_checks():
    """LB_WMB_CONTENT_PROBE wiring + the pure `_wmc_gate` in BOTH directions (no brain build)."""
    from research.runners.d6_multiref_wm_production_organ import extract_referents, is_hold_query
    t = _TURN_BY_LABEL
    default_labels = {x[0] for x in PROBE_TURNS}

    def turn_rec(intro_mr, drive_answer, drive_extra=None):
        return {"intro": {"answer": "x", "multiref": intro_mr}, "drive": dict({"answer": drive_answer},
                                                                               **(drive_extra or {}))}

    def mk(c, intact_ans, lesion_ans, lesion_rep_ans=None, intact_b_ans=None, les_scope="recur", les_alive=0.0,
           intro_n=2, drive_extra=None):
        intro, drive, _ = _WMC_TURNS[c]
        mi = {"kind": "maintain", "n_referents": intro_n, "hold_alive_min": 0.06}
        ml = {"kind": "maintain", "n_referents": intro_n, "hold_alive_min": les_alive, "lesion_scope": les_scope}

        def arm(m, ans):
            r = turn_rec(m, ans, drive_extra)
            return {intro: r["intro"], drive: r["drive"]}
        return {"intact_a": arm(mi, intact_ans), "intact_b": arm(mi, intact_b_ans or intact_ans),
                "lesion": arm(ml, lesion_ans), "lesion_rep": arm(ml, lesion_rep_ans or lesion_ans)}
    A_chg = mk("A", "did the wolf watch the owl?", "who watched whom?")
    B_chg = mk("B", "did the dog watch the owl?", "who watched whom?")
    A_same = mk("A", "did the wolf watch the owl?", "did the wolf watch the owl?")
    B_same = mk("B", "did the dog watch the owl?", "did the dog watch the owl?")
    return {
        "wmc flag parses to a real bool": isinstance(LB_WMB_CONTENT, bool),
        "wmc turns are label-only (NOT in the default roster)":
            not ({"wmc_intro", "wmc_drive", "wmcx_intro", "wmcx_drive"} & default_labels),
        "wmc groups are intro->drive": turn_group("wmc_drive") == ["wmc_intro", "wmc_drive"]
            and turn_group("wmcx_drive") == ["wmcx_intro", "wmcx_drive"],
        "wmc intros name exactly their two lexicon referents":
            extract_referents(t["wmc_intro"][1]) == ["fox", "wolf"]
            and extract_referents(t["wmcx_intro"][1]) == ["cat", "dog"],
        "wmc drives are ordinary (not hold-queries) and structurally identical":
            not is_hold_query(t["wmc_drive"][1]) and not is_hold_query(t["wmcx_drive"][1])
            and t["wmc_drive"][1].replace("wolf", "X") == t["wmcx_drive"][1].replace("dog", "X"),
        "wmc confined-lesion knob resolves in source": _flag_resolves("BRAIN_MULTIREF_LESION_SCOPE"),
        "wmc gate: both contents change -> load_bearing True": _wmc_gate({"A": A_chg, "B": B_chg})[:2]
            == (True, "regressed"),
        "wmc gate: NEITHER changes -> load_bearing False (a real negative; it CAN fail)":
            _wmc_gate({"A": A_same, "B": B_same})[:2] == (False, "pass"),
        "wmc gate: only one content changes -> UNDEFINED content-dependent-effect":
            _wmc_gate({"A": A_chg, "B": B_same})[:2] == (None, "content-dependent-effect"),
        "wmc gate: lesion arm NOT confined (scope missing) -> probe-inadequate:route":
            _wmc_gate({"A": mk("A", "the wolf", "x", les_scope=None), "B": B_chg})[1] == "probe-inadequate:route",
        "wmc gate: organ out of scope on the intro -> probe-inadequate:route":
            _wmc_gate({"A": mk("A", "the wolf", "x", intro_n=1), "B": B_chg})[1] == "probe-inadequate:route",
        "wmc gate: drive is an inner-state read-out -> probe-inadequate:not-ordinary":
            _wmc_gate({"A": mk("A", "the wolf", "x", drive_extra={"inner_state_readout": True}), "B": B_chg})[1]
            == "probe-inadequate:not-ordinary",
        "wmc gate: lesion hold still alive -> lesion-not-effective":
            _wmc_gate({"A": mk("A", "the wolf", "x", les_alive=0.05), "B": B_chg})[1] == "lesion-not-effective",
        "wmc gate: intact rebuild differs -> noisy-null-control":
            _wmc_gate({"A": mk("A", "the wolf", "x", intact_b_ans="the fox"), "B": B_chg})[1]
            == "noisy-null-control",
        "wmc gate: reply ignores the input content -> probe-inadequate:content":
            _wmc_gate({"A": mk("A", "I don't know.", "x"), "B": B_chg})[1] == "probe-inadequate:content",
        "wmc gate: reply names the OTHER content -> probe-inadequate:content":
            _wmc_gate({"A": mk("A", "the wolf and the dog", "x"), "B": B_chg})[1] == "probe-inadequate:content",
        "wmc gate: lesion rebuild differs -> noisy": _wmc_gate({"A": mk("A", "the wolf", "x", lesion_rep_ans="y"),
                                                               "B": B_chg})[1] == "noisy",
        "wmc gate: a missing arm -> arm-build-failed":
            _wmc_gate({"A": dict(A_chg, lesion=None), "B": B_chg})[1] == "arm-build-failed",
    }


def _wmf_selftest_checks():
    """LB_WMB_FOCUS_PROBE wiring + the pure `_wmf_gate` / `_wmf_headline` in BOTH directions (no brain build)."""
    from research.runners.d6_multiref_wm_production_organ import extract_referents, is_hold_query
    t = _TURN_BY_LABEL
    default_labels = {x[0] for x in PROBE_TURNS}
    held = {"A": (("dog", "cat"), ("cat", "dog")), "B": (("cat", "bird"), ("bird", "cat"))}

    def mk(reply, resolved=None, lesion_reply=None, les_alive=0.0, les_scope="recur", intact_b_reply=None,
           lesion_rep_reply=None, no_resolve=False, svo=None, ask_extra=None, les_ask_rates=(0.0, 0.0, 0.0, 0.0, 0.0)):
        """Synthetic arms. reply/resolved/lesion_reply/svo: {pair: (session1, session2)}."""
        def arm(kind):
            a = {}
            for p, ((i1, a1), (i2, a2)) in _WMF_PAIRS.items():
                for s, (intro, ask) in enumerate(((i1, a1), (i2, a2))):
                    mi = {"kind": "maintain", "n_referents": 2, "hold_alive_min": 0.08}
                    if kind.startswith("lesion"):
                        mi = {"kind": "maintain", "n_referents": 2, "hold_alive_min": les_alive,
                              "lesion_scope": les_scope}
                    a[intro] = {"answer": "x", "multiref": mi}
                    src = {"intact_a": reply, "intact_b": intact_b_reply or reply, "lesion": lesion_reply or reply,
                           "lesion_rep": lesion_rep_reply or lesion_reply or reply}[kind]
                    d = {"answer": src[p][s], "recalled_svo": (svo or {}).get(p, (None, None))[s]}
                    if kind.startswith("intact") and not no_resolve:
                        # default resolution: each session's first-mentioned referent (register 0 won the race)
                        res_pair = (resolved or {}).get(p) or (held[p][0][0], held[p][1][0])
                        d["multiref"] = {"kind": "resolve", "resolved": res_pair[s],
                                         "register_rates": [0.09, 0.08, 0.0, 0.0, 0.0]}
                    elif kind.startswith("lesion") and not no_resolve:
                        d["multiref"] = {"kind": "resolve", "resolved": None, "register_rates": list(les_ask_rates)}
                    d.update(ask_extra or {})
                    a[ask] = d
            return a
        return {k: arm(k) for k in _WMF_ARMS}
    content = {"A": ("the dog chases the cat", "I don't know about that."),
               "B": ("the cat eats the fish", "I don't know about that.")}
    same = {"A": ("the dog chases the cat", "the dog chases the cat"),
            "B": ("the cat eats the fish", "the cat eats the fish")}
    svo_ok = {"A": (["dog", "chase", "cat"], None), "B": (["cat", "eat", "fish"], None)}
    good = mk(content, lesion_reply=same, svo=svo_ok)
    positional = mk(same, lesion_reply=same, no_resolve=True)
    offorgan = mk(content, lesion_reply=content, svo=svo_ok)
    return {
        "wmf flag parses to a real bool": isinstance(LB_WMB_FOCUS, bool),
        "wmf turns are label-only (NOT in the default roster)": not (set(_WMF_TURNS) & default_labels),
        "wmf turns resolve by label": all(lab in t for lab in _WMF_TURNS),
        "wmf pairs: order-swapped intros over the SAME two lexicon referents":
            all(extract_referents(t[i1][1]) == list(held[p][0]) and extract_referents(t[i2][1]) == list(held[p][1])
                for p, ((i1, _a1), (i2, _a2)) in _WMF_PAIRS.items()),
        "wmf pairs: identical ordinary anaphor question within each pair":
            all(t[a1][1] == t[a2][1] and "it" in t[a1][1].split() and not is_hold_query(t[a1][1])
                and not extract_referents(t[a1][1]) for _p, ((_i1, a1), (_i2, a2)) in _WMF_PAIRS.items()),
        "wmf sessions are distinct per session and intro-first":
            all(turn_group(a) == [i, a] for _p, ss in _WMF_PAIRS.items() for i, a in ss),
        "wmf mechanism flag resolves in source": _flag_resolves("BRAIN_MULTIREF_FOCUS_BIND"),
        "wmf gate: reply follows the held referent + lesion removes it -> load_bearing True":
            _wmf_gate(good)[:2] == (True, "regressed"),
        "wmf gate: POSITIONAL route (OFF, identical replies) -> load_bearing False (it CAN fail)":
            _wmf_gate(positional, require_resolution=False)[:2] == (False, "pass"),
        "wmf gate: difference survives the hold lesion -> UNDEFINED off-organ-route":
            _wmf_gate(offorgan)[:2] == (None, "off-organ-route"),
        "wmf gate: ON arm never resolved -> probe-inadequate:no-resolution":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, no_resolve=True))[1]
            == "probe-inadequate:no-resolution",
        "wmf gate: both sessions resolved the SAME referent -> probe-inadequate:same-referent":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok,
                         resolved={"A": ("dog", "dog"), "B": ("cat", "bird")}))[1] == "probe-inadequate:same-referent",
        "wmf gate: reply recalls a DIFFERENT agent than resolved -> probe-inadequate:content":
            _wmf_gate(mk(content, lesion_reply=same, svo={"A": (["cat", "eat", "fish"], None),
                                                          "B": svo_ok["B"]}))[1] == "probe-inadequate:content",
        "wmf gate: lesion hold still alive -> lesion-not-effective":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, les_alive=0.05))[1] == "lesion-not-effective",
        "wmf gate: lesion no longer holds at the ask (a live register) -> lesion-not-effective":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, les_ask_rates=(0.0, 0.07, 0.0, 0.0, 0.0)))[1]
            == "lesion-not-effective",
        "wmf gate: lesion arm not confined -> probe-inadequate:route":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, les_scope=None))[1] == "probe-inadequate:route",
        "wmf gate: ask is an inner-state read-out -> probe-inadequate:not-ordinary":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, ask_extra={"inner_state_readout": True}))[1]
            == "probe-inadequate:not-ordinary",
        "wmf gate: intact rebuild differs -> noisy-null-control":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, intact_b_reply=same))[1] == "noisy-null-control",
        "wmf gate: lesion rebuild differs -> noisy":
            _wmf_gate(mk(content, lesion_reply=same, svo=svo_ok, lesion_rep_reply=content))[1] == "noisy",
        "wmf gate: only one pair follows -> UNDEFINED content-dependent-effect":
            _wmf_gate(mk({"A": content["A"], "B": same["B"]}, lesion_reply=same, svo=svo_ok))[:2]
            == (None, "content-dependent-effect"),
        "wmf gate: a missing arm -> arm-build-failed": _wmf_gate(dict(good, lesion=None))[1] == "arm-build-failed",
        "wmf headline: positional (OFF) route reading load-bearing VOIDS the ON verdict":
            _wmf_headline((True, "regressed", {}), (True, "regressed", {})) == (None,
                                                                             "probe-inadequate:positional-passes"),
        "wmf headline: OFF not load-bearing -> the ON verdict stands":
            _wmf_headline((True, "regressed", {}), (False, "pass", {})) == (True, "regressed")
            and _wmf_headline((False, "pass", {}), (None, "off-organ-route", {})) == (False, "pass"),
    }


# ── self-test: prove the instrument's LOGIC without building a brain (no memcap needed) ──────────────────────────
def selftest(out_path=None):
    """Verify the verdict inversion + change classification on synthetic responses (no brain build). Mirrors the
    battery's --skip-real synthetic demo: a real load-bearing lesion is DETECTED, a no-op is NOT, absence is handled.
    With out_path, ALSO write a citable static-verification artifact (the checks + episodic-driving remap wiring)."""
    row = _faculty_row("curiosity-followup")  # fields: curiosity.curious / curiosity.on
    intact = {"unknown": {"abstained": True, "curiosity": {"curious": True, "on": True}}}
    lesioned = {"unknown": {"abstained": True, "curiosity": {"curious": False, "on": True}}}  # crave silenced
    no_op = {"unknown": {"abstained": True, "curiosity": {"curious": True, "on": True}}}       # identical
    absent = {"unknown": {"abstained": True}}                                                   # field gone both arms

    d_change = compare(intact, lesioned, faculties=[row])["per_faculty"][0]
    d_noop = compare(intact, no_op, faculties=[row])["per_faculty"][0]
    d_absent = compare(absent, absent, faculties=[row])["per_faculty"][0]

    checks = {
        "load_bearing_detected (regressed on a real lesion)": d_change["verdict"] == "regressed",
        "change_kind value": _classify_diffs(d_change["diffs"]) == "value",
        # attribution wiring: a change absent from the null control is 100% attributable to the lesion; a change equally
        # present in the null control is 0% attributable (noise) -> the load-bearing gate must reject it.
        "attribution clean-null == 1.0": attributable_to("selftest-clean", 1, 0) == 1.0,
        "attribution dirty-null == 0.0": attributable_to("selftest-dirty", 1, 1) == 0.0,
        "attribution both-null UNDEFINED": attributable_to("selftest-none", 0, 0) is None,
        "not_load_bearing (pass on a no-op lesion)": d_noop["verdict"] == "pass",
        "not_exercised (fields absent both arms)": d_absent["verdict"] == "not-exercised",
        "structural change classified": _classify_diffs(
            [{"field": "x.y", "on": "v", "off": None}]) == "structural",
        "turn_group hold->held": turn_group("held") == ["hold", "held"],
        "turn_group dr chain": turn_group("dr_c") == ["dr_a", "dr_b", "dr_c"],
        "turn_group bc pair": turn_group("bc_b") == ["bc_a", "bc_b"],
        "turn_group single": turn_group("well") == ["well"],
        "null lesion flag unread in source": not _flag_resolves(NULL_LESION_FLAG),
        "a real lesion flag resolves in source": _flag_resolves("BRAIN_CURIOSITY_LESION"),
        # episodic-driving remap (LB_EPISODIC_DRIVE_PROBE): the store->recall pair exists and its group is store-first,
        # the forced-write flag is real, and remapping episodic to `epi_recall` still reads the same in_memory field.
        "episodic-drive turns exist": all(l in _TURN_BY_LABEL for l in (_EPISODIC_DRIVE_TURN, "epi_store")),
        "episodic-drive group is store->recall": turn_group(_EPISODIC_DRIVE_TURN) == ["epi_store", _EPISODIC_DRIVE_TURN],
        "episodic-drive forces the BTSP write": _flag_resolves("BRAIN_EPISODIC_STORE") and "1" in _EPISODIC_DRIVE_ENV.values(),
        # surprise-confirm remap (LB_SURPRISE_CONFIRM_PROBE): the CONFIRM turn is already in the default roster (no new
        # turn/session), its group is the single confirm turn, and the surprise neural-cut lesion flag resolves in source.
        "surprise-confirm turn is in the default roster": _SURPRISE_CONFIRM_TURN in {t[0] for t in PROBE_TURNS},
        "surprise-confirm group is the single confirm turn": turn_group(_SURPRISE_CONFIRM_TURN) == [_SURPRISE_CONFIRM_TURN],
        "surprise lesion flag resolves in source": _flag_resolves("BRAIN_SURPRISE_LESION"),
        # discourse-register-driving remap (LB_DISCOURSE_REGISTER_DRIVE_PROBE): the dr2 clause->shift->before triple
        # exists, its group is the 3-clause 'dr2' chain, and the correct before-agent ('bird' = dr2_a's subject) maps to
        # a referent index that is NOT the register's identity index 0 (which the lesion forces the held prev slot to).
        # This is the guard against the exact index collision the lone `dr_c` probe fell into (correct-before 'dog' ==
        # referents[0] == ident, so intact and lesion returned the identical agent).
        "discourse-drive turns exist": all(l in _TURN_BY_LABEL for l in (_DISCOURSE_DRIVE_TURN, "dr2_a", "dr2_b")),
        "discourse-drive group is the dr2 chain": turn_group(_DISCOURSE_DRIVE_TURN) == ["dr2_a", "dr2_b", _DISCOURSE_DRIVE_TURN],
        "discourse-drive before-agent avoids the ident collision (bird=3 != dog=0)": (
            _TURN_BY_LABEL["dr2_a"][1].split()[0] in _DR2_PROD_REFERENTS
            and _DR2_PROD_REFERENTS.index(_TURN_BY_LABEL["dr2_a"][1].split()[0]) == 3
            and 3 != _DR2_IDENT
            and _DR2_PROD_REFERENTS[_DR2_IDENT] == "dog"),
        "discourse-drive lesion knob resolves": _flag_resolves("BRAIN_DISCOURSE_REGISTER_LESION"),
        # common-ground-driving remap (LB_CG_DRIVE_PROBE): the mention->re-mention pair exists and its group is
        # mention-first, the lesion knob is real, and no forced-write/backend env is needed (the ledger self-pins numpy).
        "cg-drive turns exist": all(l in _TURN_BY_LABEL for l in (_CG_DRIVE_TURN, "cg_mention1")),
        "cg-drive group is mention->re-mention": turn_group(_CG_DRIVE_TURN) == ["cg_mention1", _CG_DRIVE_TURN],
        "cg-drive lesion knob resolves": _flag_resolves("BRAIN_CG_DRIVES_LESION"),
        "cg-drive needs no forced env": _CG_DRIVE_ENV == {},
        # noncontradiction-driving remap (LB_NONCONTRADICTION_DRIVE_PROBE): the driving turn exists, is a SINGLE-turn
        # group (the boot fact dog/chase/cat=AFFIRM is present at every tiny-demo build -> no store turn needed), the
        # lesion knob resolves in source, and the probe text parses to the NEGATED form of that boot fact (so intact
        # recalls "yes"/AFFIRM -> REJECT, lesion forces "unknown" -> ACCEPT: reject/recalled_yn/stored_polarity flip).
        "noncontradiction-drive turn exists": _NONCONTRA_DRIVE_TURN in _TURN_BY_LABEL,
        "noncontradiction-drive group is single": turn_group(_NONCONTRA_DRIVE_TURN) == [_NONCONTRA_DRIVE_TURN],
        "noncontradiction-drive lesion knob resolves": _flag_resolves("BRAIN_NONCONTRADICTION_LESION"),
        "noncontradiction-drive probe negates the dog/chase/cat boot fact": _noncontra_probe_parses_negated_boot_fact(),
        # affect-coloring-driving remap (LB_AFFECT_DRIVE_PROBE): the strongly-affective turn is ALREADY in the default
        # roster (no _EXTRA_TURNS / no battery edit needed), its group is the lone self-contained turn (no dependency
        # chain, no env-forcing), and the affect-coloring neural-cut lesion flag resolves so the remapped read can flip.
        "affect-drive turn is in the default roster": _AFFECT_DRIVE_TURN in {t[0] for t in PROBE_TURNS},
        "affect-drive group is the lone turn": turn_group(_AFFECT_DRIVE_TURN) == [_AFFECT_DRIVE_TURN],
        "affect-drive lesion flag resolves": _flag_resolves(FACULTY_LESIONS["affect-coloring"]["flag"]),
        # bg-select-driving remap (LB_BG_SELECT_DRIVE_PROBE): the driving turn is already in the default roster (no
        # _EXTRA_TURNS needed) and is a single-turn group; the CONFOUNDED `abstained` field reads pass on both arms
        # (True intact via the BG HOLD short-circuit, True lesion via the independent no-content abstain) while the
        # remapped `bg_select.on` field is a structural regressed diff (present+True intact, absent lesioned).
        "bg-select turn in default roster": _BG_SELECT_DRIVE_TURN in {t[0] for t in PROBE_TURNS},
        "bg-select group is single bgdots": turn_group(_BG_SELECT_DRIVE_TURN) == [_BG_SELECT_DRIVE_TURN],
        "bg-select confounded field passes (the bug)": compare(
            {_BG_SELECT_DRIVE_TURN: {"abstained": True, "bg_select": {"on": True}}},
            {_BG_SELECT_DRIVE_TURN: {"abstained": True}},
            faculties=[("bg-action-selection", _BG_SELECT_DRIVE_TURN, ["abstained"], False)]
        )["per_faculty"][0]["verdict"] == "pass",
        "bg-select remapped field is structural regressed (the fix)": (lambda pf: pf["verdict"] == "regressed"
            and _classify_diffs(pf["diffs"]) == "structural")(compare(
            {_BG_SELECT_DRIVE_TURN: {"abstained": True, "bg_select": {"on": True}}},
            {_BG_SELECT_DRIVE_TURN: {"abstained": True}},
            faculties=[("bg-action-selection", _BG_SELECT_DRIVE_TURN, list(_BG_SELECT_DRIVE_FIELDS), False)]
        )["per_faculty"][0]),
        # prospective-memory-driving remap (LB_PMEM_DRIVE_PROBE): the formation -> intervening -> cue chain exists and
        # its group holds the intention across >=1 intervening turn (the operating-point condition -- a zero-delay
        # formation->cue does not fire even intact), and remapping prospective to `pmem_cue` reads the load-bearing
        # `prospective.fired` field (whose lesion flag BRAIN_PMEM_LESION resolves in source). No forced write.
        "pmem-drive turns exist": all(l in _TURN_BY_LABEL for l in (_PMEM_DRIVE_TURN, "pmem_form2")),
        "pmem-drive group is formation->intervening->cue": turn_group(_PMEM_DRIVE_TURN) == ["pmem_form2", "pmem_d0", "pmem_d1", "pmem_d2", _PMEM_DRIVE_TURN],
        "pmem-drive holds across >=1 intervening turn": len(turn_group(_PMEM_DRIVE_TURN)) >= 3,
        "pmem-drive lesion knob resolves": _flag_resolves("BRAIN_PMEM_LESION"),
        # open-ended-generation driving remap (LB_OPEN_ENDED_DRIVE_PROBE): the teach->ask group exists, its group is
        # the 9 teach turns then the ask, the lesion knob resolves, and draw_from_weights now HONORS ablate_likelihood
        # (so the lesion bites the production draw -- the v1 wiring gap this v2 branch fixed).
        "open-ended-drive turns exist": all(l in _TURN_BY_LABEL for l in (_OPEN_ENDED_DRIVE_TURN, "oe_t1")),
        "open-ended-drive group is teach->ask": turn_group(_OPEN_ENDED_DRIVE_TURN) == [
            "oe_t1", "oe_t2", "oe_t3", "oe_t4", "oe_t5", "oe_t6", "oe_t7", "oe_t8", "oe_t9", _OPEN_ENDED_DRIVE_TURN],
        "open-ended lesion knob resolves": _flag_resolves("BRAIN_SPIKING_DRAW_LESION"),
        "open-ended lesion bites production draw": _draw_from_weights_honors_ablate(),
        # DA tag-and-capture next-day remap (LB_DA_TAG_CAPTURE_PROBE): the group is tell -> night -> recall, one
        # session, the night is a world step (never a brain_chat turn), and the companion flag resolves in source.
        "da-tag-capture group is tell->night->recall": (
            turn_group(_DA_TAG_CAPTURE_TURN)[-2:] == ["datc_night", _DA_TAG_CAPTURE_TURN]
            and len(turn_group(_DA_TAG_CAPTURE_TURN)) == 7),
        "da-tag-capture night is a world step": "datc_night" in __import__(
            "research.runners.onebrain_regression_battery", fromlist=["_WORLD_STEPS"])._WORLD_STEPS,
        "da-tag-capture companion flag resolves": _flag_resolves("BRAIN_DA_TAG_CAPTURE"),
        # open-ended-generation DISTRIBUTIONAL ruler (LB_OPEN_ENDED_DISTRIB_PROBE): pure decision-logic checks (NO
        # brain build -- `_oed_score` takes already-measured numbers) + static wiring checks (source inspection /
        # import + signature only, no SimulationBridge construction), mirroring how _classify_diffs()/compare() are
        # tested directly on synthetic values above.
        "oed-score: clean null + a real effect -> load-bearing":
            _oed_score(0.337, 0.337, 0.01) == (0.0, abs(0.337 - 0.01), True, True, "regressed"),
        "oed-score: clean null + no effect -> NOT load-bearing":
            _oed_score(0.337, 0.337, 0.337) == (0.0, 0.0, True, False, "pass"),
        "oed-score: dirty null -> UNDEFINED (never a positive OR a negative)":
            _oed_score(0.337, 0.20, 0.01)[2:4] == (False, None)
            and _oed_score(0.337, 0.20, 0.01)[4] == "noisy-null-control",
        "oed-score: a dirty null with NO lesion effect is STILL reported unclean, not silently 'pass'":
            _oed_score(0.20, 0.10, 0.20)[2] is False,
        "the distributional branch precedes every _spawn_arm call (never touches the webapp brain_chat path)": (
            lambda src: ("LB_OPEN_ENDED_DISTRIB" in src) and ("_spawn_arm(" in src)
            and src.find("LB_OPEN_ENDED_DISTRIB") < src.find("_spawn_arm(")
        )(__import__("inspect").getsource(measure_faculty)),
        # co_names (the compiled function's referenced globals/attrs) -- NOT raw source text, which would false-
        # positive on the docstring's own prose naming what it does NOT call (the exact pitfall _followon2's own
        # `_code_only` docstring-stripping helper works around).
        "measure_open_ended_distributional's CODE never references _spawn_arm / onebrain_regression_battery":
            "_spawn_arm" not in measure_open_ended_distributional.__code__.co_names
            and "onebrain_regression_battery" not in measure_open_ended_distributional.__code__.co_names,
        "SpikingWTASampler exposes the ablate_likelihood knob the distributional lesion uses": (
            "ablate_likelihood" in __import__("inspect").signature(
                __import__("research.runners._followon2_spiking_wta_sampler_derisk",
                           fromlist=["SpikingWTASampler"]).SpikingWTASampler.__init__).parameters
        ),
        "open-ended distributional flag parses to a real bool (env-string parsing didn't degrade to truthy-string)":
            isinstance(LB_OPEN_ENDED_DISTRIB, bool),
        # ── AFFECT->TONE OPEN-OUTPUT RULER (LB_AFFECT_TONE_OPEN_PROBE): pure decision-logic + static wiring checks ──
        "affect-tone-open flag parses to a real bool":
            isinstance(LB_AFFECT_TONE_OPEN, bool),
        "affect-tone-open: correct/correct + clean control -> load-bearing":
            _affect_tone_open_row({"pos_state": "correct", "neg_state": "correct",
                                   "real_directional_gap": 0.30, "ctrl_directional_gap": 0.00}, 0.05)
            == (True, "load-bearing", 0.30, 0.00, True),
        "affect-tone-open: a wrong-sign direction -> NOT load-bearing (must fail closed)":
            _affect_tone_open_row({"pos_state": "wrong", "neg_state": "correct",
                                   "real_directional_gap": 0.30, "ctrl_directional_gap": 0.00}, 0.05)[:2]
            == (False, "wrong-sign"),
        "affect-tone-open: a null (within-band) direction -> pass (not load-bearing)":
            _affect_tone_open_row({"pos_state": "null", "neg_state": "correct",
                                   "real_directional_gap": 0.02, "ctrl_directional_gap": 0.00}, 0.05)[:2]
            == (False, "pass"),
        "affect-tone-open: an UNCLEAN control -> UNDEFINED (never a positive OR negative)":
            _affect_tone_open_row({"pos_state": "correct", "neg_state": "correct",
                                   "real_directional_gap": 0.30, "ctrl_directional_gap": 0.20}, 0.05)[:2]
            == (None, "noisy-null-control"),
        "the affect-tone-open branch precedes every _spawn_arm call (never touches the webapp brain_chat path)": (
            lambda src: ("LB_AFFECT_TONE_OPEN" in src) and ("_spawn_arm(" in src)
            and src.find("LB_AFFECT_TONE_OPEN") < src.find("_spawn_arm(")
        )(__import__("inspect").getsource(measure_faculty)),
        "measure_affect_tone_open_output's CODE never references _spawn_arm / onebrain_regression_battery":
            "_spawn_arm" not in measure_affect_tone_open_output.__code__.co_names
            and "onebrain_regression_battery" not in measure_affect_tone_open_output.__code__.co_names,
        "oed n-attempts knob is a positive int (the _followon2 GO's 800 unless explicitly overridden)":
            isinstance(_OED_N_ATTEMPTS, int) and _OED_N_ATTEMPTS > 0,
        # ── SWAP-DRIVES ADEQUATE PROBE (LB_SWAP_DRIVE_PROBE): static wiring + the pre-registered decision logic ──
        "swap-drive flag parses to a real bool": isinstance(LB_SWAP_DRIVE, bool),
        "swap-drive turns exist (label-only, NOT in the default roster)": (
            all(l in _TURN_BY_LABEL for l in _SWAP_DRIVE_CONTRAST_TURNS + (_SWAP_DRIVE_TURN,))
            and not ({_SWAP_DRIVE_TURN, *_SWAP_DRIVE_CONTRAST_TURNS} & {t[0] for t in PROBE_TURNS})),
        "swap-drive group is open->hold->switch": turn_group(_SWAP_DRIVE_TURN) == ["sw_open", "sw_hold", "sw_switch"],
        "swap-drive lesion knob resolves": _flag_resolves("BRAIN_SWAP_DRIVES_LESION"),
        "swap-drive production extractor reads dog/dog/cat (and None on the old `held` probe)":
            _swap_probe_topics_static() == {"held": None, "sw_open": "dog", "sw_hold": "dog", "sw_switch": "cat"},
        "swap-score: reply change + clean null + clean contrast + reproduced -> load-bearing":
            _swap_drive_score("regressed", [{"field": "answer"}, {"field": "swap_drives.swapped"}], True, True, [], True)
            == (True, "regressed"),
        "swap-score: trace changes but reply identical -> trace-only (NOT load-bearing)":
            _swap_drive_score("regressed", [{"field": "swap_drives.swapped"}], True, True, [], True)
            == (False, "trace-only"),
        "swap-score: lesion changes a no-swap-due contrast turn -> nonspecific (UNDEFINED, never a pass)":
            _swap_drive_score("regressed", [{"field": "answer"}], True, True, [{"field": "answer"}], True)
            == (None, "nonspecific-lesion"),
        "swap-score: contrast trace absent -> contrast-undefined":
            _swap_drive_score("regressed", [{"field": "answer"}], True, True, [], False) == (None, "contrast-undefined"),
        "swap-score: dirty null on any turn -> noisy-null-control":
            _swap_drive_score("regressed", [{"field": "answer"}], False, True, [], True) == (None, "noisy-null-control"),
        "swap-score: identical switch -> pass (NOT load-bearing)":
            _swap_drive_score("pass", [], True, True, [], True) == (False, "pass"),
        "swap-score: fields absent both arms -> not-exercised":
            _swap_drive_score("not-exercised", [], True, True, [], False) == (None, "not-exercised"),
        "swap-score FULL PATH (_score_swap_drive on synthetic arms): swap + clean contrast -> load-bearing":
            _swap_score_synthetic(nonspecific=False) == (True, "regressed"),
        "swap-score FULL PATH: lesion alters the hold turn -> nonspecific-lesion":
            _swap_score_synthetic(nonspecific=True) == (None, "nonspecific-lesion"),
        "swap-score: unreproduced lesion -> noisy":
            _swap_drive_score("regressed", [{"field": "answer"}], True, False, [], True) == (None, "noisy"),
        "every FACULTY_LESIONS key is a real battery faculty":
            all(k in faculty_list() for k in FACULTY_LESIONS),
        "every battery faculty is mapped": all(k in FACULTY_LESIONS for k in faculty_list()),
        # SEED THREADING (research/seed-threading-lbf, 2026-09-20): seed=42 (the pre-existing hardcoded value)
        # keeps the ORIGINAL un-suffixed filenames -- byte-identical to every on-disk artifact from before --seed
        # existed; only a non-42 seed adds the _s<seed> tag, so a 6-seed sweep sharing one --out dir never collides.
        "seed_suffix default(42) is empty (byte-identical filenames)": _seed_suffix(42) == "",
        "seed_suffix non-default namespaces the filename": _seed_suffix(43) == "_s43",
        "seed_suffix accepts the full 6-seed roster": [_seed_suffix(s) for s in (42, 43, 44, 100, 101, 102)]
            == ["", "_s43", "_s44", "_s100", "_s101", "_s102"],
    }
    checks.update(_wmb_selftest_checks())
    checks.update(_wmc_selftest_checks())
    checks.update(_wmf_selftest_checks())
    ok = all(checks.values())
    print("=== LOAD-BEARING INSTRUMENT SELF-TEST ===")
    for name, passed in checks.items():
        print("  [%s] %s" % ("PASS" if passed else "FAIL", name))
    # coverage summary of the lesion map
    from collections import Counter
    kinds = Counter(v["kind"] for v in FACULTY_LESIONS.values())
    print("  lesion-map coverage:", dict(kinds), "over", len(faculty_list()), "battery faculties")
    print("VERDICT:", "PASS" if ok else "FAIL")
    if out_path:
        # A citable STATIC-verification artifact (no brain build): the instrument-logic checks + the episodic-driving
        # remap wiring + the roster-unchanged facts. Written through the runner so provenance sidecars it.
        # NB: key is `selftest_result`, NOT `verdict`/`status`/`go` — a selftest is an instrument-logic check, not a
        # scientific GO/NO-GO verdict, so it deliberately does not trip the verdict-preconditions gate.
        art = {"runner": "research.runners.load_bearing_fraction", "kind": "selftest",
               "selftest_result": "PASS" if ok else "FAIL", "checks": checks,
               "n_probe_turns_default_roster": len(PROBE_TURNS),
               "episodic_drive_group": turn_group(_EPISODIC_DRIVE_TURN),
               "episodic_drive_env": _EPISODIC_DRIVE_ENV,
               "surprise_confirm_turn": _SURPRISE_CONFIRM_TURN,
               "surprise_confirm_group": turn_group(_SURPRISE_CONFIRM_TURN),
               "discourse_drive_group": turn_group(_DISCOURSE_DRIVE_TURN),
               "discourse_drive_before_agent": _TURN_BY_LABEL["dr2_a"][1].split()[0],
               "discourse_drive_env": {},
               "cg_drive_group": turn_group(_CG_DRIVE_TURN),
               "cg_drive_env": _CG_DRIVE_ENV,
               "noncontradiction_drive_group": turn_group(_NONCONTRA_DRIVE_TURN),
               "noncontradiction_drive_env": {},   # no forced-write env: the boot fact is present on any backend
               "affect_drive_turn": _AFFECT_DRIVE_TURN,
               "affect_drive_group": turn_group(_AFFECT_DRIVE_TURN),
               "affect_drive_in_default_roster": _AFFECT_DRIVE_TURN in {t[0] for t in PROBE_TURNS},
               "bg_select_drive_group": turn_group(_BG_SELECT_DRIVE_TURN),
               "bg_select_drive_fields": _BG_SELECT_DRIVE_FIELDS,
               "pmem_drive_group": turn_group(_PMEM_DRIVE_TURN),
               "lesion_map_coverage": dict(kinds)}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(art, open(out_path, "w"), indent=2, default=str)
        print("  wrote", out_path)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Load-bearing fraction instrument (lesion-verified, reuses the regression battery).")
    ap.add_argument("--out", default="research/findings/raw/_load_bearing/load_bearing.json")
    ap.add_argument("--smoke", default=None, help="measure ONE faculty (its key) intact-vs-lesion, for the tiny memcap smoke")
    ap.add_argument("--only", default=None, help="comma-separated faculty keys to restrict to")
    ap.add_argument("--repeats", type=int, default=1, help="lesion-arm rebuilds for the anti-noise reproduce check (>=1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="substrate seed for every arm this invocation builds (research/seed-threading-lbf, "
                         "2026-09-20): sets BRAIN_CHAT_SEED for the whole run + namespaces every per-arm output "
                         "filename (_seed_suffix) so a multi-seed sweep sharing one --out dir cannot collide or "
                         "false-skip under LB_RESUME_SKIP_EXISTING. Default 42 is BYTE-IDENTICAL to before this "
                         "flag existed (unsuffixed filenames, BRAIN_CHAT_SEED=42 behaves exactly like unset). The "
                         "mandated 6-seed validation is 42/43/44/100/101/102, one invocation per seed.")
    ap.add_argument("--selftest", action="store_true", help="verify the instrument logic without building a brain (no cap needed)")
    ap.add_argument("--selftest-out", default=None, help="also write the selftest checks to this JSON artifact (no brain build)")
    ap.add_argument("--map", action="store_true", help="print the per-faculty lesion map and exit (no brain build)")
    args = ap.parse_args()

    if args.selftest or args.selftest_out:
        return 0 if selftest(out_path=args.selftest_out) else 1
    if args.map:
        for k in faculty_list():
            s = FACULTY_LESIONS.get(k, {})
            print("%-42s %-14s %-38s %s" % (k, s.get("kind", "UNMAPPED"), s.get("flag") or "-",
                                            (("val=%s " % s["value"]) if s.get("value") else "")))
        return 0

    # CORPUS GUARD (2026-09-23). Corpus-LEARNED organs (the comprehension animacy / verb-selects lexicons, the
    # open-ended world) silently fall back to their standalone/hand paths when data/corpus/ is absent, so a remote
    # battery reads them as NOT load-bearing -- a false negative with a clean null control. Measured: an AWS shard
    # read comprehension-monitor s100 `pass` (treat=0); the identical command locally, with the corpus, read
    # `regressed` (treat=1, ctrl=0) with and without the fix flags. Refuse to measure instead of mis-measuring.
    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _missing = [f for f in ("tinystories.txt", "wikitext.txt", "simplewiki.txt", "websters1913.json")
                if not os.path.exists(os.path.join(_proj, "data", "corpus", f))]
    if _missing and os.environ.get("LB_ALLOW_NO_CORPUS") != "1":
        print("⛔ load_bearing_fraction: data/corpus/ is missing %s -- corpus-learned faculties would silently read "
              "NOT load-bearing. Sync the corpus (tools/pool_sync_assets.sh / the AWS provisioner) or set "
              "LB_ALLOW_NO_CORPUS=1 to measure the degraded brain on purpose." % _missing, file=sys.stderr)
        return 3

    # SEED THREADING: set the process-wide substrate seed ONCE, before any arm build. _spawn_arm_raw (reused
    # verbatim from onebrain_regression_battery.py) passes `env=dict(os.environ)` to every arm subprocess, so this
    # single assignment is what makes EVERY arm (intact + lesion, every faculty) build at args.seed. Unconditional
    # even at the default (--seed unset -> 42): BRAIN_CHAT_SEED=42 reads identically to unset everywhere it is
    # consumed (webapp/server.py._brain_chat_seed + every per-organ workspace's own _DEFAULT_SEED), so this is a
    # byte-identical no-op for the shipped default.
    os.environ["BRAIN_CHAT_SEED"] = str(args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        report = run(out_dir=out_dir, only=[args.smoke], repeats=max(2, args.repeats), seed=args.seed)
    else:
        only = args.only.split(",") if args.only else None
        report = run(out_dir=out_dir, only=only, repeats=args.repeats, seed=args.seed)

    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("\n===== LOAD-BEARING FRACTION =====")
    print("  seed=%d (BRAIN_CHAT_SEED=%s)" % (args.seed, os.environ.get("BRAIN_CHAT_SEED")))
    if "determinism" in report:
        d = report["determinism"]
        print("  determinism/null-control (intact vs intact-rebuild): deterministic=%s (checked=%s, dirty=%s)"
              % (d.get("deterministic"), d.get("n_null_controls_checked"), d.get("dirty_null_controls")))
    if "UNRELIABLE" in report:
        print("  ⛔ UNRELIABLE:", report["UNRELIABLE"])
    c = report["counts"]
    print("  faculties: total=%d coverable(env-lesion)=%d exercised=%d" % (
        c["n_faculties_total"], c["n_coverable_env_lesion"], c["n_exercised"]))
    print("  LOAD-BEARING=%d  not-load-bearing=%d  not-exercised=%d  noisy=%d  knob-missing=%d"
          % (c["n_load_bearing"], c["n_not_load_bearing"], c["n_not_exercised"], c["n_noisy"], c["n_lesion_knob_missing"]))
    print("  (reported separately: in-process=%d thin=%d mechanism-only=%d proposed=%d)"
          % (c["n_in_process"], c["n_thin"], c["n_mechanism_only"], c["n_proposed"]))
    frac = report["load_bearing_fraction"]
    print("  LOAD_BEARING_FRACTION = %s  (%d/%d exercised env-lesion faculties)"
          % (("%.3f" % frac) if frac is not None else "n/a", c["n_load_bearing"], c["n_exercised"]))
    if report.get("load_bearing_faculties"):
        print("  load-bearing:", report["load_bearing_faculties"])
    if report.get("not_load_bearing_faculties"):
        print("  NOT load-bearing (cosmetic on this probe):", report["not_load_bearing_faculties"])
    print("  wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
