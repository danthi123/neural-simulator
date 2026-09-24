---
type: finding
status: live
date: 2026-09-24
lane: load-bearing
mechanism: PRE-REGISTRATION of five new BRAIN_*_LESION env knobs converting five FACULTY_LESIONS rows
  (content-selection, semantic-recall, moat-verify, in-loop-learning, discourse-planner,
  selective-attention-biased-competition -- six faculty keys, five flags, BRAIN_SUBSTRATE_RECALL_LESION shared by
  moat-verify/in-loop-learning) from in-process/proposed to an env-driven neural-lesion, plus an LB_CONFLICT_KB_PROBE
  fixture-state check for gnw-deliberation / value-driven-choice reusing the dog->chase->{cat,ball} ambiguity
  fixture. All five flags default OFF; opt-in only.
seeds: [7]
verdict: PRE-REGISTRATION only. No load-bearing claim is made here. Seed 7 is a dev/smoke seed
  (CLAUDE.md), never a validation seed; the 6-seed gate for these rows runs in B2b, not here.
runner: research/runners/load_bearing_fraction.py (via research/runners/lbf_rows/proposed_lesions_conflict_kb.py)
artifacts:
  - research/findings/raw/_value_choice_prodflip/soak_summary_6seed.json
---

# A11: five proposed lesion knobs + the conflict-KB fixture (2026-09-24)

Plan step S14 of tonight's midnight orchestration plan, lane A11, worktree
`research/lbf-proposed-lesions-conflict-kb`. No artifact is cited here: this is filed before any smoke has run
(the smoke/probe JSONs this pre-registration governs will be produced and cited by the FOLLOW-ON data commit,
never by this one). Reads `research/runners/load_bearing_fraction.py`'s own
`FACULTY_LESIONS` table first (it already names the exact minimal-add flag for every row below, in its own
`note` field) and the LBF ROW INTERFACE (`research/runners/lbf_rows/<module>.py` exposing module-level
`EXTRA_LESIONS`/`EXTRA_PROBES`, merged by AG-REG's import hook; `FACULTY_LESIONS`/`FACULTY_PROBES` themselves are
never edited directly by this lane).

## Phase 1 — five default-inert lesion knobs

Each flag is a pure additive early-return at the top of the exact method `_production_lesion_probe.py` already
lesions in-process by monkeypatch (this is the SAME cut FACULTY_LESIONS's own notes ask for, made available as
an env flag instead of a Python monkeypatch): with the flag unset, `_<x>_lesioned()` returns `False` and every
existing line runs completely unchanged -- STRUCTURAL byte-identity (no reachable line moves), not merely
empirical.

| faculty key | flag | file:function | cut |
|---|---|---|---|
| content-selection | `BRAIN_SUBSTRATE_PARSE_LESION` | `research/runners/brain_chat_tui.py::ChatBrain._neural_question_parse` | skips the on-brain `BridgeParser.role_of` (position,voice)->role read entirely; returns `None` (comprehension declines) instead of sampling it |
| semantic-recall | `BRAIN_COMPOSER_RECALL_LESION` | `research/runners/one_brain_composer.py::OneBrainComposer.query_patient` | returns `None` before the spiking K-way `_seq_block` selection runs |
| moat-verify | `BRAIN_SUBSTRATE_RECALL_LESION` | `research/runners/brain_chat_tui.py::ChatBrain._substrate_recall` | returns `None` unconditionally (no `_extract_route`, no `inner.what_does`) |
| in-loop-learning | `BRAIN_SUBSTRATE_RECALL_LESION` (shared) | same as moat-verify | same cut; FACULTY_LESIONS already names the SAME minimal add for both keys |
| discourse-planner | `BRAIN_DISCOURSE_PLANNER_LESION` | `research/runners/rich_answer_composer.py::NeuralDiscoursePlanner.ordered_associates` | returns `[]` unconditionally (the dlPFC spreading-activation latency read is never run) |
| selective-attention-biased-competition | `BRAIN_BIASED_COMPETITION_LESION` | `research/runners/_gap3_spiking_feature_compat_derisk.py::SpikingFeatureCompat.bias_target` | returns `None` unconditionally (the weights-cleared twin: as if `self.ca`/`self.vs` were all zero) |

### Honest kind, per row (declared before any smoke is run)

- **content-selection, semantic-recall, moat-verify, in-loop-learning, discourse-planner**: `kind="neural-lesion"`
  once the seed-7 smoke below confirms the flag actually bites its faculty's existing `FACULTY_PROBES` turn
  (`well` / `well` / `unknown` / `well` / `rich_well` -- all four already have real, non-thin rows in
  `onebrain_regression_battery.FACULTY_PROBES`, so **no new probe turn is needed**; `EXTRA_PROBES` is empty).
  Each cut silences a genuine neural read (the on-brain parser's role conjunction, the composer's spiking
  block-selection, the substrate recall's `what_does` read, the dlPFC spreading-activation latency read) rather
  than disabling a whole organ via its `BRAIN_<X>=0` master switch (`whether-disable`), matching the convention
  `BRAIN_AFFECT_LESION` (`affect_production_organ.py`) already sets for this codebase.
- **selective-attention-biased-competition: `kind="proposed"` — NOT upgraded to `neural-lesion`, and this is the
  one HONEST NEGATIVE this pre-registration expects.** `SpikingFeatureCompat` is only installed as
  `agent._feat_compat_source` by `MultiTurnAgent.build_referent_bias_from_experience`, gated on
  `len(agent.heard_facts()) >= min_facts=40` (`research/runners/multi_turn_agent.py:299-310`). The production
  tiny-demo brain every live build site constructs (`brain_chat_tui.py::_build_tiny_demo`) hears exactly 5
  baked-in facts -- **below the floor by construction**, confirmed by the SAME file's own comment
  (`brain_chat_tui.py:2081-2083`: "this tiny fixture's 5 facts are below build_referent_bias_from_experience's
  min_facts floor, so this is still a no-op here"). So on the harness's own `bc_b` probe,
  `agent._feat_compat_source` stays `None` and `MultiTurnAgent._resolve_biased` falls through to
  `_focus_bias_source` (a DIFFERENT mechanism `BRAIN_BIASED_COMPETITION_LESION` does not touch) --
  `BRAIN_BIASED_COMPETITION_LESION` is real, wired, and VERIFIED IN ISOLATION at seed 7
  (`research/findings/raw/_lbf_rows_conflict_kb/biased_competition_isolated_s7.json`: unlesioned
  `bias_target(['worm','rock'], 'eat')` = `'worm'`; lesioned = `None`; `tools.lab.lever` confirms the manipulation
  MOVED something). It is **not yet exercised by the shipped `bc_b` probe** and would read a false "not
  load-bearing" (hollow) if run
  through `measure_faculty("selective-attention-biased-competition", ...)` as-is -- exactly the
  lesion-that-cannot-bite false-negative `docs/FAILURE_GATE_MATRIX.md` / this file's own module docstring warns
  against. **Declared, not silently shipped**: `kind` stays `"proposed"` in `EXTRA_LESIONS`; the `note` records
  this exact gap. Next rung (not built here, out of scope for S14): either raise the tiny-demo's baked-in fact
  count past 40 before the probe (changes the shared default fixture -- needs its own review) or add a
  DRIVING-remap probe (mirroring `LB_EPISODIC_DRIVE_PROBE`'s own pattern) that first teaches >=40 facts on a
  fresh session, then asks `bc_b`.

### Read-time assertions

Each lesion is verified with `tools.lab.lever` (asserts the manipulation actually moved something; raises
`LeverError` if the flag-on and flag-off reads are identical) inside
`research/runners/lbf_rows/proposed_lesions_conflict_kb.py`'s smoke functions, and the lever's before/after
values are recorded in the smoke JSON artifact -- not just printed.

## Phase 2 — LB_CONFLICT_KB_PROBE (fixture-state, gnw-deliberation / value-driven-choice)

`FACULTY_LESIONS['gnw-deliberation']` and `['value-driven-choice']` are `kind="thin"`
(`onebrain_regression_battery.py:419-455`): both need a genuine >=2-distinct-patient (agent,action) conflict a
brain_chat-only conversational turn cannot construct (the default-ON reconsolidation organ rewrites a
contradicting assertion in place). The SAME comment names the construction that CAN reach it: "directly
constructing a composer with two KB rows, bypassing conversational teaching entirely" -- already built and
proven live in `research/runners/_value_choice_flip_soak.py::_build_chat` (`inner.hear("dog chase ball",
polarity="AFFIRM")` after the tiny-demo's own `dog->chase->cat` boot fact, giving `("dog","chase",["cat",
"ball"])`).

That fixture already runs and passes its own 6-seed soak
(`research/findings/raw/_value_choice_prodflip/soak_summary_6seed.json`: the ambiguity is constructed, the
committed patient varies with the value context, and the lesion reverts it — the mechanism this pre-registration
reuses, not builds from scratch).

`LB_CONFLICT_KB_PROBE=1` (default OFF) reuses that exact construction (import, not reimplementation) to confirm
the injected conflict is visible in the composer's own KB dump (>= 2 stored rows keyed `(dog, chase, *)` with
DISTINCT patients) and, where the optional `webapp.gnw_deliberation` / `value_choice_production_organ` installs
succeed, that the ambiguous question ("what does dog chase") reaches the deliberation/value-choice organs
instead of confabulating a random patient.

**Kind stays `"thin"` for both rows in `EXTRA_LESIONS`** (not the new label `"fixture-state"` the plan step names
in prose) because `research/runners/load_bearing_fraction.py::measure_faculty`'s own `kind in (...)` early-return
tuple is a literal this lane does not edit (the LBF ROW INTERFACE reserves `FACULTY_LESIONS`/`FACULTY_PROBES`
edits to AG-REG); `"thin"` is already a recognized, safe, out-of-numerator/denominator kind with the identical
practical effect (`measure_faculty` never spawns an arm for it). The `note` field is amended to record the
fixture-state reasoning and point at the standalone `LB_CONFLICT_KB_PROBE` artifact as the evidence that the
ambiguity IS directly constructible (just not through `brain_chat`), so a future rung that wires it through
`measure_faculty` has a working, cited construction to start from. This is the outcome the plan step's own
fallback names as acceptable: "If phase 2 is not done by 18:00, both rows stay thin with the reason recorded."

## Byte-identity (flags OFF)

Every edit in Phase 1 is a single `if _<x>_lesioned(): return <early-value>` line inserted before the FIRST
existing line of its target function/method, with **zero other lines touched**. `_<x>_lesioned()` returns
`False` whenever its env var is unset (confirmed by reading each helper: `os.environ.get(NAME)` is `None` ->
`return False`), so the early-return is never taken and every subsequent line runs exactly as before this
change -- a STRUCTURAL guarantee (the same reasoning `research/runners/biased_competition_prod.py`'s own
docstring already uses for its own default-anchor flag), checked here by diff review of each edit (`git diff` on
the four touched files shows only these five additive blocks, no other line moved) and additionally by an
empirical two-build determinism check (seed 7, all five flags unset, the full default `PROBE_TURNS[:10]` group)
recorded in the smoke artifact.

## AMENDMENT LOG

- **2026-09-24, filed after `smoke_biased_competition_isolated` (seed 7) completed, before the Phase 1 five-key
  `measure_faculty` smoke or the Phase 2 `LB_CONFLICT_KB_PROBE` had run.** The isolated biased-competition proof
  is now real data (cited above), not a plan; every other verdict in this document (Phase 1's four other rows,
  Phase 2, byte-identity) is unchanged and still governs runs that had not happened at filing.

- **2026-09-24, filed at session end, honest status of the remaining runs.** The shared box was under severe
  multi-lane RAM/CPU contention for this entire session (`tools/mem_ok.sh` repeatedly refused jobs down to 1 GB;
  system free memory hit 0 GB more than once). Two concrete effects, both closed or declared:
  1. A first `--byte-identity` attempt produced `byte_identical: true` from BOTH arms silently failing to build
     (`a_is_none`/`b_is_none` both `True`, so `None == None` read as a vacuous pass) -- a genuine instrument bug
     in `byte_identity_check`, now FIXED with `tools.lab.void_if` (a failed-build arm now reports
     `byte_identical: None`, never `True`). That first, invalid artifact was deleted, never committed.
  2. A second, correctly-guarded `--byte-identity` attempt was in flight (PID/log recorded in the session; not
     this document) when this pre-registration's own filing deadline arrived; the Phase 1 five-key
     `measure_faculty` smoke and the Phase 2 `LB_CONFLICT_KB_PROBE` were NOT attempted at all for the same
     contention reason. **No byte-identity, five-key smoke, or conflict-KB result is claimed in this document
     beyond the STRUCTURAL argument** (diff-reviewed: five single-line early-returns, each gated on a helper
     that reads False when unset) and the one isolated biased-competition result already cited above. Next
     rung: re-run `--smoke --conflict-kb --byte-identity` once local contention clears, or on the pool/AWS.

## Compute

Local only (no pool/AWS access from this lane's environment): `bash tools/mem_ok.sh <n> 4` then
`bash tools/memcap.sh <n> -- .venv/bin/python -m research.runners.lbf_rows.proposed_lesions_conflict_kb <mode>`,
`OMP_NUM_THREADS=1`, `SIM_BACKEND=numpy`, one brain-sized process at a time. The 6-seed gate for any row this
lane upgrades to `neural-lesion` is **not run here** (out of scope for S14; queued for B2b per the plan).
