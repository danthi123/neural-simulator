---
type: finding
status: wired
date: 2026-09-04
mechanism: Two pre-existing Stage-2 Touchpoint-A blockers fixed, plus one independent, newly-discovered
  test-comparison-validity defect fixed alongside them. (1) `RichAnswerComposer._render_one_verified`'s
  Touchpoint-A branch (`research/runners/rich_answer_composer.py`, flag `BRAIN_TOUCHPOINT_A_FACT_CLAUSE`,
  default OFF) now wraps its `webapp.wkv_mouth_generator.render_fact_sentence` call in
  `webapp.wkv_mouth_generator._RngIsolation.run`, per that function's own documented contract. (2)
  `research/runners/_touchpoint_a_fact_clause_derisk.py::compute_go_gate`'s `content_preserved` check is
  redefined from "identical per-row `supporting_facts`" to "no fact is lost" (a battery-wide UNION superset
  check), an OWNER-DECIDED relaxation. (3) The SAME runner's `run_pass` now resets
  `research.runners._affect_marker_wta_derisk`'s process-global `AffectMarkerWTA` reader cache
  (`reset_readers()`) before each of its two passes, closing a newly-discovered, independent cross-pass state
  leak through that SEPARATE module (see Provenance + FAILURE_LOG.md).
lane: language (own-voice mouth / retire the Qwen scaffold) + one-brain (substrate consolidation)
seeds: [42]
seed-waiver: this runner's three STRUCTURAL invariants (`scope_untouched`, `content_preserved`,
  `flag_off_inert`) are deterministic CORRECTNESS properties of a fixed probe battery against a fixed store
  sample (a scope leak either exists or it does not; a fact is either lost or it is not) -- not a statistical
  efficacy claim that could look better on a favorable seed. This is the SAME seed-waiver precedent
  `research/findings/2026-09-04-onebrain-stage2-touchpoint-a-fact-clause-BUILD-AHEAD.md`,
  `research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md`, and
  `research/findings/2026-09-04-onebrain-stage1-qwen-fallback-retire-GO.md` already established for this
  exact instrument family, run here at the SAME `--seed 42` the task's own specified command uses (no
  `--seed` override). The READINESS signal (`touchpoint_a_render_calls_delta`,
  `fact_clause_engaged_on_known_rows`) is informational, not a generalization claim, consistent with the
  module's own docstring. A different store sample / additional seeds would strengthen confidence before any
  DEFAULT-ON promotion decision (not attempted here, out of this task's scope) but would not change whether
  the underlying code-level contract fixes (RNG isolation; a cache reset between two same-seed comparison
  passes) are correct -- those are seed-independent properties of the code, not measurements that could get
  lucky.
instrument: research/runners/_touchpoint_a_fact_clause_derisk.py --n-known 4 (the full battery, not the
  earlier --smoke), CPU-forced (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`). Also re-verified
  tests/test_touchpoint_a_fact_clause_flag.py (20 tests, mocked fixtures) still pass unchanged after fix (1).
runner: research.runners._touchpoint_a_fact_clause_derisk
external: none searched this session -- this is a bug-fix + an owner-directed gate redefinition on an
  ALREADY-SCOPED, already-diagnosed pair of blockers (FAILURE_LOG.md 2026-09-04 rows 112/113,
  `research/coordination/build_ahead_ready.md` item #3), not a new mechanism-lever against a wall; the third
  fix (the WTA cache reset) reuses an EXISTING, already-validated pattern from this same repo
  (`research/runners/_affect_marker_wta_verify.py`'s own `reset_readers()` calls between its own test
  conditions), not a fresh lever needing literature research.
verdict: STRUCTURAL GO on the full n=4 battery. `research/findings/raw/_touchpoint_a_fact_clause_full.json`
  (`go_gate`): `structural_checks_passed=true` -- `scope_untouched=true` (0 scope_problems, was 1),
  `content_preserved=true` (0 content_problems under the redefined check, was 2), `flag_off_inert=true` (0
  flag_off_problems, unchanged -- this one never failed). READINESS: `touchpoint_a_render_calls_delta=-19`
  (19 fewer Qwen/template renders across the 4 known topics with the flag on) and
  `fact_clause_engaged_on_known_rows=4` (all 4 known topics engaged the fact-clause path at least once).
  `facts_off_union_count=17` and `facts_on_union_count=17` are EQUAL on this store sample
  (`facts_rescued=[]`) -- the specific idx=3/idx=5 pair FAILURE_LOG.md row 113 named (a fact told at an
  earlier turn instead of a later one) nets to the SAME total reach here, not a net NEW rescue on this
  particular sample; the check is built to also pass (and to report a non-empty `facts_rescued`) on a store
  where Touchpoint-A recovers a fact the old renderer could never verify-render at all. This is a STRUCTURAL
  and INSTRUMENT-VALIDITY result (the de-risk's own three anti-cheats now hold, and the comparison itself is
  now apples-to-apples), NOT a claim that `BRAIN_TOUCHPOINT_A_FACT_CLAUSE` is ready for a DEFAULT-ON flip --
  the flag remains default OFF, `wired` (reachable, not `on-by-default`) per docs/TERMS.md, and the
  production ledger is untouched by this task (see Honest residuals).
artifacts:
  - research/findings/raw/_touchpoint_a_fact_clause_full.json (this session's OWN re-run, n=4 full battery,
    seed 42, complete: true, 16 probes x 2 passes; `go_gate.structural_checks_passed=true`)
---

# One-brain Stage-2 Touchpoint-A: the two pre-existing blockers are fixed, plus one discovered along the way

Follow-on to `research/findings/2026-09-04-onebrain-stage2-touchpoint-a-fact-clause-BUILD-AHEAD.md` (the
flag + runner + GO-gate build-ahead) and the two structural failures it left open
(`research/FAILURE_LOG.md` 2026-09-04 rows 112/113, `research/coordination/build_ahead_ready.md` item #3).
This task's job: fix both, relax `content_preserved` per an explicit owner decision, and re-run the full
`--n-known 4` battery. A THIRD, independent, previously-undiscovered defect surfaced while verifying the
first fix did not by itself close the gate -- it is fixed here too, for this runner's own comparison
validity, with the production-level version of that defect flagged as a separate follow-up (not fixed here).

## 1. Fix #1 -- the RNG-isolation scope leak (FAILURE_LOG row 112)

`RichAnswerComposer._render_one_verified`'s Touchpoint-A branch called
`webapp.wkv_mouth_generator.render_fact_sentence` directly. That function's own docstring requires being
called "from inside `_RngIsolation.run`" because `SpikingClauseProducer.__init__` (on a cache miss) builds a
real `SimulationBridge` that reseeds the process-global RNGs (the repo's well-known "#77 footgun"). Fixed by
wrapping the call exactly the way `webapp/wkv_mouth_generator.py`'s own `generate()` wraps its equivalent
call (`text = _RNG.run(seed, _run)`):

```python
from webapp.wkv_mouth_generator import _RNG, render_fact_sentence
fc_seed = int(getattr(self.chat.inner, "seed", 42))
fc_surface = _RNG.run(fc_seed, lambda: render_fact_sentence([svo], seed=fc_seed))
```

`tests/test_touchpoint_a_fact_clause_flag.py`'s full 20-test suite passes unchanged (the wrap is transparent
to every existing mock, since `render_fact_sentence` is still resolved fresh from the module attribute each
call and `_RNG.run` simply calls its argument function and returns its result, propagating any exception).

**This fix is necessary and correct, but it was NOT sufficient by itself** -- see Section 3.

## 2. Fix #2 -- `content_preserved` redefined to "no fact is lost" (FAILURE_LOG row 113, OWNER DECISION)

Touchpoint-A deliberately rescues a fact the pre-existing Qwen/template renderer failed to verify-render.
Recovering MORE grounded facts is the goal, not a regression -- so a per-row `supporting_facts` EQUALITY
check can never pass while the mechanism does its job. The owner decided (2026-09-04) this is a feature, and
directed the gate be relaxed to a superset ("no fact lost") check. `compute_go_gate` now computes the UNION
of `supporting_facts` across every known-topic row in the WHOLE battery for each pass (`facts_off_all`,
`facts_on_all` -- battery-wide, not per-row, because a `known_followup` row's own remaining reach is only
meaningful relative to everything already told earlier in the SAME un-reset session) and requires
`facts_off_all` to be a SUBSET of `facts_on_all`. A fact present only in `facts_on_all` (a genuine rescue,
reported as `facts_rescued`) no longer fails the gate; only a fact `facts_off_all` has that `facts_on_all`
lacks would. The redefinition and its rationale are documented inline in both the module docstring and
`compute_go_gate`'s own docstring, citing this owner decision and FAILURE_LOG.md row 113 by name.

## 3. The discovered THIRD defect -- a cross-pass process-global cache in an unrelated module

Fix #1 alone did not close `scope_untouched`: re-running the n=4 battery with ONLY the RNG-isolation fix in
place reproduced the byte-identical scope failure the original FAILURE_LOG row 112 measured (the `unknown`
probe at idx=6 still gained an unwarranted affective lead-in word flag-on vs flag-off).

Root-caused by direct empirical diagnosis (a standalone 7-probe two-pass trace of the real `affect_drives`
response field at every turn, not committed as a separate artifact -- see FAILURE_LOG.md's own row for the
full numeric trace): the continuous felt-mood value the graded-affect ladder computed was IDENTICAL between
the two passes at the diverging turn, but the DISCRETE decision of whether to attach an affective lead-in
word ("Sure -- ", etc.) differed. That discrete decision is made by
`research.runners._affect_marker_wta_derisk.AffectMarkerWTA` (board #86's spiking lateral-inhibition WTA
circuit, wired into `webapp/affect_drives_chat.py::expression_lead`, default-ON in production). Its reader is
cached by `get_reader(seed)` in a MODULE-LEVEL `_READERS` dict keyed only by raw seed value -- not by session.
Both of the de-risk runner's OFF and ON passes build their session at the same seed, so this ONE process-warm
reader was SHARED across the two independently-compared passes, and its per-read washout was not sufficient
to make an identical continuous input always yield the same discrete winner: the outcome depended on how many
times the shared reader had ALREADY been called earlier in the process, not on the mood value itself.

This is a genuine, independent, PRE-EXISTING defect -- present on `main` before this task touched anything,
in a module this task did not otherwise modify -- not something Fix #1 could have addressed (it lives
entirely in a different file, reached through a completely separate call chain: the incoming message's
appraisal, not anything Touchpoint-A renders). It is also, as far as this session could determine, a LIVE
PRODUCTION defect: `webapp/server.py`'s own call site (`_ADC.observe_turn(chat, msg)`, under
`_AFFECT_DRIVES_DEFAULT_ON=True`) never overrides `seed` either, so the same shared reader singleton is used
by every concurrent or sequential live chat session -- meaning, in principle, one conversation's affective
lead-in word choice can be silently influenced by another (or an earlier, unrelated) session's own turns. This
is a SURFACE-TONE-only leak (which of Wonderful/Gladly/Sure/Hm/Honestly/Frankly leads a reply); it does not
touch conversation content, the honesty moat, or which facts any session can see.

**Fixed here, for this runner's own comparison validity only:** `_touchpoint_a_fact_clause_derisk.py`'s
`run_pass` now calls a new helper, `_reset_cross_process_affect_wta_cache()`, before each pass, clearing
`_affect_marker_wta_derisk`'s `_READERS` singleton via its own existing public `reset_readers()` function --
the SAME helper `research/runners/_affect_marker_wta_verify.py` (board #86's own verify suite) already calls
between its own test conditions, so this is an established, sanctioned idiom in this exact module family, not
a novel workaround. This is a test-harness-level isolation fix: no production code, default, or flag is
touched by it.

**NOT fixed here: the production-level defect.** Making `AffectMarkerWTA`/`get_reader` properly session-scoped
(or giving it a full RNG/state isolation wrapper, mirroring `webapp.wkv_mouth_generator._RngIsolation` /
`affect_drives_chat.AffectDrivesWorkspace._isolated`, both of which this same file's sibling mechanisms
already use) is default-ON-production-facing work with its own blast radius and its own verification
obligations (board #86's lesion/shuffle anti-cheats must be re-run against whichever fix is chosen, so the
existing GO is not silently broken). That is out of scope for this Touchpoint-A de-risk and is tracked as a
separate follow-up task, not chased here. See FAILURE_LOG.md's new 2026-09-04 row for the full write-up and
candidate fixes.

## 4. The full n=4 battery, with all three fixes in place

`research/findings/raw/_touchpoint_a_fact_clause_full.json` (seed 42, 16 probes x 2 passes, `complete: true`):

| check | before this task | after Fix #1 alone | after all 3 fixes |
|---|---|---|---|
| `scope_untouched` | false (1 problem) | false (byte-identical 1 problem) | **true** (0 problems) |
| `content_preserved` | false (2 problems) | (not re-checked in isolation) | **true** (0 problems) |
| `flag_off_inert` | true | true | true |
| `structural_checks_passed` | **false** | **false** | **true** |

Readiness (informational): `touchpoint_a_render_calls_delta=-19`, `fact_clause_engaged_on_known_rows=4`
(engaged on all 4 sampled known topics: `angora_turkey`, `college_for_interdisciplinary_studies`,
`imperial_roman`, `l_quipe_de_france`), `facts_off_union_count=17`, `facts_on_union_count=17`,
`facts_rescued=[]` on this particular store sample.

## 5. Is Touchpoint-A promotable?

The full n=4 GO battery -- the actual gate this task exists to satisfy -- now passes cleanly on every
structural invariant, with the comparison itself made valid by Fix #3. Read strictly, this is the correctness
+ instrument-validity result the task asked for: the scope leak is closed, the content-preservation
definition matches the owner's stated intent, and the readiness signal (`-19` render calls, 4/4 known topics
engaged) is a genuinely positive, not-vacuous signal for retiring Touchpoint-A's live-Qwen recall share.

It is NOT, by itself, a claim that flipping `BRAIN_TOUCHPOINT_A_FACT_CLAUSE` to default-ON is ready today.
Per `docs/TERMS.md`, the flag is `wired` (reachable from the real `/api/brain-chat` endpoint on a request
that sets it) but not `on-by-default`, and nothing here changes that -- the ledger
(`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`) is deliberately untouched by this task, matching the predecessor
BUILD-AHEAD finding's own stated discipline. A genuine promotion decision would additionally want: (a) this
same battery run against more than one probe-set sample / seed, since today's single run is a structural
(deterministic-correctness) check, not a statistical efficacy claim, per the seed-waiver above; (b) the
production-level fix for the Section 3 defect, since an un-fixed cross-session affect-marker leak is an
orthogonal but real correctness gap in the SAME production surface a promotion would rely on; and (c) the
batched sibling render path (`_render_paragraph_batched`, `BRAIN_RICH_BATCH_RENDER=1`) remains entirely
untouched by Fix #1 -- production runs the sequential path by default, so this is a disclosed scope
boundary, not a hidden gap, but a full promotion would need it addressed too.

## 6. Honest residuals

- **The production-level WTA cross-session cache defect (Section 3) is real, live, and unfixed at the
  source.** Flagged as a follow-up task; not chased here (own blast radius, own verification obligations).
- **Single-seed, single-store-sample.** See the seed-waiver above for why this is adequate for the
  STRUCTURAL claim this task asked for, and what would strengthen a PROMOTION claim beyond it.
- **`facts_rescued=[]` on this run** -- the specific rescue FAILURE_LOG row 113 named nets to the same total
  reach on this store sample (a fact told earlier instead of later), not a demonstrated net gain in
  knowledge surfaced. The mechanism is built to report a genuine gain when one occurs; none happened to occur
  in this particular 4-topic sample.
  <!--derived-->
- **The batched render path (`_render_paragraph_batched`) is untouched**, as disclosed in the predecessor
  BUILD-AHEAD finding -- production's default `BRAIN_RICH_BATCH_RENDER=0` means this does not affect the
  live default today, but it is a real scope boundary for whoever extends Fix #1.
- **The ledger is untouched by this task, on purpose** -- consistent with the predecessor finding's own
  stated discipline (moving a `retire_status` before a genuine default-flip decision would be exactly the
  "closed capability whose production default still runs the host shortcut" misuse `docs/TERMS.md` warns
  against).

## Provenance

Shipped code this session: `research/runners/rich_answer_composer.py` (the `_RNG`/`_RngIsolation.run` wrap,
Fix #1, ~8 new lines including comments); `research/runners/_touchpoint_a_fact_clause_derisk.py`
(`compute_go_gate`'s redefinition, Fix #2; `_reset_cross_process_affect_wta_cache` + its call from
`run_pass`, Fix #3; plus updated module-docstring criteria text for both). Commit `59f5c825b`. Read, not
modified: `webapp/wkv_mouth_generator.py` (`_RngIsolation`, `render_fact_sentence`, `generate()`'s own
`_RNG.run(seed, _run)` pattern), `webapp/affect_drives_chat.py` (`AffectDrivesWorkspace`, `_LEAD_WORD`,
`expression_lead`, `observe_turn`), `research/runners/_affect_marker_wta_derisk.py` (`AffectMarkerWTA`,
`get_reader`, `reset_readers`, `_build_bridge`), `research/runners/_affect_marker_wta_verify.py` (confirming
`reset_readers()` is an already-established idiom in this module family), `webapp/server.py` (confirming
`_ADC.observe_turn(chat, msg)` never overrides `seed`, and `_AFFECT_DRIVES_DEFAULT_ON=True`).

Tests: `CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest
tests/test_touchpoint_a_fact_clause_flag.py -q` -> `20 passed`, unchanged after Fix #1.

Full battery: `CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -u -m
research.runners._touchpoint_a_fact_clause_derisk --n-known 4 --out
research/findings/raw/_touchpoint_a_fact_clause_full.json` -> exit 0,
`go_gate.structural_checks_passed=true`. Run twice this session: once with Fix #1 alone (still failed
`scope_untouched`, confirming it was necessary but not sufficient -- that intermediate artifact was
overwritten by the final run and is not separately retained), and once with all three fixes (the artifact
this finding cites).

`research/FAILURE_LOG.md`: rows 112 and 113 marked resolved with this commit; a new 2026-09-04 row records
the Section 3 discovery, its fix here, and the production-level follow-up.

Builds on: `research/findings/2026-09-04-onebrain-stage2-touchpoint-a-fact-clause-BUILD-AHEAD.md` (the
flag/runner/gate build-ahead this task closes out), `research/findings/2026-08-28-affect-marker-spiking-wta-derisk.md`
(board #86, the WTA mechanism Section 3's defect lives in -- that finding's own 6-seed GO is UNAFFECTED by
this discovery, since it never exercised two independently-compared conditions at the same seed in one
process the way this de-risk runner does).
