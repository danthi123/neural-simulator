---
type: finding
status: go
date: 2026-09-04
mechanism: (1) MEASUREMENT — re-ran the project's own real-traffic instrument
  (`research/runners/_per_touchpoint_qwen_share_measure.py --phase open_ended`) against `webapp.server.brain_chat`
  now that both the recall-gate→LTM fix (`bbff50765`) and the linattn-mouth-flip (`4ea2ff74`, which sets
  `BRAIN_WKV_MOUTH_SCOPE=broad` by default) are live on `main`; (2) BUILD — a new, default-OFF, scope-limited flag
  `no_qwen_fallback_enabled()` / `BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK` in `webapp/open_ended_chat.py::answer_turn`
  that skips the literal one-shot Qwen forward pass on a brain-UNKNOWN (`known=False`) turn and produces the
  IDENTICAL fixed honest-abstain string `post_filter` would have reduced Qwen's own reply to anyway.
lane: language (own-voice mouth / retire the Qwen scaffold) + one-brain (substrate consolidation)
seeds: [42]
seed-waiver: a real-traffic measurement soak through the REAL `/api/brain-chat` entry point
  (`webapp.server.brain_chat`, in-process), reusing the project's own `_per_touchpoint_qwen_share_measure.py`
  instrument VERBATIM (unmodified) — the SAME seed-waiver precedent `research/findings/2026-09-04-recall-gate-
  reaches-real-ltm-GO.md` and `research/findings/2026-09-04-per-touchpoint-qwen-call-share.md` (commit 64fc4d5f)
  already established for this exact instrument: `seed=42` draws a reproducible sample of known topics and is
  the fixed internal generation seed every real turn already uses; a 6-seed repeat of a deterministic real-traffic
  comparison would reproduce the identical pass/fail pattern, not additional evidence. The NEW flag's OWN
  regression suite (`tests/test_no_qwen_fallback_flag.py`) is not a stochastic training run either — it is
  fast, deterministic, mocked-Qwen unit coverage (21/21 pass, <2s) exercising the flag's own branch logic
  directly, independent of any seed.
instrument: `research/runners/_per_touchpoint_qwen_share_measure.py --phase open_ended` (Surface B, REUSED
  VERBATIM, unmodified — diffed byte-identical against the committed copy before use), CPU-forced
  (`CUDA_VISIBLE_DEVICES=""`, `SIM_BACKEND=numpy`); plus a new `tests/test_no_qwen_fallback_flag.py` (21 tests,
  mocked Qwen, no GPU/heavy-model dependency) and a throwaway live end-to-end script (not committed — see
  Provenance) exercising the REAL warm Qwen faculty with the WKV mouth forced off, to confirm the flag's
  answer-preserving claim against a genuinely live Qwen call, not just the mocked unit tests.
runner: research/runners._per_touchpoint_qwen_share_measure ; tests/test_no_qwen_fallback_flag.py
external: Chen, Zaharia & Zou (2023) "FrugalGPT: How to Use Large Language Models While Reducing Cost and
  Improving Performance", arXiv:2305.05176, https://arxiv.org/abs/2305.05176 <!--derived--> — the LLM-cascade cost-routing
  precedent this finding's mechanism mirrors in the opposite direction: FrugalGPT escalates a query to a MORE
  expensive model only when a cheaper stage's answer is not good enough; this flag RETIRES the more expensive
  stage's call entirely once the moat-filtered answer is already provably equivalent (not merely cheaper) —
  recorded via `tools/record_external_search.sh`, lane-tagged, clearing `gates/deep_research_at_wall` for this
  finding's lane (20+ findings landed in this lane within the 3-day window).
verdict: GO. MEASUREMENT — the literal one-shot Qwen fallback on Surface B (`BRAIN_OPEN_ENDED=1`) now fires on
  0/15 forked turns of the project's own 16-probe battery (was 9/15 = 60% pre-flip, ALL on known=False traffic —
  research/findings/2026-09-04-per-touchpoint-qwen-call-share.md, commit 64fc4d5f): the linattn flip's
  `BRAIN_WKV_MOUTH_SCOPE=broad` default already routes every known=False turn in this battery to the WKV mouth's
  own free generation instead. Every known=False turn (11/11) converges to the IDENTICAL fixed honest-abstain
  string regardless of generator, confirming the moat is generator-agnostic. BUILD — the flag that closes the
  narrower residual (WKV mouth disabled/out-of-scope/exception) is implemented, default-OFF, and verified
  byte-identical-off + answer-preserving-on + scoped away from known=True traffic, by 21/21 new tests + 45/45
  pre-existing sibling tests, all passing, plus a live real-Qwen end-to-end check (see SS4).
artifacts:
  - research/findings/raw/_per_touchpoint_qwen_share_open_ended_AFTER_flip_and_recallgate.json (this session's
    re-measurement; complete: true, 16/16 rows, peak RSS 3954.7 MB, wall 148.4 s)
  - research/findings/raw/_no_qwen_fallback_flag_verify.json (this session's live end-to-end flag-behavior check
    through the real warm Qwen faculty)
  - tests/test_no_qwen_fallback_flag.py (21 new regression tests, all passing)
---

# One-brain Stage-1: the Qwen one-shot fallback is already ~0% live-share post-flip; the residual is now flagged closed

**One-brain Stage-1** (`research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS4: "retire the
open-ended one-shot Qwen fallback"), scoped and closed. The per-touchpoint measurement that found Qwen firing
on 60% of Surface-B turns (`research/findings/2026-09-04-per-touchpoint-qwen-call-share.md`, commit 64fc4d5f)
predates two fixes that landed the SAME day: the recall-gate→LTM fix (`bbff50765`, 19:40) and the linattn-mouth
production flip (`4ea2ff74`, 19:22, which sets `BRAIN_WKV_MOUTH_SCOPE=broad` by default). This finding
re-measures Surface B with both live, finds the Qwen share has already collapsed to 0% on the project's own
probe battery, and implements + verifies a flagged, scoped retirement of the narrower residual that measurement
discloses.

## 1. The re-measurement: Qwen's one-shot share, before vs after

Same instrument, same 16-probe battery, same `seed=42` sample (`angora_turkey`,
`college_for_interdisciplinary_studies`, `imperial_roman`, `l_quipe_de_france`), run through the real
`webapp.server.brain_chat` entry point, CPU-forced, `BRAIN_OPEN_ENDED=1` (Surface B):

| | BEFORE (commit 64fc4d5f, pre-flip) | AFTER (this session, post-flip + recall-gate fix) |
|---|---|---|
| forked turns (excl. self-initiated) | 15 | 15 |
| `generator == "qwen"` | 9 (60.0%) | **0 (0.0%)** |
| `generator == "wkv_mouth"` | 2 (13.3%) | 11 (73.3%) |
| `generator == "spiking_clause"` | 4 (26.7%) | 4 (26.7%) |
| known=True turns | 4/4 all `spiking_clause` | 4/4 all `spiking_clause` (unchanged) |
| known=False turns | 11 total: 9 qwen + 2 wkv_mouth | 11 total: **0 qwen + 11 wkv_mouth** |

The mechanism: `webapp/wkv_mouth_generator.py::scope_mode()` now defaults to `"broad"` under the new default
`recurrence_mode()=="linattn"` (`_SCOPE_ENV` comment, `wkv_mouth_generator.py:246-272`), and `in_vocab_scope()`
under `scope_mode()=="broad"` "bypasses all of the above and returns True unconditionally" (`in_vocab_scope`'s
own docstring, `:734-739`). Every prompt is now admitted to the WKV mouth's own try-block in `answer_turn`
(`webapp/open_ended_chat.py:633-688`), so `wkv_attempted` is True for essentially every turn — and whenever it
is, `if wkv_used or fact_clause_used: pass` (`:716-717`) skips the literal Qwen dispatch below entirely. This is
a SIDE EFFECT of the already-landed, owner-authorized flip, not a result of any change this finding makes — the
flip's own design doc (`research/findings/2026-09-03-linattn-production-mouth-wiring-DESIGN.md`) did not name
"the Qwen one-shot share collapses" as a design goal; it fell out of routing every prompt through the mouth's
own free generation. The recall-gate fix (`bbff50765`) touches a DIFFERENT surface entirely (Surface A's
`RichAnswerComposer._direct_fact`, not `webapp/open_ended_chat.py`'s own direct `facts.json` retrieval) and does
not mechanically change Surface B's `known` computation or routing — confirmed by reading both diffs (Provenance)
and by this measurement's own known=True row (4/4 `spiking_clause`, IDENTICAL count and mechanism before and
after the recall-gate fix, because Surface B's `retrieve()` already read `facts.json` directly and never had
the bug Surface A's fix closed).

## 2. Every known=False reply converges to the SAME fixed string, regardless of generator

The full AFTER artifact (`research/findings/raw/_per_touchpoint_qwen_share_open_ended_AFTER_flip_and_recallgate.json`)
shows all 11 known=False turns — `known_multi_sentence`, `known_followup`, `unknown` x2, `dangerous` x2,
`open_ended_opinion` x3, `greeting_social` x2 — produce the IDENTICAL template with only the extracted topic
substituted:

> *"I'm not sure about {topic} — I don't have anything about it in what I've actually learned, so I'd only be
> guessing."*

verbatim for `zorplaxian`, `flibberwock`, `paris`, `python`, `music`, `hello`, `how are you`, etc. This is
`_open_ended_verify_postfilter_derisk.post_filter`'s unknown-topic branch (imported unmodified as
`_base_post_filter` — `webapp/open_ended_chat.py:174,229`): it keeps only hedge/uncertainty sentences from
`raw`, and prepends this exact fixed string whenever none survive. **The SAME convergence held BEFORE the flip**
(the root-cause finding's own SS4: "on every `known=False` turn, the FINAL answer is the SAME fixed
honest-hedge template", 9/9 sampled Qwen replies) — this measurement confirms it holds identically for the
WKV mouth's own free generation, 11/11. The fixed string is a property of the post-filter's hedge/abstain logic,
not of which generator wrote the discarded raw text.

## 3. The build: a scoped, flagged retirement of the residual

Given (1) and (2), the literal Qwen one-shot branch (`webapp/open_ended_chat.py`'s final `else: raw, secs =
gen.generate(...)`) is reached today only when the WKV mouth genuinely does not cover a turn — disabled
(`BRAIN_OPEN_ENDED_WKV_MOUTH=0`), reverted to narrow vocab scope (`BRAIN_WKV_MOUTH_RECURRENCE=ssm` without also
setting `BRAIN_WKV_MOUTH_SCOPE=broad`), or a genuine exception inside `_WKV.generate()`. On a `known=False` turn,
paying for that Qwen forward pass only to have `post_filter` discard it and substitute the fixed string is pure
waste — the FrugalGPT cascade precedent (Chen, Zaharia & Zou 2023, arXiv:2305.05176 <!--derived-->) names this exact category of
saving (route away from the expensive stage once a cheaper path is provably as good), applied here in the
retirement direction rather than the escalation direction the paper studies.

`no_qwen_fallback_enabled()` (`BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK`, default OFF) adds one `elif` branch to
`answer_turn`'s final dispatch, reached only when `not known` and the WKV/fact-clause paths above it did not
already produce a reply:

```python
elif (not known) and no_qwen_fallback_enabled():
    raw, secs = "", 0.0
    generator_name = "no_qwen_fallback"
else:
    raw, secs = gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)
```

`raw=""` reproduces the fixed string exactly: `_sentences("")` is `[]`, so `hedges=[]`, `filtered=""`,
`uncertainty_signaled("")` is False, and the identical prefix is prepended — byte-for-byte the same visible
answer a live Qwen (or wkv_mouth) call would have produced on this battery, at zero forward-pass cost.
`generator` reports `"no_qwen_fallback"`, never `"qwen"` — deliberately avoiding a repeat of the exact mislabel
the 2026-09-04 generator-trace fix (commit c08ce8fb3) had to correct.

**Scoped to `known=False` only, on purpose.** A `known=True` turn that reaches this same branch (the WKV mouth's
`sentence_facts` path AND the separate fact-clause fallback both declined — e.g. a topic whose relation
`RELATION_LEXICON` does not cover) is UNCHANGED: Qwen stays the honest residual there. That case was not
measured by this battery (`RELATION_LEXICON` covered 34/34 live relation types in the sampled store — the
recall-gate finding's own SS3), so retiring Qwen on it is not validated safe — it would trade a possibly-correct,
moat-checked multi-sentence paragraph for the terser `_empty_known_fallback` string, a real richness loss this
flag does not force.

## 4. Verification

**Fast, deterministic, mocked-Qwen regression (`tests/test_no_qwen_fallback_flag.py`, new, 21 tests, <2s, no
GPU/heavy-model dependency)**, mirroring `tests/test_generator_trace_matches_producer.py`'s own
`_FakeQwenGenerator` isolation strategy:
- Flag OFF (default, and explicit `=0`): an out-of-vocab + unknown-topic turn calls the fake Qwen exactly once,
  traced `"qwen"` — the pre-existing path, unchanged.
- Flag ON: the IDENTICAL turn calls the fake Qwen **zero** times (`fake.calls == 0` — the forward pass is
  genuinely skipped, not merely discarded), traces `"no_qwen_fallback"`, `raw == ""`.
- **The load-bearing equivalence**: two independent `answer_turn` calls (flag off with a real fake-Qwen call vs
  flag on with the call skipped) produce `answer` values that are asserted equal — `res_on["answer"] ==
  res_off["answer"]` passes, because the fake reply's own text carries no hedge language and gets reduced to
  the identical fixed string anyway.
- **Scope guard**: a known=True turn on the same out-of-vocab, uncovered-relation branch still calls the fake
  Qwen once and traces `"qwen"` with the flag ON — confirming the flag never touches known=True traffic.
- Flag-parsing contract: the standard truthy/falsy set (`1/true/on/yes` vs `0/false/off/no/unset/garbage`,
  case-insensitive) matches every other flag in this module.

All 21 pass. The two sibling test files this change could plausibly regress
(`tests/test_generator_trace_matches_producer.py`, 27 tests; `tests/test_wkv_mouth_bpe_decode_wiring.py` +
`tests/test_wkv_invocab_scope_leadin_fix.py`, combined into the same run, 45 total) still pass unchanged.

**Live end-to-end confirmation through the REAL warm Qwen faculty** (not committed — a throwaway script, see
Provenance), CPU-forced, WKV mouth forced OFF (`BRAIN_OPEN_ENDED_WKV_MOUTH=0`) to force the residual path
deterministically. Three known=False probes (`"Tell me about zorplaxian."`, `"Tell me about paris."`, `"hello"`),
each run once with the flag OFF (a genuine Qwen forward pass, `generator="qwen"`) and once with the flag ON
(`generator="no_qwen_fallback"`, zero forward passes): **`answer` was byte-identical between the two arms on
all three probes** — the same fixed string a real Qwen call produces is exactly what the retirement produces,
confirming the pytest suite's mocked-Qwen claim against the genuine off-bridge model, not just a stand-in.
A fourth check — the SAME known=True probe (`angora_turkey`) with the WKV mouth still forced off and the flag
ON — traced `generator="spiking_clause"`, not `"qwen"`: on the real `wikidata_core_15k` store, the INDEPENDENT
fact-clause fallback (`fact_clause_fallback_enabled()`, gated on `known`, not on the WKV mouth at all) already
covers this topic's relation regardless of WKV-mouth state, so this specific probe does not exercise the
known=True→Qwen edge case at all (a real-store confirmation that `RELATION_LEXICON`'s coverage is what keeps
known=True traffic off Qwen in practice, independent of this flag). The scope guard itself — a known=True turn
on a relation `RELATION_LEXICON` does NOT cover, which DOES reach the literal Qwen branch — is proven instead
by `tests/test_no_qwen_fallback_flag.py::test_known_topic_on_the_same_branch_is_unaffected_by_the_flag`
(SS4 above), which constructs that exact uncovered-relation condition directly. A fifth check confirmed the
flag is inert under the shipped default (WKV mouth ON, broad scope): the SAME unknown-topic probe still traced
`generator="wkv_mouth"` with the flag ON, matching SS1/SS2's full-battery numbers.

## 5. Honest residuals

- **This flag's own branch is unreached on the tested battery today.** Under the current shipped defaults (WKV
  mouth ON, broad scope), `no_qwen_fallback_enabled()` being ON or OFF makes zero observable difference on this
  16-probe battery — the branch it guards is simply never reached there. Its practical value is a safety net for
  configurations where the WKV mouth genuinely does not cover a known=False turn (explicitly disabled, reverted
  to narrow vocab scope, or a genuine exception) — not a change to today's live traffic pattern.
- **known=True traffic reaching the literal Qwen branch is a real, unclosed residual, left alone on purpose.**
  Retiring Qwen there was not validated safe by this task's own measurement (the sampled store's relations were
  all lexicon-covered) and would trade richer content for a terser fallback — named, not forced.
- **The `BRAIN_WKV_MOUTH_SCOPE=broad` threshold itself remains an un-measured "admit everything" gate** (the
  roadmap's own de-risk 1, `research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md` SS3) — this
  finding measures its DOWNSTREAM effect on Qwen call share, not its own coverage/confidence calibration, which
  stays a separate, already-named next step.
- **N=16 probes, one seed-drawn topic sample, matching the precedent's own disclosed scope** — a larger,
  independently-sampled battery is the natural next rung, not a 6-seed repeat of this deterministic comparison
  (see seed-waiver).
- **`BRAIN_OPEN_ENDED` is still not the production default** — this closes a Stage-1 residual on a channel that
  is itself still opt-in; Stage 3 (reconciling Surface A/B) and the roadmap's genuine owner-forks are unaffected.

## Provenance

Shipped code read/edited this session (2026-09-04): `webapp/open_ended_chat.py` (`answer_turn`'s full dispatch
`:622-753`, the new `no_qwen_fallback_enabled` `:456-489`, the flag-family docstrings), `webapp/server.py`
(the `generator` trace comment `:4718-4727`, read + updated, no logic change), `webapp/wkv_mouth_generator.py`
(`recurrence_mode`/`scope_mode`/`in_vocab_scope` `:226-272,718-747`, read only, confirming the broad-scope
mechanism), `research/runners/_open_ended_verify_postfilter_derisk.py` (`post_filter`'s unknown-topic branch
`:44-57`, read only, confirming the fixed-string mechanism this flag reproduces).

Instrument reuse: `research/runners/_per_touchpoint_qwen_share_measure.py` used VERBATIM, unmodified (diffed
against the committed copy before use — no changes). A filesystem-only, git-untracked dependency
(`data/corpus/tinystories.txt`, gitignored, required by `SpikingQwenFaculty.held_out_text()`) was symlinked from
the primary checkout into this worktree to satisfy the warm-up turn — the SAME disclosed convenience
`research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md` and the root-cause finding both used; not a
code or git-tracked change.

New tests: `tests/test_no_qwen_fallback_flag.py` (21 tests, `CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy
.venv/bin/python -m pytest tests/test_no_qwen_fallback_flag.py -q` → `21 passed`). Regression:
`tests/test_generator_trace_matches_producer.py tests/test_wkv_mouth_bpe_decode_wiring.py
tests/test_wkv_invocab_scope_leadin_fix.py` → `45 passed`, run once before and once after this session's edits.

External search recorded via `tools/record_external_search.sh`, lane-tagged
`"language (own-voice mouth / retire the Qwen scaffold) + one-brain (substrate consolidation)"`, clearing
`gates/deep_research_at_wall` for this finding (20+ findings landed in this lane within the 3-day window).

Builds on: `research/findings/2026-09-04-per-touchpoint-qwen-call-share.md` (commit 64fc4d5f, the root-cause
measurement), `research/findings/2026-09-04-recall-gate-reaches-real-ltm-GO.md` (the recall-gate fix landed the
same day), `research/findings/2026-09-04-generator-trace-mislabel-fix.md` (the `generator` trace field this
finding's own new value follows the same discipline as), `research/findings/2026-09-03-one-brain-mouth-
integration-ROADMAP.md` (Stage 1, the mission this finding closes), `ac58b81e6`/`4ea2ff742` (the linattn
production flip whose `BRAIN_WKV_MOUTH_SCOPE=broad` default is the mechanism behind SS1's headline number).
