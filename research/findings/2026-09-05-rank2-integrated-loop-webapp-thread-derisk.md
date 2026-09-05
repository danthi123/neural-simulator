---
type: finding
status: contributing
date: 2026-09-05
mechanism: scaffold-retirement de-risk (backlog rank 2) — thread `integrated_loop` (OneBrainComposer's spiking
  K-way cue-match SEQUENCER) through webapp/server.py -> brain_chat_tui / developed_brain_io -> MultiTurnAgent ->
  BrainConversationalAgent -> OneBrainComposer, so the production chat construction can opt the (agent, action)
  cue-match SELECTION onto the substrate instead of the host first-match `_scan`, both flag-gated default OFF
lane: integration-first (WIRING BACKLOG rank-2)
integration_faculty: content-selection
artifacts:
  - research/runners/_rank2_integrated_loop_webapp_thread_derisk.py
  - tests/test_rank2_integrated_loop_thread.py
  - research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partA.json
  # ^ Part A's corrected (post-fix) re-run: GO, 6/6 checks (see "PART A" below). Part B (production-scale,
  # GPU, 6 seeds) was still in flight at this commit and is NOT cited here (gates/claim_check refuses a
  # citation of a file not yet on disk); added in the follow-up commit that reports its verdict.
verification: |
  SIM_BACKEND=numpy python -u -m research.runners._rank2_integrated_loop_webapp_thread_derisk --skip-production \
      --out research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partA.json
  SIM_BACKEND=cupy  python -u -m research.runners._rank2_integrated_loop_webapp_thread_derisk --skip-mechanical
  # pass --out to choose Part B's result path; not spelled out here until that file exists on disk --
  # gates/claim_check refuses a citation of a .json path with no file on disk
  SIM_BACKEND=numpy python -m pytest tests/test_rank2_integrated_loop_thread.py -q
---

# Rank-2 de-risk: thread `integrated_loop=True` through webapp/server.py -> brain_chat_tui -> OneBrainComposer

## The residual this targets

`research/coordination/scaffold_retirement_backlog.md` rank-2: *"Thread `integrated_loop=True` (webapp/server.py
-> brain_chat_tui -> OneBrainComposer) — the spiking K-way cue-match SELECTION (GO 4/4 at V=320) replaces the
host string-`==` for-loop under ~15 faculties. Config-flip + re-verify at production vocab."*

**Premise check against current code (before writing anything), per the workflow's own instruction that the map
has been wrong before.** `OneBrainComposer(integrated_loop=True)` and `BrainConversationalAgent(...,
integrated_loop=...)` already existed — the 2026-06-21 #3 fold
(`2026-06-21-shortcut3-fold-integrated-loop-BUILD.md`) built the flag and validated it GO 4/4 (answer-identity,
moat, reconsolidation-abstain, anti-cheats) at K in {2,4,8,32}, V in {72,320}, multi-seed at V=72 and single-seed
(42) at V=320. **What did NOT exist, confirmed by grep before any edit:** `webapp/server.py` had **zero**
references to `integrated_loop` anywhere. `MultiTurnAgent.__init__` (the class every production chat brain is
built through — `_build_chat_brain` always passes `use_multiturn=True`) had **no `integrated_loop` parameter at
all**. Neither `brain_chat_tui._build_tiny_demo` nor `developed_brain_io.load_developed_brain` threaded one
through either. So the premise held exactly as scoped: the composer-level capability was GO, but there was no
path from the production entry point to opt into it — the task was genuinely open, not already done.

**Corpus/gate check** (`tools/before_you_build.sh`, RAG `--corpus all`): surfaced the 2026-06-21 build + de-risk
findings (read in full, summarized above) and the 2026-08-11 production-pipeline inventory (confirms the same
gap independently: the live turn's recall MECHANISM was still host-scaffolded at that reading). No existing
research gate or scoping doc covers this specific seam; this is a WIRING task composing two already-de-risked
pieces (the #3 fold's sequencer + the production construction chain), so `deep_research_at_wall` does not fire.

## The build (additive, default OFF everywhere, NO `sim/` edit)

One new parameter threaded through four call sites, plus one webapp env knob:

| site | change |
|---|---|
| `research/runners/multi_turn_agent.py` | `MultiTurnAgent.__init__` gains `integrated_loop=False`, passed through to the inner `BrainConversationalAgent`. Previously **absent from the signature entirely**. |
| `research/runners/brain_chat_tui.py` | `_build_tiny_demo(..., integrated_loop=False)` threads to `MultiTurnAgent`/`BrainConversationalAgent`. `_resolve_integrated_loop(args)` (mirrors `_resolve_composer_kind`) + `--integrated-loop`/`--no-integrated-loop` CLI flags + `BRAIN_INTEGRATED_LOOP` env resolution for the standalone TUI's `load_brain()`. |
| `research/runners/developed_brain_io.py` | `load_developed_brain(..., integrated_loop=False)` threads to `MultiTurnAgent`/`BrainConversationalAgent`. |
| `webapp/server.py` | New `_INTEGRATED_LOOP_DEFAULT_ON = False` (named constant, CLASS-PI-anchorable) + `_integrated_loop_enabled()` (mirrors `_ltm_ship_default_on()`'s exact idiom: unset -> the constant; `BRAIN_INTEGRATED_LOOP` in {1,true,on,yes} -> True). `_build_chat_brain` reads it for **both** the tiny-demo branch (`_build_tiny_demo(..., integrated_loop=_il)`) and the developed-brand branch (`load_developed_brain(..., integrated_loop=_integrated_loop_enabled())`) — the same env var spans both, mirroring `BRAIN_COMPOSER_KIND`'s existing precedent. |

**Not touched (documented, not silent):** `brain_chat_tui._load_self_knowledge` hard-codes `composer_kind="rf"`
— `integrated_loop` only reads on the `'onebrain'` branch, so threading it there would be a no-op; left alone
with a comment explaining why, rather than adding dead plumbing.

**A genuinely new fact surfaced while building this, not assumed going in:** the production onebrain composer
today is **not** the bare `OneBrainComposer` — `BRAIN_COMPOSER_MERGE`'s pool-1 default is ON
(`_COMPOSER_IN_POOL1_DEFAULT_ON = True` in `onebrain_merge_production.py`), so `BrainConversationalAgent`'s
onebrain branch actually constructs a `Pool1BoundOneBrainComposer` via `make_pool1_onebrain_composer(...)` by
default. `integrated_loop` is forwarded to that call site too (pre-existing code, unchanged by this rank); Part A
(below) checks whether it threads through correctly there as well — the de-risk targets the composer the webapp
*actually* builds today, not an idealized bare one.

**Byte-identical when off, confirmed in the diff, not just by inspection:** every new parameter defaults to
`False`; `_integrated_loop_enabled()` returns the `False` constant when `BRAIN_INTEGRATED_LOOP` is unset. `git
diff --stat -- sim/` is empty (reuse-by-import throughout, matching the original #3 fold).

## Verification

### PART A — mechanical thread-check (numpy-CPU, GPU-free)

Builds a REAL `ChatBrain` via `webapp.server._build_chat_brain('tiny-demo', 'stub')` — the actual `/api/brain-chat`
entry point for the default (no explicit bundle) brain, carrying the same LTM-attach / discourse-event-register /
biased-competition wiring a live turn does — with `BRAIN_INTEGRATED_LOOP` unset vs `"1"`, checking: the composer
is an onebrain-family composer either way; OFF gives `integrated_loop is False` and the tiny 5-fact battery
answers exactly as documented (`brain`-use/learn/store, `dog`-chase-`cat`, `cat`-eat-`fish`, abstain on the
never-stored `dog`/`eat`); ON gives `integrated_loop is True` (confirming the flag reaches the constructed
composer through the full `webapp.server -> brain_chat_tui -> MultiTurnAgent -> BrainConversationalAgent ->
OneBrainComposer`, or today's actual `Pool1BoundOneBrainComposer`, chain) with the moat still holding on the
absent `dog`/`eat` pair even at this small, over-abstention-prone vocab.

**An earlier version of this runner (same session) was run and FAILED two of its own checks** — not a mechanism
failure but an instrument bug in the check itself: it asserted the composer's exact class name (`OneBrainComposer`),
which broke on discovering the `Pool1BoundOneBrainComposer` fact above, and it passed a legitimately-`None`
query result straight to `Verdict.require()`, which treats a raw `None` measured-value as its own UNMEASURED
sentinel (a real collision between "abstained, correctly" and "the instrument never measured this" — worth
naming since it is exactly the class of trap `tools/verdict.py` exists to prevent, just tripped by this caller,
not the tool). Both were code bugs in the CHECK, not evidence against the plumbing (the two checks that were
never confounded by either bug — `integrated_loop is False`/`is True` reaching the composer, and the byte-identical
answer battery — passed cleanly on that same run). Fixed (relaxed the class check to `"OneBrainComposer" in
type(...).__name__`; compare `probe is None` as a bool before handing it to `require()`) and re-launched.

**Re-launched after the fix — GO, all 6 checks:**

| check | measured |
|---|---|
| OFF composer is onebrain-family (`Pool1BoundOneBrainComposer`, per the pool-1 fact above) | True |
| OFF `integrated_loop is False` | True |
| OFF tiny 5-fact battery answers correctly (`{brain_use: spikes, brain_learn: words, brain_store: memory, dog_chase: cat, cat_eat: fish, dog_eat_moat: None}`) | True |
| ON composer is onebrain-family | True |
| ON `integrated_loop is True` | True |
| ON moat holds (`dog`/`eat` abstains) | True |

`VERDICT rank2-mechanical-thread-check => GO`. `research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partA.json`.

### PART B — production-scale (V=320, K=32) answer-identity + moat, 6 seeds (GPU/cupy)

Extends the 2026-06-21 gate's 1-seed V=320 CONFIRMATION (a bare, hand-built `OneBrainComposer`) to the CLAUDE.md
6-seed standard (42/43/44/100/101/102), **through the SAME `MultiTurnAgent` construction shape**
`load_developed_brain(use_multiturn=True)` / `webapp._build_chat_brain` always build (not a bare composer) — the
newly-threaded plumbing exercised end to end at the validated production vocab tier, not a repeat of the
composer-API test that already existed. Reuses the validated K=32 fact set + V=320 padding recipe + the
memory-safe per-seed teardown by import (NO reimplementation) from
`_phaseB_onebrain_sequencerK_k32_margin_derisk` / `_phaseB_onebrain_integrated_loop_fold_derisk`. Op-point is
OneBrainComposer's own validated default (match_thresh=0.06, gain=0.11, sigma=1.0, input_gain=1.0) — not
overridden.

*(Result table to follow in the next commit, filled from Part B's own output once that run lands: per-seed
answer-identical + moat FA_total across 42/43/44/100/101/102, plus the OVERALL verdict. Not spelled out as a
file path here — the file does not exist on disk yet.)*

**Status at commit time: PART B WAS LAUNCHED (GPU, background) and had not yet completed when the plumbing +
Part A results were committed** (this finding is being updated in a follow-up commit once it lands, per the
workflow's "commit before any long verify" instruction — the plumbing, the runner, and Part A's mechanical GO do
not depend on Part B's result and should not wait behind it). Reproduce with the `verification:` command above.

## Honest scope (what this rank does NOT claim)

- **Not flipped default-on.** `_INTEGRATED_LOOP_DEFAULT_ON = False`; this is a de-risk for an owner-gated flip
  decision, matching the task's explicit instruction. Per `docs/TERMS.md`: this makes the mechanism **wired
  (default-off)** — reachable from `/api/brain-chat` on some request (`BRAIN_INTEGRATED_LOOP=1`) — not
  **on-by-default**, and the host `_scan` is not **scaffold-retired** (it remains the default path).
- **The tiny-demo's own vocab stays outside the validated margin.** Flipping `BRAIN_INTEGRATED_LOOP=1` today
  would apply to the tiny-demo composer too (~15 words), which is BELOW the documented small-vocab
  over-abstention boundary (`_burndown_1A_c2_smallvocab_derisk.json`) — a real but SAFE-direction cost (moat
  0-FA, never a false-accept), not a defect in this rank's plumbing. The knob is threaded uniformly because the
  webapp has one code path for both brain sizes; an operator turning it on should do so for a real (large-vocab)
  developed bundle, not the GPU-free smoke fallback.
- **No developed bundle on disk is `composer_kind='onebrain'` yet.** `scale787/day_33` (the only production
  bundle manifest found) is still `composer_kind='rf'` — RANK-1 (bundle rebuild rf->onebrain) is a separate,
  independent backlog item and was NOT touched here. `integrated_loop` on the developed-brand branch is
  consequently a no-op against every bundle on disk TODAY (composer_kind_changed is narrowly allowlisted to
  `'slotbinder'` only, an existing, deliberate safety narrowing this rank did not touch); it activates
  automatically, with no further code change, once a bundle's own saved `composer_kind` is (or becomes)
  `'onebrain'`.
- **"under ~15 faculties"** is satisfied by construction (Part A goes through the actual `_build_chat_brain`
  entry point with its LTM/discourse/biased-competition wiring genuinely present; Part B goes through the same
  `MultiTurnAgent` shape production always builds, not a bare composer) rather than by driving every one of those
  faculties through a scripted turn — a broader end-to-end conversational regression (discourse anaphora +
  biased competition actually exercised alongside `integrated_loop=True`) is a natural next rung, not attempted
  here.

## No-regression check (already confirmed, not pending)

`tests/test_multi_turn_agent.py` (multi-turn anaphora + multi-hop + moat) and
`tests/test_developed_brain_io_codes_roundtrip.py` — both touch code this rank edited
(`MultiTurnAgent`/`load_developed_brain`) — pass unchanged: 6/6, 129s, numpy-CPU.

## CI guard

`tests/test_rank2_integrated_loop_thread.py`: every new/extended signature defaults `integrated_loop=False`
(pure introspection); `webapp.server._integrated_loop_enabled()` unset->False, explicit on/off tokens resolve
correctly; a monkeypatch-based WIRING check (< 1s, `OneBrainComposer`/`LearnedAssocGraph` replaced with instant
recorders, `composer_merge_enabled` forced False for determinism) confirms `MultiTurnAgent`'s new parameter
reaches the composer's constructor kwarg for both `True`/`False` — a repeat of the real ~90-180s onebrain build
this rank's own runner already pays for would not be a proportionate CI cost (`BrainConversationalAgent`'s
onebrain branch always sizes the sequencer fabric at `k_max=32` regardless of vocab, so there is no cheap
small-K lane through this specific branch the way `test_onebrain_integrated_loop_fold.py` gets by calling
`OneBrainComposer` directly). 5/5 pass, ~40s (dominated by importing `webapp.server`, not by anything new here).

## Bottom line

The plumbing is built and additive (default-off everywhere; `git diff --stat -- sim/` empty). **Part A
(mechanical thread-check, numpy-CPU) is GO, 6/6 checks** — the flag genuinely reaches the actual production
composer through the real `webapp.server._build_chat_brain` entry point, byte-identical when off, verified in
the data. **Part B (production-scale V=320/K=32, 6-seed answer-identity + moat, GPU/cupy) was launched and had
not completed at commit time**, per the workflow's "commit the runner + finding before any long verify"
instruction — this finding is updated in a follow-up commit with Part B's own artifact-reported verdict once it
lands, rather than asserting that result ahead of its instrument. Not flipped; an owner decision on
`BRAIN_INTEGRATED_LOOP`'s default is the next step, gated on Part B's result.
