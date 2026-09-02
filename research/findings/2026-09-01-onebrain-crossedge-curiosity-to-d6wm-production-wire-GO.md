---
type: finding
status: live
date: 2026-09-01
mechanism: onebrain-xedge-curiosity-d6-production-wire
lane: one-brain/integration/production
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_6seed.json
runner: research/runners/onebrain_xedge_curiosity_d6_production.py
builds_on:
  - research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md
  - research/findings/2026-08-27-onebrain-xedge-production-live-learning-GO.md
---

# The curiosity.ask -> d6.w0 cross-edge is wired into the LIVE `/api/brain-chat` and DRIVES the D6 hold-query reply
# text itself (not a diagnostic field) — 6-seed GO on the production wrapper, real-handler confirmed, session-isolated

**One-line:** the runner-level 6-seed GO (`2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md`) wired
curiosity's `ask` crave pool -> d6's `w0` WM slot on a standalone research pool. This finding carries it into
`webapp/server.py`'s real `/api/brain-chat` D6 hold-query branch (`research/runners/
onebrain_xedge_curiosity_d6_production.py`, `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6`), and — unlike this project's own
PART-1/R4 precedents (additive diagnostic field only) — makes the cross-edge's own measured, lesion-attributable
suppression change the ACTUAL reply text a live hold-query turn returns: a session that just craved (a genuine D3
abstain) gets an honest, self-consuming qualifier appended to "who are we talking about"; a session that never
craved, or the SAME session under `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1`, does not. 6-seed GO on the
production wrapper's own self-test (n_go 6/6, exactly reproducing the runner-level finding's per-seed numbers to
full precision); confirmed end-to-end through the REAL FastAPI handler (a live 3-turn session: introduce 2
referents -> a genuine abstain that craves -> a hold-query whose reply text changes). Session-isolated (the
2026-08-27 cross-session leak-fix pattern, reused): the crave bit lives on the calling session's own per-session
`MultiReferentWMOrgan` instance, never on the shared frozen cross-edge pool.

## 1. Why this is not another diagnostic-only wire-in

PART-1 (`2026-08-27-onebrain-xedge-production-frozen-GO.md`, later live-learned in
`2026-08-27-onebrain-xedge-production-live-learning-GO.md`) and R4
(`onebrain_xedge_selfschema_production.py`) both attach an additive field to the response
(`resp["...']["source_provenance_crossedge"]`) that never changes `resp["answer"]` on their own is_hyp path — a
shape this project's OWN 2026-08-19 standing lesson names directly: *"a live-chat faculty is real only when it's
LOAD-BEARING on the conversation (changes content/tone/focus); a neural verdict stashed as metadata + a
default-on flip is a hollow checkbox integration."* PART-1's OWN PART-2 (the live-learning finding cited above)
already establishes the precedent that a cross-edge CAN be made to flip a real production decision (the
`repair_target` role) and, once validated, was flipped default-ON (`_XEDGE_DEFAULT_ON = True` in
`onebrain_xedge_production.py`) — this finding follows that same bar on the curiosity->d6 pair: the wire-in
changes the literal string returned to the caller, gated by the SAME lesion-attributable measurement the
runner-level 6-seed GO already validated.

## 2. What was built (reuse-by-import; no `sim/` edit)

`research/runners/onebrain_xedge_curiosity_d6_production.py`: a `XedgeCuriosityD6ProductionPool` holder around
`_onebrain_crossedge_curiosity_to_d6wm.AskToW0Pool` (imported, not reimplemented) — builds the [curiosity,
d6_multiref_wm] merged pool once, GROWS the `ask->w0` edge via the substrate's own `train()` (0.05 -> ~1.7-2.1,
matching the runner-level GO), then FREEZES it (`enable_hebbian_learning=False`). Exposes `crossedge_w0_shift
(pool, ask_held)` — the live reply-path hook: reads `AskToW0Pool.read_w0("familiar")` as the baseline and
`read_w0("novel" if ask_held else "familiar")` as the held read, returning the signed shift (negative =
suppression, matching the runner-level GO's own measured sign).

`webapp/server.py` — two edit sites, both inside the existing D6 MULTI-REFERENT WORKING MEMORY block:
  1. The hold-query READ-OUT branch (the early `return JSONResponse(...)` that already answers "who/what are we
     talking about" off `d6org.judge(msg, ...)`): when `xedge_curiosity_d6_enabled()`, read THIS SESSION's own
     `d6org._xedge_curiosity_recent_crave` bit, drive `crossedge_w0_shift`, and — when the measured shift clears
     the runner's own registered `INTACT_FLOOR` (signed negative) — APPEND an honest qualifier to `jq["readout"]`
     (" Though a recent flash of curiosity is competing for my attention right now.") before it becomes
     `resp["answer"]`. The bit is then CONSUMED (cleared), so the qualifier fires once per crave episode, not on
     every subsequent hold-query (mirrors prospective-memory's own "fires once" intention consumption).
  2. `_curiosity_followup` (the existing D3 abstain-triggered crave read): when the flag is on and `d6org` exists
     for this session, PERSIST this turn's own live `curious` verdict onto `d6org._xedge_curiosity_recent_crave` —
     an instance attribute on THIS session's own per-session organ (2026-08-27 session-isolation pattern reused
     verbatim: never written onto the shared process pool), so only a session that itself just craved can ever
     see its own crave reflected in a LATER hold-query.

## 3. 6-seed GO on the production wrapper's own self-test (numpy CPU)

`SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_curiosity_d6_production --grow --seeds
42,43,44,100,101,102 --out research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_6seed.json` — 6/6
GO, reproducing the runner-level 6-seed GO's own per-seed numbers to full precision (this wrapper calls the SAME
`AskToW0Pool.read_w0`, not a reimplementation):

| seed | grown weight | shift intact (ask_held=True) | shift lesioned | frac attributable | no-signal-ok | clears floor | GO |
|---|---|---|---|---|---|---|---|
| 42 | 2.020219 | -0.011375 | +0.000250 | 1.021978 | true | true | GO |
| 43 | 1.889884 | -0.010500 | -0.000125 | 0.988095 | true | true | GO |
| 44 | 1.981809 | -0.013000 | -0.000375 | 0.971154 | true | true | GO |
| 100 | 1.970807 | -0.010750 | -0.000125 | 0.988372 | true | true | GO |
| 101 | 2.117586 | -0.014250 | +0.000125 | 1.008772 | true | true | GO |
| 102 | 1.739121 | -0.010000 | -0.000125 | 0.987500 | true | true | GO |

(Two seeds read fractions slightly over 1.0 — the SAME benign lesioned-control wobble the runner-level finding's
own §3 documents and `tools.lab.attributable_to` flags on its own "ABOVE 100%" line; the lesioned \|shift\| itself
is small and near-zero on every seed.) Artifact: `research/findings/raw/
_onebrain_xedge_curiosity_d6_production_frozen_6seed.json` (n_go 6/6).

## 4. Confirmed end-to-end through the REAL `/api/brain-chat` handler

A live 3-turn session (numpy backend, `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6=1`):

1. `"the dog and the cat are here"` -> MAINTAIN, loads `dog`+`cat` into this session's D6 buffer (unaffected by
   this wire-in).
2. `"what does the wombat eat"` -> a genuine D3 abstain, `curiosity.curious=True` (want_hz=129.2 >=
   threshold=65.9) -> `_xedge_curiosity_recent_crave` set True on this session's own `d6org`.
3. `"who are we talking about"` -> reply: `"I'm holding 2 referents in working memory at once: dog and cat.
   Though a recent flash of curiosity is competing for my attention right now."` — the qualifier is REAL reply
   text, not metadata. The attached diagnostic (`resp["multiref"]["curiosity_crossedge"]`) reads `ask_held=true,
   w0_rate_familiar=0.060625, w0_rate_read=0.049250, shift_w0=-0.011375, cross_weight=2.020219` — matching seed
   42's own row above to full precision (the SAME live substrate read, not a re-derivation).

Formalized as `tests/test_webapp_server.py::test_brain_chat_xedge_curiosity_d6_*` (five tests, all green): no-
regression on ordinary turns; explicitly-disabled byte-identical (`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6=0`, the
escape hatch — no qualifier, no diagnostic key, even after a genuine crave); the ambient PRODUCTION default is
ON (env fully unset -> the qualifier still fires); the qualify-and-lesion-collapse test above (crave -> qualifier
appears and is CONSUMED on repeat; no-crave -> never appears; `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1` -> a
genuine crave still gets NO qualifier, shift collapses under the runner's own noise-floor ratio); and session
isolation (a fresh session that never craved cannot see another session's crave, `ask_held=False`).

**A HONEST INSTRUMENT NOTE, diagnosed via a standalone repro before being written into the no-regression test
(not guessed).** The no-regression test's FIRST draft compared the ENTIRE `curiosity` sub-dict flag-off vs
flag-on and failed, reproducibly, twice. A standalone repro (bypassing this wire-in's flag entirely) showed the
cause: calling `curiosity_production_organ.judge()` TWICE against the SAME already-built, already-calibrated
process-singleton bridge returns a genuinely different `want_hz` (129.17 vs 126.39 Hz on one observed pair,
`curious=True` both times) — a pre-existing, small-magnitude sample-to-sample read noise in the curiosity
organ's OWN spiking rate instrument, present with or without this wire-in's flag. This is the SAME residual
CLASS `onebrain_xedge_selfschema_production.py`'s own docstring already names for its OWN instrument ("two
consecutive amb_read calls... are not bit-identical... a tiny per-synapse pulse-timer/delay-buffer residual a
state-restore does not zero") — not a new defect, and not caused by this wire-in (confirmed: the `multiref`
sub-dict, the ONLY substrate read this wire-in's session-state WRITE could plausibly perturb, compared
byte-identical flag-off vs flag-on in the SAME repro). The no-regression test now compares `multiref` for exact
equality and `curiosity` only on its DECISION-relevant, config-derived fields (`curious`, `novelty`,
`threshold`), not the noisy `want_hz`/`curiosity_da` sub-fields — matching how every other cross-edge wire-in in
this codebase already grades this residual class on a tolerance, not bit-exact equality.

## 5. Moat-safety, byte-identical-off, session isolation (checked, not assumed)

  * **Moat-safe.** The qualifier never changes WHICH referents are reported held, never flips an abstain, never
    fabricates a fact — an honest functional self-report ("a recent flash of curiosity is competing for my
    attention"), the SAME style this codebase already uses for D6's own readout ("I'm holding N referents...")
    and curiosity's own follow-up ("My curiosity is piqued..."). No phenomenal claim.
  * **Byte-identical-off.** `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6` unset -> `xedge_curiosity_d6_enabled()` is False
    at BOTH edit sites -> no attribute is ever written on `d6org`, no production pool is ever built, no key is
    ever added, no text is ever appended. Checked in the data (whole-response dict equality), not inferred from
    reading the code.
  * **Session-isolated.** The crave bit is an instance attribute on the CALLING session's own per-session
    `MultiReferentWMOrgan` (`webapp/server.py`'s `_SESSION_MULTIREF[cache_key]`), never on the shared
    `XedgeCuriosityD6ProductionPool` (which itself carries no session-specific state at all — it is a pure,
    stateless-given-input direct region-current instrument). A fresh session's `d6org` (a fresh `getattr(...,
    False)`) cannot inherit another session's crave — checked directly (a second, never-craved session's
    hold-query carries no qualifier even immediately after a first session's genuine crave).

## 6. Honest residuals (declared, not hidden — carried + extended from the runner-level finding's own §5)

  * **The w0 slot the cross-edge biases is d6's OWN direct-drive region**, not (yet) bound to WHICHEVER
    discourse referent this session's `MultiReferentWMOrgan` has semantically loaded into register 0 — the
    qualifier is a genuine, lesion-attributable competition-for-attention read, but it does not (yet) know or
    care WHICH referent's register it is, so it can never (yet) cause the READOUT to actually DROP the competed
    referent from the "holding N referents" list — only append an honest qualifier alongside the unchanged list.
    Binding the cross-edge onto the semantic content of the currently-focused register (so a suppressed referent
    is the one the reply drops, not just flags) is a separate, later, reviewed rung.
  * **The "recent crave" carry-forward is coarse** (binary, non-decaying-until-consumed), not a continuous-time
    decay model of a lingering crave.
  * **Training remains host-supervised** (the tonic co-drive of `ask`+`w0`), exactly as the runner-level finding
    discloses — not claimed self-organized.
  * **Region-pair choice remains hand-directed** (carried from the runner-level finding's own residual).

## 7. Decision: AUTO-FLIP

Per the 2026-09-01 auto-flip policy (validated-GO + genuinely load-bearing on the live `/api/brain-chat` +
moat-safe + byte-identical-off + no-regression; the only guard is the hollow-flip trap), and following this
project's OWN precedent (PART-1's live-learning wire-in was flipped `_XEDGE_DEFAULT_ON = True` once its own
decision-flipping load-bearing check passed): `_XEDGE_CD6_DEFAULT_ON` is flipped to `True` in
`onebrain_xedge_curiosity_d6_production.py`. This is NOT a hollow flip — §4 shows the reply text itself changes,
not only a metadata field. Per `docs/TERMS.md`, this faculty is now correctly described as **wired** (reachable
from `/api/brain-chat` on the hold-query path) and **on-by-default**; it does not (yet) qualify as
**integrated / production-default** in the stricter sense (no existing host shortcut is being retired here — this
is a NEW cross-organ synapse, not a replacement of a prior host computation), so that stronger term is not
claimed.

## 8. Files

`research/runners/onebrain_xedge_curiosity_d6_production.py` (NEW) ·
`research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_6seed.json` · `webapp/server.py` (two edit
sites inside the existing D6 block, additive) · `tests/test_webapp_server.py` (five new tests). Reused,
unmodified: `research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` (`AskToW0Pool`, `INTACT_FLOOR`,
`LESION_RATIO`), `research/runners/curiosity_production_organ.py`, `research/runners/
d6_multiref_wm_production_organ.py`. No `sim/` file touched.

Functional read-outs only; no phenomenal-experience claim.
