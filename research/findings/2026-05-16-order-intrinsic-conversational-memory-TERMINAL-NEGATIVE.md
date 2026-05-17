# Order-intrinsic conversational memory — honest TERMINAL NEGATIVE (the conversational-generation arc concludes; the validated grounded-memory asset stands)

## TL;DR

The order-intrinsic line — the genuinely-different architecture built
*after* the 6-negative generative-production line and targeting its
precise converged diagnosis (make order INTRINSIC to the
representation via the multi-seed-validated D.11/P4.1 DG/CA3 store, and
read it back with a DETERMINISTIC position sweep — NO learned sequence
model anywhere) — **FAILED its pre-registered multi-seed capability
gate.** Per the design's explicit pre-registration, this is the
**terminal, decision-relevant** conclusion of the conversational-
*generation/production* arc. Across **seven** maxed-integrity, honestly
propagated attempts (Inc-1/2/3, G1, G1.5, P, and now order-intrinsic),
self-contained, locally-trained generative *production* of
order-correct propositions does not work on this substrate/hardware
under the no-cheating/local constraints — whether order is read OUT of
an order-blind pool (the first six) OR made INTRINSIC and read by a
deterministic sweep (this line). The project's robust, multi-seed,
anti-cheat-validated **trustworthy grounded continual memory with
no-confabulation abstention is untouched, LOAD-BEARING-no-harm-
re-proven, and stands as the deliverable.** This is an honest
scientific boundary, not a project failure.

## The pre-registered gate result (FIXED bars, never touched)

`order_intrinsic_gate.py`, seeds 42/43/44, **proper pre-registered
config** `encoding_steps=100` (the multi-seed-validated P4.1 DG/CA3
store's validated co-drive regime — deliberately the STRONGER config,
NOT the Task-6 no-harm speed override of 60; FIXED, not tuned
post-hoc), per-seed sidecar-frozen control-max abstention floor
(NEVER 650, NEVER recomputed at gate time), `permuted_order_controls`
load-bearing, `order_intrinsic_verdict` reusing UNMODIFIED
`song_g1_core.score_order`/`g1_verdict` (`_G1_MARGIN=0.10`/
`_G1_ABS_FLOOR=0.5` byte-untouched), `aggregate_multiseed(min_seeds=3)`:

| seed | frozen floor | held-out props | true_score | best_perm | gate_cleared | verdict |
|---|---|---|---|---|---|---|
| 42 | 0.01125 | 3 | 0.000 (one 0.500) | 0.000 | N (all) | 0/3 FAIL |
| 43 | 0.00750 | 3 | 0.000 | 0.000 | N (all) | 0/3 FAIL |
| 44 | 0.01750 | 3 | 0.000 | 0.000 | N (all) | 0/3 FAIL |

Aggregate (`aggregate_multiseed`, min_seeds=3): n_seeds 3,
enough_seeds Y, all_seeds_have_props Y, **n_props_pass 0/9** →
**GATE: FAIL**. Wall clock 133.6 s (3 seeds).

## What the failure actually is (honest mechanism, no spin)

This is **not** an anti-learning result (permuted did NOT beat true —
`true ≈ best_perm ≈ 0` everywhere). It is a **near-noise read-back +
correct abstention** result:

- Position-sweep read-back top-rates are 0.005–0.0175 (≈0.5–1.75%
  pool rate) — near the OU-noise floor. The trained additive
  `ec_context(position) → motor_{N,E,S,W}` read-back pathway, driven
  by position ALONE, does not carry a recoverable concept signal at
  the pre-registered multi-seed bar.
- `decode_position_sweep` therefore **abstains** on essentially every
  slot (`true_decoded` is mostly `None`; `gate_cleared` False for
  all 9). The per-slot no-confabulation moat fires exactly as designed
  — the system **correctly refuses to emit a confabulated concept**
  rather than guessing.
- The single non-zero (seed 42, `['east','north']`, true_score 0.500)
  is one position decoded, the other abstained — still FAIL (a slot
  abstained; no margin over permuted; below the FIXED 0.5+margin bar).

So: the order-intrinsic STORE is validated and untouched (P4.1 3/3
multi-seed; LOAD-BEARING no-harm Task-6 re-proved store distinctness
3/3 + no-confab moat 3/3 — the additive pathway is **harmless**), and
the agent **correctly abstains rather than confabulate** — that is
exactly the deliverable property. What does NOT work is recovering the
intrinsically-stored order via a trained position-alone read-back
pathway: it stays weak (design-pre-registered outcome (a)), now shown
at the proper config, multi-seed, 0/9.

## The converged conclusion (across the whole arc)

Seven attempts, one consistent, decision-relevant boundary:

- **Inc-1/2/3** (char-level BPTT): memorization ≠ held-out
  generalization.
- **G1 / G1.5 / P** (songbird / trajectory / predictive-coding over
  the recognition-only pool): the order signal cannot be read OUT of
  an order-blind pool and made to generalize through substrate
  realization (G1 AUC 0.775; P learned order in isolation but did not
  generalize the held-out gate).
- **Order-intrinsic** (order made INTRINSIC; deterministic sweep
  read-back; the categorically-different thing the diagnosis pointed
  to): the store is genuinely order-intrinsic and validated, the
  no-harm asset is preserved, but the trained position-alone read-back
  does not recover the stored order at the pre-registered multi-seed
  bar — near-noise, abstention-dominated, 0/9.

Self-contained, locally-trained generative *production* of
order-correct propositions — the sim emitting novel order-correct
propositions through its own substrate, judged by its own
comprehension, with no external teacher/corpus/templates — does not
work on this substrate/hardware under the no-cheating/local
constraints. Continuing to spin further variants would be the
garden-of-forking-paths / config-cranking the project's anti-cheat
discipline explicitly forbids past a pre-registered terminus. The
honest action is to record this boundary and stop the conversational-
generation/production arc here.

## Anti-cheat discipline (maxed-integrity terminal negative)

The strongest possible form of an honest negative: pre-registered,
multi-seed, run at the PROPER (stronger) config, maxed-integrity.

- Bars `_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5` NEVER tuned (verified
  byte-identical; `order_intrinsic_core`/`order_intrinsic_encode`/
  `song_g1_core` byte-untouched by the gate commit).
- 650 NEVER used anywhere (functional grep clean; only in
  "NEVER 650" prose).
- Control-max abstention floor pre-registered, computed ONCE per seed
  BEFORE held-out eval, sidecar-frozen, NEVER recomputed at gate time;
  control-MAX ONLY (the encoded distribution provably cannot move it).
- **PROPER pre-registered config** `encoding_steps=100` — the
  validated-store regime, deliberately the STRONGER config, NOT the
  Task-6 no-harm speed override of 60; FIXED in code with recorded
  rationale; NOT tuned after seeing the result (a maxed shot, not an
  under-powered one).
- Permuted-ORDER control load-bearing (here moot since `true ≈ 0`,
  but applied to every held-out prop).
- ≥3 seeds MANDATORY (3/3, all FAIL — explicitly NOT a single-seed /
  near-noise pass; the cheap probe had already proved single-seed
  unreliable at 50%).
- LOAD-BEARING no-harm check PASSED *before* the gate was trusted
  (Task-6 e9cd7b2: validated DG/CA3 store distinctness 3/3 + the
  no-confabulation moat 3/3 — the additive read-back pathway did NOT
  regress the validated asset; the asset is provably un-regressed).
- Gate runner independently spec+quality reviewed APPROVED (8/8
  pre-registered anti-cheat invariants PASS) BEFORE launch; the one
  Important review issue (resumed-seed record schema) fixed pre-run
  and proven display/record-only (verdict provably unaffected).
- A category-error in the pre-gate Task-6 framing was caught and
  corrected from recorded data with NO GPU re-run / no pass-chasing
  (same integrity class as the documented C1/C2, Inc-3-held-out,
  P-bias, Task-5-retarget corrections).
- The gate was run ONCE at the pre-registered config and is NOT being
  config-cranked. This terminal FAIL is propagated, not iterated away.

## The robust validated asset is the deliverable (untouched, no-harm-re-proven)

The project's genuinely validated, multi-seed, anti-cheat
contribution stands entirely intact and was LOAD-BEARING-re-proven
un-regressed by this line's no-harm check: the **trustworthy grounded
continual memory with no-confabulation abstention** — G.20 sparse
distributed ensemble (160 concepts @ 100% / 320 @ 98.4%, multi-seed,
cross-bridge), engram tagging / stim-recall (D.14, 87.5%), multi-tag
cue retrieval (90% FULL / 100% PARTIAL multi-seed), compositional
intersection / yes-no / role queries (90–100%), hippocampal
trisynaptic separation/completion (D.12/D.13), CLS no-catastrophic-
forgetting (Marr/McClelland), and the no-confabulation moat that
*refuses to make things up* — the very property this gate showed
holding (correct abstention rather than confabulated order). It is the
honest, robust deliverable. The conversational-generation/production
arc is concluded; the grounded-memory line is validated and shippable.

## Files

- Gate: `research/runners/order_intrinsic_gate.py` (commit 32a7a96 +
  resume-schema fix 7fde87d) + `tests/test_order_intrinsic_gate_smoke.py`
- Pure core (reviewed SOUND, hardened):
  `research/runners/order_intrinsic_core.py` +
  `tests/test_order_intrinsic_core.py`
- Encode/readback on the validated DG/CA3 store:
  `research/runners/order_intrinsic_encode.py`
- LOAD-BEARING no-harm: `research/runners/order_intrinsic_noharm.py`
  (Task-6 corrected e9cd7b2: store 3/3 + no-confab 3/3 PASS)
- Evidence: `research/findings/raw/g11_bg/order_intrinsic_gate.json`,
  `_order_intrinsic_gate_stdout.log`, per-seed frozen sidecars
  `order_intrinsic_gate.ckpt.npz.{42,43,44}.json`
- Design/plan: `docs/plans/2026-05-16-order-intrinsic-conversational-memory-{design,implementation}.md`
- Prior arc: `2026-05-16-generator-increment{1,2,3}-*.md`,
  `2026-05-16-generator-G1-songbird-NEGATIVE.md`,
  `2026-05-16-generator-G1.5-trajectory-readout-NEGATIVE.md`,
  `2026-05-16-generator-P-predictive-coding-NEGATIVE.md`
