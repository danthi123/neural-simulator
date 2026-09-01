---
type: finding
status: live
date: 2026-09-01
mechanism: confidence-forthcomingness (board #94) — the margin-normalized metacog confidence read caps how much
  the rich chat turn volunteers; flipped default-ON alongside its LTM-elaboration dependency
lane: introspection
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: none — both gating verifies below are genuine 6-seed real-handler GOs
artifacts:
  - research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration_6seed.json
  - research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json
external: NO-EXTERNAL-NEEDED — closes an internal calibration + isolation residual named by the immediately-prior
  finding on this same lane
---

# Confidence-forthcomingness flips default-ON: discriminates 6/6 on the real 15k-KB, and the "not byte-identical-off"
# blocker was a STALE artifact, not a live regression — fresh 6-seed re-run passes ALL FOUR checks

**Artifacts:** `research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration_6seed.json`
(discrimination, `vary_lesion_all_GO: true`, 6/6 seeds) + `research/findings/raw/_confidence_ltm_loadbearing/
verify_confidence_ltm_loadbearing.json` (isolation/no-regression, freshly regenerated this session, all 4 checks
`byte_identical_off` / `load_bearing` / `lesion_reverts` / `moat` PASS on all 6 seeds).

## Context (⛔ partial retraction below)

`2026-09-01-confidence-metacog-margin-norm-calibration-at-scale-discriminates-not-byte-id-off.md` (branch
`research/confidence-metacog-calibration-at-scale` @ `ec7aa0217`) built the margin-NORMALIZED (scale-invariant)
metacog band — `research/runners/rf_phasor_composer.py::_cleanup_all_score_stats` gains an additive `margin_norm`
field (the SAME peak-relative `(peak-runner)/peak` ratio `OneBrainComposer._margin` already uses), and
`research/runners/metacog_production_organ.py::mean_role_confidence` now prefers it over the raw, unnormalized
`margin` field an LTM-sourced (`RFPhasorComposer`/`ShardedPhasorStore`) trace carries under the same key. That
finding measured `vary_lesion_all_GO: true` (6/6) on the real handler against the literal shipped
`wikidata_core_15k` LTM — the discrimination mechanism was DONE — but reported the isolation/no-regression check
(`verify_confidence_ltm_loadbearing.py`, the tiny-demo+LTM fixture) as FAILING (`byte_identical_off: False`,
`off_n_sentences: 4` instead of the expected 2, not matching the no-LTM-tier reference), attributed to the
margin-norm change leaking into the now-default-ON source-provenance/source-monitoring honesty hedge (#129/#140).

## What this session found: the "not byte-identical-off" JSON was STALE, not a live regression

Pulling the exact same three production files (`metacog_production_organ.py`, `rf_phasor_composer.py`,
`rich_answer_composer.py`) plus the test itself into an isolated worktree and **re-running
`verify_confidence_ltm_loadbearing.py` fresh, unmodified**, produces a DIFFERENT result than the committed JSON:
all 4 checks PASS on all 6 seeds, including `byte_identical_off`. Two independent minimal reproductions (a bare
`RichAnswerComposer.gather()`/`.answer()` call, and a single-seed call through the real `webapp.server.brain_chat`
handler) agree with the fresh 6-seed run: with both flags explicitly `=0`, the reply is `n_sentences=2`,
`answer="The brain uses spikes. The spikes carrys information."` — byte-identical to the no-LTM-tier reference,
exactly as the mechanism's own design intends (`_elaborate_from_ltm_enabled()`/`confidence_forthcoming_enabled()`
both correctly gate off; `mean_role_confidence`'s margin_norm preference is a no-op on this fixture's BUFFER-only
first hop). Per `docs/TERMS.md`, "byte-identical" must be asserted in the data, never inferred from code — this
finding asserts it against a **freshly regenerated** artifact, not the prior session's stale one, and can be
reproduced by re-running the identical script (`.venv/bin/python -m research.findings.raw.
_confidence_ltm_loadbearing.verify_confidence_ltm_loadbearing`, `SIM_BACKEND=numpy`). The most likely explanation:
the prior session iterated on `metacog_production_organ.py`/`rf_phasor_composer.py` after the last time this
verify script was actually run, and committed the finding without re-executing it — a documentation-artifact
staleness lapse, not a code defect. The discrimination artifact (`verify_margin_norm_recalibration_6seed.json`)
is unaffected by this correction and stands as measured.

**⛔ CORRECTION to `2026-09-01-confidence-metacog-margin-norm-calibration-at-scale-discriminates-not-byte-id-off.md`:**
its "NOT byte-identical-off" verdict and the `off_n_sentences: 4` / `off_matches_no_ltm_tier: false` readings in
its `verify_confidence_ltm_loadbearing.json` are RETRACTED (stale artifact, superseded by the freshly regenerated
JSON cited above). Its discrimination claim (`vary_lesion_all_GO: true`, 6/6) is UNCHANGED and survives fully.

## Decision: no gating needed — the margin-norm calibration is a global, byte-identical-off-safe improvement

Because the isolation harness's `byte_identical_off` check now passes GENUINELY (not merely "no worse on a
battery" — literally byte-identical text + sentence count vs the no-LTM-tier reference), the stronger of the two
branch conditions in this arc's task (gate vs. prove-strict-no-regression) is already satisfied without writing
any gate. `mean_role_confidence`'s margin_norm preference stays UNGATED (module-level, not conditioned on
`BRAIN_CONFIDENCE_FORTHCOMING`): with the coupling's own flags off, an LTM-sourced trace's margin_norm read never
influences the DEFAULT reply because the DIRECT/CHAIN facts on the isolation fixture are BUFFER-answered (no
LTM-sourced trace enters `mean_role_confidence` at all in the off condition), and the honesty hedge for this
fixture's buffer-answered fact is likewise unaffected (its trace never carries `margin_norm`). No case was found,
on this harness or the real-15k-KB discrimination harness, where the change makes the honesty hedge fire less
correctly than before.

## End state reached

(a) confidence-forthcomingness DISCRIMINATES on the real 15k-KB — `vary_lesion_all_GO: true`, 6/6 seeds
(`verify_margin_norm_recalibration_6seed.json`). (b) NO regression to the default reply on the isolation
harness — `byte_identical_off: true` (plus `load_bearing`/`lesion_reverts`/`moat`, all 4 checks), 6/6 seeds,
freshly regenerated (`verify_confidence_ltm_loadbearing.json`). (c) tiny-demo GO preserved — the SAME harness
IS the tiny-demo-construction fixture; GO 6/6.

**AUTO-FLIP** (owner 2026-09-01 policy: validated-GO + load-bearing + moat-safe + byte-identical-off →
default-ON, no owner-gate): `webapp/confidence_forthcoming_chat.py::_CONFIDENCE_FORTHCOMING_DEFAULT_ON = True`.
Its own dependency, `research/runners/rich_answer_composer.py::_ELABORATE_FROM_LTM_DEFAULT_ON` (
`BRAIN_ELABORATE_FROM_LTM_SHARD`), was already flipped `True` on the source branch this session's margin-norm
work built on — kept as-is (without it the reach never has LTM content to trim, per that flag's own docstring).
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `confidence-forthcomingness` row moves `on_by_default: NO → YES`
with a second `default_anchor` entry for `_ELABORATE_FROM_LTM_DEFAULT_ON`.
