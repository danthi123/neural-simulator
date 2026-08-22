---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-learn-through-use-production-noregression-recalled-topic
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_learn_through_use_noregression.py — through the REAL production functions
  (EpisodicRecallOrgan.note_topic/recall, d5_episodic_production_organ.recall_disclosure,
  continuous_engine.mark_recall/consolidate_used_memory/d5_consolidate_enabled) at the PRODUCTION encode
  (train_events=40), with the step-5/6 snapshot/restore weight-attribution isolation. Plus a no-GPU disclosure-layer
  unit guard tests/test_d5_learn_through_use_noregression.py.
runner: research/runners/_d5_learn_through_use_noregression.py
external: NO-EXTERNAL-NEEDED — a production-integration no-regression verification of the in-repo graded-read wiring
  under the knob-2-validated read criterion; no literature question. The saturating plateau ceiling is Bittner et al.
  2017 Science 357:1033.
artifacts:
  - research/findings/raw/_d5_ltu_noregression/summary_6seed.json
  - research/findings/raw/_d5_ltu_noregression/seed42.json
---
# D5 learn-through-use: the graded-read production wiring is no-regression-CLEAN on the recalled topic (OFF byte-identical, ON moat-unchanged, ON first-use rise conversation-visible); the default-on flip is HANDED BACK, gated on knob 1

Artifact: research/findings/raw/_d5_ltu_noregression/summary_6seed.json (6-seed production no-regression guard).

**One line.** The graded apical read is already folded into the production episodic memory (EpisodicDapMemory.recall emits depth_hold beside the binary in_memory gate; recall_disclosure surfaces it behind BRAIN_D5_CONSOLIDATE; consolidate_used_memory runs the between-turn plateau-gated BTSP — all default-OFF). This finalization brings the knob-2 read criterion (relative-tolerance floor + saturating-tail exclusion) into the verification instrument and adds a PRODUCTION no-regression guard that proves, through the real production functions at the production encode, that the wiring is safe FOR THE RECALLED TOPIC. The default-on flip is handed back to the owner and remains gated on knob 1 (the memory separator, board #73).

## What was finalized (additive; NO sim/ edit; OFF byte-identical to HEAD by construction)
- The knob-2 read criterion — `_mono_rel` with an absolute floor `max(2%-of-move, MONO_TOL_ABS)` + a saturating-tail dead-step exclusion + a flat-trace self-test — is brought from branch research/d5-graded-apical-read into research/runners/_d5_step6_graded_apical_read_derisk.py (the change is PURELY the criterion; the self-test runs on every invocation and BLOCKS if the loosening ever admits a flat/decreasing/collapsing trace).
- research/runners/_d5_learn_through_use_noregression.py — the production no-regression guard (below), imports that criterion.
- tests/test_d5_learn_through_use_noregression.py — a fast, no-GPU unit guard on the disclosure-layer invariants (7 tests, runs every commit): the OFF reply is the exact HEAD text (no strength clause), ON only APPENDS the strength to an admitted memory, the honest-abstain line is byte-identical off vs on, and a not-in-memory record never surfaces a strength on or off.
- NO production-code edit: the OFF path is byte-identical to HEAD BY CONSTRUCTION.

## The production no-regression verdict (6 seeds, cupy, production encode te=40)
<!--derived-->
Result: 4/4 instrument-valid seeds are a clean GO on all three claims; 2/6 seeds (43, 102) are instrument-invalid (self-ignition — dog's assembly self-completes so the binary MOAT correctly abstains, the honesty gate working, NOT a regression). Per valid seed (surfaced read = depth_hold, the BTSP IS_post):

| seed | A: OFF byte-identical | B: ON moat-unchanged | C: ON first-use rise | first-use ΔmV | knob-2 monotone (reported) |
|---|---|---|---|---|---|
| 42  | yes | yes | yes | 0.46 | yes |
| 44  | yes | yes | yes | 0.62 | yes |
| 100 | yes | yes | yes | 0.28 | yes |
| 101 | yes | yes | yes | 0.30 | yes |
| 43  | instrument-invalid (self-ignition) | | | | |
| 102 | instrument-invalid (self-ignition) | | | | |

- **A (OFF byte-identical):** BRAIN_D5_CONSOLIDATE unset -> recall_disclosure emits NO recall-strength clause (the default reply is the exact HEAD text), consolidate_used_memory returns None, and a full mark_recall -> consolidate cycle leaves the store weights hash-identical. 4/4.
- **B (ON moat-unchanged):** BRAIN_D5_CONSOLIDATE=1 -> the binary in_memory gate is flag-INDEPENDENT (recall never reads the flag): formed dog completes, never-formed cat abstains, formation-lesion collapses — identical off vs on. The honest-abstain reply for cat is byte-identical off vs on. ON only APPENDS the strength clause to a memory the binary gate already admitted (the completion text is unchanged), and the surfaced strength equals the record's real depth_hold. 4/4.
- **C (ON first-use rise conversation-visible):** one consolidation tick (the production budget, BRAIN_D5_CONSOLIDATE_BUDGET=1) raises the surfaced depth_hold (first-use ΔmV 0.28-0.62), the recall_disclosure STRING changes (the reply's recall-strength mV rises), in_memory holds through consolidation, and the OFF control is a byte-identical no-op (store hash unchanged + read flat). The knob-2 saturating-tail-tolerant monotone also holds 4/4 (vs 3/6 strict under the old criterion — the value of the floor). 4/4.

## The plateau saturates at the production encode (why the signal is a FIRST-USE rise)
At te=40 the binary UP-fraction is saturated (~1.0) so it is FLAT — exactly the quantization ceiling the graded read was built to escape. depth_hold is near its ceiling too: one tick RAISES it (the conversation-visible signal), but the regenerative NMDA plateau is amplitude-bounded (Bittner et al. 2017), so over further ticks it saturates/decays rather than growing monotonically. Production runs ONE tick per recall, so the FIRST-USE rise is the real production signal; the guard gates on it and reports the multi-turn trajectory + the knob-2 monotone for honesty.

## Scope (honest boundary): recalled topic vs neighbor crosstalk
This guard covers the RECALLED-TOPIC no-regression — the memory the turn actually used. The distinct NEIGHBOR-CROSSTALK no-regression (consolidating one memory perturbing an OVERLAPPING assembly's surfaced strength on ~1/6 emergent builds) is the existing research/runners/_d5_graded_flip_soak.py's domain and the residual that BLOCKS the flip until knob 1 lands — the memory separator (sep_bias=1000 -> 6/6 DISJOINT+healthy, board #73, commit e62113ef on branch research/memory-separator-readout), per finding 2026-08-21-d5-graded-apical-read-conversation-visible-in-production-flip-blocked-on-emergent-assembly-crosstalk. The self-ignition seen here (2/6) is the same emergent-assembly build-reliability residual knob 1 (disjoint+healthy) also addresses. A clean verdict here is NECESSARY, not sufficient, for the flip.

## The recommendation (HANDED BACK — the owner does the flip)
The graded-read WIRING is no-regression-clean on the recalled topic and the read criterion is finalized to the knob-2-stable form. Recommendation: PROCEED with the default-on flip ONCE knob 1 (the memory separator, board #73) is on main so emergent assemblies stay disjoint+healthy — which removes the two documented residuals (neighbor-crosstalk and self-ignition); re-run research/runners/_d5_graded_flip_soak.py with the separator to confirm 6/6 neighbor byte-identity, then flip BRAIN_D5_CONSOLIDATE default 0->1 in webapp/continuous_engine.py and move the docs/PRODUCTION_INTEGRATION_LEDGER.yaml d5-live-consolidation row to on_by_default:YES. This finding does NOT flip it. Scope honesty: the surfaced strength is a faithful spiking read (not a phenomenal claim); the snapshot/restore determinism guard + the single full-strength encode remain host idealizations.
