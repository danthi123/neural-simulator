---
type: finding
status: live
claim_check: measured (re-read of shard JSON already on disk; no brain was built, no job was queued)
date: 2026-09-29
lane: load-bearing (the #1 metric, owner 2026-09-19; the reply rule, owner 2026-08-19)
mechanism: the registered load-bearing fraction (research/runners/load_bearing_fraction.py over FACULTY_PROBES) scored
  at two levels on the same raw arms -- TRACE (the registered decision-field verdict in each lb.json, reproduced) and
  REPLY (answer / abstained / recalled_svo on the probe turn, intact_a vs lesion, intact_a vs intact_b as the null,
  reproduced against the lesion's own repeat) -- adjudicated from two independent re-scorers plus 14 adversarial
  skeptic checks, one per disputed or headline claim
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_load_bearing/_shards/b2b0924-base/s<seed>/<faculty>/{lb.json,intact_a_*,intact_b_*,lesion_*}
  - research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/ (aggregate.json = the reported ~0.95 / 24)
  - research/findings/raw/_load_bearing/_shards/flipdefaults-thin/ (aggregate.json = the reported ~0.60 / 15)
  - research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s<seed>.json (summaries only, no raw arms)
  - .claude/worktrees/_reply_level_rescore_2026-09-29/{scorer_A.py,scorer_B.py,scorer_B_output.json,adjudicator/}
  - research/coordination/b2d_decision_fields.json (the self-describing-field audit this finding's S1 reading uses)
verdict: The registered fraction measures decision fields, not replies. At reply level the B2b robust core is 17 of
  33 reply-defined coverable rows (~0.54 mean, corrected 27/33 trace-core once comprehension-monitor's broken lesion
  arm is excluded); the flip-adequate battery's reply core is 15 of 25. Of the 10 names the raw scoring calls
  "trace-only," one (comprehension-monitor) is not trace-only at all -- its lesion arm crashes -- and of the
  remaining 9, 7 are instrument limits (no Qwen mouth in the battery, or a probe that never reaches the reply path),
  2 are tautological (a lesion-branch-written label counted as the decision field), and 1
  (source-provenance-honesty) is genuinely trace-only ONLY for the currently-registered probe turn: on the
  chain-route turn the same monitor is already shown reply-load-bearing under the production default
  (research/findings/2026-09-01-source-provenance-honesty-129-default-on-flip-GO.md). The comprehension-monitor
  crash is fixed in this session (research/runners/comprehension_production_organ.py); re-measurement is pending.
---

# The load-bearing fraction measures decision fields, not replies: reply-level robust core 17 of 33 (B2b) and 15 of 25 (flip-adequate); 7 of 9 genuine trace-only rows are instrument limits, 2 are tautological, comprehension-monitor's lesion crash is now fixed

> **Provenance caveat (Claude, 2026-10-06, RULING s1-c4-audit):** affect-faculty rows here that ran on pool or AWS
> nodes before 2026-10-05 read the raw Warriner norm, not the DR-2 learned valence map (the runtime input was not
> shipped; `docs/drafts/stage1-exit/AUDIT-affect-pool-results.md`). Intact and lesion arms shared that fallback, so a
> row's load-bearing reading stands; it says nothing about the learned values.

## Headline

Three numbers per battery: the registered TRACE headline (reproduced exactly), the REPLY-level secondary (the
owner's 2026-08-19 rule -- "a faculty counts only if its lesion changes the reply" -- measured on the same raw
arms), and the corrected TRACE reading once comprehension-monitor's broken lesion arm is excluded (its "6/6
load-bearing" was a harness crash, never a lesion effect -- see below). Means are per seed (n load-bearing over n
exercised, as `lb_shard.py aggregate` computes), population SD. The robust core counts faculties positive on all six
DEFINED seeds.

<!--derived-->
| battery | probe set | trace mean +/- SD (as registered) | trace core | reply mean +/- SD | reply core | trace, comprehension-monitor excluded |
|---|---|---|---|---|---|---|
| flipdefaults-thin (2026-09-24, the shipped thin headline) | thin | 0.603 +/- 0.018 (26 exercised) | 15 | 0.372 +/- 0.018 (26) | 9 | 0.587 / 14 |
| flipdefaults-adequate (2026-09-24, the reported adequate headline) | adequate, 9 LB flags | 0.949 +/- 0.018 (26) | 24 | 0.627 +/- 0.019 (25) | 15 | 0.947 / 23 |
| B2b b2b0924-base (F = a308f1e09, 36 coverable rows, incl. the two 2026-09-29 seed-102 A1.4 re-runs) | adequate, 9 LB flags | 0.843 +/- 0.014 (34) | 28 | 0.535 +/- 0.014 (33) | 17 | 0.838 / 27 |
| _adequate6 (2026-09-21, earlier adequate set) | adequate v1 | 0.853 +/- 0.026 (26) | 20 | UNDEFINED (no raw arms kept) | UNDEFINED | -- |

**The honest reading.**
- **Reply-level load-bearing is 15-17 faculties**, not the 24-28 the trace headline reports. That is the number that
  answers "does the faculty's lesion change what the user sees" (owner, 2026-08-19).
- Of the names the raw per-faculty scoring calls "trace-only" (10 in B2b, same membership in all three batteries
  with raw arms): **1 is not really trace-only -- it is UNDEFINED** (comprehension-monitor's lesion arm raises an
  exception in 88 of 88 retained files; the "load-bearing" reading is the harness's own except-branch catching a
  crash, not a lesion effect). Of the remaining 9: **7 are instrument limits** (no Qwen mouth for the two
  affect-manner rows; a probe that never reaches the reply path for the rest -- da-mode is folded in as
  partly-tautological, partly-genuine, see below), **2 are tautological** (the only registered diff is a label the
  lesion branch writes about itself, regardless of any neural effect: da-mode-drives-response, affect-drives-response),
  and **1 is genuinely trace-only, but only for the registered probe** (source-provenance-honesty -- on a different,
  already-validated turn it is reply-load-bearing).
- **B2b includes the two seed-102 cells re-run on 2026-09-29** (discourse-register, episodic-memory; both
  regressed, null clean, lesion reproduced, landed 12:15/12:17, confirmed by an independent skeptic re-check against
  the timestamped files). Whether they count as the one A1.4 re-run is still an open board ruling; if voided, both
  cells are UNDEFINED and both faculties leave both cores, taking B2b's trace core to 26/27 and reply core to 15/16.
- **swap-drives-response is registered as "not-exercised" in every canonical battery** (the `held` turn never
  reaches it), but a separate, preregistered, reply-level probe -- `LB_SWAP_DRIVE_PROBE` -- has ALREADY been run on
  all 6 canonical seeds at one pinned revision (`7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`, git-clean, manifest
  verified) and reads reply-load-bearing on 6 of 6 seeds: `answer` differs ("Setting the held thread aside -- On
  cat, then -- the cat eats the fish" vs "the cat eats the fish"), with clean null and no-swap-due contrast
  controls. Only the seed-42 leg of that run is committed; the other five sit uncommitted in a working tree at the
  same revision. This is cited as existing evidence, not folded into any battery's registered count.

## Method

- **Trace level.** `lb.json` `per_faculty[0]` read verbatim: `verdict`, `load_bearing`, `diffs`. Never recomputed. A
  cell is exercised when its verdict is `regressed` or `pass`.
- **Reply level.** The response dict for the probe turn named in `lb.json` `turn`. Treatment is intact_a vs lesion.
  The null is intact_a vs intact_b (an independent same-seed rebuild). Reproduction is lesion vs `lesion_*.rep0`.
  The primary fields are `answer`, `abstained` and `recalled_svo`: the bubble text, its abstain styling and the
  recalled caption `webapp/static/app.js` renders. A cell is reply-load-bearing when a primary field differs in the
  treatment, stays the same in the null, and reproduces.
- **Secondary fields.** `verified`, `n_sentences`, `supporting_facts`, `hypothesis`, `hypothesis_svo` and `rich` were
  also compared. They changed no verdict in any of the 630 cells two independent scorers checked.
- **Integrity.** Across the three batteries with raw arms: 0 noisy reply cells, 0 unreproduced cells, 0 reply diffs
  on earlier turns of the same arm group. The probe turn matched `lb.json` in every cell. Every renderer value is
  `template-stub (GPU-free)`; **no battery has the Qwen mouth in the loop**
  (`research/runners/onebrain_regression_battery.py:884-885,902` hardcode `renderer="stub"` and
  `SIM_DISABLE_LLM=1`; `MoodConditionedRenderer` is a declared NO-OP on the stub,
  `research/runners/affect_production_organ.py:417-421`).
- **Scope of "byte-identical."** Where this document says byte-identical, the `answer` strings were compared
  exactly in the data.
- **Adjudication.** Two independent scorers (A: 36-coverable-row denominator; B: 34-exercised-row denominator) each
  scored all 630 cells; their per-cell verdicts agreed everywhere except one cell (episodic-memory seed 102, resolved
  below). An adjudicator re-derived the headline from the raw shard JSON with its own script
  (`.claude/worktrees/_reply_level_rescore_2026-09-29/adjudicator/adjudicate.py`) and issued a per-faculty verdict for
  all 36 coverable B2b rows. **14 independent adversarial skeptic checks** then tried to refute the adjudicator's
  headline and every disputed or load-bearing-adjacent per-faculty call; 5 of the 14 were REFUTED as stated
  (confidence-forthcomingness, source-provenance-honesty, swap-drives-response, wm-binding-advanced's "fix" framing,
  and the headline's raw episodic-memory count) and are corrected below. The other 9 were CONFIRMED, several with
  additional evidence the adjudicator had not cited.

## Per-faculty table (trace seeds / reply seeds; `--` = row absent from that battery's registry; n/e = not exercised)

<!--derived-->
| faculty | B2b | flip-adequate | flip-thin | adequate6 trace | class (adjudicated, corrections applied) |
|---|---|---|---|---|---|
| affective-tom | 6/6 . 6/6 | -- | -- | -- | REPLY-LOAD-BEARING (registered trace field is a lesion-written label; rescued by `answer`) |
| bg-action-selection | 6/6 . 6/6 | 6/6 . 6/6 | **0/6 . 6/6** | 6/6 | REPLY-LOAD-BEARING (thin trace misses it -- `abstained` is confounded there) |
| causal-whatif | 6/6 . 6/6 | -- | -- | -- | REPLY-LOAD-BEARING |
| common-ground-drives | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 6/6 | REPLY-LOAD-BEARING |
| curiosity-followup | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| discourse-register | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 6/6 | REPLY-LOAD-BEARING (B2b s102 cell is the 2026-09-29 A1.4 re-run, pending ruling) |
| episodic-memory | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 5/6 | REPLY-LOAD-BEARING (B2b s102 cell is the 2026-09-29 A1.4 re-run, pending ruling) |
| gnw-multistep-deliberation | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| metacog-monitor | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| noncontradiction-gate | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 6/6 | REPLY-LOAD-BEARING |
| pragmatic-implicature | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| prospective-memory | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 0/6 (different probe, pmem_form) | REPLY-LOAD-BEARING on the adequate set |
| reconsolidation | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING (registered trace value is lesion-written; rescued by `answer`) |
| self-initiated-utterance | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| surprise-monitor | 6/6 . 6/6 | 6/6 . 6/6 | 0/6 . 0/6 | 6/6 | REPLY-LOAD-BEARING |
| vision-identity-spiking-hmax | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| worldmodel-forward | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 . 6/6 | 6/6 | REPLY-LOAD-BEARING |
| affect-marker-spiking-wta | 4/6 . 4/6 | 4/6 . 4/6 | 4/6 . 4/6 | 4/6 | REPLY-LOAD-BEARING on 4/6 at F, via the "Gladly! " prepend retired at HEAD (2026-09-25, `969894855`); **expected 0/6 at HEAD under the stub** |
| affect-appraisal-interoceptive | 6/6 . 0/6 | -- | -- | -- | TRACE-ONLY, instrument limit (no mouth) |
| affect-coloring | 6/6 . 0/6 | 6/6 . 0/6 | 0/6 . 0/6 | 6/6 | TRACE-ONLY, instrument limit (no mouth) |
| affect-drives-response | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | TRACE-ONLY, tautological (pre-2026-09-25 it was genuinely reply-driving via the marker prepend on an affective turn; now double-dead: probe is mood-neutral AND the marker is retired) |
| comprehension-learned-animacy-cue | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | TRACE-ONLY, instrument limit (well-formed probe never reaches the abstain/repair branch) |
| comprehension-learned-verb-selects | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | TRACE-ONLY, instrument limit (same mechanism) |
| comprehension-monitor | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | **UNDEFINED**: the lesion arm raises an exception in every retained file (not trace-only, not load-bearing -- a broken instrument). **Fixed in this session**; re-measurement pending. |
| confidence-forthcomingness | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | TRACE-ONLY, **instrument limit** (corrected from tautological -- see below; a separately-run reply-level protocol already shows a genuine 6/6 GO) |
| da-mode-drives-response | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 | TRACE-ONLY, tautological on 4/6 seeds; a genuine (non-tautological) trace effect exists on 2/6, but never reaches the reply on this probe |
| multiref-competition | 6/6 . 0/6 | -- | -- | -- | TRACE-ONLY, instrument limit (the reply channel is a later hold-query turn the probe never asks) |
| source-provenance-honesty | 6/6 . 0/6 | 6/6 . 0/6 | 6/6 . 0/6 | 4/6 | TRACE-ONLY **for this probe only** (corrected -- see below; reply-load-bearing on the chain-route turn, already shown GO) |
| open-ended-generation | 6/6 . undef | 6/6 . undef | 0/6 . 0/6 | 0/6 (rich_open probe) | TRACE-ONLY, instrument limit (the distributional ruler never runs a chat turn; the dedicated reply instrument reads NO-GO, 0.094 < 0.10 floor) |
| swap-drives-response | n/e in every canonical battery | n/e | n/e | n/e | **UNDEFINED in-battery; REPLY-LOAD-BEARING (6/6) on a separately-run, preregistered probe** -- see headline |
| wm-binding-advanced | n/e | n/e | n/e | n/e | UNDEFINED (not exercised; two prior remediation attempts already tried -- see below) |
| d5-consolidate, sleep-replay, gnw-bus, spiking-anaphor | 0/6 . 0/6 | -- | -- | -- | NOT load-bearing at either level |
| da-gated-encoding | 0/6 . 0/6 | 0/6 . 0/6 | 0/6 . 0/6 | 0/6 | NOT load-bearing at either level |

## The trace-only and UNDEFINED rows, one by one

**comprehension-monitor: UNDEFINED. The lesion crashed in every retained file -- fixed in this session.**
- Every retained comprehension-monitor lesion arm carries `comprehension = {"on": true, "error": "ValueError: could
  not broadcast input array from shape (7002,) into shape (1032,)"}` -- 84 of 88 files with that exact string, the
  remaining 4 (an INVALID early tag and two others) with a related but not-identical broadcast error. The server's
  except branch writes that block (`webapp/server.py:6108-6109`); the harness reads `comprehended` True -> None as a
  regression, so B2b's "6/6 load-bearing" for this row is a crash, not a lesion effect. `answer` / `abstained` /
  `recalled_svo` are byte-identical on every checked seed.
- **Root cause, confirmed by code read and now fixed.** `ComprehensionProductionOrgan._snapshot_rest`
  (`research/runners/comprehension_production_organ.py`, was line 417) branched on `self._shared is not None and
  self._shared.snap is not None` -- the ORGAN's own pool-attachment state -- not on which comp bridge was being
  snapshotted. `_lesion_comp()` builds its twin standalone (`_build_comp(self.seed)`, no `shared=`) precisely so "it
  needs no pool guard" (the code's own comment at the `judge()` call site) -- but when the organ itself runs
  pool-attached (the production default, `BRAIN_ONEBRAIN_WAVE3_POOL`), the old condition still took the
  shared-snapshot branch for the twin, storing the pool's N=7002 rest snapshot into the twin's own N=1032-neuron
  arrays; `_hard_reset` then raised on the in-place assignment. **Fix**: the shared snapshot is now used only for
  the bridge the pool actually owns (`comp is self.comp`); every other bridge, including the twin, always takes the
  manual 40-step settle. This changes nothing for the intact path (`comp is self.comp` was already always true
  there). Verified with a before/after regression test that reproduces the exact (7002,)-vs-(1032,) crash on the
  pre-fix branch condition and confirms it is gone post-fix, plus a test that the intact path's snapshot is
  byte-identical to the shared pool's own snap before and after
  (`tests/test_comprehension_lesion_twin_snapshot.py`, 4/4 passing).
- **Not yet done**: re-measuring comprehension-monitor's load-bearing status now that its lesion arm can actually
  run. The expected outcome -- not yet measured -- is that a working lesion drops the margin toward chance and sends
  the turn to the abstain/repair path, changing the reply; this is a hypothesis, not a result, until B2d (or a
  targeted re-run) measures it.

**da-mode-drives-response: TRACE-ONLY-TAUTOLOGICAL on 4 of 6 seeds; genuinely trace-load-bearing (not tautological)
on 2 of 6; never reply-load-bearing on this probe.**
- On seeds 43, 44, 100 and 102 the intact mode is already `rest` (DA 0.37-0.39). The only registered diff there is
  `da_drives.reason` `low_engagement` -> `lesion_collapsed`, written by `"lesion_collapsed" if lesion and mode in
  ("rest","neutral")` (`webapp/da_mode_drives_chat.py:459`) -- true regardless of what the lesion actually did.
- A genuine (non-tautological) mode diff (`neutral` -> `rest`, DA 0.52/0.51 -> ~0.046 <!--derived-->) exists on seeds 42 and 101:
  both collapse to the identical ~0.0462 floor <!--derived-->, confirming `silence_snc` always drives mode to rest. Without the
  label the row is trace-load-bearing on 2 of 6 seeds, genuinely.
- The answer is byte-identical on all 6 seeds because rest and neutral both map to an empty suffix
  (`da_mode_drives_chat.py:175-178,251-256`). The reply path is real: on the SAME battery's `emo` turn, the intact
  answer carries the focus-mode suffix " -- worth going further here." on seeds 42, 44, 100 and 101 (DA 0.64-0.71,
  `da_drives_suffix` appended at `server.py:6729-6730`). `emo` would reach the reply path on at most 4 of 6 seeds, so
  a different remap is needed for a clean 6-seed reply probe.

**affect-drives-response: TRACE-ONLY-TAUTOLOGICAL on 6/6, and now double-dead at HEAD.**
- The only registered diff on all 6 seeds, in every battery including adequate6, is `affect_drives.reason`
  `neutral_hold` -> `lesion_collapsed`, written unconditionally by `"lesion_collapsed" if lesion"`
  (`webapp/affect_drives_chat.py:487`).
- This was NOT always a vacuous faculty: `docs/PRODUCTION_INTEGRATION_LEDGER.yaml:739-740` documents that
  pre-2026-09-25, on an affective turn, the same mechanism genuinely changed `answer` ("Wonderful -- X" / "Frankly!
  X" vs an unmarked reply). Two things independently kill the registered probe's ability to show this: (1) the
  `well` probe is mood-neutral even pre-flip (mood ~-0.003 <!--derived--> -> level 0, and `expression_lead()` returns `''`
  unconditionally at level 0 regardless of the flag); (2) since `969894855` (2026-09-25, owner decision,
  `BRAIN_AFFECT_MARKER_SURFACE` default OFF), the lead is computed and recorded but never prepended to `answer` at
  all, on any turn. Fix direction: a mood-engaged turn AND a mouth arm (or the open-output ruler), not this probe.

**confidence-forthcomingness: TRACE-ONLY, INSTRUMENT LIMIT -- corrected from tautological.**
- On the registered `rich_well` probe, the only diffs are the block's own presence: `granted` False -> None,
  `reason` `high_confidence` -> None. This LOOKS tautological (a whether-disable row whose own trace block
  disappears), but a skeptic check found the real cause is different: the probe topic has exactly 1 matching fact
  (`requested_sentences=1`) regardless of confidence, so `apply_cap()`'s reach-vs-floor branch has zero headroom to
  act on -- `reason='high_confidence'` is a REAL value computed from a genuine spiking metacog margin, not a
  disable-branch-written label (contrast da-mode-drives-response above, which is a true tautology).
- **A separately-run, already-existing 6-seed protocol already shows this faculty IS reply-load-bearing**:
  `research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json`, built by
  `verify_confidence_ltm_loadbearing.py`, exercises the faculty with the 2026-09-01 LTM-elaboration protocol
  (`BRAIN_ELABORATE_FROM_LTM_SHARD=1` + `BRAIN_CONFIDENCE_FORTHCOMING=1` on a topic with real elaboration headroom)
  and shows a genuine reply-level causal effect on all 6 seeds (a confident turn keeps more elaborated sentences than
  a low-confidence one). The row's B2b/adequate6 trace-only reading is an artifact of the registered probe's content
  exhaustion, not evidence the faculty is inert.

**affect-coloring and affect-appraisal-interoceptive: TRACE-ONLY, INSTRUMENT LIMIT (no mouth, no render turn).**
- On 6/6 seeds the lesion zeroes the affect differential (`valence_sign '+'->'0'`, `tone_level 2->0`) and empties
  `affect.manner_template` -- the literal instruction to the fluent mouth ("Keep the words ... write ONE short, warm,
  friendly sentence"). `MoodConditionedRenderer` is a declared NO-OP on the StubRenderer
  (`research/runners/affect_production_organ.py:417-421`), and the battery hardcodes `renderer="stub"`.
  `answer`/`abstained`/`recalled_svo` are byte-identical on every checked seed in every battery (12 arms checked
  independently for affect-appraisal-interoceptive alone).
- A cheaper reply-level ruler than a GPU/Qwen arm exists and needs no mouth: `affect_production_organ.py`'s
  `is_feel_query`/`feel_readout` (wired at `webapp/server.py:5022-5027`) writes the mood word directly into `answer`,
  bypassing the renderer entirely -- but no registered probe turn (`well` or `emo`) matches its feel-query regex.
  Earlier mouth-level evidence through a real (linattn) mouth was a directional NO-GO, positive-asymmetric
  (`research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO.md`), so a
  mouth arm is not assured to flip this to GO.

**comprehension-learned-animacy-cue and comprehension-learned-verb-selects: TRACE-ONLY, INSTRUMENT LIMIT (a
well-formed probe never reaches the abstain/repair branch).**
- The lesion silences the learned lexicon, so the probe's noun/verb falls out of competence and `judge()` returns
  None (`comprehension_production_organ.py:731-734` in the pre-fix line numbering). The intact arm comprehends the
  same sentence. Both routes pass the turn through unchanged -- `answer` is byte-identical on 6/6 seeds across every
  battery (18-30 seed x battery cells checked). The gate changes the reply only when `comprehended=False` with no
  known binding (`server.py` around line 6054/6060); a well-formed probe sentence can never reach that branch by
  construction of the probe, independent of the faculty.
- One skeptic ran the proposed counter-probe live (no job queue, no repo edit): an ill-formed/role-reversed
  transitive whose nouns are covered only by the learned cue. This confirms the remap direction is sound; it has not
  been run at 6 seeds as a registered probe.

**multiref-competition: TRACE-ONLY, INSTRUMENT LIMIT (the reply channel is a later turn the probe never asks).**
- On the `hold` (MAINTAIN) turn, the lesion flips `multiref.all_recovered` True->False on 6/6 seeds and
  `multiref.recovered[0]` fox->wolf. `answer`/`abstained`/`recalled_svo` are byte-identical; a full flattened diff
  of the whole response payload found no other differing field. The registered probe
  (`research/runners/lbf_rows/live_organs.py`) compares only `multiref.all_recovered`, never a reply field. The
  reply channel is the DISJOINT hold-query turn (`answer = jq['readout']`, `webapp/server.py` ~5676-5800), which the
  registered probe group never sends.

**open-ended-generation: TRACE-ONLY, INSTRUMENT LIMIT (the distributional ruler never runs a chat turn; the
production-turn instrument reads NO-GO).**
- In B2b and flip-adequate, `measure_open_ended_distributional` (`load_bearing_fraction.py` ~line 1113) builds a
  standalone `SpikingWTASampler` and never calls `_spawn_arm` -- no `answer`/`abstained`/`recalled_svo` is ever
  computed, so the reply cell is structurally UNDEFINED, not merely unchanged.
- The dedicated production-turn reply instrument reads NO-GO on its own terms:
  `research/findings/2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md` (5/6 DEFINED, mean delta
  0.094 under the 0.10 floor <!--derived-->) and its independent successor,
  `research/findings/2026-09-25-open-ended-gated-turn-part-a-6seed-gate-NOT-GO-UNDEFINED.md` (same seed-100
  degenerate null on different code, only 2 of 6 seeds individually clear the floor). No reply-level field has yet
  been found that robustly flips under this lesion across 6 seeds. In `_adequate6`, where the row was measured on
  the `rich_open` chat turn with `answer` among its fields, it reads `pass` on 6/6 -- not load-bearing at either
  level there either.

**source-provenance-honesty: TRACE-ONLY -- but ONLY for the currently-registered probe turn. Corrected: it is
already shown reply-load-bearing at the production default, on a different turn.**
- On the registered `well` (non-chain) probe, the lesion moves `provenance.label` from `perceived` to None
  (abstain-at-tie, default ON since 2026-09-23) on 6/6 seeds; `provenance_framed_text` returns the answer unchanged
  for both PERCEIVED and None on the non-chain path (`research/runners/source_provenance_honesty.py:175-193`,
  applied at `server.py:6851-6863`). `answer`/`abstained`/`recalled_svo` are byte-identical there on every seed.
- **The original write-up's reason was wrong, and its "proposed fix" describes work that already shipped.** The
  hedge that would surface on the reply, `BRAIN_SOURCE_MONITORING_FRAMES_HONESTY`, is NOT default-off: it and the
  underlying `BRAIN_SOURCE_PROVENANCE_HONESTY` organ flag were both flipped to default-ON on 2026-09-01 (commits
  `bedb9ad6e`, `e73bea486`) and remain default-ON at HEAD (verified live, both gate functions return True with the
  env fully unset).
- **On a chain-route (derived-answer) turn -- which the registered probe never sends -- the lesion DOES change
  `answer` under real production defaults**, already measured 6/6 GO:
  `research/findings/2026-09-01-source-provenance-honesty-129-default-on-flip-GO.md`
  (`research/findings/raw/_source_provenance_honesty_flip/flip_verify.json`): intact chain answer "I believe the
  deer eats the worm, but I reasoned that myself rather than being told it directly..." vs. lesioned chain answer
  "I derived this from: wolf hunt deer; deer eat worm. the deer eats the worm...". So this row's correct label is
  "trace-only for the registered non-chain probe; already reply-load-bearing at the production default on the
  chain-route turn" -- not a faculty that cannot reach the reply, and not comparable to the genuine instrument-limit
  rows above.
- Before 2026-09-23 the tie was a host coin flip, which could spuriously hedge a perceived fact -- this also
  explains `_adequate6`'s 4/6 against B2b's 6/6.

**swap-drives-response: registered UNDEFINED (not-exercised) in every canonical battery; a separate,
already-completed, preregistered probe shows REPLY-LOAD-BEARING on 6/6.**
- The default/registered batteries (b2b0924-base, allfixes2, `_adequate6`) score this row `not-exercised` on 6/6
  seeds because the `held` turn is answered by a role-binding repair short-circuit before `swap_drives` is ever
  attached -- `LB_SWAP_DRIVE_PROBE` is genuinely excluded from every canonical battery today.
- Separately, `LB_SWAP_DRIVE_PROBE` (reply-level, preregistered) has already been run on all 6 canonical seeds at
  one single, verified git revision (`7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`; `git_dirty:false`; both manifest
  checks true on every seed): `load_bearing:true`/`verdict:regressed` on EVERY seed, `answer` genuinely differing
  ("Setting the held thread aside -- On cat, then -- the cat eats the fish" vs "the cat eats the fish"), backed by a
  real analog mechanism difference (`mm_peak`/`boost_max`), clean null controls (0 diffs) and clean no-swap-due
  contrast turns (0 diffs, with the preregistration's own honest caveat that 3 of the 4 compared fields on those
  contrast turns are pass-by-construction -- only `answer` is a free field there, and it reads 0 diff as required),
  and `lesion_reproduced:true`. Only the seed-42 leg is committed to git; the other five sit as untracked artifacts
  in a working tree at the same revision. This is genuine evidence, cited here, not folded into any battery's
  registered denominator.

**wm-binding-advanced: UNDEFINED (not exercised) -- two remediation attempts have already been tried and neither
closes it.**
- Confirmed across three independent battery runs: every seed of b2b0924-base, allfixes2 and `_adequate6` reads
  `verdict='not-exercised'`. Root cause: the registered `held` turn ("the wolf watches the owl") never populates a
  `multiref` key at all -- the organ's hand-typed referent lexicon does not admit `owl`, and the turn exits through
  the comprehension-repair clarification branch before reaching any organ reply path.
- `LB_WMB_HOLDQUERY_PROBE` (6 seeds + off-flag variants) was already tried and found PASS-BY-CONSTRUCTION: its
  reply is a host template whose only input is the lesioned buffer, so a reply diff is guaranteed once the route is
  reached; the code (`load_bearing_fraction.py` ~1794-1806) deliberately overwrites its own verdict to
  `integrity-smoke`/`load_bearing=None` so it cannot enter the load-bearing count. A third, un-mentioned probe was
  also already tried with the same outcome. No remediation for this row is open; a genuinely new probe is needed,
  not a re-run of either existing one.

## Instrument limits (defects of the measurement, not verdicts on faculties)

1. **No mouth.** The renderer is hardcoded to the stub, so any faculty whose reply channel is the Qwen manner
   instruction is invisible (affect-coloring, affect-appraisal-interoceptive).
2. **Single probe turn.** Effects that land on a later turn (multiref's hold-query) or need a render turn are not
   sampled.
3. **Probes that never reach the reply path by construction of the sentence chosen**, independent of the faculty:
   the comprehension cues on a well-formed sentence, confidence on a content-exhausted turn, da-mode on a turn that
   never lifts DA into focus.
4. **Lesion-written labels count as decision evidence.** `*.reason = lesion_collapsed`, `reconsolidation.action =
   lesioned_nowrite`.
5. **An organ exception inside an arm was scored as a regression** (comprehension-monitor) -- now fixed at the code
   level; the harness-side rule (an `error` key anywhere in a compared field's parent dict makes the cell class C /
   UNDEFINED, never `regressed`) is pre-registered for B2d.
6. **Revision.** B2b and the flip batteries predate the 2026-09-25 marker retirement. affect-marker-spiking-wta's
   4/6 reply hits ride the "Gladly! " prepend, which HEAD no longer surfaces by default.
7. **Out-of-chat rulers.** The open-ended distributional ruler never produces a reply.
8. **Registered probes are sometimes narrower than the faculty**, and a wider probe has SOMETIMES already been run
   and shows GO (confidence-forthcomingness's LTM-elaboration protocol; source-provenance-honesty's chain-route
   turn; swap-drives-response's dedicated probe) -- these are not open research questions, they are unmerged
   measurements.

## What changes (this session; what is still open)

1. **Done, this session: fixed comprehension-monitor's lesion-twin crash.**
   `research/runners/comprehension_production_organ.py`'s `_snapshot_rest` now checks `comp is self.comp` before
   taking the shared-pool-snapshot branch, so the standalone lesion twin always settles its own rest state. Test:
   `tests/test_comprehension_lesion_twin_snapshot.py` (4/4 passing; verified to fail on the pre-fix condition and
   pass post-fix; the intact path's snapshot is proven byte-identical to the shared pool's own snap, unchanged by
   the fix). **Not done**: re-measuring the row's load-bearing status now that the lesion can run.
2. **Pre-registered for B2d** (`research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md`):
   a reply-level secondary score (S2) beside the trace fraction (S0/S1), self-describing decision fields dropped
   from S1 (`research/coordination/b2d_decision_fields.json`), and the organ-error rule (comprehension-monitor's
   crash pattern) as A1.2 rule 7 -- an `error` key in the probe-turn parent dict of any compared field, in any arm,
   makes the cell class C / UNDEFINED, never `regressed`, for S0 as reported and for S1 and S2. This finding's
   corrections to confidence-forthcomingness and source-provenance-honesty are reflected in B2d's own probe-remap
   plan (see that document's "Probe remaps," added this session).
3. **Open, not yet run:** probe remaps for the genuine instrument-limit rows -- da-mode (a turn whose intact DA
   reaches focus on all 6 seeds; `emo` only reaches 4/6), multiref (`hold` -> a hold-query turn), the comprehension
   cues (an ill-formed probe using a learned-only noun/verb, verified live by one skeptic but not run at 6 seeds),
   open-ended-generation (the production-turn instrument, which currently reads NO-GO, not the distributional
   ruler). Each needs a dev-seed verification before a 6-seed run (B2d's plan lists this as a pre-gate step).
4. **Open, not yet run:** a GPU arm with the Qwen2.5-0.5B mouth (greedy decoding, intact-intact null) for the
   mouth-channel rows (affect-coloring, affect-appraisal-interoceptive, and at HEAD affect-drives-response), via
   `tools/gpu_queue.sh`. Earlier evidence through a different real mouth (linattn) was a directional NO-GO,
   positive-asymmetric, so this is not assured to flip to GO.
5. **Wording.** The 2026-09-24 flip finding's honesty line ("lesioning ... changes the chat reply") describes the
   reply level; what it measured was the trace level -- 9-11 of its 24 core rows (depending on the
   comprehension-monitor correction) do not change the reply on the registered probe. `docs/TERMS.md` now defines
   "load-bearing" with this code condition (added same-cycle as this finding).

## What this does NOT show

- **Trace-only does not mean inert.** Of the 9 genuinely trace-only B2b rows (comprehension-monitor excluded as
  UNDEFINED), 7 have a reply path the battery does not exercise (instrument limits) and 1 (source-provenance-honesty)
  is already shown reply-load-bearing on a different, production-default turn. Only the 2 tautological rows
  (da-mode-drives-response, affect-drives-response) currently have no demonstrated reply channel on ANY measured
  turn under the production default.
- **The reply-level numbers are a lower bound for mouth-channel faculties**, because the stub hides the manner
  instruction. For affect-marker at HEAD they are an OVER-count (the prepend is retired).
- **Reply-level load-bearing means the faculty's neural decision selects what the user sees.** It does not mean the
  brain composed the words. The stub surfaces are host templates. The "who owns the computation" host-decision
  share (plan 2026-09-25) is a separate measurement; S2 in the B2d prereg is explicitly named as an input to that
  plan's seam map, not a substitute for it.
- **All measurements are functional read-outs.** Nothing here bears on phenomenal experience.

## Reconciled scorer and skeptic disagreements

- **episodic-memory seed 102 / discourse-register seed 102.** Scorer A counted the 2026-09-29 re-run cells as core;
  scorer B initially excluded them for having landed mid-session. Both cells' `lb.json` files are confirmed landed
  (12:15 and 14:07 respectively) as A1.4 re-runs, both 6/6 at trace and reply level. Whether they count as "the one"
  A1.4 re-run under B2b's own procedure is an open board ruling on branch research/b2-rulings-2026-09-29, not
  resolved by this finding.
- **Denominators.** Scorer A used 36 coverable rows, scorer B used 34 exercised rows (swap-drives-response and
  wm-binding-advanced are not-exercised on every seed). Once s102 is in, both give the same numerator: trace core
  28 (including open-ended-generation), reply core 17.
- **Reply field set.** 8 fields (scorer A) vs 3 primary fields (scorer B) agree on all 630 cells; the secondary
  fields changed no verdict.
- **adequate6 open-ended-generation.** Confirmed not load-bearing at either level there (`rich_open` is a genuine
  chat turn with `answer` among its compared fields, verdict pass on 6/6) -- not UNDEFINED as one scorer initially
  read it.
- **prospective-memory, adequate6 0/6 vs B2b 6/6.** A probe difference (`pmem_form`/`held` field
  `prospective.held`, a compile-time literal True, vs B2b's `LB_PMEM_DRIVE` remap, `pmem_cue`/`prospective.fired`),
  not a code change in the faculty.
- **source-provenance-honesty, 4/6 (adequate6) vs 6/6 (B2b).** The abstain-at-tie default flip on 2026-09-23; before
  it, the tie was a host coin flip that happened to land on `perceived` on 2 of 6 seeds in the earlier battery.
- **Reported 0.949/24 and 0.603/15.** <!--derived--> Both come from the flipdefaults shards
  (`research/findings/2026-09-24-flip-validated-fixes-production-default-battery-GO.md`), which have raw arms and
  reproduce exactly; not from `_adequate6`, which is a different, earlier battery (0.853/20 <!--derived-->).
- **Missed by the first-pass scorers, caught by adjudication + skeptic review:** the comprehension-monitor crash
  (now fixed), the da-mode verdict resting on `reason` on only 4 of 6 seeds (genuine on the other 2, never
  tautological there), and the fact that confidence-forthcomingness, source-provenance-honesty and
  swap-drives-response each already have unmerged evidence that changes their classification.

## Terms checked against docs/TERMS.md

- **"load-bearing"** is added to `docs/TERMS.md` in this cycle with the code condition this finding uses (reply
  field differs beyond the intact_a-vs-intact_b null, reproduced; otherwise "trace-level load-bearing (decision
  field)"). Every load-bearing claim above already carries that qualifier explicitly (REPLY-LOAD-BEARING vs
  TRACE-ONLY vs UNDEFINED) rather than the bare word.
- **"lesion"**: comprehension-monitor's lesion arm did not hold prior to this session's fix (it raised before
  producing a measurement at all) -- this finding calls it UNDEFINED, never "lesion", for every prior battery.
- **"byte-identical"**: used only where the `answer` string was compared exactly in the retained JSON, as stated in
  "Method" above; no instance in this document is inferred from reading code alone.
- **"GO"**: reserved for the gate's own verdict (confidence-forthcomingness's LTM-elaboration protocol,
  source-provenance-honesty's 2026-09-01 flip, swap-drives-response's dedicated probe) -- none of those GOs are
  claimed as part of THIS finding's own measurement; they are cited as prior, separately-verdicted evidence.
