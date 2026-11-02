---
type: result
status: live
date: 2026-11-02
mechanism: reply-level load-bearing fraction (the #1 metric) at pinned main f2885df70, stub mouth, battery b2d1018
prereg: research/findings/2026-10-20-lbf-at-head-gate-seed-measurement-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
verdict: MEASUREMENT (no bar). Reply-level S2 mean 0.5365853658536586 (22 of 41 faculties change the reply on all six seeds); decision-field S0 mean 0.8208333333333333 (core 32 of 40). COMPLETE 246/246 cells.
runner: tools/b2d_score.py
artifacts:
  - research/findings/raw/_load_bearing/_shards/b2d1018/score_gate6_1102.json
---

# The #1 metric at a pinned main, six gate seeds: reply-level 22 of 41 (stub mouth)

Date: 2026-11-02 · prereg: `research/findings/2026-10-20-lbf-at-head-gate-seed-measurement-PREREGISTRATION.md`
(measurement only, no bar) · pin `f2885df70cfb99a8715c376bca12ac3c2134ef46` · registered command
`tools/b2d_score.py --tag b2d1018 --pin f2885df70cfb99a8715c376bca12ac3c2134ef46 --seeds 42 43 44 100 101 102` ·
score: `research/findings/raw/_load_bearing/_shards/b2d1018/score_gate6_1102.json`.

**This is the stub (template) mouth, not the Qwen mouth.** The Phase A leg "reply-level fraction at HEAD with the
mouth" stays open.

## 1. The numbers (all of them)

Verdict COMPLETE: 246 of 246 coverable cells defined, 0 fail cells, stop-gate row (reasoning-transitive-chat) not
blocked on 6/6.

| level | what it counts | per seed (42, 43, 44, 100, 101, 102) | mean | sd | robust core (6/6) |
|---|---|---|---|---|---|
| **S2, reply-level (the headline)** | lesion changes answer / abstained / recalled_svo | 22/41 on every seed <!--derived--> | **0.5366** | 0.0 | **22** |
| S0, decision-field | lesion changes ANY compared field (organ trace fields included) | 33, 32, 33, 33, 33, 33 of 40 <!--derived--> | 0.8208 | 0.0093 | 32 (union 33) |
| S1 | self-describing fields excluded | UNDEFINED: the fields file is pinned at 5818dc629, not this pin | - | - | - |

Dev read beside it (seeds 7/8, labelled DEV, never the result): S0 33/40 = 0.825 each, S2 22/41 each <!--derived-->.

**Correction to the prereg's wording.** Section 2 item 1 calls S0 "faculties whose lesion changes the reply". The
scorer's S0 reads `load_bearing` over every compared field. 59 of the 203 S0-load-bearing cells leave the reply
unchanged <!--derived-->. The reply-level number is S2. The provisional figure Claude reported to the owner earlier
today as "0.820 with the fuller tests" <!--derived--> is this trace-level S0, not a reply-level reading.

## 2. Per faculty

- **Reply-level 6/6 (S2 core, 22):** affective-tom, bg-action-selection, category-inherit, causal-whatif,
  common-ground-drives, comprehension-monitor, curiosity-followup, d6-hebbian-write, discourse-register,
  episodic-memory, gnw-multistep-deliberation, metacog-monitor, neural-selfid, noncontradiction-gate,
  pragmatic-implicature, prospective-memory, reconsolidation, self-initiated-utterance, surprise-monitor,
  vision-identity-spiking-hmax, worldmodel-forward, xedge-surprise-worldmodel.
- **Decision-field only (S0 load-bearing, reply unchanged on every seed, 10):** affect-appraisal-interoceptive,
  affect-coloring, affect-drives-response, affect-marker-spiking-wta (S0 5/6: s43's intact lead is empty, so the
  lesion has nothing to remove), comprehension-learned-animacy-cue, comprehension-learned-verb-selects,
  confidence-forthcomingness, da-mode-drives-response, multiref-competition, source-provenance-honesty. For five of
  these the S0 diff is self-describing (an organ's own `on` flag or a `reason` the lesion branch writes). The affect
  rows cannot reach the reply through the stub mouth, so for them this may be an instrument limit, not a brain one.
- **open-ended-generation:** S0 6/6 from a distributional plausible-fraction ruler with no intact/lesion arm files;
  S2 UNDEFINED on all six seeds (no reply fields) and dropped from S2's denominator.
- **Not load-bearing 0/6:** da-gated-encoding (engaged: lesion records `lesioned=true`, g 1.0 vs intact ~1.5-1.7,
  genuinely hollow); d5-consolidate, gnw-bus, sleep-replay, spiking-anaphor (no engagement record: the lesion arm's
  probe turn is byte-identical to intact, so these are hollow OR not engaged; engagement UNDEFINED).
- **Not exercised 6/6:** swap-drives-response, wm-binding-advanced (the battery probe does not reach them; the
  adequate swap-drives probe is LOAD-BEARING 6/6 in `research/findings/2026-11-02-swap-drives-gate-6seed-RESULT.md`).
- **Denominator-only rows (Amendment B):** tom-false-belief (host-input, load-bearing 6/6 but not in the numerator),
  reasoning-transitive-chat (stop gate).
- **Not covered (7):** gnw-deliberation, learned-referent, onebrain-xedge, spiking-qroute,
  surprise-salience-snc-afferent, value-driven-choice (thin), one-brain-substrate (mechanism-only).

## 3. Verification (verify-go, four refuters: provenance, instrument re-score, engagement/mass, where-ran)

- **Reproduced.** The pinned scorer re-run on the shards gives a byte-identical result; a hand recompute from every
  `lb.json` agrees per seed.
- **Provenance.** 0 of 294 gate-seed cells fail the clean rule: every sidecar is at the pin, `git_dirty` false,
  manifest verified at start and exit, no BRAIN_ key but BRAIN_CHAT_SEED in the recorded env and no FIX_ENV flag in
  any argv `--env` payload, every arm in one run with its `lb.json`, no cell started before 10-28. This battery's
  first gate run (10-21) carried four extra BRAIN_ flags; those cells were re-run clean and replaced (RULING
  `lbf-replace-1028`; commits c9d8d2d9a, d30ae6303, e04c756e9, 7848fd823). The replace filter covered seeds
  100-102 only; seeds 42-44 were audited directly and are clean.
- **Sham null.** intact_a == intact_b on the probe turn in all 240 armed cells; lesion == lesion.rep0 in every
  cell. No reply change is nondeterminism.
- **Where it ran.** `tools/where_ran.py` over the 294 `lb.json`: pool = 172, workstation = 122 (node-dl4g243,
  node-dl5d243, node-mxl1462drl; dant123-wk). No cell mixes hosts; intact replies are identical across hosts in all
  33 testable same-turn groups. The registered command passes no `--corpus`, so the corpus-hash rule was not checked
  and `research/coordination/b2d_corpus_sha256.tsv` has no row for pool40 (node-mxl1462drl): host-equivalence rests on
  identical outputs, not a hash.
- **S2 denominator defect.** S2 counts swap-drives-response and wm-binding-advanced as `pass` although S0 records them
  not-exercised. Under the engagement rule they are UNDEFINED; excluding them gives S2 = 22/39 = 0.564 <!--derived-->.
  Reported, not substituted: the registered scorer's 0.5366 stands as the number. FAILURE_LOG line lands
  with the fix.
- **Not like-for-like with earlier readings.** 09-21 (0.853, core 20, 26 faculties) and 09-29 (trace 0.843 core 28; <!--derived-->
  reply 0.535 core 17) <!--derived--> used a different roster, scorer and pin. Read the reply-level mean as unchanged (0.535 ->
  0.537) <!--derived--> and the reply core as larger (17 -> 22) only with that caveat; neither is a measured regression or gain.

## 4. What this means for the mission

Twenty-two faculties, on the stub mouth, change what the brain says when lesioned, on every gate seed; that is the
integration-first unit of progress. The gap between S0 and S2 is ten faculties whose organs act but whose effect
never reaches the words: five are affect rows that need the real mouth (the Qwen-mouth arm), the rest are
comprehension / confidence / provenance rows whose output stops at a decision field. Next: (1) the Qwen-mouth arm of
this battery once the claim checker unblocks it (RULING `moat-slug-hold-1018`); (2) engagement traces for the four
lesions with no record, so their 0/6 reads as hollow or as unexercised; (3) fix S2 to score not-exercised rows
UNDEFINED.

## 5. Honesty boundary

Every number is a functional read-out (reply fields, organ trace fields, lesion flags). Nothing here asserts
phenomenal experience.
