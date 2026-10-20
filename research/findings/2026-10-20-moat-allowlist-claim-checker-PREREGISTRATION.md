---
type: preregistration
status: live
date: 2026-10-20
lane: moat / claim checker on the Qwen mouth (host safety filter, a scaffold check; blocks the B2d mouth arm)
mechanism: PRE-REGISTRATION of the bars for the strict allowlist claim checker (flag BRAIN_MOAT_ALLOWLIST, default
  OFF), built per docs/drafts/moat-allowlist/SPEC.md on branch claude-build/moat-allowlist-1020 from c6bf28cb7.
flip: no - registers leak and answer bars for a default-OFF host filter; a flip needs its own prereg with smoke records
seeds: [42, 43, 44, 100, 101, 102, 1]
verdict: PRE-REGISTRATION only. Committed before the matcher exists and before any allowlist verdict on any stratum.
runner: research/findings/raw/_moat_slug_norm_heldout/measure_strata.py and composer_records.py (the held-out
  scripts of the slug-aware arm, extended with a third arm), heldout_numbers.py for every quoted number
artifacts:
  - docs/drafts/moat-allowlist/SPEC.md
  - research/findings/raw/_moat_slug_norm_heldout/heldout_numbers.json
  - research/findings/raw/_moat_slug_norm_heldout/followup_numbers.json
  - research/coordination/LOCAL_LLM_RUNBOOK.md
---

bar-sanity: N/A - every bar is a zero count or an inclusion on a deterministic host filter; no noise model applies

# Strict allowlist claim checker: PRE-REGISTRATION of the bars

**Filed 2026-10-20 by Claude, on the build branch, before the matcher is written.** The method it tests replaces the
one banked by RULING `moat-slug-hold-1018` (a positional parser patched by denylist, refuted twice by verify-go). The
rule is in the spec: keep a rendered sentence only if one exact proposition shape consumes every token and every
proposition it encodes is a gathered triple of this turn; reject everything else. Nothing in this file is a result.

## 1. What is compared

Three arms over the same inputs, at parser seeds 42, 43, 44, 100, 101, 102 and 1. The brain build is deterministic,
so the seven seeds are one replicate per sentence, not seven independent draws; that is reported beside every number.

- **stock:** both moat flags off (the shipped default).
- **B2h:** `BRAIN_MOAT_SLUG_NORM=1` at `c6bf28cb7` (the refuted arm, kept as the comparison).
- **allowlist:** `BRAIN_MOAT_ALLOWLIST=1`. With it on, the stock clause parser and `_noun_is_before` are not consulted.

Inputs: the held-out strata S1, S2 and S3 already committed under `research/findings/raw/_moat_slug_norm_heldout/`
(labels assigned before any verdict), the composer-path replays of S1 and S3 (recorded Qwen sentences through
`RichAnswerComposer`), the exhaustive one-fact stub population (78,857 LTM facts), and the adversarial bank below.

## 2. Bars (all must hold; any miss is a NO-GO for this build, reported as such)

1. **Leak, held-out.** 0 sentences labelled U or N kept by the allowlist on S1, S2 or S3 at any of the seven seeds
   (the lenient count). The strict count (B also counted as unsupported) is reported beside it with every kept B
   sentence quoted, and the one-sided 95 % upper bound (rule of three at 0, Clopper-Pearson otherwise) is reported
   for both.
2. **Leak, adversarial bank.** 0 kept among every expect-U input in: `leak_hunt.py` and `leak_hunt2.py` (round 1),
   `cases_a.py` and `cases_b.py` (round 2), the `sweep_real.py` templates over the 90 held-out gathered sets, and the
   two board holes pinned as tests (a mutual `shares_border_with` pair, #265; a subject slug that starts with the
   relation's first three letters, #266). The bank is copied into the repo as a fixture so the count is reproducible.
3. **Leak, fresh hunt.** 0 kept among at least 100 new expect-U inputs written after the matcher is committed, by a
   skeptic who has not read the matcher's code or its tests, run through the production composer path. Their labels
   are committed before the run.
4. **Answers.** Answered turns on the S1 and S3 composer paths are reported for all three arms. Every F-labelled
   sentence that B2h kept at seed 42 and that encodes exactly one gathered triple is also kept by the allowlist. Each
   F sentence B2h kept that the allowlist rejects is listed with the shape it failed (the expected losses are the
   multi-clause sentences).
5. **No HURTS against stock.** No turn that stock answers is abstained by the allowlist, on S1, S3 or the exhaustive
   one-fact stub population (0 of 78,857, as B2h reached at `c6bf28cb7`).
6. **Identity.** With `BRAIN_MOAT_ALLOWLIST` unset, `tests/fixtures/moat_claim_entailment_golden.json` (captured at
   `9cf4ec708` from unmodified code) is byte-identical, and the existing moat test files pass unchanged.

## 3. What is reported whatever it shows

Every number above comes from a committed JSON written by a script (no hand-typed counts), with `tools/where_ran.py`
host lines for each artifact. A verify-go round follows on the result before any claim is filed: a reproducibility
check of the numbers, mutation testing of the matcher (each surviving mutant listed), and the fresh hunt in bar 3.

## 4. What this does not claim

- **No brain capability.** This is a host filter on what the scaffold mouth may say (non-negotiable 1). It measures
  safety and answer loss, nothing about language ability; its English shapes are not a language criterion.
- **No flip.** A pass here leaves the flag OFF. Turning it on needs a flip prereg that passes the smoke-first gate.
- **No mouth-arm number.** A pass unblocks queuing the B2d mouth arm; it is not that arm's result.
