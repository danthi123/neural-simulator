---
type: finding
status: live
date: 2026-09-08
tags: [scaffold-retirement, rank-1, onebrain-composer, ask_yes_no, dg-ca3, spiking-recall]
verdict: The spiking OneBrainComposer's ask_yes_no now scans ALL (agent,action) blocks for the asserted patient, reaching FULL parity with the host rf composer (strict_parity 1.0, zero residual gap) with the moat intact — closing rank-1 flip gate (2).
---

# Rank-1 flip gate (2): onebrain ask_yes_no multi-block scan — FULL parity with rf, moat intact

## Question

Rank-1 (retire the host `rf` composer → spiking DG-CA3 `onebrain`) de-risked GO on recall
(`2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md`) but carried ONE characterized
gap: on ambiguous SVO cues (same (agent,action), different patient) `ask_yes_no` returned `'unknown'` 75/264
times where the host `rf` scanned the full SVO and said `'yes'`. Root cause (confirmed by reading the code, not
assumed): `ask_yes_no` selected the FIRST (agent,action) block only (`_seq_block` / `_fact_shard_first_match`),
so a stored `(agent,action,patient)` triple whose patient lived in a LATER same-(agent,action) block was missed.
This is a recall-completeness gap (a genuinely-stored fact not found), never a case where `'unknown'` was honest.

## Fix (additive, moat-preserving) — `research/runners/one_brain_composer.py` (commit `4feff352a`)

`ask_yes_no` rewritten to scan EVERY (agent,action) candidate for the asserted patient before concluding:
`_fact_shard_yesno_match` (scans all DG-CA3 shard candidates on the production fast path) and `_host_yesno_match`
(the equivalent full `_read_blocks()` scan when the fast path is off) — each returns the first full-SVO match or
abstains (`None`), NEVER a false `'no'`. The opt-in (default-OFF) `integrated_loop` spiking K-way sequencer path
is untouched (its single-decision limit is explicitly out of scope). Production default `composer_kind` NOT
changed — still `'rf'`.

## Results

<!--derived: all values read from research/findings/raw/_rank1_composer_rebuild/verify_yesno_rung.json-->

Real 404-fact `bridges/developed/scale787/day_33` bundle, seed 42, `SIM_BACKEND=numpy`, production default
`BRAIN_FACT_SHARD_RETRIEVAL=1`:

| ask_yes_no metric | before | after |
|---|---|---|
| strict_parity vs rf | <1.0 (75/264 mismatch) | **1.0000** |
| ob_yes_rate (all 264 SVOs) | 0.716 | **0.99621 (== rf's 0.99621)** | <!--derived-->
| ob_unknown_on_stored | 75 | **1** | <!--derived-->
| ob_no_on_stored (CONFAB if >0) | 0 | **0** |
| unambiguous ob_yes_rate (n=161) | 0.994 | **0.99379** (unchanged) | <!--derived-->

No regression: `query_patient`/`query_agent` strict parity **1.0**, recall **0.9947 / 0.9952** (unchanged);
scramble control collapses 0.99→0.0, attribution 1.0; moat 80/80 abstain out-of-store, 0 confab.

**The single residual `ob_unknown_on_stored=1` is a SOURCE-DATA artifact, not an onebrain gap:** the one SVO
(`atom, share, electron`) returns `'unknown'` because the host `rf` baseline ALSO returns `'unknown'` for it —
the bundle's `facts.json` stores the identical `(atom, share, electron, AFFIRM)` tuple twice. So onebrain's
`ask_yes_no` has ZERO residual gap versus rf on this real bundle.

## Status

**Rank-1 flip gate (2) CLOSED.** With gate (1) (the `onebrain_k_max` load-path thread,
`2026-09-08-onebrain-kmax-loadpath-thread-GO.md`, merge `9bd0d749`) also closed, both characterized gates on the
composer production flip are GO. Remaining before flipping `composer_kind` `'rf'`→`'onebrain'` in production: a
live end-to-end `/api/brain-chat` round-trip on an onebrain bundle (the k_max finding's noted residual), and the
flip mechanism itself (the deployed bundle's manifest pins `'rf'`). Speed: the spiking composer is inherently
slower than the host closed-form `rf` (the accepted speed<faithfulness cost of going brain-based); the multi-block
scan is bounded (candidates per (agent,action) are few). Honesty boundary: all read-outs functional; no
phenomenal claim.
