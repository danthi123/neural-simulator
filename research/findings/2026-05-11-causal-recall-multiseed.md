# Liu 2012-style causal recall multi-seed — 0/3 FAIL (test methodology issue, not P2 broken)

**Date:** 2026-05-11
**Catalog:** D.14 (engram tagging, Tonegawa lab 2012-2015)
**Reference:** Liu et al. 2012 Nature — Optogenetic stimulation of a
hippocampal engram activates fear memory recall.
**Seeds:** [42, 43, 44]
**Verdict:** 0/3 OVERALL PASS (0/3 word-driven, 0/3 causal)

## Per-seed results

| Seed | Tag size | Word ratio | Word | Causal ratio | Causal | Overall | Wall |
|---|---|---|---|---|---|---|---|
| 42 | 20 | 0.57x | FAIL | 0.38x | FAIL | FAIL | 105s |
| 43 | 29 | 1.15x | FAIL | 1.21x | FAIL | FAIL | 118s |
| 44 | 29 | 0.21x | FAIL | 0.81x | FAIL | FAIL | 120s |

Targets: word-driven target/other > 1.3x; causal target/other > 1.5x.

## Multi-seed averages

- Word-driven target/other ratio: 0.64x (FAIL, expected >1.3x)
- Causal target/other ratio: 0.80x (FAIL, expected >1.5x)
- Mean wall clock: 114 sec/seed

## Diagnosis (honest)

This is NOT a failure of the engram-tagging API (P2). It's a
failure of the **single-direction paired-stim training paradigm**
to overcome seed-specific structural bias.

Three reasons:

1. **No contrastive training.** The test trains ONLY "north" →
   motor_N. With only one direction, STDP grows lang→motor_N
   weights but doesn't suppress lang→motor_E/S/W. Random init
   biases dominate.

2. **No topographic prior + FS lateral inhibition.** The
   2026-05-06 Tier 1 BREAKTHROUGH showed that real W↔A binding
   requires:
   - Topographic prior matching cortical somatotopy
     (Pulvermüller 2001-2003)
   - PV-FS lateral inhibition between motor pools
     (Vogels 2011, Hofer 2011)
   - 200-400 events per direction × 4 directions
   This runner uses NONE of those mechanisms.

3. **Per-seed alignment is random.** Seed 42 motor_E wins
   (target_word=7 vs motor_E=18). Seed 43 has reasonable
   structure (target=10 vs other=8.67). Seed 44 target gets
   1/5 the firing of other pools (target=2 vs S=29). The
   "alignment" between language word and target action is
   seed-dependent randomness, not learned binding.

This is the EXACT same finding as the 2026-05-03 permuted-label
control: single-word paired-stim training produces marginally-
above-chance structure that's randomly oriented per seed.

## What P2 engram-tagging actually validates

P2's engram-tagging API is independently validated by:

- `tests/test_engram_tagging.py` — 12 unit tests covering:
  start_engram_recording, commit_engram_tag, stimulate_tag,
  clear_tag_drive, top_k vs threshold_hz selection,
  region_filter, persistence across save/load_checkpoint
- `tests/test_concept_replay.py` — 5 unit tests for the
  P3.1 concept replay mechanism that uses these tags
- P1 trisynaptic loop multi-seed PASS uses the same API for
  CA3 ensemble tagging — 3/3 BIOLOGY-FAITHFUL PASS
- P4.1 positional binding multi-seed PASS uses the same API
  for (word, position) tag distinction — 3/3 PASS

P2's API works. This Liu 2012 behavioral test fails because the
DOWNSTREAM training paradigm (lang→motor pure paired-stim with
no contrastive cross-pool mechanism) is not sufficient to make
the engram tag drive the correct motor pool.

## What it would take to PASS

To replicate Tonegawa 2012 fear-memory result in our architecture:
1. Train all 4 word→motor bindings simultaneously (north→N,
   east→E, south→S, west→W) with cross-pool contrast.
2. Enable topographic bias (`--apply-topographic-bias`) and
   FS lateral inhibition (`--enable-motor-fs`).
3. Scale arch (n_lang_input=2048, n_motor_per_action=500 — the
   Tier 1 BREAKTHROUGH params from 2026-05-06).
4. 200-400 events per direction.

This is essentially what `bio_three_factor.py` already does.
A `validate_causal_recall_v2.py` would build on top of that
trained network, tag CA3 during a final fresh exposure, then
test causal recall via tag stimulation.

## Wall clock

~115 sec/seed for the current (failing) test. Future v2 with
Tier 1 architecture would be ~5-10 min/seed.

## Path forward (deferred)

The Liu 2012 behavioral validation is parked. The engram-tagging
API is independently validated via unit tests + multi-seed P1
+ P4.1 PASS. The behavioral validation can be revisited later
with proper contrastive training.

## Production status

P2 engram-tagging API: PRODUCTION READY (used by P3.1, P4.1, P5).
P2 unit tests: PASS (12 tests).
P2 single-direction Liu 2012-style behavioral test:
DEFERRED (this finding) pending architectural prereqs.
