# R-iii SWR reactivation probe (6-seed) — the 2026-05-24 hypothesized fix ("add the missing CA3 drive to trigger_swr_replay") is REFUTED as the sole fix: the direct-drive specificity is an ARTIFACT (collapses to chance in the post-seed COMPLETION window, even with a strengthened+trained CA3 autoassociator). The R-iii bottleneck is DEEPER than the missing drive — a sustainable recurrent attractor (the full D.13 training regime) or the consolidation/decoder path. The adversarial post-seed control caught what would have been a false "diagnosis-confirmed." NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_swr_reactivation_probe.py` (minimal fresh hippocampus substrate + engram tagging + SWR, numpy-CPU ~9s/seed). NO `sim/` edit.
**Verdict:** INCONCLUSIVE / naive-fix-REFUTED (6-seed) — an honest diagnostic that corrects the 2026-05-24 hypothesized fix.

## Why this ran
The #5 imaginative-recombination gate flagged the fully-spiking SWR generative-replay loop as a documented decisive NEGATIVE (2026-05-24, at chance) and recommended the gate's cheap "probe-before-rebuild": the diagnosed failure-mode #1 was that the (c) loop's `trigger_swr_replay` OMITS the explicit CA3 drive (`stimulate_tag`) that the validated Phase 1.3 `run_concept_replay_phase` uses (confirmed at the code level: `trigger_swr_replay` opens the `ca3_swr_burst` gate but never calls `stimulate_tag`; `run_concept_replay_phase` calls `bridge.stimulate_tag(...)`). This probe tests whether adding that drive makes SWR reactivation SPECIFIC.

## The result — 6-seed (a minimal fresh hippocampus substrate: EC/DG/CA3/CA1 + engram tags on CA3)
```
FULL-window capture (steps 0-99, INCLUDING the direct-drive window):
  NO-DRIVE spec = -0.006      WITH-DRIVE spec = +0.523     <- looks like a clean "confirmation"...
POST-SEED completion window (steps 20-99, EXCLUDING the drive -- pure recurrence completion):
  NO-DRIVE spec = -0.029      WITH-DRIVE spec = -0.040     <- COLLAPSES to chance
POST-SEED window WITH a strengthened+trained CA3 autoassociator (recurrent w=5.0 + Hebbian encoding):
  NO-DRIVE spec = -0.035      WITH-DRIVE spec = -0.036     <- STILL at chance
```
(spec = post-replay CA3 activity overlap with the CORRECT engram minus the mean overlap with the OTHER engrams.)

## The honest finding (the adversarial-verify catch)
The FULL-window +0.523 "specificity" was an ARTIFACT: it counted the direct-drive window (steps 0-19), where `stimulate_tag` is literally firing the tagged neurons — so of course they overlap the correct tag. When the capture is restricted to the POST-SEED window (steps 20-99, pure recurrence-driven COMPLETION), the specificity COLLAPSES to chance for both conditions — the seeded ensemble does NOT sustain via CA3 recurrence. This holds EVEN with a strengthened+trained autoassociator (recurrent weight 5.0 + Hebbian during encoding). 

⇒ the naive 2026-05-24 fix ("add the CA3 drive to trigger_swr_replay") is REFUTED as the sole fix: the drive makes the neurons fire WHILE driven (an artifact), but does NOT produce sustained recurrence-driven pattern completion. The R-iii bottleneck is DEEPER — the CA3 autoassociator on this substrate/config does not form a sustainable attractor for the stored ensembles.

Had the probe stopped at the full-window +0.523, it would have recorded a FALSE "diagnosis-confirmed." The adversarial post-seed control (isolating completion from the direct drive) prevented the over-claim — the same discipline that caught the derangement bugs, the multi-level overclaim, and the #4 WTA reframe earlier this session.

## Honest scope + the refined R-iii diagnosis
- This is a MINIMAL fresh substrate with a SHORT Hebbian encoding — NOT the full D.13 Marr-autoassociator training regime (the 2026-05-11 pattern-completion validation needed ~400 training events + a direct-CA3-drive-of-PARTIAL-pattern protocol + specific params to reach completion cos 0.748). A fully-trained autoassociator MIGHT sustain completion; this probe shows the naive short-encoding + drive does NOT.
- So the refined R-iii diagnosis: sustained specific SWR reactivation needs BOTH (i) a genuinely-trained CA3 recurrent attractor (the D.13 regime) AND (ii) a seeding drive — the drive ALONE (failure-mode #1) is insufficient. The proper R-iii test needs either the D.13 training regime on this substrate or the original consolidated substrate (the deferred rebuild), plus the decoder path (failure-mode #3).
- The fully-spiking SWR generative-replay loop remains a characterized DEEP boundary (probe-before-rebuild done; the naive fix refuted; the real fix is a heavier trained-attractor rebuild). This does NOT affect the propositional generative-replay CORE (the b2 proposer, GO) or the R-i/R-ii recombination mechanisms (GO this session, which use the FHRR algebra, not the SWR loop).

## What this establishes
The R-iii probe (the gate's specified cheap diagnostic) is done: the naive "add the CA3 drive" fix is REFUTED (the direct-drive specificity is an artifact; post-seed completion is at chance even with a strengthened autoassociator). The R-iii deep boundary stands with a REFINED diagnosis (needs a sustainable trained attractor + drive, not just the drive) and a precise next step (the D.13 training regime / consolidated-substrate rebuild + the decoder). The adversarial post-seed control is the load-bearing methodology contribution.

## Files
`research/runners/_riii_swr_reactivation_probe.py` (`--train` strengthens the autoassociator; `run_seed(seed, train_ca3=)`). Prior: `2026-05-24-c-generative-replay-decisive-NEGATIVE-*.md`; the #5 gate `2026-07-08-imaginative-recombination-frontier-research-gate.md`; the propositional core `2026-06-23-genfrontier-b2-generative-replay-derisk`; the R-i/R-ii recombination GOs (this session).
