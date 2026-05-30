# Direction 7 production adversarial reviewer verdict: CLEAR -- VALIDATED BOUNDARY

Date: 2026-05-30 ~12:00 EDT
Reviewer: independent adversarial review subagent (general-purpose), 9 scrutiny items + independent diagnostic re-run.

## Verdict: CLEAR -- promote pillar n=110 as VALIDATED BOUNDARY

All 9 items addressed with recomputed numbers:
1. Verdict module frozen + untampered (predates result by 3 days; git diff byte-empty; 11/11 grounding tests pass).
2. Smell-test recompute: L=5 OI mean = (0.925+0.700+0.865)/3 = 0.830000 exactly; seed 43 included at full weight, not dropped.
3. "PASS by frozen mean-rule but BOUNDARY in reality" is the HONEST characterization; n=106 (D5) precedent apt; D7 is STRONGER than n=106 (D7 mean 0.830 clears bar; n=106 mean 0.790 did not). VALIDATED BOUNDARY correct.
4. Crash-retrain confound GENUINELY ruled out (reviewer re-ran diagnostic, reproduced exactly; found a more robust argument -- see correction 1).
5. Anti-cheat: 320 distinct concepts; seed 43 cross-bridge cos 0.22-0.50 (distinct, not collapsed); seed-offset fix present; batched-vs-scalar 2.78e-17.
6. seed 43 NOT a broken cell: full shape (16x12800), firing 0.0025 mid-pack (not silent); 98.7KB npz = benign compression variance. Valid boundary measurement.
7. Protocol [42,43,44] x {2,3,5} pre-registered; cache has ONLY 42/43/44 (no gamed seeds).
8. "Envelope bends at V=320" supported by 3 independent signals (whole distribution dropped; 22x variance increase; geometry degraded all seeds); but "LOCATES ceiling" overstates n=3 precision -> soften to direction/trend (correction 2).
9. Protected set byte-empty diff (e739543..HEAD); no autograd; abstention 7/7 PASS.

## Required corrections (documentation only, all applied):
1. Crash-confound argument: "D_spatial NOT retrained" was factually wrong (D_spatial trained post-relaunch per .h5 mtimes). Replaced with stronger argument: seeds 42/44 clean on the SAME post-relaunch bridges where seed 43 degrades -> process-global crash cannot selectively damage only seed 43. Also corrected commit 95af2c8's "12/15 cached before crash" (only A_nouns+B_verbs predate crash; cache-skip reused surviving .h5 artifacts).
2. Headline: "LOCATES the ceiling between V=160 and V=320" -> "begins to bend at or below V=320 (seed-robust at V=160, seed-sensitive at V=320)."
3. BOUNDARY headline preserved (was already correct).

All three corrections applied to the finding doc 2026-05-30 before promotion.
