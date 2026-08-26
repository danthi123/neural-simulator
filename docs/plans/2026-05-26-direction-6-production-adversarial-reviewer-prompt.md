---
type: plan
status: live
date: 2026-05-26
---

# Direction 6 PRODUCTION Adversarial Reviewer Prompt (pre-staged 2026-05-26)

Status: Direction 6 (D4 dedicated-pool extended to V=32 per bridge x 5 bridges = 160 cross-bridge concepts) SMOKE in flight ~13:30 EDT 2026-05-26 (background; watcher bgkp5r743; ETA 2.5 hr). Production launch conditional on smoke PASS/PARTIAL per the runner discipline.

Pre-registered reviewer prompt for when D6 PRODUCTION completes.

## Inputs

- D6 builder + runner + vocab + verdict + probe at research/findings/raw/direction_6_*
- D6 grounding pin: tests/test_direction_6_grounding.py
- D6 frozen verdict module: research/findings/raw/direction_6_verdict.py
- D6 smoke training: research/findings/raw/direction_6_5bridge_smoke.json + .log (will exist when smoke completes)
- D6 production training: research/findings/raw/direction_6_5bridge_production.json + .log (will exist when production completes)
- D6 cross-bridge probe: research/findings/raw/direction_6_cross_bridge_*.json + .log
- D4 pillar n=108 reference: research/findings/raw/direction_4_cross_bridge_production_bugfix.json (V=80 baseline; OI L=5 = 0.977)
- Pillar n=95 G.20 sparse reference: research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md
- D6 commit history: 6becaa5 + 724048c (infrastructure)

Predicted result via FHRR algebra capacity ratio: if D4 V=80 hits boundary at L=6/L=7, D6 V=160 (2x vocab) should hit boundary at L=3/L=4 (doubling V drops boundary ~2 rungs). So D6 production expected to PASS at L=2 and L=3, BOUNDARY at L=4, FAIL at L=5+.

## Scrutiny items (9 items; ALL must PASS for CLEAR)

1. Bug fix correctness: 5 distinct seed offsets at 100k apart; activity vectors DISTINCT across bridges (cos < 0.99). Confirm via diagnostic probe analog of D4 distinctness probe. BLOCK if cross-bridge cos > 0.99.

2. Multi-seed reproducibility at production scale: 15 cells trained (5 bridges x 3 seeds); 3 seeds in probe; loads {2, 3, 5} with n_trials=200. BLOCK if missing.

3. Smell-test recomputation: independently recompute multi-seed OB+OI from per-seed JSON values; match aggregate within 0.001.

4. OB characterisation: expect OB perfect (1.000) at L=2 and L=3; possibly PARTIAL at L=5 given doubled vocab. BLOCK only if OB at L=2 falls below 0.80 (would contradict the FHRR capacity prediction).

5. OI characterisation (the key scaling test): expect OI PASS at L=2, possibly L=3; BOUNDARY at L=5 (per FHRR capacity prediction). BLOCK only if OI at L=2 falls below 0.80.

6. Comparison to D4 pillar n=108 (V=80): D6 (V=160) should hit boundary 1-2 rungs lower than D4 per capacity ratio. If D6 actually beats D4 at L=5, that's surprising (worth flagging but not blocking - might be characterisation noise).

7. Anti-cheat: parallel-matching primitive (cross_bridge_mode_unification_probe.py from pillar n=95) byte-unchanged via git log. BLOCK if non-empty diff on primitive.

8. Builder fix non-default-breaking: omitting label arg defaults cleanly per the D4 fix pattern. BLOCK if breaks pre-existing code paths.

9. Score-tuning / threshold-tampering check: bar 0.80; seeds [42, 43, 44]; no post-hoc adjustment. BLOCK if any tampering found.

## Reviewer verdict

End with EXACTLY ONE of:
- CLEAR: all 9 items PASS. Pillar n=109 candidate APPROVED. Specify whether PASS or BOUNDARY based on OI L=5.
- BLOCK: specify failed items + strengthening fix needed.

## Pillar n=109 framing (if CLEAR)

Pillar n=109 [PASS or BOUNDARY]: Direction 6 D4-architecture extended to V=160 cross-bridge (5 bridges x V=32 = 160 unique concepts on dedicated-pool bio_brain_regions). [OB/OI characterisation]. Confirms FHRR algebra capacity ratio (capacity proportional to N_dim/V) on the cleanest cross-bridge substrate. Matches G.20 sparse pillar n=95 vocab scale (V=160) on dedicated-pool architecture with [comparable/better/worse] L=5 OI.
