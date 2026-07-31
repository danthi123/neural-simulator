---
type: plan
status: live
date: 2026-05-26
---

# Direction 4 PRODUCTION Adversarial Reviewer Prompt (pre-staged 2026-05-26)

Status: Direction 4 bugfix SMOKE = FULL PASS 6/6 (commit efbad3d). All cells multi-seed clear 0.80 bar with margins +0.175 to +0.200 (L=5 OI = 0.983 vs the 0.80 bar). DRAMATICALLY outperforms D5 hybrid at the L=5 cell (0.983 vs D5 0.790 BOUNDARY). D4 PRODUCTION decisive multi-seed run launched ~06:35 EDT 2026-05-26 (full scale; ~7-15 hr ETA; watcher b6bvjv9tg chains the cross-bridge probe on training completion).

This is the pre-registered, pre-staged adversarial reviewer prompt for when D4 production completes.

## Inputs

- Bug fix: research/findings/raw/direction_4_bridge_builder.py (added _DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS map at 100k offsets per bridge)
- Bugfix smoke result: research/findings/raw/direction_4_5bridge_smoke_bugfix.json + .log
- Bugfix smoke probe: research/findings/raw/direction_4_cross_bridge_bugfix_smoke.json + .log
- Production training: research/findings/raw/direction_4_5bridge_production_bugfix.json + .log
- Production probe: research/findings/raw/direction_4_cross_bridge_production_bugfix.json + .log
- D4 bug-fix commit: efbad3d
- D4 NEGATIVE invalidation finding: research/findings/2026-05-26-DIRECTION-4-NEGATIVE-INVALIDATED-same-systematic-cross-bridge-uniformity-bug-as-D5-had.md
- D5 hybrid production reference: pillar n=106 BOUNDARY
- D4 frozen verdict module: research/findings/raw/direction_4_verdict.py
- D4 cross-bridge probe: research/findings/raw/direction_4_cross_bridge_probe.py

## Scrutiny items (9 items)

1. Bug fix correctness: 5 distinct seed offsets at 100k apart; activity vectors DISTINCT across bridges (cos < 0.99). BLOCK if cross-bridge cos > 0.99.

2. Multi-seed reproducibility at production scale: 15 cells trained; 3 seeds in probe; loads {2, 3, 5} with n_trials=200. BLOCK if missing.

3. Smell-test recomputation: independently recompute multi-seed OB+OI from per-seed JSON values; match aggregate within 0.001.

4. OB PASS every cell: each load multi-seed must be >= 0.80; smoke had perfect 1.000.

5. OI characterisation (the key differentiator vs D5): L=2 and L=3 must clear 0.80; L=5 should be >= 0.80 (smoke 0.983; expect production similar). BLOCK only if OI L=2 falls below 0.80.

6. Comparison to D5 hybrid + pillar n=95: D4 should produce OI L=5 >= D5 hybrid (smoke ratio 2.1x). The biology insight: dedicated-pool may be CLEANER for cross-bridge than the hybrid.

7. Anti-cheat: cross_bridge_mode_unification_probe.py byte-unchanged; verify via git log.

8. Builder fix non-default-breaking: omitting label arg defaults cleanly; preserves prior single-bridge behavior.

9. Score-tuning/threshold-tampering check: bar 0.80; seeds [42, 43, 44]; no post-hoc adjustment.

## Verdict

CLEAR (all 9 PASS) or BLOCK (specify failed items + strengthening fix needed).

If CLEAR: pillar n=108 candidate (PASS or BOUNDARY depending on OI L=5).

## Pillar n=108 framing (if CLEAR)

Pillar n=108: Direction 4 dedicated-pool bio_brain_regions cross-bridge composition (5 bridges x V=16 = 80 concepts). [OB / OI characterisation]. Dramatic improvement over D5 hybrid at L=5 OI. The hybrid architecture's shared sparse pool was a workaround for the cross-bridge uniformity bug, not a necessary architectural component.
