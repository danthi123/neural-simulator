# Direction 7 PRODUCTION Adversarial Reviewer Prompt (pre-staged 2026-05-27)

Status: Direction 7 (D6 dedicated-pool extended to V=64 per bridge x 5 bridges = 320 cross-bridge concepts; vocab byte-identical to Direction M G.20 sparse production deliverable) SMOKE in flight 08:28 EDT 2026-05-27 (background; bash watcher b29xs5cm9; ETA 2-3 hr). Production launch conditional on smoke PASS / PARTIAL per the runner discipline.

Pre-registered reviewer prompt for when D7 PRODUCTION completes. Pillar n=110 candidate if reviewer CLEARs.

## Inputs

- D7 builder + runner + vocab + verdict + probe at research/findings/raw/direction_7_*
- D7 grounding pin: tests/test_direction_7_grounding.py (11/11 PASS at commit 72e8964)
- D7 frozen verdict module: research/findings/raw/direction_7_verdict.py
- D7 smoke training: research/findings/raw/direction_7_5bridge_smoke.json + .log (will exist when smoke completes)
- D7 production training: research/findings/raw/direction_7_5bridge_production.json + .log (will exist when production completes)
- D7 cross-bridge probe: research/findings/raw/direction_7_cross_bridge_*.json + .log
- D6 pillar n=109 reference: research/findings/raw/direction_6_cross_bridge_production.json (V=160 baseline; OI L=5 = 0.987)
- D4 pillar n=108 reference: research/findings/raw/direction_4_cross_bridge_production_bugfix.json (V=80 baseline; OI L=5 = 0.977)
- Pillar n=95 G.20 sparse reference at V=160: research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md (BOUNDARY 0.790)
- G.20 sparse 320-tier reference: research/findings/2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md (per-bridge 98.4% PASS)
- D7 commit history: 72e8964 (infrastructure)

## What the FHRR algebra capacity-ratio prediction says vs what D6 actually showed

Predicted (per FHRR algebra capacity ratio: capacity proportional to N_dim/V; doubling V drops boundary ~2 rungs):
- D4 V=80 OI L=5 = 0.977 (above boundary)
- D6 V=160 should hit boundary at L=3/L=4
- D7 V=320 should hit boundary at L=2

Actual (decisive multi-seed at pillar n=109 D6):
- D4 V=80 OI L=5 = 0.977 (confirmed; pillar n=108)
- D6 V=160 OI L=5 = 0.987 (slightly BETTER than D4 at half the vocab; SHATTERED prediction)

Updated prediction for D7 V=320: per the D6 SHATTER pattern, the bio_brain_regions dedicated-pool architecture's grounded-symbol geometry is substantially cleaner than uniform-random FHRR algebra assumes. The actual capacity envelope appears to have more headroom than algebra predicts. D7 V=320 should plausibly PASS at L=2 and L=3 (slight degradation possible at L=5 but no two-rung drop).

## Scrutiny items (9 items; ALL must PASS for CLEAR)

1. **Bug fix correctness**: 5 distinct seed offsets at 100k apart via `_DIRECTION_7_BRIDGE_LABEL_SEED_OFFSETS`; activity vectors DISTINCT across bridges (cos < 0.99 across the 5 V=64 bridge slot pairs). Confirm via diagnostic probe analog of D4 distinctness probe / D6 distinctness probe. BLOCK if cross-bridge cos > 0.99.

2. **Multi-seed reproducibility at production scale**: 15 cells trained (5 bridges x 3 seeds); 3 seeds in probe; loads {2, 3, 5} with n_trials=200. Per-seed and aggregate OI values logged. BLOCK if missing.

3. **Smell-test recomputation**: independently recompute multi-seed OB+OI from per-seed JSON values; match aggregate within 0.001. BLOCK on > 0.001 mismatch.

4. **OB characterisation**: expect OB perfect (1.000) at L=2 and L=3; possibly PARTIAL at L=5 given doubled vocab vs D6. BLOCK only if OB at L=2 falls below 0.80 (would contradict the D6 trajectory).

5. **OI characterisation (the key scaling test)**: expect OI PASS at L=2, probably L=3; L=5 is the open question. PASS at L=5 would extend pillar n=109's SHATTER pattern; BOUNDARY at L=5 would establish a clean envelope at V=320. BLOCK only if OI at L=2 falls below 0.80.

6. **Comparison to pillar n=109 D6 (V=160)**: D7 (V=320) doubles vocab vs D6. Per the SHATTER pattern, D6's OI L=5 = 0.987 essentially preserved at V=160 (no boundary). D7 V=320 should show whether the dedicated-pool grounded-symbol geometry remains near-orthogonal at V=320 OR finally hits a real envelope. Either outcome is informative; honest characterization required. If D7 actually BEATS D6 at L=5, that's strong evidence the per-bridge mean-centring sharpens further at V=64 per bridge (more concepts to average against the shared baseline -> cleaner concept-specific residual). If D7 hits BOUNDARY at L=5, that's the long-predicted FHRR capacity envelope.

7. **Comparison to G.20 sparse 320-tier**: G.20 sparse 5-bridge × V=64 per bridge (320 unique) achieved 98.4% per-bridge PASS rate (pillar n=? from 2026-05-16 SHIPPED doc). D7 uses BYTE-IDENTICAL vocab. Compare D7's V_total=320 cross-bridge mode-unification OI/OB to G.20's per-bridge readout. Different metrics (cross-bridge composition vs per-bridge classification) but same vocab tier — relative performance establishes whether biology-faithful architecture matches the user-facing G.20 chat capability deliverable.

8. **Anti-cheat**: parallel-matching primitive (cross_bridge_mode_unification_probe.py from pillar n=95) byte-unchanged via `git log -p research/findings/raw/cross_bridge_mode_unification_probe.py | head -20`. Protected set byte-empty diff: `git diff e739543..HEAD -- research/runners/abstention_gate.py tests/test_abstention_gate.py sim/bridge.py sim/kernels.py sim/neuromodulators.py sim/backend.py research/runners/text_minimal_isolation.py`. BLOCK if non-empty diff on primitive or protected set.

9. **Score-tuning / threshold-tampering check**: bar 0.80 frozen; seeds [42, 43, 44] frozen; no post-hoc adjustment. Verify `_DIRECTION_7_OB_MIN`, `_DIRECTION_7_OI_MIN`, `_DIRECTION_7_LOADS`, `_DIRECTION_7_MIN_SEEDS` byte-identical to commit 72e8964. BLOCK if any tampering found.

## Reviewer verdict

End with EXACTLY ONE of:
- **CLEAR**: all 9 items PASS. Pillar n=110 candidate APPROVED. Specify whether PASS or BOUNDARY based on OI L=5.
- **BLOCK**: specify failed items + strengthening fix needed.

## Pillar n=110 framing (if CLEAR)

Pillar n=110 [PASS or BOUNDARY]: Direction 7 D4-architecture extended to V=320 cross-bridge (5 bridges x V=64 = 320 unique concepts on dedicated-pool bio_brain_regions; vocab byte-identical to Direction M G.20 sparse production deliverable). [OB/OI characterisation]. [PASS framing: BEATS/MATCHES/EXTENDS pillar n=109 D6 V=160 trajectory; the FHRR algebra capacity-ratio prediction (boundary should drop ~2 rungs at doubled V) again SHATTERED at production scale; dedicated-pool grounded-symbol geometry confirmed substantially cleaner than uniform-random across vocab scales V=80 -> V=160 -> V=320. Unifies the user-facing chat capability tier with biology-faithful architecture (pillars n=98/n=105/n=108/n=109/n=110).] [BOUNDARY framing: D6 V=160 (0.987) was at-ceiling; D7 V=320 establishes the genuine FHRR capacity envelope at L=5 OI. Honest characterization of architectural ceiling; substrate scales to V=160 cleanly but V=320 hits the predicted envelope.]
