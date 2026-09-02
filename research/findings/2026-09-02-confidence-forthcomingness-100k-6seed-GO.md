---
type: finding
status: live
date: 2026-09-02
mechanism: confidence-forthcomingness-reverify-at-wikidata-100k
board: 108 (cluster R3) / 94
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_confidence_100k_recalib/verify_100k_6seed_cupy.json
---

# Confidence→forthcomingness (#94) re-verified GO at the full 100k knowledge scale — 6/6, load-bearing

**2026-09-02, cupy, 6 seeds (42/43/44/100/101/102), elapsed ~3871s.** The confidence→forthcomingness coupling
(the brain elaborates more when its recall is confident and hedges/caps when it is not) was re-verified against
the **wikidata_100k** LTM bundle (~79k facts <!--derived-->) rather than the small demo store, as the second of the two
blockers on flipping the #108 large-knowledge default. This closes the scale question the earlier
`margin_norm` recalibration (2026-09-01) opened: the peak-normalized `mean_role_conf` reads correctly at 100k.

Artifact: `research/findings/raw/_confidence_100k_recalib/verify_100k_6seed_cupy.json`.
Flags: `BRAIN_ELABORATE_FROM_LTM_SHARD=1 + BRAIN_CONFIDENCE_FORTHCOMING=1 + BRAIN_LTM_SHIP_DEFAULT=1 +
BRAIN_KB_RELATION_QUESTIONS default-ON`, true floor (moat escape residual 1, unrelated).

## Verdict: GO

- **`measurement_all_GO: True`** — all 6 seeds: `confident=True`, `mean_role_conf=0.4769`, `reason=high_confidence`
  on the clean (confident) recall. The `mean_role_conf` sits squarely inside the existing metacog band
  (0.30/0.50 <!--derived-->) — the recalibration's whole point (comparing like-with-like across codebook sizes,
  no per-scale threshold re-tuning).
- **`vary_lesion_all_GO: True`** — the coupling is **load-bearing, not observational**: in the lesion arm the
  same recall reads `confident=False`, `reason=low_confidence_capped`, `elaborations_dropped=1`
  (forthcomingness collapses when the confidence link is cut). This is the anti-hollow bar (the effect must
  VANISH when the coupling is lesioned) met per-seed.

## What this unblocks

One of the two #108 (large-knowledge default) blockers is now cleared. The remaining blocker is **latency**:
the 100k recall median sits ~1.3s on cupy (a backend-specific per-query cost, diagnosed 2026-09-02 as NOT the
decode-escalation trigger margin — see `2026-09-02-escalation-gating-tighten-latency-correctness-safe-not-the-lever.md`).
The latency lever in progress is vectorizing the per-query winner-code gather in `_escalate_role_match`; the
owner-set fallback is to accept ~1.3s (inside the stated 1.1–1.3s tolerance <!--derived-->) if that fix does not clear <1s.
