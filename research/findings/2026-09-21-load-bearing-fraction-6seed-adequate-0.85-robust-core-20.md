---
type: finding
status: live
lane: load-bearing
date: 2026-09-21
---

# Robust adequate-probe load-bearing fraction (6-seed) = 0.85 ± 0.03; robust core 20/26 — the 6-seed constraint trims the single-seed 0.88 (2026-09-21)

The 6-seed validation of the #1 metric (lesion-verified LOAD-BEARING FRACTION — % of production faculties where
lesioning the brain's contribution provably changes the reply), measured with the ADEQUATE probes (all 7 verified
drive-probe flags ON), on the integration branch. This is the constraint-mandated robust number that supersedes the
single-seed adequate reading (~0.88) from `2026-09-20-hollow-set-attack-6of9-probe-artifacts-verified.md`.

## Result (artifacts: research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s{42,43,44,100,101,102}.json)

<!--derived-->
**Robust adequate-probe fraction = 0.8526 mean, 0.0264 std (n_seeds=6).**
- Per-seed load_bearing_fraction: s42=0.8462, s43=0.8462, s44=0.8077, s100=0.8846, s101=0.8846, s102=0.8462.
  (Each value is the `load_bearing_fraction` field of the correspondingly-named committed artifact
  research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s42.json etc.)
- **Robust core = 20 faculties** load-bearing in ALL 6 seeds (the honest robust count).
- Union = 23 faculties load-bearing in AT LEAST 1 seed.
- **3 seed-dependent borderline** (load-bearing in some but not all seeds): `episodic-memory`,
  `affect-marker-spiking-wta`, `source-provenance-honesty`.
- Backend numpy CPU (AWS r7i.4xlarge), thread-capped (OPENBLAS_NUM_THREADS=2), per-seed prov sidecars stamp the device.

## The honest correction (why the 6-seed matters)

The single-seed adequate reading was **~0.88 = 23/26** (seed 42). The robust 6-seed is **0.85 mean with a robust core
of 20, not 23** — because 3 of those 23 faculties are **seed-dependent, not robustly load-bearing**. Most notably
**episodic-memory** drops to borderline: its load-bearing verdict depends on the forced BTSP store succeeding, which
is sensitive to the per-seed heterogeneity (the same class of near-threshold seed-dependence documented for
prospective-memory below). This is exactly what the 6-seed constraint exists to catch: a single seed over-counted the
adequate fraction by ~3 faculties. **Report 0.85 ± 0.03 / robust-core-20, not 0.88.**

## The shipped default is still 0.59 (flags OFF)

All 7 drive-probe flags are DEFAULT-OFF. The production battery with no flags reads the **0.59 ± 0.02** floor
(`2026-09-20-load-bearing-fraction-6seed-0.59-robust-core-14.md`). The 0.85 is "what the brain drives WHEN each
faculty is probed adequately" — the honest ceiling of the current substrate under adequate instrumentation — not the
shipped-default number. The gap between 0.59 and 0.85 is a PROBE-COVERAGE gap (the instrument, not the brain), now
quantified 6-seed on both ends.

## The three gap agents, folded in (all honest, no tuning — HARD RULE 2 held)

- **prospective-memory** (measured separately, `2026-09-21` pmem-v2 6-seed): flips load-bearing in **5/6 seeds**
  (seed 44 a null-clean near-FIRE_THR miss) — a seed-dependent-borderline flip, same tier as the 3 above. It is a
  27th faculty-turn (the intervening-turns protocol), not in the 26-faculty default roster, so it is reported
  alongside rather than inside the 0.85. Artifacts research/findings/raw/_load_bearing/_pmem_v2_6seed/.
- **open-ended-generation**: NOT load-bearing on the single-turn decision-diff (the wrong ruler for a faculty whose
  contribution is a DISTRIBUTION of guesses); distributionally it IS load-bearing (draw_many plausible-frac
  0.83->0.04 under lesion, already GO'd). Honest-negative on the single-turn metric + a genuine banked wiring fix
  (the neural draw-lesion was a silent no-op). Finding
  `2026-09-21-open-ended-generation-single-turn-not-load-bearing-spiking-plausibility-gate-masks-draw.md`.
- **da-gated-encoding**: honest-negative confirmed — load-bearing only under a tuned swept read-damage operating
  point, rejected as not a natural conversational drive.

## Integration + provenance

- All measured on the consolidation branch **research/lb-consolidation @ bacd4e3c** = lb-integration ∪ pmem-v2 ∪
  open-ended-v2 (all 9 drive-flags + seed-threading + the open-ended wiring fix; selftest PASS, byte-identical
  default, PROBE_TURNS=26, default-path no-regression verified). The AWS 6-seed ran the 7-flag adequate battery via
  `--seed` per the seed-threading feature.
- Aggregation: research/findings/raw/_load_bearing/_adequate6/ (6 per-seed JSONs + prov sidecars, backend-stamped).

## Next
1. Merge research/lb-consolidation to main (de-risked; all flags default-OFF, seed-threading byte-identical default).
2. Wire open-ended-generation's DISTRIBUTIONAL lesion metric as its load-bearing ruler (its own recommendation) —
   the no-defer step that would move it from single-turn-negative to a proper distributional load-bearing read.
3. The seed-dependent borderline set (episodic, affect-marker, source-provenance, prospective-memory) is the honest
   actionable remainder: each is near an operating-point threshold, not wired-hollow. Characterizing the threshold
   (what companion process would stabilize the flip across seeds) is the deeper faculty work.
