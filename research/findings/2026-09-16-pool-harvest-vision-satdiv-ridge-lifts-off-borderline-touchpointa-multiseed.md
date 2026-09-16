---
type: finding
status: contributing
date: 2026-09-16
mechanism: vision-satdiv-divisive-norm-readout + touchpoint-a-multiseed
lane: perception + language
seeds: [42, 43, 44, 100, 101, 102]
verdict: D-vision satdiv finer grid ADVANCES off the BORDERLINE — ridge=1.0 at sigma8/scale760 crosses the runner's
  beat-NO-GO-floor(5/6)+load-bearing(6/6) GO bar (capability_go 0->2/6), the ridge axis being the lever; but full
  per-seed capability is still only 2/6, so it is a step off the wall, NOT a solved wall. Touchpoint-A rank-3 n=4
  structural battery holds across all 5 seeds (multi-seed robustness confirmed). 0-Claude-token pool compute.
runner: research/runners/_vision_lindiscrim_readout_derisk.py + research/runners/_touchpoint_a_fact_clause_derisk.py
artifacts:
  - research/findings/raw/_pool_harvest_20260916/vision_satdiv_refine_grid.json
  - research/findings/raw/_pool_harvest_20260916/touchpoint_a_multiseed.json
  - research/findings/raw/lanes/perception/satdiv_refine_sig8_sc760_r1p0_6seed.json
external: NO-EXTERNAL-NEEDED -- a finer grid of an existing readout lever (Reynolds-Heeger divisive normalization,
  already banked in the D-vision arc) + a multi-seed repeat of an existing battery; no new mechanism or claim.
builds_on:
  - research/findings/2026-09-09-vision-configural-binding-competitive-selection-NEXT-MECHANISM-PREREGISTERED.md
---

# Pool harvest: the vision satdiv ridge lever lifts off the BORDERLINE (still partial); Touchpoint-A multi-seed holds

Harvest of the 14-cell pool batch staged while the owner gamed (all remote mini-PC CPU, 0 Claude tokens).

## D-vision satdiv readout — the finer (sigma, scale, ridge) grid, the walls-ledger's named next lever

The 2026-09-03 walls-ledger marked the satdiv divisive-normalization readout **BORDERLINE** at sigma=8/scale=760
(capability_go 0/6, first-ever single-seed capability True) and named a finer (sigma, scale, ridge) grid as next.
Ran it: sigma {6,7,9,10}@sc760, scale {700,740,780,820,850}@sig8, ridge {0.2,1.0}@sig8/sc760, all 6-seed.

<!--derived-->
(verdicts + counts from `research/findings/raw/_pool_harvest_20260916/vision_satdiv_refine_grid.json` and the cited
per-cell artifacts.)

- **ridge=1.0 at sigma8/scale760 is the lever: it crosses the runner's GO bar** (`overall_verdict:
  LINDISCRIM-READOUT-GO`, `task_go_5of6_beat_and_lb: True`). sigma {6,10} collapse to NO-GO; the sigma=8 scale row
  and ridge=0.2 stay strong PARTIAL (beat 2-4/6, load-bearing 6/6).
- **What "GO" means here (the honest read — this is a beats-floor GO, not a solved task).** That GO rests on
  `beats_config_c_nogo` 5/6 (clears the 0.34 config-C NO-GO floor) AND `learning_load_bearing` 6/6. The STRICTER
  per-seed `capability_go` is **2/6** (up from the frontier's 0/6), with held-out accuracy ~0.44-0.57 vs chance 0.25.
  So the finer grid genuinely moved the arc OFF the BORDERLINE (learning is now robustly load-bearing + beats the
  floor, capability_go 0->2/6), but it did NOT solve the binding task (full capability still 2/6).
- **Mechanistically:** higher ridge (more readout regularization) at the divisive-norm operating point is what tips
  it over the beat+lb bar — consistent with the Reynolds-Heeger framing (normalization + a stable readout). The
  named next lever is now pushing per-seed `capability_go` up (the ~0.5 held-out accuracy has headroom vs the
  capability bar), not more (sigma,scale) — those are mapped.

## Touchpoint-A (rank-3) — multi-seed structural robustness confirmed

The 2026-09-04 Touchpoint-A finding ran the n=4 fact-clause battery at seed 42 (structural) + 43/100, and named a
multi-seed/probe-set repeat as strengthening a promotion decision. Ran seeds 44/101/102.

<!--derived-->
(from `research/findings/raw/_pool_harvest_20260916/touchpoint_a_multiseed.json`.)

All 5 available seeds (43/44/100/101/102) pass the full battery's structural gate:
`scope_untouched` + `content_preserved` + `flag_off_inert` + `structural_checks_passed` all True on every seed. So
the wire-in's byte-identical-off + structural soundness is seed-robust — the promotion case is strengthened (a
production-flip would still want the integrated no-regression soak, as with the other wire-ins).

## Honest scope

Both are 0-token pool results. The vision GO is a beats-floor+load-bearing GO with capability_go 2/6 — an advance
off the BORDERLINE, explicitly NOT "vision binding solved". Functional read-outs only.
