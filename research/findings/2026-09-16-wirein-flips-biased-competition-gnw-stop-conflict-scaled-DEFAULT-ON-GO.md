---
type: finding
status: verified
date: 2026-09-16
mechanism: two scaffold-retirement wire-in flips — learned spiking referent-bias (biased-competition) + spiking-conflict-scaled GNW stop-boost (gnw-global-stop)
integration_faculty: selective-attention-biased-competition
lane: language + focus (scaffold-retirement, owner's #1 metric)
seeds: [42, 43, 44, 100, 101, 102]
verdict: Two host-cognition residuals identified by the 2026-09-16 retirement-readiness audit are replaced by
  already-validated spiking reads and flipped DEFAULT-ON, each verified by BOTH a 6-seed runner soak AND the
  integrated /api/brain-chat no-regression battery (all_pass, 0/38, AWS r7i). (1) selective-attention-biased-
  competition: the host animacy/verb LEXICON that scored the WTA referent-bias is demoted; the learned spiking
  feature-compatibility map (SpikingFeatureCompat, gap#3 A1) is now the production default -> the row moves to
  RETIRABLE_NOW (host lexicon reached only in the <40-fact fallback, never in the production KB). (2)
  gnw-global-stop: the fixed-scalar conflict-boost is replaced by the brain's OWN upstream spiking conflict
  magnitude -> that residual is RETIRED, but the row STAYS BLOCKED:neural-render because its verdict->clearing-
  STRING template still needs the own-voice mouth (a concrete instance of the mouth being the retirement keystone).
  HONEST: scaffold_retired stays 4 this round — neither host SYMBOL is deleted yet (the deletes are careful
  follow-ons); this lands the spiking paths as the production defaults, not full RETIRED.
runner: research/runners/_learned_referent_bias_flip_soak.py + research/runners/_gnw_stop_conflict_scaled_boost_derisk.py + research/runners/onebrain_regression_battery.py
artifacts:
  - research/findings/raw/_learned_referent_bias_flip_soak/soak_summary_6seed.json
  - research/findings/raw/_gnw_stop_conflict_scaled_boost_6seed.json
  - research/findings/raw/_wirein_flips_20260916/battery_BRAIN_BIASED_COMPETITION_LEARNED_BIAS.json
  - research/findings/raw/_wirein_flips_20260916/battery_BRAIN_GNW_STOP_CONFLICT_SCALED.json
external: NO-EXTERNAL-NEEDED -- both wire in EXISTING already-validated spiking mechanisms (SpikingFeatureCompat 6/6
  GO 2026-07-18; the gnw-deliberation conflict magnitude already computed); no new mechanism.
builds_on:
  - research/findings/raw/_retirement_readiness_audit_20260916/result.json
  - research/findings/2026-07-18-gap3-A1-learned-feature-compatibility-cheap-first-GO.md
---

# Two scaffold-retirement wire-in flips default-ON — biased-competition (RETIRABLE_NOW) + gnw conflict-boost (retired, row still mouth-blocked)

The 2026-09-16 retirement-readiness audit (12-agent source-verified fan-out) found these two rows carry a host-
cognition residual that is NOT the reply-text mouth and is retirable by wiring an already-validated spiking read.
Build agents wired each (additive, default-off), 6-seed pool soaks passed, and one batched AWS r7i integrated verify
cleared both. Flipped default-ON.

## The two flips (each: 6-seed soak GO AND integrated no-regression all_pass)

<!--derived-->
(verdicts read from `research/findings/raw/_wirein_flips_20260916/battery_BRAIN_BIASED_COMPETITION_LEARNED_BIAS.json`,
its `battery_BRAIN_GNW_STOP_CONFLICT_SCALED.json` sibling, and the two soak artifacts
`research/findings/raw/_learned_referent_bias_flip_soak/soak_summary_6seed.json` +
`research/findings/raw/_gnw_stop_conflict_scaled_boost_6seed.json`.)

- **selective-attention-biased-competition** — `BRAIN_BIASED_COMPETITION_LEARNED_BIAS` default-ON. The learned
  spiking feature-compatibility map replaces the host `content_bias_target` animacy/verb lexicon as the per-referent
  WTA-bias source. Soak: 6/6 seeds, each `off_byte_identical` + `on_matches_host` + `lesion_diverges` (abstains under
  lesion = load-bearing). Integrated battery `battery_BRAIN_BIASED_COMPETITION_LEARNED_BIAS.json`: all_pass, 0/38.
  -> **RETIRABLE_NOW**: the host lexicon is now reached only in the <40-heard-fact fallback (never in the ≥40-fact
  production KB); its sole host-cognition residual is demoted.
- **gnw-global-stop** — `BRAIN_GNW_STOP_CONFLICT_SCALED` default-ON. The conflict-stop boost is now scaled by the
  brain's own upstream spiking conflict magnitude (gnw-deliberation's `n_ignited`/`conf`) instead of a fixed host
  scalar. Soak: verdict GO 6-seed (byte-identical-off, lesion holds, swap-fallback unchanged; the moderate-conflict
  conf~0.5 case is characterized-not-cleared, a named residual). Integrated battery
  `battery_BRAIN_GNW_STOP_CONFLICT_SCALED.json`: all_pass, 0/38.

## Honest scope — why scaffold_retired stays 4 (not 6)

Neither is a clean RETIRED this round, and I will not inflate the metric:
- biased-competition keeps `content_bias_target` as a genuine (if production-unreachable) <40-fact fallback; deleting
  it + handling that degrade is a reviewed follow-on -> RETIRED. It is RETIRABLE_NOW.
- gnw-global-stop retired ONE of its three residuals (the fixed-scalar conflict boost). It STAYS BLOCKED:neural-render
  because its verdict->clearing-STRING template still needs the own-voice mouth — a concrete confirmation of the
  audit's keystone finding (the mouth blocks the majority of remaining retirements). `BOOST_GAIN` remains only as the
  base magnitude the spiking scale multiplies + the no-upstream-conflict fallback (an architectural constant).

Real progress: two host-cognition shortcuts (a feature lexicon; a fixed-scalar boost) are replaced by spiking reads
as the production defaults, integrated-verified answer-preserving. The full deletes (→ scaffold_retired++) are the
documented next follow-ons; the biggest lever remains finishing the own-voice mouth (neural-render).
