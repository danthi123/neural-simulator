---
type: finding
status: interim
claim_check: measured
date: 2026-09-07
mechanism: AGI-fork — a systematic sweep of mechanisms for making an emergent PLACE code persist + be load-bearing
  in the single rate-recurrent predictive substrate, plus TWO instrument findings showing the emergence METRIC
  itself was masking the result.
lane: agi-fork (emergence / continual-learning substrate)
branch: agi-fork
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_fork_pcs_emergence_derisk.py
artifacts:
  - research/findings/raw/_fork_pcs_sr10_6seed.json
  - research/findings/raw/_fork_pcs_sr20_6seed.json
  - research/findings/raw/_fork_pcs_sr10_nav_shap20_6seed.json
  - research/findings/raw/_fork_pcs_shaping_ns10_6seed.json
  - research/findings/raw/_fork_pcs_shaping_ns20_6seed.json
  - research/findings/raw/_fork_pcs_floor_scaling.json
  - research/findings/raw/_fork_pcs_auxloc10_6seed.json
builds_on:
  - research/findings/2026-09-06-agi-fork-why-place-fades-successor-representation-fix.md
  - research/findings/2026-09-06-agi-fork-firstmove-emergence-transient-objective-does-not-retain.md
verdict: >
  TWO DURABLE FINDINGS + one promising lead. (1) MECHANISM SWEEP — NEGATIVE for every OBJECTIVE lever tried: a
  longer prediction horizon, a nav-to-remembered-goal task, potential-based reward shaping (two magnitudes), and a
  SUCCESSOR-REPRESENTATION target (two magnitudes + composed) all leave the linear-decode place metric at or below
  the untrained-reservoir floor (EMERGENCE_GO false in every case). SR is worse than doing nothing — a COMPETING
  objective that SUPPRESSES place (progressively worse with SR strength, worst when composed) and degrades
  navigation; SR is retired as a method. (2) THE INSTRUMENT WAS (PARTLY) THE WALL — two measured results converge:
  the untrained-reservoir linear-decode FLOOR rises monotonically with n_hidden (a larger random reservoir decodes
  position better with ZERO learning, near-perfect rank correlation), masking a genuine emergent code; AND a
  floor-independent Skaggs SPATIAL-INFORMATION place-cell metric SEPARATES a trained aux-localization core from the
  untrained reservoir (on the SI real/shuffle ratio, the significant-place-cell fraction, and rate-map stability)
  exactly where the linear decode does NOT, and the SI metric stays FLAT across n_hidden while the decode floor
  inflates substantially — capacity-invariant where the decode is not. So the fork's "place fades below floor"
  negatives were, at least in part, a capacity-artifact of the decode-vs-reservoir metric, not an absence of place
  tuning. (3) THE LEAD — an AUXILIARY SELF-LOCALIZATION LOSS (Cueva-Wei/Banino) is the best mechanism: it does NOT
  suppress place (place margin ≈ base, vs SR's suppression), has the highest decode load-bearing of any mechanism,
  and its trained core carries a genuine (SI-significant) place code. HONEST SCOPE: the SI validation is a SHORT
  single-seed run; the decisive floor-independent 6-seed 200k characterization of aux-loc across n_hidden (esp. the
  low-floor n_hidden=128 regime) is QUEUED, not yet run. Fork branch only; nothing wired. Functional read-outs only.
---

# AGI-fork: a place-mechanism sweep, and the finding that the INSTRUMENT was (partly) the wall

## What ran
The single rate-recurrent predictive substrate (`sim/pcs_substrate.py`) lived the grounded egocentric-crop grid
world for 200k online steps, 6 seeds, and the emergence battery read a PLACE faculty (+ object/permanence/value)
post-hoc. Across the arc we swept every OBJECTIVE lever we could build, each `--lesion-mode both` (decoding- AND
behavioral-importance lesions). All numbers below are read from the cited committed artifacts.

## 1. Mechanism sweep — every objective lever is NEGATIVE on the decode metric
<!--derived-->
| mechanism | place margin vs floor | place decode-LB | note |
|---|---|---|---|
| base (plain next-latent) | −0.04 | 2/6 | place emerges early, fades to floor by 200k |
| shaping ns=1.0 / ns=2.0 | ~floor | 2/6 | homing up, place not load-bearing — 2nd-order lever |
| SR sr=1.0 (alone) | −0.11 | 0/6 | SR SUPPRESSES place (competing objective) |
| SR sr=2.0 (alone) | −0.19 | 0/6 | stronger SR = worse |
| SR sr=1.0 + nav + shaping | −0.14 | 0/6 | worst; navigation also collapsed (approach_off below chance) |
| aux-loc w=1.0 (alone) | −0.037 | 3/6 | does NOT suppress; best decode-LB of any mechanism |

(Place margin = mean over 6 seeds of `place r2 − floor_untrained`, from `research/findings/raw/_fork_pcs_sr10_6seed.json`,
`research/findings/raw/_fork_pcs_sr20_6seed.json`, `research/findings/raw/_fork_pcs_sr10_nav_shap20_6seed.json`,
`research/findings/raw/_fork_pcs_shaping_ns10_6seed.json`, `research/findings/raw/_fork_pcs_shaping_ns20_6seed.json`,
and `research/findings/raw/_fork_pcs_auxloc10_6seed.json`.)
**SR is retired as a method** (banked negative): a successor-feature target, under this world's exploration
policy, reallocates capacity AWAY from a clean allocentric code and degrades both place and navigation. Mechanism:
`2026-09-06-agi-fork-why-place-fades-successor-representation-fix.md`.

## 2. The instrument was (partly) the wall — two measured findings
**(a) The linear-decode floor is a capacity artifact.**
<!--derived-->
`research/findings/raw/_fork_pcs_floor_scaling.json`: the untrained-reservoir place-decode R² RISES monotonically with n_hidden —
0.49@128, 0.63@512, 0.72@2048 (Δ+0.23, Spearman ρ=0.986) — a larger random reservoir decodes position better from
richer dynamics, with zero learning; `attributable_to` shows only ~1-5% of a trained core's decode is attributable
to training vs the size-matched reservoir. So "beat the floor by 0.05" is a stringent, size-dependent bar that
MASKS a genuine emergent code at n_hidden=512.

**(b) A floor-independent metric separates trained place tuning from the reservoir.** A Skaggs SPATIAL-INFORMATION
place-cell metric (SI = Σ_i p_i (λ_i/λ) log₂(λ_i/λ) per unit, + rate-map split-half stability, + significance vs a
100× position-shuffle null) was added (additive; GO gate unchanged). Validation (seed42, 25k):
<!--derived-->
| metric | trained (aux-loc) | untrained reservoir | separates? |
|---|---|---|---|
| linear-decode place R² | 0.650 | 0.631 | NO (floor masks, +0.02) |
| SI real/shuffle ratio | 3.06 | 1.71 | YES |
| frac significant place-cells | 0.928 | 0.516 | YES |
| mean rate-map stability | 0.672 | 0.394 | YES |

<!--derived-->
Across n_hidden 128→1024 the SI metrics stay FLAT while the decode floor inflates ~+0.20 — the SI metric is
capacity-invariant where the linear decode is not. (Honest: raw mean-SI is skewed and does NOT discriminate; the
normalized ratio / frac-place-cells / stability are the discriminators. The untrained reservoir is not near-zero
on SI — the world has genuine position-correlated wall cues — but it is capacity-invariant and the trained core
rises clearly above it.)

## 3. Reading it (no-defer)
The fork spent five mechanism levers hitting a "place doesn't persist" wall, and the deepest cause was the
**instrument**, exactly the CLAUDE.md lesson ("the instrument is part of the emulation; a mechanism you cannot
measure correctly you will tune in the wrong direction for weeks"). The linear-decode-vs-reservoir-floor metric
is capacity-artifact-flawed; a genuine place code (SI-significant, stable, above the reservoir) was sitting under
the inflated floor the whole time, and the aux-localization loss makes it stronger, not weaker.

## Next (queued, floor-independent)
The SI metric now rides EVERY run. The QUEUED aux-loc sweep — n_hidden {128, 256, 512, 1024} × magnitudes {0.5,
1.0, 2.0}, + retention (400k) + the behavioral combos — will give the decisive 6-seed floor-independent read:
does aux-loc produce a persistent, SI-significant, behaviorally-load-bearing place code, and does the low-floor
regime (n_hidden=128) let it clear even the linear-decode bar? GPU-gated (owner gaming); resumes on
`gpu_queue.sh resume`. The remaining gap is BEHAVIORAL load-bearing (aux-loc's place is decodable/SI-significant
but the base world does not require position); the composed aux-loc+nav+shaping run tests it.

## Honest scope
6-seed on the mechanism sweep + floor-scaling; the SI validation is single-seed/25k (the 6-seed 200k SI is
pending). A NEGATIVE-heavy but PRODUCTIVE arc: two durable instrument findings + a promising lead. Fork branch
only; nothing wired to production. Every read-out is a functional instrument reading; nothing asserts felt
experience.
