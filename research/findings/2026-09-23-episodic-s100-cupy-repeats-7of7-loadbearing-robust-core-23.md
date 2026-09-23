---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
verdict: GO
---

# Episodic-memory is load-bearing 6/6 → ROBUST CORE 23: s100 resolved by a 7×-repeat cupy majority-vote (7/7 clean-deterministic LB) (2026-09-23)

Resolves the last-open borderline from the episodic in-situ verification
(`2026-09-22`, board CURRENT STATE): episodic read clean load-bearing on s42/s43/s44/s101/s102 but s100's single
cupy in-situ run flagged `UNRELIABLE` (a non-deterministic intact/null-control arm — GPU-reduction noise at a
near-threshold `episodic.in_memory` read), leaving episodic at 5/6-measurable + s100-unresolved (robust core held at
22). The numpy deterministic re-measure proved impractical (~48 rebuilds of the 3.65M-synapse brain). So s100 was
re-measured on cupy with a **7×-repeat majority vote** (serial, one GPU brain at a time, memcap 12).

## Result — s100 is load-bearing, 7/7, all clean-deterministic
<!--derived from research/findings/raw/_lbf_fix_episodic_store/_insitu5_cupyrepeats/epi_s100_run1.json -->
All 7 independent fresh-subprocess cupy runs of `load_bearing_fraction --only episodic-memory --seed 100 --repeats 1`
(BRAIN_EPISODIC_STORE_VERIFY=1, LB_EPISODIC_DRIVE_PROBE=1) read **episodic-memory LOAD-BEARING**:
| run | load_bearing | verdict | treatment_diffs | control_diffs (null) | deterministic | UNRELIABLE |
|---|---|---|---|---|---|---|
| 1-7 (all) | True | regressed | 1 | 0 | True | False |

**7/7 load-bearing AND 7/7 clean-deterministic** (null control `control_diffs=0` every run; the lesion flips
`episodic.in_memory` True→False, `treatment_diffs=1`). The earlier single in-situ `UNRELIABLE` (one non-deterministic
null) did NOT recur in any of 7 fresh repeats → it was a one-off GPU-reduction fluctuation, not a fragile faculty.
Artifacts: `research/findings/raw/_lbf_fix_episodic_store/_insitu5_cupyrepeats/epi_s100_run1.json` (representative; run1-run7 all identical) + `research/findings/raw/_lbf_fix_episodic_store/_insitu5_cupyrepeats/epi_s100_run7.json`.

## Verdict — episodic 6/6 → ROBUST CORE 23 (Option-C reported)
Episodic-memory is now lesion-verified load-bearing on all 6 seeds (42/43/44/100/101/102): s42/s43/s44/s101/s102
clean on their single in-situ runs (each `deterministic=True`), s100 clean 7/7 on repeat. So episodic joins the robust
core: **22 → 23** (the count of faculties load-bearing in ALL 6 seeds under adequate probes). Per the owner's OPTION-C
standard the #1 metric is reported as a PAIR: **robust core 23/26 adequate-probe** — always with the context that the
shipped thin-probe default remains ~0.59 (probe coverage, not a brain limit) and that **s100 is the closest-to-
threshold member** (1 prior non-deterministic read; 7/7 clean on repeat — load-bearing but nearest the determinism
band, so a future operating-point stabilizer for episodic store-reliability remains the honest hardening rung).

## Verification / anti-cheat (the majority-vote IS the adversarial check)
The 7×-repeat design directly attacks the near-threshold refutation: if s100 were fragile, the repeats would split;
they were 7/7 clean-deterministic instead. Each run is an INDEPENDENT fresh subprocess (own brain build), the null
control is clean in every run (`control_diffs=0` → the `treatment_diffs=1` is caused by the lesion, not run-to-run
noise), and BRAIN_EPISODIC_STORE_VERIFY is default-off / additive (no sim/ edit; the store-reliability substrate is
the already-committed `research/lbf-fix-episodic-store` mechanism). Honest residual: s100 sits nearest the
determinism band (the one prior flicker) — reported, not hidden.

## Honesty
Functional read-out only (the brain's episodic store→recall provably drives the reply; lesioning it changes the
answer) — no phenomenal claim. This advances charter dimension D1 (express/load-bearing) by one faculty to robust
core 23.
