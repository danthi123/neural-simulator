---
type: finding
status: no-go
claim_check: measured
date: 2026-09-23
lane: B — curiosity (novelty / epistemic-gap crave) x E1 metacognition
mechanism: curiosity-metacog-lowconfidence-coupling (host linear map of metacog's INPUT evidence scalar onto the
  curiosity organ's `judge(novelty=)` argument; research/runners/_curiosity_metacog_lowconfidence_coupling_derisk.py)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_curiosity_metacog_lowconfidence_coupling_derisk.py
artifact: research/findings/raw/_curiosity_metacog_lowconfidence_coupling_6seed.json
verdict: NOT-GO 4/6 as measured by the runner (G4 fails seeds 43 and 44); the artifact's retrofitted preconditions
  block marks it UNDEFINED as evidence about the mechanism (two unmet preconditions, §2). BANKED on 2026-09-23, 22 days after the run. The instrument was
  defective in three ways (G4 is a random draw, the tested "coupling" is not a coupling, G5 is not an exact compare),
  so this NOT-GO is a verdict on the METHOD, not on the capability. The real rung is built separately (see §4).
---

# Curiosity x metacog low-confidence coupling: 6-seed NOT-GO (4/6), banked late, with the instrument defects named

## 1. What happened

On 2026-09-01 the pool ran the pre-registered 6-seed gate of
`_curiosity_metacog_lowconfidence_coupling_derisk.py` (pool41, run_id `1788309083-427492`, provenance sidecar
`research/findings/raw/_curiosity_metacog_lowconfidence_coupling_6seed.json.prov.json`). The runner's own verdict
was **GO=False, n_go=4/6**. G4 (shuffle control) failed on seeds 43 and 44. G1, G2, G3 and G5 passed on all 6.

The artifact was never committed and no finding was written. On 2026-09-23 a build agent checked only `git log`,
concluded the runner had "never been run", and queued the same gate again as two pool jobs (`..._seedsA.json`,
`..._seedsB.json`). A `ls research/findings/raw | grep curiosity_metacog` would have found the banked result. The
re-run finished before it could be cancelled. It reproduced the same verdict: seeds 42/100/101/102 GO, seeds
43/44 fail G4 only. The want_hz values differ from 09-01 because the curiosity organ's operating point was
recalibrated on 2026-09-18 (the FAITHFUL/CALMER regime, want_novel about 42 Hz instead of about 129 Hz). The re-run
is redundant and is **not** used as evidence here. It is not committed.

(Table values are the artifact's per-seed `rho` / `rho_shuffle` rounded to 3 decimals.)

<!--derived-->
| seed | rho(evidence, want) | G4 shuffle rho | G4 | GO |
|---|---|---|---|---|
| 42  | -1.000 | +0.107 | pass | GO |
| 43  | -1.000 | +0.500 | **fail** | NOT-GO |
| 44  | -1.000 | +0.714 | **fail** | NOT-GO |
| 100 | -1.000 | +0.214 | pass | GO |
| 101 | -1.000 |  0.000 | pass | GO |
| 102 | -1.000 | +0.321 | pass | GO |

(All values are from `research/findings/raw/_curiosity_metacog_lowconfidence_coupling_6seed.json`.)

## 2. Why this NOT-GO says nothing about the mechanism: three instrument defects

**(a) G4 is a random draw, not a null control.** The gate draws ONE permutation per seed
(`np.random.default_rng(90000 + seed).permutation(7)`). want_hz is strictly monotone in novelty, so the shuffled
rho is fixed entirely by that one permutation. It does not depend on the mechanism at all. For n=7, a random
permutation gives |rho| >= 0.5 with probability 0.267 (exact enumeration of all 5040 permutations: 1344 reach it). <!--derived--> So about 27% of seeds fail G4 by chance, whatever the
organ does. Seeds 43 and 44 drew "bad" permutations. Seed 42 passed the smoke only because its permutation was
benign. The fix is a permutation null distribution: at least 1000 permutations, with the observed statistic
reported as a percentile.

**(b) The tested "coupling" is a host linear map of metacog's INPUT, not of anything metacog computed.**
`_novelty_of(ev)` maps the host evidence scalar, which is metacog's input, straight onto curiosity's `novelty`
argument. Curiosity never receives a spike, balance or margin that metacog produced. G1 therefore measures only
that the curiosity ASK pool's want_hz is monotone in its own novelty input. That is already banked: DR-1
corr(gap, spiking-want) = +0.996 (quoted from `research/findings/2026-07-23-DR1-curiosity-inversion-ONBRIDGE-spiking.md`) <!--derived-->, and the rank-10 graded-novelty GO of 2026-09-05. Under the BRAIN-BASED-ONLY
standard this is a host shortcut, not an inter-organ coupling. G3 (removing curiosity's own excitability drive)
is tautological for the same reason. It lesions curiosity's input pathway, not a metacog-to-curiosity edge.

**(c) G5 is not an EXACT compare.** The docstring says metacog reads are "byte-identical whether or not this runner
imports the curiosity organ". The code compares two `MetacogProductionOrgan(seed)` instances in one process, with
curiosity imported in both, using `np.allclose`. Under [`docs/TERMS.md`](../../docs/TERMS.md), *byte-identical*
requires an exact or hash compare. At most this was a within-process determinism check of metacog.

## 3. Verdict

**NOT-GO 4/6, banked as measured.** Because of (a), no G4 outcome from this runner can be read as evidence either
way. Because of (b), even a 6/6 GO would not have shown that metacognition drives curiosity. This runner is closed
as a METHOD. Do not re-run it. The CAPABILITY is not closed: "a low-confidence recall makes the brain curious" is
still the named next rung in `curiosity_production_organ.py`'s own docstring.

## 4. What replaces it

The real rung: curiosity's ASK pool is driven through actual synapses by metacognition's OWN spiking read-out, on
one shared merged pool. The gate is pre-registered with a permutation null distribution, a lesion of ONLY the
metacog-to-curiosity edge, an EXACT metacog-unchanged compare, and a repeat-run hash. See
`research/runners/_curiosity_metacog_conflict_xedge_derisk.py` and its pre-registration in
`docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md`.

## 5. Process lapse (logged)

`research/FAILURE_LOG.md` 2026-09-23 records the lapse: a gate used ONE seed-drawn permutation as a pass/fail null.
It also records that "never run" was checked with `git log` only, which misses untracked raw artifacts.

Functional read-outs only. No phenomenal claim.
