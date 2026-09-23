---
type: finding
status: no-go
claim_check: synthesis
date: 2026-09-23
mechanism: ONE declared spiking CrossEdge (fixed weight 4.0) from a NEW margin-comparator circuit -- built on
  metacog's production-unused meta_schema region, hand-set weights calibrated on seed 42, not Hebbian-grown --
  into curiosity's ASK pool. No curiosity neuromodulator, no host novelty signal, no STDP/homeostasis/OU noise
  on the pool (fixed-weight by design). Runs on a 2-organ REGISTRY-descriptor pool
  (metacog-with-production-heterogeneity + curiosity), not the production 11-organ wave3 pool. Harvest of
  research/curiosity-lane-next (HEAD 4d8168c97310ff02ca4ac4e2b8e7b67845371f23, NOT merged here) -- no new
  mechanism, runner, or sim/ edit in this finding, only a read of the 6-seed artifact that landed after the
  fix round returned.
lane: curiosity (metacog low-confidence recall -> ASK coupling)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO -- 4/6 seeds pass all evidence gates (n_go=4), 3/5 held-out seeds pass (seed 42 is the
  calibration seed). This holds under BOTH the runner's own all-8-checks scoring and the re-review's corrected
  evidence-only scoring (G1, G4, G7, G8; see below) -- the reclassification does not change the pass/fail
  pattern in this data.
artifacts:
  - research/findings/raw/_curiosity_metacog_conflict_xedge_6seed.json (landed on pool42/pool41 after the fix
    round returned; pulled by `tools/pool_sync.sh` this session)
  - research/findings/raw/_curiosity_metacog_conflict_xedge_6seed.json.prov.json
external: NO-EXTERNAL-NEEDED -- reads an already-run, already-committed pre-registered gate; no new claim.
builds_on:
  - docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md (research/curiosity-lane-next branch,
    NOT merged)
  - research/runners/_curiosity_metacog_conflict_xedge_derisk.py (same branch)
  - re-review of that branch (verdict: fix-required; prior_issues_resolved: true for the ORIGINAL fix-round
    issues, but 5 NEW issues raised on the new mechanism -- honored below)
---

# Curiosity CPU-lane harvest: the 6-seed conflict-crossedge gate, corrected

NO-EXTERNAL-NEEDED: this is a harvest read of an already-run, already-committed pre-registered gate against
thresholds fixed before any seed ran; no new mechanism or capability claim is made here that would require
external literature.

**Artifact read:** `research/findings/raw/_curiosity_metacog_conflict_xedge_6seed.json`
(+ `.prov.json` sidecar).

## What this is

`research/curiosity-lane-next` (HEAD `4d8168c97`) built a new spiking mechanism (a margin comparator on
metacog's `meta_schema` region driving curiosity's ASK pool through one declared CrossEdge) and staged a
6-seed pre-registered GO gate on the pool, then returned without waiting. Its re-review passed the fix
round's original 7 issues but raised 5 new ones on the new mechanism (journal line 9). This harvest pulls the
now-landed 6-seed artifact (it was still queued at re-review time) and scores it, **applying the re-review's
corrections to how the result is characterized**, not re-deriving new thresholds.

## The corrected reading: three re-review constraints applied

**1. G3, G3b and G5 are integrity smokes, not evidence, and reclassifying them does not change this
verdict.** The re-review's core finding: ASK has no drive except the one declared edge (no noise, no
neuromodulator, no curiosity-side input to metacog), so under an edge lesion 100% attribution is guaranteed by
construction (G3 cannot fail), and because the edge is feed-forward with no return path into metacog, `G3b`
and `G5` (metacog reads unchanged under lesion / vs. no-edge / vs. bare pool) are also guaranteed. **Evidence
gates are G1, G4, G7, G8** (G4 itself only weakly independent of G1 — see below). Reading the landed data
against evidence-only gates gives the identical per-seed pass/fail pattern as the runner's own all-8-check
scoring, because G3/G3b/G5 are in fact `true` on every seed in this data too — the reclassification is a
correction to what "8/8" is allowed to mean, not a change to the numeric result.

**2. The mechanism is a NEW circuit, not production metacog's own confidence read.** Production's confidence
decision is the `nmda_norm_margin` read; `meta_schema` is documented as "left present but unused" in
production (`_second_order_metacog_monitor_derisk.py:714`). The comparator gating ASK here is a *separate*,
hand-tuned circuit sitting on that unused region. The claim tested is "a parallel margin comparator built on
metacog's own substrate correlates with recall confidence," not "production's own confidence signal drives
curiosity" — no gate here checks separation against production's actual `confident` flag, and at the
calibration seed the confident/uncertain ASK ranges are adjacent at the boundary (0.14–0.37 Hz vs. 0.49–3.11
Hz).

**3. S1 (secondary, non-gating) does not cross the production curiosity threshold on any of the 6 seeds** —
confirmed here directly from the per-seed `secondary.production_curiosity_calib` block:

| seed | threshold (Hz) | max uncertain-level ASK (Hz) | crosses? |
|---|---|---|---|
| 42  | 19.10 | 3.11 | no |
| 43  | 21.88 | 5.75 | no |
| 44  | 23.26 | 4.06 | no |
| 100 | 20.14 | 6.53 | no |
| 101 | 23.96 | 1.68 | no |
| 102 | 21.88 | 5.69 | no |

The synaptic route alone never reaches within a factor of ~3 of production's own curious threshold on any
seed. This mechanism, wired as-is, would not make production curiosity fire on a low-confidence recall; it
would need summation with the organ's existing novelty drive or a substantially stronger learned (not
hand-set) edge.

## Per-seed evidence gates (G1, G4, G7, G8)

| seed | G1 (ρ≤−0.8) | G4 (perm p≤0.01) | G7 (swap ρ≤−0.8) | G8 (relay-lesion ρ>−0.5 or UNDEF) | ρ | ρ_swap | ρ_relay-lesion | held-out? |
|---|---|---|---|---|---|---|---|---|
| 42  | PASS | PASS | PASS | PASS | −0.991 | −0.909 | +0.955 | calibration |
| 43  | PASS | PASS | PASS | PASS | −0.916 | −0.907 | +0.891 | yes |
| 44  | **FAIL** | PASS | PASS | PASS | **−0.691** | −1.000 | +0.709 | yes |
| 100 | PASS | PASS | **FAIL** | PASS | −0.834 | **−0.795** | −0.145 | yes |
| 101 | PASS | PASS | PASS | PASS | −0.954 | −0.909 | +0.518 | yes |
| 102 | PASS | PASS | PASS | PASS | −0.943 | −0.970 | +0.727 | yes |

**Raw: 4/6 GO. Held-out (excluding the seed-42 calibration seed): 3/5 GO.** Seed 44 fails G1 (monotonicity
weakens to ρ=−0.69, short of the −0.8 bar) despite a clean class-swap. Seed 100 fails G7 (class-swap ρ=−0.795,
just short of −0.8) and is also the one seed where the relay-lesion coefficient (`ρ_relay-lesion = −0.145`)
sits far from every other seed's +0.5..+0.96 range — its coupling under a relay lesion doesn't just weaken,
it goes slightly negative, suggesting seed 100's coupling may be routed differently through the comparator
than the other 5 seeds. G8 still formally passes on seed 100 (−0.145 > −0.5), but this is the seed to
re-examine first if this mechanism is revisited.

**G4's pseudo-replication caveat (re-review issue 3), reported not re-derived:** the 10,000-permutation null
shuffles all 88 per-rep observations as exchangeable, but each level's 8 reps are consecutive reads on the
same pool, not independent draws, which makes the reported p-values (all ≈1e-4) anti-conservative. Since
G1 already requires the stronger level-mean-ρ≤−0.8 bar over 11 levels, G4 adds little independent evidence
here and should not be read as a second, separate confirmation.

**G8's UNDEFINED-as-pass bug did NOT trigger on this data.** The re-review flagged that the scorer treats
`rho_relay is None` as a pass (a relay lesion that flattens ASK entirely would pass by default). Every one of
the 6 landed seeds has a defined, non-null `rho_relay_lesion` value (see table above) — the bug is real and
unfixed on the lane branch's scorer, but it did not affect this verdict.

## Bottom line

**NO-GO**, under both the runner's own scoring and the re-review's corrected evidence-only scoring. The
mechanism is a real, wired, spiking cross-edge with no by-construction confound on its non-integrity gates,
but it fails held-out generalization on 2 of 5 truly held-out seeds (44 on monotonicity, 100 on the
class-swap direction check), and even where it passes, the synaptic drive alone (1.7–6.5 Hz peak ASK) sits
**3x to 14x below** production's own 19–24 Hz curious threshold (per-seed ratio range, from the S1 table
above: seed 101 is the widest gap at ~14x, seed 100 the narrowest at ~3x) — not uniformly "an order of
magnitude," as an earlier version of this finding said. Per `docs/TERMS.md`, this is reported as
NO-GO, not "partial" or "characterized limit" — the gate's own verdict is negative and that verdict is
reported as-is.
