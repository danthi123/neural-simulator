---
type: finding
status: live
lane: load-bearing
date: 2026-09-22
---

# Borderline faculties: intact/lesion reads SEPARATE across 6 seeds → the operating-point stabilizer is BUILDABLE (2026-09-22)

Follow-on to the operating-point diagnosis (`2026-09-21-load-bearing-borderline-operating-point-diagnosis.md`). That
finding measured, per seed, each borderline faculty's continuous spiking read + its fixed host threshold. This is a
zero-new-compute analysis OVER those committed reads (`research/findings/raw/_lbf_borderline/op_s{42,43,44,100,101,102}.json`)
answering the go/no-go precondition for NEXT ACTION #5 (build the stabilizer): **does a FIXED operating point exist that
separates the intact read from the lesion read across ALL 6 seeds?** If yes, a principled homeostatic regulator that
targets that operating point can make the flip robust-6/6 (the fix is buildable). If the reads OVERLAP, no fixed
operating point works and the seed-dependence is fundamental (honest-negative).

## Result (derived from the committed op_s*.json reads; no new brain builds)
<!--derived-->
Separability = (min over seeds of the intact read) − (max over seeds of the lesion read); a POSITIVE gap means a fixed
operating point in the gap separates all six seeds.

<!--derived-->
| faculty | read | min-intact | max-lesion | gap | verdict |
|---|---|---|---|---|---|
| episodic-memory | apical_cue | 0.500 | 0.000 | +0.500 | SEPARABLE |
| source-provenance-honesty | d | 1.000 | 0.000 | +1.000 | SEPARABLE |
| prospective-memory | rel | 0.184 | 0.041 | +0.143 | SEPARABLE |
| affect-marker-spiking-wta | WTA | — | — | — | INDETERMINATE (read not scalar) |

## What this means for the #5 stabilizer build
- **episodic, source-provenance, prospective-memory: the stabilizer is BUILDABLE.** Their intact reads sit strictly
  above their lesion reads on every seed, so a principled operating point exists; a homeostatic regulator (e.g. a
  Turrigiano-style scaling that normalizes the read toward a target the way the substrate does elsewhere) that centers
  the operating point in the gap would make the flip robust-6/6. This is NOT tuning a constant to inflate a count — it
  is a biology-grounded regulator whose TARGET is derived from the substrate's own separable reads.
- **affect-marker-spiking-wta: fix its RNG-isolation bug FIRST.** Its read is not a scalar the intact/lesion arms share
  (the WTA OU-noise RNG is non-isolated across the read pair, per the diagnosis caveat), so separability cannot even be
  assessed until the reads are made comparable. That bug fix is the prerequisite; only then can separability be judged.

## Honest nuance (read-level vs integration-level seed-dependence)
episodic's diagnosis READ separates cleanly (its intact reads sit above the lesion reads and above threshold on every
seed — see the table), so by the read alone all six seeds should clear; yet the adequate BATTERY called episodic
borderline (off at one seed). The two measure different things: the diagnosis runner reads the organ's apical_cue
directly; the battery reads the INTEGRATED store→recall decision field. So episodic's seed-dependence lives in the
INTEGRATION path (the numpy BTSP store→recall pipeline at that seed), not in read-overlap. Its stabilizer target is
clear at the read level, but making the BATTERY verdict robust means stabilizing the integration path, not (only) the
operating point. prospective-memory's gap is the narrowest of the three (see the table) — its read sits near threshold,
so its stabilizer must lift the held×cue coincidence, not just move a constant.

## Method + provenance
Artifacts analyzed (committed, prov-sidecarred, numpy CPU): research/findings/raw/_lbf_borderline/op_s42.json,
research/findings/raw/_lbf_borderline/op_s43.json, research/findings/raw/_lbf_borderline/op_s44.json,
research/findings/raw/_lbf_borderline/op_s100.json, research/findings/raw/_lbf_borderline/op_s101.json,
research/findings/raw/_lbf_borderline/op_s102.json.
`python3` analysis over those six committed per-seed reads. No new brain
builds, no threshold moved, no artifact re-generated — this is a read-only separability measurement of already-committed
data, so it cannot tune anything.

## Next
NEXT ACTION #5 is GO (not honest-negative) for 3 of 4: build the principled operating-point stabilizer for episodic /
source-provenance / prospective-memory (regulator targeting the separable gap; for episodic, stabilize the store→recall
integration path; for pmem, lift the near-threshold held×cue coincidence). affect-marker: fix the WTA RNG-isolation bug
first, then re-assess separability. Each stays honest-negative if the principled regulator fails to hold 6-seed.
