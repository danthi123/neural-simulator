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

## Harvest: affect-marker RNG-isolated 6-seed (2026-09-22, the prerequisite resolved)
The affect-marker RNG-isolation bug (shared reader advancing the OU-noise RNG between the intact and lesion reads) was
fixed (fresh reader per arm, same seed; commit on research/runners/_lbf_borderline_operating_point.py) and the
diagnosis re-run 6-seed (numpy CPU, serial, memcap). Artifacts:
research/findings/raw/_lbf_borderline_isolated/op_s42.json … op_s102.json + research/findings/raw/_lbf_borderline_isolated_run.out.
<!--derived-->
- **RNG-isolated affect-marker is load-bearing in 1/6 seeds (only s100), down from the pre-fix 4/6** — the extra
  pre-fix flips were RNG-confounded (the non-isolated reader gave the two arms different noise draws).
- ROOT CAUSE (now unambiguous): the INTACT WTA winner-minus-runner-up margin clears DEAD_MARGIN=0.05 on only 1/6 seeds
  (s100 margin ~0.16; the other five ~0.033–0.044, i.e. BELOW the dead-zone → no clean winner even INTACT → empty
  lead). So affect-marker's problem is NOT operating-point placement (as for the separable three) — the intact WTA
  simply does not commit to a clean marker at the 'emo' mood level on most seeds. It is therefore essentially
  NOT-load-bearing under a fair (isolated) read, and is NOT a stabilizer candidate the way episodic/source-provenance/
  prospective-memory are. HONEST consequence for the #1 metric: the earlier 23/26 union count was partly inflated by
  affect-marker's RNG-confounded flips; the robust core 20/26 is unaffected (affect-marker was never in it).

## Next — the #5 stabilizer build, DECOMPOSED (it is two different problems, not one)
The full harvest shows the three genuinely-separable faculties split by WHERE the seed-dependence actually lives, and
they need DIFFERENT fixes. Attacking them all as "operating-point stabilizers" would waste effort on two of them.
- **prospective-memory = READ-LEVEL (the one true operating-point case).** Its intact `rel` (held×cue coincidence)
  sits just BELOW FIRE_THR=0.2 at s44 while clearing it on the other five. The fix is a biology-grounded short-term
  FACILITATION of the held-intention assembly across the intervening turns (the SFA/NMDA facilitation the diagnosis
  already named) that lifts `rel` to reliably clear FIRE_THR by the cue — a homeostatic regulator, NOT a moved constant.
- **episodic-memory + source-provenance-honesty = INTEGRATION-LEVEL (NOT operating-point).** Their diagnosis READS
  separate cleanly on ALL 6 seeds (episodic apical_cue vs 0; source-provenance `d` = a PERFECT 1-vs-0 split), yet the
  battery called them 5/6 and 4/6. So the seed-dependence is NOT at the read's threshold — a stabilizer there fixes
  nothing. It is DOWNSTREAM, in the integrated decision pipeline (episodic: the numpy store→recall/BTSP step at s44;
  source-provenance: whatever discretizes/consumes the perfectly-separated `d` at s44/s102). The #5 work for these two
  is to LOCATE the seed-fragile integration step (an investigation), not to regulate the read.
- **affect-marker: DONE (not a candidate).** Isolated = 1/6; its intact WTA does not commit (dead-zone) on 5/6 — a fair
  near-negative; no operating-point stabilizer applies.
Each build stays honest-negative if the principled mechanism fails to hold 6-seed.
