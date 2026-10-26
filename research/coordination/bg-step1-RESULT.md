# BG step-1 grid RESULT - [bg-step1], STEP1-SPEC STOP rule 4

claim_check: measured (research.runners.host_share_bg_seam --score --seeds ... --commit, 16 runs: 8 sets x {all-8 seeds 71-78, clean-5 71 73 74 76 78}; verdict read via the registered host_share_bg_seam.verdict() on each score.json; artifacts research/findings/raw/_bg_live_seam/step1/e0b8608 generated at pool revision 3f1249c798006f58e5963a893cc5d167e4f46c1e; spec docs/drafts/bg-tonic-loop/STEP1-SPEC.md approved at e0b8608c2; scorer file at main sha f64d3a991; backend numpy CPU, SIM_BACKEND=numpy, CUDA_VISIBLE_DEVICES= empty, scoring only, no simulation)

## Verdict: STOP rule 4 fires - every set UNDEFINED

Registered verdict() on each all-8 score.json (prereg precondition order: UNDEFINED never NO-GO). Exactly one precondition fails on seven sets - membrane in [-150, +100] mV in every neural-seam trial; stn50_gpe100 fails that plus the exact seam-event count (84/96, its s71 cell). No set reaches a bar read, so the spec's rule applies verbatim: if every set is UNDEFINED, STOP and report which regions before any bar uses v.

| set | all-8 status | membrane_ok | sham_ok | failed preconditions |
|---|---|---|---|---|
| stn50_gpe50 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn50_gpe100 | UNDEFINED | 0.5714 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial; exactly one neural-seam event per seam-arm turn (96) |
| stn100_gpe50 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn100_gpe100 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn200_gpe50 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn200_gpe100 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn400_gpe50 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |
| stn400_gpe100 | UNDEFINED | 0.625 | 1.0 | membrane in [-150, +100] mV in every neural-seam trial |

membrane_ok 0.625 on every intact set = 60/96 in-range trials; the 36 out-of-range trials are exactly seeds 72, 75, 77 x 12 seam-arm turns (stn50_gpe100 0.5714 = 48/84 with the same 36 breaches). This reproduces the step-1 breach record: seed-locked blow-up with byte-identical minima across sets (WORKLOG cross-set scans 03:48, 09:48, 21:27 on 10-25).

## Which regions: NOT in the step-1 record

Each neural-seam event stores one aggregate v_range (network min/max), not per-region voltage: e.g. stn50_gpe50 s72 seam_b hsb_lo = [-1425.8, +39.98] mV while the same event on clean seed 71 = [-97.0, +39.9]. Latch rates and tonic rates are per-latch/per-population; voltages are not. Naming the breaching region is therefore not derivable from step-1 artifacts - [bg-breach-diag] already recorded exactly this (per-cell indices, currents, jittered params absent, 126/126). The approved one-seed observe run (RULING batch-1026 (3): seed 72, event hsb_hi, per-cell currents/params) is the instrument that names it; until it runs, region identity stays ABSENT, never inferred.

## Clean-5 bars: reported context, never a selection

Per RULING batch-1026 (2): score_clean5.json pooled values with verdict() gates:

| set | F | empty_held | content_held | ns (G3) | commit_thr_change | bus_unique_agree | membrane_ok | gated keys true |
|---|---|---|---|---|---|---|---|---|
| stn50_gpe50 | 1.0 | 0.8 | 0.0 | 0.5 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn50_gpe100 | 1.0 | 0.75 | 0.0 | 0.4375 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn100_gpe50 | 1.0 | 0.8 | 0.0 | 0.55 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn100_gpe100 | 1.0 | 0.8 | 0.0 | 0.55 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn200_gpe50 | 1.0 | 0.9 | 0.0 | 0.6 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn200_gpe100 | 1.0 | 0.8 | 0.0 | 0.6 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn400_gpe50 | 1.0 | 0.9 | 0.0 | 0.55 | 0.0 | 1.0 | 1.0 | 6/6 |
| stn400_gpe100 | 1.0 | 0.8 | 0.0 | 0.65 | 0.0 | 1.0 | 1.0 | 6/6 |

Every set computes all six gated keys true on its clean-5 pooled values (F 1.0000, content_held 0.0000, bus_unique_agree 1.0000, commit_thr_change 0.0000, membrane and sham 1.0; empty_held 0.75-0.90 vs bar 0.75; ns 0.4375-0.6500 vs bar 0.30). Three reasons this is context and can never become a selection: (a) the frozen selection rule only considers sets whose full 8-seed block passes its preconditions, and all eight fail via STOP rule 4; (b) at exactly five seeds the registered verdict() itself returns UNDEFINED - content_held has 5 events, below min_events_per_measure 6 (one content event per seed), a structural property of scoring five seeds, not a set defect; (c) stn50_gpe100's clean-5 additionally lacks its s71 cell (48/60 seam events).

## Cell completeness and the s71 requeue (STOP rule 3)

63 of 64 seam cells landed (8 sets x seeds 71-78 except stn50_gpe100 s71) plus 8/8 prescreens. stn50_gpe100 s71 was excluded from the 71-line queue on purpose: it was the designated first canary (the 615-char checked reason: canary stn50_gpe100 s71 runs first via pool_canary auto --wait-s 5400 and its PASS lands in pool_canaries.tsv before this grid queues; the canary row exists - 2026-10-24T22:40:33, wspool, 3f1249c79, rc=0, wall=2930s, json=7, PASS). But the cell's output is absent everywhere: the empty dir on wspool (mtime 10-24 21:51 = its mkdir), no s71 json on any node, and $P's dir is empty. The canary's json=7 matches no s71 artifact path - record and artifacts disagree.

STOP rule 3 (a cell that dies without output is requeued VERBATIM exactly once with a canary on record): requeued once on 2026-10-26 - pool.queue depth 1; the line is the stn50_gpe100 s72 claims row with ONLY the two path occurrences s72->s71 and --seed 72->71 (verified: substituting back reproduces the original bytes), same revision, env currents (GPE 100 pA / STN 50 pA), mem_ok 8 2 + memcap 12; --checked carries the same reason plus the STOP 3 note, spec and prereg token, max_age_h=168; canary-on-record = the two PASS rows for this runner. On landing, re-score stn50_gpe100 (all-8 and clean-5) and amend this RESULT.

## Honesty boundary

Every quantity above is a functional read-out of the registered scorer over recorded artifacts - bars, rates, membrane fractions, event counts. Nothing asserts phenomenal experience; UNDEFINED is a preregistered measurement state, not a verdict on the mechanism's worth.

## Next per the spec

After a STOP, STEP1-SPEC section 6's work is not the driver's: reporting is done here; naming the breaching region needs the approved observe run; choosing the region and the parameter change before any bar uses v is Claude/owner work (bars, preregs and spec edits are section-2 reserved). Inputs for that decision: this table, spec 3.9's region table, and [bg-breach-diag].

## AMENDMENT 2026-10-26 12:05 - s71 landed, full grid re-scored, verdict unchanged

The requeued stn50_gpe100 s71 cell landed complete (7/7 arms + 7 prov sidecars, 11:06-11:47), so the STOP 3 loop is closed and the grid is 64/64 cells. s71's own membrane is clean (worst -97.9 mV), so clean-5 {71,73,74,76,78} now rests on data, not elimination. All sixteen scores were re-run with the registered scorer at 12:00 (flat symlink view rebuilt, 896 links, artifacts untouched).

all-8 = UNDEFINED on all eight sets with gates computing true and membrane_ok 0.625 (60/96; seeds 72/75/77 x 12; worst -1916.3 / -1639.7 / -460.6, tonic-pa independent); clean-5 = UNDEFINED structural (n_content 5 < min 6) with membrane 1.0 and gates true. Section 2's reason (c) is stale - stn50_gpe100's clean-5 now carries 60/60 seam events - but (a) and (b) alone keep clean-5 non-selectable.

STOP 3's disposition (re-score on landing) is fulfilled; the original canary record/artifact disagreement stands as logged in FAILURE_LOG. Next: the approved observe run (RULING batch-1026 (3)) to name the breaching region.
