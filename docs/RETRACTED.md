# Retraction registry

Rows are appended when a document's **central claim** dies. Governed files (see [`docs/WRITING.md`](WRITING.md))
may not cite a registered path without `⛔` on the same line or bullet.

A row here does not mean the whole document is worthless — retractions are often **partial**, and the "why" column
states what died. The superseding document is authoritative for the part that died.

| path or commit | date | superseded by | why (<=20 words) |
|---|---|---|---|
| `research/findings/2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md` | 2026-07-26 | `research/findings/2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md` | Dense-CA1 attribution was an artifact of a 333x apical_R miscalibration; the code is sparse and fact-specific. |
| `research/findings/2026-08-06-source-monitor-coresidency-v6-calibration-GO-learning-off-silent-by-construction.md` | 2026-08-06 | `research/findings/2026-08-06-source-monitor-stepping-history-instrument-FIXED-v6-and-v9-calibration-GO-were-artifacts.md` | v6 "calibration GO" on the weakest-source margin was a stepping-history instrument artifact (min(M)==min(L) under fixed instrument); leak-closure sub-result survives. |
