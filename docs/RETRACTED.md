# Retraction registry

Rows are appended when a document's **central claim** dies. Governed files (see [`docs/WRITING.md`](WRITING.md))
may not cite a registered path without `⛔` on the same line or bullet.

A row here does not mean the whole document is worthless — retractions are often **partial**, and the "why" column
states what died. The superseding document is authoritative for the part that died.

| path or commit | date | superseded by | why (<=20 words) |
|---|---|---|---|
| `research/findings/2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md` | 2026-07-26 | `research/findings/2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md` | Dense-CA1 attribution was an artifact of a 333x apical_R miscalibration; the code is sparse and fact-specific. |
| `research/findings/2026-08-06-source-monitor-coresidency-v6-calibration-GO-learning-off-silent-by-construction.md` | 2026-08-06 | `research/findings/2026-08-06-source-monitor-stepping-history-instrument-FIXED-v6-and-v9-calibration-GO-were-artifacts.md` | v6 "calibration GO" on the weakest-source margin was a stepping-history instrument artifact (min(M)==min(L) under fixed instrument); leak-closure sub-result survives. |
| `research/findings/2026-08-10-shared-channel-arity-capacity-located-M-star-grows-with-dimension.md` | 2026-08-10 | `research/findings/2026-08-10-shared-channel-arity-capacity-CORRECTED-DC-offset-artifact.md` | The "M* grows ~√d / disjoint isolates crosstalk" break was a removable readout DC-offset (common-mode) artifact; adversarial-verify caught it. With label-free common-mode removal, shared-channel superposition composes 1.00 through M=6 at d=8 — no break in range. |
| `research/findings/2026-08-10-episodic-dialogue-recall-wired-to-spiking-dAP-readout-numpy-backend-honest-negative.md` (PARTIAL: the numpy "backend-block" claim only) | 2026-08-10 | same file, §⛔CORRECTION (2026-08-10) | The numpy "dendritic apical read does not fire / forward-Euler backend-blocked (0.000)" was measured at kthresh=30 — a non-firing point on BOTH backends. At the corrected kthresh=8 the module fires cue-specifically on numpy (0.93/0.91) AND cupy 6/6 (fresh isolated builds). The wiring / no-regression / store-fires sub-results survive. |
