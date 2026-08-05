---
type: research-finding
status: complete
lane: gateb-v14-source-constrained-identification
date: 2026-08-05
claim_check: measured-negative-result
artifacts:
  - research/specs/v14_snr_stageB_fourth_blind_extraction_consensus_v1.json
  - research/measurements/v14-stageB/extraction-groups-v2.json
  - research/measurements/v14-stageB/extractor_d/inspection/verification_manifest.json
  - research/measurements/v14-stageB/adjudications-four-way
---

# Stage B four-extractor population targets remain unresolved

**Decision:** do not issue calibration, validation, or held-out target packets and do not start parameter fitting.

Extractor D independently remeasured all seven eligible source panels from native pixels while blind to extractors A/B/C, their comparisons, prior adjudications, and target partitions. It produced seven available records containing 83 numeric measurements and four explicitly unavailable cells. Controller-side validation reproduced every digitization exactly and verified all record, annotation, source-asset, output, and self-digests.

The prospective four-way consensus rule was committed as `950fab579` before D began, and its implementation was committed as `a040f4628` before D completed. The rule requires a three-of-four panel and exact command-set vote, followed by a unique maximum complete pairwise-agreement clique of at least three extractors for every point. No nearest-x matching, tolerance change, or post-hoc marker association was used.

## Result

| Source panel | Panel vote | Command set | Resolved points | Unresolved points | Verdict |
|---|---|---|---:|---:|---|
| Kv3 activation, Fig. 8 C1 | available | resolved | 6 | 8 | unresolved |
| Kv3 inactivation, Fig. 8 C2 | available | resolved | 6 | 6 | unresolved |
| Kv3 deactivation, Fig. 9 B | available | resolved | 0 | 5 | unresolved |
| Fast Na activation, Fig. 6 A4 | available | resolved | 0 | 14 | unresolved |
| Fast Na inactivation, Fig. 6 B4 | available | unresolved | 0 | 0 | unresolved |
| Fast Na recovery, Fig. 7 D | unresolved | not applicable | 0 | 0 | unresolved |
| Fast Na deactivation, Fig. 9 C | available | resolved | 2 | 7 | unresolved |

All seven adjudications have `status = four_extractions_unresolved`, `optimization_allowed = false`, and no scientific verdict. Partial target packets are prohibited, so the resolved subsets above cannot be used for fitting.

## Interpretation

The source images do not support a complete custody-safe numerical target set under the preregistered uncertainty and agreement rules. This is a measurement limitation, not evidence against any channel model. Repeating parameter searches against hand-selected or averaged points would confound model error with extraction choice.

The next scientific action must seek stronger measurement authority: original numeric source data, a higher-fidelity/vector source, or a newly preregistered acquisition method with an independent error model. A fifth manual extractor alone is not automatically informative and must not be used under an invented post-hoc voting rule. Existing thresholds and four-way outputs remain frozen.

## Bounded data-availability search

A same-day DOI-, title-, author-, and repository-specific web search found the two primary articles, the PMC HTML/figure assets, the published summary tables, NeuroElectro's sodium summary table, and the Kv3 supplementary-figures PDF. It found no Dryad, Figshare, Zenodo, GitHub, institutional, or DOI-linked archive containing the original point-level activation, inactivation, recovery, or deactivation data. PMC's open-access-package API reports both article identifiers as not open access. APS denied direct terminal retrieval of the article PDFs with HTTP 403, so whether an authorized publisher PDF contains recoverable vector paths remains unverified. The papers identify Fu-Ming Zhou as the corresponding author; requesting original numeric data is a valid later user-approved external action, not something this result silently assumes will succeed.

## Execution evidence

- Group manifest: `research/measurements/v14-stageB/extraction-groups-v2.json`, self-digest `c8d5036698dcdb77e2808013ab0db699ad6a82a5c9a96093798287db89ca2859`.
- D verification manifest: seven panels, 83 numeric points, four unavailable cells; all hashes, records, native crops, and deterministic outputs verified.
- Four-way implementation verification: 58 focused and adjacent local tests plus 41 target/digitization/scorer tests on pool40 at immutable revision `a040f4628`.
- No target packet directory was created and no optimizer or GPU campaign was launched.
