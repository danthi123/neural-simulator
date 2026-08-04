---
status: qualified
lane: parameter-research
date: 2026-08-04
type: finding
---

# Parameter Research Full-Text Retrieval Smoke

The complete parameter-research handoff was exercised on the question of
NALCN conductance density in adult substantia nigra pars reticulata neurons.
Planning completed eight local checks across the project RAG and prior-failure
corpus before any network search. Four bounded scholarly queries then produced
six deduplicated metadata leads.

The workflow selected the open eLife paper *The leak channel NALCN controls
tonic firing and glycolytic sensitivity of substantia nigra pars reticulata
neurons* (DOI `10.7554/eLife.15271`). <!--derived--> It retrieved the provider's PDF, stored
1,031,328 bytes under its verified SHA-256 identity, and linked the receipt to
the exact metadata lead. State reload revalidated the closed receipt schema,
source binding, URL and MIME provenance, byte count, content-addressed path,
and stored-file digest.

The automated locator found one `Hz` passage on PDF page 13. It describes a
1 Hz high-pass filter used for spike detection and does not resolve the missing
NALCN conductance value. It remains a `candidate_parameter_locator` with
`claim_status: not_a_claim` and `review_status: pending_review`. No numerical
claim, parameter choice, or scientific conclusion was accepted.

The durable workflow state and receipt are in
`research/findings/raw/parameter-research-fulltext-retrieval-smoke.json`. The
downloaded PDF remains in the local content store rather than the repository;
its SHA-256 is
`26457fde81995524cd3e656c6630ad4d70b6ccb4b432d9db2a0efe2031b598e2`.

This smoke establishes that RAG-first planning, live metadata discovery,
full-text retrieval, durable linking, and the pending-review evidence boundary
operate together. It does not establish that automated passage matching has
found the required biological parameter.
