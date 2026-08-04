---
status: qualified
lane: parameter-research
date: 2026-08-04
type: finding
---

# Parameter Research Live Discovery Smoke

Date: 2026-08-04

The parameter-gap workflow's live scholarly adapter was checked against the
current OpenAlex and Crossref APIs with this bounded query:

`substantia nigra pars reticulata NALCN conductance pacemaking`

Both providers responded successfully on the first attempt with three metadata
records each. The normalized result retained six unique source leads and
identified open/full-text URL leads where providers supplied them. The output
remained explicitly `metadata_only`: no numerical value, exact locator,
experimental condition, or scientific claim was inferred from search metadata.
The machine-readable smoke receipt is
`research/findings/raw/parameter-research-live-discovery-smoke.json`.

This proves that live discovery is operational, not that any returned paper
contains the missing SNr parameter. Full text must still be retrieved and read,
then any proposed claim must pass the existing research-packet review and RAG
intake gates.
