# Research escalation at an experimental wall

Use this workflow after two distinct attempts fail against the same biological
or behavioral defect. It stops another tuning cycle, checks prior work and local
scientific sources, turns missing knowledge into answerable questions, and
leaves a versioned record that future RAG searches can recover.

## 1. Start the gate

```bash
python tools/research_escalation.py start \
  --slug gpi-tonic-output \
  --blocked-experiment "GPi output remains silent without host current" \
  --wall-reason "Two mechanism changes did not produce autonomous firing" \
  --failed-attempt "Raised synaptic drive; firing stayed input-dependent" \
  --failed-attempt "Retuned reset parameters; zero-input cells stayed silent" \
  --parameter-question "What firing-rate and variability bounds are measured without fast synaptic input?" \
  --wiring-question "Which GPe cell classes contact GPi output cells, at which compartments and sign?"
```

The command checks prior project work, verifies that the canonical index is
current, queries the plan corpus, and runs the primary-source research gate. It
records one of three states:

- `local-reading-required`: relevant local primary material was found. Read it
  before changing the implementation.
- `no-relevant-local-evidence`: retrieval worked, but no relevant local primary
  evidence passed the relevance floor. Begin a documented external search.
- `retrieval-unavailable`: a command failed, the index is stale/unavailable, or
  output was malformed. This is an infrastructure failure, not evidence that
  the literature is silent.

After `retrieval-unavailable`, repair the index or RAG environment and run:

```bash
python tools/research_escalation.py retry-retrieval --gate research/findings/<gate>.md
```

Finalization remains blocked until a successful retry is recorded. Every
mutation uses a file lock, an atomic replacement, and a monotonic revision so
parallel agents cannot silently overwrite one another.

## 2. Record external searches

A normal discovery search can be recorded with one database and query. A claim
that a parameter or wiring detail was *not found* has a higher bar: one question
per protocol, at least two databases, at least two query variants, the searched
publication-date range, and a result/search URL for each database.

```bash
python tools/research_escalation.py record-search \
  --gate research/findings/<gate>.md --questions P1 --claim-absence \
  --database PubMed --database "Crossref citation graph" \
  --query "GPi autonomous firing receptor blockade quantitative" \
  --query "entopeduncular nucleus spontaneous rate synaptic blockers" \
  --date-from 1900-01-01 --date-to 2026-08-04 \
  --url "https://pubmed.ncbi.nlm.nih.gov/?term=..." \
  --url "https://search.crossref.org/?q=..." \
  --outcome "No primary report supplied the requested preparation-matched bound"
```

`answer --status not-found` only accepts a complete absence protocol tied to
that exact question. One broad search cannot dispose of every open question.

## 3. Intake useful sources

```bash
python tools/research_escalation.py record-source \
  --gate research/findings/<gate>.md --questions P1 \
  --kind peer-reviewed-primary --citation "Authors (year), title" \
  --url "https://doi.org/..." --query "..." \
  --locator "Methods, Figure 3" \
  --evidence "Preparation, measured range, variance, and stated exclusions" \
  --license-status metadata-only
```

Every source gets a durable metadata/evidence record in the canonical source
catalog and an intake-ledger entry. The existing incremental index updater then
runs, followed by a source-specific RAG query. A source cannot resolve a
question until that query retrieves its intake record.

Use `--license-status open-access`, `public-domain`, or `permission-granted`
with `--local-file <path>` to archive a lawful local copy. `metadata-only`
preserves citation, locator, evidence, and URL but refuses a local copy. The
workflow does not infer licensing or bypass paywalls. PDF text extraction
remains the responsibility of the existing catalog extraction workflow; the
metadata record itself is immediately indexed.

Reviews and secondary sources remain useful discovery aids but cannot resolve a
parameter or wiring question. A primary preprint can resolve one only with its
lower evidence classification visible.

## 4. Answer and finalize

```bash
python tools/research_escalation.py answer \
  --gate research/findings/<gate>.md --question P1 --status resolved \
  --answer "Use the preparation-matched range" --references S1,S2

python tools/research_escalation.py finalize \
  --gate research/findings/<gate>.md --decision "Use a bounded scan" \
  --next-experiment "One preregistered lesion test that separates the remaining mechanisms"
```

Finalization refuses open questions, unresolved retrieval failures, source
records that are absent from RAG, and incomplete absence claims. It then runs
the repository's normal finding evidence gates. Inspect all captured commands,
outputs, revisions, sources, and search protocols with:

```bash
python tools/research_escalation.py inspect --gate research/findings/<gate>.md
```
