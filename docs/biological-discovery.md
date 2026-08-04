# Biological discovery

`tools/biological_discovery.py` is the first online discovery step used when a simulation experiment reaches a documented research wall. It searches Europe PMC, Crossref, and OpenAlex for primary-study candidates that may contain missing biological parameters, wiring measurements, or experimental methods.

It does not decide that a paper contains the needed value. Abstracts and metadata are discovery leads only. A candidate can influence implementation only after someone reads the primary source, records an exact page, figure, table, or methods locator, and sends it through the existing research-escalation and source-intake checks.

## Input

Create a JSON wall with the blocked experiment, preparation, mechanisms, and at least one parameter and wiring question:

```json
{
  "schema": "biological-discovery-wall-v1",
  "wall_id": "gpi-tonic-output",
  "blocked_experiment": "GPi output cells do not sustain autonomous tonic firing.",
  "wall_reason": "Two bounded scans remained trial-bound.",
  "preparation": {
    "species": "mouse",
    "brain_region": "globus pallidus internus",
    "cell_type": "GPi projection neuron",
    "state": "adult ex vivo slice",
    "recording": "whole-cell patch clamp"
  },
  "mechanisms": ["HCN current", "persistent sodium current"],
  "parameter_questions": [
    {"id": "P1", "text": "What autonomous firing rate and HCN conductance are measured?"}
  ],
  "wiring_questions": [
    {"id": "W1", "text": "Which inhibitory inputs set tonic firing and where do they terminate?"}
  ],
  "prior_attempts": ["Current sweep did not generalize."]
}
```

Run discovery with a new output path:

```bash
python tools/biological_discovery.py \
  --wall research/queue/gpi-tonic-output-wall.json \
  --output research/queue/gpi-tonic-output-discovery.json
```

The destination is create-only. If any provider request fails or returns invalid JSON, discovery fails and no packet is emitted. A successful search with no candidates is recorded as complete, but every question remains unresolved.

## Output and review

The packet records every generated query, request timestamp, exact API URL, normalized source metadata, deduplication identities, lawful full-text links reported by the providers, ranking factors, and unresolved fields. Ranking favors preparation matches, named mechanisms, quantitative methods, likely primary studies, and accessible full text.

Every abstract excerpt keeps its source locator and is labeled `review-required`. It never counts as an exact parameter claim. Before source intake:

1. Confirm that the work is a primary experiment and matches the simulated preparation.
2. Read the methods and results, not only the abstract.
3. Record the exact value, units, conditions, uncertainty, and page/figure/table/methods locator.
4. Check the license before archiving any local full-text copy.
5. Use `tools/research_escalation.py record-source`; that command delegates durable registration to `tools/rag/source_intake.py`.

## Current limits

- The three services expose overlapping but incomplete metadata; ranking is lexical and deterministic, not a scientific conclusion.
- The tool does not download or parse full text, infer missing values, assess study quality, or resolve conflicting measurements.
- Open-access links are retained only when provider metadata supplies an access basis. Crossref full-text links require attached license metadata.
- Provider completion proves that the configured searches ran, not that the literature contains no relevant paper.
