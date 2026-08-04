# Experiment planner

`tools/experiment.py` converts a machine-readable preregistration into reproducible job commands. It automates
bookkeeping and mechanical gates, not scientific interpretation, parameter selection, or promotion decisions.

## Spec contract

The loader accepts the existing `sim-experiment-spec-*` JSON format and requires explicit, non-overlapping
`calibration` and held-out seed partitions plus a backend list. A spec becomes executable by adding:

```json
{
  "execution": {
    "command": [
      ".venv/bin/python", "-u", "-m", "research.runners.example",
      "--seed", "{seed}", "--phase", "{partition}",
      "--arm", "{arm}", "--out", "{output}"
    ],
    "output": "research/findings/raw/example/{partition}/{backend}/{arm}-{seed}.json",
    "arms": ["control", "treatment"],
    "targets": {
      "numpy": {"device": "cpu", "lane": "pool"},
      "cupy": {
        "device": "cuda:0", "lane": "gpu",
        "env": {"CUDA_VISIBLE_DEVICES": "0"}
      }
    },
    "corpus_reason": "new-config"
  }
}
```

Commands must use `.venv/bin/python -m research.runners...`, preserving automatic run provenance. GPU and pool
jobs are emitted through `tools/queue_add.sh`; local jobs contain the direct command. Backend, seed, partition,
and arm ordering is sorted, so the same source and spec produce the same job IDs and plan ordering.

Every output is repository-relative and unique. A job atomically creates `<output>.claim` before launch. Existing
outputs or claims block planning, and a claim prevents two queued copies from silently writing the same result.

## Scientific gates

An optional prerequisite names evidence that must exist at an exact digest before selected partitions can be
planned:

```json
{"id": "calibration", "path": "path/result.json", "sha256": "...", "partitions": ["replication"]}
```

An optional stop rule blocks later partitions until a person or analysis runner records a decision:

```json
{"id": "calibration-gate", "blocks": ["replication"], "decision_file": "path/decision.json"}
```

The decision file must contain the same `rule_id`, the canonical `spec_sha256`, and `decision` set to exactly
`continue` or `stop`. The planner verifies the record and obeys it; it does not decide which value is warranted.

## Seal and plan

Create seals and plans outside the Git worktree so they do not make the source dirty:

```bash
.venv/bin/python tools/experiment.py seal \
  --spec research/specs/example.json --seal /tmp/example.seal.json

.venv/bin/python tools/experiment.py plan \
  --spec research/specs/example.json --partition calibration \
  --plan-dir /tmp/example-calibration-plan

.venv/bin/python tools/experiment.py plan \
  --spec research/specs/example.json --partition held_out \
  --seal /tmp/example.seal.json --plan-dir /tmp/example-heldout-plan
```

A seal records an exact canonical spec hash and a clean Git revision, or a verified cluster source manifest.
Held-out planning is refused without a matching seal and is also refused if source or config changed afterward.
Plan directories are create-only and contain a read-only JSON manifest and command file for every job.
