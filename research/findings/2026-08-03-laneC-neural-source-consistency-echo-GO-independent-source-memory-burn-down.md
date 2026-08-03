---
type: finding
status: contributing
date: 2026-08-03
mechanism: laneC-neural-source-consistency-echo
runner: research/runners/_laneC_self_schema_honesty_wirein_derisk.py
artifacts:
  - research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_6seed.json.prov.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_smoke_s44.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_smoke_s44.json.prov.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_source_consistency_floor_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json
---

# Lane C neural source-consistency echo: six-seed GO for the first source-metadata burn-down

<!--derived-->
**One-line verdict.** The `neural_source_consistency` mode replaces the exact composer source-metadata floor with an
independent RF source-memory echo for confidence selection. On the same six-seed stressed known-fact battery, it
downgraded 46/46 wrong recalls, left 0/46 wrong recalls asserted, added 0 false accepts, preserved 475/475 hard-moat
abstains, and had 0 correct source-mismatch false positives. This is the preferred current safety mode over
`source_consistency_floor`, but it is still a bounded source-monitoring scaffold, not final biological honesty.

## Role In The Whole Brain

The self-schema honesty path should let the conversation system answer, hedge, or abstain based on what the brain's own
state says about reliability. The immediate failure mode was familiar-but-wrong recall: the memory cue matched a stored
region, but cleanup returned a confident wrong answer. Raw trace confidence could not catch every case, and exact source
metadata caught it only by reading a Python fact record.

This run tests the next step: ask a separate RF memory trace, with independent concept and role codes, whether the live
candidate answer matches the same cue. If that source echo disagrees, the self-schema confidence input is floored before
speech rendering.

## What Changed

- `RFPhasorComposer` now has an opt-in source monitor:
  `enable_source_monitor=True, source_monitor_D=64`.
- The monitor writes a redundant RF/FHRR source-memory echo using independent role and concept codebooks.
- `source_consistency_record(kind, cue, raw_answer)` decodes cue and answer roles from that echo and returns only
  source-monitor evidence, not the exact `source_fact`.
- `known_fact_confidence_record(...)` adds `confidence_source_mode="neural_source_consistency"`.
- If that mode is selected but the source echo is unavailable or fails to match, confidence fails closed instead of
  silently falling back to raw trace confidence.
- `BrainConversationalAgent.known_fact_record(...)` passes the source-monitor evidence into the default-off
  self-schema honesty hook.
- The previous exact-metadata floor remains available and visibly named as a scaffold.

Default behavior remains unchanged unless self-schema honesty is enabled and the caller selects the neural source
consistency mode.

## Six-Seed Result

Command:

```bash
env SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._laneC_self_schema_honesty_wirein_derisk \
  --confidence-source-mode neural_source_consistency \
  --json research/findings/raw/lanes/metacog/laneC_self_schema_neural_source_consistency_6seed.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | GO |
| matched known-fact queries | 288 |
| matched-query hard abstains | 109 |
| correct matched recalls | 133 |
| wrong matched recalls | 46 |
| wrong recalls still asserted | 0 |
| low-confidence wrong recalls downgraded | 46/46 |
| source-mismatched wrong recalls downgraded | 46/46 |
| source-mismatched correct recalls | 0 |
| correct recalls asserted | 72/133 |
| hard-moat abstains preserved | 475/475 |
| added false accepts | 0 |
| self-schema invocations on hard moat | 0 |

The seed-44 smoke also passed before the six-seed promotion:
`laneC_self_schema_neural_source_consistency_smoke_s44.json`.

## Interpretation

This is a real burn-down of the bluntest shortcut in `source_consistency_floor`: confidence selection no longer needs to
read the exact Python source fact attached to the trace. The selected source consistency signal now comes from an
independent RF source-memory echo. That makes the current production safety story more brain-like and less
database-like.

The caveat remains load-bearing. The source echo is still engineered at storage time with a separate codebook. It is a
neural-style redundant memory trace, not a learned developmental source-monitoring circuit. It also still sits inside a
known-fact stress battery rather than a full lived conversation loop.

## Negative Probes That Shaped This

Two cheap alternatives were rejected before this promotion:

- Asking a yes/no verifier about the candidate was invalid because polarity confused existence; wrong and correct
  recalls both produced mixed yes/no answers.
- A reciprocal query-agent check failed because wrong recalls often stayed internally self-consistent at the cue level.

Replay-stability probing also showed the wrong recalls were stable representation/source ambiguities, not momentary
noise. That is why a redundant source-memory signal was the useful next burn-down.

## Next Mechanism

1. Treat `neural_source_consistency` as the preferred current known-fact safety mode over exact source metadata.
2. Do not call it final honesty. Add it to the scaffold ledger as an engineered source-memory echo.
3. Replace the store-time echo with plastic source tags or a learned source-monitoring circuit.
4. Feed that learned source-monitoring signal through dynamic ACC/aPFC and the self-schema, with source-code
   permutation/disable controls.
5. Move the test from known-fact stress batteries into the minimal lived speech-action loop.
