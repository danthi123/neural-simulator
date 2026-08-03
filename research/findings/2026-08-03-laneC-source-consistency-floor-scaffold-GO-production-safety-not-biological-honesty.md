---
type: finding
status: contributing
date: 2026-08-03
mechanism: laneC-source-consistency-floor
runner: research/runners/_laneC_self_schema_honesty_wirein_derisk.py
artifacts:
  - research/findings/raw/lanes/metacog/laneC_self_schema_source_consistency_floor_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_source_consistency_floor_6seed.json.prov.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json.prov.json
---

# Lane C source-consistency floor: production scaffold GO, not final biological honesty

<!--derived-->
**One-line verdict.** A named `source_consistency_floor` scaffold closes the observed production high-confidence wrong
assertions: on the same six-seed stressed known-fact battery it downgraded 46/46 wrong recalls, left 0/133 correct
recalls source-mismatched, preserved 475/475 hard-moat abstains, added 0 false accepts, and left 0 wrong assertions.
This is a useful production safety floor, but it reads composer source metadata and is therefore a scaffold to burn
down into a neural source-memory consistency signal.

## What Changed

- `RFPhasorComposer` trace metadata now exposes read-only cleanup runner-up, margin, and conflict fields for each
  traced role, plus the matched source fact when trace is enabled.
- `SelfSchemaHonestyConfig.confidence_source_mode` defaults to the previous `trace` mode.
- `known_fact_confidence_record(...)` can run in `source_consistency_floor` mode: the selected source confidence is
  floored to zero when the matched source fact's cue or answer disagrees with the live decoded recall.
- `BrainConversationalAgent.known_fact_record(...)` keeps the hard moat first and routes the chosen scalar through the
  same spiking `meta_schema -> self_schema` relay.
- `CommunicableTurn._known_fact_channel(...)` now surfaces the confidence mode/evidence metadata.

Default public retrieval remains unchanged unless `enable_self_schema_honesty=True` and the caller opts into this
confidence source mode.

## Six-Seed Result

Command:

```bash
env SIM_BACKEND=numpy .venv/bin/python -m research.runners._laneC_self_schema_honesty_wirein_derisk \
  --confidence-source-mode source_consistency_floor \
  --json research/findings/raw/lanes/metacog/laneC_self_schema_source_consistency_floor_6seed.json
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
| source-mismatched wrong recalls | 46 |
| source-mismatched wrong recalls downgraded | 46 |
| source-mismatched correct recalls | 0 |
| hard-moat abstains preserved | 475/475 |
| added false accepts | 0 |
| self-schema invocations on hard moat | 0 |
| correct assert rate | 72/133 |

The trace-only baseline artifact remains the scientific boundary:
`laneC_self_schema_honesty_wirein_6seed.json` is still **PARTIAL**, with 4/46 wrong recalls asserted when raw trace
confidence was high. That is why this finding is explicitly scoped as a scaffold, not a solved metacognitive truth
signal.

## Interpretation

The scaffold catches the actual failure mode in this stressed RF composer: the query selects a source block, but the
cleanup-decoded cue or answer disagrees with the exact source metadata attached to that block. In production terms, the
brain can now avoid confidently speaking those mismatched recalls when the scaffold is enabled.

The caveat is load-bearing. An end-state biological brain cannot consult an exact Python source dictionary as a truth
oracle. The replacement should be a neural source-memory/source-monitoring readout: the source trace, cue trace, and
decoded answer should agree through substrate activity, and the dynamic ACC/aPFC/self-schema path should learn to treat
disagreement as low confidence. Until then, `source_consistency_floor` belongs in the scaffold ledger.

## Next Mechanism

1. Keep the scaffold default-off and visibly named.
2. Build a neural source-consistency readout that compares cue/source/answer activity without exact host labels.
3. Feed that readout, or a calibrated dynamic ACC/aPFC correctness signal, into the existing self-schema hook.
4. Re-run the same six-seed battery with source metadata disabled or permuted as a collapse control.

## 2026-08-03 Follow-Up

The first burn-down step is now banked. `confidence_source_mode="neural_source_consistency"` uses an independent RF
source-memory echo instead of the exact `source_fact` metadata for the selected consistency signal. It matched this
scaffold's safety result on the same six-seed battery: 46/46 wrong recalls downgraded, 0 wrong assertions, 0 source-
mismatch false positives on correct recalls, 475/475 hard-moat abstains preserved, and 0 added false accepts. Prefer
that mode over this exact metadata floor for current production safety experiments.

The new mode is still not final biological honesty: the redundant source echo is engineered at store time, not learned
developmentally. Finding:
`research/findings/2026-08-03-laneC-neural-source-consistency-echo-GO-independent-source-memory-burn-down.md`.
