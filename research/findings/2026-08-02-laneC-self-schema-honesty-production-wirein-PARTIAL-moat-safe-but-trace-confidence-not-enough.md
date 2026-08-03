---
type: finding
status: contributing
date: 2026-08-02
mechanism: laneC-self-schema-honesty-production-wirein
runner: research/runners/_laneC_self_schema_honesty_wirein_derisk.py
artifacts:
  - research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json
  - research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json.prov.json
---

# lane C self-schema honesty: production wire-in is moat-safe, but only partial as a truth signal

<!--derived-->
**One-line verdict.** The production conversation hook is now built default-off and preserves the hard no-confab moat:
the six-seed stressed battery recorded 475/475 hard-moat abstains preserved, 0 added false accepts, and 0 self-schema
invocations on hard moat misses. It also downgraded most low-confidence familiar-but-wrong recalls, 31/32, but one
low-confidence wrong recall and 9/46 total wrong recalls still asserted. So this is a useful **PARTIAL** production
wire-in, not an honesty GO: trace confidence alone is not enough.

## What Changed

The new default-off path is deliberately narrow:

- `BrainConversationalAgent.known_fact_record(cue)` wraps existing `what_does` and yes/no retrieval with structured
  certainty metadata.
- `research/runners/self_schema_honesty.py` adds a small fixed spiking `meta_schema -> self_schema` confidence relay.
- `CommunicableTurn._known_fact_channel` uses the structured record only when `enable_self_schema_honesty=True`.
- The old hard moat remains first. If retrieval returns `None` or `"unknown"`, self-schema is not built or invoked.
- The self-schema path can only downgrade a matched answer into a hedge or soft abstain. It cannot turn an unknown cue
  into an answer.

This is a production behavior hook, not a claim of subjective experience.

## Six-Seed Result

Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneC_self_schema_honesty_wirein_derisk \
  --json research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | PARTIAL |
| matched known-fact queries | 288 |
| matched-query hard abstains | 109 |
| correct matched recalls | 133 |
| wrong matched recalls | 46 |
| wrong recalls still asserted | 9 |
| low-confidence wrong recalls | 32 |
| low-confidence wrong recalls downgraded | 31 |
| hard-moat abstains preserved | 475/475 |
| added false accepts | 0 |
| self-schema invocations on hard moat | 0 |

Per seed:

| seed | wrong | wrong asserted | low-conf wrong | low-conf downgraded | hard moat | added false accepts |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 1 | 6 | 6 | 80/80 | 0 |
| 43 | 5 | 2 | 3 | 2 | 79/79 | 0 |
| 44 | 9 | 3 | 3 | 3 | 78/78 | 0 |
| 100 | 9 | 1 | 7 | 7 | 79/79 | 0 |
| 101 | 8 | 1 | 7 | 7 | 80/80 | 0 |
| 102 | 8 | 1 | 6 | 6 | 79/79 | 0 |

## Interpretation

The integration shape is right: default-off behavior is preserved, the moat is still load-bearing, and the
self-schema relay is in the normal conversation path rather than only in an isolated research runner. The failure is
also clear: source trace confidence is an operating-point-dependent recall score, not a learned correctness estimate.
When the composer is confidently wrong, the self-schema relay receives a strong confidence current and may assert.

## Next Mechanism

Do not promote this as production honesty. The next Lane C step is to feed the self-schema from a calibrated or learned
correctness-confidence signal, probably the dynamic ACC/aPFC monitor already validated in the isolated runner, plus a
per-domain calibration window. The production hook can stay default-off as the seam for that signal.
