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
invocations on hard moat misses. A follow-up monotonic source-confidence floor fixed the low-confidence assertion
defect: 32/32 low-confidence familiar-but-wrong recalls were downgraded. The result remains **PARTIAL**, not an
honesty GO, because 4/46 total wrong recalls still asserted when trace confidence was high.

## What Changed

The new default-off path is deliberately narrow:

- `BrainConversationalAgent.known_fact_record(cue)` wraps existing `what_does` and yes/no retrieval with structured
  certainty metadata.
- `research/runners/self_schema_honesty.py` adds a small fixed spiking `meta_schema -> self_schema` confidence relay.
- `CommunicableTurn._known_fact_channel` uses the structured record only when `enable_self_schema_honesty=True`.
- The old hard moat remains first. If retrieval returns `None` or `"unknown"`, self-schema is not built or invoked.
- The self-schema path can only downgrade a matched answer into a hedge or soft abstain. It cannot turn an unknown cue
  into an answer.
- Assertion now also requires source confidence above the configured assert floor. This prevents coarse spiking-rate
  quantization from making a low-confidence recall assertive.

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
| wrong recalls still asserted | 4 |
| low-confidence wrong recalls | 32 |
| low-confidence wrong recalls downgraded | 32 |
| hard-moat abstains preserved | 475/475 |
| added false accepts | 0 |
| self-schema invocations on hard moat | 0 |

Per seed:

| seed | wrong | wrong asserted | low-conf wrong | low-conf downgraded | hard moat | added false accepts |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 0 | 6 | 6 | 80/80 | 0 |
| 43 | 5 | 0 | 3 | 3 | 79/79 | 0 |
| 44 | 9 | 1 | 3 | 3 | 78/78 | 0 |
| 100 | 9 | 1 | 7 | 7 | 79/79 | 0 |
| 101 | 8 | 1 | 7 | 7 | 80/80 | 0 |
| 102 | 8 | 1 | 6 | 6 | 79/79 | 0 |

## Interpretation

The integration shape is right: default-off behavior is preserved, the moat is still load-bearing, and the
self-schema relay is in the normal conversation path rather than only in an isolated research runner. The monotonic
source floor closes the relay quantization failure for low-confidence errors. The remaining failure is higher level:
source trace confidence is an operating-point-dependent recall score, not a learned correctness estimate. When the
composer is confidently wrong, the self-schema relay receives a strong confidence current and may assert.

## Next Mechanism

Do not promote this as production honesty. The next Lane C step is to feed the self-schema from a calibrated or learned
correctness-confidence signal, probably the dynamic ACC/aPFC monitor already validated in the isolated runner, plus a
per-domain calibration window. The production hook can stay default-off as the seam for that signal.

## 2026-08-03 Follow-Up

A named `source_consistency_floor` scaffold now catches the observed high-confidence wrong recalls by requiring the
matched composer source metadata to agree with both the asked cue and decoded answer. That six-seed artifact is GO
(`research/findings/raw/lanes/metacog/laneC_self_schema_source_consistency_floor_6seed.json`: 46/46 wrong recalls
downgraded, 0 wrong assertions, 0 added false accepts), but it is explicitly not the final biological mechanism because
it reads exact source metadata. Finding:
`research/findings/2026-08-03-laneC-source-consistency-floor-scaffold-GO-production-safety-not-biological-honesty.md`.
