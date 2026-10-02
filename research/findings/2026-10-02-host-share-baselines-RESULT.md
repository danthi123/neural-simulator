---
type: finding
status: live
date: 2026-10-02
lane: C-self-workspace
mechanism: host-decision share baseline, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/host_share.py
artifacts:
  - research/findings/raw/_host_share/gate/aggregate.json
  - research/findings/raw/_host_share/gate/report.md
---

# Host-decision share baselines: result (2026-10-02)

Prereg: research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md (gate revision f7f2bd1690783e60200d443c7442f63d3e5994a5). Scored by `python -m research.runners.host_share --aggregate ... --seeds 42,43,44,100,101,102`. Artifact: research/findings/raw/_host_share/gate/aggregate.json.

<!--derived-->

**Verdict (preregistered validity): BASELINE-VALID** (seeds [42, 43, 44, 100, 101, 102])

| seam | reported as | class | HS | NS | MS | SC | DA | n events with F | live seeds |
|---|---|---|---|---|---|---|---|---|---|
| k-select | MEASURED | HOST-OWNED | 1.000 | 0.000 | 0.000 | 0.233 | 1.000 | 20 | 6 |
| v-choice | MEASURED | HOST-OWNED | 1.000 | 0.000 | 0.000 | 0.500 | 0.545 | 10 | 6 |
| bg-select | MEASURED | HOST-OWNED | 0.889 | 0.111 | 0.000 | 0.667 | 0.500 | 18 | 6 |
| organ-c | CALL-GRAPH CHECK | HOST-OWNED | 1.000 | 0.000 | 0.000 | 0.667 | 0.333 | 18 | 6 |
| metacog | CALL-GRAPH CHECK | HOST-OWNED (replaceable) | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 18 | 6 |
| ignition | CALL-GRAPH CHECK | MIXED | 0.643 | 0.000 | 0.357 | UNDEFINED | 1.000 | 28 | 6 |

Panel, measured seams (k-select, v-choice, bg-select): host_decision_share 0.958, neural_decision_fraction 0.042, masked_fraction 0.000, replaceable_share_of_host 0.543, n events 48.

Panel, call-graph checks (organ-c, metacog, ignition): host_decision_share 0.844, neural_decision_fraction 0.000, masked_fraction 0.156, replaceable_share_of_host 0.444, n events 64.

Content (30 turns): learned_content_fraction 0.667, python_selected_fraction 0.333, unattributed 0.000, fact_word_share 0.795.

Validity: V1 sham-clean seeds [42, 43, 44, 100, 101, 102] (pass True); V3 applied True; V4 True; V5 drift events 0; V6 sources ['tiny-demo +LTM'] (pass True); PYTHONHASHSEED True; seed threading True; missing shards 0.
- k-select UNDEFINED reasons: F: the host alternative coincides with the intact outcome (not observable)
- v-choice UNDEFINED reasons: F: the host alternative coincides with the intact outcome (not observable)
- ignition UNDEFINED reasons: F: the host alternative coincides with the intact outcome (not observable); S: no neural-derived alternative exists for this seam (by design)


## Per-seam reading (fixed sentences from HANDOFF section 9)

- k-select: host-owned. Flipping the host decision moved the reply on at least 80% of events.
- v-choice: host-owned. Flipping the host decision moved the reply on at least 80% of events.
- bg-select: host-owned. Flipping the host decision moved the reply on at least 80% of events.
- organ-c: host-owned. Flipping the host decision moved the reply on at least 80% of events. (call-graph check, not pooled)
- metacog: host-owned, and the neural-derived rule gave the same reply on these probes, so it is a replacement candidate. (call-graph check, not pooled)
- ignition: mixed; no single owner on these probes. (call-graph check, not pooled)
