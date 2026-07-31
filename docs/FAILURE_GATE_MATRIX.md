# The failure→gate matrix — every known failure class, and what mechanically prevents it

**This file is the SPECIFICATION, not a retrospective.** It exists because the failure record was already written
down — CLAUDE.md documents the plasticity bound trap for four rules, the 2026-07-28 session wrote up nine
retractions, `verify-go` states the rules — and it was read as *history to respect* rather than as *requirements
to implement*. The bound trap then hit a fifth rule.

Classes are from the 2026-07-31 taxonomy pass (21 agents, 11 classes, **zero rejected** by adversarial verifiers),
ranked by cost × recurrence. A row is only "GATED" if something **blocks** on an unavoidable path. Advisory tools
are marked ADVISORY and count as ungated — the measured reason: **1330 runners, `tools/lab.py` imported by 2,
`tools/experiment.py` by 0.**

| # | Failure class | n | Gate | Where it blocks | State |
|---|---|---|---|---|---|
| 1 | manipulation-never-engaged (lever/arm/lesion inert) | 10 | `Experiment._assert_one_variable`, `lab.lever` | import-time only | ⚠️ ADVISORY |
| 2 | plasticity-bound-trap | 7 | `lab.bound_check` + `biology_check` `constraints_config` | pre-commit GATE 3 | ✅ GATED |
| 3 | check-that-cannot-fail / was bypassed | 9 | every gate ships a **failing-direction test** | `tests/test_experiment_harness.py` | 🟡 PARTIAL |
| 4 | comparison-with-no-discriminating-power | 7 | `Experiment.validate_instrument` (power **and** FPR) | import-time only | ⚠️ ADVISORY |
| 5 | record-not-read-before-building | 6 | `pool_queue.sh --checked` + dispatcher refusal | execution path | ✅ GATED |
| 6 | wrong-quantity-comparison | 7 | — | — | ⛔ UNGATED |
| 7 | liveness-mistaken-for-progress | 4 | heartbeat + throughput check at launch | reporting only | ⚠️ ADVISORY |
| 8 | stale-pointer / unmaintained registry | 5 | `check_docs` W1/W2; `status:` frontmatter | pre-commit GATES 1, 4 | 🟡 PARTIAL |
| 9 | single-seed-headline | 4 | `Experiment(n_seeds=)` refusal below 6 | not implemented | ⛔ UNGATED |
| 10 | single-axis-sweep-reported-as-absolute | 3 | — | — | ⛔ UNGATED |
| 11 | terminology-overclaim | 3 | `docs/TERMS.md` | prose only | ⛔ UNGATED |
| + | claim-not-traced-to-artifact | — | `claim_check.py` | pre-commit GATE 2 | ✅ GATED |
| + | biology-not-bound-to-code | — | `biology_check.py` | pre-commit GATE 3 | ✅ GATED |

**Score: 4 gated, 2 partial, 3 advisory, 4 ungated.**

## The rule this file enforces on itself

**A class is not closed by a tool existing.** It is closed when something *blocks* on a path that cannot be
avoided, and when that blocking has been demonstrated in its **failing direction** — the check must be shown to
fail on a case it should catch, not merely to pass on a good one. Class 3 exists precisely because checks here
have twice been made mandatory and then failed *inside themselves* (`;` instead of `&&`; a pipe eating the exit
status; a relevance count that made a gate unfailable).

## Why the advisory ones are the priority

Classes 1 and 4 are the two most expensive (10 and 7 incidents) and both live in `tools/experiment.py`, which has
**zero importers**. A fail-closed harness nobody imports prevents nothing. Closing them means making the harness
unavoidable for anything that produces a verdict — not writing more of it.

## Known-ungated, stated plainly

- **6 · wrong-quantity-comparison** — two correct numbers of different quantities. No mechanical check exists;
  `circ` vs `circ_dW` cost a retracted 6-seed GO. Candidate: require metrics to carry a declared unit/quantity tag.
- **9 · single-seed-headline** — trivially gateable (`n_seeds < 6` refuses without a waiver); simply not built yet.
- **10 · single-axis-sweep-reported-as-absolute** — a sweep over interacting parameters gives a conditional
  answer; the density optimum was not merely narrow but *inverted*. Candidate: require sweeps to record which
  other axes were held, and at what values.
- **11 · terminology-overclaim** — `docs/TERMS.md` defines code conditions for ~10 loaded words in prose only.
  Candidate: check findings for those words against their stated conditions.

## What no gate can fix

Prioritisation and taste — knowing *which* question is worth asking. The gates make wrong answers expensive to
publish; they cannot make the right question salient. That remains a judgement call, and the place where owner
steer is load-bearing.
