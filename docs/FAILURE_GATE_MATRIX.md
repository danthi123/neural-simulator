# The failure→gate matrix — every known failure class, and what mechanically prevents it

**This file is the SPECIFICATION, not a retrospective.** It exists because the failure record was already written
down — CLAUDE.md documents the plasticity bound trap for four rules, the 2026-07-28 session wrote up nine
retractions, `verify-go` states the rules — and it was read as *history to respect* rather than as *requirements
to implement*. The bound trap then hit a fifth rule.

Classes are from the 2026-07-31 taxonomy pass (21 agents, 11 classes, **zero rejected** by adversarial verifiers),
ranked by cost × recurrence. A row is only "GATED" if something **blocks** on an unavoidable path. Advisory tools
are marked ADVISORY and count as ungated — the measured reason: **1330 runners, `tools/lab.py` imported by 2,
`tools/experiment.py` by 0.**

| # | Failure class | n | Gate (module) | Where it blocks | State |
|---|---|---|---|---|---|
| 1 | manipulation-never-engaged | 10 | `gates/lever_efficacy` | registry, reporting | 🟡 REPORTS (40 live hits) |
| 2 | plasticity-bound-trap | 7 | `lab.bound_check` + `biology_check` | pre-commit G3 | ✅ BLOCKS |
| 3 | check-that-cannot-fail | 9 | registry refuses a gate whose selftest passes vacuously | `gates/__init__` | ✅ STRUCTURAL |
| 4 | no-discriminating-power | 7 | `gates/discriminating_power` | registry, reporting | 🟡 REPORTS |
| 5 | record-not-read-before-building | 6 | `pool_queue --checked` + dispatcher refusal | execution path | ✅ BLOCKS |
| 6 | wrong-quantity-comparison | 7 | `gates/quantity_mismatch` | registry | ✅ BLOCKS |
| 7 | liveness-mistaken-for-progress | 4 | `gates/throughput` + dispatcher exit-status + heartbeat | registry + heartbeat | 🟡 REPORTS |
| 8 | stale-pointer / unmaintained registry | 5 | `gates/stale_pointer`, `check_docs` W1/W2, `dead_citations.sh` | pre-commit G1 + registry | 🟡 PARTIAL (sees 1% until statuses declared) |
| 9 | single-seed-headline | 4 | `gates/single_seed` | registry | ✅ BLOCKS |
| 10 | single-axis-sweep-as-absolute | 3 | `gates/conditional_sweep` | registry, reporting | 🟡 REPORTS |
| 11 | terminology-overclaim | 3 | `gates/terminology` | registry, reporting | 🟡 REPORTS |
| P | artifact-provenance | — | `runners/__init__` capture + `gates/artifact_provenance` | execution path + registry | ✅ BLOCKS |
| D | doc-type / placement | — | `gates/doc_type` | registry | ✅ BLOCKS |
| C | claim-not-traced-to-artifact | — | `claim_check.py` | pre-commit G2 | ✅ BLOCKS |
| B | biology-not-bound-to-code | — | `biology_check.py` | pre-commit G3 | ✅ BLOCKS |
| M | mechanism-status conflict | — | `biology_check.check_mechanism_status` | pre-commit G3 | ✅ BLOCKS |
| S | finding-status undeclared | — | pre-commit GATE 4 | pre-commit G4 | ✅ BLOCKS (new findings) |
| X | invalid queued command | — | `pool_queue` argparse validation | execution path | ✅ BLOCKS |
| Y | job died silently | — | dispatcher exit-status log + heartbeat | execution path | ✅ REPORTS |
| L | CPU lanes starved while work continues elsewhere | — | `gates/lane_starvation` | registry | ✅ BLOCKS |
| COV | a NOTICED failure never became a gate | — | `gates/coverage` + `research/FAILURE_LOG.md` | registry | ✅ BLOCKS |

**Score: 13 BLOCKING · 1 structural · 6 reporting · 0 ungated.**

## The loop that keeps this file honest

Every gate here was added because a failure was NOTICED and then acted on — which made closure depend on memory,
the exact dependency the system exists to remove. `gates/coverage` closes that: a newly-noticed failure gets ONE
LINE in [`research/FAILURE_LOG.md`](../research/FAILURE_LOG.md), and the gate BLOCKS until that line names a gate
or declares `NOT-GATEABLE: <reason>`. It also checks the reverse — a module absent from this matrix, or a matrix
row naming a module that does not exist, is spec/code drift and fails.

**What it cannot do:** notice. If a failure is never written down, nothing fires. It closes
*noticed-but-forgotten*, not *never-noticed* — and that limit is stated rather than papered over.

Every class from the 2026-07-31 taxonomy now has a module. Six REPORT rather than block, each
because it declared limits it cannot check reliably at commit time — an honest reporting gate
beats a false-positive generator that gets disabled. Nothing is left unaddressed.

## The rule this file enforces on itself

**A class is not closed by a tool existing.** It is closed when something *blocks* on a path that cannot be
avoided, and when that blocking has been demonstrated in its **failing direction** — the check must be shown to
fail on a case it should catch, not merely to pass on a good one. Class 3 exists precisely because checks here
have twice been made mandatory and then failed *inside themselves* (`;` instead of `&&`; a pipe eating the exit
status; a relevance count that made a gate unfailable).

## What the gates found on their FIRST corpus run

Not hypothetical. On the registry's first pass over the live repo:

- **`lever-efficacy`: 40 identical-arm pairs across 11 banked artifacts.** In
  `_emerge6_recurrent_microcircuit_seq.json`, THREE arms agree on all three metrics
  (`apical_feedback_lesion` = `no_teaching_null` = `untrained`, `onestep = -0.0698238953499733`). Two distinct
  manipulations cannot agree to sixteen digits. Finding:
  [`2026-07-31-audit-lever-efficacy-...`](../research/findings/2026-07-31-audit-lever-efficacy-40-identical-arm-pairs-in-banked-artifacts.md).
- **`single-seed`** blocked a finding written the same hour for headlining SOLVED on 3 declared seeds.
- **`artifact-provenance`** blocked the audit's own output artifact, generated from an ad-hoc heredoc.
- **`claim_check`** blocked a commit for a citation path that did not resolve, and separately found a
  **wrong number in an already-published table** (density-1.0 null 0.4977, the document said 0.5894).
- **`stale-pointer`** reported that it can currently check **1% of 407 citations**, because only 4 declare a
  status — it states its own blindness instead of passing quietly.

**Eight blocks on the author, in one session.** That is the intended behaviour, not a defect rate.

## Calibration is measured, not asserted

A gate that cries wolf gets disabled, which is worse than no gate. So each was measured against the real corpus:
`single-seed` fires on 1 of 1841 findings, and its verifier simulated adding frontmatter to every legacy file to
find that **260 would fire** — which is why the frontmatter scope limit is load-bearing rather than arbitrary.
`terminology` measures 2.3% corpus-wide against its own claimed ~3%, i.e. it does not flatter itself.

## What no gate can fix

Prioritisation and taste — knowing *which* question is worth asking. The gates make wrong answers expensive to
publish; they cannot make the right question salient. That remains a judgement call, and the place where owner
steer is load-bearing.
