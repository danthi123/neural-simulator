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
| AP | pending work SERIALISED while dispatchable agents sit unused, including completed agents whose lanes remain falsely `running` | — | `gates/agent_parallelism` | registry; tracked JSON workboard | ✅ BLOCKS |
| L | CPU lanes starved while work continues elsewhere | — | `gates/lane_starvation` | registry | ✅ BLOCKS |
| IR | a GO reports a SIZE without a SOURCE (no decomposition) | — | `gates/instrument_required` | registry | ✅ BLOCKS |
| OP | a run misses an operating-point target recorded in its own artifact | — | `gates/operating_point` | registry | ✅ BLOCKS |
| COV | a NOTICED failure never became a gate | — | `gates/coverage` + `research/FAILURE_LOG.md` | registry | ✅ BLOCKS |
| BC | every arm of an A/B lands BELOW chance, reported as a NO-GO | 1 | `gates/below_chance` | registry | ✅ BLOCKS |
| R | the record's own RETRIEVAL layer cannot see part of the record (a flat findings glob; or `**` written without `recursive=True`, which is a silent no-op) | 42 findings | `gates/retrieval_completeness` | registry | ✅ BLOCKS |
| AT | a treatment/control pair is MEASURED but the difference is never ATTRIBUTED (`tools/lab` imported by 2 of 1330 runners) | 1 (gap#5: 97% was the clamp) | `gates/attribution_required` | registry | ✅ BLOCKS |
| V | an artifact asserts a VERDICT without carrying what earned it — the run-time relationships no file-scanner can see (precondition changed, control cannot reach its mechanism, ceiling below chance, knob inert, validity computed then ignored) | 5 in one day | `tools/verdict.Verdict` (runtime) + `gates/verdict_preconditions` (artifact) | registry | ✅ BLOCKS |
| SV | a finding STATES a named quantity that disagrees with the artifact it cites — `claim_check` passes it because the number exists SOMEWHERE, and existence is not agreement | 1 (chance 0.200 derived vs 0.167 reported) | `gates/stated_value_mismatch` | registry | ✅ BLOCKS |
| DC | an artifact cannot say what DEVICE it ran on, or burned hours without ever projecting its cost | 2 (a 30-min GPU test that ran on CPU; a 9h run heading for ~23h/cell) | `gates/device_and_cost` + runtime `lab.assert_backend` / `lab.project_cost` | registry | ✅ BLOCKS |
| CC | an EXPENSIVE run whose question was never checked against the record — the first gate here that looks for a REDUNDANT claim rather than a wrong one | 1 (9h x 8 cells re-deriving a 6-seed result from 3 weeks earlier) | `gates/corpus_check_required` + `before_you_build.sh` recording + door stamp | registry | ✅ BLOCKS |
| CM | a CLOSURE claim that names no mechanism, so nothing can adjudicate it against other live claims | 2 contradictory gap#4 findings live for 17 days | `gates/closure_names_mechanism` (forces the entry `biology_check.check_mechanism_status` needs) | registry | ✅ BLOCKS |
| BV | a FUNDAMENTAL-LIMIT / "different-paradigm" verdict banked from a MEMORY model of the mechanisms, without reading the adjacent findings OR the external literature that refutes it (the symmetric negative half of CM; closes CC's cheap-run hole by triggering on the CLAIM type, not compute cost) | 1 (b7549514 "fundamental transport-free ceiling" overturned within the hour by WF-Act-PC arxiv 2607.13380 + our own prior findings) | `gates/boundary_verdict_external_check` (+ `tools/record_external_search.sh` writes the previously-dead `.last_external_search` marker) | registry | ✅ BLOCKS |
| RM | re-proposing a mechanism the record ALREADY refuted, as a next/remaining surpass, from memory — the UPBEAT + CHEAP seam that BV (loud-boundary titles only) and CC (>1h runs only) both miss; a recurring reflex (dendrites) the owner had caught before | 1 (2026-08-02 "remaining surpass = two-compartment dendritic credit" for gap#4 written into a finding + board + both roadmaps, while dendritic/BDSP credit is tested-and-NEGATIVE per 2026-05-17 / 2026-07-12 / 2026-07-22-real-issue-NOT-dendrites / 2026-08-01) | `gates/refuted_mechanism_reproposal` (refuted register: dendritic/two-compartment/BDSP/burstprop; must cite the refuting finding stem or an already-tested token near a proposal phrase) | registry | ✅ BLOCKS |
| DR | DEEP RESEARCH (local record + EXTERNAL literature) not done while HAMMERING A WALL — lever after lever, cheap and PARTIAL-framed, so CC (>1h only) and BV (loud-boundary titles only) both miss it; the recurring owner-flagged miss, and the most expensive (re-derives an existing solution + circles a proven external mechanism) | 1 (2026-08-09: 5 forgetting levers before any research; the record already had CLS + Phase-1.4 103%, "replay caps ~55%" already characterised, and the external SOTA = PS-SNN pattern-separation / EWC / van de Ven) | `gates/deep_research_at_wall` (a lane with >=3 findings in 3 days requires a fresh `.external_searches.jsonl` entry with a NON-EMPTY source; `tools/deep_research.sh` does both halves, `record_external_search.sh` now REQUIRES a real source) | registry | ✅ BLOCKS |
| KR | a knob that changes the SUBSTRATE but cannot be set from the command line — so a prescribed fix is unrunnable and the config is unrecoverable | 3 in one runner (16 of 1333 corpus-wide) | `gates/knob_reachable` | registry | ✅ BLOCKS |
| SF | the forward-looking summary docs (roadmap §7/§8 + `ROADMAP.md`) drift while findings pile up — keeping them current was a REMEMBERED skill-run, not a check | ~11 findings with the roadmap untouched a whole session (gap#4's on-bridge wall stayed "wall" after being surpassed) | `gates/summary_doc_freshness` (staleness BUDGET: THRESHOLD findings, then the next finding-commit blocks until a forward-doc is synced) | registry | ✅ BLOCKS |
| CVV | a `status: live` finding claims a GO/surpass/closure in its TITLE while the artifact it cites printed `SIGNAL: false` / a HONEST-NEGATIVE verdict — silent-failure rule #1, which `verdict_preconditions` (artifact-internal) and `claim_check` (numbers only) both miss | 1 (2026-08-01 gap#4 "closure" banked as a session headline + roadmap surpass banner while its e-prop artifacts each read SIGNAL=false, deep_credit_share≈0.005) | `gates/claim_verdict_consistency` (title-scoped, negation-guarded; escape = status superseded/contributing/retracted or a ⛔/RETRACT title; corpus-calibrated to 0 false positives) | registry | ✅ BLOCKS |

**Score: 26 BLOCKING · 1 structural · 7 reporting · 0 ungated — 34 rows.** (The previous line read
`14 · 1 · 6`, which sums to 21 against 22 rows: row **Y**, green but non-blocking, was in no bucket. Corrected
here rather than carried forward — an arithmetic drift in the score of the anti-drift spec is the joke this
file cannot afford. Reporting = 6 🟡 rows + Y.)

> **Adding a row: RE-DERIVE this line, do not increment it.** The buckets must sum to the row count, and the
> `coverage` gate will tell you if a module is missing a row but not if the arithmetic is wrong. Count it:
> `grep -c '^| ' docs/FAILURE_GATE_MATRIX.md` minus the header row, against ✅ BLOCKS + ✅ STRUCTURAL +
> (🟡 rows + ✅ REPORTS).

## The loop that keeps this file honest

Every gate here was added because a failure was NOTICED and then acted on — which made closure depend on memory,
the exact dependency the system exists to remove. `gates/coverage` closes that: a newly-noticed failure gets ONE
LINE in [`research/FAILURE_LOG.md`](../research/FAILURE_LOG.md), and the gate BLOCKS until that line names a gate
or declares `NOT-GATEABLE: <reason>`. It also checks the reverse — a module absent from this matrix, or a matrix
row naming a module that does not exist, is spec/code drift and fails.

**What it cannot do:** notice. If a failure is never written down, nothing fires. It closes
*noticed-but-forgotten*, not *never-noticed* — and that limit is stated rather than papered over.

**The loop has now closed one row for the first time.** Class **BC** was logged as
`NOT-GATEABLE yet: the guard exists in ONE runner, which does not cover the class ... widening the coverage
recogniser to accept a single runner would have made this pass while covering nothing.` It is a gate as of
2026-07-31. `NOT-GATEABLE` is therefore a *state*, not a verdict — the log row is the queue, and
[`research/FAILURE_LOG.md`](../research/FAILURE_LOG.md) still carries the superseded `NOT-GATEABLE` text for
this class and needs updating to name `gates/below_chance`.

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
- **`below-chance`: 34 files / 35 sites — 16 of 7151 artifacts (0.22%) in `research/findings/raw/`, plus 18 of
  234 in `raw/`.** Three
  banked `"verdict": "NEGATIVE"` artifacts have **no interpretable result underneath the verdict**:
  `_ml_stacked_s42.json` `/per_seed[0]/cooc` scores acc 0.13 against deranged 0.14 with chance 0.25 — the
  derangement CONTROL beat the treatment, both under chance; `_lge_gpu_seed42.json` records
  `NEGATIVE_no_structure` where graded 0.2375 / orthogonal 0.11875 / permuted 0.24375 all sit under 0.25; and
  `_npwall_spiking_s42.json` asserts `"shuffle_collapses": true` and `"GO": false` where np 0.460, shuffle
  0.446 and hidden_frozen 0.468 are all under chance 0.549. Each is UNDEFINED, not a negative. Reproduce:
  `.venv/bin/python -m tools.gates.below_chance`.

**Eight blocks on the author, in one session.** That is the intended behaviour, not a defect rate.

## Calibration is measured, not asserted

A gate that cries wolf gets disabled, which is worse than no gate. So each was measured against the real corpus:
`single-seed` fires on 1 of 1841 findings, and its verifier simulated adding frontmatter to every legacy file to
find that **260 would fire** — which is why the frontmatter scope limit is load-bearing rather than arbitrary.
`terminology` measures 2.3% corpus-wide against its own claimed ~3%, i.e. it does not flatter itself.
`below-chance` publishes its REACH before its rate — it can read a floor in only **763 of 7151** artifacts
(10.7%), so 89% of the corpus is beyond it — and discloses that **13 of its 18 `raw/` hits** reach the required
two arms only via a `coupling` sub-dict that holds coefficients, not scores; no structural rule separates those
from the true positives, because the incident this gate exists for also keeps all its arms in one sub-dict.

**Selftests are mutation-tested, not trusted.** `below_chance` was deliberately broken ten ways (unfailable
check; substring deny list; integer arms; bare-`k` derivation; controls-only arms; empty-staged-list fallthrough;
`<=` for `<`; pre-filter desynchronised from the recogniser; over-wide acknowledgement escape; over-wide verdict
escalation) and its `selftest()` was required to fail on each. **It caught 6 of 8, then 9 of 10** — one miss was
a bare-`k` rule masked by the read pre-filter rather than tested (a seam defect inside a single file), the other
an assertion whose control string could not match its own regex. Both were found by attacking the check, not the
code. This is the concrete procedure behind class 3 and it is cheap: substitute one string into the module's
source, exec it, assert `selftest()` returns non-empty. A rule with no mutation that breaks it is untested.

## What no gate can fix

Prioritisation and taste — knowing *which* question is worth asking. The gates make wrong answers expensive to
publish; they cannot make the right question salient. That remains a judgement call, and the place where owner
steer is load-bearing.
