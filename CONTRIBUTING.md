# Contributing

Neural Simulator is both a software project and an experimental neuroscience
project. A useful contribution must be sound code and must support an honest
claim about the integrated simulated brain.

## Start Here

1. Read the [README](README.md), [Current State](docs/CURRENT-STATE.md), and
   [Roadmap](ROADMAP.md). The current priority is integration: joining separately
   validated mechanisms into shared causal loops, including a live conversational
   loop, rather than adding another isolated capability.
2. Follow the [Quickstart](QUICKSTART.md) for CPU or NVIDIA GPU setup.
3. Check the [Scaffold Ledger](docs/SCAFFOLD-LEDGER.md) before adding a shortcut
   that may already have a planned replacement.
4. Search the record before building. `tools/before_you_build.sh "<defect>"`
   queries the local findings corpus and lists existing checks; also read the
   code, tests, and [`research/findings/`](research/findings/) for earlier
   attempts, negative results, and corrections.
5. Enable the commit hooks so the automated checks run for you:
   `git config core.hooksPath tools/githooks`.

The repository has no `pyproject.toml` or package installer. Run commands from
the repository root.

## Choose A Clear Scope

For a simulator or tooling change, state the behavior being fixed or added and
which callers may be affected.

For a brain mechanism, write down these points before implementation:

- the role the mechanism must serve in the whole brain;
- the biological process or evidence motivating it;
- its neural inputs, outputs, learning signals, and expected time scale;
- how it connects to perception, action, memory, value, affect, or language;
- what observable behavior should change when it is present;
- what control should fail when the mechanism is removed or disrupted;
- every temporary host-side calculation or hand-designed representation;
- the condition under which each temporary shortcut can be removed;
- the expected memory, runtime, and hardware cost.

This prevents a narrow test from becoming the objective. Passing an isolated
unit test is necessary, but a cognitive mechanism is not complete until it
performs its intended role in an integrated behavior.

## Architecture Expectations

The long-term system is one shared spiking brain. Specialized regions and
pathways are expected, but cognition should move toward neural activity and
synaptic state rather than host-side parsing, lookup, routing, or answer logic.

Host code is appropriate for the external world, body and sensor interfaces,
file input/output, visualization, experiment control, and measurement. A
temporary cognitive shortcut is acceptable only when it is explicit, bounded,
recorded in the scaffold ledger, and paired with a replacement plan.

Prefer existing simulator abstractions and local learning rules. Do not create a
new subsystem when the same behavior belongs in an existing region, pathway, or
shared simulation step.

## Implementation Standards

- Keep changes focused. Do not mix unrelated cleanup with a behavioral change.
- Follow the surrounding Python style; no repository-wide formatter is pinned.
- Preserve CPU and GPU behavior unless the change is explicitly backend-specific.
- Select the backend explicitly with `SIM_BACKEND=numpy` or
  `SIM_BACKEND=cupy` in tests and recorded commands.
- Keep random seeds configurable. Do not hide randomness in module-level state.
- Validate configuration and fail visibly. Silent fallback can invalidate an
  experiment while leaving the process apparently healthy.
- Add comments only where the reason or biological mapping is not evident from
  the code.
- Avoid copying production logic into a test; test observable behavior and the
  important intermediate signal independently.

## Biological And Integration Evidence

A neuroscience name is not evidence. Cite the source that motivates a mechanism
and explain which part is modeled, simplified, or omitted. Use primary papers or
standard neuroscience references when practical.

Test at two levels:

1. **Mechanism:** confirm the expected dynamics, learning rule, timing, sign,
   bounds, or state transition.
2. **Whole behavior:** confirm the mechanism changes the integrated brain in the
   way its stated role predicts.

Prefer a whole-behavior test that exercises the integrated loop the mechanism is
meant to serve, such as the live conversational loop, over a standalone harness.
In an integrated conversational evaluation most prompts fall outside the brain's
grounded knowledge, and the correct response is to abstain; a silence that avoids
fabricating an answer is a measured success, not a gap.

Use causal controls where the claim requires them. Examples include removing the
pathway, withholding the consequence, permuting labels, changing the relevant
internal drive, or replaying the same reward without the required experience.
Verify that a disruption is still active at the moment of measurement; a pathway
that regrows during the test is not a valid removal control.

Controls must distinguish the proposed explanation from easier alternatives.
A high score alone does not show which mechanism caused it.

## Reproducible Results

Every reported research result should make another contributor able to rerun it.
Record:

- the exact command and repository commit;
- Python and important dependency versions;
- selected backend, device, and relevant hardware;
- all seeds and configuration overrides;
- input data or checkpoint identity;
- wall time and important resource use;
- raw output path and provenance sidecar;
- failed runs, exclusions, and analysis decisions.

Use a fixed multi-seed set before making a general capability claim when runtime
allows; the standard set here is 42, 43, 44, 100, 101, and 102. Report per-seed
values as well as summaries, and treat a single-seed result as indicative only. A
smoke run proves that a path executes; it does not prove that the behavior is
reliable.

Store durable experiment evidence under `research/findings/raw/` and write a
dated report under `research/findings/`. Keep negative and corrected results.
Do not promote a result from an uncommitted scratch file or from memory.

For a new or changed finding, run:

```bash
python tools/finding_lint.py --include-untracked research/findings/<finding>.md
python tools/claim_check.py research/findings/<finding>.md
python tools/finding_status.py --check research/findings/<finding>.md
```

These checks help with provenance, stale claims, controls, and unsupported
measurements. They do not replace reading the artifact or checking the
experiment's measurement code.

## Wording Of Claims

Several words carry a precise, checkable meaning here. Before using one in a
findings document, commit message, or status document, confirm its code
condition in [`docs/TERMS.md`](docs/TERMS.md). In particular, "GO" names only the
positive verdict of one specific gated test at its own seeds; it is never a
faculty-wide, product, or consciousness claim.

Describe internal-state read-outs — affect, valence, confidence, familiarity,
self-model, uncertainty — as functional read-outs of measured neural signals. Do
not describe them as felt experience, sentience, emotion, or a person inside the
simulation. This distinction is a deliverable of the work, not a disclaimer added
at the end.

Keep negative, corrected, and superseded findings in the record. When a
document's central claim dies, register it in
[`docs/RETRACTED.md`](docs/RETRACTED.md); no current document may then cite that
path without the retraction marker on the same line.

## Performance Evidence

Performance is part of the architecture because the target should remain usable
on high-end consumer hardware.

- Measure a baseline and the changed version with the same seed, configuration,
  backend, and device.
- Synchronize GPU work before timing it.
- Report wall time, simulated steps per second, peak memory, and problem size
  when relevant.
- Prefer sparse, local, event-driven work and avoid unnecessary host-device
  transfers or per-step synchronization.
- Keep a correct reference path when optimizing shared numerical behavior.
- Test numerical and behavioral equivalence within an explicitly justified
  tolerance.

Biological fidelity can be expensive. That does not excuse avoidable dense work,
duplicate computation, or an unmeasured performance regression.

## Testing

Install development dependencies with:

```bash
python -m pip install -r requirements-dev.txt
```

Run the smallest relevant tests while iterating:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
SIM_BACKEND=numpy python -m pytest tests/<affected_test>.py -q
```

Run GPU-specific tests with an explicit GPU backend on a suitable machine:

```bash
SIM_BACKEND=cupy python -m pytest tests/<affected_test>.py -q
```

Broaden testing when changing `sim/bridge.py`, shared configuration, checkpoint
formats, backend behavior, or cross-region integration. State which tests were
not run and why.

For public-document changes, run:

```bash
python tools/check_docs.py
```

## Automated Checks

Commit-time checks enforce the conventions above so they are not left to memory.
Enable them once with `git config core.hooksPath tools/githooks`. The pre-commit
hook blocks a commit that, for example, breaks the document-structure rules in
[`docs/WRITING.md`](docs/WRITING.md), states a measurement in a new findings
document that appears in no artifact the document cites, binds a biological
mechanism to source anchors or configuration that no longer resolve, or leaves a
new finding's status undeclared or a capability headlined from a single seed.

The failure-class specification is
[`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md); the individual
checks live in [`tools/gates/`](tools/gates/). A deliberate, reviewable override
is `git commit --no-verify`; it is visible in the reflog and should be rare and
explained. When you notice a new failure mode these checks would not catch, add
one line to [`research/FAILURE_LOG.md`](research/FAILURE_LOG.md) naming a gate
that covers it or recording why it is not yet gateable.

## Pull Request Checklist

- The change has one clear purpose.
- Existing work and negative findings were checked first.
- The whole-brain role and biological basis are explained when relevant.
- New scaffolds are named, bounded, and assigned a removal condition.
- Load-bearing terms are used only where their `docs/TERMS.md` conditions hold.
- Unit, integration, and causal-control evidence match the claim.
- Commands, seeds, backend, artifacts, and hardware are recorded.
- Performance was measured when the changed path is performance-sensitive.
- Public status documents were updated only when committed evidence supports the
  new wording.
- No current document silently relies on a retracted or superseded result.
- Automated document and finding checks pass, or a `--no-verify` override is
  explained.

In the pull request description, separate what the code does, what the evidence
shows, what remains uncertain, and what is still temporary.
