# Contributing

Neural Simulator is both a software project and an experimental neuroscience
project. A useful contribution must be sound code and must support an honest
claim about the integrated simulated brain.

## Start Here

1. Read the [README](README.md), [Current State](docs/CURRENT-STATE.md), and
   [Roadmap](ROADMAP.md).
2. Follow the [Quickstart](QUICKSTART.md) for CPU or NVIDIA GPU setup.
3. Check the [Scaffold Ledger](docs/SCAFFOLD-LEDGER.md) before adding a shortcut
   that may already have a planned replacement.
4. Search the code, tests, and [`research/findings/`](research/findings/) for
   earlier attempts, negative results, and corrections.

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

Use more than one seed before making a general capability claim when runtime
allows. Report per-seed values as well as summaries. A smoke run proves that a
path executes; it does not prove that the behavior is reliable.

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

## Pull Request Checklist

- The change has one clear purpose.
- Existing work and negative findings were checked first.
- The whole-brain role and biological basis are explained when relevant.
- New scaffolds are named, bounded, and assigned a removal condition.
- Unit, integration, and causal-control evidence match the claim.
- Commands, seeds, backend, artifacts, and hardware are recorded.
- Performance was measured when the changed path is performance-sensitive.
- Public status documents were updated only when committed evidence supports the
  new wording.
- No current document silently relies on a retracted or superseded result.

In the pull request description, separate what the code does, what the evidence
shows, what remains uncertain, and what is still temporary.
