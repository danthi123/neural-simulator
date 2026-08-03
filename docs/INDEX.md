# Documentation

The repository contains public guides, technical references, experiment records,
and many dated plans. They do not all have the same authority.

## Canonical Public Path

New readers should follow this short path:

1. [README](../README.md) - purpose, architecture goal, and honest overview.
2. [Quickstart](../QUICKSTART.md) - separate CPU and NVIDIA GPU setup.
3. [User Guide](../USER_GUIDE.md) - working interfaces and workflows.
4. [Current State](CURRENT-STATE.md) - capabilities, limitations, and major gaps.
5. [Roadmap](../ROADMAP.md) - current priorities and planned development.
6. [Scaffold Ledger](SCAFFOLD-LEDGER.md) - temporary shortcuts and their intended
   replacements.
7. [Contributing](../CONTRIBUTING.md) - engineering and research standards.

These documents define the current public description of the project. Keep them
short, mutually consistent, and readable without project-specific vocabulary.

## Technical Reference

These documents add depth but are not required for onboarding:

| Resource | Purpose |
|---|---|
| [Project charter](PROJECT-CHARTER.md) | Architectural commitments, host boundary, scaffold policy, and scientific standard. |
| [Technical overview](TECHNICAL-OVERVIEW.md) | Implemented substrate, evidence boundaries, intended architecture, and main technical gaps. |
| [Structural mechanism map](plans/2026-08-02-structural-mechanism-map.md) | Brain functions, candidate biological mechanisms, present implementation, and open gaps. |
| [Biology guide](biology.md) | Neuroscience background used by simulator mechanisms. |
| [Architecture diagrams](diagrams/) | Current, detailed, and target brain wiring views. |
| [Automated research checks](FAILURE_GATE_MATRIX.md) | What the repository checks before accepting experimental claims. |
| [Web dashboard reference](webapp-frontend-guide.md) | Dashboard implementation and interface notes. |

Reference documents may be narrower or more technical than the public guides.
They should not be treated as proof that a capability works.

## Evidence Record

| Resource | How to use it |
|---|---|
| [`research/findings/`](../research/findings/) | Dated experiment reports, including negative, corrected, superseded, and retracted results. Check each document's status. |
| [`research/findings/raw/`](../research/findings/raw/) | Raw measurements and provenance files. Use these for exact numbers and commands. |
| [Retraction registry](RETRACTED.md) | Known invalid or replaced claims that must not be repeated as current results. |

An experiment report records what was known when it was written. It is not a
living roadmap. Later controls or corrections may change its interpretation.

## Historical And Operational Material

- [`docs/plans/`](plans/) contains dated design proposals and old roadmaps. Treat
  them as historical or specialist reference unless a canonical public document
  explicitly points to one.
- [Project history archive](project-history-archive.md) preserves retired overview
  material.
- [`HANDOFF.md`](../HANDOFF.md), [`GAP_CLOSURE_MISSION.md`](../GAP_CLOSURE_MISSION.md),
  and [`CLAUDE.md`](../CLAUDE.md) are maintainer operations material. They are not
  public onboarding or the source of current capability claims.
- Older demo guides may remain useful for reproducing a specific experiment, but
  they do not define the present project direction.

## When Sources Disagree

1. Use raw artifacts and provenance for exact measurements.
2. Use the latest non-retracted finding for an experiment's interpretation.
3. Use `CURRENT-STATE.md` for current capability status.
4. Use `ROADMAP.md` for priorities and planned work.
5. Treat dated plans and operational notes as context, not current authority.
