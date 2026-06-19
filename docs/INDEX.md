# Documentation Index

Navigation map for all documentation in the neural-simulator project.
Find what you need fast.

---

## I just want to...

| Task | Document |
|---|---|
| Understand what this project IS | [README.md](../README.md) |
| Run the simulator | [QUICKSTART.md](../QUICKSTART.md) |
| See what works today | [CURRENT-STATE.md](CURRENT-STATE.md) |
| Understand the biology | [biology.md](biology.md) |
| Add a feature | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Configure the GUI | [USER_GUIDE.md](../USER_GUIDE.md) |
| Read AI agent guidelines | [CLAUDE.md](../CLAUDE.md) |
| Read about a specific finding | [research/findings/](../research/findings/) |
| See architecture decisions | [docs/plans/](plans/) |
| Get the full biology catalog | [references/feature-catalog.md](../references/feature-catalog.md) (catalog-build branch) |

---

## Top-level documents

### `README.md` — project overview

What the project IS, plain language. Capabilities table. How to try it.
Latest validated result. Project structure.

**Read this if:** you've never seen this project before.

### `QUICKSTART.md` — get running in 60 seconds

Install + run + 3 things to try. Cross-references for going deeper.

**Read this if:** you just want to make it work.

### `CLAUDE.md` — AI agent guidelines

Project conventions, code organization, gotchas, technical reference for
Claude/AI coding agents working on the codebase.

**Read this if:** you're an AI agent or you want the deep technical
gotchas (STDP bounds, Hebbian decay, NMDA configuration, etc.)

### `USER_GUIDE.md` — full reference manual

GUI features, CLI options, all configuration parameters explained.
Reference manual style.

**Read this if:** you want to know every flag the simulator supports.

### `CONTRIBUTING.md` — how to extend

Coding conventions, where to add what kind of code, testing requirements.

**Read this if:** you want to add a feature or submit changes.

### `CHANGELOG.md` — release-style changelog

Notable changes by date, oldest first.

**Read this if:** you want a chronological summary of changes.

---

## docs/

### `docs/CURRENT-STATE.md` — comprehensive snapshot

THE authoritative reference for what the system does today.
Validated capabilities (with numbers), architecture overview, plasticity
rules, performance, limitations, active research.

**Read this if:** you want a deep technical understanding of the
project's current capabilities.

### `docs/biology.md` — neuroscience tour, plain language

Each section: real biology + how we model it + citations to source
texts (Kandel 6e + 12 specialty references).

**Read this if:** you want to understand the neuroscience without
needing to be an expert.

### `docs/plans/` — architecture decision records

Date-stamped design docs for specific architectural decisions.
Examples:
- `2026-05-02-distributed-motor-pool-design.md`
- `2026-05-02-swr-text-io-integration-design.md`
- `2026-05-02-text-io-next-directions-biology-grounded.md`

**Read these if:** you're investigating a specific architecture choice
or trying to understand why a feature was designed a certain way.

### `docs/SCIENCE_ROADMAP.md` — long-term scientific direction

Multi-month research roadmap with experiment results table.

**Read this if:** you want to know what's next and what's been tried.

### `docs/project-history-archive.md` — old README content

The previous README.md (710 lines of cluster/phase milestone history)
preserved here for context. Don't read this unless you're researching
the project's history.

---

## research/

### `research/findings/` — chronological session findings

Date-stamped scientific findings. Latest at the top of
[`research/findings/INDEX.md`](../research/findings/INDEX.md) — start there;
its "At a glance" table is the running milestone spine.

For the single most-current state (the next concrete step the autonomous
work is on), see
[`research/findings/AUTONOMOUS_STATE.md`](../research/findings/AUTONOMOUS_STATE.md).
It is dense and internal — the README and `CURRENT-STATE.md` are the
plain-language versions.

A few landmark findings worth knowing about:
- `2026-04-27-NEW-BEST-4cheats-closed.md` — navigation: closing the
  perception shortcuts (the navigation flagship)
- `2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md` — the first
  real concept-to-concept retrieval result
- `2026-05-16-G20-failure-mechanism-FINAL-SYNTHESIS.md` — the
  distributed ~320-concept memory (continual + won't-confabulate)
- `2026-06-11-familiarity-gate-v320-GO.md` — the "learned cortex"
  arc forks (flat vs meaning-structured cortex)
- `2026-06-13-phase1-32bridge-fanout-derisk-GO.md` — scaling the
  learned cortex to ~2,048 concepts de-risked (production build in progress)
- `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` —
  a generalizing cortex that learns word meanings from a conversation
  stream, realized on the spiking network (~64 concepts)
- `2026-06-16-navigate-to-compose-then-answer.md` +
  `2026-06-16-unified-embodied-agent-stage2-GO.md` — the unified embodied
  agent: one network navigates, perceives, composes a fact, and converses
  (six seeds)
- `2026-06-19-spiking-decision-default-on-GO.md` — the navigation
  move-decision is now made in spikes by default (six seeds, honest ~16% cost)
- `2026-06-19-latency-csr-cache-GO.md` — the conversational engine sped up
  10–20× (answers identical, "won't make things up" guarantee unchanged)

**Read these if:** you want the chronological story of how the project
evolved.

### `research/runners/` — experiment scripts

Python modules implementing specific experiments (g1-g11, text_*, etc.).
Each is invocable via `python -m research.runners.<name>`.

---

## references/

### `references/feature-catalog.md`

(On `catalog-build` branch) Encyclopedia of over 300 biological mechanisms
with citations. Each entry: biological description + sim status + cluster
assignment.

**Read this if:** you want to look up a specific mechanism (e.g., "is
NPY-LTS striatal interneuron modeled?") and find its citation.

### `references/biology-buildout-roadmap.md`

(On `catalog-build` branch) Tier 0/1/2/3 implementation roadmap for
mechanisms in the catalog.

### `references/language-mechanisms-additions.md`

Language-specific neuroscience entries (G.20-G.25): Pulvermüller,
Hagoort MUC, Tomasello, Indefrey, Hickok-Poeppel, Friederici.

### `references/textbooks/`

Local PDFs of source textbooks (gitignored). README documents the
inventory:
- Kandel 6e *Principles of Neural Science* (2021) — primary
- Marr 1969, Albus 1971, Hesslow 2013 — cerebellum
- Bolam 2000, Tepper 2017/2018, PBR-160 — basal ganglia
- O'Keefe & Nadel 1978 — hippocampus
- Buzsáki 2006 — rhythms
- Sutton & Barto 2018, Schultz 1998/2016 — reward / RL

### `references/glossary.md`

Project-specific terminology.

---

## tests/

`pytest`-runnable test suite. Most relevant for new contributors:

- `test_distributed_motor_pop.py` — distributed motor architecture
- `test_bridge_text_io.py` — text I/O bridge APIs
- `test_cluster_d.py`, `test_cluster_f.py` — hippocampus + cerebellum
- `test_d1_d2_asymmetry.py` — striatal microcircuit
- `test_e_inh_override.py` — per-region E_inh
- `test_neuromodulators.py`, `test_regions.py` — framework infrastructure

Run all: `pytest tests/ -v`

---

## webapp/

Optional FastAPI dashboard. Browse runs, launch experiments, view 3D.

Start with: `uvicorn webapp.server:app --host 127.0.0.1 --port 8765`
Browse: `http://127.0.0.1:8765`

---

## When in doubt

Start at [README.md](../README.md). It points you everywhere else.
