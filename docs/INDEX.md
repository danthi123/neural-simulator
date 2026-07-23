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
| See where the project is headed | [ROADMAP.md](../ROADMAP.md) |
| See the architecture at a glance | [diagrams/brain_architecture_current.md](diagrams/brain_architecture_current.md) |
| Understand the biology | [biology.md](biology.md) |
| Add a feature | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Configure the GUI | [USER_GUIDE.md](../USER_GUIDE.md) |
| Read AI agent guidelines | [CLAUDE.md](../CLAUDE.md) |
| Read about a specific finding | [research/findings/](../research/findings/) |
| See architecture decisions | [docs/plans/](plans/) |
| Get the full biology catalog | [references/glossary.md](../references/glossary.md) (the full catalog lives in a separate worktree) |

---

## Top-level documents

### `README.md` — project overview

What the project IS, plain language. Capabilities table. How to try it.
Latest validated result. Project structure.

**Read this if:** you've never seen this project before.

### `QUICKSTART.md` — get running in 60 seconds

Install + run + 3 things to try. Cross-references for going deeper.

**Read this if:** you just want to make it work.

### `ROADMAP.md` — accomplished / in progress / what's left

The kept-current source of truth for what the project has achieved, what it is
working on now, and the open research frontiers on the path to the goal (a
single spiking brain that navigates and holds a grounded conversation). Uses
plain-language status badges (done / partial / boundary / scaffold / open).

**Read this if:** you want the honest current status and near-term direction.

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

### `docs/diagrams/` — architecture flowcharts

The live, kept-current flowcharts of the whole simulated brain, as
GitHub-native Mermaid markdown (they render directly on GitHub):
- `brain_architecture_current.md` — newcomer overview, plain language.
- `brain_architecture_detailed.md` — exhaustive per-region / per-pathway map.

Also here: three hand-authored hero images (`brain_master.svg`,
`brain_navigation.svg`, `brain_conversational.svg`, each with a rendered
`.png`) — an archived June 2026 snapshot. See `docs/diagrams/README.md`.

**Read these if:** you want to see how the regions connect at a glance.

### `docs/biology.md` — neuroscience tour, plain language

Each section: real biology + how we model it + citations to source
texts (Kandel 6e + specialty references).

**Read this if:** you want to understand the neuroscience without
needing to be an expert.

### `docs/plans/` — architecture decision records

Date-stamped design docs for specific architectural decisions, paired
with the findings that resulted. Examples span the project's history:
- `2026-05-02-distributed-motor-pool-design.md`
- `2026-07-13-np-recurrent-language-derisk-spec.md`
- `2026-07-15-months-scale-plan-to-one-brain-and-small-llm-conversation.md`

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
- `2026-04-27-NEW-BEST-4cheats-closed.md` — an early navigation milestone
  (note: some "closed all shortcuts / no heuristic" claims in older navigation
  notes were later audited and corrected — see `ROADMAP.md` and the
  navigation-audit findings for the honest current status)
- `2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md` — an early
  concept-to-concept retrieval result
- `2026-05-16-G20-failure-mechanism-FINAL-SYNTHESIS.md` — the
  distributed few-hundred-concept memory (learns continually and declines
  to answer rather than fabricate)
- `2026-06-11-familiarity-gate-v320-GO.md` — the word-meaning-layer work
  forks (a semantically flat vs. a meaning-structured version)
- `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` —
  the brain learns word meanings by "listening" to a text stream, realized
  on the spiking network (validated at a small vocabulary)
- `2026-06-16-navigate-to-compose-then-answer.md` +
  `2026-06-16-unified-embodied-agent-stage2-GO.md` — one network navigates,
  perceives an object, forms a fact about it, and converses (multiple seeds)
- `2026-06-19-spiking-decision-default-on-GO.md` — the navigation
  move-decision is now made in spikes by default (multiple seeds; an honest
  cost vs. the old hand-coded pick-the-max step is reported)
- `2026-06-19-latency-csr-cache-GO.md` — the conversational engine sped up
  substantially (answers unchanged, the no-fabrication safeguard intact)

**Read these if:** you want the chronological story of how the project
evolved. For the plain-language current status, prefer `ROADMAP.md` and
`docs/CURRENT-STATE.md`.

### `research/runners/` — experiment scripts

Roughly 1,250 headless Python scripts, each implementing one experiment
(navigation, conversation/chat, memory consolidation, the word-meaning-learning
experiments, the language generator, and many more). Each runs via
`python -m research.runners.<name>`.

---

## references/

### `references/feature-catalog.md` — *(separate worktree, not in this checkout)*

Encyclopedia of over 300 biological mechanisms with citations. Each entry:
biological description + how/whether the sim models it. It is maintained in a
separate `sim-catalog` worktree, so it will not appear in a normal checkout of
this repository.

**Look here if:** you want to check whether a specific mechanism (e.g., "is
the NPY-LTS striatal interneuron modeled?") is implemented, and find its citation.

### `references/biology-buildout-roadmap.md` — *(separate worktree, not in this checkout)*

Tiered implementation roadmap for the mechanisms in the catalog above. Same
`sim-catalog` worktree; not in a normal checkout.

### `references/language-mechanisms-additions.md`

Language-specific neuroscience entries (Pulvermüller, Hagoort, Tomasello,
Indefrey, Hickok–Poeppel, Friederici).

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

`pytest`-runnable test suite (472 files, most CPU-only). A few worth knowing:

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
