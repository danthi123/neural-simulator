# Commit checks (the lean set, 2026-11-02)

One hook for every driver (Claude, Hermes, Pi): `tools/githooks/pre-commit` (install: `git config core.hooksPath
tools/githooks`). Never bypass it with `--no-verify`. The whole hook takes about 1-2 s on a typical commit.

**The bar for a check:** it is cheap, it reads the commit's OWN content (never idle compute, board or doc
freshness, or other system state), and it guards against costly or irreversible harm or an invalid scientific
claim. Before adding one, show it meets the bar and keeps the hook under ~5 s. Everything else is a skill step
(`before-build`, `land-result`, `ops`), not a gate.

| class | check (14 blocking) | where | blocks when |
|---|---|---|---|
| - | agent merge backstop | pre-commit (uid 99 on main) | the server driver commits a non-routine class on main (`tools/agent_merge_policy.py`) |
| - | finding status | pre-commit | a new `research/findings/*.md` has no frontmatter `status:` |
| - | claims trace to artifacts | pre-commit, `tools/claim_check.py` | a number in a new finding is in no artifact it cites |
| - | biology bindings | pre-commit, `tools/biology_check.py` | the commit touches `research/biology/`, `research/runners/` or `sim/` and a binding is broken or contradicted |
| SEC | `gates/secret_scan` | registry | a staged line looks like a credential (the repo is pushed to a public remote) |
| CC | `gates/corpus_check_required` | registry | a new non-data file under `brain/`, `docs/plans/`, `research/findings/` lands with no `tools/before_you_build.sh` run in the last 24 h; or an artifact recording > 1 h of compute has no record check |
| BP | `gates/brain_prereq` | registry | a new `brain/experiments|eval/*.py` lacks `# Capability:` (a `brain/PREREQUISITES.md` section) or `# Record:` (an existing path), or a NOT-READY capability lacks `# Expected-fail:`; or a staged PREREQUISITES.md capability cites no catalog entry, `research/biology` file, Kandel chapter or DOI |
| 9 | `gates/single_seed` | registry | a new finding headlines a result from fewer than six seeds |
| PR | `gates/prereg_before_run` | registry | a pre-registration is committed together with the run data it governs |
| PRA | `gates/prereg_amendment_order` | registry | an amendment to a pre-registration is committed together with run data |
| P | `gates/artifact_provenance` | registry | a new result JSON cannot say what produced it |
| V | `gates/verdict_preconditions` | registry | an artifact asserts a verdict without the preconditions that earned it |
| HD | `gates/host_decision_declared` | registry | a plan adds host code that decides content without declaring it as a scaffold (review must still check for a dishonest `none`: the gate cannot see it) |
| CT | `gates/workflow_cost_tiering` | registry | a committed workflow script has an agent with no model tier |
| 11 | `gates/terminology` | registry, report only | a new doc uses an overclaim word (consolidation, compositional, ...) without its condition (`docs/TERMS.md`) |

Registry rules (`tools/gates/__init__.py`): one module per check exposing `NAME`, `CLASS_ID`, `BLOCKING`, `check(paths)`
(paths = files ADDED by the commit) and `selftest()`, which must FAIL on a case the check should catch, or the
registry refuses to trust it.

Retired: 41 gates and the old G1 doc-structure step, with timings, evidence and reasons, in
`docs/archive/2026-11-02-gate-inventory.md`. Their code is in `tools/gates/_archive/` (never run), and their tests
are in `tests/_archive_gates/` (not collected).
