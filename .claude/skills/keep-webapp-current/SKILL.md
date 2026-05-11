---
name: keep-webapp-current
description: Use when changes to research/runners/g11_bg_runner.py, sim/, or experiment/ may have introduced new fields, flags, or capabilities that the webapp dashboard doesn't reflect yet. Verifies that webapp/server.py endpoints, webapp/static/app.js client logic, webapp/static/world.js, and the launcher's preset list stay in sync with the underlying simulator. Reports drift; updates safe items; flags semantic changes for human review.
---

# Keep Webapp Current

Sister skill to `sync-documentation` but specialized for the FastAPI + vanilla-JS dashboard at `webapp/`. The dashboard reads from the runner's output JSON and exposes the runner's CLI flags via launcher presets — both surfaces drift the moment a new flag is added or a new field is recorded.

**Announce at start:** "Using keep-webapp-current to verify dashboard against simulator."

## When this skill should fire

Manual invocation, OR auto-suggested by the `check_doc_drift` hook (which currently nudges `sync-documentation`; we may want a sibling hook for webapp drift). Trigger conditions:

- Edits to `research/runners/g11_bg_runner.py` (esp. `argparse` block, `phase_stats` writes, `*_log` outputs)
- Edits to `sim/regions.py` or `sim/bridge.py` that add new gate/region/state types
- New `*_runner.py` files
- New finding doc (`research/findings/*.md`) — may want a launcher preset for the recipe

## Eight checks

### A. Server endpoint field names match runner output JSON

The server's `/api/runs` summary computes `sum_finalQ` from `phase_stats[*].final_quarter_mean_distance`. If the runner renames or moves this field, the dashboard silently shows `sum —` for everything (this exact bug landed 2026-04-28).

```bash
# Check the runner records what the server reads:
grep -E "final_quarter_mean_distance|finalQ" research/runners/g11_bg_runner.py | head
grep -E "final_quarter_mean_distance|finalQ" webapp/server.py | head
```

The two grep results should reference the same field names.

### B. Runner CLI flags exist as launcher presets (or extras)

```bash
# Get runner's flags
grep "add_argument" research/runners/g11_bg_runner.py | sed -E 's/.*add_argument..([^"]+)".*/\1/' | sort > /tmp/runner_flags.txt

# Get webapp's flags (in PRESETS dict + frontend defaults)
grep -E '"--[a-zA-Z-]+"' webapp/server.py webapp/static/index.html | sed -E 's/.*("--[a-zA-Z-]+").*/\1/g' | sort -u > /tmp/webapp_flags.txt

# Diff
diff /tmp/runner_flags.txt /tmp/webapp_flags.txt | head -40
```

Flags in runner that aren't in webapp are fine (webapp doesn't need to expose all flags), but flags **referenced in webapp that aren't in the runner** are bugs — clicking the preset would fail.

### C. Filename pattern in `_detect_experiment` matches actual run files

```bash
# What patterns exist?
ls research/findings/raw/g11_bg/*.json | sed -E 's|.*/(g11_seed[0-9]+_?[^/]*)\.json|\1|' | sort -u | head -20

# What does _detect_experiment expect?
grep "_EXP_SUFFIX_RE" webapp/server.py
```

If a new naming pattern appears (e.g. `g11_run42_v3lateral.json` instead of `g11_seed42_v3lateral.json`), the experiments tab will categorize it as `"(other)"`.

### D. Categorize-experiment heuristics in ui.js cover new experiment names

`webapp/static/ui.js:categorizeExperiment` has hard-coded substring rules
("v3lateral" → cheat #5, "perception" → perception arc, etc.). When new
experiments appear (e.g. "v4dev" for the developmental phase), they
fall through to "other".

```bash
# Find experiment names with no category-pill match
node -e "
const cats = require('./webapp/static/ui.js'); // pseudo — not actually a node module
" || true
# Or browser eval: list experiments, find rows with category 'other'.
```

When you spot a new experiment falling through to "other", add a clause
to `categorizeExperiment`.

### E. World-tab playback uses fields that the runner records

`webapp/static/world.js:loadRun` reads:
- `data.grid_size`, `data.trajectory`, `data.goal_log`, `data.action_log`,
  `data.reward_log`, `data.phase_stats`

Verify the runner still writes these:

```bash
grep -E "trajectory|goal_log|action_log|reward_log|grid_size" research/runners/g11_bg_runner.py | grep -v "^#" | head
```

### F. Live-mode regex matches actual stdout format

`webapp/server.py:_PROGRESS_RE` matches:
```
[g11 seed=42] step 800/1800  pos=(6,1)  goal=(1,6)  recent_dist=7.58 ...
```

If the runner's print format changes, live mode silently shows nothing.
Test with a sample line:

```bash
python -c "
from webapp.server import _try_parse_progress
print(_try_parse_progress(
    '[g11 seed=42] step 800/1800  pos=(6,1)  goal=(1,6)  recent_dist=7.58',
    0.0
))
"
# Should output a ProgressEvent, not None.
```

### G. Webapp tests still pass

```bash
python -m pytest tests/test_webapp_server.py -q
```

If runner-side changes break the webapp, tests are the first line of defense.

### H. Latest finding doc has a corresponding run group in Experiments

```bash
ls -t research/findings/2026-*.md | head -3
# For each, eyeball whether the experiment it describes is visible at /api/experiments
```

If a finding describes "v4dev" but the experiments tab doesn't show it
(because no runs exist with that suffix), either the finding is premature
or the runs were named with a different suffix.

### I. capability_status.json reflects the latest validated capability

The Home tab "Project capability status" panel is backed by
`webapp/capability_status.json` (added 2026-05-09). It is a manual
source-of-truth — when significant milestones land (multi-seed GO,
new validated tier, capacity rule extension) the JSON should be
updated so the widget doesn't lie.

```bash
# Check the as_of date vs the most recent CONFIRMED finding
jq -r '.as_of' webapp/capability_status.json
ls -t research/findings/*CONFIRMED*.md research/findings/*BREAKTHROUGH*.md 2>/dev/null | head -3
```

If a finding doc with `CONFIRMED` or `BREAKTHROUGH` in the name is
newer than `capability_status.json`'s `as_of`, the JSON is stale.

```bash
# Verify the JSON parses + matches the test schema
python -m pytest tests/test_webapp_server.py -k capability_status -q
```

If the schema test fails, the JSON has drifted from the documented
shape — fix it (don't change the test).

**What to update if stale:**
- `headline.tier` / `result` / `metrics` / `wall_clock` / `finding_doc`
  / `summary` to the new validated capability
- Append a new pillar to `pillars[]` (or update the most-recent one
  if it's a refinement)
- Extend `capacity_rule.rows[]` if a new vocab tier was validated
- Update `phase_status.active` / `next` / `after_next` as the master
  plan progresses
- Bump `as_of` to today

## What to fix automatically

- Field-name drift (Check A): patch `webapp/server.py` to read the new
  field name, with the old name as fallback (don't remove old support).
- Add new experiment to `categorizeExperiment` (Check D): add a clause
  with a sensible category and color.
- Add new flag to launcher's `extra_args` examples (Check B): only as
  documentation, not as a default preset.

## What to flag for human review

- New preset suggestions (Check H): recipe text from a finding doc may
  warrant a new entry in the `PRESETS` dict in `webapp/server.py`. Show
  the user the recipe and ask whether to add as a preset.
- Print-format changes (Check F): if regex breaks, ask whether to
  generalize the regex or update the runner's print format.
- New simulator capability (e.g. neuromodulator subsystem when first
  added): may warrant a new tab. Don't auto-add tabs; flag for human.

## What this skill must NOT do

- Add new tabs without confirmation (UX surface decisions are human-only)
- Change the dashboard's color palette or theming
- Remove existing endpoints or backwards-compat fields
- Touch the bridge or runner code from this skill (out of scope)

## Output format

```
## Webapp drift report — <date>

### ✓ Verified clean
- [list of checks that passed]

### Auto-updated (safe drift)
- [file: change made]

### Needs human review
- [item: what changed, where, suggested action]

### Skipped (out of scope)
- [things this skill doesn't touch]
```

Keep under 30 lines. If everything is clean: "All 8 checks pass."

## Why this skill exists

The webapp depends on undocumented contracts with the simulator (field names, print formats, filename conventions, flag names). These contracts break silently the moment the simulator changes. This skill is the explicit, automatable check that those contracts hold.

## Companion files

- `webapp/server.py` — the FastAPI app
- `webapp/static/app.js` — main frontend bootstrap
- `webapp/static/world.js` — world viz + live mode
- `webapp/static/charts.js` — chart utilities
- `webapp/static/ui.js` — toasts, shortcuts, state, experiment helpers
- `webapp/capability_status.json` — capability snapshot for Home panel
  (manual source of truth, updated with major milestones)
- `tests/test_webapp_server.py` — server-side smoke tests
- `webapp/README.md` — usage + Phase 1/2/3 plan
