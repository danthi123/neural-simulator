# 2026-05-09 — Frontend audit arc (during Phase 1.5 wait window)

User request: "Any frontend updates/additions for you to work on while this
phase runs? I know you added some presets, but maybe worth looking at the
frontend as a whole and seeing what can be added/improved?"

Context: Phase 1.5 multi-seed at scaled arch in flight (~9 hrs total),
3 commits this iteration. Holistic frontend pass while GPU is busy.

## What shipped

| # | Commit | Change | Why it matters |
|---|---|---|---|
| 1 | `3ae8220` | **Launcher dropdown — 14 missing presets** | Backend ships → frontend lags. Of 34 backend `PRESETS`, only 19 had dropdown entries. Surfaced all chat / consolidation / Phase 1.x / 2.3 presets under 6 new section headers with empirical-result text inline (e.g. "★ 12-word at scaled n_motor=2000; 3/3 GO 95%/115% retention [~220min]"). |
| 2 | `d14711c` | **Capability status panel on Home tab** | New `webapp/capability_status.json` source-of-truth + `/api/capability-status` endpoint + Home `<section>` showing latest validated capability, 6 Path F empirical pillars with status badges, capacity scaling rule table, and master-plan position. 6 new tests passing. Section gracefully hides if endpoint 404s (so older webapps don't break). |
| 3 | `b6974a3` | **Findings tag classifier rewrite** | Recent findings (Phase 1.3 / 1.4 / 1.5 / 2.x, Tier 1 / 2.1 / 2.2 / 2.3, capacity hypothesis, multi-seed validations, anti-cheat) all fell into "Other" because the classifier predated the Path F arc. Added 11 new patterns ahead of older Cluster letters. Sanity-checked against 11 real recent filenames — all now tagged correctly. |

Plus skill updates (outside repo, durable on disk for future autonomous arcs):

- `autonomous-runs/SKILL.md` principle #10 extended to mention
  `capability_status.json` as a frontend touchpoint to maintain when
  major milestones land
- `keep-webapp-current/SKILL.md` Check I added — explicit drift check
  that compares `capability_status.json` `as_of` against most recent
  `*CONFIRMED*` / `*BREAKTHROUGH*` finding doc

Tests: **68/68** passing across 5 test files (test_webapp_server +
test_chat_demo_aggregate + test_phase_1_5_aggregate + test_chat_repl +
test_synonym_consistency).

## Gap analysis methodology

Walked the 10 dashboard tabs (overview, launcher, runs, experiments,
world, brain, language, findings, plans, info) plus the static assets
(app.js 3479 lines, brain3d.js 1726, world.js 1869, charts.js 411,
ui.js 260, index.html 685, style.css 2481).

**Findings:**

| Tab | Surface | Drift level | Action |
|---|---|---|---|
| Overview | KPIs, in-flight, distribution chart, activity feed, findings feed | High — no "what's currently validated" summary | ✅ Added capability status panel |
| Launcher | Preset dropdown + extra args | High — 15/34 backend presets missing | ✅ Surfaced all 14 reachable + 6 group headers |
| Runs | Filter chips + live panel + detail split | Clean | (no action) |
| Experiments | Auto-grouped by suffix | Clean | (no action) |
| World | Scrubber, retina, run HUD | Clean (recent additions for visual cortex / live mode) | (no action) |
| Brain | Three.js 3D viz + region pathway toggles | Clean | (no action) |
| Language | Text I/O W→A breakdown | Clean | (no action) |
| Findings | Search + tag chips + detail panel | Medium — classifier missed Phase 1.x / Tier 2.x / capacity | ✅ Rewrote classifier |
| Plans | Search + detail panel | Clean — Phase 2 resumption plan reachable | (no action) |
| Info | About panel | Clean | (no action) |

## Architecture: capability_status.json source-of-truth pattern

The decision was to make the Home capability widget JSON-backed rather
than inline HTML, with these properties:

1. **Single edit point** — when a milestone lands, edit one JSON file
   instead of three (HTML + CSS + JS).
2. **Stable schema** — tests enforce shape (`pillars[].status` must be
   one of VALIDATED / BOUNDARY / PREDICTED / NEGATIVE), so future
   milestone-edits can't accidentally break the panel.
3. **Graceful fallback** — if the JSON is missing, endpoint returns
   a `_warning` stub and the frontend hides the section. Fresh
   checkouts don't show broken UI.
4. **Sister-skill awareness** — `keep-webapp-current` Check I now
   compares the JSON's `as_of` to recent `*CONFIRMED*` finding docs
   and flags drift. Future autonomous arcs that run that skill will
   surface stale capability data automatically.

Schema:

```json
{
  "as_of": "YYYY-MM-DD",
  "headline": {
    "tier": "...", "result": "...", "metrics": "...",
    "wall_clock": "...", "finding_doc": "...md", "summary": "..."
  },
  "pillars": [{ "name", "status", "metric", "doc", "date" }, ...],
  "capacity_rule": {
    "rule": "...",
    "rows": [{ "vocab", "subpops", "n_motor", "neurons_per_subpop", "status" }, ...]
  },
  "phase_status": { "active", "next", "after_next", "master_plan", "resumption_plan" }
}
```

## Caveat: webapp restart required for endpoint to go live

The running uvicorn instance is a parent of the Phase 1.5 runner
subprocess chain. Restarting now would risk losing the in-flight
training. The new endpoint code is committed and tested via FastAPI
TestClient (6 tests passing), but won't go live until the next clean
webapp restart. The frontend gracefully hides the section on 404, so
no broken UI — just hidden until restart.

After Phase 1.5 batch completes, a single uvicorn restart will pick
up:
- `/api/capability-status` endpoint (new)
- The 14 launcher dropdown options (already live — static-served)
- The findings tag classifier improvements (already live — static-served)

Static-served changes require only a browser refresh.

## What's next (for the next user prompt or autonomous tick)

Per master plan:
1. Wait for Phase 1.5 multi-seed (in flight, ~9 hrs total)
2. Aggregate results, decide if 3/4 benchmark threshold passes
3. If yes: master plan named milestone — "BIOLOGY-GROUNDED CONTINUAL
   LEARNING VALIDATED"
4. Then 16-word smoke (~3.5 hrs) → 16-word multi-seed (~10 hrs)
5. Then Phase 2.2b 10M-param overnight (~14 hrs)
6. Then Phase 2.3b transfer test (~3-6 hrs)

Frontend follow-ups for future arcs (NOT urgent, none of these are gaps
the user explicitly asked for):
- Live multi-seed batch progress widget — currently the bash launcher
  chains seeds via PowerShell `Wait-Process` but the dashboard shows
  individual seed runs without "X of N seeds complete" rollup.
- Vocab-size hint overlay on launcher when a chat / consolidation
  preset is selected (cf. inline option text, which arguably already
  covers this).
- Per-pillar finding-doc preview-on-hover on the capability panel
  (small interaction polish).

## Related

- Capability JSON: `webapp/capability_status.json`
- Endpoint: `webapp/server.py` `/api/capability-status`
- Render: `webapp/static/app.js` `renderOverviewCapability()`
- Tests: `tests/test_webapp_server.py` `test_capability_status_*` (6)
- Skill update (auto-runs principle #10): documents the maintenance
  responsibility for future arcs
- Skill update (keep-webapp-current Check I): documents the drift
  detection responsibility for future arcs
