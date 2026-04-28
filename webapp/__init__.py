"""Web dashboard for the neural simulator research workflow.

Phase 1 (2026-04-28): runs as a separate FastAPI process. Surfaces:
- List of completed runs (research/findings/raw/g11_bg/*.json) with sortable finalQ
- Detail view of a single run (per-trial finalQ chart, motor counts, phase stats)
- Browse markdown findings (research/findings/*.md)
- Kick off a new runner from preset configs (flagship, perception-only, baseline)
- Live tail of stdout via WebSocket for in-flight runs

Phase 1 deliberately runs out-of-process from the simulation: a launched
runner is a subprocess with its own GPU context. Phase 2 will add the
agent-in-environment Three.js viz; Phase 3 will replace the gridworld
with a PyBullet 3D world.

Run with: uvicorn webapp.server:app --reload --port 8765
"""
