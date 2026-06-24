# Frontend → INTERACT-FIRST console — scoping (read-only)

**2026-06-23. READ-ONLY scoping. No build, no `sim/` edit.** Owner directive (2026-06-23): the webapp should
NOT showcase capabilities / history / milestones. It is a **functional console** with **3 jobs**:

1. **LAUNCH + MANAGE runs** (start a run, watch it, kill/pause it).
2. **VISUALIZE the brain + its activities** (spikes / active regions / world).
3. **INTERACT — chat/talk to the brain**, with the **CHAT surface as the centerpiece**.

This doc assesses what the existing webapp HAS per job (reuse), what to REMOVE/demote, designs the INTERACT
surface (backed by the real brain-chat agent, not the deprecated MockLLM), the brain-visualization plan during a
conversation, and a cheapest-first build sequence. Honest about scope; `sim/`-touch / restart needs flagged.

---

## TL;DR

- **Job 1 (launch/manage) — already solid, KEEP almost verbatim.** `POST /api/runs/launch` + the detached-run
  monitor (`/api/inflight`, `/ws/runs/{id}`, log tail, kill/pause, orphan-recovery across restart) is the
  console's strongest existing pillar. The `Lab` tab + the `Home` "Active runs" panel are the management surface.
  The only debt is the enormous PRESETS catalog (~90 presets) — curate it down to a short menu, don't rebuild it.
- **Job 2 (visualize) — partial reuse, with one important GAP.** The 2D `World` (world.js) + 3D `Brain`
  (brain3d.js) + the live `[ACTIVITY]` per-region firing channel all exist and work — **but they are driven by a
  navigation runner's stdout (`g11_bg_runner --emit-activity`), NOT by a conversational/chat session.** There is
  no live brain-activity feed for the chat brain today. Showing "which regions fired / which facts were recalled"
  during a conversation is **new work** (the chat path returns answers, not activity frames).
- **Job 3 (interact) — the real work.** The webapp's ONLY chat surface today is the **deprecated Path-3
  `/api/llm-chat` MockLLM↔BridgeMemory** panel, buried inside the Lineages tab. It is a regex pattern-matcher over
  a 4-motor key/value memory — NOT the brain-chat agent, NOT the grounded-language faculty, NO no-confab moat as
  the owner means it. **Replace it.** The correct backend already exists, fully built, as a runner:
  `research/runners/brain_chat_tui.py` (`ChatBrain`) over `developed_brain_io.load_developed_brain` +
  `BrainConversationalAgent`/`MultiTurnAgent` + the off-bridge Qwen render→verify. The build is: wrap `ChatBrain`
  behind a new `/api/brain-chat` endpoint + a dedicated **Interact** tab (the centerpiece), streaming the answer
  and clearly surfacing **abstentions (the moat)**.

- **REMOVE / DEMOTE per the new directive:** the capability/milestone/progress surfaces. Most were ALREADY removed
  from `Home` (the index.html comment at lines 65-71 confirms KPIs / "capability status" / Path-F pillars /
  distribution / activity+findings feeds were stripped). The remaining capability machinery to retire from the UI:
  the `/api/capability-status` endpoint + `capability_status.json` + `renderCapabilityStatus()` (dead in `Home`
  but still shipped), and the research-y tabs (`Experiments`, `Findings`, `Plans`, `Language` text-IO aggregates,
  `About`/capability copy). These are NOT the console's 3 jobs — demote behind a single "Research/Archive" area or
  drop from nav. (Runs history stays — it is job 1.)

- **Build sequence (cheapest-first):** (0) curate presets; **(1) `/api/brain-chat` endpoint + a minimal Interact
  tab — the centerpiece, highest leverage, no `sim/` edit;** (2) reframe nav to the 3 jobs + demote the
  capability/research surfaces; (3) wire chat → live brain visualization (the new GAP: emit activity from the chat
  agent). Each step ships independently. Only step 3 needs an `[ACTIVITY]`-style emit from the conversational path
  (runner/endpoint code, still NO `sim/` edit). **Every new endpoint/route needs a `uvicorn` reload to take
  effect** (in-flight detached runs survive it; the server is built for restart).

---

## Job 1 — LAUNCH + MANAGE RUNS  (KEEP; curate, don't rebuild)

### What the webapp HAS (reusable, strong)

| Piece | Location | Reuse |
|---|---|---|
| Launch a runner subprocess (preset + seed(s) + grid + extra flags) | `POST /api/runs/launch` (`server.py` ~1300-1490) | KEEP. Detached, own GPU ctx, survives webapp restart. |
| Per-preset arg lists + per-preset runner module | `PRESETS` dict (`server.py:365-1179`), `PRESET_RUNNERS` (`server.py:1186+`) | KEEP the mechanism; **CURATE the list** (~90 entries → a short menu). |
| In-flight monitor (detached runs write `*.pid`/`*.log`, server scans + tails) | `/api/inflight`, `_drain_log` (`server.py:1865`), `_process_alive`/orphan recovery | KEEP — this is the management core. |
| Live stdout + parsed progress over WS | `WS /ws/runs/{id}`, `_try_parse_progress` (`server.py:196`) | KEEP. |
| Kill / pause a run | `POST /api/runs/launch/{id}/kill`, control-file write (`server.py`), `world.js` pause | KEEP. |
| Runs history + detail (phase stats, curves) | `/api/runs`, `/api/runs/{name}`, Runs tab | KEEP (this IS "manage runs"; it's not "milestones"). |
| Frontend: `Lab` tab (launch form) + `Home` "Active runs" panel (auto-refresh 5s) | `index.html` lines 549-664 / 65-83, `app.js` `refreshInflightPanel` (~3170) | KEEP both. `Home` is already a run-management surface per its own comment. |

### What to REMOVE / demote
- **The PRESETS catalog is the only real debt.** ~90 presets (every find-the-ceiling / phase-1.5 / consolidation
  variant) overwhelm the launcher `<select>` (index.html 553-625 is a wall of `<option>`s). Curate to the handful
  the console actually needs (a flagship nav run, a smoke, the conversational/develop demo, maybe a couple). Keep
  the full dict for power-users behind "extra flags" or an "all presets" expander — but the default menu is short.

### Honest scope
Job 1 needs essentially no new backend. It's a **curation + nav-placement** task, not a build.

---

## Job 2 — VISUALIZE the brain + its activities  (partial reuse; one GAP)

### What the webapp HAS

| Surface | Location | What it shows | Driven by |
|---|---|---|---|
| 2D World playback + Live mode | `world.js` (1869 ln), World tab | agent/goal/trajectory on the grid; live `recent_dist` chart; retina (V1) panel | nav runner progress events over `/ws/runs/{id}` |
| 3D Brain (Three.js, no build step) | `brain3d.js` (2057 ln), Brain tab | ~50 region spheres, synaptic pathways, traveling-spike pulses, bloom; replay + live | nav runner; region brightness from action/reward + the `[ACTIVITY]` channel |
| **Live per-region firing channel** | `[ACTIVITY] {json}` line → `_try_parse_activity` (`server.py:163`) → `ActivityFrame` ring buffer → `/ws/runs/{id}` (latest-wins coalescing) | `{region_name: firing_fraction, flux: {pathway: val}}` per frame | **`g11_bg_runner --emit-activity` ONLY** (`server.py:1386-1392` gates `--emit-activity` to runners that support it) |

The plumbing for "show live brain activity" is **already built and good**: a tiny fire-and-forget stdout line
(~30 region floats), ring-buffered server-side, coalesced per WS client so a slow browser never backs up the sim.
The 3D renderer already consumes it.

### The GAP (this is the new-work part of job 2)
**The activity channel is wired to NAVIGATION, not to the CHAT brain.** During a conversation we want to show:
spikes / which regions are active / **which facts were recalled** / when the moat abstained. None of that is
emitted today — the chat path (`brain_chat_tui.ChatBrain.answer`) returns a string, not activity frames. So:

- **Reusable as-is:** the WS activity transport (`ActivityFrame` + ring buffer + coalescing), the 3D region scene,
  the region-name → sphere mapping, the "active region glows / pulse travels" rendering.
- **New work:** make the conversational agent emit an `[ACTIVITY]`-shaped signal (or a chat-specific equivalent)
  so the renderer has something to show during a turn. Cheapest first cut is **semantic, not per-neuron**: emit,
  per answered turn, the recalled fact (agent/action/patient), the gate decision (answered vs **abstained**), and
  which conceptual regions lit (parser → composer/KB → cleanup → renderer). That is a faithful "what the brain did
  on this turn" view without needing live `cp_firing_states` off the GPU. A later upgrade can stream real
  per-region firing fractions from the live co-resident bridge the chat agent runs on.

### What to show during a conversation (recommendation)
1. **A "what the brain did" strip under each answer** (cheapest, highest signal): `parsed: dog chase cat` →
   `recalled: (dog, chase, cat)` → `verified ✓` → fluent answer; OR `recalled: ∅ → ABSTAINED (moat)`.
2. **The 3D Brain in a side panel**, regions pulsing along the conversational pathway (language-in → composer/KB →
   cleanup → language-out), reusing brain3d.js. Drive it from the per-turn semantic signal first; upgrade to live
   firing fractions later.
3. **The moat made loud**: abstentions render distinctly (a greyed "I don't know about that — the brain wasn't
   taught this" card), because the moat is the project's headline property and the owner wants it clearly shown.

### Honest scope
Reusing the **renderers** is free; producing a **chat-driven activity feed** is the genuine build. Recommend
shipping the per-turn semantic strip (job-3-adjacent, trivial) first, and treating the live 3D-during-chat as a
follow-on once the chat endpoint exists.

---

## Job 3 — INTERACT (chat) — the CENTERPIECE  (replace the MockLLM; the real backend already exists)

### What the webapp HAS today (and why it's wrong for this directive)
- **`/api/llm-chat` + `/api/llm-chat/{name}/transcript` + `/reset`** (`server.py:2526-2698`) drive a
  **MockLLM ↔ BridgeMemory** tool-use loop. `MockLLM` is a **regex pattern dispatcher** ("remember that my X is
  north", "what's my X") over a **4-motor (N/E/S/W) key/value** memory (`sim/bridge_memory.py`). It is the
  DEPRECATED Path-3 secondary application (CLAUDE.md marks Path 3.2 "SECONDARY", 3.3 "DEPRECATED for primary
  path"). The frontend panel (`renderLLMChatPanel`, `app.js:2265-2421`) is **buried inside the Lineages tab** and
  even labels itself "(Phase 3.2 demo)".
- This is NOT the brain-chat the owner means: it doesn't use `BrainConversationalAgent`, has no grounded codes, no
  who/what/yes-no/describe/reason, no fluent generative render, and its "moat" is just "pattern not recognized".

**Verdict: deprecate the `/api/llm-chat` MockLLM panel; build a new `/api/brain-chat` surface.**

### The CORRECT backend already exists (reuse-by-import, NO `sim/` edit)
`research/runners/brain_chat_tui.py` is the production brain-chat agent, already assembled:

- **`ChatBrain`** (`brain_chat_tui.py:199-318`) — the full conversational turn:
  `answer(question)` → **GATE** (`gate()`: route the free-text question to a stored SVO fact via `QuestionRouter`,
  resolve self-aliases + multi-turn anaphora, then **VERIFY against the brain's spiking recall** `inner.what_does`)
  → **CONSTRAIN+VERIFY render** (`render()`) → `(answer_string, abstained_bool)`. On no match it returns
  **`"I don't know about that.", True`** — the no-confab **moat**, surfaced as a clean boolean the UI can render.
- **The brain it serves** — `developed_brain_io.load_developed_brain(path)` (`developed_brain_io.py:234`)
  reconstructs the EXACT developed brain (grounded codes `.npz` + facts.json + vocab + seed) as a
  `BrainConversationalAgent` (or `MultiTurnAgent` for anaphora). The "developed brain" bundle is what the
  longitudinal develop loop / `_self_knowledge_demo` produce + `save_developed_brain` writes. **Note:** no bundle
  exists on disk yet in this repo (no `brain.json`, no `_self_knowledge_grounded_codes.json`) — so the endpoint
  must handle: a `--load <bundle>` path, the self-knowledge codes path, OR the **tiny-demo CPU fallback**
  (`_build_tiny_demo`, GPU-free) for an out-of-box smoke. The TUI already encodes exactly this load precedence
  (`load_brain`, `brain_chat_tui.py:415-435`).
- **The fluent renderer** — `QwenRenderer` (`brain_chat_tui.py:169-192`) wraps the **off-bridge Qwen-0.5B grounded
  faculty** (`_grounded_lang_integration_derisk.SpikingQwenFaculty`): GATE→CONSTRAIN→**VERIFY** (re-parse the
  generated prose back to an SVO, require it to MATCH the gated fact, else speak the raw fact). This is the
  hallucination-proof render: **the brain supplies + verifies CONTENT; the LLM supplies only fluent surface form.**
  For GPU-free / CI, `StubRenderer` (template-stub) or `--no-renderer` (raw triples) substitute identically.

### The INTERACT-surface design

**New endpoint `POST /api/brain-chat`** (sibling of the existing `/api/llm-chat`, but backed by `ChatBrain`):

- Body: `{ session, message, brain?, renderer? }`. `brain` selects the load source (a developed-brain bundle dir,
  the self-knowledge codes, or `tiny-demo`); defaults to whatever the server is configured to serve. `renderer` ∈
  `{qwen, stub, raw}` (qwen needs `SIM_BACKEND=cupy` + a free GPU; stub/raw are GPU-free).
- Server keeps an **in-process `ChatBrain` cache keyed by session** (exactly like `_LLM_ORCHESTRATORS` does for the
  MockLLM, `server.py:2513`), because the first load (and the Qwen warm-up, `SpikingQwenFaculty.load_seconds`) is
  expensive and must be paid once and kept warm (the TUI loads the faculty once and keeps it warm by design).
- Returns: `{ answer, abstained, recalled_svo, verified, renderer, gen_seconds }` (and, when wired, a per-turn
  activity payload for job 2). `abstained=true` ⇒ the moat fired; the UI renders it distinctly.

**Streaming the answer + showing abstentions:**
- The Qwen render is a short greedy generation (`max_new_tokens≈24`, `_generate`); true token streaming is
  possible later, but the **cheapest first cut is a non-streamed POST with an optimistic "brain> thinking…"**
  placeholder (the TUI already prints exactly this, `brain_chat_tui.py:512-513`), then replace it with the verified
  answer. This matches the existing chat-panel UX (`app.js` `appendChatTurn` + "Thinking…" button state, 2371).
- **Abstention is a first-class boolean** (`ChatBrain.answer` returns it), so the UI never has to guess: render
  abstained turns as a greyed "I don't know about that — the brain wasn't taught this" card. This is the moat made
  visible, per the owner's emphasis.

**Where the RICH-answer mode plugs in:** `ChatBrain.render()` is the single seam. Today it renders ONE verified
SVO. The upcoming rich mode (multi-fact / reasoned / multi-hop answers) slots in by (a) having the GATE return more
than one fact (the agent already exposes `reason_chain` / `query_chain` / `elaborate`), and (b) feeding the faculty
a richer constrain prompt — **without changing the endpoint contract** (still `{answer, abstained, ...}`, the
VERIFY step still guards every emitted clause). So the endpoint + tab built now are forward-compatible with rich
answers; only `ChatBrain` internals grow.

**Frontend: a dedicated `Interact` tab (the centerpiece).** Reuse the existing chat-panel widgets almost verbatim
(`renderLLMChatPanel`'s log + input + Enter-to-send + transcript load, `app.js:2265-2470`) but: (1) point them at
`/api/brain-chat`; (2) make it a **top-level tab, not a sub-panel of Lineages**, registered via the documented
one-line `TAB_REGISTRY` pattern (`app.js:339`, the comment block 317-337 spells out the exact 3 steps); (3) put it
**first/leftmost** in nav (it's the centerpiece); (4) render `abstained` distinctly; (5) add a renderer toggle
(qwen / stub / raw) and a "what the brain knows" (`/facts`) affordance (the TUI's `/facts` lists `composer.kb`).

### Honest scope / caveats
- **`ChatBrain` is import-clean and `sim/`-free** (it imports `developed_brain_io`, `brain_conversational_agent`,
  `multi_turn_agent`, `_grounded_lang_integration_derisk`). Wrapping it in an endpoint is genuinely a thin adapter.
- **The Qwen renderer needs a GPU** (`cupy` backend + CUDA torch); on a GPU-less host the endpoint must default to
  `stub`/`raw` so the console still works (the moat + recall are CPU; only fluent surface needs the GPU). The TUI
  already warns + degrades (`brain_chat_tui.py:180-182`).
- **No developed-brain bundle is checked in.** The console's first-run experience must be the **tiny-demo brain**
  (a handful of self-facts, GPU-free), with a clear "load a developed brain" affordance for when a bundle exists.
  Producing/serving a real bundle (self-knowledge or a develop-loop save) is a separate content step, not blocking.

---

## REMOVE / DEMOTE (per the "no capabilities/milestones" directive)

| Surface | Status | Action |
|---|---|---|
| `Home` KPIs / "capability status" / Path-F pillars / distribution / activity+findings feeds | **Already removed** from the rendered Home (index.html 65-71 comment confirms) | Done — just delete the now-dead `renderCapabilityStatus()` (`app.js:3304+`) + the `/api/capability-status` endpoint (`server.py:3608`) + `capability_status.json`. |
| `Experiments` tab (auto-grouped per-experiment aggregates, delta-vs-flagship) | Research/milestone framing | Demote out of primary nav (move to a single "Archive/Research" area or drop). Not one of the 3 jobs. |
| `Findings` / `Plans` tabs (markdown browsers) | Research docs | Demote (Archive area) or drop from the console nav. |
| `Language` tab (text-IO I→W/W→A aggregates) | Old research metric | Drop from nav (superseded; not interact). |
| `About` tab (project overview + CURRENT-STATE + biology copy) | Capability/marketing copy | Drop or fold into a tiny footer link. |
| `Bridges` / `Lineages` tabs | Asset libraries | KEEP only the parts that feed jobs (e.g. choosing a developed brain to chat with / a bridge to launch); the **MockLLM chat panel inside Lineages → REMOVE** (replaced by the Interact tab). |

Net: nav collapses from ~12 tabs to roughly **Interact · Visualize (World/Brain) · Lab (launch) · Runs**, plus an
optional folded-away "Archive". Keep `keep-webapp-current` in mind: the launcher/preset/world contract with the
simulator is brittle and undocumented; curating presets and removing tabs is low-risk, but verify the launch path
still works after the cut.

---

## BUILD SEQUENCE (cheapest-first)

0. **Curate PRESETS** (no code risk): trim the launcher `<select>` to a short default menu; keep the full dict
   behind an expander/extra-flags. *(server.py PRESETS unchanged; index.html option list trimmed.)*
1. **`/api/brain-chat` endpoint + minimal `Interact` tab — THE CENTERPIECE.** Wrap `brain_chat_tui.ChatBrain`
   (load via `developed_brain_io` / self-knowledge / tiny-demo fallback; session-cached + warm). Reuse the existing
   chat-panel widgets pointed at the new endpoint; render `abstained` distinctly; renderer toggle (qwen/stub/raw).
   **No `sim/` edit.** Needs a `uvicorn` reload to register the route. Highest leverage, smallest surface.
2. **Reframe nav to the 3 jobs + demote capability/research surfaces.** Make Interact leftmost; collapse
   Experiments/Findings/Plans/Language/About into an Archive (or drop); remove the dead capability endpoint +
   `renderCapabilityStatus` + the MockLLM panel.
3. **Wire chat → live brain visualization (the job-2 GAP).** Emit a per-turn "what the brain did" signal from the
   chat agent (recalled SVO + gate decision + conceptual regions touched), render it as the under-answer strip and
   drive brain3d.js region pulses. Upgrade later to real per-region firing fractions from the live co-resident
   bridge. Runner/endpoint code only — **NO `sim/` edit** for the semantic first cut.

**Restart/`sim/` flags:** none of the above edits `sim/`. Every new endpoint/route requires a `uvicorn` reload to
take effect; in-flight **detached** runs survive the restart (the server is explicitly built for it — `_drain_log`
+ orphan recovery), so a reload mid-conversation only drops the in-process `ChatBrain` cache (re-warms on next
message), it does not kill running jobs.

---

## Key file references

- Webapp: `webapp/server.py` (3700 ln; PRESETS 365-1179, `/api/info` 2184, MockLLM chat 2506-2698,
  `_drain_log`/activity 1865-1935, capability endpoint 3608), `webapp/static/index.html` (nav 43-56, launcher
  549-664, Home 65-83, World/Brain tabs), `webapp/static/app.js` (4232 ln; TAB_REGISTRY 339, MockLLM chat panel
  2265-2470, inflight panel ~3170), `webapp/static/world.js` (1869 ln; live mode + WS), `webapp/static/brain3d.js`
  (2057 ln; 3D scene + activity consumption), `webapp/README.md`.
- Interact backend (reuse): `research/runners/brain_chat_tui.py` (`ChatBrain` 199-318, `QwenRenderer`/`StubRenderer`
  153-192, `load_brain` 415-435), `research/runners/developed_brain_io.py` (`load_developed_brain` 234,
  `save_developed_brain` 105), `research/runners/_grounded_lang_integration_derisk.py` (`SpikingQwenFaculty` 168,
  GATE→CONSTRAIN→VERIFY 282+).
