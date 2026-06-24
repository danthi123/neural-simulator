# PAIRED scoping — the artificial-life CAPSTONE + the interact-first CONSOLE (the owner wants them TOGETHER)

**Date:** 2026-06-24
**Type:** READ-ONLY research-gate scoping for the next arc (owner-approved, done TOGETHER). NO `sim/` edit, NO build,
NO commit. Deliverables = `research/findings/raw/_scoping_capstone_console.json` + this doc.
**North star:** artificial life with a proper brain analogue (`project_actual_goal_artificial_life_brain_analogue`).
The two pair so the owner can **WATCH + TALK to a brain developing over simulated time.**

---

## 0. One-paragraph answer

**FAR more is already BUILT than the two prior 2026-06-23 scopings assumed — both have been substantially REALIZED
since (the commits prove it: the develop-loop GPU-GO, `rich_answer_composer`, the `/api/brain-chat` endpoint, the
self-knowledge build).** (A) The longitudinal develop loop is **GPU-GO at 1-seed** — `_longitudinal_develop_loop_gpu.py`
runs the whole develop→converse→consolidate→grow→persist cycle with REAL stream-cortex learning (vocab grows, facts
accumulate, retention holds, moat 0-FA, persists+resumes), per-day ~2.2 min → a "year" ≈ ~13.5 hr LOCAL. (B) The
console's **`/api/brain-chat` endpoint + the `Interact` tab (leftmost, the default landing tab) ALREADY SHIP**, backed
by the real `brain_chat_tui.ChatBrain` (GATE+VERIFY+the no-confab moat), session-cached + warm, abstentions rendered
distinctly. **So this is now mostly an ASSEMBLY + WIRING + SCALE-HORIZON job, not a frontier.** The two **genuine
gaps**: (A) the lineage persists `DevelopState`-JSON only — NOT `cp_connections`/the learned codes — so resume
RE-HEARS to re-derive the codes (a wall-clock tax + a long-horizon fidelity caveat), AND `save_developed_brain` is
never CALLED, so **no developed-brain bundle exists for the console to load** (the GROWTH/CONSOLIDATE stages are
owner-approved STAND-INS); (B) the console's **RICH-answer path is NOT wired** — the endpoint accepts `rich=True` and
calls `chat.render_rich(...)`, but `ChatBrain` has NO `render_rich` (the rich multi-sentence logic lives in the
SEPARATE `RichAnswerComposer` wrapper) — and there is no chat-driven brain-ACTIVITY feed (the `[ACTIVITY]` viz is
gated to the nav runner). **The PAIRING SEAM** is `developed_brain_io.save_developed_brain` (a self-contained bundle)
↔ the console's existing bundle-load path: produce ONE bundle from the capstone, point the console at it, talk to it.
**Cheapest-first: wire rich answers (B1) → save a bundle (A1) → list/select bundles (B2) → scale the horizon 6-seed
(A2) → chat-activity viz (B3) → persist `cp_connections` (A3).** NO `sim/` edit on the critical path.

---

## 1. (A) CAPSTONE — current state (read directly)

### What is BUILT (reuse-by-import, runs end-to-end)
- **The develop loop, GPU-GO 1-seed** — `research/runners/_longitudinal_develop_loop_gpu.py` (+ the CPU scaffold
  `_longitudinal_develop_loop.py` it imports the loop machinery from). Five stages every simulated day:
  **WAKE** (the REAL stream-cortex Hebbian co-occurrence learning on a *persistent* GPU bridge — `StreamCortex` over
  `_phaseB_onbridge_stream_cortex_derisk.build_stream_bridge`; the brain HEARS the TinyStories corpus window-by-window
  + its rate-Hebbian synapses learn the concept codes → grounded phasors) → **CONVERSE** (`MultiTurnAgent` on the
  *stream-learned* grounded codes) → **SLEEP** (self-replay + an OLD-fact retention re-test) → **GROWTH**
  (`TierPromoter` mastery→promote decision) → **PERSIST** (`BridgeLineage` atomic save/load).
- **The result** (`research/findings/2026-06-23-longitudinal-develop-loop-GPU-GO.md`): 4 days, vocab 6→24, facts 2→11,
  recall 1.0, retention 1.0 (no catastrophic forgetting), moat 0-FA, **corr(M,C) 0.894** (REAL learning, day0≠dayN);
  it **persists + resumes** (presented day 4 on a reload, lived 5 more days); the **frozen-brain anti-cheat holds**
  (plasticity-off → 0 facts, 0 fidelity). Built-in anti-cheats: frozen-brain (`do_frozen`) + persistence-resume
  (`do_resume`) + moat-clean. Per-day ~133 s → **compressed-week 15.6 min, month ~1 hr, year ~13.5 hr LOCAL**.
- **Wall-clock / VRAM** — LOCAL, comfortably **<24 GB** (the 320-stream cortex ~10K neurons / ~25M synapses fit on the
  3090). CONFIRMED no cloud needed (per `feedback_long_local_runs_ok_confirm_cloud_cause` — the only possible cloud
  trigger is the OPTIONAL generative-faculty 50–200M scale-up, itself likely-local).
- **The human REPL** — `brain_chat_tui.py` (`ChatBrain`): GATE (spiking recall + the no-confab moat) → CONSTRAIN+VERIFY
  fluent render → `(answer, abstained)`; loads a `developed_brain_io` BUNDLE **or** the self-knowledge codes **or** a
  tiny-demo; `--rich` routes to `RichAnswerComposer` (substantive multi-sentence grounded replies). So *talk to a
  developed brain* is DONE as a TUI.

### The TWO genuine gaps (A)
1. **Persistence DEPTH** — `BridgeLineage.save/load` persists the `DevelopState` payload (facts/vocab/tier/day/metrics)
   as JSON only; it does **NOT** persist `cp_connections` (the stream cortex's learned synapses). On resume the loop
   **RE-HEARS the cumulative vocab to RE-DERIVE the codes** (`develop_gpu` resume branch — the in-code comment: *"cheap
   stand-in for loading the bridge's synaptic store; the GPU full-persist of `cp_connections` is a follow-on"*). At a
   long horizon this is a per-resume **wall-clock tax + a fidelity drift risk**. The fix exists in-project
   (`BridgeLineage.export_shards` / `save_checkpoint` persist `cp_connections`).
2. **No BUNDLE is produced** — `developed_brain_io.save_developed_brain` (the self-contained `brain.json` +
   `grounded_codes.npz` + `facts.json` + `lineage/` the console can `--load`) is **never CALLED** by the loop OR
   `_self_knowledge_demo`. **Verified: `find . -name brain.json` = empty.** The self-knowledge demo saves codes to a
   flat `_self_knowledge_grounded_codes.json` (229 KB, ON DISK) — a *different* artifact the console loads via the
   `'self-knowledge'` brain source, not the bundle path. **So a develop-loop-developed brain cannot be loaded by the
   console yet** (only the self-knowledge codes + the tiny-demo can).

### The owner-approved STAND-INS (honest)
- **GROWTH** = `TierPromoter` DECISION + a lineage growth-event (the real neuron-count arch rebuild + weight-transfer
  is a heavy GPU follow-on; `auto_grow_chat.py`'s orchestration runs but with mock train/transfer).
- **CONSOLIDATE** = self-replay + retention re-test (the full-SWR-on-the-conv-bridge is deferred — `consolidation_trainer.py`
  hard-imports cupy AND builds a *different* direction-vocab bridge).

---

## 2. (B) CONSOLE — current state (read directly)

Per `feedback_frontend_purpose_console_not_dashboard`: the webapp is a **functional console** with 3 jobs —
launch/manage runs · visualize the brain · **INTERACT (chat, the centerpiece)**. The 2026-06-23 scoping's recommended
build (a `/api/brain-chat` endpoint + a leftmost Interact tab) has **already been built**.

### What is BUILT
- **`/api/brain-chat` + `/api/brain-chat/reset`** (`webapp/server.py` ~2806) — wraps `brain_chat_tui.ChatBrain`
  (reuse-by-import, NO `sim/` edit), **session-cached + kept WARM** (cache key `(session, brain, renderer)`),
  **GPU-light by default** (`_default_brain_renderer` picks `qwen` only when a free CUDA GPU + cupy backend, else the
  GPU-free `stub`). `brain ∈ {tiny-demo, self-knowledge, a developed-brain bundle DIR}`. Returns
  `{answer, abstained, recalled_svo, verified, renderer, source}` — the **moat is a first-class boolean**.
- **The `Interact` tab** — `index.html` (FIRST/leftmost + active by default) + `app.js setupBrainChat` (posts
  `/api/brain-chat`, renders abstentions distinctly with a `∅` icon + a *"moat fired"* note, an SVO toggle, a renderer
  selector auto/qwen/stub/raw, suggestion chips incl. the moat probe *"who wrote romeo and juliet?"*) + `style.css`.
  The nav already collapsed to the 3 jobs (Interact is the default landing tab).

### The TWO genuine gaps (B)
1. **RICH-answer NOT wired** — the endpoint ACCEPTS `rich=True` and calls `chat.render_rich(gate_svo)` *if*
   `hasattr(chat,'render_rich')` — but **`ChatBrain` has NO `render_rich`**. The rich multi-sentence logic lives in the
   SEPARATE `RichAnswerComposer` wrapper (a different `.answer(question)->dict` API, *not* a `render_rich(svo)`). So
   today the endpoint silently falls back to the single-fact render for `rich=True`. This is the owner's *"conversation
   is too thin"* upgrade and it is **unwired**.
2. **No chat-driven ACTIVITY feed** — the live per-region brain-activity channel (`[ACTIVITY] {json}` →
   `_try_parse_activity` → `ActivityFrame` ring buffer → `/ws` → `brain3d.js`) is gated to the **nav runner only**
   (`--emit-activity`, `g11_bg_runner`). The chat path returns answers, not activity frames. So *show the brain's live
   activity during a conversation* is NEW work.

---

## 3. Reusable-vs-to-build

### Reuse-by-import (NO build)
`(A)` the whole develop loop · the stream cortex · `MultiTurnAgent`/`BrainConversationalAgent`/`RFPhasorComposer`/
`one_brain_composer` (+ the moat) · `BridgeLineage` · `developed_brain_io.{save,load,is}_developed_brain` (built, just
never *called* by the loop) · the bundle format (the PAIRING SEAM). `(B)` the `/api/brain-chat` endpoints + the warm
session cache · the `Interact` tab + `setupBrainChat` · `brain_chat_tui.ChatBrain` + `RichAnswerComposer` +
`QwenRenderer`/`StubRenderer` · the `[ACTIVITY]` WS transport + `ActivityFrame` + the `brain3d.js` region scene.

### To build (cheapest-first)
- **B1 — wire RICH answers into `/api/brain-chat`.** Session-cache a `RichAnswerComposer` around the warm `ChatBrain`;
  on `rich=True` call `rich.answer(message)` → `{answer, abstained, facts/supporting SVOs, n_sentences}`; add a *rich*
  toggle + a *"tell me more"* affordance to the Interact tab. **NO `sim/` edit.** *The highest-leverage console
  upgrade.*
- **A1 — produce a developed-brain BUNDLE.** Call `developed_brain_io.save_developed_brain(agent, bundle_dir,
  develop_state=...)` at the END of a develop-loop run (and/or from `_self_knowledge_demo`) so the console/TUI can
  `--load` it. **NO `sim/` edit.** *Closes the A↔B pairing seam.*
- **B2 — `/api/brains` list + a console dropdown** to pick which developed brain to talk to (incl. the capstone
  bundle). **NO `sim/` edit.**
- **A2 — scale the HORIZON 6-seed** (the standing 6-seed gate) + measure the real per-day wall-clock at the chosen
  scale (compressed-week → month, a longer LOCAL run). **NO `sim/` edit.**
- **B3 — chat-driven ACTIVITY viz** — emit a per-turn semantic *"what the brain did"* strip (recalled SVO + answered/
  **ABSTAINED** + the conceptual pathway parser→composer/KB→cleanup→render; the endpoint already returns
  `recalled_svo`/`abstained`/`verified`) under each answer; optionally drive `brain3d.js` region pulses. **NO `sim/`
  edit for the semantic cut.**
- **A3 — persist `cp_connections` in the lineage** (`export_shards` / `save_checkpoint`) so RESUME **loads** the
  learned synapses instead of RE-HEARING — removes the per-resume wall-clock tax + the long-horizon fidelity drift.
  **NO `sim/` edit likely.** *The true-persistence long-horizon build.*
- **A4 — real auto-growth weight-transfer** (heavy GPU; the brain genuinely GROWS in neuron-count across tiers). May
  surface a **possible `sim/` touch** at a *conversational*-arch tier boundary (`set_pathway_weights` is validated for
  vocab tiers; the conversational-arch promotion is unverified).
- **A5 — OPTIONAL free-generation deepening** (the generative-loop generator + grow-no-forget), gated on a likely-LOCAL
  50–200M generator scale-up (the C2 in-band capacity wall). NOT on the critical path.

---

## 4. `sim/` edit flags

**NONE on the cheap-first critical path** (B1, A1, B2, A2, B3, A3 are all reuse-by-import / webapp / runner). The ONLY
possible `sim/` touch is **A4** (real auto-growth weight-transfer at a conversational-arch tier boundary, *if*
`set_pathway_weights` doesn't cover a needed pathway shape). **A5** (free-generation) is its own separately-tracked
deepening. Per `feedback_dont_gate_on_approval`, a justified `sim/` edit is fine with a byte-level diff review — but
none is needed for the paired build.

---

## 5. The cheapest-first BUILD ORDER

0. *(housekeeping, low-risk)* remove the dead capability endpoint + the deprecated MockLLM↔BridgeMemory chat panel;
   curate the launcher PRESETS to a short menu (mostly done).
1. **B1 — wire RICH answers** into `/api/brain-chat` + the Interact `rich` toggle + *"tell me more"*. *(NO `sim/`;
   needs a `uvicorn` reload.)* **The headline conversational upgrade.**
2. **A1 — save a developed-brain BUNDLE** from the develop loop. *(NO `sim/`.)* **Closes the pairing seam.**
3. **B2 — `/api/brains` list + a console dropdown** to pick a developed brain (incl. the capstone bundle). *(NO `sim/`.)*
4. **A2 — scale the develop-loop HORIZON 6-seed** + measure per-day wall-clock. *(NO `sim/`.)*
5. **B3 — chat-driven ACTIVITY viz** (per-turn semantic strip first, then brain3d region pulses). *(NO `sim/` for the
   semantic cut.)*
6. **A3 — persist `cp_connections`** so RESUME loads (not re-hears) the learned synapses. *(NO `sim/` likely.)*
7. **A4 — real auto-growth weight-transfer** (heavy GPU; possible `sim/` touch — byte-review if so).
8. **A5 — OPTIONAL free-generation deepening** (gated on a likely-local 50–200M generator scale-up).

---

## 6. HOW THEY PAIR

The console **LOADS + TALKS to** the capstone's developing brain through ONE shared artifact:
`developed_brain_io`'s **`brain.json` BUNDLE**. **A1** has the develop loop SAVE a bundle at the end of (or
periodically during) a run; **B2** has the console LIST + SELECT bundles; the **EXISTING** `/api/brain-chat` already
loads a bundle (`is_developed_brain_bundle`/`load_developed_brain`), and **B1** makes the answer rich. So the owner:

1. **launches the develop loop** from the console's Lab tab (job 1) — the brain develops over simulated days,
   **PERSISTING bundles**;
2. **opens the Interact tab** (job 3) and **picks the developed brain** (a day-N bundle), then **TALKS to it** — rich
   multi-sentence, gate→render→**verify**, multi-turn anaphora, the **moat abstaining** on the untaught;
3. **watches the brain's activity** (job 2) — the per-turn semantic strip / 3D region pulses.

The **day-0-vs-day-N comparison** (talk to an early bundle vs a late one) is the *"watch + talk to a brain developing
over time"* deliverable, end-to-end on the console. **NOTE:** the cheap cut works **TODAY** with the tiny-demo + the
self-knowledge codes (both already loadable); A1's bundle is what makes the **CAPSTONE's** developing brain the thing
the owner talks to.

---

## 7. Anti-cheats

**Capstone:** held-out probe (generalization ≠ memorization; `probe_heldout` per day) · retention / **no-replay
control** (consolidation OFF MUST catastrophically forget while the consolidated loop retains — ADD at A2; the
frozen-brain arm already covers *"is learning load-bearing"*) · **frozen-brain** (plasticity OFF → no rise, BUILT) ·
**persistence-resume** (BUILT) · **moat 0-FA across all days** (BUILT) · permuted-curriculum (ADD at A2).
**Console:** abstention is a first-class boolean surfaced distinctly (BUILT — the `∅ moat fired` note) · the
RICH-answer **per-sentence VERIFY** drops a confabulated sentence (BUILT in `RichAnswerComposer`'s adversarial
confab-drop check — preserve it through the endpoint wiring so the moat EXTENDS to multi-sentence) · the moat probe
chip stays (the firewall, one click).

---

## 8. Honest caveats
- GROWTH + CONSOLIDATE are owner-approved STAND-INS (decision-only growth, self-replay not full-SWR); the development
  SIGNAL (vocab/facts/retention) is real, the structural-growth + full-SWR are the heavier A4/follow-on builds.
- Resume RE-HEARS to re-derive codes (not loads `cp_connections`) — fine at the smoke horizon, a wall-clock + fidelity
  concern at a long horizon (A3 fixes it).
- The output is terse SVO under the brain's own renderer; the off-bridge Qwen gives fluent prose (GPU); the rich path
  (B1) is the multi-sentence upgrade — both keep the moat by per-sentence VERIFY.
- 1-seed GPU-GO only so far; the 6-seed gate applies before any *"works at scale"* claim (A2).
- Free-generation distribution-growth hit a scale wall (C2 in-band) — OPTIONAL (A5), decoupled from the development
  metric.

---

## Sources (re-verified this pass, file:line / finding)
- `research/runners/_longitudinal_develop_loop_gpu.py` (read in full — the 5-stage loop, the resume RE-HEAR stand-in,
  the anti-cheat arms, the wall-clock + ETA), `_longitudinal_develop_loop.py` (the imported loop machinery +
  `_save_state`/`_load_state`/`DevelopState`). Finding `2026-06-23-longitudinal-develop-loop-GPU-GO.md`.
- `research/runners/developed_brain_io.py` (read in full — `save_developed_brain`/`load_developed_brain`/
  `is_developed_brain_bundle`; the bundle format; **never CALLED** by the loop). `research/runners/brain_chat_tui.py`
  (read in full — `ChatBrain` GATE/VERIFY/moat, `QuestionRouter`, `QwenRenderer`/`StubRenderer`, `load_brain`
  precedence, the `--rich` wiring). `research/runners/rich_answer_composer.py` (read — `RichAnswerComposer.answer`,
  the per-sentence VERIFY confab-drop, the neural discourse-planner).
- `webapp/server.py:2700-2899` (read — `/api/brain-chat` + reset + `_build_chat_brain` + `_default_brain_renderer` +
  the `render_rich` forward-compat hook that `ChatBrain` does not satisfy). `webapp/static/{app.js,index.html,
  style.css}` (grepped — the Interact tab, leftmost + default, `setupBrainChat`, the abstention rendering).
- `sim/lineage.py` (grepped — `save`/`load` persist `DevelopState` JSON via `save_fn`; `export_shards`/`save_checkpoint`
  persist `cp_connections` — the A3 lever). `research/runners/_self_knowledge_demo.py` (read header + grepped — DEVELOPs
  via the develop loop, saves flat `_self_knowledge_grounded_codes.json`, has a `--repl`; does NOT save a bundle).
- On-disk check: `_self_knowledge_grounded_codes.json` (229 KB) + `_curriculum_self_knowledge.json` EXIST; **no
  `brain.json` bundle exists** anywhere. `research/findings/AUTONOMOUS_STATE.md` CYCLE 513-518 (the burndown roadmap +
  all 3 optional follow-ons just COMPLETED — the project is clear to start this arc).

### Memory pointers honored
`project_actual_goal_artificial_life_brain_analogue` (north star) · `feedback_frontend_purpose_console_not_dashboard`
(the console's 3 jobs, chat centerpiece) · `project_post_conversational_roadmap_tiers` (Tier-3 capstone) ·
`feedback_long_local_runs_ok_confirm_cloud_cause` (compressed-horizon LOCAL; no cloud) · `feedback_6seed_validation`
(6-seed before "works at scale", A2) · `feedback_moat_not_hard_lossy_memory_ok` (the moat held, surfaced distinctly) ·
`feedback_dont_gate_on_approval` (a justified `sim/` edit is fine with a byte-review — none needed on the critical
path) · `project_grounded_language_faculty` (the gate→render→verify faculty the rich path reuses).
